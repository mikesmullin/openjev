#!/usr/bin/env python
"""Doom (ViZDoom "defend the center") played by the NLI cross-encoder, same recipe as flappy.py.

Each decision step (4 game tics = 114 ms of game time) the visible enemies from the labels buffer are rendered as text
(premise); the three actions are the options: "turn left" / "turn right" / "attack".
Policies: random, oracle (heuristic on the labels), nli (zero-shot entailment), mlp (frozen latent + MLP with soft BCE
trained on noisy-oracle rollouts). The model has ~60 ms per decision on a 4B, under the 114 ms budget -> real time.

    python doom.py --ckpt ckpt/qwen3.5-4b-nli --episodes 5 --out results/doom_4b.json --video results/doom_mlp.mp4
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import vizdoom as vzd

from latent_mlp import fit, predict, grouped_split

ACTIONS = ["turn left", "turn right", "attack"]
BUTTONS = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
FRAME_SKIP = 4
# Hypothesis variants. Each variant is a list of statements about the state; every statement is bound to an action
# (index into ACTIONS). "action" is the naive variant shipped in the published script (it collapses to random);
# "position"/"position_none" are the state-statement variants reported in results/full_report.md.
HYPS = {
    "action": ([f"The correct action is: {a}" for a in ACTIONS], [0, 1, 2]),
    "position": ([
        "The nearest enemy is to the left of the crosshair.",
        "The nearest enemy is to the right of the crosshair.",
        "The nearest enemy is exactly on the crosshair.",
    ], [0, 1, 2]),
    "position_none": ([
        "The nearest enemy is to the left of the crosshair.",
        "The nearest enemy is to the right of the crosshair.",
        "The nearest enemy is exactly on the crosshair.",
        "There is no enemy in view.",
    ], [0, 1, 2, 0]),
}

ENEMY_NAMES = {"Zombieman": "zombie soldier", "ShotgunGuy": "shotgun guard", "Imp": "imp", "Demon": "pinky demon",
               "MarineChainsaw": "chainsaw marine", "MarineChainsawVzd": "chainsaw marine", "ChaingunGuy": "chaingunner",
               "HellKnight": "hell knight", "Cacodemon": "cacodemon", "LostSoul": "lost soul", "Revenant": "revenant", "BaronOfHell": "baron of hell"}


def make_game(res=vzd.ScreenResolution.RES_640X480):
    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, "defend_the_center.cfg"))
    game.set_screen_resolution(res)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_labels_buffer_enabled(True)
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_episode_timeout(2100)
    game.init()
    return game


def parse_state(game):
    st = game.get_state()
    if st is None:
        return None
    W, H = st.screen_buffer.shape[1], st.screen_buffer.shape[0]
    enemies = []
    for lab in st.labels:
        if lab.object_name not in ENEMY_NAMES or lab.width == 0:  # skip player, blood splats, bullet puffs, ...
            continue
        cx = (lab.x + lab.width / 2) / W - 0.5
        enemies.append({"name": ENEMY_NAMES.get(lab.object_name, lab.object_name.lower()), "off": float(cx),
                        "size": float(lab.height / H)})
    enemies.sort(key=lambda e: abs(e["off"]))
    ammo, health = game.get_game_variable(vzd.GameVariable.AMMO2), game.get_game_variable(vzd.GameVariable.HEALTH)
    return {"enemies": enemies, "ammo": int(ammo), "health": int(health), "kills": int(game.get_game_variable(vzd.GameVariable.KILLCOUNT)),
            "frame": st.screen_buffer}


def oracle(s, tol=0.03):
    if not s["enemies"]:
        return 0  # scan left
    e = s["enemies"][0]
    if abs(e["off"]) < tol:
        return 2 if s["ammo"] > 0 else (0 if e["off"] < 0 else 1)
    return 0 if e["off"] < 0 else 1


def render_text(s):
    if s["enemies"]:
        parts = []
        for e in s["enemies"][:4]:
            side = "right of" if e["off"] > 0.015 else ("left of" if e["off"] < -0.015 else "exactly on")
            dist = "very close" if e["size"] > 0.45 else ("close" if e["size"] > 0.25 else "far")
            parts.append(f"a {e['name']} {abs(e['off']):.2f} to the {side} the crosshair ({dist})".replace("to the exactly on", "exactly on"))
        seen = "Visible enemies: " + "; ".join(parts) + "."
    else:
        seen = "No enemies are visible right now."
    return (f"Doom, Defend the Center. You stand in the middle of a circular arena with a pistol ({s['ammo']} bullets, health {s['health']}). "
            f"Enemies walk toward you from all sides and attack when close; you can only turn left, turn right, or fire. "
            f"Screen offsets are fractions of the screen width (0 = crosshair, 0.5 = screen edge); one turn step moves the view by about 0.05. "
            f"{seen} A shot hits only if an enemy is within about 0.03 of the crosshair.")


class Scorer:
    def __init__(self, ckpt, hyp="action"):
        self.hyp_name = hyp
        self.hypotheses, self.action_map = HYPS[hyp]
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(ckpt)
        self.model = AutoModelForSequenceClassification.from_pretrained(ckpt, dtype=torch.bfloat16).cuda().eval()
        self.template = getattr(self.model.config, "nli_template", None) or "Premise: {premise}\nHypothesis: {hypothesis}"
        if self.model.config.get_text_config().pad_token_id is None:
            self.model.config.get_text_config().pad_token_id = self.tok.pad_token_id
        self.tok.padding_side = "right"
        self.backbone = getattr(self.model, self.model.base_model_prefix)

    @torch.no_grad()
    def latents(self, texts, bs=64):
        X, L = [], []
        for s in range(0, len(texts), bs):
            enc = self.tok(texts[s:s + bs], truncation=True, max_length=512, padding=True, return_tensors="pt").to("cuda")
            h = self.backbone(**enc).last_hidden_state
            last = enc["attention_mask"].sum(1) - 1
            pooled = h[torch.arange(h.shape[0], device=h.device), last]
            X.append(pooled.float().cpu().numpy()); L.append(self.model.score(pooled).float().cpu().numpy())
        return np.concatenate(X), np.concatenate(L)

    def pair_texts(self, s):
        return [self.template.format(premise=render_text(s), hypothesis=h) for h in self.hypotheses]

    def action_probs(self, p_ent):
        """Fold per-hypothesis P(entailment) into one score per action (max over the hypotheses bound to it)."""
        out = np.zeros(len(ACTIONS), dtype=np.float64)
        for h, a in enumerate(self.action_map):
            out[a] = max(out[a], float(p_ent[h]))
        return out


def play(game, policy, seed, record=False):
    game.set_seed(seed)
    game.new_episode()
    frames, lats, steps = [], [], 0
    while not game.is_episode_finished():
        s = parse_state(game)
        if s is None:
            break
        t0 = time.perf_counter()
        a = policy(s)
        probs = None
        if isinstance(a, tuple):
            a, probs = a
        lats.append(time.perf_counter() - t0)
        if record:
            frames.append({"frame": s["frame"], "text": render_text(s), "a": int(a), "probs": None if probs is None else [float(p) for p in probs],
                           "kills": s["kills"], "ammo": s["ammo"], "health": s["health"], "lat_ms": lats[-1] * 1000})
        game.make_action(BUTTONS[a], FRAME_SKIP)
        steps += 1
    kills = int(game.get_game_variable(vzd.GameVariable.KILLCOUNT))
    return {"kills": kills, "reward": game.get_total_reward(), "steps": steps, "lat_ms": float(np.mean(lats) * 1000) if lats else 0.0, "frames": frames}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/qwen3.5-4b-nli")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--collect-episodes", type=int, default=12)
    ap.add_argument("--noise", type=float, default=0.2)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/doom_4b.json")
    ap.add_argument("--video", default=None, help="mp4 of the best MLP episode with option probabilities")
    ap.add_argument("--video-nli", default=None)
    ap.add_argument("--hyp", default="action", choices=list(HYPS))
    ap.add_argument("--zero-shot-only", action="store_true", help="skip the noisy-oracle collection + latent MLP stage")
    args = ap.parse_args()
    if args.hyp != "action" and not args.zero_shot_only:
        ap.error("--hyp other than 'action' requires --zero-shot-only (the MLP stage assumes one hypothesis per action)")
    rng = random.Random(args.seed)
    game = make_game()
    results, replays = {}, {}

    def evaluate(name, policy, record=False):
        eps = [play(game, policy, 100 + i, record=record) for i in range(args.episodes)]
        k = [e["kills"] for e in eps]
        results[name] = {"mean_kills": float(np.mean(k)), "max_kills": int(max(k)), "mean_reward": float(np.mean([e["reward"] for e in eps])),
                         "mean_steps": float(np.mean([e["steps"] for e in eps])), "lat_ms": float(np.mean([e["lat_ms"] for e in eps]))}
        if record:
            replays[name] = max(eps, key=lambda e: e["kills"])["frames"]
        print(f"{name:8s} kills mean {np.mean(k):5.2f} max {max(k):2d}  reward {np.mean([e['reward'] for e in eps]):6.1f}  "
              f"steps {np.mean([e['steps'] for e in eps]):6.1f}  latency {results[name]['lat_ms']:.1f} ms", flush=True)

    evaluate("random", lambda s: rng.randrange(3))
    evaluate("oracle", oracle)
    scorer = Scorer(args.ckpt, args.hyp)

    def nli_policy(s):
        _, L = scorer.latents(scorer.pair_texts(s))
        p_ent = torch.softmax(torch.tensor(L), -1).numpy()[:, 1]
        pa = scorer.action_probs(p_ent)
        return int(pa.argmax()), pa
    evaluate("nli", nli_policy, record=bool(args.video_nli))
    results["nli"]["hyp"] = args.hyp
    results["nli"]["hypotheses"] = scorer.hypotheses
    results["nli"]["action_map"] = scorer.action_map

    if args.zero_shot_only:
        game.close()
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump({"args": vars(args), "results": results}, open(args.out, "w"), indent=2)
        if args.video_nli and "nli" in replays:
            write_video(replays["nli"], args.video_nli, "nli")
        return

    states, labels = [], []
    for i in range(args.collect_episodes):
        game.set_seed(i); game.new_episode()
        while not game.is_episode_finished():
            s = parse_state(game)
            if s is None:
                break
            a_or = oracle(s)
            states.append({k: v for k, v in s.items() if k != "frame"}); labels.append(a_or)
            a = rng.randrange(3) if rng.random() < args.noise else a_or
            game.make_action(BUTTONS[a], FRAME_SKIP)
    print(f"collected {len(states)} states; action dist {np.bincount(labels, minlength=3) / len(labels)}", flush=True)
    X, _ = scorer.latents([t for s in states for t in scorer.pair_texts(s)])
    qid = np.repeat(np.arange(len(states)), 3)
    gold = np.array([[int(j == l) for j in range(3)] for l in labels]).ravel()
    tr, va = grouped_split(qid, 0.1, args.seed)
    ns = argparse.Namespace(hidden=512, dropout=0.1, lr=1e-3, wd=1e-2, bs=512, epochs=60, patience=8, eps=args.eps, seed=args.seed)
    model, stats, va_acc, _ = fit(X[tr], gold[tr], qid[tr], X[va], gold[va], qid[va], ns)
    print(f"mlp val agreement with oracle: {va_acc:.3f}", flush=True)
    results["mlp_val_acc"] = va_acc

    def mlp_policy(s):
        Xs, _ = scorer.latents(scorer.pair_texts(s))
        z = predict(model, stats, Xs)
        return int(z.argmax()), 1 / (1 + np.exp(-z))
    evaluate("mlp", mlp_policy, record=bool(args.video))
    game.close()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"args": vars(args), "results": results}, open(args.out, "w"), indent=2)
    for name, path in [("mlp", args.video), ("nli", args.video_nli)]:
        if path and name in replays:
            write_video(replays[name], path, name)


def write_video(frames, path, name):
    import imageio
    from PIL import Image, ImageDraw, ImageFont
    import matplotlib
    fd = matplotlib.get_data_path() + "/fonts/ttf/"
    F = lambda sz, b=False: ImageFont.truetype(fd + ("DejaVuSansMono-Bold.ttf" if b else "DejaVuSansMono.ttf"), sz)
    f_s, f_m, f_l = F(15), F(19), F(24, True)
    W, H = 1280, 720
    BG, PANEL, INK, MUTE, LINE = (11, 21, 25), (18, 34, 41), (228, 239, 236), (138, 166, 171), (36, 64, 74)
    COLS = [(242, 177, 52), (63, 191, 127), (229, 83, 61)]
    w = imageio.get_writer(path, fps=17, codec="libx264", quality=8, macro_block_size=None)
    win = 120
    for i, fr in enumerate(frames):
        img = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(img)
        title = {"mlp": "latent + MLP", "nli": "zero-shot NLI", "ft": "cross-encoder fine-tuned on frames (NLI loss)"}.get(name, name)
        d.text((30, 18), f"Doom · Defend the Center · Qwen3.5-4B NLI cross-encoder · {title}", fill=INK, font=f_l)
        d.text((30, 50), "one decision per 4 tics (114 ms game time), shown at ~x2 · 3 options scored per step", fill=MUTE, font=f_s)
        img.paste(Image.fromarray(fr["frame"]).resize((600, 450)), (30, 80))
        d.text((30, 540), f"step {i:4d}   kills {fr['kills']:2d}   ammo {fr['ammo']:2d}   health {fr['health']:3d}   decision {fr['lat_ms']:.0f} ms", fill=MUTE, font=f_s)
        # probability panel
        PX, PY, PW, PH = 680, 100, 560, 260
        d.rectangle([PX, PY, PX + PW, PY + PH], fill=PANEL)
        for yy, lab in [(0, "0"), (0.5, "0.5"), (1, "1")]:
            y = PY + PH - yy * PH; d.line([(PX, y), (PX + PW, y)], fill=LINE); d.text((PX - 28, y - 8), lab, fill=MUTE, font=f_s)
        start = max(0, i - win + 1)
        xs = lambda k: PX + (k - start) / win * PW
        for j, col in enumerate(COLS):
            pts = [(xs(k), PY + PH - frames[k]["probs"][j] * PH) for k in range(start, i + 1) if frames[k]["probs"]]
            if len(pts) > 1:
                d.line(pts, fill=col, width=3)
        for k in range(start, i + 1):
            d.line([(xs(k), PY + PH - 6), (xs(k), PY + PH)], fill=COLS[frames[k]["a"]], width=2)
        if fr["probs"]:
            for j, (a, col) in enumerate(zip(ACTIONS, COLS)):
                d.text((PX + j * 190, PY + PH + 14), f"{a}: {fr['probs'][j]:.2f}", fill=col, font=f_m)
        d.text((PX, PY + PH + 46), f"chosen: {ACTIONS[fr['a']].upper()}", fill=INK, font=f_m)
        y0 = PY + PH + 90
        words, line, lines = fr["text"].split(), "", []
        for wd in words:
            if len(line) + len(wd) + 1 > 62:
                lines.append(line); line = wd
            else:
                line = (line + " " + wd).strip()
        lines.append(line)
        for j, ln in enumerate(lines[:11]):
            d.text((PX, y0 + j * 20), ln, fill=MUTE, font=f_s)
        w.append_data(np.asarray(img))
    for _ in range(17):
        w.append_data(np.asarray(img))
    w.close()
    print("wrote", path, len(frames), "frames")


if __name__ == "__main__":
    main()
