#!/usr/bin/env python
"""Vibe test: can the NLI cross-encoder (and its latent + MLP) play Flappy Bird?

The game state is rendered as text (premise); the two actions are the options ("flap" / "do nothing").
Policies:
  random, never (always "do nothing"), oracle (heuristic on the true state),
  nli   : zero-shot, argmax P(entailment) over "The correct action is: {a}",
  mlp   : latent of the frozen NLI model for both options -> MLP trained with soft BCE on oracle labels
          collected from noisy oracle rollouts (same recipe as latent_mlp.py).

    python flappy.py --ckpt ckpt/qwen3.5-4b-nli --episodes 20 --out results/flappy_4b.json
"""
import argparse
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn

from latent_mlp import MLP, fit, predict, grouped_split

# ----------------------------------------------------------------------------- game
GRAVITY, FLAP_V, SPEED = 0.003, 0.015, 0.015
PIPE_EVERY, GAP, PIPE_HW, BIRD_X, BIRD_HW = 0.45, 0.28, 0.04, 0.2, 0.03


class Flappy:
    def __init__(self, seed=0, max_steps=2000):
        self.rng = random.Random(seed)
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.y, self.vy, self.t, self.score, self.done = 0.5, 0.0, 0, 0, False
        self.pipes = [[1.2, self._gap()]]  # [x, gap_lo]
        return self.state()

    def _gap(self):
        return self.rng.uniform(0.15, 0.85 - GAP)

    def next_pipe(self):
        for x, lo in self.pipes:
            if x + PIPE_HW >= BIRD_X - BIRD_HW:
                return x, lo
        return None

    def step(self, flap):
        if self.done:
            return self.state(), 0.0, True
        self.vy = FLAP_V if flap else self.vy - GRAVITY
        self.y += self.vy
        self.t += 1
        for p in self.pipes:
            p[0] -= SPEED
        if self.pipes[-1][0] < 1.2 - PIPE_EVERY:
            self.pipes.append([self.pipes[-1][0] + PIPE_EVERY, self._gap()])
        if self.pipes[0][0] + PIPE_HW < BIRD_X - BIRD_HW:
            self.pipes.pop(0)
            self.score += 1
        x, lo = self.next_pipe()
        hit = self.y <= 0 or self.y >= 1
        if abs(x - BIRD_X) < PIPE_HW + BIRD_HW and not (lo < self.y < lo + GAP):
            hit = True
        if hit or self.t >= self.max_steps:
            self.done = True
        return self.state(), (0.0 if hit else 1.0), self.done

    def state(self):
        x, lo = self.next_pipe()
        return {"y": self.y, "vy": self.vy, "dx": x - BIRD_X, "lo": lo, "hi": lo + GAP, "score": self.score, "t": self.t}


def oracle(s, margin=0.03, lookahead=3):
    """flap if the predicted height a few frames ahead falls below the gap centre (minus a margin)."""
    target = (s["lo"] + s["hi"]) / 2 - margin
    y_pred = s["y"] + s["vy"] * lookahead - GRAVITY * lookahead * (lookahead - 1) / 2
    return y_pred < target


def render_text(s):
    pos = "inside the gap" if s["lo"] < s["y"] < s["hi"] else ("above the gap" if s["y"] >= s["hi"] else "below the gap")
    move = "rising" if s["vy"] > 0 else "falling"
    centre = (s["lo"] + s["hi"]) / 2
    return (f"Flappy Bird. The bird is at height {s['y']:.2f} (0 = ground, 1 = ceiling) and is {move} "
            f"with vertical velocity {s['vy']:+.3f} per frame; gravity pulls it down every frame and flapping pushes it up. "
            f"The next pipe is {s['dx']:.2f} ahead; its gap spans heights {s['lo']:.2f} to {s['hi']:.2f} (centre {centre:.2f}). "
            f"The bird is currently {pos}, {s['y'] - centre:+.2f} relative to the gap centre. "
            f"The bird must fly through the gap without touching the pipe, the ground or the ceiling.")


def render_numeric(s):
    return (f"Flappy Bird state: y={s['y']:.2f} vy={s['vy']:+.3f} pipe_dx={s['dx']:.2f} "
            f"gap_lo={s['lo']:.2f} gap_hi={s['hi']:.2f} gap_centre={(s['lo'] + s['hi']) / 2:.2f} "
            f"offset_from_centre={s['y'] - (s['lo'] + s['hi']) / 2:+.2f}")


def render_coach(s):
    centre = (s["lo"] + s["hi"]) / 2
    off = s["y"] - centre
    move = "rising" if s["vy"] > 0 else "falling"
    return (f"Flappy Bird. Rule of thumb: flap when the bird is below the centre of the next gap or falling towards it; "
            f"do nothing when it is above the centre or rising. Right now the bird is {abs(off):.2f} {'above' if off > 0 else 'below'} "
            f"the gap centre and {move} at {abs(s['vy']):.3f} per frame. The pipe is {s['dx']:.2f} ahead.")


def render_ascii_prompt(s):
    return "Flappy Bird screen (the bird is '>', pipes are '#', top row is the ceiling, bottom row is the ground):\n" + render_ascii(s)


PROMPTS = {"base": render_text, "numeric": render_numeric, "coach": render_coach, "ascii": render_ascii_prompt}
HYPS = {"action": lambda a: f"The correct action is: {a}",
        "should": lambda a: "The bird should flap now." if a == "flap" else "The bird should not flap now."}


def render_ascii(s, h=12, w=30):
    grid = [[" "] * w for _ in range(h)]
    col = int(min(max(s["dx"] / 0.6, 0), 1) * (w - 1))
    for r in range(h):
        yy = 1 - r / (h - 1)
        if not (s["lo"] < yy < s["hi"]):
            grid[r][col] = "#"
    br = int(round((1 - s["y"]) * (h - 1)))
    if 0 <= br < h:
        grid[br][2] = ">"
    return "\n".join("".join(r) for r in grid)


ACTIONS = ["flap", "do nothing"]


# ----------------------------------------------------------------------------- model-backed policies
class Scorer:
    def __init__(self, ckpt, prompt="base", hyp="action"):
        self.render, self.hyp = PROMPTS[prompt], HYPS[hyp]
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
        return [self.template.format(premise=self.render(s), hypothesis=self.hyp(a)) for a in ACTIONS]


def play(env_seed, policy, max_steps, fps=None, record=False):
    """fps=None: turn-based (the game waits for the policy). fps=30: real time - while the policy is thinking the
    game keeps ticking with no flap, so a slow policy acts on stale states and skips frames."""
    env = Flappy(seed=env_seed, max_steps=max_steps)
    s = env.reset()
    frames, replay, lats, skipped_total = [], [], [], 0

    def snap(a, lat_ms, skipped, probs=None):
        if record:
            replay.append({"t": env.t, "y": round(env.y, 4), "pipes": [[round(x, 3), round(lo, 3)] for x, lo in env.pipes[:3]],
                           "a": int(a), "lat_ms": round(lat_ms, 1), "skipped": skipped, "score": env.score,
                           "probs": [round(float(x), 4) for x in probs] if probs is not None else None})

    while not env.done:
        t0 = time.perf_counter()
        a = policy(s)
        probs = None
        if isinstance(a, tuple):
            a, probs = a
        lat = time.perf_counter() - t0
        lats.append(lat)
        skipped = int(lat * fps) if fps else 0
        if record and env.t % 3 == 0 and len(frames) < 12:
            frames.append((render_ascii(s), "FLAP" if a else "----"))
        snap(a, lat * 1000, 0, probs)
        s, _, _ = env.step(a)
        for _ in range(skipped):  # model still thinking: bird glides
            if env.done:
                break
            skipped_total += 1
            snap(False, 0.0, 1)
            s, _, _ = env.step(False)
    return {"score": env.score, "steps": env.t, "frames": frames, "replay": replay,
            "lat_ms": float(np.mean(lats) * 1000), "skipped": skipped_total, "decisions": len(lats)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/qwen3.5-4b-nli")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--collect-episodes", type=int, default=60, help="noisy oracle rollouts for MLP training data")
    ap.add_argument("--noise", type=float, default=0.15)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--out", default="results/flappy.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=float, default=30.0, help="real-time tick rate; 0 = turn-based")
    ap.add_argument("--record-only", action="store_true", help="few episodes, replay with probabilities (for the video)")
    ap.add_argument("--prompt", default="base", choices=list(PROMPTS))
    ap.add_argument("--hyp", default="action", choices=list(HYPS))
    ap.add_argument("--lookahead", type=int, default=3, help="oracle lookahead used for labels and the oracle policy")
    ap.add_argument("--margin", type=float, default=0.03)
    ap.add_argument("--skip-nli", action="store_true")
    args = ap.parse_args()
    fps = args.fps or None
    rng = random.Random(args.seed)
    results = {}

    def evaluate(name, policy, record_first=True):
        eps = [play(1000 + i, policy, args.max_steps, fps=fps, record=record_first) for i in range(args.episodes)]
        best = max(eps, key=lambda e: e["score"])  # keep the replay of the best episode
        sc = [e["score"] for e in eps]
        lat = float(np.mean([e["lat_ms"] for e in eps])); skip = float(np.mean([e["skipped"] / max(e["steps"], 1) for e in eps]))
        results[name] = {"mean_score": float(np.mean(sc)), "median_score": float(np.median(sc)), "max_score": int(max(sc)),
                         "mean_steps": float(np.mean([e["steps"] for e in eps])), "lat_ms": lat, "skipped_frac": skip,
                         "frames": best["frames"], "replay": best["replay"], "replay_score": best["score"]}
        print(f"{name:10s} score mean {np.mean(sc):6.2f} median {np.median(sc):5.1f} max {max(sc):3d}  steps {np.mean([e['steps'] for e in eps]):7.1f}"
              f"  latency {lat:6.1f} ms  skipped frames {skip:5.1%}", flush=True)

    if not args.record_only:
        evaluate("random", lambda s: rng.random() < 0.1)
        evaluate("never", lambda s: False)
    orc = lambda s: oracle(s, margin=args.margin, lookahead=args.lookahead)
    evaluate("oracle", orc)

    scorer = Scorer(args.ckpt, args.prompt, args.hyp)

    # zero-shot NLI: argmax entailment over the two action hypotheses
    def nli_policy(s):
        _, L = scorer.latents(scorer.pair_texts(s))
        p = torch.softmax(torch.tensor(L), -1).numpy()
        return int(p[:, 1].argmax()) == 0, p[:, 1]  # index 0 = flap; probs = P(entailment) per option
    if not args.skip_nli:
        t0 = time.time(); evaluate("nli", nli_policy); print(f"  ({time.time()-t0:.0f}s)")

    # latent + MLP trained on noisy-oracle rollouts
    states, labels = [], []
    for i in range(args.collect_episodes):
        env = Flappy(seed=i, max_steps=600); s = env.reset()
        while not env.done:
            a_or = orc(s)
            states.append(dict(s)); labels.append(int(a_or))
            a = (not a_or) if rng.random() < args.noise else a_or
            s, _, _ = env.step(a)
    print(f"collected {len(states)} states, flap rate {np.mean(labels):.2f}", flush=True)
    texts = [t for s in states for t in scorer.pair_texts(s)]
    X, _ = scorer.latents(texts)
    qid = np.repeat(np.arange(len(states)), 2)
    gold = np.array([[1, 0] if l == 1 else [0, 1] for l in labels]).ravel()
    tr, va = grouped_split(qid, 0.1, args.seed)
    ns = argparse.Namespace(hidden=512, dropout=0.1, lr=1e-3, wd=1e-2, bs=512, epochs=60, patience=8, eps=args.eps, seed=args.seed)
    model, stats, va_acc, _ = fit(X[tr], gold[tr], qid[tr], X[va], gold[va], qid[va], ns)
    print(f"mlp val agreement with oracle: {va_acc:.3f}", flush=True)
    results["mlp_val_acc"] = va_acc

    def mlp_policy(s):
        Xs, _ = scorer.latents(scorer.pair_texts(s))
        z = predict(model, stats, Xs)
        return int(z.argmax()) == 0, 1 / (1 + np.exp(-z))  # probs = sigmoid score per option (flap, do nothing)
    t0 = time.time(); evaluate("mlp", mlp_policy); print(f"  ({time.time()-t0:.0f}s)")

    json.dump({"args": vars(args), "results": results}, open(args.out, "w"), indent=2)
    for name in [n for n in ["nli", "mlp"] if n in results]:
        print(f"\n=== {name}: first frames of episode 0 (bird '>', pipe '#')")
        for fr, a in results[name]["frames"][:4]:
            print(fr); print("action:", a); print("-" * 30)


if __name__ == "__main__":
    main()
