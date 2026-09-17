#!/usr/bin/env python
"""Headed, real-time Doom driven by the openjev NLI cross-encoder.

Differs from doom.py in the two ways that matter for watching it live:

* the ViZDoom window is visible and the engine runs in ASYNC_PLAYER at a fixed ticrate, so the world clock
  keeps running while the model thinks (doom.py uses PLAYER, where the sim blocks on every forward pass);
* inference runs on its own thread. The main thread advances the game one tic at a time and applies the most
  recent decision, so the window stays at a smooth ticrate no matter how long a forward pass takes. The model
  re-decides as fast as it can (~20 Hz on a 4B) instead of on a fixed 4-tic cadence.

    python doom_live.py --ckpt ../openjev_hf/qwen3.5-4b-nli --hyp position --episodes 3
"""
import argparse
import os
import threading
import time

import numpy as np
import torch
import vizdoom as vzd

from doom import ACTIONS, BUTTONS, HYPS, parse_state, render_text


def make_game(res, ticrate, sound, window=True):
    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, "defend_the_center.cfg"))
    game.set_screen_resolution(res)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_labels_buffer_enabled(True)
    game.set_window_visible(window)
    game.set_sound_enabled(sound)
    game.set_render_hud(True)
    game.set_render_weapon(True)
    game.set_render_crosshair(True)
    game.set_render_all_frames(True)   # draw the tics between decisions; without it the window is a slideshow
    game.set_mode(vzd.Mode.ASYNC_PLAYER)
    game.set_ticrate(ticrate)
    game.set_episode_timeout(2100)
    game.init()
    return game


class Policy:
    """Scores the hypothesis set against a premise. Holds the model; called only from the worker thread."""

    def __init__(self, ckpt, hyp):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.hypotheses, self.action_map = HYPS[hyp]
        self.tok = AutoTokenizer.from_pretrained(ckpt)
        self.model = AutoModelForSequenceClassification.from_pretrained(ckpt, dtype=torch.bfloat16).cuda().eval()
        self.template = getattr(self.model.config, "nli_template", None) or "Premise: {premise}\nHypothesis: {hypothesis}"
        if self.model.config.get_text_config().pad_token_id is None:
            self.model.config.get_text_config().pad_token_id = self.tok.pad_token_id
        self.tok.padding_side = "right"
        self.backbone = getattr(self.model, self.model.base_model_prefix)

    @torch.no_grad()
    def __call__(self, premise):
        texts = [self.template.format(premise=premise, hypothesis=h) for h in self.hypotheses]
        enc = self.tok(texts, truncation=True, max_length=512, padding=True, return_tensors="pt").to("cuda")
        h = self.backbone(**enc).last_hidden_state
        last = enc["attention_mask"].sum(1) - 1
        logits = self.model.score(h[torch.arange(h.shape[0], device=h.device), last]).float()
        p_ent = torch.softmax(logits, -1)[:, 1].cpu().numpy()
        pa = np.zeros(len(ACTIONS))
        for i, a in enumerate(self.action_map):     # fold hypotheses onto actions (max over those bound to each)
            pa[a] = max(pa[a], float(p_ent[i]))
        return int(pa.argmax()), pa


class Brain(threading.Thread):
    """Consumes the latest premise, publishes the latest decision. Stale premises are dropped, never queued."""

    daemon = True

    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.premise = None
        self.action, self.probs, self.lat_ms, self.n = 0, np.zeros(len(ACTIONS)), 0.0, 0
        self.stop_flag = False

    def submit(self, premise):
        with self.lock:
            self.premise = premise
        self.wake.set()

    def latest(self):
        with self.lock:
            return self.action, self.probs.copy(), self.lat_ms, self.n

    def run(self):
        while not self.stop_flag:
            self.wake.wait(0.1)
            self.wake.clear()
            with self.lock:
                premise, self.premise = self.premise, None
            if premise is None:
                continue
            t0 = time.perf_counter()
            a, pa = self.policy(premise)
            dt = (time.perf_counter() - t0) * 1000
            with self.lock:
                self.action, self.probs, self.lat_ms, self.n = a, pa, dt, self.n + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="../openjev_hf/qwen3.5-4b-nli")
    ap.add_argument("--hyp", default="position", choices=list(HYPS))
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ticrate", type=int, default=35, help="vanilla Doom is 35")
    ap.add_argument("--res", default="RES_800X600")
    ap.add_argument("--sound", action="store_true")
    ap.add_argument("--no-window", action="store_true", help="for testing the loop without a display")
    ap.add_argument("--log-every", type=float, default=1.0, help="seconds between terminal status lines")
    args = ap.parse_args()

    print(f"loading {args.ckpt} ...", flush=True)
    brain = Brain(Policy(args.ckpt, args.hyp))
    for h in brain.policy.hypotheses:
        print(f"   {h!r} -> {ACTIONS[brain.policy.action_map[brain.policy.hypotheses.index(h)]]}")
    print("warming up kernels ...", flush=True)
    for _ in range(3):
        brain.policy("Doom, Defend the Center. No enemies are visible right now.")
    brain.start()

    game = make_game(getattr(vzd.ScreenResolution, args.res), args.ticrate, args.sound, window=not args.no_window)
    print(f"window {'off' if args.no_window else 'ON'}  ticrate {args.ticrate}  ASYNC_PLAYER  hyp={args.hyp}\n", flush=True)

    all_kills = []
    for ep in range(args.episodes):
        game.set_seed(args.seed + 100 + ep)
        game.new_episode()
        t_ep, last_log, decisions0 = time.perf_counter(), 0.0, brain.latest()[3]
        tics = 0
        while not game.is_episode_finished():
            s = parse_state(game)
            if s is None:
                break
            brain.submit(render_text(s))
            a, pa, lat, n = brain.latest()
            game.set_action(BUTTONS[a])
            game.advance_action(1)          # one tic, real-time -- ASYNC blocks until the tic elapses
            tics += 1
            now = time.perf_counter() - t_ep
            if now - last_log >= args.log_every:
                last_log = now
                rate = (n - decisions0) / max(now, 1e-9)
                print(f"  t{now:6.1f}s  kills {s['kills']:2d}  hp {s['health']:3d}  ammo {s['ammo']:2d}  "
                      f"| {ACTIONS[a]:10s} " + " ".join(f"{x:.2f}" for x in pa) +
                      f"  | {lat:5.1f} ms/decision, {rate:4.1f} decisions/s", flush=True)
        k = int(game.get_game_variable(vzd.GameVariable.KILLCOUNT))
        wall = time.perf_counter() - t_ep
        n = brain.latest()[3] - decisions0
        all_kills.append(k)
        print(f"episode {ep}: {k} kills  {tics} tics in {wall:.1f}s "
              f"({tics / max(wall, 1e-9):.1f} tics/s vs {args.ticrate} target)  {n} decisions\n", flush=True)

    brain.stop_flag = True
    brain.wake.set()
    brain.join(timeout=5)   # let the worker leave its wait() before CUDA tears down, else abort() at exit
    game.close()
    print(f"kills mean {np.mean(all_kills):.2f} max {max(all_kills)}  over {args.episodes} episodes")


if __name__ == "__main__":
    main()
