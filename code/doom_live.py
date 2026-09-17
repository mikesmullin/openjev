#!/usr/bin/env python
"""Headed, real-time Doom driven by the openjev NLI cross-encoder.

Differs from doom.py in the two ways that matter for watching it live:

* the ViZDoom window is visible and the engine runs in ASYNC_PLAYER at a fixed ticrate, so the world clock
  keeps running while the model thinks (doom.py uses PLAYER, where the sim blocks on every forward pass);
* inference runs on its own thread. The main thread advances the game one tic at a time and applies the most
  recent decision, so the window stays at a smooth ticrate no matter how long a forward pass takes. The model
  re-decides as fast as it can (~25 Hz on a 4B) instead of on a fixed 4-tic cadence.

    python doom_live.py --scenario defend_the_center      # the tweet's arena, 3 actions
    python doom_live.py --scenario deadly_corridor        # walk a corridor under fire
    python doom_live.py --scenario freedoom2 --map map01  # a real level, doors and all
    python doom_live.py --list

Everything past defend_the_center is our extension, not part of the published result: the checkpoint was only
ever demonstrated as a 3-action turret. Walking levels need movement actions, so the hypothesis set grows
(see PROFILES) and the premise carries items, weapon and a stuck flag. Same frozen model, still zero-shot.
"""
import argparse
import os
import re
import threading
import time

import numpy as np
import torch
import vizdoom as vzd

from doom import ENEMY_NAMES

# ----------------------------------------------------------------------------- actions
# An action is a set of (button, value) pairs; a scenario supports it only if its cfg declares every button
# (or the button is a delta, which we add ourselves).
#
# Aiming geometry, measured on this build: the binary TURN_LEFT/TURN_RIGHT buttons turn 1.76 deg per tic, and
# an action is held until the next decision (1.3-2.5 tics), so one decision swings 2.3-4.4 deg = 0.026-0.049
# in screen-offset units. The hit window is +/-0.03. A single turn step is therefore as wide as the entire
# window the policy is trying to stop inside, which is what makes it oscillate around the target.
#
# The fix is not simply a smaller step -- that fixes settling but doubles the time to swing onto a target
# across the screen, which costs kills when enemies close from all sides. Instead give the model a coarse and
# a fine turn and let it pick, which is what a classifier is actually good at. TURN_LEFT_RIGHT_DELTA takes an
# exact degrees-per-tic value (positive = right).
TURN_FAST, TURN_SLOW = 6.0, 1.0   # degrees per tic; slow is ~0.011 offset per decision, well inside the window
DELTA = "TURN_LEFT_RIGHT_DELTA"

ACTION_BUTTONS = {
    "turn left": [("TURN_LEFT", 1)],
    "turn right": [("TURN_RIGHT", 1)],
    "swing left": [(DELTA, -TURN_FAST)],
    "swing right": [(DELTA, TURN_FAST)],
    "nudge left": [(DELTA, -TURN_SLOW)],
    "nudge right": [(DELTA, TURN_SLOW)],
    "attack": [("ATTACK", 1)],
    "move forward": [("MOVE_FORWARD", 1)],
    "move left": [("MOVE_LEFT", 1)],
    "move right": [("MOVE_RIGHT", 1)],
    "use": [("USE", 1)],
    # Compound recovery actions. A bare turn or a bare USE leaves the player's position unchanged, so if the
    # premise says "stuck" the next premise says "stuck" too and the policy spins in place forever. Both
    # recoveries have to actually translate the player.
    "veer right": [(DELTA, TURN_FAST), ("MOVE_FORWARD", 1)],
    "open and step through": [("USE", 1), ("MOVE_FORWARD", 1)],
}

# Offset bands used by both the premise wording and the hypotheses -- they have to agree or the model is
# asked to confirm a sentence the premise never says. CENTRE matches the actual hit tolerance (0.03); the
# original render_text called anything past 0.015 "left of", so it kept turning off targets it could already hit.
CENTRE, NEAR = 0.03, 0.10

# ----------------------------------------------------------------------------- hypothesis profiles
# Every hypothesis is a statement *about the state*, bound to an action -- never the name of the action.
# That distinction is the whole result: see REPRO.md.
TURRET = [   # the published 3-way set: one coarse turn each way, no magnitude control
    ("The nearest enemy is to the left of the crosshair.", "turn left"),
    ("The nearest enemy is to the right of the crosshair.", "turn right"),
    ("The nearest enemy is exactly on the crosshair.", "attack"),
]
AIM = [      # same job, but the model also chooses how far to turn
    ("The nearest enemy is far to the left of the crosshair.", "swing left"),
    ("The nearest enemy is just slightly to the left of the crosshair.", "nudge left"),
    ("The nearest enemy is lined up with the crosshair.", "attack"),
    ("The nearest enemy is just slightly to the right of the crosshair.", "nudge right"),
    ("The nearest enemy is far to the right of the crosshair.", "swing right"),
    ("No enemy is visible right now.", "swing left"),
]
EXPLORER = AIM + [
    ("No enemy is in view and the way ahead is open.", "move forward"),
    ("The player is pressed against a wall and has stopped moving.", "veer right"),
    ("There is a door or a switch directly ahead.", "open and step through"),
]
DODGER = [
    ("The incoming fireball is to the left of the player.", "move right"),
    ("The incoming fireball is to the right of the player.", "move left"),
]
PROFILES = {"turret": TURRET, "aim": AIM, "explorer": EXPLORER, "dodger": DODGER}

SCENARIOS = {
    "defend_the_center":        dict(cfg="defend_the_center.cfg",        profile="turret",   blurb="circular arena, turn and shoot -- the tweet"),
    "defend_the_line":          dict(cfg="defend_the_line.cfg",          profile="turret",   blurb="enemies advance from one side"),
    "predict_position":         dict(cfg="predict_position.cfg",         profile="turret",   blurb="rocket a moving target, no autoaim"),
    "deadly_corridor":          dict(cfg="deadly_corridor.cfg",          profile="explorer", blurb="walk a corridor under fire, skill 5"),
    "health_gathering_supreme": dict(cfg="health_gathering_supreme.cfg", profile="explorer", blurb="acid floor, find medkits"),
    "my_way_home":              dict(cfg="my_way_home.cfg",              profile="explorer", blurb="maze of rooms, find the goal"),
    "take_cover":               dict(cfg="take_cover.cfg",               profile="dodger",   blurb="strafe to dodge fireballs"),
    "deathmatch":               dict(cfg="deathmatch.cfg",               profile="explorer", blurb="open arena, many weapons"),
    "freedoom2":                dict(cfg="freedoom2.cfg",  iwad="freedoom2.wad", map="map01", profile="explorer", blurb="the actual game, Doom II-style maps"),
    "freedoom1":                dict(cfg="freedoom1.cfg",  iwad="freedoom1.wad", map="e1m1", profile="explorer", blurb="the actual game, Doom I-style maps"),
}

IGNORE_LABELS = ("DoomPlayer", "Blood", "BulletPuff", "Puff", "TeleportFog", "Smoke")
VARS = ["HEALTH", "ARMOR", "KILLCOUNT", "SELECTED_WEAPON", "SELECTED_WEAPON_AMMO", "POSITION_X", "POSITION_Y"]


def pretty(name):
    return re.sub(r"(?<!^)(?=[A-Z])", " ", name).lower()


def make_game(scen, args, profile, window=True):
    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, scen["cfg"]))
    if "iwad" in scen:
        game.set_doom_game_path(os.path.join(os.path.dirname(vzd.__file__), scen["iwad"]))
        game.set_doom_map(args.map or scen["map"])
    game.set_screen_resolution(getattr(vzd.ScreenResolution, args.res))
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_labels_buffer_enabled(True)
    game.set_window_visible(window)
    game.set_sound_enabled(args.sound)
    game.set_render_hud(True)
    game.set_render_weapon(True)
    game.set_render_crosshair(True)
    game.set_render_all_frames(True)   # draw the tics between decisions, else the window is a slideshow
    game.set_mode(vzd.Mode.ASYNC_PLAYER)
    game.set_ticrate(args.ticrate)
    game.set_episode_timeout(args.timeout)
    # Delta turning is our addition, so add the button ourselves. Binary buttons are never added: a scenario
    # that withholds MOVE_FORWARD is withholding it on purpose.
    if any(b == DELTA for _, a in PROFILES[profile] for b, _ in ACTION_BUTTONS[a]):
        if vzd.Button.TURN_LEFT_RIGHT_DELTA not in game.get_available_buttons():
            game.add_available_button(vzd.Button.TURN_LEFT_RIGHT_DELTA)
    for v in VARS:
        if getattr(vzd.GameVariable, v) not in game.get_available_game_variables():
            game.add_available_game_variable(getattr(vzd.GameVariable, v))
    game.init()
    return game


def build_actions(game, profile):
    """Keep only the hypotheses whose action this scenario's buttons can express."""
    names = [b.name for b in game.get_available_buttons()]
    idx = {n: i for i, n in enumerate(names)}
    hyps, actions, vectors = [], [], []
    for text, act in PROFILES[profile]:
        need = ACTION_BUTTONS[act]
        if not all(b in idx for b, _ in need):
            continue
        v = [0] * len(names)
        for b, val in need:
            v[idx[b]] = val
        hyps.append(text)
        actions.append(act)
        vectors.append(v)
    if not hyps:
        raise SystemExit(f"no usable action for profile {profile}; buttons are {names}")
    return hyps, actions, vectors


class World:
    """Turns the labels buffer + game variables into the premise string."""

    def __init__(self, game, title, actions):
        self.game = game
        self.title = title
        self.actions = actions
        self.trail = []

    def var(self, n):
        return self.game.get_game_variable(getattr(vzd.GameVariable, n))

    def observe(self):
        st = self.game.get_state()
        if st is None:
            return None
        H, W = st.screen_buffer.shape[0], st.screen_buffer.shape[1]
        enemies, items = [], []
        for lab in st.labels:
            if lab.width == 0 or lab.object_name.startswith(IGNORE_LABELS):
                continue
            e = {"name": ENEMY_NAMES.get(lab.object_name, pretty(lab.object_name)),
                 "off": (lab.x + lab.width / 2) / W - 0.5, "size": lab.height / H}
            (enemies if lab.object_name in ENEMY_NAMES else items).append(e)
        enemies.sort(key=lambda e: abs(e["off"]))
        items.sort(key=lambda e: abs(e["off"]))
        x, y = self.var("POSITION_X"), self.var("POSITION_Y")
        self.trail.append((x, y))
        self.trail = self.trail[-35:]                      # ~1 s of positions
        moved = float(np.hypot(x - self.trail[0][0], y - self.trail[0][1])) if len(self.trail) > 20 else 999.0
        return {"enemies": enemies, "items": items, "moved": moved, "stuck": moved < 8.0,
                "health": int(self.var("HEALTH")), "armor": int(self.var("ARMOR")),
                "ammo": int(self.var("SELECTED_WEAPON_AMMO")), "kills": int(self.var("KILLCOUNT"))}

    def render(self, s):
        def describe(e):
            off, side = abs(e["off"]), "right" if e["off"] > 0 else "left"
            dist = "very close" if e["size"] > 0.45 else ("close" if e["size"] > 0.25 else "far")
            if off <= CENTRE:
                where = "lined up with the crosshair"
            else:
                where = f"{'far' if off > NEAR else 'just slightly'} to the {side} of the crosshair"
            return f"a {e['name']} {where}, offset {off:.2f} ({dist})"

        seen = ("Visible enemies: " + "; ".join(describe(e) for e in s["enemies"][:4]) + "."
                if s["enemies"] else "No enemies are visible right now.")
        stuff = (" Also in view: " + "; ".join(describe(e) for e in s["items"][:3]) + "." if s["items"] else "")
        move = (" You have barely moved in the last second, so you are probably blocked by a wall or a closed door."
                if s["stuck"] else f" You moved {s['moved']:.0f} map units in the last second.")
        return (f"Doom, {self.title}. You are holding a weapon with {s['ammo']} rounds, health {s['health']}, "
                f"armor {s['armor']}. You can {', '.join(self.actions[:-1])} or {self.actions[-1]}. "
                f"Screen offsets are fractions of the screen width (0 = crosshair, 0.5 = screen edge); one turn step "
                f"moves the view by about 0.05. {seen}{stuff}{move} "
                f"A shot hits only if an enemy is within about 0.03 of the crosshair.")


class Policy:
    """Scores the hypothesis set against a premise. Holds the model; called only from the worker thread."""

    def __init__(self, ckpt, hyps):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.hyps = hyps
        self.tok = AutoTokenizer.from_pretrained(ckpt)
        self.model = AutoModelForSequenceClassification.from_pretrained(ckpt, dtype=torch.bfloat16).cuda().eval()
        self.template = getattr(self.model.config, "nli_template", None) or "Premise: {premise}\nHypothesis: {hypothesis}"
        if self.model.config.get_text_config().pad_token_id is None:
            self.model.config.get_text_config().pad_token_id = self.tok.pad_token_id
        self.tok.padding_side = "right"
        self.backbone = getattr(self.model, self.model.base_model_prefix)

    @torch.no_grad()
    def __call__(self, premise):
        texts = [self.template.format(premise=premise, hypothesis=h) for h in self.hyps]
        enc = self.tok(texts, truncation=True, max_length=512, padding=True, return_tensors="pt").to("cuda")
        h = self.backbone(**enc).last_hidden_state
        last = enc["attention_mask"].sum(1) - 1
        logits = self.model.score(h[torch.arange(h.shape[0], device=h.device), last]).float()
        p = torch.softmax(logits, -1)[:, 1].cpu().numpy()   # P(entailment) per hypothesis
        return int(p.argmax()), p


class Brain(threading.Thread):
    """Consumes the latest premise, publishes the latest decision. Stale premises are dropped, never queued."""

    daemon = True

    def __init__(self, policy, n):
        super().__init__()
        self.policy = policy
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.premise = None
        self.choice, self.probs, self.lat_ms, self.n = 0, np.zeros(n), 0.0, 0
        self.stop_flag = False

    def submit(self, premise):
        with self.lock:
            self.premise = premise
        self.wake.set()

    def latest(self):
        with self.lock:
            return self.choice, self.probs.copy(), self.lat_ms, self.n

    def run(self):
        while not self.stop_flag:
            self.wake.wait(0.1)
            self.wake.clear()
            with self.lock:
                premise, self.premise = self.premise, None
            if premise is None:
                continue
            t0 = time.perf_counter()
            i, p = self.policy(premise)
            dt = (time.perf_counter() - t0) * 1000
            with self.lock:
                self.choice, self.probs, self.lat_ms, self.n = i, p, dt, self.n + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="../openjev_hf/qwen3.5-4b-nli")
    ap.add_argument("--scenario", default="defend_the_center", choices=list(SCENARIOS))
    ap.add_argument("--map", default=None, help="map for freedoom1/freedoom2, e.g. map07 or e1m3")
    ap.add_argument("--profile", default=None, choices=list(PROFILES), help="override the scenario's hypothesis set")
    ap.add_argument("--list", action="store_true", help="show the scenarios and exit")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ticrate", type=int, default=35, help="vanilla Doom is 35")
    ap.add_argument("--timeout", type=int, default=4200, help="episode timeout in tics (35 = 1 s)")
    ap.add_argument("--res", default="RES_800X600")
    ap.add_argument("--sound", action="store_true")
    ap.add_argument("--no-window", action="store_true", help="for testing the loop without a display")
    ap.add_argument("--log-every", type=float, default=3.0, help="seconds between terminal status lines")
    args = ap.parse_args()

    if args.list:
        for k, v in SCENARIOS.items():
            print(f"  {k:26s} {v['profile']:9s} {v['blurb']}")
        return

    scen = SCENARIOS[args.scenario]
    profile = args.profile or scen["profile"]
    game = make_game(scen, args, profile, window=not args.no_window)
    hyps, actions, vectors = build_actions(game, profile)
    title = args.scenario.replace("_", " ") + (f" {args.map or scen.get('map','')}" if "iwad" in scen else "")
    world = World(game, title, sorted(set(actions)))

    print(f"scenario {args.scenario}  profile {profile}  ({scen['blurb']})")
    for h, a in zip(hyps, actions):
        print(f"   {h!r} -> {a}")
    dropped = len(PROFILES[profile]) - len(hyps)
    if dropped:
        print(f"   ({dropped} hypotheses dropped: this scenario's cfg has no button for them)")
    print(f"loading {args.ckpt} ...", flush=True)
    brain = Brain(Policy(args.ckpt, hyps), len(hyps))
    for _ in range(3):
        brain.policy("Doom. No enemies are visible right now.")   # warm the kernels; first pass is ~500 ms
    brain.start()
    print(f"window {'off' if args.no_window else 'ON'}  ticrate {args.ticrate}  ASYNC_PLAYER\n", flush=True)

    all_kills = []
    for ep in range(args.episodes):
        game.set_seed(args.seed + 100 + ep)
        game.new_episode()
        world.trail = []
        t_ep, last_log, n0, tics = time.perf_counter(), 0.0, brain.latest()[3], 0
        s = None
        while not game.is_episode_finished():
            s = world.observe()
            if s is None:
                break
            brain.submit(world.render(s))
            i, p, lat, n = brain.latest()
            game.set_action(vectors[i])
            game.advance_action(1)      # one tic, real-time -- ASYNC blocks until the tic elapses
            tics += 1
            now = time.perf_counter() - t_ep
            if now - last_log >= args.log_every:
                last_log = now
                print(f"  t{now:6.1f}s  kills {s['kills']:2d}  hp {s['health']:3d}  ammo {s['ammo']:3d}  "
                      f"{'STUCK' if s['stuck'] else '     '} | {actions[i]:13s} " +
                      " ".join(f"{x:.2f}" for x in p) +
                      f" | {lat:5.1f} ms, {(n - n0) / max(now, 1e-9):4.1f} dec/s", flush=True)
        k = int(world.var("KILLCOUNT"))
        wall = time.perf_counter() - t_ep
        all_kills.append(k)
        print(f"episode {ep}: {k} kills  hp {s['health'] if s else 0}  {tics} tics in {wall:.1f}s "
              f"({tics / max(wall, 1e-9):.1f} tics/s vs {args.ticrate})  {brain.latest()[3] - n0} decisions\n", flush=True)

    brain.stop_flag = True
    brain.wake.set()
    brain.join(timeout=5)   # let the worker leave its wait() before CUDA tears down, else abort() at exit
    game.close()
    print(f"kills mean {np.mean(all_kills):.2f} max {max(all_kills)}  over {args.episodes} episodes")


if __name__ == "__main__":
    main()
