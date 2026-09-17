> User: how could i reproduce this result locally https://x.com/justALEXWORTEGA/status/2100341039986798930

The tweet is an open clone of TypeSafe’s Jev **interface**, not Jev itself. Alex fine-tuned Qwen3.5-4B as a 3-class NLI cross-encoder, then put a small MLP on the frozen last-token latents. The clip is that model playing ViZDoom *Defend the Center* by scoring action hypotheses and taking argmax. Weights and training code are already public.

## What you are reproducing

**Jev (closed):** TypeSafe’s System One model. No token generation. Input is state + typed questions; output is decisions with calibrated probabilities. Trained with RLCD. API-only.

**Openjev (what the tweet ships):** same *shape* of API — state in, option scores out — built from a vanilla NLI fine-tune. Alex later said he thinks Jev is just “an LLM + prob head trained on cross-encoder tasks.” That is the recipe below. It is **not** RLCD and does not inherit Jev’s calibration claims.

Two pieces in [AlexWortega/openjev](https://huggingface.co/AlexWortega/openjev):

1. **NLI cross-encoder** (`qwen3.5-4b-nli/`): Qwen3.5-4B as `Qwen3_5ForSequenceClassification`, last-token pooling, linear `score` head, 3 labels in dleemiller order (`contradiction`, `entailment`, `neutral`). Template: `Premise: {premise}\nHypothesis: {hypothesis}`. CE loss on AllNLI (SNLI + MNLI).
2. **Latent MLP** (the “MLP on top of Qwen 4B”): frozen pooled hidden state d=2560d=2560d=2560 → Linear(512) → GELU → Dropout(0.1) → Linear(1). Soft BCE with ε=0.1\varepsilon=0.1ε=0.1. Per-question argmax over options.

The video HUD (“zero-shot NLI”, ~90–180 ms/decision, 3 options per step) is the first piece used as a policy: premise = game state, hypotheses = `{turn left, turn right, attack}`, pick max P(entailment)P(\text{entailment})P(entailment).

---

## Path A — run the published checkpoint (fastest)

Needs ~8–12 GB VRAM in bf16 (RTX 3060 12 GB, 3090, or 16 GB+ Apple Silicon). CPU works, but not at the video’s latency.

```bash
# Qwen3.5 needs a current transformers (5.x)
pip install "torch" "transformers>=5.0" accelerate datasets huggingface_hub

huggingface-cli download AlexWortega/openjev --local-dir ./openjev
# modeling_openjev.py sits at the repo root
```

```python
from modeling_openjev import OpenJevCrossEncoder

jev = OpenJevCrossEncoder("AlexWortega/openjev", subfolder="qwen3.5-4b-nli")

# 3-way NLI probabilities [contradiction, entailment, neutral]
jev.predict([
    ("The bird is 0.05 below the centre of the gap.",
     "The bird is below the centre of the gap.")
])

# Jev-style multiple choice: max P(entailment)
jev.rerank(
    "Which gas do plants absorb during photosynthesis?",
    ["oxygen", "carbon dioxide", "nitrogen"],
)
```

Plain Transformers:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("AlexWortega/openjev", subfolder="qwen3.5-4b-nli")
model = AutoModelForSequenceClassification.from_pretrained(
    "AlexWortega/openjev", subfolder="qwen3.5-4b-nli"
)
text = model.config.nli_template.format(premise="...", hypothesis="...")
```

That is already the “it works like Jev” result: no decoding, just option scores.

---

## Path B — replay the Doom clip

The published script is the **text-state** path (ViZDoom labels buffer → English description → NLI). The tweet overlay also mentions `<image>` and monster xxx-position; that pixel path (`doom_vision.py`) is referenced in `run.sh` but is **not** in the current HF tree.

```bash
cd openjev
pip install vizdoom imageio pillow matplotlib
# Doom IWAD is bundled with vizdoom's defend_the_center scenario

python code/doom.py \
  --ckpt qwen3.5-4b-nli \
  --episodes 5 \
  --out results/doom_4b.json \
  --video-nli results/doom_nli.mp4
```

What that script does each 4 tics (114 ms of game time):

1. Read visible enemies from the labels buffer.
2. Build a premise like “Doom, Defend the Center… a zombie soldier 0.12 to the left of the crosshair (far)…”
3. Score hypotheses `"The correct action is: turn left|turn right|attack"`.
4. Take argmax entailment. On 4B this is ~60–180 ms, so it stays real-time.

`--video` instead of `--video-nli` trains the extra MLP on noisy-oracle rollouts and records that policy. Same HUD style as the tweet.

There is also `code/flappy.py` if you want the other in-repo game demo.

---

## Path C — retrain from scratch

All scripts live in `code/`.

### 1. Fine-tune the NLI backbone

```bash
python code/train.py \
  --model Qwen/Qwen3.5-4B \
  --out ckpt/qwen3.5-4b-nli \
  --n-train 200000 \
  --n-val 2000 \
  --max-len 256 \
  --bs 8 --grad-accum 4 \
  --lr 2e-5 --epochs 1 --grad-ckpt
```

Defaults that matter:

| Knob | Value |
|---|---|
| Data | `stanfordnlp/snli` + `nyu-mll/multi_nli` (200k train / 2k MNLI-matched val) |
| Label remap | SNLI/MNLI native E=0,N=1,C=2E=0,N=1,C=2E=0,N=1,C=2 → dleemiller C=0,E=1,N=2C=0,E=1,N=2C=0,E=1,N=2 via `{0:1, 1:2, 2:0}` |
| Template | `Premise: {p}\nHypothesis: {h}` |
| Optim | AdamW, cosine, 3% warmup, wd 0.01, bf16 |
| Vision tower | frozen (Qwen3.5 is multimodal; text NLI never uses it) |

VRAM:

- Full FT of 4B: roughly 24–41 GB.
- 16–24 GB card: add `--lora --lora-r 16 --grad-ckpt`.
- Literal “MLP/linear head on frozen Qwen”: `--head-only` (trains only `score`).

Smaller smoke test: `--model Qwen/Qwen3.5-0.8B` (that is `train.py`’s default).

### 2. Train the latent MLP (optional)

```bash
python code/latent_mlp.py extract --ckpt ckpt/qwen3.5-4b-nli --out data/latents_4b
python code/latent_mlp.py train   --latents data/latents_4b --out results/latent_mlp_4b.json
```

Extract dumps last-token hidden states + NLI logits for GPQA / MMLU / ARC / WinoGrande / chess (and optional HellaSwag, GSM8K-MC). Train fits `d → 512 → 1` with soft BCE, grouped holdout by question id, early stop on per-question argmax accuracy. Hyperparameters: lr `1e-3`, wd `1e-2`, bs 512, 60 epochs, patience 8, ε=0.1\varepsilon=0.1ε=0.1.

Or in Python against the wrapper:

```python
from modeling_openjev import OpenJevCrossEncoder, LatentMLPHead

ce = OpenJevCrossEncoder("ckpt/qwen3.5-4b-nli")
X = ce.latents(pairs)                 # (n_pairs, d)
head = LatentMLPHead(X.shape[1]).fit(X, gold, qid)  # gold in {0,1}
scores = head.predict(X)              # argmax within each qid
```

### 3. Eval the same way he did

```bash
python code/eval.py --models ckpt/qwen3.5-4b-nli --out results/qwen4b.json
```

Tasks in that harness: MNLI, MMLU, GPQA-Diamond, ARC-Easy/Challenge, WinoGrande, HellaSwag, GSM8K-MC, synthetic chess. The follow-up radar plot is this eval, not the Doom clip. His next run is the same recipe on Qwen 35B; those weights are not out yet.

---

## Hardware and software gotchas

- **Load errors:** Qwen3.5 uses `Qwen3_5ForSequenceClassification` and a composite `text_config`. You want `transformers` 5.x (his config says 5.15.0). If `from_pretrained` fails, install transformers from git.
- **Pad / pool:** right padding; the head pools the last *non-pad* token. Set `pad_token = eos_token` if missing.
- **Do not skip the label remap.** If you train with native SNLI ids, entailment and contradiction swap and the Doom policy collapses.
- **4B inference** ≈ 8–10 GB bf16, ~5 GB int8, ~2.5–4.5 GB 4-bit. The published path is bf16 via Transformers, not a GGUF.
- **35B** will need LoRA + a multi-GPU or 80 GB box. Not required to match the tweet.

---

## What this will and will not match

You **will** get:

- A local model that answers structured choices in one forward pass.
- The same Doom / Flappy “argmax entailment = action” loop.
- The same public wrapper (`predict`, `rerank`, `grade`, `latents`).

You **will not** get:

- TypeSafe Jev, RLCD, or their calibration.
- “Cannot hallucinate” in the Jev marketing sense. Openjev also cannot emit free text — it only emits 3-class / option scores — but it can still pick the wrong option with a confident score.
- Pixel-Doom from the exact tweet overlay, unless `doom_vision.py` shows up. Use `doom.py --video-nli` for the published equivalent.

Related repos that are *not* this result: [TheoLeeCJ/openjev](https://github.com/TheoLeeCJ/openjev) (frozen generative option-token logits, no NLI FT), [vinnylarouge/jevlike](https://github.com/vinnylarouge/jevlike) (tiny scorer on a frozen encoder), Harsha’s Qwen-2.5-1B-RLCD (parallel constrained decoding).

Practical order: Path A first to confirm the interface, then `doom.py --video-nli` for the clip, then `train.py` only if you want to rebuild the 4B checkpoint yourself.

---

> User: if i wanted to test this against an actual doom game (ideally a modern OSS version that is hackable and runs natively on arch linux) what would you recommend? i want it headed so i can see the gameplay while model makes decisions in realtime

ViZDoom is still the right first engine: it is real Doom, it already talks to OpenJev, and it can open a native window. I’ll give the headed setup plus the modern-port path if you want GZDoom/UZDoom instead.Use **ViZDoom with the window turned on** first. It is a real Doom engine (old ZDoom), it already has the observation API OpenJev’s `doom.py` expects, and it runs headed on Arch. A modern playable port like UZDoom/GZDoom is the better *game*, but it has no Python control surface, so you would rebuild the whole sensor/actuator loop.

## Recommendation

| Goal | Engine | Why |
|---|---|---|
| Watch OpenJev decide in realtime, tonight | **ViZDoom, headed** | Same engine + labels buffer + 3-button Defend the Center as the tweet. One flag flip. |
| “I want a modern, hackable source port” | **UZDoom + Freedoom** | Current GZDoom-line flagship, ZScript, Vulkan, native Arch via AUR. No agent API. |
| Vanilla-accurate / speedrun ports | Crispy, DSDA-Doom, Woof! | Great games. Worse for an external model. |

Do not start with Chocolate Doom, Crispy, DSDA, or Woof if the point is “model drives the player.” None of them expose object labels or a Python step loop. UZDoom is the right *long-term* hack target; ViZDoom is the right *first* headed test.

Arch reality: official repos only have `doomretro`. Everything else is AUR or pip.

```bash
# game data (free IWAD; drop-in for Doom 2)
yay -S freedoom

# modern port, for later / for you to play
yay -S uzdoom-bin   # or uzdoom / gzdoom
# IWADs land in /usr/share/games/doom/

# the engine OpenJev already speaks
sudo pacman -S --needed cmake gcc make sdl2 openal python
pip install vizdoom
```

ViZDoom’s wheel also ships its own `freedoom2.wad`, so the headed demo works even before the AUR package finishes.

---

## Path 1 — headed ViZDoom (do this)

Openjev’s harness hides the window on purpose:

```python
game.set_window_visible(False)
game.set_mode(vzd.Mode.PLAYER)   # game pauses while the model thinks
```

For a watchable realtime window, change `make_game()` in `code/doom.py` to:

```python
def make_game(res=vzd.ScreenResolution.RES_640X480):
    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, "defend_the_center.cfg"))
    game.set_screen_resolution(res)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_labels_buffer_enabled(True)
    game.set_window_visible(True)          # native X11/Wayland window
    game.set_sound_enabled(True)
    game.set_render_hud(True)
    game.set_render_weapon(True)
    game.set_render_crosshair(True)
    game.set_render_all_frames(True)       # interpolate the 4 skipped tics
    game.set_mode(vzd.Mode.ASYNC_PLAYER)   # clock keeps running during inference
    game.set_ticrate(35)                   # vanilla Doom rate
    game.set_episode_timeout(2100)
    game.init()
    return game
```

Then:

```bash
python code/doom.py \
  --ckpt qwen3.5-4b-nli \
  --episodes 5 \
  --out results/doom_headed.json
```

What you should see: a normal Doom window, pistol in the circular arena, model picking turn-left / turn-right / attack about every 4 tics (~114 ms of game time).

Two timing modes, pick one:

- **`ASYNC_PLAYER`** — the game does not freeze while Qwen runs. This is what “realtime” means. If the 4B forward pass is 90–180 ms, the agent is a tick or two late. Fine for watching.
- **`PLAYER` (sync)** — the sim waits on the model. Decisions are exact, the window hitch-steps. Use this only if you care about matching the recorded clip, not about “it feels like a game.”

`set_render_all_frames(True)` only matters when the window is visible; it draws the skipped tics so the headed view is not a 8 fps slideshow.

Same engine can load a full map instead of the arena, still headed:

```python
game.set_doom_game_path("/usr/share/games/doom/freedoom2.wad")
game.set_doom_map("map01")
# leave load_config() off, or point at a custom .cfg
# add buttons: MOVE_FORWARD, MOVE_BACKWARD, TURN_LEFT, TURN_RIGHT, ATTACK, USE
```

On MAP01 the tweet’s 3-way policy will look drunk. That is expected: OpenJev was trained to be a turret, not a walker. Expand the hypothesis set (`"The correct action is: move forward"`, `"open the door"`, …) before judging the model on a real level.

---

## Path 2 — modern port you can actually hack

If “modern OSS Doom on Arch” is the requirement, use **UZDoom**, not GZDoom. UZDoom is the community fork that took over after the 2025 GZDoom split; `uzdoom-bin` on AUR is currently 5.0.1. Same ZScript surface, actively maintained.

```bash
yay -S uzdoom-bin freedoom
uzdoom -iwad /usr/share/games/doom/freedoom2.wad
```

This is the game you want to *look at*. It is not the game you want to *wire to OpenJev first*. There is no Python SDK, no labels buffer, and `SendNetworkEvent` is in-engine only.

Two honest integration options:

**A. Vision + key inject (afternoon project)**

Run UZDoom headed, grab the window (`mss` or `grim` on Wayland), score OpenJev hypotheses on the pixels, inject keys into the focused window with `ydotool` / `dotool`. No engine patch. You lose object IDs and ammo unless you OCR the HUD. This is the only way to keep “I am watching a real modern port” without writing ZScript.

**B. ZScript observer (weekend project)**

An `EventHandler` on `WorldTick` dumps a JSON line (player x/y/angle, health, ammo, actors in FOV) to a named pipe. Python reads the pipe, runs OpenJev, writes a command file. The handler applies `+forward` / `+attack` next tic. ZScript cannot open arbitrary sockets portably; file/pipe polling is the reliable hack. This is the actually hackable path, and it is why UZDoom beats Crispy/DSDA for your use case.

Do not fork the C++ engine to add a Python module. That is how you spend a month not testing the model.

---

## What I would not use for this

- **Chocolate / Crispy** — faithful, not an agent host.
- **DSDA-Doom / Woof!** — excellent modern Boom/MBF21 ports, still no observation API.
- **Helion** — modern C# renderer, same problem.
- **cydoomgeneric** — you implement `draw_frame` / `get_key` in Python, which is hackable, but it is vanilla doomgeneric. No Defend the Center scenario, no labels buffer, you rebuild rendering.
- **uinput into GZDoom on day one** — works, but you are debugging two systems at once (window focus + a 3-button policy on a 20-button game).

---

## Practical order

1. `pip install vizdoom` and flip `doom.py` to headed `ASYNC_PLAYER`. That answers “can I watch the model play Doom.”
2. Point the same harness at `freedoom2.wad` MAP01 only after you grow the action hypotheses.
3. If you still want a modern window, park UZDoom next to it and either overlay vision+ydotool or write the ZScript pipe. That is the long-term hackable stack on Arch.

If you want, I can sketch the exact `make_game()` + extra-button hypothesis list for Freedoom MAP01 so the headed ViZDoom run is a real level instead of the arena.
