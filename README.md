# openjev-repro

A local reproduction of [AlexWortega/openjev](https://huggingface.co/AlexWortega/openjev) — Qwen3.5-4B
fine-tuned as a 3-class NLI cross-encoder, used as a game policy by scoring hypotheses about the game state
and taking the argmax entailment. Reproduced on a single RTX 5090, then extended so it can be watched live.

**This is not a GitHub fork.** The original lives only on Hugging Face — all 122 repos on
[github.com/AlexWortega](https://github.com/AlexWortega) were checked and none is an equivalent, which fits
how he worked: his `code/SKILL.md` describes rsyncing to remote GPU boxes and publishing straight to the Hub.
There is nothing on GitHub to fork from.

Attribution is carried in the history instead. The root commit `63ab36c` is his `code/` and
`modeling_openjev.py` fetched verbatim from the HF repo at `8c9db06` and left untouched, so:

```bash
git diff 63ab36c            # exactly our contribution, nothing of his mixed in
```

Hugging Face repos are real git repos, so his history is fetchable directly if you want the true lineage
(his tree tracks the 8.5 GB checkpoint through LFS — clone with `GIT_LFS_SKIP_SMUDGE=1`):

```bash
git remote add upstream https://huggingface.co/AlexWortega/openjev
git fetch upstream          # upstream/main is 8c9db06, the SHA our root commit vendors
```

His work is MIT; so is this.

The 8.5 GB checkpoint is not in git. Fetch it with:

```bash
huggingface-cli download AlexWortega/openjev --local-dir ./openjev_hf
```

---

## 1. The reproduction

Ran his harness against his published `results/*.json`. Same seeds, 5 episodes each.

| policy | his kills | ours | his max | ours | his ms | ours |
|---|---|---|---|---|---|---|
| random | 1.00 | **1.00** | 2 | 2 | – | – |
| oracle (heuristic) | 18.80 | **18.80** | 22 | 22 | – | – |
| zero-shot NLI, `hyp=action` | 1.00 | **1.00** | 1 | 1 | 60.4 | 33.9 |
| zero-shot NLI, `hyp=position` | 11.00 | 10.40 | 16 | 16 | 56.7 | 64.3 |
| zero-shot NLI, `hyp=position_none` | 10.20 | 10.40 | 14 | 16 | 57.5 | 51.5 |
| latent + MLP head | 16.00 | 14.40 | 22 | 20 | 66.4 | 34.5 |

The three deterministic rows match to the last decimal. The two that consult the model drift because bf16
kernels flip an occasional argmax and Doom compounds one different turn into a different episode.

### The published scripts do not reproduce the clip

`code/doom.py` as published predates the run that made the video. Its only hypotheses are
`"The correct action is: {turn left|turn right|attack}"`, which **his own `results/doom_4b.json` scores at
1.0 kills — identical to random.** The clip is `videos/doom_zs_position.mp4` (11.0 kills), produced by a
`--hyp position --zero-shot-only` revision that was never uploaded. We reconstructed those flags from the
hypothesis table in his `results/full_report.md`. `code/flappy.py` has the same gap — the `sign` variant that
scores 28/28 in the report is missing.

That gap *is* the finding. Same frozen weights, same game state; only the wording of the hypothesis changes:

```
"The correct action is: turn left"                     ->  1.0 kills  (= random)
"The nearest enemy is to the left of the crosshair."   -> 10.4 kills
```

Asking the model to **name an action** fails. Asking it to **verify a statement about the world**, and
binding that statement to an action, works. It's a prompt-shape effect, not a capability the NLI fine-tune
conferred.

---

## 2. What we added

### `code/doom.py` — `--hyp` / `--zero-shot-only`

The hypothesis variants needed to reproduce the clip, plus a flag to skip the noisy-oracle collection and
latent-MLP stage that the published script always runs.

### `code/doom_live.py` — headed, real-time

His script runs the engine in `PLAYER` mode with the window hidden, so the simulation blocks on every forward
pass. Correct for scoring, but not watchable and not real-time. Changed:

- **`ASYNC_PLAYER` at a fixed ticrate, window visible** — the world clock keeps running while the model
  thinks. `set_render_all_frames(True)` draws the tics between decisions.
- **Inference on a worker thread** — the main thread advances one tic at a time and applies the most recent
  decision; stale premises are dropped rather than queued. The window holds ticrate regardless of forward-pass
  time, and the policy re-decides as fast as it can instead of on a fixed 4-tic cadence.

Measured: **34.6–35.0 tics/s against a 35 target**, 33–49 ms per decision at 25–29 decisions/s, 13.00 mean
kills (max 14) over 3 episodes. Higher than the recorded 10.4 because the policy now updates at ~27 Hz rather
than 8.75 Hz — same model, same hypotheses, more decisions per second.

### Scenarios — it plays more than the one arena

All 10 bundled ViZDoom scenarios plus the full game (`freedoom1.wad` / `freedoom2.wad` ship with the pip
wheel), via `--scenario` / `--map`. Hypotheses whose action a scenario's cfg cannot express are dropped
automatically, so `deadly_corridor` loses the `use` hypothesis instead of crashing.

Walking a real level needed two fixes beyond adding movement actions:

**Deadlock.** A bare turn or a bare `USE` leaves the player's position unchanged. Once the premise said
"stuck" it said "stuck" forever and the policy spun in place for an entire episode. Recovery actions now
translate the player (`veer right` = turn + forward, `open and step through` = use + forward).

**Aiming.** Measured on this build:

| | degrees | screen offset |
|---|---|---|
| binary turn, 1 tic | 1.76° | 0.020 |
| binary turn, held one decision (1.3–2.5 tics) | 2.3–4.4° | 0.026–0.049 |
| **the hit window** | **±2.7°** | **±0.03** |

One turn step is as wide as the entire window it is trying to stop inside, so overshoot is structural — it
oscillates around the target. Halving the step fixes settling but doubles acquisition time, which costs kills
when enemies close from every side. Instead the model gets a coarse and a fine turn and picks between them,
which is the thing a classifier is actually good at (`TURN_LEFT_RIGHT_DELTA` takes exact degrees/tic):

```
"far to the left of the crosshair."            -> swing left   (6.0°/tic)
"just slightly to the left of the crosshair."  -> nudge left   (1.0°/tic)
"lined up with the crosshair."                 -> attack
"just slightly to the right of the crosshair." -> nudge right
"far to the right of the crosshair."           -> swing right
"No enemy is visible right now."               -> swing left   (scan)
```

Also widened the premise's centre band from 0.015 to 0.03 to match the real hit tolerance. The original
wording called an enemy "left of the crosshair" at an offset a shot would already hit, so it kept turning off
live targets.

> Not yet validated. Single episodes with graded aim scored 16, 11 and 6 kills against 13.00 mean for the
> coarse 3-way set. That is an anecdote, not a result — the matched-seed A/B has not been run. Graded aim also
> doubles the batch (6 hypotheses vs 3), dropping decisions from ~27/s to ~14/s at ~62 ms.

### `code/doom_web.py` — browser front-end

Headless ViZDoom streamed into a web page: MJPEG into an `<img>`, SSE for telemetry. Stdlib plus PIL, no
other dependencies. The sidebar carries what the recorded mp4 HUD carried — per-hypothesis P(entailment),
chosen action, rolling chart, the exact premise text — except live.

Reloading the page only reconnects the streams and never touched the simulation, so there are real controls:
**restart episode** (`r`, same seed — deterministic re-run), **next episode** (`n`, skipped episodes are
excluded from the mean), **pause/resume** (`space`, genuinely freezes the world since ASYNC only advances
when the loop asks).

---

## Running it

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python "transformers>=5.0" accelerate datasets huggingface_hub \
    vizdoom imageio imageio-ffmpeg pillow matplotlib scikit-learn flash-linear-attention
huggingface-cli download AlexWortega/openjev --local-dir ./openjev_hf

cd code

# reproduce the clip (headless, writes json + mp4)
../.venv/bin/python doom.py --ckpt ../openjev_hf/qwen3.5-4b-nli \
    --episodes 5 --seed 0 --hyp position --zero-shot-only \
    --out ../results_repro/doom_zs_position.json \
    --video-nli ../results_repro/doom_zs_position.mp4

# watch it live in a native window
../.venv/bin/python doom_live.py --scenario defend_the_center --profile aim

# watch it live in a browser at http://127.0.0.1:8732/
../.venv/bin/python doom_web.py --scenario freedoom2 --map map01

../.venv/bin/python doom_live.py --list   # all scenarios
```

Needs ~10 GB free VRAM. `flash-linear-attention` is what gets decisions to ~35 ms; `causal-conv1d` is skipped
because the only published version is built against a newer torch ABI than 2.11.

---

## Errata in the write-up this started from (`tmp/GROK1.md`)

- `doom.py --video-nli` does **not** give the clip's policy. `--video` vs `--video-nli` only selects which
  policy gets recorded; the latent MLP is trained either way.
- Training hyperparameters were wrong. `qwen3.5-4b-nli/train_result.json` records the real run:
  `--n-train 120000 --bs 32 --grad-accum 1 --max-len 256 --lr 2e-5 --grad-ckpt`, seed 42, final MNLI-matched
  eval accuracy 0.8985.

## What this is not

Not TypeSafe's Jev, not RLCD, and it inherits none of Jev's calibration claims. openjev cannot emit free text
— only 3-class scores over supplied options — but it can still pick the wrong option with a confident score.
