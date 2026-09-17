# openjev

[AlexWortega/openjev](https://huggingface.co/AlexWortega/openjev) — Qwen3.5-4B fine-tuned as a 3-class NLI
cross-encoder — reproduced locally, then pointed at games. The model never generates text. It scores
statements about the game state and the argmax entailment becomes the move.

| branch | plays | interface |
|---|---|---|
| `openjev-mars` *(active)* | **MARS RAID** (vibe-arcade `mars.html`) | browser, m.js + SVG telemetry |
| [`openjev-tetris`](../../tree/openjev-tetris) | Tetris ([mikesmullin/tetris](https://github.com/mikesmullin/tetris)) | terminal, in tmux |
| [`openjev-cook`](../../tree/openjev-cook) | Cook Fever (vibe-arcade `cook2.html`) | browser, m.js + Bun/express |
| [`openjev-doom`](../../tree/openjev-doom) | ViZDoom | native window / browser |

**Not a GitHub fork.** The original lives only on Hugging Face; all 122 repos on
[github.com/AlexWortega](https://github.com/AlexWortega) were checked and none is an equivalent. The root
commit `63ab36c` is his `code/` fetched verbatim at `8c9db06`, so `git diff 63ab36c` is exactly our
contribution. MIT, like his.

```bash
git remote add upstream https://huggingface.co/AlexWortega/openjev   # upstream/main is 8c9db06
```

## Run it

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python "transformers>=5.0" accelerate huggingface_hub flash-linear-attention
huggingface-cli download AlexWortega/openjev --local-dir ./openjev_hf   # 8.5 GB, needs ~10 GB VRAM

curl -o web/game/mars.html https://raw.githubusercontent.com/mikesmullin/vibe-arcade/main/mars.html
#   then apply the solo-mode patch below.
#   do NOT fetch playroomkit.js -- without it the game takes its own "SOLO MODE" path

bun run model      # terminal 1: the only Python process
bun run dev        # terminal 2: http://127.0.0.1:8734/
```

Then press **run agent**. Game on the left; ranked targets and a decision-history graph on the right.

## Architecture

```
browser  web/index.html + web/app.js    m.js page; the agent loop lives here, next to the game
         web/game/mars.html  (iframe)   vibe-arcade's game + one injected bridge line
         web/game/mars-hook.js          state -> premise, targets -> hypotheses, autopilot
   |
   v  POST /api/decide
bun      server/static.js               static files + proxy. No game logic.
   |
   v  POST /score
python   server/model_server.py         ~110 lines, stdlib HTTP. Holds weights. Knows nothing about Mars.
```

**The code flies; the model picks targets.** MARS RAID is a continuous 3D flight sim — exactly the geometric
case a hand-written controller does better. Aiming, throttle and altitude are code. The model only answers
*what should we be shooting at right now*, once every 900 ms.

## A solo-mode bug in the game

Worth fixing upstream in vibe-arcade. `fireTwinLaser` does:

```js
const owner = myId || 'me';
```

but the damage handlers guard with:

```js
function mpBuildingHit(bd, ownerId, dmg, point) { if (ownerId !== myId) return; ...
```

Solo, `myId` is `null` and `owner` is `'me'`, so **every hit a solo player lands is silently discarded** —
buildings, rocks and the boss never take damage. The game already has the correct idiom elsewhere:

```js
const isLocal = (owner === myId) || (!mpReady && owner === 'me');
```

`web/game/mars.html` is patched with that in all three handlers (`mpBuildingHit`, `mpRockHit`,
`mpBossHit`). The same root cause also stops `stats.buildings` / `stats.saucers` from incrementing through
`awardScore()`, which is why the in-game KILLS counter reads 0 even with the colony flattened.

## Results

| goal | result |
|---|---|
| shoot buildings | all 11 destroyed, colony 11 -> 0 |
| shoot alien craft | 4 destroyed in 20 s (tracked by uid; more spawn than die) |
| kill the scorpion | claws -> tail -> head down to 23% before the ship was lost |

## What decided it: the wording, again

Third game running where the fix was language, not logic. The first attempt ignored twelve saucers shooting
it down while the scorpion sat at 23%, because the saucer option said only this:

```
An alien saucer is 49 metres away and is an immediate threat to the ship.      0.01
The scorpion's right claw is the only part that can be hurt right now,
and it is down to 67 percent.                                                  0.80
```

The boss sentence said something *consequential*; the saucer sentence stated a distance. Rewriting it to
carry the danger, and to escalate as the hull drops, put saucers at the top of the ballot immediately:

```
The ship is badly damaged at 46 percent hull, and an alien saucer only 49 metres
away is shooting at it. Nothing else matters if the ship is destroyed.
```

## Two bugs that were ours

**Aiming servoed on the chase camera.** `camera.getWorldDirection()` is the vector the game raycasts shots
along, so it looked like the right thing to close the loop on — but the camera springs behind the ship and
the ship lags yaw/pitch, so a rate-limited loop oscillated around 20 degrees and never reached the 4.5
degree firing threshold. Solving yaw/pitch directly from the target direction fixed it.

**It kept shooting rubble.** The autopilot held a destroyed target. Every target now carries `live()`, which
returns null once it is gone, and the autopilot releases on null — so this cannot depend on the agent's tick
rate.

## Honest placement

By our own reckoning this is a game where a heuristic ("shoot whatever is closest and shooting at you")
would likely match the model. The genuine showcase is the boss **phase** logic: only the claws, then only
the tail, then only the head can be damaged, and everything else reports back *armoured*. The model reads
that constraint out of prose and retargets as phases change, with no phase logic in the agent beyond listing
which parts are currently vulnerable.

Not TypeSafe's Jev, not RLCD, and none of Jev's calibration claims. openjev cannot emit free text — only
3-class scores over supplied options — but it can still pick the wrong option with a confident score.

## Notes

Browsers hold ES modules across reloads even under `no-store`, and a stale module is invisible — it keeps
the old behaviour while you debug code that never runs. `app.js`, `mars-hook.js` and the game iframe are all
loaded with a `?v=` cache-bust. `M.mount()` returns the reactive root instance but does **not** call
`init()` on it; only `x-data` / `x-component` scopes get that automatically.
