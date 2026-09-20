# nimble-mars

[bespokelabs/nimble](https://github.com/bespokelabsai/nimble) — **Bespoke-Nimble-9B**, a LoRA fine-tune of
Qwen3.5-9B that makes typed decisions in one forward pass — playing **MARS RAID**.

This is the [`openjev-mars`](../../tree/openjev-mars) demo again, fourth model through it, and the closest
comparison in the repo so far. [`simplejev-mars`](../../tree/simplejev-mars) already measured **stock
Qwen3.5-9B** on this exact harness. Nimble *is* stock Qwen3.5-9B plus a decision fine-tune. Same game, same
state, same questions, same ballot — so the only variable is the tuning, and it can be ablated directly
against its own base, which is what [below](#nimble-against-its-own-base) does.

| | `simplejev-mars` | `nimble-mars` *(here)* |
|---|---|---|
| model | qwen3.8-27b-nvfp4 (and stock Qwen3.5-9B / 2B) | Bespoke-Nimble-9B = Qwen3.5-9B + LoRA |
| engine | llama.cpp `llama-server`, over HTTP | Transformers, weights in our own process |
| asks | one state + N named questions | one context + one flat **schema**, N fields |
| per decision | 1 prefill per question | **1 prefill per decision**, all fields |
| decision RTT | p50 427 ms | **p50 132 ms**, p95 146 ms, 3–4 questions |
| outcome | colony 11→0, scorpion killed | colony 11→0 **at hull 100**, scorpion killed |

The model never writes text. Each answer is one letter code (`A`–`Z`, or true/false), read straight off the
logits and softmaxed over the permitted codes only.

## Run it

```bash
git clone --recurse-submodules git@github.com:mikesmullin/openjev.git -b nimble-mars
cd openjev

uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -r vendor/nimble/requirements/training.txt \
    huggingface_hub flash-linear-attention
uv pip install --python .venv/bin/python torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

bun run model:prepare   # 18 GB: downloads the pinned base and merges the LoRA on the CPU
bun run game            # fetch + patch vibe-arcade's mars.html

bun run model           # terminal 1: the only Python process, holds the weights
bun run dev             # terminal 2: http://127.0.0.1:8734/
```

Then press **run agent**. Game on the left; ranked targets, judgement and latency on the right.
`bun run model:verify` checks the fast scoring modes against upstream's reference before you trust them.

Needs ~18 GB of VRAM for the weights, so nothing else can be resident on a 32 GB card.

**Pin torchvision.** Installing the requirements pulls `torchvision` 0.26 against torch 2.8.0, and its
compiled ops then fail to register (`operator torchvision::nms does not exist`). Transformers' lazy
importer swallows that and re-raises it somewhere else entirely — `ModuleNotFoundError: Could not import
module 'BloomPreTrainedModel'`, from `peft`, which has nothing to do with it. `torchvision==0.23.0+cu128`
is the matching build.

## Architecture

```
browser  web/index.html + web/app.js      m.js page; the agent loop lives here, next to the game
         web/game/mars.html  (iframe)     vibe-arcade's game + one injected bridge line
         web/game/mars-hook.js            state -> context, targets -> enum choices, autopilot
   |
   v  POST /api/decide
bun      server/static.js                 static files + proxy. No game logic.
   |
   v  POST /v1/classifier
python   server/nimble_server.py          ~500 lines. Holds the weights. Knows nothing about Mars.
         vendor/nimble/nimble/scoring/    prompt construction + candidate tokens, from the submodule
```

**The code flies; the model judges.** Aiming, throttle and altitude are code. The model answers *what
should we be shooting at*, *should we still be here*, *how bad is this* and *should we wake the scorpion*
— every tick, in one call.

### The wire contract is still simple-jev v1

Nothing needs it to be. It is v1 so that `web/app.js`, `web/game/mars-hook.js` and `server/static.js` are
the `simplejev-mars` files with the model's name changed, and the two branches stay comparable. The
translation lives in `server/nimble_server.py`:

| v1 | Nimble |
|---|---|
| `choice` with `criteria: {label: why}` | `enum` field, `choices` + `choice_descriptions` |
| `choice` over exactly yes/no | **`boolean`** field — Nimble's own second type — mapped back |
| `score` with an ordered rubric | `enum` over ordered levels, read back as `Σ i · pᵢ` |

The `score` mapping is the one upstream sanctions: *"If a field is an ordered rating scale, your
application can use the probabilities to calculate an expected level."* So the danger reading stays
continuous — 2.21 is a real value, not a rounded 2.

v1's third type, `noul`, is deliberately **not** mapped. It encodes a probability as one of the digits
`1`–`9`, and `simplejev-mars` measured that failing on a Qwen: the model wants to write `0.9`, so it
reaches for `0`, every permitted label sits 8+ nats down and the softmax runs on noise. Nimble's boolean is
the type that question actually wants.

`vendor/nimble` is a submodule, never a copy, so prompt text and the candidate-token checks stay upstream's
and `git submodule update --remote` is the whole upgrade path.

### The prompt budget is 2048, not 8192

Upstream's serving path now permits 8192 tokens, but `schema_config.json` records `max_length: 2048` and
that is what the adapter was trained at. Going past it is not rejected — it answers, just from outside its
training distribution, which is the worse failure. The harness budgets against the contract. In practice a
full four-question decision with nine candidates is ~660–1100 tokens.

The release is a **165 MiB LoRA adapter**, not a checkpoint: it pins `Qwen/Qwen3.5-9B` at an exact revision
and must be merged against it. `scripts/prepare-model.py` does that and treats `prompt_code_sha256` as
fatal — the prompt text is part of the trained contract, so a submodule whose `parallel_schema.py` has
drifted is a different model, not a warning.

## One prefill per decision, not one per field: 315 ms → 115 ms

Upstream's `CudaCandidateScorer` loops the fields and runs each one's full prompt with `use_cache=False`.
Their docs say so plainly: *"The CUDA scorer scores each field on its own, with the full prompt each
time."* Only the Mac/MLX `ParallelScorer` prefills the shared context once. That leaves the same
N-prefills shape that cost `simplejev-mars` its latency — except here the weights are in our own process
and nothing forces it.

It is worth avoiding because Nimble's prompt is the *opposite shape* to v1's. v1 puts the selected question
after the context and renders its options twice, so each question carries a large tail of its own. Nimble
renders the entire schema once and varies only a trailing `Requested field: "name"`. Measured on the real
payload: **648 of 661 tokens are shared by all four fields**, and each field's suffix is 12–13 tokens.

`prepare_prompts()` already returns that split as `prefix_ids`/`suffix_ids`. Only MLX used it. Two modes now do:

| mode | what it does |
|---|---|
| `independent` | upstream's, unchanged — the reference for correctness |
| `batched` | one padded batch of all N full prompts; one pass, still N prefixes of compute |
| `prefix` | prefill the prefix once, then one batched step over the suffixes against its cache |

```
                     1 question   2        3        4 questions
independent              54.8    107.4    230.1    315.6 ms
batched                  54.8     91.8    168.0    238.5 ms
prefix                   76.5     83.5    110.8    115.1 ms
```

The fourth question costs **4.5 ms**. Below two questions `prefix` is a *loss* — chunking the pass costs
something and there is nothing to share yet — so the crossover is at two, and the default is `prefix`
because a decision here is three or four.

### Widening the cache is the hard part, because Qwen3.5 is hybrid

Only 8 of its 32 layers hold an ordinary `keys`/`values` KV cache. The other 24 are linear attention,
holding a fixed-size recurrent state plus the causal conv's left context:

```
LinearAttentionLayer   conv_states {0: (1, 8192, 4)}   recurrent_states {0: (1, 32, 128, 128)}
DynamicLayer           keys (1, 4, P, 256)             values (1, 4, P, 256)
```

Replicating only the KV half — what a pure-attention model would need — fails loudly, which is the good
case. Replicating the recurrent state across rows is legitimate: every row continues from the same prefix,
so they genuinely share both the summary and the conv's left context.

### The leftover logit differences are arithmetic, and here is the control that shows it

BF16 matmul is not associative, so changing the *shape* of the computation moves the last bits, and both
fast modes change it. Three measurements separate that from a bug:

```
batched, rows all the same length, zero padding      0.19     so it is not padding
batched, one row at a time (batch 1, no padding)     0.0000   exact
chunked prefill, batch 1, NO cache widening at all   0.3618   == prefix mode's own gap, exactly
```

The last line is the one that matters: splitting the pass reproduces the whole of `prefix` mode's
disagreement **without widening anything**, so the gap is not a broken mask or a mis-replicated recurrent
state. So `--verify` checks what actually matters — the argmax and the probabilities the ballot is ranked
on. Over **24 varied game states and 86 field decisions, both fast modes agree with upstream's on 86/86**,
max |Δp| 0.056.

`causal_conv1d` is still missing — its prebuilt wheel fails against torch 2.8.0 with the same
`undefined symbol: _ZN3c104cuda19CUDAErrorLogCaptureC1Ev` the `simplejev-mars` branch hit — so the
linear-attention conv is on its reference PyTorch path and there is speed still on the table.

## Nimble against its own base

The interesting experiment, because it is controlled: identical prompts, identical schema, same scorer,
same GPU. Only the LoRA differs.

| case | Bespoke-Nimble-9B | stock Qwen3.5-9B |
|---|---|---|
| hull 15, bleeding → posture | **break_off 0.950** | press 0.625 ✗ |
| hull 100, untouched → threat | **0.03 / 3** | 0.73 / 3 |
| hull 15, bleeding → threat | 2.21 / 3 | 2.61 / 3 |
| colony still standing → wake? | **no 0.990** | **yes 0.628** ✗ |
| colony flattened → wake? | yes 0.993 | yes 0.785 |
| boss phase, only the claw vulnerable | claw 0.993, armoured ~0.00 | claw 0.999, armoured 0.000 |
| hold offered beside real targets | building 0.883 | building 0.856 |
| hold is the honest answer | hold 0.974 | hold 0.969 |

The fine-tune earns its place on the **judgement** calls, not on target selection. The base model keeps
pressing the attack at 15 percent hull while bleeding, and wakes the scorpion with six buildings still
standing — the exact failure that loses the game. Nimble gets both right and is far better calibrated on
an idle threat reading (0.03 vs 0.73 at full hull, untouched).

Two honest negatives:

**The boss-phase showcase is not evidence of the fine-tune.** Both models read the armoured constraint out
of prose perfectly and both retarget on the vulnerable part. This branch family has claimed that as its
showcase since `openjev-mars`; on a 9B it is simply not hard.

**The `other:hold` trap did not reproduce.** `simplejev-mars` found stock Qwen3.5-9B taking "hold fire" at
0.98 every tick and flying a whole game with `shots: 0`. Under Nimble's prompt neither model does — both
pick a real target when one exists (0.883 / 0.856) and both correctly hold when nothing is in range (0.974
/ 0.969). So that trap was a property of **v1's prompt shape**, not of the 9B weights. Worth recording,
because the obvious story — "the fine-tune fixed the hold trap" — is wrong.

## Results

One continuous run, agent driving from the menu, `prefix` mode:

| goal | result |
|---|---|
| destroy the colony | **11 → 0 buildings**, hull still **100** as the last one fell |
| wake the scorpion | flipped to yes on its own once the colony was gone |
| kill the scorpion | **dead** — worked through all three phases, claws → tail → head |
| survive | ship lost to saucers *after* the boss died, at hull 20, both objectives complete |

**151 decisions, 522 questions, mean RTT 135 ms, p50 132 ms, p95 146 ms**, ~3 ms of that HTTP/proxy
overhead. 3 questions per decision for most of the raid and 4 while the scorpion was still buried — the
RTT trace is almost flat, because the extra questions ride the same prefill.

For comparison: `openjev-mars` got the scorpion to 23 percent before the ship was lost; `simplejev-mars`
killed it with a 27B at p50 427 ms and hull 93 when the colony fell. This is a 9B at p50 132 ms that
flattened the colony without taking a hit.

### What decided it: bounding the break-off

`posture: break_off` vetoes shooting entirely, and it was firing on a bare plurality. Measured mid-game at
hull 82: posture read `break_off` **0.58** while target read *destroy the colony building* **0.75**, and
the raid sat at 7 of 11 buildings for ninety seconds — evading, never firing. The target question was
right and the veto was overriding it on a coin flip.

Requiring the veto to be *decisive* (p ≥ 0.75) is what the probabilities are for. An argmax-only interface
would have to act on 0.58; a distribution lets the agent treat "narrowly break off" as "keep fighting, but
this is going badly". Not calibration — upstream is explicit that 0.9 does not mean right 90 percent of the
time — just a threshold tested on the task, which is exactly what they recommend doing with it.

That alone was not enough, and the second half is the more interesting failure. `evade()` flies 600 m from
the nearest saucer; saucers respawn without limit; once the colony is gone and twelve are airborne,
breaking off is defensible on nearly every tick. The scorpion's tail then sat at **73 percent for four
minutes** while `target` kept correctly picking it (0.36–0.63 across ticks) and the ship never closed —
approach, take fire, flee 600 m, repeat. The aim telemetry named it exactly:

```
tail radius 4.2 m   dist 297 m   angle 1.53 deg   ->   miss 7.9 m   gate needs < 4.6 m
```

`miss` is `sin(angle) * dist`, and the gate is the game's own tolerance, `radius + 0.4`. At that range a
4.2 m part needs the aim inside 0.81 degrees; the jink holds 1.53. The veto was not wrong
about the danger; it was simply never allowed to end. So a break-off now lasts at most 6 consecutive
decisions, after which the agent must press for 8 before it may break off again. With the bound the ship
closes to ~60 m, and the same fight that stalled at 73 percent went claws → tail → head and killed it.

Same lesson the `laya-mars` branch wrote down independently: *break off as a last resort, and bound it.*

### Honest placement

A heuristic ("shoot whatever is closest and shooting at you") would still likely match this on target
selection, as on every branch before it. What this model actually buys is the other three questions, and
the ablation above is the evidence: its own base model, on identical prompts, presses the attack at 15
percent hull and wakes the boss too early. Those are the two decisions that lose the run.

The scorpion kill is not attributable to the model alone — it took the bounded break-off, which is harness
logic. And the in-game `KILLS` counter still reads `0 saucers · 0 buildings` with the colony flattened:
the solo-mode damage patch restores damage, but `awardScore()` is reached on a path solo play does not
take. Our own `colony left` counter reads the live building list and is the one to trust.

Nimble is not Jev, and upstream says so: 2,676 training examples across ten domains, built in a day, 90.1
percent reference-label agreement against Jev's 93.2. It cannot emit free text — only codes over supplied
answers — and it can still pick the wrong one with a confident score.

## The solo-mode bug in the game

`web/game/mars.html` is fetched and patched by `scripts/fetch-game.sh`, pinned to vibe-arcade commit
`0cc97efd`. Worth fixing upstream. `fireTwinLaser` does `const owner = myId || 'me'`, but the three damage
handlers guard with `if (ownerId !== myId) return`. Solo, `myId` is `null` and `owner` is `'me'`, so **every
hit a solo player lands is silently discarded**. The game already has the correct idiom elsewhere —
`(owner === myId) || (!mpReady && owner === 'me')` — and the script applies it to `mpBuildingHit`,
`mpRockHit` and `mpBossHit`.

## Notes

Browsers hold ES modules across reloads even under `no-store`, and a stale module is invisible — it keeps
the old behaviour while you debug code that never runs. `app.js`, `mars-hook.js` and the game iframe are all
loaded with a `?v=` cache-bust. `M.mount()` returns the reactive root instance but does **not** call
`init()` on it; only `x-data` / `x-component` scopes get that automatically.

Bun's default 10 s `idleTimeout` would abort a cold first decision; `server/static.js` raises it to 120 s.
