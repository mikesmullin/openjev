# simplejev-mars

[featherless-ai/simple-jev](https://github.com/featherless-ai/simple-jev) playing **MARS RAID**, driven by
a local `qwen3.8-27b-nvfp4-mtp-q8attn` on llama.cpp.

This is the [`openjev-mars`](../../tree/openjev-mars) demo with the model swapped out. Same game, same
test/control harness, same telemetry idea — different classifier underneath, and a different shape of
question.

| | `openjev-mars` | `simplejev-mars` *(here)* |
|---|---|---|
| model | AlexWortega/openjev, Qwen3.5-4B NLI cross-encoder | qwen3.8-27b-nvfp4-mtp-q8attn (GGUF, NVFP4) |
| engine | Transformers, weights in our own process | llama.cpp `llama-server`, already resident |
| asks | one premise + N hypotheses → P(entailment) each | one state + N **named questions** → typed answers |
| per decision | 1 batched forward pass | 1 prefill per question (3–4 here) |
| decision RTT | ~35 ms | ~1.5 s |

The interesting difference is not the size. openjev could only score *"how true is this sentence"*, so
every decision had to be disguised as a hypothesis — and a sentence can be perfectly true while being a
terrible reason to act. simple-jev asks instead: `choice` over live targets, `choice` for posture,
`score` on a danger rubric, each answered from the logits of its own permitted labels.

## Run it

```bash
git clone --recurse-submodules git@github.com:mikesmullin/openjev.git -b simplejev-mars
cd openjev

uv venv --python 3.12 .venv                                  # pydantic + numpy only
uv pip install --python .venv/bin/python pydantic numpy      # the weights live in llama-server
bun run game                                                 # fetch + patch vibe-arcade's mars.html

~/inference.mjs qwen3.8-27b-nvfp4-mtp-q8attn                 # terminal 1: the model (any llama-server)
bun run model                                                # terminal 2: the simple-jev adapter
bun run dev                                                  # terminal 3: http://127.0.0.1:8734/
```

Then press **run agent**. Game on the left; ranked targets, judgement and latency on the right.

## Architecture

```
browser  web/index.html + web/app.js      m.js page; the agent loop lives here, next to the game
         web/game/mars.html  (iframe)     vibe-arcade's game + one injected bridge line
         web/game/mars-hook.js            state -> classifier state, targets -> choice candidates, autopilot
   |
   v  POST /api/decide
bun      server/static.js                 static files + proxy. No game logic.
   |
   v  POST /v1/classifier
python   server/jev_server.py             ~290 lines. Holds no weights. Knows nothing about Mars.
   |     vendor/simple-jev/common/        the v1 contract, imported from the submodule, never copied
   v  POST /apply-template, /tokenize, /completion
llama.cpp  llama-server :1234             qwen3.8-27b-nvfp4-mtp-q8attn
```

**The code flies; the model judges.** MARS RAID is a continuous 3D flight sim, and geometry is exactly
what a hand-written controller does better. Aiming, throttle and altitude are code. The model answers
*what should we be shooting at*, *should we still be here*, and *how bad is this*, once every ~1.5 s.

## Why llama.cpp and not upstream's `hf-server`

Upstream ships a complete server, `vendor/simple-jev/hf-server/hf_server.py`: it loads a Hugging Face
checkpoint with Transformers, holds the weights itself, and reads next-token logits straight out of the
forward pass. **That is the reference implementation, and none of it is used here.** Three reasons:

1. `qwen3.8-27b-nvfp4-mtp-q8attn` is an **NVFP4 GGUF**. Transformers cannot load it. Using `hf-server`
   would mean a different checkpoint in a different format — not this model.
2. It is already on the GPU. The card has 32 GB and llama-server holds ~19 GB of it; a second 27B copy
   under Transformers does not fit beside it.
3. `hf-server` would be a second inference stack to configure and keep warm, for a model we already have
   loaded and serving.

So `server/jev_server.py` is an **adapter**, not a fork. It imports `vendor/simple-jev/common/` — the
actual v1 contract: prompt text, label assignment, softmax, response shape — and replaces only the engine
boundary. That is the seam upstream documents: *"Shared request validation, versioned prompt
instructions, and response scoring live in the plain Python `common/` folder so other inference
implementations can use the same rules."* `common/PROMPT_STRUCTURE_V1.md` is a language-independent spec
precisely so this is possible. Nothing in `common/` is copied or patched; `git submodule update --remote`
is the whole upgrade path.

Nothing here depends on `inference.mjs` — that is just how this machine happens to start llama-server.
The adapter takes `--llama <url>` and talks to any llama-server. To use upstream's path instead, run
`hf_server.py` with an HF-format checkpoint and point `MODEL_URL` at it; the request/response shape is
the same, because it is the same `common/`.

## Getting label logits out of llama.cpp

v1 needs the raw next-token logits for a few permitted labels, at a deliberately unfinished assistant
turn (`{"answer": "`). Two approaches look right and are not — both measured, not assumed:

- **A GBNF grammar restricting output to the labels, with `post_sampling_probs`.** llama.cpp computes
  those probabilities before the grammar sampler runs, so unconstrained tokens keep their mass and
  nothing is renormalised over the labels.
- **`logit_bias` to force the labels into the top-N.** The returned `logprob` values are the raw ones;
  the bias reaches neither the values nor the list membership.

What works is `n_probs`. `top_logprobs` is the top-N of the true vocabulary log-softmax, so a label that
appears carries its exact value — and log-softmax differs from the raw logit by `log Z`, one constant
shared by every token at that position, which cancels in v1's softmax over the permitted labels alone.
Coverage is handled by escalation (64 → 512 → 4096); the prefill makes label tokens overwhelmingly
likely, so the first rung answers in practice. Measured cost of `n_probs`: **none** (64 vs 0 is 102 ms
either way — it is a sort, not a forward pass).

All 50 choice labels and all 10 digit labels are single-token-stable at the rendered boundary for this
tokenizer, checked per v1 section 9 and cached on the text tail the label follows.

## Latency: why a decision costs ~1.5 s

openjev answered in ~35 ms because it was **one batched forward pass** over all hypotheses by a 4B
encoder, with no generation. simple-jev v1 is structurally different, and the difference is not model
size. Profiled, with the label-boundary cache warm:

```
question     render  tokenize  complete  prompt_n
target           1ms       0ms     637ms       653      <- tokens actually re-evaluated
posture          1ms       0ms     528ms       393
threat           1ms       0ms     260ms       469
wake             1ms       0ms     242ms       387
TOTAL                             1671ms
```

Template rendering and tokenization are free. Decode is free (`predicted_ms = 0.0` — it is one token).
**All of it is prompt evaluation**, and the reason is where v1 puts the question:

```
[system: base + briefing of all questions]  [context/state]  [reminder]  [THE SELECTED QUESTION]
                     shared, cached                                       differs per question
```

The selected question sits *after* the context, and v1 renders its options **twice** (section 5). With
one llama-server slot, cycling through the questions evicts each previous tail, so every question pays
full prefill for its own tail, every tick. The state changes each tick anyway, which invalidates
everything after it regardless. So the cost is `eval(state) + Σ eval(question tails)`, plus a ~100 ms
per-request floor.

That makes the tails the only real lever, and they responded well — trimming candidate counts (3+3+2 → 2+2+2)
and shortening option/rubric wording took a decision from **1920 ms to ~1470 ms** with no loss of quality.
Untested levers, in rough order of expected value: `-np 4` so the questions batch in parallel slots
instead of serialising; a larger `-ub` (currently 512) so a ~650-token tail is not split into micro-batches;
and asking `threat`/`wake` on a slower cadence than `target`.

## `noul` does not survive contact with this model

v1's `noul` type permits the digit labels `1`–`9`, encoding probability 0.1–0.9. Qwen3.8 does not want to
write a bare digit there — it wants to write `0.9`, so the token it actually reaches for is `0`:

```
{"answer":            0    logprob  -0.000     <- not a permitted label
                      1    logprob  -7.892
                      9    logprob -10.998
```

Everything permitted sits 8+ nats down, and the softmax over those runs on noise. Measured: *"Is Mia's
dog named Max?"* (true) → 0.0785 and *"Does Mia own a cat?"* (false) → 0.0706 — no discrimination.

The same judgements as a two-candidate `choice` are clean: **yes 0.9992** and **no 0.9888**. So this
harness uses `choice` and `score` only, and asks yes/no questions as two-candidate choices. This is a
model/label-set mismatch, not a bug in `common/` — and fixing it properly would mean a new template
version, which v1 explicitly reserves.

## The wording lesson, again

Third game in this repo where the fix was language rather than logic — and it survived the change of
model, which is the interesting part. The first saucer option read:

```
Shoot down the alien saucer 447 metres away. It is shooting at the ship,
though more saucers keep spawning, so clearing them is endless.          0.00
Destroy the colony building 656 metres away. Flattening the colony is
the mission and 5 buildings are still standing.                          0.48
```

That trailing clause is true, it is useful context — and it is an argument *against* the option it is
attached to. Saucers scored ~0.01 while the ship was being shot down. **A candidate description has to
argue for its action.** The caveat moved into the shared state, where it informs every question equally,
and the saucer sentence now escalates with the hull:

```
Shoot down the alien saucer 148 metres away. The ship is at 30 percent hull
and cannot finish the mission if it is destroyed first.
```

The `threat` rubric had the mirror-image problem: its levels described the *rate* of incoming damage, so
it read 1.83 with 15% hull left. Not wrong — nothing was getting through at that instant — but not the
question either. Danger is the rate **and** the margin left to absorb it. With both in every level it
reads 0.09 at full hull and 2.90 at 30%.

## Results

One continuous run, agent driving from the menu:

| goal | result |
|---|---|
| destroy the colony | **11 → 0 buildings**, hull still 100 when the last one fell |
| wake the scorpion | `wake` flipped to yes on its own once the colony was gone |
| fight the boss | engaged claws-first by phase; hull 100 → 30 over the fight |
| posture under fire | `break_off` at 0.97 with threat 2.90; `press` at 1.00 while untouched |

Telemetry over that run: **51 decisions, 189 questions, mean RTT 1466 ms, p50 1498, p95 1532**,
~44 ms of that HTTP/proxy overhead, 11–15 labels scored per decision.

Questions per inference moves with the situation — **4** while the scorpion is buried (`wake` is on the
ballot), **3** once it is awake. That is the shared-prefix design doing its job: the state is sent once
and several independent judgements come back from it.

### Honest placement

A heuristic ("shoot whatever is closest and shooting at you") would likely match this on target
selection, as it would have on the openjev branch. The genuine showcase is the boss **phase** constraint —
only the claws, then the tail, then the head can be damaged, and everything else reports *armoured* —
which the model reads out of the candidate list and retargets on, with no phase logic in the agent. And
`posture` is a real second judgement, not a threshold: it vetoes the target choice when the fight is lost.

An in-game caveat: the `KILLS` counter in vibe-arcade's own HUD still reads `0 saucers · 0 buildings`
with the colony flattened. The solo-mode damage patch (below) restores damage, but `awardScore()` is
reached on a path that solo play does not take. Our own `colony left` counter reads the live building
list and is the one to trust.

## The solo-mode bug in the game

`web/game/mars.html` is fetched and patched by `scripts/fetch-game.sh`, pinned to vibe-arcade commit
`0cc97efd`. Worth fixing upstream. `fireTwinLaser` does `const owner = myId || 'me'`, but the three
damage handlers guard with `if (ownerId !== myId) return`. Solo, `myId` is `null` and `owner` is `'me'`,
so **every hit a solo player lands is silently discarded**. The game already has the correct idiom
elsewhere — `(owner === myId) || (!mpReady && owner === 'me')` — and the script applies it to
`mpBuildingHit`, `mpRockHit` and `mpBossHit`.

## Notes

Browsers hold ES modules across reloads even under `no-store`, and a stale module is invisible — it keeps
the old behaviour while you debug code that never runs. `app.js`, `mars-hook.js` and the game iframe are
all loaded with a `?v=` cache-bust. `M.mount()` returns the reactive root instance but does **not** call
`init()` on it; only `x-data` / `x-component` scopes get that automatically.

Bun's default 10 s `idleTimeout` will abort a cold first decision; `server/static.js` raises it to 120 s.
