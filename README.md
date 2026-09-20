# verdict-mars

[openJev-verdict-2.0](https://github.com/Heman10x-NGU/openJev-verdict-2.0) — a ~151M ModernBERT-class
encoder that scores a whole ballot in one non-autoregressive forward pass — playing **MARS RAID**.

Built on [`nimble-mars`](../../tree/nimble-mars), so the game, the harness, the ballot and the wire
contract are unchanged and only the model moved. It is the **fastest** thing this repo has put in the
cockpit — a four-question decision is 12 ms, p50 in-game is 18 ms — and it is also the one doing the
**least** of the deciding, for reasons measured below rather than asserted.

| | `simplejev-mars` | `nimble-mars` | `verdict-mars` *(here)* |
|---|---|---|---|
| model | Qwen3.8-27B NVFP4 | Bespoke-Nimble-9B | openJev-verdict-2.0, 151.4M |
| shape | autoregressive, logits at a prefill | autoregressive, one code token | **non-autoregressive encoder** |
| per decision | 1 prefill per question | 1 prefill per decision | **1 forward pass, always** |
| decision RTT | p50 427 ms | p50 132 ms | **p50 18 ms** |
| types used | choice, score | enum, boolean, ordered enum | **choice, score, noul** |
| outcome | colony 11→0, scorpion killed | colony 11→0 at hull 100, killed | colony 11→0, scorpion killed |

## Read this before comparing any number to the upstream README

**The benchmarked model cannot be downloaded.** The project contains two models and only one is
obtainable:

- `verdict2/model.py` — the marker-pointer network with the **dual-channel correctness head**. Every
  headline figure describes this: 77.10% top-1, 1.44% correctness ECE, 0.7664 AUROC, the
  selective-classification curves. Its checkpoint `artifacts/verdict2-base/model.pt` is a git-LFS
  pointer, and the object is not on GitHub's LFS server — the batch API answers
  `404 Object does not exist`. The advertised HF repo `heman10x/openJev-verdict-2.0` holds a config, a
  tokenizer and two PNGs. **No weights.**
- `core/engine_encoder.py` + `heman10x/rlcd-modernbert-151m` — a GLiClass fine-tune of
  `knowledgator/gliclass-modern-base-v2.0`, 151.4M, checksummed in `artifacts/ARTIFACTS.json`. This one
  downloads and runs, and **this is what the branch serves.**

So the second channel does not exist here, this branch never uses or claims it, and none of the
accuracy or calibration headlines can be attributed to what is running.

**The calibration scope is narrower than "calibrated" suggests.** `calibrator.json` is one scalar
temperature (1.4265) whose recorded scope is `restricted_5_candidate_selection`. Upstream's own engine
only reports `calibrated_for_scope` when a query has exactly five candidates and `unvalidated_scope`
otherwise. The ballot here is usually not five, so most answers are outside the fitted scope. The page
shows that status rather than hiding it.

## Run it

```bash
git clone --recurse-submodules git@github.com:mikesmullin/openjev.git -b verdict-mars
cd openjev

uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python transformers gliclass safetensors pydantic

bun run model:prepare     # ~580 MB, SHA-256 verified against artifacts/ARTIFACTS.json
bun run game              # fetch + patch vibe-arcade's mars.html

bun run model             # terminal 1  (bun run model:cpu also works)
bun run dev               # terminal 2: http://127.0.0.1:8734/
```

`bun run model:selftest` runs one decision and a latency sweep. The model loads in 2.3 s and needs
about 0.6 GB, so it shares a GPU with anything else — or runs on the CPU.

## The first model here whose own vocabulary is v1's

Upstream's primitives are `Choice`, `Score` and `Noul`. That is simple-jev v1 exactly, so nothing is
invented in translation:

```
v1 choice {criteria: {id: why}}  ->  Choice(options=[Option(id, description=why)])
v1 score  {criteria: [rubric]}   ->  Score(levels=[Level(id=str(i), description, value=i)])
v1 noul   {instructions}         ->  Noul(proposition=instructions)
```

**`noul` runs for the first time in this repo.** `simplejev-mars` had to abandon it: asked for a bare
digit `1`-`9`, a Qwen reaches for `0` and every permitted label sits 8+ nats down, so the softmax ran on
noise. That was a property of making a generative model emit a digit, not of the type. Here noul is a
two-outcome query scored like any other — though see below for what it actually answers.

**Abstention is a real outcome.** Every query gets `__insufficient_evidence__` appended by upstream's
formatter, and the id is *reserved* — constructing an `Option` with it raises. So this branch is the
first with no `other:hold` candidate on the ballot. Every earlier branch had to write "hold fire and
attack nothing right now" as an option, and `simplejev-mars` documented the cost: on Qwen3.5-9B it won
at 0.98 every tick and flew a whole game at `shots: 0`, because a safe, agreeable sentence is what a
choice question rewards. Here "none of these" is a property of the query type instead of a sentence
someone had to write persuasively, and the server reports it as `abstained` / `p_abstain` beside the
renormalised distribution.

## The state has to be prose, and that is not a style preference

Every other branch sends `state` as canonical JSON, because v1 renders it that way and treats it as
data. Doing that here **measurably breaks the model**. With only the state changing:

| | JSON state | prose state |
|---|---|---|
| posture, hull 100 / 0 damage | break_off 0.78 | press 0.91 |
| posture, hull 15 / 26 damage | break_off 0.785 | press 0.70 |
| threat, any hull | **abstains** (p_abstain 0.45–0.54) | gives a reading |
| boss phase: armoured head / tail | 0.40 / 0.53 | 0.006 / 0.041 |
| boss phase: the only vulnerable claw | **0.07** | **0.953** |

Under JSON it preferred the two parts whose descriptions say *"armoured and cannot be hurt"* over the
only one that could be damaged, and abstained overall. Under prose it picks the claw at 0.953. It is a
151M encoder fine-tuned on prose support, security and finance tickets; a wall of `snake_case` keys is
not what it reads. So `situation()` narrates the state into sentences, and `server/verdict_server.py`
passes a string state through verbatim.

### Irrelevant context flips answers

The more uncomfortable measurement. Holding the question and the decisive facts identical and only
appending *true but irrelevant* sentences (altitude, saucer count, mission restatement):

```
posture, press probability        hull 100        hull 46       hull 15
  2-sentence state                  0.91            0.72          0.70
  full 8-sentence state             0.35            0.28          0.27
```

The answer inverts. Upstream reports a 4.76% option-**order** flip rate, and symmetric permutation-KL
is one of the five headline breakthroughs — but that measures shuffling the options, not padding the
context. On this task, context length moves the decision further than the decisive fact does.

## What the model can and cannot answer here

Probed directly, with the prose state the agent actually sends:

| question | result | verdict |
|---|---|---|
| **target** — which thing to attack | tracks the candidate descriptions, prefers the colony over saucers | **works** |
| **boss phase** — only the claw is vulnerable | claw 0.953, armoured parts 0.006 / 0.041 | **works, and well** |
| **posture** — press or break off | `break_off` 0.70–0.71 at *every* hull from 100 to 15 | pinned |
| **threat** — 0–3 danger rubric | 2.54 at every hull; abstained on 26 of 162 live decisions | flat |
| **wake** — is the colony finished | p_true 0.778 at 11 buildings, 0.731 at 0 | **slightly inverted** |

The pattern is consistent: it reads the **candidate descriptions** well and barely conditions on the
**state**. The boss-phase question works because the discriminating fact ("armoured and cannot be hurt")
is written into the options themselves. Posture, threat and wake all require relating a number in the
state to a judgement, and it cannot.

### So the harness does more here, and that is the honest headline

Two gates in `web/app.js` exist purely because the model cannot answer the question:

- **`wake` is only asked once the colony is already flattened.** Asked every tick, p_true ~0.78 clears
  any threshold, and the first run woke the scorpion on tick one with 11 buildings standing — the
  failure that loses this game and the one the 2B made on `simplejev-mars`. The ordering constraint is
  now enforced in code and the model only confirms the final go.
- **The bounded break-off inherited from `nimble-mars` is load-bearing in a way it was not there.**
  `break_off` sits at 0.62–0.68 in game, below the 0.75 conviction threshold, so the veto mostly never
  fires. On `nimble-mars` that threshold discriminated; here it mostly just suppresses a constant.

`target` — including the whole boss-phase sequence — is genuinely the model's. The mission structure is
not.

## Results

One continuous run, agent driving from the menu, `wake` gated:

| goal | result |
|---|---|
| destroy the colony | **11 → 0 buildings**, before the scorpion was woken |
| wake the scorpion | on the model's confirmation, once the colony was gone |
| kill the scorpion | **dead** — claws → tail → head |
| survive | ship lost at the very end, hull 2, both objectives complete |

**162 decisions, 486 questions, mean RTT 19 ms, p50 18 ms, p95 22 ms**, 7 ms per question, 225 shots
fired. 12 ms of that is the model and ~7 ms is HTTP and proxy — at this speed the harness overhead is
a third of the budget, which has not been true on any previous branch.

Latency by question count, measured directly:

```
1 question 6.8 ms     2 -> 7.8 ms     3 -> 9.9 ms     4 -> 12.0 ms
```

One padded batch, `forward_call_count: 1`, always. `nimble-mars` had to rebuild upstream's CUDA scorer
to get four questions into a single pass; here it is what the architecture does. There is no cadence to
tune and nothing to cache.

### Honest placement

This is the fastest and the cheapest model in the repo by a wide margin — 151M parameters, 0.6 GB, 2.3 s
to load, 18 ms a decision, and it will run on a CPU. It flattened the colony and killed the scorpion.

But it reached that outcome with more of the mission encoded in the harness than any previous branch,
because three of its four questions do not respond to the game state. And the model that the upstream
benchmarks actually describe — the marker-pointer network with the correctness head, which is the
interesting idea in the project — could not be obtained at all. If those weights are published, the
branch is worth re-running: the second channel is exactly what the break-off veto on `nimble-mars` had
to approximate with a hand-tuned threshold.

## The solo-mode bug in the game

`web/game/mars.html` is fetched and patched by `scripts/fetch-game.sh`, pinned to vibe-arcade commit
`0cc97efd`. `fireTwinLaser` does `const owner = myId || 'me'`, but the three damage handlers guard with
`if (ownerId !== myId) return`. Solo, `myId` is `null` and `owner` is `'me'`, so **every hit a solo
player lands is silently discarded**. The game already has the correct idiom elsewhere —
`(owner === myId) || (!mpReady && owner === 'me')` — and the script applies it to `mpBuildingHit`,
`mpRockHit` and `mpBossHit`. The in-game `KILLS` counter still reads `0 saucers · 0 buildings` with the
colony flattened, because `awardScore()` is reached on a path solo play does not take; the page's own
`colony left` reads the live building list and is the one to trust.

## Notes

A stale answer is worse than no answer. `wake` is only on the ballot while the scorpion is dormant, so
once it is awake the answer stops arriving — and an earlier version of the response handler *held* the
last value instead of clearing it, which sent the act chain down the `wakeBoss()` branch on every
subsequent tick and silently skipped aiming. That was a whole game at `shots fired: 0`, and it was a
harness bug, not the model. Answers that are not re-asked are now cleared.

Browsers hold ES modules across reloads even under `no-store`. `app.js`, `mars-hook.js` and the game
iframe are all loaded with a `?v=` cache-bust.
