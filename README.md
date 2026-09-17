# openjev

[AlexWortega/openjev](https://huggingface.co/AlexWortega/openjev) — Qwen3.5-4B fine-tuned as a 3-class NLI
cross-encoder — reproduced locally, then pointed at a cooking game. The model never generates text. It scores
statements about the game state and the argmax entailment becomes the move.

| branch | what it plays |
|---|---|
| `main` | **Cook Fever** (`cook2.html` from [mikesmullin/vibe-arcade](https://github.com/mikesmullin/vibe-arcade)), in the browser |
| [`openjev-doom`](../../tree/openjev-doom) | ViZDoom, the original reproduction — see that branch's README |

**This is not a GitHub fork.** The original lives only on Hugging Face; all 122 repos on
[github.com/AlexWortega](https://github.com/AlexWortega) were checked and none is an equivalent. Attribution
is in the history: the root commit `63ab36c` is his `code/` fetched verbatim at `8c9db06`, so
`git diff 63ab36c` is exactly our contribution. His work is MIT; so is this. His HF repo is a real git repo
if you want the true lineage:

```bash
git remote add upstream https://huggingface.co/AlexWortega/openjev   # upstream/main is 8c9db06
```

---

## Run it

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python "transformers>=5.0" accelerate huggingface_hub flash-linear-attention
huggingface-cli download AlexWortega/openjev --local-dir ./openjev_hf   # 8.5 GB, needs ~10 GB VRAM

bun install
bun run fetch-assets     # cook2.html + ~16 MB of art from vibe-arcade, and injects the agent bridge

bun run model            # terminal 1: the only Python process
bun run dev              # terminal 2: http://127.0.0.1:8733/
```

Then press **run agent**.

## Architecture

```
browser  web/index.html + web/app.js      m.js page; the agent loop lives here, next to the game
         web/game/cook2.html (iframe)     vibe-arcade's game, with one injected bridge line
         web/game/agent-hook.js           state -> premise, affordances -> hypotheses, argmax -> act
   |
   v  POST /api/decide
bun      server/index.js                  express: static files + proxy. No game logic.
   |
   v  POST /score
python   server/model_server.py           ~110 lines, stdlib HTTP. Holds weights. Knows nothing about cooking.
```

The agent loop runs in the browser because that is where the game is — reading state and applying an action
are direct calls, not round trips. Python is one file that takes a premise and a list of hypotheses and
returns P(entailment) for each.

### Why not llama.cpp

Checked in the source, not assumed. Sequence-classification heads in llama.cpp exist **only for BERT-family
encoders** — `convert_hf_to_gguf.py` registers `ForSequenceClassification` for Bert, DistilBert, Roberta,
XLMRoberta, NeoBERT and ModernBert, and no decoder-family model has one. `Qwen3_5` is not a known
architecture there at all; `llama-arch.cpp` knows `QWEN3NEXT`, which is a different arch and causal-LM only.
Hosting this checkpoint would mean implementing the Qwen3.5 hybrid attention stack *and* inventing a 3-label
pooled head for a decoder. Hence the one Python file.

### Driving the game

cook2.html is click-and-drag, but we never synthesise pointer events at guessed coordinates. It already has
the right seams — `G`, `interactives`, `stations`, `selected`, `select()`, `tryDrop()` — so
`scripts/fetch-assets.mjs` injects one line at the end of its module handing that scope to our hook
(`selected` as a getter, since it is a `let`). Actions are then enumerated from the game's own `accepts()`
and `onTap()` and executed as direct calls.

### The hypotheses

Straight from the Doom result on the other branch: every candidate action is offered to the model as a
**statement about the world that would justify it**, never as the name of the action. Naming the action
scores at chance; verifying a statement works.

```
"There is payment sitting on the counter, and the seat it is on stays blocked
 until it is picked up."                            -> collect the payment    0.98
"A customer has ordered a soda and none has been poured yet."
                                                    -> start pouring a soda   0.89
"The raw patty in your hands needs to be cooked on the grill."
                                                    -> put it on the grill    0.58
```

## Livelocks, and what they have in common

Every bug worth recording here was the same bug: **an action that leaves the state unchanged wins the argmax
again on the next tick, forever.** The model is not wrong in any of these; the action set is.

| symptom | cause |
|---|---|
| three identical `"The grill should be used now."` | `Station.onTap()` is a no-op on the base class, so every station offered a dead hypothesis — one per grill pan, each costing a forward pass |
| tapped the soda machine 8× with empty hands | the second tap returns `false`, and the premise never mentioned the machine, so the model could not tell the first one worked |
| `pick up the full cup` ⇄ `put it back down` | a finished dish nobody ordered — the only move after picking it up is to put it down |
| `pick up the empty plate` ⇄ `put it back down` | plates are assembled in place; lifting an empty one is never useful |
| `pick up the cooking patty` ⇄ `put it back down` | taking food off the heat undoes progress |
| `take a patty` ⇄ `put it back down` | no free pan to put it in |

The fixes are all the same shape: do not offer an action that cannot change anything, and make sure the
premise mentions every machine that works on its own, so "nothing happened" is distinguishable from
"something is underway". There is also always a `wait` action, because food cooks and customers arrive on
their own, so waiting is a real strategy and the list is never empty.

**Uncollected payment stalls the level.** `freeSlot()` skips any seat that still has a coin pile on it —
the game's own comment reads *"sitting piles throttle the wave"* — so money left on the counter stops the
next customer from ever arriving. That one is not a livelock, just a rule worth knowing.

## What it does and doesn't do

It serves customers, collects tips, pours sodas and cooks patties, at **~80–110 ms and 10–17 hypotheses per
decision, ~2.5 decisions/s**. It is not good at the game: it has no plan beyond the current tick, and on a
quiet board its confidence collapses to 0.05–0.10 across every option, where the argmax is close to
arbitrary. Nothing here is trained — it is the same frozen NLI checkpoint, zero-shot, being asked to check
sentences about a diner.

Not TypeSafe's Jev, not RLCD, and none of Jev's calibration claims. openjev cannot emit free text — only
3-class scores over supplied options — but it can still pick the wrong option with a confident score.

## Dev notes

The browser holds ES modules across reloads even under `no-store`, and a stale module is invisible — it just
keeps the old behaviour while you debug code that is never running. Both `web/app.js` and the game's
`agent-hook.js` are therefore imported with a `?v=` cache-bust. `M.mount()` returns the reactive root
instance but does **not** call `init()` on it; only `x-data` / `x-component` scopes get that automatically.
