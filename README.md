# openjev

[AlexWortega/openjev](https://huggingface.co/AlexWortega/openjev) — Qwen3.5-4B fine-tuned as a 3-class NLI
cross-encoder — reproduced locally, then pointed at games. The model never generates text. It scores
statements about the game state and the argmax entailment becomes the move.

| branch | plays | interface |
|---|---|---|
| `main` | **Tetris** ([mikesmullin/tetris](https://github.com/mikesmullin/tetris)) | terminal, in tmux |
| [`openjev-cook`](../../tree/openjev-cook) | Cook Fever (vibe-arcade `cook2.html`) | browser, m.js + Bun/express |
| [`openjev-doom`](../../tree/openjev-doom) | ViZDoom | native window / browser |

**This is not a GitHub fork.** The original lives only on Hugging Face; all 122 repos on
[github.com/AlexWortega](https://github.com/AlexWortega) were checked and none is an equivalent. Attribution
is in the history: the root commit `63ab36c` is his `code/` fetched verbatim at `8c9db06`, so
`git diff 63ab36c` is exactly our contribution. His work is MIT; so is this.

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

git clone git@github.com:mikesmullin/tetris.git && (cd tetris && make)   # or set TETRIS_BIN

bun run model              # terminal 1: the only Python process
bun run demo               # terminal 2: splits your current tmux window
```

`scripts/tmux-demo.sh` **reuses an existing tmux session** rather than creating one — board on the left,
what the model was asked and answered on the right. `TMUX_SESSION` picks a specific one. Without tmux:
`bun run tetris -- --reset --pieces 40`.

## Architecture

```
bun   agent/tetris.js        enumerate placements, describe outcomes, score, press keys
  |
  |  fork state -> press -> dump          the C binary is its own simulator
  v
      $TETRIS_BIN            mikesmullin/tetris: turn-based, JSON `dump`, state in $TETRIS_STATE_FILE
  |
  v  POST /score
python server/model_server.py  ~110 lines, stdlib HTTP. Holds weights. Knows nothing about Tetris.
```

### The state file is the whole trick

`tetris` persists to `$TETRIS_STATE_FILE` and has an undocumented `dump` command that emits the board,
current piece and next piece as JSON. So every candidate placement is evaluated by **copying the live state
to a scratch file, pressing the keys against the copy, and dumping the result.** The simulator is the real
game: nothing here reimplements gravity, wall kicks, locking or line clears, and there is no chance of the
agent's model of the rules drifting from the game's.

Illegal moves are clamped rather than rejected, so many key sequences land the piece in the same place —
those collapse to the shortest sequence.

### Why not llama.cpp

Checked in the source, not assumed. Sequence-classification heads in llama.cpp exist **only for BERT-family
encoders** (`convert_hf_to_gguf.py` registers `ForSequenceClassification` for Bert, DistilBert, Roberta,
XLMRoberta, NeoBERT, ModernBert); no decoder-family model has one. `Qwen3_5` is not a known architecture
there at all — `llama-arch.cpp` knows `QWEN3NEXT`, a different arch, causal-LM only. Hosting this checkpoint
would mean implementing the Qwen3.5 hybrid attention stack *and* inventing a 3-label pooled head for a
decoder. Hence the one Python file.

### The hypotheses

Doom and Cook Fever used the model as a *truth test*: one statement about the world was true, and the action
bound to it was the move. Tetris is different — every candidate outcome is computed by the game, so every
description is factually true. Here the model is used as a **reranker** instead, which is his own `rerank`
protocol: the premise states the objective, the options compete on P(entailment).

```
premise      ... Column heights left to right are 1:0 2:0 3:2 ...  A good move clears lines,
             buries no empty cells, keeps the stack low, and leaves the surface flat.
             Burying an empty cell is bad because it cannot be filled until every row above it is cleared.

hypothesis   This move completes and clears 1 line, buries no new empty cells, leaves the
             tallest column 2 rows high, and leaves the surface with a roughness of 4,
             where 0 is perfectly flat.
```

The roughness number matters more than it looks. Without it, most quiet placements produced the *same
sentence*, every probability tied at 0.018, and the argmax collapsed to whichever candidate came first —
the agent just dropped everything down the left wall. Descriptions that do not discriminate cannot be
reranked. Identical sentences are now deduped before scoring, since a duplicate costs a forward pass and
cannot be told apart anyway.

## How well it plays

Four runs: **5, 7, 8 and 12 lines** (500–1500 points) over 52–69 pieces, each ending in game over. 4–19
outcomes are scored per piece in 30–800 ms.

It started much worse — 0 to 4 lines — and three changes got it there. They are worth recording because two
of them are about *language*, not search:

**Tell it how close a clear is.** The original description had no signal for line-clear progress at all, so
it optimised flat-and-low forever and never aimed at finishing a row. Outcomes now report the nearest
unfinished row's remaining cells.

**Trim the ballot to the Pareto front.** Drop any placement that another placement beats on *every* axis at
once (lines, buried cells, height, roughness). This removes strictly-worse options without taking a position
on the trade-offs between them — the model still chooses.

> A first attempt simply dropped every hole-creating move whenever a clean one existed, and it backfired
> badly. Flat-topping the stack never buries anything, while filling a gap beside it usually does, so the
> filter left only tower-building moves. The stack went 2 → 5 → 7 rows in three pieces, and by piece 4 there
> was no clean move left and it took seven holes at once. A filter that looks like it encodes the objective
> can quietly encode the opposite.

**Say it in words, not numbers.** This was the single biggest win — 3 lines to 8. The model is a *language*
model, and the difference between "roughness 9" and "roughness 13" is a much weaker signal than "the surface
is left jagged and full of gaps". Descriptions now lead with the decisive fact in plain language:

```
This move traps 1 empty cell under the blocks, ruining those rows.
The stack is getting high, and the surface is left a little uneven.

This move is clean and traps no empty cells. The stack stays low,
and the surface is left flat and easy to build on. A row is left
needing only 1 more cell to clear.
```

It is still a one-ply greedy reranker with no lookahead past the current piece, and a hand-written heuristic
would beat it comfortably.

Not TypeSafe's Jev, not RLCD, and none of Jev's calibration claims. openjev cannot emit free text — only
3-class scores over supplied options — but it can still pick the wrong option with a confident score.

## Notes

`pane-base-index` is commonly `1`, so the demo script resolves the first pane index instead of assuming
`0` — hardcoding it fails with `can't find pane: 0`.
