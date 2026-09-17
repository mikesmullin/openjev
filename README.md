# openjev

[AlexWortega/openjev](https://huggingface.co/AlexWortega/openjev) — Qwen3.5-4B fine-tuned as a 3-class NLI
cross-encoder — reproduced locally, then pointed at real work. The model never generates text. It scores
statements and the argmax entailment is the answer.

| branch | task |
|---|---|
| `openjev-email` *(active)* | **Inbox triage** — four classification questions per email |
| [`openjev-mars`](../../tree/openjev-mars) | MARS RAID — target selection in a 3D shooter |
| [`openjev-tetris`](../../tree/openjev-tetris) | Tetris in the terminal |
| [`openjev-cook`](../../tree/openjev-cook) | Cook Fever |
| [`openjev-doom`](../../tree/openjev-doom) | ViZDoom — the original reproduction |

**Not a GitHub fork.** The original lives only on Hugging Face. Root commit `63ab36c` vendors his `code/`
verbatim at `8c9db06` — this branch then deletes the parts it does not use (his training and evaluation
scripts, and the game demos), so `git show 63ab36c` is the pristine upstream and everything after it here is
ours. The other branches keep his scripts. MIT, like his.

---

## What this replaces

A generative LLM pass over an inbox, of the kind `agl-agents/personal-email` runs. Instead of prompting a
model to emit JSON, each email is put to the cross-encoder as **four independent multiple-choice questions**:

| question | answers |
|---|---|
| Category | Brand deal · Cold pitch · Personal · Newsletter · Marketing · Receipt · Service alert · Notification · Other |
| Reply | Yes · No |
| Sender | Human · Auto |
| Urgency | Today · This week · Whenever |

Each answer is a sentence. All 16 sentences are scored against the email in **one batched forward pass**;
the argmax within each group is the answer and its share of the group's mass is the confidence.

**Tokens out is structurally zero** — there is nothing to parse, nothing to retry, and no way to get invalid
JSON back.

## Run it

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python "transformers>=5.0" accelerate huggingface_hub flash-linear-attention
huggingface-cli download AlexWortega/openjev --local-dir ./openjev_hf   # 8.5 GB, ~10 GB VRAM

bun install
bun run model                          # terminal 1: the only Python process
EMAIL_DB=/path/to/db bun run dev       # terminal 2: http://127.0.0.1:8735/
```

`EMAIL_DB` points at a directory holding `entities/` and/or `_archive/` of entity YAML files (each with an
`origin.raw` Gmail payload). **No corpus ships with this repo — it is someone's mail.** `data/` is gitignored.

## Performance, measured

500 emails, 16 statements each, one RTX 5090:

| | value |
|---|---|
| per email | **180 ms** (all four questions) |
| per question | ~45 ms |
| throughput | **5.2 emails/s** with 3 workers |
| tokens out | 0 |

### Why not cache the premise across questions?

It is the right instinct and it is mathematically sound: the model is a *causal* decoder and the template is
`Premise: …\nHypothesis: …`, so the premise's hidden states cannot depend on the hypothesis. Computing them
once and replaying them for all 16 statements would be exact, not an approximation.

**It does not work on this architecture.** Qwen3.5 is a hybrid — 24 of its 32 layers are gated-DeltaNet
linear attention, which keep a *recurrent* state rather than a KV cache. Transformers cannot replicate that
state across a batch:

```
AttributeError: 'LinearAttentionLayer' object has no attribute 'batch_repeat_interleave'
```

On a pure-attention model this optimisation would be a straightforward ~7x.

### Batching more emails does not help either

Already compute-bound, not launch-bound — cost is linear in total tokens:

```
1 email  175 ms   174.5 ms/email
2 emails 338 ms   169.0 ms/email
8 emails 1391 ms  173.8 ms/email
```

### So the only lever is fewer tokens

The premise is re-encoded once per statement, so its length is paid 16 times:

| body characters | tokens | ms/email |
|---|---|---|
| 0 | 1014 | 61 |
| 240 | 1846 | 110 |
| 900 | 4086 | 230 |

Capping the body at 240 characters took the app from 365 ms to 180 ms per email — **2x** — with no
measurable loss of accuracy, because subject, sender and the opening lines carry nearly all the signal.

## The wording matters more than anything else

Three rounds of this, each measured:

**1. Vague catch-alls swallow everything.** "This does not fit any usual category of email" is loosely true
of almost any email, so **Other took 63% of the inbox**. Replaced with something concrete — a bounce, a
delivery failure, a test message — and Other fell to **0%**, with a realistic spread across the rest.

**2. Negative and universal statements barely entail at all.** This is the sharpest finding here. Measured
P(entailment) across a marketing email, a receipt, a security alert and a personal note:

```
"Nothing is asked of the recipient at all…"       0.003  0.030  0.000  0.001
"Nobody is waiting for an answer…"                0.022  0.085  0.011  0.001
"There is a deadline, an expiry or a security…"   0.037  0.009  0.669  0.012   <- positive, and it fires
```

An NLI head asks whether a claim is *supported by the text*. A universal negative has nothing in the text
to support it, so it scores near zero and its option can never win. Rewriting both as positive assertions
fixed urgency instantly: Whenever went from 0% to a correct majority, and the security alert still lands on
Today at 0.669.

**3. Pick wording by measurement, not taste.** Three phrasings of the Reply question, scored against five
hand-labelled emails:

| phrasing | correct |
|---|---|
| "A named person is waiting… and will notice if no answer comes." | 3/5 |
| "The sender asks the recipient a direct question…" | 4/5 |
| **"The sender personally addresses the recipient and asks them to respond."** | **5/5** |

The two losers both called a security alert and a marketing blast "needs a reply".

## Honest placement

This is the model's native domain — judging whether a sentence follows from a piece of prose — and it shows:
no heuristic baseline competes, unlike the games on the other branches. What is *not* established here is
accuracy against ground truth. The distributions look right and the spot checks pass, but nobody has
hand-labelled 500 emails, so treat the numbers as plausibility, not precision.

Not TypeSafe's Jev, not RLCD, and none of Jev's calibration claims.
