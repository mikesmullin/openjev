# gliner-email

[GLiNER 2.5](https://github.com/fastino-ai/GLiNER2) triaging a real inbox — 500 emails, **6.6/s, on the
CPU**, with the GPU never touched.

The [`openjev-email`](../../tree/openjev-email) branch did this task with a Qwen3.5-4B NLI
cross-encoder: each email became 16 hypothesis sentences and the argmax entailment was the answer,
~180 ms per email on an RTX 5090. This is the same task, the same page, and the same 18-folder
taxonomy, with a 194M DeBERTa-v3 classification encoder instead — and it is faster on a CPU than the
4B model was on a GPU.

| | `openjev-email` | `gliner-email` *(here)* |
|---|---|---|
| model | Qwen3.5-4B NLI cross-encoder | GLiNER 2.5, 194M DeBERTa-v3 |
| device | RTX 5090 | **CPU**, 12 threads |
| per email | 180 ms | **109 ms** |
| throughput | 5.2 emails/s (3 workers) | **6.6 emails/s** |
| shape | premise re-encoded once **per hypothesis** | text encoded **once**, scored against a label set |
| tokens out | 0 | 0 |

This branch is an orphan — it shares no history with the others.

## Results

500 emails, 21 operations (delete / skip / archive + 18 Gmail folders) plus a two-class spam question,
all in one forward pass per email:

| | |
|---|---|
| throughput | **6.6 emails/s**, 0 errors |
| per email | 109 ms avg |
| avg confidence | 60% |
| tokens out | 0 — nothing to parse, no invalid JSON possible |

| outcome | count | share |
|---|---|---|
| archive | 154 | 31% |
| delete | 137 | 27% |
| move to Newsletters | 107 | 21% |
| move to Expenses | 52 | 10% |
| move to (4th folder) | 13 | 3% |
| move to (5th folder) | 8 | 2% |
| skip | 7 | 1% |
| move to (6th folder) | 7 | 1% |
| 7 more folders | 15 | 3% |

**40% filed to a folder, 27% deletable, 1% needing a human.** Folder names beyond the generic ones are
withheld here: the taxonomy is personal and lives only in the gitignored `config.yaml`. Spam: 62 likely, 23 borderline, 415 clean.

Spot-check of the first ten, redacted — this is a real inbox, so senders, subjects and amounts are
replaced with the shape of each message. The classification and spam score are the actual outputs:

```
local political newsletter                   -> delete               spam   1%
music-software marketing blast               -> move to Newsletters  spam  99%
bank card charge alert, small amount         -> move to Expenses     spam   2%
calendar notification for a utility bill     -> archive              spam   0%
bank card charge alert, larger amount        -> move to Expenses     spam   2%
HOA community announcement                   -> move to Newsletters  spam   0%
car-forum promotional mail                   -> delete               spam 100%
retailer discount promotion                  -> delete               spam   1%
online-retailer content promotion            -> delete               spam   1%
```

Card charges land in `Expenses`, promotions in `delete`, community and subscription mail in
`Newsletters`, and an automated calendar ping in `archive`. Those are the calls a human would make.

## Labels are class names, not assertions

openjev scored checkable assertions — *"This is a receipt, invoice, or order confirmation for something
the recipient bought, with an amount stated."* Feeding those same sentences to GLiNER is worse than
useless. Measured on eight real emails:

| label form | speed | behaviour |
|---|---|---|
| descriptions (openjev's hypotheses) | 236 ms/email | collapses to `skip` on 7 of 8 |
| bare folder names (`Expenses`) | 86 ms/email | discriminates, but confuses neighbours |
| **short phrases** (`purchase receipt`) | 89 ms/email | **discriminates, fewer errors** |

Bare names are nearly as good and 2.6x faster than sentences, but they confuse categories that sit next
to each other: a card charge filed as `Income` until `Expenses` became *"purchase receipt"*. So
each folder and operation carries a short `label` in `config.yaml` — two or three words naming the
class — alongside the `description`, which stays for humans and for the openjev branch.

This is the same finding as [`gliner-mars`](../../tree/gliner-mars), where sentence-shaped labels pinned
`threat` to *critical* on every tick. A generative model wants an option that argues for itself; a
classification encoder wants a label that names a class.

## Spam: the one place GLiNER is strictly better

openjev could not ask about intent. *"This is a scam trying to trick the reader"* scored **0.060** on a
blatant prize scam, and no rephrasing helped — an NLI head judges whether a claim is supported by the
text, and a scam is precisely a text that conceals its intent. That branch had to score three
*observable* patterns (windfall, urgency, credentials) and take the strongest.

A classification encoder has no such problem. Spam is a canonical text class, so it can simply be asked,
as two classes:

```js
export let SPAM_SIGNALS = ['spam', 'legitimate email'];
```

Measured across this inbox: a marketing blast 0.99–1.00, a bank charge alert 0.02, a community
newsletter 0.00. Two classes also means the confidence **is** the probability, rather than the maximum
of a bank of hand-written signals.

## The date line was eating the inbox

The single largest quality bug on this branch, and the most familiar one.

`premiseFor()` used to open with `An email received on Thu, 19 Jun 2026 18:28:00.` It looks like
harmless metadata. It is on all 500 emails, it reads as routine correspondence, and it dragged the
whole inbox into the catch-all. Isolated on four card-charge alerts from the same bank:

```
From/Subject/Body only        -> Expenses  (0.45, 0.35, 0.34, 0.30)
with the date line prepended  -> archive   (0.39, 0.38, 0.36, 0.47)
```

Across the full run:

| | with the date line | without |
|---|---|---|
| archive | **61%** | **31%** |
| delete | 15% | 27% |
| move to Newsletters | 14% | 21% |
| move to Expenses | 2% | **10%** |
| filed to a folder | 24% | **40%** |
| throughput | 6.5/s | 6.6/s |

The openjev branch warned that "a catch-all category sentence once took 63% of the inbox", and at 61%
this looked like exactly that — a bad `archive` label. It was not. The label was fine; the premise had
a constant in it. **Text that appears on every input cannot discriminate between inputs, but it can
still move every answer in the same direction**, which makes it look like a category problem rather
than an input problem. That is now four separate times in this repo, across four different models.

The date is still shown in the table. It is just not shown to the model, where it was never signal —
no folder in the taxonomy is about *when* something arrived.

## Run it

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python "gliner2[local]" protobuf sentencepiece
bun install

cp config.yaml.example config.yaml     # then edit: your folders, each with a short `label`
bun run model                          # terminal 1: the classifier, CPU, port 8750
EMAIL_DB=/path/to/db bun run dev       # terminal 2: http://127.0.0.1:8735/
```

`EMAIL_DB` points at a directory holding `entities/` and/or `_archive/` of entity YAML files, each with
an `origin.raw` Gmail payload. **No corpus ships with this repo — it is someone's mail.** `config.yaml`
is gitignored and holds the real folder names; `config.yaml.example` has generic placeholders.

`protobuf` and `sentencepiece` are required: DeBERTa-v3's tokenizer fails to load without them.
`bun run model` runs under `systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0`, so a
runaway load kills its own process rather than the desktop.

## Architecture

```
browser  web/index.html + web/app.js    m.js page; the worker pool and all the counters live here
         web/questions.js               the label sets and the premise. The part that matters.
   |
   v  POST /api/classify
bun      server/static.js               static files, /api/emails, /api/config, proxy
         server/emails.js               entity YAML -> {subject, from, body}
   |
   v  POST /classify
python   server/gliner_server.py        ~140 lines. fastino/gliner2.5-base-v1 on the CPU.
```

## Honest placement

`classify_text` returns the winning label and its confidence, not a distribution — `format_results=False`,
`threshold=0.0` and multi-label mode all still return only the argmax. The page wants a `probs` array,
so the remaining mass is spread uniformly over the losers: **`confidence` is real, the per-label spread
is not**, and nothing should read meaning into a loser's value. openjev, which scored every hypothesis
independently, gave a genuine ranking over all 21 operations. Spam is the exception here, because two
classes means the confidence is the probability.

60% average confidence is honest for a 21-way choice and should not be read as 60% accuracy — it is the
winner's share, and the folders genuinely overlap (`Expenses` vs `Statements`, `Opportunities` vs
`Job Applications`). What this branch demonstrates is that a 194M encoder on a CPU does this job faster
than a 4B NLI model on a 5090, and that almost all of the quality came from what the labels and the
premise say, not from the model.
