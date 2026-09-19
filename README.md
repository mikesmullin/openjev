# laya-email

[Laya](https://huggingface.co/convaiinnovations/laya) triaging a real inbox — **fastest of the three,
and the worst at the job.** Recorded because the failure is specific, measured, and useful.

Laya is a 421M ModernBERT-large "System 1" decision model that answers `choice` / `score` / `noul`
without generating text. On [`laya-mars`](../../tree/laya-mars) it is the best model this repo has run:
4.7 ms per question on the GPU and the only calibrated danger reading anyone produced. Pointed at 21-way inbox
routing it collapses, and it does so *confidently*.

| | [`openjev-email`](../../tree/openjev-email) | [`gliner-email`](../../tree/gliner-email) | `laya-email` *(here)* |
|---|---|---|---|
| model | Qwen3.5-4B NLI | GLiNER 2.5, 194M | Laya, 421M |
| device | RTX 5090 | **CPU** | **GPU** (see below) |
| throughput | 5.2 emails/s | 18.6 emails/s | 28.4 emails/s on GPU, **3.5 on CPU** |
| per email | 180 ms | 109 ms | 55 ms on GPU |
| filed to a folder | — | **40%** | 20% |
| fell into `archive` | — | **31%** | 67% |
| flagged likely spam | — | **62** (12%) | 216 (43%) |
| avg confidence | — | 60% | 35% |

It is 1.5x faster than GLiNER *on a GPU* and files half as much mail correctly. On equal footing --
both on the CPU -- it is **5x slower** as well. **Use GLiNER for this task.**

### Correction: an earlier version of this branch claimed CPU

The first commit here reported "27.4 emails/s on the CPU". That was wrong, and the error is worth
recording. `laya.load()` places the model on CUDA whenever a GPU is visible; the code set
`torch.set_num_threads()` and assumed that meant CPU. Eight forked workers therefore each took ~1.7 GB
of weights plus a ~1.5 GB CUDA context -- **25 GB of VRAM**, which filled the card while the README
said "the GPU is never touched".

Measured properly, `--device` now explicit:

| device | workers | throughput | VRAM |
|---|---|---|---|
| cuda | 1 | **28.4 emails/s** | ~3.1 GB |
| cuda | 8 | 27.4 emails/s | ~25 GB |
| cpu | 8 | 3.5 emails/s | 0 |

One GPU worker does everything eight were doing -- they were contending for the same device, so the
extra seven bought nothing and cost 22 GB. The default is now `--device cuda --workers 1`, with
`bun run model:cpu` for the CPU configuration. The process-not-threads result from `gliner-email` is
a CPU result and does not transfer to a single GPU.

## The six-option cliff

The headline finding, and the reason the flat design had to be abandoned. Four bank card-charge
alerts, correct answer `Expenses`, offered an increasing number of folders:

| options | correct | mean confidence |
|---|---|---|
| 2 | **4/4** | 0.17 |
| 3 | **4/4** | 0.14 |
| 4 | **4/4** | 0.27 |
| 6 | **4/4** | 0.63 |
| 9 | 0/4 | 0.82 |
| 12 | 0/4 | **1.00** |
| 18 | 0/4 | 0.96 |

Perfect to six, gone by nine. **Confidence moves the opposite way to accuracy** — lowest where the
model is right, 1.00 where it is wrong. So confidence cannot be used to detect this failure. The flat
21-option version answered `eCommerce` and `Stock` for card charges at 1.00, which reads as certainty
and is noise.

This is worth stating plainly because every other branch here has been able to trust confidence at
least directionally. Here it is actively misleading.

## Staging helps the structure and not the answer

`openjev-email` argued against two-stage routing, and was right *for that model*: a two-stage design
needs an intermediate option meaning "belongs in SOME folder", which is vague, and vague options beat
specific ones against an NLI head. Neither half of that argument applies to Laya — the stage-1 options
below each name a concrete kind of mail, and the cost objection is gone too, because openjev
re-encoded the premise once per hypothesis while Laya encodes the text once per call at ~15 ms. Three
cheap calls beat one wrong one.

So the tree keeps every node inside the cliff:

```
stage 1 (6)    delete | skip | archive | money | work | life
  money (5)    Expenses, Statements, Income, Stock, Taxes
  work  (4)    Opportunities, Job Applications, Job Interviews, employer
    employer(4)  Job/Ancestry, Job/Blizzard, Job/CrowdStrike, Job/HSA
  life  (6)    Travel, Newsletters, Kids, Myself, eCommerce, Real Estate Investment
```

It routes card charges into `money` correctly — and then picks `Stock` out of the five, because
*"You made a $5.36 transaction with AMAZON PRIME"* matches *"brokerage or crypto trade"*. Three label
wordings were tried on that group alone:

```
purchase receipt / bank statement / paycheck / brokerage or crypto trade / tax filing   0/4 Expenses
a card charge for something bought / a monthly account statement / ...                  1/4 Expenses
money spent / account statement / money received / investments / taxes                  0/4 Expenses
```

Best was 1 of 4, with confidence honestly low throughout (0.12–0.43). So the cliff is real but it is
not the whole story: Laya also cannot separate near-synonym financial folders regardless of how few it
is shown. That is a capability limit, not a wording bug, and tuning was stopped there. `Stock` ends up
with 41 of 500 emails in the full run.

## Spam is reported but should not be trusted

Laya's `noul` is well calibrated on the MARS questions. Here it flags **43% of the inbox** as likely
spam against GLiNER's 12%. On routine bank charge alerts it reads 0.80–0.95 across three wordings:

```
"Is this email spam or a promotional blast?"                    chase 0.88, 0.85
"Is this unsolicited spam from a sender they do not know?"      chase 0.95, 0.85
"Is this a marketing email trying to sell something?"           chase 0.80, 0.57
```

GLiNER reads 0.02 on the same mail. Obvious marketing is 1.00 and community mail 0.06–0.13, so the
signal is not absent — it false-positives hard on transactional mail, which is most of an inbox.

## What this says about the model

The two branches together are the useful result. The same model, on the same machine, in the same week:

- **[`laya-mars`](../../tree/laya-mars)** — small, semantically distinct option sets; ordinal severity;
  a yes/no about an observable state. Best-in-repo, 4.7 ms per question on the GPU, calibrated.
- **`laya-email`** — many-way topical routing over a personal taxonomy with near-synonym categories.
  Worst-in-repo, and confidently so.

Laya is a decision model, not a classifier. It is very good at *judging a described situation* over a
handful of clearly different choices, and poor at *looking up which of eighteen bins a document belongs
in*. GLiNER, an information-extraction model built for label matching, is the opposite. Neither
ordering is about size — Laya is more than twice GLiNER's parameter count.

## Run it

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python laya
bun install

cp config.yaml.example config.yaml     # your folders, each with a short `label`
bun run model                          # 1 GPU worker, port 8750  (bun run model:cpu for CPU)
EMAIL_DB=/path/to/db bun run dev       # http://127.0.0.1:8735/
```

`EMAIL_DB` points at a directory of entity YAML files with an `origin.raw` Gmail payload. **No corpus
ships with this repo — it is someone's mail.** `config.yaml` is gitignored.

`bun run model` runs a single GPU worker (~3.1 GB VRAM) under
`systemd-run --user --scope -p MemoryMax=24G -p MemorySwapMax=0`. `bun run model:cpu` forks 8 CPU
workers of 4 threads sharing the port via `SO_REUSEPORT` — that scaling is a CPU result measured on
[`gliner-email`](../../tree/gliner-email) and does not help on a single GPU, where one worker already
saturates the device.

## Architecture

```
browser  web/index.html + web/app.js      the worker pool and the counters
         web/questions.js                 label sets and the premise
   |
   v  POST /api/classify
bun      server/static.js + emails.js     static files, /api/emails, /api/config, proxy
   |
   v  POST /classify
python   server/laya_email_server.py      the staged tree. 1 GPU worker by default.
```

The wire contract is `openjev-email`'s, unchanged, so the page is untouched: the staged answer is
reported as an index into the flat `[operations..., folders...]` list the page builds from the same
`config.yaml`.
