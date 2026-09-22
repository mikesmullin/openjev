# Laya Core ML Browser

**A local browser Tetris demo driven by Laya typed-decision inference on Apple Silicon.**

The browser renders the board and generates every legal landing. A local Swift/Core ML
service asks Laya *“Is this a clean placement?”* for each candidate, and the landing with
the highest `P(true)` is selected. Bun/Express serves the UI and proxies the inference API;
no cloud model or generated output tokens are involved.

```text
HTML/CSS/JavaScript Canvas
          │
          ▼
Bun + Express gateway :3000
          │
          ▼
Swift LayaServer / Core ML :8787
          │
          ▼
Laya multilingual L128 fp16 bucket + tokenizer.json
```

## Local MacBook Pro M1 performance

Measured locally on a **MacBookPro18,1 with Apple M1 Pro, 16 GB RAM**, macOS 26.6.2.
The service used the 322M-parameter multilingual Laya Core ML model, the L128 bucket,
and CPU + Neural Engine execution. Model loading was excluded.

| Warm local measurement | Result |
|---|---:|
| One-state HTTP API, P50 | **9.42 ms** |
| One-state HTTP API, P95 | **10.06 ms** |
| 20-state batch, total P50 | **145.80 ms** |
| 20-state batch, per candidate | **~7.29 ms** |
| Core ML service time in a 20-state batch | **~7.18 ms/candidate** |

The HTTP measurements used 100 single-state requests after 10 warmups and 20 batches of
20 states. The batch endpoint keeps browser/network overhead out of the per-candidate
model loop. An earlier 2,083-decision headless Tetris run measured a **7.26 ms median**
and **7.37 ms P95** for warm Core ML decisions.

These are measurements from one local development machine, not a cross-device benchmark.
Different model buckets, prompt lengths, OS activity, and power/thermal conditions will
change the results.

## Run the browser demo

Requirements:

- Apple Silicon Mac
- macOS 14+
- Xcode command-line tools with the license accepted
- Swift 6+
- Bun 1.3+
- The Laya Core ML model assets

The demo uses the existing local model cache when available:

```text
~/Library/Application Support/FluidUse/Models/laya-coreml/
```

It must contain `tokenizer.json` and:

```text
laya_multilingual_fp16_L128_options32.mlmodelc/
```

The fp16 L128 bucket plus tokenizer is approximately 650 MB. Model files are excluded
from Git. `LayaServer` can download the L128 assets automatically when `LAYA_MODEL_DIR`
is not set.

Install the Bun gateway dependency once:

```bash
bun install --cwd web
```

Start the Swift inference service and browser gateway together:

```bash
./scripts/run-browser.sh
```

Then open <http://127.0.0.1:3000>. The UI supports `laya`, `heuristic`, and `random`
policies. Increase the delay sliders to watch the orange candidate outline move through
the scored landings before the selected green landing is placed.

To select a different model directory or ports:

```bash
LAYA_MODEL_DIR="$HOME/Library/Application Support/FluidUse/Models/laya-coreml" \
LAYA_PORT=8787 PORT=3000 ./scripts/run-browser.sh
```

## API

The Swift service exposes a small localhost API:

```text
GET  /healthz
POST /v1/laya/tetris/score
```

Scoring is batched so the browser does not make one HTTP request per legal landing:

```bash
curl -X POST http://127.0.0.1:8787/v1/laya/tetris/score \
  -H 'content-type: application/json' \
  -d '{"states":["The T piece dropped at column 3 leaves one hole under it."]}'
```

A response contains the calibrated probabilities, token count, bucket and per-call
latency:

```json
{
  "model": "laya-multilingual",
  "bucket": 128,
  "results": [
    {
      "pTrue": 0.005990498,
      "pFalse": 0.9940095,
      "latencyMs": 7.2,
      "tokenCount": 46
    }
  ]
}
```

## Project layout

- `Sources/LayaEngine/` — exact tokenizer, prompt builder, Core ML bridge and model store
- `Sources/LayaServer/` — minimal localhost HTTP inference service
- `web/public/` — browser Tetris and Canvas UI
- `web/server.js` — Bun/Express static server and API proxy
- `scripts/run-browser.sh` — builds and starts both services
- `Tests/LayaEngineTests/` — prompt and typed-answer unit tests

This branch intentionally removes the unrelated form automation, FluidAudio dependency,
old Swift command-line demos, SwiftUI demo, media, benchmarks and unused desktop assets.
It retains only the Laya inference path needed by the browser application.

## Tests and build

```bash
swift test
swift build -c release --product LayaServer
bun install --cwd web
```

The release build and the browser gateway have been smoke-tested against the local model
on the M1 Pro machine described above.

## Model and license

The Core ML model is an independent conversion of
[`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya), published at
[`FluidInference/laya-coreml`](https://huggingface.co/FluidInference/laya-coreml). It
provides typed `choice`, `score`, and `noul` decision heads without autoregressive
text generation. The model and retained runtime code follow their Apache-2.0 licensing;
see [LICENSE](LICENSE).

Core ML inference requires macOS and Apple Silicon. The Bun/Express layer is a local
browser gateway, not a portable Linux inference implementation.
