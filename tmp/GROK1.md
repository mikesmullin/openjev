> User: how could i reproduce this result locally https://x.com/justALEXWORTEGA/status/2100341039986798930

The tweet is an open clone of TypeSafe’s Jev **interface**, not Jev itself. Alex fine-tuned Qwen3.5-4B as a 3-class NLI cross-encoder, then put a small MLP on the frozen last-token latents. The clip is that model playing ViZDoom *Defend the Center* by scoring action hypotheses and taking argmax. Weights and training code are already public.

## What you are reproducing

**Jev (closed):** TypeSafe’s System One model. No token generation. Input is state + typed questions; output is decisions with calibrated probabilities. Trained with RLCD. API-only.

**Openjev (what the tweet ships):** same *shape* of API — state in, option scores out — built from a vanilla NLI fine-tune. Alex later said he thinks Jev is just “an LLM + prob head trained on cross-encoder tasks.” That is the recipe below. It is **not** RLCD and does not inherit Jev’s calibration claims.

Two pieces in [AlexWortega/openjev](https://huggingface.co/AlexWortega/openjev):

1. **NLI cross-encoder** (`qwen3.5-4b-nli/`): Qwen3.5-4B as `Qwen3_5ForSequenceClassification`, last-token pooling, linear `score` head, 3 labels in dleemiller order (`contradiction`, `entailment`, `neutral`). Template: `Premise: {premise}\nHypothesis: {hypothesis}`. CE loss on AllNLI (SNLI + MNLI).
2. **Latent MLP** (the “MLP on top of Qwen 4B”): frozen pooled hidden state d=2560d=2560d=2560 → Linear(512) → GELU → Dropout(0.1) → Linear(1). Soft BCE with ε=0.1\varepsilon=0.1ε=0.1. Per-question argmax over options.

The video HUD (“zero-shot NLI”, ~90–180 ms/decision, 3 options per step) is the first piece used as a policy: premise = game state, hypotheses = `{turn left, turn right, attack}`, pick max P(entailment)P(\text{entailment})P(entailment).

---

## Path A — run the published checkpoint (fastest)

Needs ~8–12 GB VRAM in bf16 (RTX 3060 12 GB, 3090, or 16 GB+ Apple Silicon). CPU works, but not at the video’s latency.

```bash
# Qwen3.5 needs a current transformers (5.x)
pip install "torch" "transformers>=5.0" accelerate datasets huggingface_hub

huggingface-cli download AlexWortega/openjev --local-dir ./openjev
# modeling_openjev.py sits at the repo root
```

```python
from modeling_openjev import OpenJevCrossEncoder

jev = OpenJevCrossEncoder("AlexWortega/openjev", subfolder="qwen3.5-4b-nli")

# 3-way NLI probabilities [contradiction, entailment, neutral]
jev.predict([
    ("The bird is 0.05 below the centre of the gap.",
     "The bird is below the centre of the gap.")
])

# Jev-style multiple choice: max P(entailment)
jev.rerank(
    "Which gas do plants absorb during photosynthesis?",
    ["oxygen", "carbon dioxide", "nitrogen"],
)
```

Plain Transformers:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("AlexWortega/openjev", subfolder="qwen3.5-4b-nli")
model = AutoModelForSequenceClassification.from_pretrained(
    "AlexWortega/openjev", subfolder="qwen3.5-4b-nli"
)
text = model.config.nli_template.format(premise="...", hypothesis="...")
```

That is already the “it works like Jev” result: no decoding, just option scores.

---

## Path B — replay the Doom clip

The published script is the **text-state** path (ViZDoom labels buffer → English description → NLI). The tweet overlay also mentions `<image>` and monster xxx-position; that pixel path (`doom_vision.py`) is referenced in `run.sh` but is **not** in the current HF tree.

```bash
cd openjev
pip install vizdoom imageio pillow matplotlib
# Doom IWAD is bundled with vizdoom's defend_the_center scenario

python code/doom.py \
  --ckpt qwen3.5-4b-nli \
  --episodes 5 \
  --out results/doom_4b.json \
  --video-nli results/doom_nli.mp4
```

What that script does each 4 tics (114 ms of game time):

1. Read visible enemies from the labels buffer.
2. Build a premise like “Doom, Defend the Center… a zombie soldier 0.12 to the left of the crosshair (far)…”
3. Score hypotheses `"The correct action is: turn left|turn right|attack"`.
4. Take argmax entailment. On 4B this is ~60–180 ms, so it stays real-time.

`--video` instead of `--video-nli` trains the extra MLP on noisy-oracle rollouts and records that policy. Same HUD style as the tweet.

There is also `code/flappy.py` if you want the other in-repo game demo.

---

## Path C — retrain from scratch

All scripts live in `code/`.

### 1. Fine-tune the NLI backbone

```bash
python code/train.py \
  --model Qwen/Qwen3.5-4B \
  --out ckpt/qwen3.5-4b-nli \
  --n-train 200000 \
  --n-val 2000 \
  --max-len 256 \
  --bs 8 --grad-accum 4 \
  --lr 2e-5 --epochs 1 --grad-ckpt
```

Defaults that matter:

| Knob | Value |
|---|---|
| Data | `stanfordnlp/snli` + `nyu-mll/multi_nli` (200k train / 2k MNLI-matched val) |
| Label remap | SNLI/MNLI native E=0,N=1,C=2E=0,N=1,C=2E=0,N=1,C=2 → dleemiller C=0,E=1,N=2C=0,E=1,N=2C=0,E=1,N=2 via `{0:1, 1:2, 2:0}` |
| Template | `Premise: {p}\nHypothesis: {h}` |
| Optim | AdamW, cosine, 3% warmup, wd 0.01, bf16 |
| Vision tower | frozen (Qwen3.5 is multimodal; text NLI never uses it) |

VRAM:

- Full FT of 4B: roughly 24–41 GB.
- 16–24 GB card: add `--lora --lora-r 16 --grad-ckpt`.
- Literal “MLP/linear head on frozen Qwen”: `--head-only` (trains only `score`).

Smaller smoke test: `--model Qwen/Qwen3.5-0.8B` (that is `train.py`’s default).

### 2. Train the latent MLP (optional)

```bash
python code/latent_mlp.py extract --ckpt ckpt/qwen3.5-4b-nli --out data/latents_4b
python code/latent_mlp.py train   --latents data/latents_4b --out results/latent_mlp_4b.json
```

Extract dumps last-token hidden states + NLI logits for GPQA / MMLU / ARC / WinoGrande / chess (and optional HellaSwag, GSM8K-MC). Train fits `d → 512 → 1` with soft BCE, grouped holdout by question id, early stop on per-question argmax accuracy. Hyperparameters: lr `1e-3`, wd `1e-2`, bs 512, 60 epochs, patience 8, ε=0.1\varepsilon=0.1ε=0.1.

Or in Python against the wrapper:

```python
from modeling_openjev import OpenJevCrossEncoder, LatentMLPHead

ce = OpenJevCrossEncoder("ckpt/qwen3.5-4b-nli")
X = ce.latents(pairs)                 # (n_pairs, d)
head = LatentMLPHead(X.shape[1]).fit(X, gold, qid)  # gold in {0,1}
scores = head.predict(X)              # argmax within each qid
```

### 3. Eval the same way he did

```bash
python code/eval.py --models ckpt/qwen3.5-4b-nli --out results/qwen4b.json
```

Tasks in that harness: MNLI, MMLU, GPQA-Diamond, ARC-Easy/Challenge, WinoGrande, HellaSwag, GSM8K-MC, synthetic chess. The follow-up radar plot is this eval, not the Doom clip. His next run is the same recipe on Qwen 35B; those weights are not out yet.

---

## Hardware and software gotchas

- **Load errors:** Qwen3.5 uses `Qwen3_5ForSequenceClassification` and a composite `text_config`. You want `transformers` 5.x (his config says 5.15.0). If `from_pretrained` fails, install transformers from git.
- **Pad / pool:** right padding; the head pools the last *non-pad* token. Set `pad_token = eos_token` if missing.
- **Do not skip the label remap.** If you train with native SNLI ids, entailment and contradiction swap and the Doom policy collapses.
- **4B inference** ≈ 8–10 GB bf16, ~5 GB int8, ~2.5–4.5 GB 4-bit. The published path is bf16 via Transformers, not a GGUF.
- **35B** will need LoRA + a multi-GPU or 80 GB box. Not required to match the tweet.

---

## What this will and will not match

You **will** get:

- A local model that answers structured choices in one forward pass.
- The same Doom / Flappy “argmax entailment = action” loop.
- The same public wrapper (`predict`, `rerank`, `grade`, `latents`).

You **will not** get:

- TypeSafe Jev, RLCD, or their calibration.
- “Cannot hallucinate” in the Jev marketing sense. Openjev also cannot emit free text — it only emits 3-class / option scores — but it can still pick the wrong option with a confident score.
- Pixel-Doom from the exact tweet overlay, unless `doom_vision.py` shows up. Use `doom.py --video-nli` for the published equivalent.

Related repos that are *not* this result: [TheoLeeCJ/openjev](https://github.com/TheoLeeCJ/openjev) (frozen generative option-token logits, no NLI FT), [vinnylarouge/jevlike](https://github.com/vinnylarouge/jevlike) (tiny scorer on a frozen encoder), Harsha’s Qwen-2.5-1B-RLCD (parallel constrained decoding).

Practical order: Path A first to confirm the interface, then `doom.py --video-nli` for the clip, then `train.py` only if you want to rebuild the 4B checkpoint yourself.
