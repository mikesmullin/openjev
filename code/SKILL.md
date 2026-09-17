---
name: openjev
description: How to turn a Qwen3.5 checkpoint into an NLI cross-encoder ("jev" model), evaluate it zero-shot on multiple-choice benchmarks, make it play Flappy Bird / Doom in real time (text state or raw pixels), and publish to the HF Hub. Use for any work in this repo - training, eval, games, radar plot, HF card.
---

# openjev: Qwen3.5 as an NLI cross-encoder

Everything runs on remote GPU boxes; the Mac only edits and rsyncs. Two hosts:

| host | GPU | python | notes |
|---|---|---|---|
| `eva02` | 1x A6000 48 GB | `/home/alexw/miniconda3/envs/aisci/bin/python` | HF token present (gated GPQA ok); project dir `~/qwen_nli` |
| `azrtx` | 2x RTX Pro 6000 96 GB | `/home/azureuser/dense-trainer/.venv/bin/python` | `HF_HOME=/mnt/hf`; no HF token (pipe it from eva02); GPU 1 often has someone's server (9 GB), still usable |

`run.sh` rsyncs the scripts and launches jobs: `./run.sh sync|train|gen|eval`. For azrtx prefix
`HOST=azrtx REMOTE=/home/azureuser/qwen_nli PY=/home/azureuser/dense-trainer/.venv/bin/python ENVS="HF_HOME=/mnt/hf CUDA_VISIBLE_DEVICES=0"`.
Launch long jobs detached: `(nohup env ... python x.py > logs/x.log 2>&1 < /dev/null &)` inside one `ssh` call, then poll the log
with a `for i in $(seq 1 9); do sleep 60; grep -q DONE_PATTERN log && break; done` loop in a later ssh (make the pattern specific:
`^mlp .*score mean`, not `^mlp `). Never `pkill -f` a pattern that also appears in your own ssh command line (it kills your shell):
use `pkill -f "doom_vision.py --ck[p]t"` style patterns.

## 1. Train the cross-encoder (`train.py`)

`python train.py --model Qwen/Qwen3.5-4B --out ckpt/qwen3.5-4b-nli --n-train 120000 --grad-ckpt` (`--lora` for models that do not fit).

* Model: `AutoModelForSequenceClassification` on a Qwen3.5 checkpoint -> `Qwen3_5ForSequenceClassification`, 3 labels in dleemiller
  order `0 contradiction, 1 entailment, 2 neutral`, last-non-pad-token pooling, input = `"Premise: {p}\nHypothesis: {h}"` (stored as
  `config.nli_template`), right padding, loss = cross-entropy over 3 classes.
* Gotchas: Qwen3.5 config is composite -> set `config.get_text_config().pad_token_id`; transformers 5.x Trainer needs
  `label_names=["labels"]`; `warmup_ratio` is gone in 5.15 (use `warmup_steps`); the checkpoint carries a vision tower - freeze
  it (`"visual" in name`); `lm_head.weight` UNEXPECTED on load is fine; SNLI/MNLI native labels are `0 ent,1 neu,2 con` -> remap.
* Timing: 0.8B ~30 min / 2B ~40 min on A6000; 4B ~70 min on RTX Pro 6000 (40 GB). 9B full FT fits on azrtx (68 GB) but was not finished.
* `flash-linear-attention` (pip) gives the fused Gated DeltaNet path (4B decision 138 ms -> ~50 ms); without it torch fallback.

## 2. Zero-shot evaluation (`eval.py`)

`python eval.py --models ckpt/qwen3.5-4b-nli dleemiller/ModernCE-large-nli --out results/x.json --tasks mnli gpqa mmlu arc_easy arc_challenge winogrande hellaswag gsm8k_mc4 gsm8k_mc10 chess gsm8k`

* Two modes per MC task, straight from the blog: **rerank** (premise = question, hypothesis = `The correct answer is: {opt}`, argmax
  P(entailment)) and **grading** (premise = question + `Reference answer: {gold}`, entailment <=> option is gold).
* ModernCE's `config.json` label order is wrong; `LABEL_ORDER_OVERRIDES` in `eval.py` fixes it (else 3% MNLI).
* GPQA is gated: on azrtx read it from `data/gpqa_diamond.csv` (copied from eva02's HF cache). Chess = synthetic 4 SAN moves, 1 legal
  (`load_chess`). GSM8K k-choice = gold + numeric distractors (`load_gsm8k_mc`). GSM8K best-of-k uses candidates from `--gen-only`
  (a llama-server on eva02:18085 was used; cached in `data/gsm8k_cands.jsonl`).
* `--tasks mmlu_fewshot gpqa_fewshot --fewshot 5 --mc-n 2000 --bs 16` puts demos in the premise (does not help below 4B).
* `summarize.py` merges `results/*.json` into `results/summary.md`; `radar.py` draws the Jev/Terra/openjev radar (Jev/Terra values are
  hard-coded estimates read off their chart).

## 3. Games, zero-shot (`flappy.py`, `doom.py`, `doom_vision.py`)

The trick that makes zero-shot work: hypotheses are **statements about the state**, not action names, and each statement maps
to an action. `The correct action is: flap` scores 0; `The offset relative to the gap centre is negative / positive` scores 28/28.

* Flappy: `python flappy.py --ckpt ckpt/qwen3.5-4b-nli --episodes 6 --fps 15 --max-steps 900 --record-only --zero-shot-only --hyp sign`
  (`--hyp position|sign|should|action`, `--prompt base|numeric|coach|ascii`). Real-time = fixed tick; while the model thinks the bird
  glides (skipped frames reported). Video: `flappy_video.py --json ... --policy nli --out x.mp4` (game 15 fps shown at x2).
* Doom (ViZDoom `defend_the_center`, pip `vizdoom`): `python doom.py --ckpt ... --episodes 5 --zero-shot-only --hyp position --video-nli x.mp4`.
  Text state comes from the labels buffer; keep only real monsters (`ENEMY_NAMES`), blood splats otherwise get shot at.
  One decision per 4 tics = 114 ms budget; 4B text decision ~57 ms.
* Doom from pixels: `python doom_vision.py --ckpt ... --mode zeroshot --variants pixels --episodes 5 --video x.mp4`. Frame (320x240,
  80 image tokens) goes in as `<|vision_start|><|image_pad|>*n<|vision_end|>` inside the premise; pass `mm_token_type_ids`
  (`input_ids == image_pad_id`) and `pixel_values`/`image_grid_thw` repeated per hypothesis. Needs torchvision (installed `--no-deps`).
  **Replace the vision patch-embed Conv3d with an fp32 conv** (`FastPatchEmbed`): the bf16 cuDNN path takes ~2 s per frame on
  Blackwell, fp32 takes 0.3 ms. Best hypotheses so far: monster x-position in pixels (`pixels` variant, 5.2 kills vs random 1.0).
* `--mode mlp` / `--mode finetune` / `latent_mlp.py` exist but the user wants zero-shot only - do not use them unless asked.

## 4. Publish (`hf_publish.py`, `make_card.py`)

Repo `AlexWortega/openjev`: checkpoint in `qwen3.5-4b-nli/`, `modeling_openjev.py`, `code/`, `results/`, `videos/`, `assets/`.
Token handling: never write it on azrtx; pipe it:
`ssh eva02 'cat ~/.cache/huggingface/token' | ssh azrtx 'cd ~/qwen_nli && read -r TOK; HF_TOKEN="$TOK" python hf_publish.py ...'`.
The card is marketing-only by user request: title `openjev — Qwen3.5 trained as jev model`, the two Doom videos (`<video>` tags with
`resolve/main/videos/...`), the radar, a short pitch, files, usage. No measurement tables, no dataset descriptions (architecture + loss
only). `make_card.py` still generates the full report -> keep it in `results/full_report.md`, not in README.
The rendered Hub page caches ~15 min; check the raw README to verify.

## 5. Results to remember (all zero-shot, 4B)

MNLI 0.904/0.907 (ModernCE 0.909/0.921); rerank w/o reference: ARC-E 0.77, ARC-C 0.59, MMLU 0.47, GPQA ~chance; grading with reference
0.94-0.99. Flappy 28/28 (`sign`), Doom text 11 kills (oracle 18.8), Doom pixels 5.2. Few-shot in the premise: no gain below 4B.
