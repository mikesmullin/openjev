# openjev reproduction — local notes

Everything lives in this directory.

    .venv/                      python 3.12, torch 2.11+cu128, transformers 5.17, flash-linear-attention 0.5.2
    openjev_hf/                 pristine `AlexWortega/openjev` snapshot (8.5 GB checkpoint + code/ + results/)
    code/                       working copy of his scripts; doom.py is PATCHED (doom.py.orig = pristine)
    results_repro/              our runs + videos

## Rerun

    cd code

    # the tweet: text state, state-statement hypotheses, no training at all
    ../.venv/bin/python doom.py --ckpt ../openjev_hf/qwen3.5-4b-nli \
        --episodes 5 --seed 0 --hyp position --zero-shot-only \
        --out ../results_repro/doom_zs_position.json \
        --video-nli ../results_repro/doom_zs_position.mp4

    # his published pipeline as-shipped (degenerate zero-shot nli + latent MLP)
    ../.venv/bin/python doom.py --ckpt ../openjev_hf/qwen3.5-4b-nli \
        --episodes 5 --seed 0 --out ../results_repro/doom_4b.json \
        --video ../results_repro/doom_mlp.mp4 --video-nli ../results_repro/doom_nli.mp4

Needs ~10 GB free VRAM. Nothing else on the box may be holding the GPU.

## The patch to doom.py

The published `code/doom.py` predates the run that made the clip: it only has the
`"The correct action is: {turn left|turn right|attack}"` hypotheses, which his own
`results/doom_4b.json` scores at 1.0 kills — exactly random. The clip is
`videos/doom_zs_position.mp4` (11.0 kills), produced by a `--hyp position --zero-shot-only`
version of the script that was never published. Added here:

* `HYPS` — `action` / `position` / `position_none`, wordings taken verbatim from the
  hypothesis table in `results/full_report.md`.
* `Scorer(ckpt, hyp)` and `Scorer.action_probs()` — folds per-hypothesis P(entailment)
  into one score per action (max over the hypotheses bound to that action).
* `--hyp`, `--zero-shot-only` — the latter skips the noisy-oracle collection + latent MLP
  stage, which the published script always runs.

`code/flappy.py` has the same gap: only `action` / `should`; the `sign` variant that scores
28/28 in the report is missing. Not reconstructed here.

## Errata in tmp/GROK1.md

* `doom.py --video-nli` does NOT give the clip's policy. `--video` vs `--video-nli` only
  selects which policy is recorded; the MLP is trained either way.
* Training hyperparameters were wrong. `qwen3.5-4b-nli/train_result.json` records the real
  run: `--n-train 120000 --bs 32 --grad-accum 1 --max-len 256 --lr 2e-5 --grad-ckpt`,
  seed 42, final MNLI-matched eval accuracy 0.8985.
