#!/usr/bin/env python
"""Download Bespoke-Nimble-9B, merge its LoRA into the pinned base, and record the result.

This is upstream's own preparation step (nimble/README.md, "Download the model"), kept here as a script
so `bun run model:prepare` is reproducible and so the contract check is not optional.

The release is a 165 MiB LoRA adapter, not a checkpoint: it pins `Qwen/Qwen3.5-9B` at an exact revision
in `schema_config.json` and must be merged against that revision before either scorer can load it. The
merge runs on the CPU and needs ~18 GB of the base in RAM plus room for the merged copy on disk.

`prompt_code_sha256` ties the adapter to the exact text of `parallel_schema.py`. The prompt is part of
the trained contract -- change the prompt and the weights no longer match it -- so a mismatch is fatal
rather than a warning.
"""
import hashlib, json, sys
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "vendor" / "nimble"
CACHE = ROOT / ".cache"
REPO = "bespokelabs/Bespoke-Nimble-9B"

if not (VENDOR / "nimble" / "scoring" / "parallel_schema.py").exists():
    sys.exit(f"missing submodule at {VENDOR}\n  git submodule update --init --recursive")

snapshot = Path(snapshot_download(REPO, cache_dir=str(CACHE / "huggingface" / "hub")))
contract = json.loads((snapshot / "schema_config.json").read_text())

prompt_hash = hashlib.sha256((VENDOR / "nimble" / "scoring" / "parallel_schema.py").read_bytes()).hexdigest()
if contract["task"] != "schema_candidate_classification_v1" or contract["prompt_code_sha256"] != prompt_hash:
    sys.exit(f"contract mismatch: the submodule's scoring prompt is not the one these weights were trained against\n"
             f"  expected {contract['prompt_code_sha256']}\n  found    {prompt_hash}")
print(f"contract ok: {contract['task']}, base {contract['model']} @ {contract['revision'][:12]}")

model_path = snapshot
if (snapshot / "adapter_config.json").exists():
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    model_path = CACHE / "models" / ("nimble-9b-" + snapshot.name)
    if (model_path / "config.json").exists():
        print(f"merged weights already present at {model_path}")
    else:
        print("downloading pinned base and merging the adapter on the CPU (slow, needs ~18 GB RAM)")
        base = Qwen3_5ForConditionalGeneration.from_pretrained(
            contract["model"], revision=contract["revision"],
            dtype=torch.bfloat16, device_map="cpu", cache_dir=str(CACHE / "huggingface" / "hub"),
        )
        merged = PeftModel.from_pretrained(base, snapshot).merge_and_unload(safe_merge=True)
        merged.save_pretrained(model_path)
        AutoTokenizer.from_pretrained(snapshot).save_pretrained(model_path)

# `max_length` is what the adapter was TRAINED at. Upstream's server now permits 8192, but prompts past
# 2048 are outside the training distribution, so the harness budgets against this number, not that one.
config = {"model_path": str(Path(model_path).resolve()), "model_id": REPO,
          "revision": snapshot.name, "max_input_tokens": contract.get("max_length", 2048)}
(CACHE / "nimble-model.json").write_text(json.dumps(config, indent=2))
print("ready:", json.dumps(config, indent=2))
