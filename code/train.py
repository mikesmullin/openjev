#!/usr/bin/env python
"""Fine-tune a Qwen3.5 model as a 3-way NLI cross-encoder on AllNLI (SNLI + MNLI).

Label order follows dleemiller/ModernCE-large-nli: 0=contradiction, 1=entailment, 2=neutral.
Usage:
    python train.py --model Qwen/Qwen3.5-0.8B --out ckpt/qwen3.5-0.8b-nli
    python train.py --model Qwen/Qwen3.5-9B  --out ckpt/qwen3.5-9b-nli --lora --grad-ckpt
"""
import argparse
import json
import math
import os
import random

import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

ID2LABEL = {0: "contradiction", 1: "entailment", 2: "neutral"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
# SNLI / MNLI native: 0=entailment, 1=neutral, 2=contradiction
NATIVE2OURS = {0: 1, 1: 2, 2: 0}
TEMPLATE = "Premise: {premise}\nHypothesis: {hypothesis}"


def format_pair(premise: str, hypothesis: str) -> str:
    return TEMPLATE.format(premise=premise.strip(), hypothesis=hypothesis.strip())


def load_allnli(n_train: int, n_val: int, seed: int):
    snli = load_dataset("stanfordnlp/snli", split="train")
    mnli = load_dataset("nyu-mll/multi_nli", split="train")
    cols = ["premise", "hypothesis", "label"]
    train = concatenate_datasets([snli.select_columns(cols), mnli.select_columns(cols)])
    train = train.filter(lambda x: x["label"] in (0, 1, 2), num_proc=8)
    train = train.shuffle(seed=seed).select(range(min(n_train, len(train))))
    val = load_dataset("nyu-mll/multi_nli", split="validation_matched").select_columns(cols)
    val = val.filter(lambda x: x["label"] in (0, 1, 2)).shuffle(seed=seed).select(range(n_val))
    return train, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=200_000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--head-only", action="store_true", help="freeze the backbone, train only the `score` head")
    ap.add_argument("--eval-steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=-1, help="debug: stop early")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    train, val = load_allnli(args.n_train, args.n_val, args.seed)
    print(f"train={len(train)} val={len(val)}")

    def encode(batch):
        texts = [format_pair(p, h) for p, h in zip(batch["premise"], batch["hypothesis"])]
        enc = tok(texts, truncation=True, max_length=args.max_len)
        enc["labels"] = [NATIVE2OURS[l] for l in batch["label"]]
        return enc

    train = train.map(encode, batched=True, remove_columns=train.column_names, num_proc=8)
    val = val.map(encode, batched=True, remove_columns=val.column_names)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        dtype=torch.bfloat16,
    )
    # Qwen3.5 config is composite (text_config inside); the seq-cls head reads get_text_config().pad_token_id
    model.config.get_text_config().pad_token_id = tok.pad_token_id
    model.config.pad_token_id = tok.pad_token_id
    model.config.nli_template = TEMPLATE  # consumed by eval.py
    model.config.use_cache = False
    # Qwen3.5 checkpoints carry a vision tower that text-only NLI never touches: freeze it.
    n_vis = 0
    for n, p in model.named_parameters():
        if "visual" in n:
            p.requires_grad = False
            n_vis += p.numel()
    print(f"frozen visual params: {n_vis/1e6:.1f}M")
    if args.head_only:
        for n, p in model.named_parameters():
            p.requires_grad = n.startswith("score")
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"head-only: trainable params {n_tr/1e3:.1f}K")

    if args.lora:
        from peft import LoraConfig, TaskType, get_peft_model

        lcfg = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=args.lora_r,
            lora_alpha=2 * args.lora_r,
            lora_dropout=0.05,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
                "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
            ],
            modules_to_save=["score"],
        )
        model = get_peft_model(model, lcfg)
        model.print_trainable_parameters()

    def compute_metrics(p):
        logits = p.predictions[0] if isinstance(p.predictions, (tuple, list)) else p.predictions
        preds = logits.argmax(-1)
        return {"accuracy": float((preds == p.label_ids).mean())}

    total_steps = args.max_steps if args.max_steps > 0 else int(math.ceil(len(train) / (args.bs * args.grad_accum)) * args.epochs)
    targs = TrainingArguments(
        output_dir=args.out + "_trainer",
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, int(0.03 * total_steps)),  # warmup_ratio was removed in transformers 5.15
        weight_decay=0.01,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        bf16=True,
        gradient_checkpointing=args.grad_ckpt,
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="no",
        report_to="none",
        dataloader_num_workers=4,
        seed=args.seed,
        remove_unused_columns=False,
        label_names=["labels"],  # transformers 5.x leaves this empty -> no eval loss/metrics otherwise
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train,
        eval_dataset=val,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    final = trainer.evaluate()
    print("final eval:", final)

    if args.lora:
        model = model.merge_and_unload()
    model.config.nli_template = TEMPLATE
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    with open(os.path.join(args.out, "train_result.json"), "w") as f:
        json.dump({"args": vars(args), "final_eval": final}, f, indent=2)
    print("saved to", args.out)


if __name__ == "__main__":
    main()
