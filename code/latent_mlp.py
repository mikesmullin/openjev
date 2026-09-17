#!/usr/bin/env python
"""Frozen NLI-Qwen latent -> MLP head trained with soft BCE to score multiple-choice options.

extract: run the fine-tuned NLI model once per (question, option) pair, save the pooled last-token hidden state
         (the input of the `score` head) + the NLI logits for train and test splits of every MC task.
train:   MLP on the latents, soft-BCE loss (gold=1-eps, others=eps), per-task and joint; report per-question
         argmax accuracy vs the plain NLI entailment rerank on the same test pairs.

    python latent_mlp.py extract --ckpt ckpt/qwen3.5-4b-nli --out data/latents_4b
    python latent_mlp.py train --latents data/latents_4b --out results/latent_mlp_4b.json
"""
import argparse
import csv
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset

import eval as E

TASKS = ["gpqa", "mmlu", "arc_easy", "arc_challenge", "winogrande", "chess"]
EXTRA_TASKS = ["hellaswag", "gsm8k_mc4", "gsm8k_mc10"]


# ----------------------------------------------------------------------------- train splits
BIG = {"mmlu": 30000, "winogrande": 40398, "chess": 10000}  # --big train sets (GPQA/ARC have no more data)


def train_items(task, seed=0, big=False):
    if big and task == "mmlu":  # MMLU auxiliary_train (ARC/OBQA/RACE-style MC, 99.8k) subsample
        ds = load_dataset("cais/mmlu", "all", split="auxiliary_train").shuffle(seed=seed).select(range(BIG["mmlu"]))
        return [{"q": ex["question"].strip(), "opts": [c.strip() for c in ex["choices"]], "gold": int(ex["answer"])} for ex in ds]
    if big and task == "winogrande":
        ds = load_dataset("allenai/winogrande", "winogrande_xl", split="train")
        return [{"q": ex["sentence"], "opts": [ex["option1"], ex["option2"]], "gold": int(ex["answer"]) - 1,
                 "hyp": (lambda o, s=ex["sentence"]: s.replace("_", o))} for ex in ds]
    if big and task == "chess":
        return E.load_chess(BIG["chess"], seed=1)
    if task == "hellaswag":
        return E.load_hellaswag(8000, seed=seed, split="train")
    if task == "gsm8k_mc4":
        return E.load_gsm8k_mc(4, split="train")
    if task == "gsm8k_mc10":
        return E.load_gsm8k_mc(10, split="train")
    if task == "gpqa":  # gpqa_main minus the diamond questions
        diamond = {it["q"] for it in E.load_gpqa()}
        items = []
        for ex in csv.DictReader(open("data/gpqa_main.csv")):
            q = ex["Question"].strip()
            if q in diamond:
                continue
            opts = [ex["Correct Answer"], ex["Incorrect Answer 1"], ex["Incorrect Answer 2"], ex["Incorrect Answer 3"]]
            items.append({"q": q, "opts": [o.strip() for o in opts], "gold": 0})
        return items
    if task == "mmlu":  # validation + dev (test is the eval split)
        items = []
        for split in ["validation", "dev"]:
            for ex in load_dataset("cais/mmlu", "all", split=split):
                items.append({"q": ex["question"].strip(), "opts": [c.strip() for c in ex["choices"]], "gold": int(ex["answer"])})
        return items
    if task in ("arc_easy", "arc_challenge"):
        cfg = "ARC-Easy" if task == "arc_easy" else "ARC-Challenge"
        ds = load_dataset("allenai/ai2_arc", cfg, split="train")
        items = []
        for ex in ds:
            labels = ex["choices"]["label"]
            if ex["answerKey"] in labels:
                items.append({"q": ex["question"].strip(), "opts": [t.strip() for t in ex["choices"]["text"]], "gold": labels.index(ex["answerKey"])})
        return items
    if task == "winogrande":
        ds = load_dataset("allenai/winogrande", "winogrande_xl", split="train").shuffle(seed=seed).select(range(8000))
        return [{"q": ex["sentence"], "opts": [ex["option1"], ex["option2"]], "gold": int(ex["answer"]) - 1,
                 "hyp": (lambda o, s=ex["sentence"]: s.replace("_", o))} for ex in ds]
    if task == "chess":
        return E.load_chess(2000, seed=1)  # test uses seed 0
    raise ValueError(task)


def test_items(task):
    ns = argparse.Namespace(mc_n=None, chess_n=500, fewshot=0)
    return E.MC_TASKS[task](ns)


# ----------------------------------------------------------------------------- extract
@torch.no_grad()
def extract(args):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(args.ckpt, dtype=torch.bfloat16).cuda().eval()
    template = getattr(model.config, "nli_template", None) or "Premise: {premise}\nHypothesis: {hypothesis}"  # raw base models
    if model.config.get_text_config().pad_token_id is None:
        model.config.get_text_config().pad_token_id = tok.pad_token_id
    tok.padding_side = "right"
    backbone = getattr(model, model.base_model_prefix)
    os.makedirs(args.out, exist_ok=True)

    def run(items, path):
        texts, qid, gold, nopts = [], [], [], []
        for i, it in enumerate(items):
            hyp = it.get("hyp") or (lambda o: f"The correct answer is: {o}")
            nopts.append(len(it["opts"]))
            for j, o in enumerate(it["opts"]):
                texts.append(template.format(premise=it["q"].strip(), hypothesis=hyp(o).strip()))
                qid.append(i); gold.append(int(j == it["gold"]))
        X, L = [], []
        for s in range(0, len(texts), args.bs):
            enc = tok(texts[s:s + args.bs], truncation=True, max_length=args.max_len, padding=True, return_tensors="pt").to("cuda")
            h = backbone(**enc).last_hidden_state
            last = enc["attention_mask"].sum(1) - 1
            pooled = h[torch.arange(h.shape[0], device=h.device), last]
            X.append(pooled.float().cpu().numpy().astype(np.float16))
            L.append(model.score(pooled).float().cpu().numpy())
        np.savez(path, X=np.concatenate(X), nli=np.concatenate(L), qid=np.array(qid), gold=np.array(gold), nopts=np.array(nopts))
        print(f"  {path}: {len(items)} questions, {len(texts)} pairs", flush=True)

    for task in args.tasks:
        print(task, flush=True)
        run(train_items(task, big=args.big), f"{args.out}/{task}_train.npz")
        if not args.skip_test:
            run(test_items(task), f"{args.out}/{task}_test.npz")


# ----------------------------------------------------------------------------- train
class MLP(nn.Module):
    def __init__(self, d, hidden=512, p=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Dropout(p), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_npz(path, use_nli):
    z = np.load(path)
    X = z["X"].astype(np.float32)
    if use_nli:
        X = np.concatenate([X, z["nli"].astype(np.float32)], 1)
    return X, z["qid"], z["gold"], z["nopts"], z["nli"]


def per_question_acc(scores, qid, gold):
    """argmax over each question's options == gold option"""
    hits, n = 0, 0
    order = np.argsort(qid, kind="stable")
    scores, qid, gold = scores[order], qid[order], gold[order]
    starts = np.r_[0, np.flatnonzero(np.diff(qid)) + 1, len(qid)]
    for a, b in zip(starts[:-1], starts[1:]):
        hits += int(gold[a:b][scores[a:b].argmax()] == 1); n += 1
    return hits / n


def soft_bce(logits, y, eps, pos_weight):
    target = y * (1 - eps) + (1 - y) * eps
    w = torch.where(y > 0.5, pos_weight, 1.0)
    return (w * nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")).mean()


def fit(Xtr, ytr, qtr, Xva, yva, qva, args, dev="cuda"):
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    norm = lambda X: torch.tensor((X - mu) / sd, dtype=torch.float32, device=dev)
    Xtr_t, ytr_t = norm(Xtr), torch.tensor(ytr, dtype=torch.float32, device=dev)
    Xva_t = norm(Xva)
    pos_weight = torch.tensor(float((1 - ytr.mean()) / max(ytr.mean(), 1e-6)), device=dev)
    torch.manual_seed(args.seed)
    model = MLP(Xtr.shape[1], args.hidden, args.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best, best_state, bad = -1, None, 0
    n = len(Xtr_t)
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        for s in range(0, n, args.bs):
            idx = perm[s:s + args.bs]
            loss = soft_bce(model(Xtr_t[idx]), ytr_t[idx], args.eps, pos_weight)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            acc = per_question_acc(model(Xva_t).cpu().numpy(), qva, yva)
        if acc > best:
            best, bad, best_state = acc, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, (mu, sd), best, ep + 1


def predict(model, stats, X, dev="cuda"):
    mu, sd = stats
    with torch.no_grad():
        return model(torch.tensor((X - mu) / sd, dtype=torch.float32, device=dev)).cpu().numpy()


def grouped_split(qid, frac, seed):
    qs = np.unique(qid)
    rng = np.random.RandomState(seed)
    rng.shuffle(qs)
    hold = set(qs[: max(1, int(len(qs) * frac))].tolist())
    mask = np.array([q in hold for q in qid])
    return ~mask, mask


def train(args):
    train_dir = args.train_dir or args.latents
    data = {t: {"train": load_npz(f"{train_dir}/{t}_train.npz", args.use_nli),
                "test": load_npz(f"{args.latents}/{t}_test.npz", args.use_nli)} for t in args.tasks}
    if args.frac < 1.0:  # data-scaling: keep a grouped fraction of the train questions
        for t in args.tasks:
            X, qid, gold, nopts, nli = data[t]["train"]
            keep, _ = grouped_split(qid, 1 - args.frac, args.seed + 1)
            data[t]["train"] = (X[keep], qid[keep], gold[keep], nopts, nli[keep])
    results = {}
    print(f"features: {data[args.tasks[0]]['train'][0].shape[1]}d  eps={args.eps}  use_nli={args.use_nli}")

    # baseline: NLI entailment rerank on the same pairs (sanity vs eval.py numbers)
    for t in args.tasks:
        X, qid, gold, nopts, nli = data[t]["test"]
        p = torch.softmax(torch.tensor(nli), -1).numpy()
        results[t] = {"n_test_q": int(len(nopts)), "n_train_q": int(len(data[t]["train"][3])),
                      "random": float(np.mean(1.0 / nopts)), "nli_rerank": per_question_acc(p[:, E.ENT], qid, gold)}

    # per-task MLP
    for t in args.tasks:
        Xtr, qtr, ytr, _, _ = data[t]["train"]
        tr, va = grouped_split(qtr, 0.1, args.seed)
        model, stats, va_acc, eps_run = fit(Xtr[tr], ytr[tr], qtr[tr], Xtr[va], ytr[va], qtr[va], args)
        Xte, qte, yte, _, _ = data[t]["test"]
        results[t]["mlp_per_task"] = per_question_acc(predict(model, stats, Xte), qte, yte)
        results[t]["mlp_per_task_val"] = va_acc
        print(f"{t:14s} per-task: val {va_acc:.3f} test {results[t]['mlp_per_task']:.3f} ({eps_run} ep)", flush=True)

    # joint MLP: all tasks pooled (question ids offset per task)
    Xs, ys, qs, off = [], [], [], 0
    for t in args.tasks:
        Xtr, qtr, ytr, _, _ = data[t]["train"]
        Xs.append(Xtr); ys.append(ytr); qs.append(qtr + off); off += qtr.max() + 1
    Xtr, ytr, qtr = np.concatenate(Xs), np.concatenate(ys), np.concatenate(qs)
    tr, va = grouped_split(qtr, 0.1, args.seed)
    model, stats, va_acc, eps_run = fit(Xtr[tr], ytr[tr], qtr[tr], Xtr[va], ytr[va], qtr[va], args)
    for t in args.tasks:
        Xte, qte, yte, _, _ = data[t]["test"]
        results[t]["mlp_joint"] = per_question_acc(predict(model, stats, Xte), qte, yte)
    print(f"joint: val {va_acc:.3f} ({eps_run} ep)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"args": vars(args), "results": results}, open(args.out, "w"), indent=2)
    print("\n| task | train q | test q | random | NLI rerank | MLP per-task | MLP joint |\n|---|---|---|---|---|---|---|")
    for t in args.tasks:
        r = results[t]
        print(f"| {t} | {r['n_train_q']} | {r['n_test_q']} | {r['random']:.3f} | {r['nli_rerank']:.3f} | {r['mlp_per_task']:.3f} | {r['mlp_joint']:.3f} |")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--ckpt", required=True); e.add_argument("--out", required=True)
    e.add_argument("--tasks", nargs="+", default=TASKS); e.add_argument("--bs", type=int, default=32); e.add_argument("--max-len", type=int, default=1024)
    e.add_argument("--big", action="store_true", help="large train sets for mmlu/winogrande/chess"); e.add_argument("--skip-test", action="store_true")
    t = sub.add_parser("train")
    t.add_argument("--latents", required=True); t.add_argument("--out", required=True)
    t.add_argument("--tasks", nargs="+", default=TASKS)
    t.add_argument("--eps", type=float, default=0.1, help="soft-BCE label smoothing: gold=1-eps, others=eps")
    t.add_argument("--use-nli", action="store_true", help="append the 3 NLI logits to the latent")
    t.add_argument("--hidden", type=int, default=512); t.add_argument("--dropout", type=float, default=0.1)
    t.add_argument("--lr", type=float, default=1e-3); t.add_argument("--wd", type=float, default=1e-2)
    t.add_argument("--bs", type=int, default=512); t.add_argument("--epochs", type=int, default=60); t.add_argument("--patience", type=int, default=8)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--train-dir", default=None, help="take *_train.npz from here (e.g. a --big extraction)")
    t.add_argument("--frac", type=float, default=1.0, help="fraction of train questions to use")
    args = ap.parse_args()
    random.seed(args.seed if hasattr(args, "seed") else 0)
    (extract if args.cmd == "extract" else train)(args)


if __name__ == "__main__":
    main()
