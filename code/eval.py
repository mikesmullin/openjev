#!/usr/bin/env python
"""Evaluate NLI cross-encoders (ours + dleemiller/ModernCE-large-nli) on MNLI, GPQA-diamond and GSM8K.

Modes follow https://huggingface.co/blog/dleemiller/nli-xenc-ways-to-use:
  * QA rerank (#3): premise=question, hypothesis=candidate; pick argmax P(entailment)
  * grading  (#6): premise=question+reference answer, hypothesis=candidate; entailment <=> correct

Usage:
    python eval.py --models ckpt/qwen3.5-0.8b-nli dleemiller/ModernCE-large-nli --out results/qwen0.8b.json
"""
import argparse
import collections
import json
import os
import re
import time

import numpy as np
import requests
import torch
from datasets import load_dataset
from sklearn.metrics import f1_score
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

CON, ENT, NEU = 0, 1, 2
# Some HF configs mislabel the logit order. ModernCE's config.json says {0 ent, 1 neu, 2 con} but the actual
# logits are [con, ent, neu] (as the model card says; verified: trusting config gives 3% MNLI accuracy).
LABEL_ORDER_OVERRIDES = {"dleemiller/ModernCE-large-nli": ["contradiction", "entailment", "neutral"]}
LLAMA_URL = "http://127.0.0.1:18085/v1/chat/completions"


class NLIScorer:
    """Returns softmax probs [contradiction, entailment, neutral] for (premise, hypothesis) pairs."""

    def __init__(self, path, device="cuda", bs=32, max_len=1024):
        self.cfg = AutoConfig.from_pretrained(path)
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path, dtype=torch.bfloat16).to(device).eval()
        self.device, self.bs, self.max_len = device, bs, max_len
        self.template = getattr(self.cfg, "nli_template", None)  # set by train.py for Qwen models
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        if self.template:
            self.tok.padding_side = "right"
            self.model.config.get_text_config().pad_token_id = self.tok.pad_token_id
        # models differ in label order (ModernCE config: 0 ent, 1 neu, 2 con) -> permute to [con, ent, neu]
        l2i = {v.lower(): int(k) for k, v in self.cfg.id2label.items()}
        if path in LABEL_ORDER_OVERRIDES:
            l2i = {name: i for i, name in enumerate(LABEL_ORDER_OVERRIDES[path])}
        self.perm = [l2i["contradiction"], l2i["entailment"], l2i["neutral"]]
        print(f"{path}: id2label={self.cfg.id2label} perm={self.perm} template={'yes' if self.template else 'no'}")

    @torch.no_grad()
    def predict(self, pairs):
        out = []
        for i in range(0, len(pairs), self.bs):
            chunk = pairs[i : i + self.bs]
            if self.template:
                texts = [self.template.format(premise=p.strip(), hypothesis=h.strip()) for p, h in chunk]
                enc = self.tok(texts, truncation=True, max_length=self.max_len, padding=True, return_tensors="pt")
            else:
                enc = self.tok([p for p, _ in chunk], [h for _, h in chunk], truncation=True,
                               max_length=self.max_len, padding=True, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            logits = self.model(**enc).logits.float()[:, self.perm]
            out.append(torch.softmax(logits, -1).cpu().numpy())
        return np.concatenate(out, 0)


# ----------------------------------------------------------------------------- MNLI
def eval_mnli(scorer, n=None):
    res = {}
    native2ours = {0: ENT, 1: NEU, 2: CON}
    for split in ["validation_matched", "validation_mismatched"]:
        ds = load_dataset("nyu-mll/multi_nli", split=split).filter(lambda x: x["label"] in (0, 1, 2))
        if n:
            ds = ds.shuffle(seed=0).select(range(n))
        probs = scorer.predict(list(zip(ds["premise"], ds["hypothesis"])))
        gold = np.array([native2ours[l] for l in ds["label"]])
        res[split] = {"acc": float((probs.argmax(-1) == gold).mean()), "n": len(ds)}
    return res


# ----------------------------------------------------------------------------- multiple choice
# Every MC task is a list of dicts {"q": str, "opts": [str], "gold": int, "hyp": callable|None}.
# rerank (blog #3): premise=q, hypothesis="The correct answer is: {opt}" -> argmax P(ent)
# grading (blog #6): premise=q + "Reference answer: {gold}", hypothesis="Answer: {opt}" -> ent <=> gold

FEWSHOT_HYP = lambda o: f"Answer: {o}"


def with_demos(q, demos):
    """Few-shot premise: k solved (question, gold answer) pairs followed by the question."""
    return "\n\n".join(f"{dq}\nAnswer: {da}" for dq, da in demos) + "\n\n" + q


def load_gpqa(path="data/gpqa_diamond.csv", fewshot=0, seed=0):
    import csv
    import random as _r
    rows = list(csv.DictReader(open(path))) if os.path.exists(path) else load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    base = []
    for ex in rows:
        opts = [ex["Correct Answer"], ex["Incorrect Answer 1"], ex["Incorrect Answer 2"], ex["Incorrect Answer 3"]]
        base.append({"q": ex["Question"].strip(), "opts": [o.strip() for o in opts], "gold": 0})
    if not fewshot:
        return base
    rng = _r.Random(seed)
    items = []
    for i, it in enumerate(base):  # leave-one-out demos from other diamond questions
        pool = [j for j in range(len(base)) if j != i]
        demos = [(base[j]["q"], base[j]["opts"][base[j]["gold"]]) for j in rng.sample(pool, fewshot)]
        items.append({"q": with_demos(it["q"], demos), "opts": it["opts"], "gold": it["gold"], "hyp": FEWSHOT_HYP})
    return items


def load_mmlu(n=None, seed=0, fewshot=0):
    ds = load_dataset("cais/mmlu", "all", split="test")
    if n:
        ds = ds.shuffle(seed=seed).select(range(n))
    dev = {}
    if fewshot:  # standard MMLU few-shot: dev split, 5 per subject
        for ex in load_dataset("cais/mmlu", "all", split="dev"):
            dev.setdefault(ex["subject"], []).append((ex["question"].strip(), ex["choices"][int(ex["answer"])].strip()))
    items = []
    for ex in ds:
        it = {"q": ex["question"].strip(), "opts": [c.strip() for c in ex["choices"]], "gold": int(ex["answer"])}
        if fewshot:
            it["q"] = with_demos(it["q"], dev[ex["subject"]][:fewshot])
            it["hyp"] = FEWSHOT_HYP
        items.append(it)
    return items


def load_arc(cfg):
    ds = load_dataset("allenai/ai2_arc", cfg, split="test")
    items = []
    for ex in ds:
        labels = ex["choices"]["label"]
        if ex["answerKey"] not in labels:
            continue
        items.append({"q": ex["question"].strip(), "opts": [t.strip() for t in ex["choices"]["text"]], "gold": labels.index(ex["answerKey"])})
    return items


def load_winogrande():
    ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
    items = []
    for ex in ds:
        sent = ex["sentence"]
        opts = [ex["option1"], ex["option2"]]
        # hypothesis = sentence with the blank filled; premise = sentence with the blank left open
        items.append({"q": sent, "opts": opts, "gold": int(ex["answer"]) - 1,
                      "hyp": lambda o, sent=sent: sent.replace("_", o)})
    return items


def load_chess(n=500, seed=0):
    """Synthetic 'Chess (4 legal moves)': random position, 4 candidate moves in SAN, exactly one is legal."""
    import random as _r
    import chess
    rng = _r.Random(seed)
    items = []
    while len(items) < n:
        board = chess.Board()
        for _ in range(rng.randint(6, 40)):
            moves = list(board.legal_moves)
            if not moves or board.is_game_over():
                break
            board.push(rng.choice(moves))
        legal = list(board.legal_moves)
        if len(legal) < 2 or board.is_game_over():
            continue
        legal_san = {board.san(m) for m in legal}
        good = board.san(rng.choice(legal))
        bad = set()
        tries = 0
        while len(bad) < 3 and tries < 500:
            tries += 1
            sq = rng.choice([s for s in chess.SQUARES if board.piece_at(s) and board.piece_at(s).color == board.turn])
            piece = board.piece_at(sq)
            to = rng.choice(chess.SQUARES)
            if to == sq or (board.piece_at(to) and board.piece_at(to).color == board.turn):
                continue
            capture = board.piece_at(to) is not None
            if piece.piece_type == chess.PAWN:
                san = (chess.square_name(sq)[0] + "x" if capture else "") + chess.square_name(to)
            else:
                san = chess.piece_symbol(piece.piece_type).upper() + ("x" if capture else "") + chess.square_name(to)
            if san not in legal_san and san != good:
                bad.add(san)
        if len(bad) < 3:
            continue
        opts = [good] + sorted(bad)
        rng.shuffle(opts)
        pgn = chess.Board().variation_san(board.move_stack)
        q = (f"Chess position after the moves: {pgn}\nFEN: {board.fen()}\n"
             f"{'White' if board.turn else 'Black'} to move. Which of the following moves is legal in this position?")
        items.append({"q": q, "opts": opts, "gold": opts.index(good)})
    return items


def load_hellaswag(n=None, seed=0, split="validation"):
    ds = load_dataset("Rowan/hellaswag", split=split)
    if n:
        ds = ds.shuffle(seed=seed).select(range(n))
    items = []
    for ex in ds:
        ctx = (ex["ctx_a"] + " " + ex["ctx_b"].capitalize()).strip() if ex["ctx_b"] else ex["ctx_a"].strip()
        items.append({"q": f"{ex['activity_label']}: {ctx}", "opts": [e.strip() for e in ex["endings"]], "gold": int(ex["label"]),
                      "hyp": lambda o: o})  # hypothesis = the ending itself
    return items


def load_gsm8k_mc(k=4, n=None, seed=0, split="test"):
    """GSM8K as k-way multiple choice: gold final answer + k-1 numeric distractors (deterministic perturbations)."""
    import random as _r
    rng = _r.Random(seed)
    ds = load_dataset("openai/gsm8k", "main", split=split)
    if n:
        ds = ds.shuffle(seed=seed).select(range(n))
    items = []
    for ex in ds:
        g = extract_number(ex["answer"])
        gv = float(g)
        cands = set()
        gen = [lambda: gv + rng.choice([1, 2, 3, 5, 10]), lambda: gv - rng.choice([1, 2, 3, 5, 10]), lambda: gv * 2, lambda: gv / 2,
               lambda: gv + rng.choice([4, 6, 7, 8, 9, 12, 15, 20, 25, 50]), lambda: gv * 10, lambda: gv * rng.choice([3, 4, 5]),
               lambda: gv - rng.choice([4, 6, 7, 8, 9, 12, 15, 20, 25, 50]), lambda: abs(gv) + rng.randint(100, 999)]
        gi = 0
        while len(cands) < k - 1 and gi < 200:
            v = gen[gi % len(gen)](); gi += 1
            vs = str(int(v)) if float(v) == int(v) else f"{v:.2f}"
            if vs != g and vs not in cands and v >= 0:
                cands.add(vs)
        opts = [g] + sorted(cands, key=lambda x: rng.random())
        order = list(range(len(opts))); rng.shuffle(order)
        opts = [opts[i] for i in order]
        items.append({"q": ex["question"].strip(), "opts": opts, "gold": opts.index(g), "hyp": lambda o: f"The answer is {o}."})
    return items


MC_TASKS = {
    "hellaswag": lambda a: load_hellaswag(a.mc_n),
    "gsm8k_mc4": lambda a: load_gsm8k_mc(4),
    "gsm8k_mc10": lambda a: load_gsm8k_mc(10),
    "gpqa": lambda a: load_gpqa(),
    "mmlu": lambda a: load_mmlu(a.mc_n),
    "gpqa_fewshot": lambda a: load_gpqa(fewshot=a.fewshot),
    "mmlu_fewshot": lambda a: load_mmlu(a.mc_n, fewshot=a.fewshot),
    "arc_easy": lambda a: load_arc("ARC-Easy"),
    "arc_challenge": lambda a: load_arc("ARC-Challenge"),
    "winogrande": lambda a: load_winogrande(),
    "chess": lambda a: load_chess(a.chess_n),
}


def eval_mc(scorer, items):
    rerank_pairs, grade_pairs, grade_gold, offsets = [], [], [], []
    for it in items:
        q, opts, g = it["q"], it["opts"], it["gold"]
        hyp = it.get("hyp") or (lambda o: f"The correct answer is: {o}")
        offsets.append((len(rerank_pairs), len(opts)))
        for j, o in enumerate(opts):
            rerank_pairs.append((q, hyp(o)))
            grade_pairs.append((f"{q}\nReference answer: {opts[g]}", f"Answer: {o}"))
            grade_gold.append(1 if j == g else 0)
    pr = scorer.predict(rerank_pairs)
    pg = scorer.predict(grade_pairs)
    grade_gold = np.array(grade_gold)
    rerank_hits, margin_hits, rank_hits, rand = [], [], [], []
    for (s, k), it in zip(offsets, items):
        g = it["gold"]
        rerank_hits.append(pr[s:s+k, ENT].argmax() == g)
        margin_hits.append((pr[s:s+k, ENT] - pr[s:s+k, CON]).argmax() == g)
        rank_hits.append(pg[s:s+k, ENT].argmax() == g)
        rand.append(1.0 / k)
    pred_ent = (pg.argmax(-1) == ENT).astype(int)
    return {
        "n_questions": len(items),
        "random_baseline": float(np.mean(rand)),
        "rerank_acc": float(np.mean(rerank_hits)),
        "rerank_margin_acc": float(np.mean(margin_hits)),
        "grade_acc": float((pred_ent == grade_gold).mean()),
        "grade_f1": float(f1_score(grade_gold, pred_ent)),
        "grade_rank_acc": float(np.mean(rank_hits)),
        "label_dist_rerank": np.bincount(pr.argmax(-1), minlength=3).tolist(),
    }


# ----------------------------------------------------------------------------- GSM8K
NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_number(text):
    m = re.search(r"####\s*(-?[\d,]*\.?\d+)", text)
    if m:
        s = m.group(1)
    else:
        nums = NUM_RE.findall(text)
        if not nums:
            return None
        s = nums[-1]
    s = s.replace(",", "").rstrip(".")
    try:
        v = float(s)
    except ValueError:
        return None
    return str(int(v)) if v == int(v) else str(v)


def llama_chat(prompt, temperature, max_tokens=512, retries=3):
    body = {
        "model": "qwen35",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.95,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    for _ in range(retries):
        try:
            r = requests.post(LLAMA_URL, json=body, timeout=300)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            print("llama-server error:", e)
            time.sleep(5)
    return ""


def gen_gsm8k_candidates(cache, n_q, n_samples, seed=0):
    if os.path.exists(cache):
        return [json.loads(l) for l in open(cache)]
    ds = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=seed).select(range(n_q))
    rows = []
    t0 = time.time()
    for i, ex in enumerate(ds):
        prompt = (ex["question"].strip() + "\n\nSolve the problem step by step. "
                  "Finish with a final line of the form: #### <number>")
        gold = extract_number(ex["answer"])
        cands = [{"text": llama_chat(prompt, 0.0), "kind": "greedy"}]
        cands += [{"text": llama_chat(prompt, 0.7), "kind": "sample"} for _ in range(n_samples)]
        for c in cands:
            c["pred"] = extract_number(c["text"])
            c["correct"] = c["pred"] is not None and c["pred"] == gold
        rows.append({"question": ex["question"].strip(), "gold_solution": ex["answer"].strip(), "gold": gold, "cands": cands})
        if (i + 1) % 10 == 0:
            print(f"  gsm8k gen {i+1}/{n_q}  {time.time()-t0:.0f}s", flush=True)
    os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
    with open(cache, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


def eval_gsm8k(scorer, rows):
    n = len(rows)
    greedy = np.mean([r["cands"][0]["correct"] for r in rows])
    samples = [r["cands"][1:] for r in rows]
    k = len(samples[0])

    def maj_vote(cs):
        votes = collections.Counter(c["pred"] for c in cs if c["pred"] is not None)
        if not votes:
            return False
        top = votes.most_common(1)[0][0]
        return any(c["correct"] for c in cs if c["pred"] == top)

    maj = np.mean([maj_vote(cs) for cs in samples])
    oracle = np.mean([any(c["correct"] for c in cs) for cs in samples])
    pass1 = np.mean([np.mean([c["correct"] for c in cs]) for cs in samples])

    # best-of-k rerank: premise=question, hypothesis=candidate solution
    pairs = [(r["question"], c["text"]) for r, cs in zip(rows, samples) for c in cs]
    pr = scorer.predict(pairs).reshape(n, k, 3)
    pick = pr[:, :, ENT].argmax(-1)
    rerank = np.mean([samples[i][pick[i]]["correct"] for i in range(n)])
    pick_m = (pr[:, :, ENT] - pr[:, :, CON]).argmax(-1)
    rerank_margin = np.mean([samples[i][pick_m[i]]["correct"] for i in range(n)])

    # grading: premise=question+gold solution, hypothesis="The answer is <pred>"
    gpairs, ggold = [], []
    for r in rows:
        for c in r["cands"]:
            if c["pred"] is None:
                continue
            gpairs.append((f"{r['question']}\nReference solution: {r['gold_solution']}", f"The answer is {c['pred']}."))
            ggold.append(int(c["correct"]))
    pg = scorer.predict(gpairs)
    ggold = np.array(ggold)
    pred = (pg.argmax(-1) == ENT).astype(int)
    return {
        "n_questions": n, "k": k,
        "greedy_acc": float(greedy), "sample_pass1": float(pass1),
        f"maj@{k}": float(maj), f"oracle@{k}": float(oracle),
        f"nli_rerank@{k}": float(rerank), f"nli_rerank_margin@{k}": float(rerank_margin),
        "grade_acc": float((pred == ggold).mean()), "grade_f1": float(f1_score(ggold, pred)),
        "grade_n_pairs": int(len(ggold)), "grade_pos_rate": float(ggold.mean()),
    }


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gsm8k-cache", default="data/gsm8k_cands.jsonl")
    ap.add_argument("--gsm8k-n", type=int, default=200)
    ap.add_argument("--gsm8k-k", type=int, default=4)
    ap.add_argument("--mnli-n", type=int, default=None)
    ap.add_argument("--tasks", nargs="+", default=["mnli", "gpqa", "gsm8k"],
                    help="any of: mnli gsm8k " + " ".join(MC_TASKS))
    ap.add_argument("--mc-n", type=int, default=None, help="subsample size for MMLU (default: full 14k)")
    ap.add_argument("--chess-n", type=int, default=500)
    ap.add_argument("--fewshot", type=int, default=5, help="k demos for *_fewshot tasks")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--gen-only", action="store_true", help="only generate GSM8K candidates and exit")
    args = ap.parse_args()

    rows = None
    if "gsm8k" in args.tasks:
        rows = gen_gsm8k_candidates(args.gsm8k_cache, args.gsm8k_n, args.gsm8k_k)
        print(f"gsm8k candidates: {len(rows)} questions, greedy acc={np.mean([r['cands'][0]['correct'] for r in rows]):.3f}")
    if args.gen_only:
        return

    mc_items = {t: MC_TASKS[t](args) for t in args.tasks if t in MC_TASKS}
    for t, its in mc_items.items():
        print(f"{t}: {len(its)} questions; example: {its[0]['q'][:120]!r} opts={its[0]['opts'][:4]}")
    results = {}
    for m in args.models:
        print(f"\n===== {m}")
        scorer = NLIScorer(m, bs=args.bs, max_len=args.max_len)
        r = {}
        if "mnli" in args.tasks:
            r["mnli"] = eval_mnli(scorer, args.mnli_n); print("mnli", r["mnli"])
        for t in args.tasks:
            if t in MC_TASKS:
                r[t] = eval_mc(scorer, mc_items[t]); print(t, r[t], flush=True)
        if rows is not None:
            r["gsm8k"] = eval_gsm8k(scorer, rows); print("gsm8k", r["gsm8k"])
        results[m] = r
        del scorer; torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)

    # summary table
    print("\n| model | task | n | random | rerank acc | grade acc | grade F1 |")
    print("|---|---|---|---|---|---|---|")
    for m, r in results.items():
        for t in args.tasks:
            if t in MC_TASKS and t in r:
                d = r[t]
                print(f"| {m} | {t} | {d['n_questions']} | {d['random_baseline']:.3f} | {d['rerank_acc']:.3f} | {d['grade_acc']:.3f} | {d['grade_f1']:.3f} |")
        if "mnli" in r:
            print(f"| {m} | mnli m/mm | - | 0.333 | {r['mnli']['validation_matched']['acc']:.3f}/{r['mnli']['validation_mismatched']['acc']:.3f} | - | - |")
        if "gsm8k" in r:
            gs = r["gsm8k"]; k = gs["k"]
            print(f"| {m} | gsm8k (greedy {gs['greedy_acc']:.3f}, maj@{k} {gs[f'maj@{k}']:.3f}, oracle {gs[f'oracle@{k}']:.3f}) | {gs['n_questions']} | - | {gs[f'nli_rerank@{k}']:.3f} | {gs['grade_acc']:.3f} | {gs['grade_f1']:.3f} |")


if __name__ == "__main__":
    main()
