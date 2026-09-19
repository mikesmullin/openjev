#!/usr/bin/env python
"""Laya triaging an inbox, in stages, because it falls off a cliff past six options.

[convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) is a 421M ModernBERT-large
"System 1" decision model that answers `choice` / `score` / `noul` questions without generating text.
On the MARS branch it is the fastest and best-calibrated model this repo has run. On a flat 21-way
inbox routing question it is the worst -- and the reason is worth measuring rather than guessing.

## The six-option cliff

Four bank card-charge alerts, correct answer `Expenses`, offered an increasing number of folders:

    options   correct   mean confidence
          2       4/4              0.17
          3       4/4              0.14
          4       4/4              0.27
          6       4/4              0.63
          9       0/4              0.82
         12       0/4              1.00
         18       0/4              0.96

Accuracy is perfect to six and gone by nine. **Confidence moves the opposite way**: it is lowest where
the model is right and 1.00 where it is wrong. So confidence cannot be used to detect the failure --
the flat 21-option version answered `eCommerce` and `Stock` for card charges at 1.00, which looks like
certainty and is noise.

Flat-versus-staged was argued the other way on the openjev branch, and correctly for that model: a
two-stage design needs an intermediate option meaning "belongs in SOME folder", which is vague, and
vague options win against an NLI head. That argument does not apply here, because the stage-1 options
below are not a catch-all -- each names a concrete kind of mail. And the cost objection does not apply
either: openjev re-encoded the premise once per hypothesis, so two stages doubled the bill, while Laya
encodes the text once per call at ~15 ms. Three cheap calls beat one wrong one.

## The tree

Every node is at most six options, which is where the model still works:

    stage 1 (6)   delete | skip | archive | money | work | life
    money   (5)   Expenses, Statements, Income, Stock, Taxes
    work    (4)   Opportunities, Job Applications, Job Interviews, employer
      employer(4) Job/Ancestry, Job/Blizzard, Job/CrowdStrike, Job/HSA
    life    (6)   Travel, Newsletters, Kids, Myself, eCommerce, Real Estate Investment

Most emails cost two calls; only employer mail costs three.

The wire contract is openjev-email's, unchanged, so the page is untouched: the answer is reported as an
index into the flat `[operations..., folders...]` list the page builds from the same `config.yaml`.

## Spam is reported but not trusted

Laya's `noul` is well calibrated on the MARS questions. Here it reads 0.80-0.95 on routine bank charge
alerts across three different wordings, against GLiNER's 0.02 on the same mail. Obvious marketing is
1.00 and community mail 0.06-0.13, so it is not useless, but it false-positives hard on transactional
mail and the page's "likely spam" count should be read with that in mind.
"""

import argparse
import json
import os
import socket
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import yaml

MODEL_ID = "convaiinnovations/laya"
CONFIG = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
CONFIG_EXAMPLE = CONFIG + ".example"

# Which folders sit under which stage-1 branch. Names must match config.yaml; anything in the config
# that is not mentioned here is appended to `life`, so an edited taxonomy still routes somewhere.
GROUPS = {
    "money": (["Expenses", "Statements", "Income", "Stock", "Taxes"],
              "money in or out: receipts, statements, pay, trades, tax"),
    "work":  (["Opportunities", "Job Applications", "Job Interviews", "employer"],
              "anything about a job or an employer"),
    "life":  (["Travel", "Newsletters", "Kids", "Myself", "eCommerce", "Real Estate Investment"],
              "personal life, reading, and side ventures"),
}
EMPLOYER = ["Job/Ancestry", "Job/Blizzard", "Job/CrowdStrike", "Job/HSA"]


def load_taxonomy():
    path = CONFIG if os.path.exists(CONFIG) else CONFIG_EXAMPLE
    cfg = yaml.safe_load(open(path, encoding="utf8")) or {}
    ops = [(o["op"], o.get("label") or o["op"]) for o in cfg.get("operations", [])]
    folders = [(f["name"], f.get("label") or f["name"]) for f in cfg.get("folders", [])]
    return ops, folders


class Triage:
    def __init__(self, model_id, threads, device):
        import torch
        torch.set_num_threads(threads)
        import laya
        t0 = time.perf_counter()
        # laya.load() defaults to CUDA when a GPU is visible. Say which, always: an earlier version of
        # this file set torch threads and assumed that meant CPU, and eight workers quietly took 25 GB
        # of VRAM while the README claimed they were on the CPU.
        self.agent = laya.load(model_id, device=device)
        self.model_id, self.threads, self.device = model_id, threads, device
        self.ops, self.folders = load_taxonomy()
        self.label = dict(self.ops) | dict(self.folders)
        # The page's flat action list: operations first, then folders, from the same config.
        self.flat = [o for o, _ in self.ops] + [f for f, _ in self.folders]
        known = {n for g, _ in GROUPS.values() for n in g} | set(EMPLOYER)
        self.groups = {k: ([n for n in g if n in self.label or n == "employer"], d)
                       for k, (g, d) in GROUPS.items()}
        # Anything the taxonomy has that the tree does not mention still needs a home.
        extra = [f for f, _ in self.folders if f not in known]
        if extra:
            self.groups["life"] = (self.groups["life"][0] + extra, self.groups["life"][1])
        print(f"taxonomy: {len(self.ops)} operations, {len(self.folders)} folders", flush=True)
        print(f"loaded {model_id} on {device} ({threads} threads) in {time.perf_counter()-t0:.1f}s",
              flush=True)
        self.lock = __import__("threading").Lock()

    def _choice(self, text, instructions, criteria):
        q = {"q": {"type": "choice", "instructions": instructions, "criteria": criteria}}
        a = self.agent.predict(text, q)["answers"]["q"]
        return a["choice"], float(a.get("confidence") or 0.0)

    def warm(self):
        self.classify("From: someone.\nSubject: hello.\nBody: nothing much.", {})

    def classify(self, premise, _groups):
        t0 = time.perf_counter()
        with self.lock:
            # Stage 1: six options, which is inside the cliff.
            top = {op: lbl for op, lbl in self.ops}
            top.update({k: d for k, (_, d) in self.groups.items()})
            pick, conf = self._choice(premise, "What should be done with this email?", top)
            stages = 1

            chosen, confidence = pick, conf
            if pick in self.groups:
                names = self.groups[pick][0]
                crit = {n: (self.label.get(n) or "mail from or about that employer") for n in names}
                sub, c2 = self._choice(premise, "Which of these does it belong in?", crit)
                stages += 1
                confidence = conf * c2
                if sub == "employer":
                    crit3 = {n: self.label.get(n, n) for n in EMPLOYER if n in self.label}
                    sub, c3 = self._choice(premise, "Which employer is this about?", crit3)
                    stages += 1
                    confidence *= c3
                chosen = sub

            spam = self.agent.predict(
                premise, {"spam": {"type": "noul", "instructions": "Is this email spam or a promotional blast?"}}
            )["answers"]["spam"]["noul"]
        ms = (time.perf_counter() - t0) * 1000

        idx = self.flat.index(chosen) if chosen in self.flat else 0
        probs = [0.0] * len(self.flat)
        if probs:
            probs[idx] = round(float(confidence), 6)
        return {
            "answers": {
                "action": {"index": idx, "label": chosen, "confidence": round(float(confidence), 6),
                           "probs": probs},
                # Reported, not trusted -- see the module docstring.
                "spam": {"index": 0, "label": "spam", "confidence": float(spam),
                         "probs": [round(float(spam), 6)]},
            },
            "ms": round(ms, 2), "stages": stages,
            "tokensIn": len(premise.split()), "tokensOut": 0,
        }


class ReusePortServer(ThreadingHTTPServer):
    allow_reuse_port = True
    request_queue_size = 64

    def server_bind(self):
        try:
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except OSError:
            pass
        super().server_bind()


class Handler(BaseHTTPRequestHandler):
    svc = None
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/health"):
            return self._send({"ok": True, "backend": "laya-cpu-staged",
                               "model": self.svc.model_id, "threads": self.svc.threads,
                               "device": self.svc.device})
        self._send({"error": "not found"}, 404)

    def do_POST(self):
        if not self.path.startswith("/classify"):
            return self._send({"error": "not found"}, 404)
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            self._send(self.svc.classify(req.get("premise", ""), req.get("groups") or {}))
        except Exception as e:
            self._send({"error": f"{type(e).__name__}: {e}"}, 500)


def serve(args, i):
    Handler.svc = Triage(args.model, args.threads, args.device)
    Handler.svc.warm()
    print(f"worker {i} ready on :{args.port}", flush=True)
    ReusePortServer(("127.0.0.1", args.port), Handler).serve_forever()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--port", type=int, default=8750)
    ap.add_argument("--threads", type=int, default=8)
    # One worker on cuda. Each holds ~1.7 GB of weights plus a ~1.5 GB CUDA context, so eight of them
    # is 25 GB of VRAM -- which is what happened, and it filled the card.
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()
    kids = []
    for i in range(1, args.workers):
        if os.fork() == 0:
            serve(args, i)
            os._exit(0)
        kids.append(i)
    try:
        serve(args, 0)
    finally:
        pass


if __name__ == "__main__":
    main()
