#!/usr/bin/env python
"""GLiNER 2.5 triaging an inbox, on the CPU.

The openjev-email branch put each email to a Qwen3.5-4B NLI cross-encoder as a list of hypothesis
sentences and took the argmax entailment: 16 statements per email, ~180 ms on an RTX 5090. This is the
same task and the same page, with a 194M DeBERTa-v3 classification encoder in place of the 4B NLI head,
and no GPU at all.

The shapes are genuinely different, and that is the point of the comparison:

    openjev    one premise re-encoded once PER HYPOTHESIS, argmax over P(entailment)
    GLiNER     one text encoded ONCE, scored against a label set, all groups in one forward pass

    POST /classify  {"premise": "...", "groups": {"action": [...labels], "spam": [...labels]}}
                 -> {"answers": {"action": {index, confidence, probs}, "spam": {probs}}, "ms": ...}

The wire contract is openjev-email's, unchanged, so web/app.js and the page work as they are. What
changed is what the strings in `groups` should be: openjev wanted checkable assertions about the email
("This is a receipt, invoice, or order confirmation..."), GLiNER wants class names ("Receipts"). See
web/questions.js.

## Synthesized probabilities

`classify_text` returns the winning label and its confidence, not a distribution -- `format_results=False`,
`threshold=0.0` and multi-label mode all still return only the argmax. The page wants a `probs` array,
so the remaining mass is spread uniformly over the losers. `confidence` is real; the per-label spread is
not, and nothing downstream should read meaning into a loser's value.

Spam is the exception and is a real probability: it is asked as a two-class question, so P(spam) is the
winner's confidence or its complement.
"""

import argparse
import json
import os
import socket
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_ID = "fastino/gliner2.5-base-v1"


class Classifier:
    def __init__(self, model_id, threads):
        import torch
        torch.set_num_threads(threads)
        from gliner2 import AutoExtractor
        t0 = time.perf_counter()
        self.model = AutoExtractor.from_pretrained(model_id)
        self.model_id = model_id
        self.threads = threads
        print(f"loaded {model_id} on CPU ({threads} threads) in {time.perf_counter() - t0:.1f}s", flush=True)
        # Within a process, serialize: torch is already using this worker's threads for one forward
        # pass and a second concurrent call would only split them. Parallelism comes from running
        # several worker PROCESSES instead -- see main().
        self.lock = __import__("threading").Lock()

    def warm(self):
        self.classify("An email about nothing in particular.",
                      {"action": ["archive", "delete"], "spam": ["spam", "legitimate email"]})

    def classify(self, premise, groups):
        t0 = time.perf_counter()
        with self.lock:
            raw = self.model.classify_text(premise, {k: list(v) for k, v in groups.items()},
                                           include_confidence=True)
        ms = (time.perf_counter() - t0) * 1000

        answers = {}
        for name, labels in groups.items():
            labels = list(labels)
            v = raw.get(name)
            if isinstance(v, list):
                v = v[0] if v else None
            label = v.get("label") if isinstance(v, dict) else (v if isinstance(v, str) else None)
            conf = float(v.get("confidence") or 0.0) if isinstance(v, dict) else 0.0
            idx = labels.index(label) if label in labels else 0

            if name == "spam":
                # Two-class question, so this is an actual probability rather than a spread.
                p_spam = conf if idx == 0 else 1.0 - conf
                answers[name] = {"index": idx, "label": label, "confidence": conf,
                                 "probs": [round(p_spam, 6)]}
                continue

            rest = (1.0 - conf) / max(1, len(labels) - 1)
            answers[name] = {
                "index": idx, "label": label, "confidence": conf,
                "probs": [round(conf if i == idx else rest, 6) for i in range(len(labels))],
            }
        return {"answers": answers, "ms": round(ms, 2),
                "labels": sum(len(v) for v in groups.values()),
                "tokensIn": len(premise.split()), "tokensOut": 0}


class ReusePortServer(ThreadingHTTPServer):
    """Let every worker bind the same port and have the kernel deal the connections out."""

    allow_reuse_port = True
    # Default is 5 per worker. With several clients in flight the kernel started refusing connections
    # before the workers were actually saturated, which reads as an error rather than as queueing.
    request_queue_size = 64

    def server_bind(self):
        # allow_reuse_port covers 3.11+, but set it directly too rather than depend on the version.
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
            return self._send({"ok": True, "backend": "gliner2.5-cpu",
                               "model": self.svc.model_id, "threads": self.svc.threads,
                               "device": "cpu"})
        self._send({"error": "not found"}, 404)

    def do_POST(self):
        if not self.path.startswith("/classify"):
            return self._send({"error": "not found"}, 404)
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            groups = req.get("groups") or {}
            if not groups:
                return self._send({"error": "groups required"}, 400)
            self._send(self.svc.classify(req.get("premise", ""), groups))
        except Exception as e:
            self._send({"error": f"{type(e).__name__}: {e}"}, 500)


def serve(args, index):
    Handler.svc = Classifier(args.model, args.threads)
    Handler.svc.warm()
    print(f"worker {index} ready on :{args.port}", flush=True)
    ReusePortServer(("127.0.0.1", args.port), Handler).serve_forever()


def main():
    """Scale with processes, not threads.

    A 194M encoder on a short email does not parallelise well inside one forward pass: on a 32-core
    Threadripper, measured end to end,

        1 process   x  8 threads   ->  10.9 emails/s
        1 process   x 32 threads   ->  14.9 emails/s     2.7x the cores, 1.4x the work
        4 processes x  8 threads   ->  30.8 emails/s
        8 processes x  4 threads   ->  36.1 emails/s     same 32 threads, 2.4x the throughput
       16 processes x  4 threads   ->  39.4 emails/s     all 64, diminishing

    Intra-op parallelism saturates around 8 threads and then spends cores on synchronisation. Running
    independent copies of the model instead keeps every core doing useful work, and the default below
    -- 8 workers of 4 threads -- takes half the machine and leaves the rest for everything else.

    Each worker is a separate process holding its own copy of the weights (~1 GB), all bound to the
    same port with SO_REUSEPORT, so the kernel load-balances and the client sees one endpoint.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--port", type=int, default=8750)
    ap.add_argument("--threads", type=int, default=4, help="torch threads PER worker")
    ap.add_argument("--workers", type=int, default=8, help="worker processes sharing the port")
    args = ap.parse_args()

    children = []
    for i in range(1, args.workers):
        pid = os.fork()
        if pid == 0:
            serve(args, i)
            os._exit(0)
        children.append(pid)
    try:
        serve(args, 0)
    finally:
        for pid in children:
            try:
                os.kill(pid, 15)
            except ProcessLookupError:
                pass


if __name__ == "__main__":
    main()
