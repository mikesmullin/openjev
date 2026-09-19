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
        # torch is already using every core for one forward pass; letting several requests in at once
        # would split them and make each slower. The page runs workers, so serialize here.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--port", type=int, default=8750)
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()

    Handler.svc = Classifier(args.model, args.threads)
    Handler.svc.warm()
    print(f"ready  ->  http://127.0.0.1:{args.port}/classify", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
