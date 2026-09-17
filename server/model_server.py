#!/usr/bin/env python
"""The entire Python surface: hold the openjev cross-encoder on the GPU and score hypotheses.

Everything else -- static files, the game, the agent loop, telemetry -- is Bun. This process knows nothing
about the game; it takes a premise and a list of hypotheses and returns P(entailment) for each.

llama.cpp cannot host this checkpoint, which is why this file exists. Sequence-classification heads in
llama.cpp are implemented only for BERT-family encoders (Bert, DistilBert, Roberta, XLMRoberta, NeoBERT,
ModernBert); no decoder-family model has one. And `Qwen3_5` is not a known architecture there at all -- the
nearest is QWEN3NEXT, which is causal-LM only. Serving this would mean implementing the Qwen3.5 hybrid
attention stack *and* inventing a 3-label pooled head for a decoder. Transformers it is.

    python server/model_server.py --ckpt ./openjev_hf/qwen3.5-4b-nli --port 8750

    POST /score  {"premise": "...", "hypotheses": ["...", "..."]}
              -> {"probs": [0.91, 0.02], "argmax": 0, "ms": 34.2}
    GET  /health -> {"ok": true, "device": "cuda", "dim": 2560}
"""
import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

CON, ENT, NEU = 0, 1, 2


class Scorer:
    def __init__(self, ckpt, dtype=torch.bfloat16, max_len=1024):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(ckpt)
        self.model = AutoModelForSequenceClassification.from_pretrained(ckpt, dtype=dtype)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()
        self.template = getattr(self.model.config, "nli_template", None) or "Premise: {premise}\nHypothesis: {hypothesis}"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"          # the head pools the last non-pad token
        tc = self.model.config.get_text_config()
        if tc.pad_token_id is None:
            tc.pad_token_id = self.tok.pad_token_id
        self.backbone = getattr(self.model, self.model.base_model_prefix)
        self.max_len = max_len
        self.lock = __import__("threading").Lock()   # one GPU, serialize callers

    @torch.no_grad()
    def score(self, premise, hypotheses):
        """P(entailment) for each hypothesis against the premise, in one batched forward pass."""
        texts = [self.template.format(premise=premise.strip(), hypothesis=h.strip()) for h in hypotheses]
        with self.lock:
            enc = self.tok(texts, truncation=True, max_length=self.max_len, padding=True, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            h = self.backbone(**enc).last_hidden_state
            last = enc["attention_mask"].sum(1) - 1
            logits = self.model.score(h[torch.arange(h.shape[0], device=h.device), last]).float()
            return torch.softmax(logits, -1)[:, ENT].cpu().tolist()


class Handler(BaseHTTPRequestHandler):
    scorer = None
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
            s = self.scorer
            return self._send({"ok": True, "device": s.device,
                               "dim": s.model.config.get_text_config().hidden_size,
                               "labels": s.model.config.id2label})
        self._send({"error": "not found"}, 404)

    def do_POST(self):
        if not self.path.startswith("/score"):
            return self._send({"error": "not found"}, 404)
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            hyps = req.get("hypotheses") or []
            if not hyps:
                return self._send({"error": "hypotheses required"}, 400)
            t0 = time.perf_counter()
            probs = self.scorer.score(req.get("premise", ""), hyps)
            ms = (time.perf_counter() - t0) * 1000
            best = max(range(len(probs)), key=lambda i: probs[i])
            self._send({"probs": probs, "argmax": best, "ms": ms})
        except Exception as e:                       # never take the server down on one bad request
            self._send({"error": f"{type(e).__name__}: {e}"}, 500)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="./openjev_hf/qwen3.5-4b-nli")
    ap.add_argument("--port", type=int, default=8750)
    ap.add_argument("--max-len", type=int, default=1024)
    args = ap.parse_args()

    print(f"loading {args.ckpt} ...", flush=True)
    Handler.scorer = Scorer(args.ckpt, max_len=args.max_len)
    for _ in range(3):                               # first pass is ~500 ms; warm the kernels
        Handler.scorer.score("A kitchen.", ["The grill is empty."])
    print(f"ready on {Handler.scorer.device}  ->  http://127.0.0.1:{args.port}/score", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
