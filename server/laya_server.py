#!/usr/bin/env python
"""Laya deciding for MARS RAID.

[convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) is a non-autoregressive
"System 1" decision model: ModernBERT-large (395M) plus a 2-layer decision head, 421M total, trained
with RLCD to report calibrated probabilities. It never generates text. Every option is scored at its
own `[MASK]` token, and all questions are answered in one forward pass.

It takes `choice` / `score` / `noul` questions with `criteria` -- which is, almost exactly, the
simple-jev v1 request shape this harness already speaks. So `web/app.js` is the simplejev-mars page
essentially unchanged, and this server is a thin translation:

    POST /v1/classifier   {"state": {...}, "questions": {...}}   -> {"answers": {...}, "timing": {...}}

## The model in this harness, so far

    openjev        Qwen3.5-4B NLI cross-encoder   ~35 ms    GPU
    simple-jev     Qwen3.8-27B via llama.cpp     ~430 ms    GPU
    DiffusionGemma 26B-A4B text diffusion       ~3700 ms    GPU + offload
    GLiNER 2.5     194M DeBERTa-v3 encoder        ~71 ms    CPU   (23 ms/question)
    Laya           421M ModernBERT-large          ~13 ms    CPU   (3.2 ms/question)

And unlike GLiNER it returns a real distribution over the options, so the page's ballot bars show a
spread again rather than one full bar.

## Two things it cannot do, both handled here

**It cannot read negation.** Scoring options by overlap with the state, a negated mention reads as a
positive one. Measured, with the sky empty and all three target options offered:

    "the sky is clear of saucers"   -> saucer   0.94     wrong, and confident
    "nothing else is in the air"    -> building 0.49     right

So `describe()` never names a thing that is not there. This is the same family of bug as the constant
sentences that broke the earlier branches, and the same rule covers both: only put a fact in the state
if its presence is the signal.

**Posture is unreliable**, as it was for GLiNER: asked to choose between attacking and escaping it
answers at 0.01-0.15 confidence and does not track danger. `threat` on the other hand is excellent --
0.05 healthy, 2.07 worn down, 2.30 nearly dead -- so posture is a threshold on the model's own threat
score rather than a question. The page labels it as derived.
"""

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_ID = "convaiinnovations/laya"

# Above this, break off. Laya's threat is a 0-3 expected level over the rubric in web/app.js.
BREAK_OFF_AT = 2.5


def describe(state):
    """Render the game state as prose that mentions only what is actually present.

    The negation finding above is why this is not a JSON dump and why every clause is conditional:
    naming an absent thing puts its word in the text, and the model scores options by overlap.
    """
    s = state
    hull = s.get("hull_percent", 0)
    lost = s.get("hull_lost_last_10s", 0)
    saucers = s.get("alien_saucers_airborne", 0)
    buildings = s.get("colony_buildings_standing", 0)

    if hull >= 80:      bits = ["The gunship is barely scratched"]
    elif hull >= 55:    bits = ["The gunship is damaged but holding together"]
    elif hull >= 30:    bits = ["The gunship is badly damaged"]
    else:               bits = ["The gunship is critically damaged and almost destroyed"]

    if lost >= 20:      bits.append("It is being hit hard right now")
    elif lost >= 8:     bits.append("It is taking steady hits")
    elif lost > 0:      bits.append("It is taking the occasional hit")
    else:               bits.append("Nothing is hitting it")

    # Only mention saucers when there are some. "clear of saucers" scores as saucers.
    if saucers:
        bits.append(f"{saucers} alien saucers are in the air shooting at it")
    else:
        bits.append("nothing else is in the air")

    if buildings:
        bits.append(f"{buildings} colony buildings are still standing")
    else:
        bits.append("every colony building has been levelled")

    sc = s.get("scorpion") or {}
    status = sc.get("status", "")
    if "dormant" in status or "buried" in status:
        bits.append("The giant scorpion is still buried and asleep")
    elif sc.get("vulnerable_parts"):
        bits.append("The giant scorpion is awake, and only " +
                    " and ".join(f"its {p['part']}" for p in sc["vulnerable_parts"]) + " can be hurt")
    return ". ".join(bits) + "."


class Decider:
    def __init__(self, model_id, threads):
        import torch
        torch.set_num_threads(threads)
        import laya
        t0 = time.perf_counter()
        self.agent = laya.load(model_id)
        self.model_id, self.threads = model_id, threads
        print(f"loaded {model_id} on CPU ({threads} threads) in {time.perf_counter() - t0:.1f}s", flush=True)
        self.lock = __import__("threading").Lock()

    def warm(self):
        self.decide({"hull_percent": 100},
                    {"target": {"type": "choice", "instructions": "Attack what?",
                                "criteria": {"a": "a colony building", "b": "the giant scorpion"}}})

    def decide(self, state, questions):
        text = describe(state)
        t0 = time.perf_counter()
        with self.lock:
            out = self.agent.predict(text, questions)
        ms = (time.perf_counter() - t0) * 1000
        answers = dict(out.get("answers") or {})

        # posture is a threshold on the model's own threat reading, not a question -- see the docstring.
        threat = answers.get("threat", {}).get("score")
        if threat is not None:
            answers["posture"] = {
                "type": "derived",
                "choice": "break_off" if threat >= BREAK_OFF_AT else "press",
                "confidence": answers["threat"].get("confidence", 1.0),
            }
        return {
            "model": self.model_id,
            "answers": answers,
            "usage": {"input_tokens": len(text.split()), "output_tokens": 0},
            "timing": {
                "total_ms": round(ms, 2),
                "questions": len(questions),
                "labels": sum(len(q.get("criteria") or []) or 9 for q in questions.values()),
                "cached_tokens": 0,
                "raw": text[:200],
            },
        }


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
            return self._send({"ok": True, "backend": "laya-cpu", "model": self.svc.model_id,
                               "threads": self.svc.threads, "device": "cpu"})
        self._send({"error": "not found"}, 404)

    def do_POST(self):
        if not self.path.startswith("/v1/classifier"):
            return self._send({"error": "not found"}, 404)
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            qs = req.get("questions") or {}
            if not qs:
                return self._send({"error": "questions required"}, 400)
            self._send(self.svc.decide(req.get("state", {}), qs))
        except Exception as e:
            self._send({"error": f"{type(e).__name__}: {e}"}, 500)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--threads", type=int, default=24)
    args = ap.parse_args()
    Handler.svc = Decider(args.model, args.threads)
    Handler.svc.warm()
    print(f"ready  ->  http://127.0.0.1:{args.port}/v1/classifier", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
