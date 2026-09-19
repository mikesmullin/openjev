#!/usr/bin/env python
"""GLiNER 2.5 as the decision-maker for MARS RAID.

Fourth model to drive this harness, and the smallest by three orders of magnitude. The lineage:

    openjev        Qwen3.5-4B NLI cross-encoder   score hypotheses      ~35 ms   GPU
    simple-jev     Qwen3.8-27B via llama.cpp      label logits         ~430 ms   GPU
    DiffusionGemma 26B-A4B text diffusion         generate JSON       ~3700 ms   GPU + offload
    GLiNER 2.5     194M DeBERTa-v3 encoder        classify labels       ~68 ms   CPU

GLiNER2.5 is a boundary-based information-extraction encoder: instead of generating anything, it scores
a supplied set of labels against the text. `classify_text` takes a dict of independent tasks and answers
all of them in one forward pass, which is exactly the shape this harness has wanted since the beginning
-- one state, several named questions, one inference.

    POST /v1/decide   {"state": {...}, "candidates": [{"uid","description"}, ...]}
                   -> {"answers": {...}, "timing": {...}}

## It runs on the CPU, on purpose

At 194M parameters this does not need a GPU, and leaving the GPU alone means it can share the machine
with whatever else is loaded. Measured on 12 CPU threads, four questions in one call:

    gliner2.5-base-v1   (194M)   68.1 ms   -> 17.0 ms per question
    gliner2.5-small-v1   (74M)   33.1 ms   ->  8.3 ms per question

## What it gives up

`classify_text` returns the winning label and its confidence, not a distribution over all of them --
`format_results=False`, `threshold=0.0` and multi-label mode all still return just the argmax. So the
page's per-candidate bars show one full bar rather than a spread, as with the diffusion branch. The
classifier branches (openjev, simple-jev) are the only ones that could rank the whole ballot.
"""

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_ID = "fastino/gliner2.5-base-v1"

# Short, distinct class names -- not sentences.
#
# The first version used a full sentence per label ("the ship is about to be destroyed"). GLiNER read
# `threat` as maximal on every tick, including at 100 percent hull with an empty sky. That is the same
# "wording is the interface" lesson as the previous three branches, but pointing the opposite way: a
# generative model wanted an option that argued for itself, while a classification encoder wants a
# label that names a class. Long, mutually-similar sentences give it nothing to separate.
THREAT = ["safe", "minor damage", "heavy damage", "critical"]

# `wake` needs the two options to state the *reason*, not the action. Measured over three states
# (11 buildings / 2 buildings / colony flattened):
#
#   ["wake the scorpion", "leave it buried"]                      always "leave it buried"
#   ["wake the sleeping scorpion", "keep destroying the colony"]  always "wake"
#   the pair below                                                right on 2 of 3, and 0.99 on the
#                                                                 case that matters (colony flattened)
WAKE = {"yes": "the colony is finished so wake the scorpion",
        "no": "there are still buildings to destroy"}

# There is no POSTURE task, and that is a finding rather than an omission.
#
# Asked to choose between attack and retreat, GLiNER answers with total confidence and exactly
# backwards -- across three different wordings, on the same three states:
#
#   ["attack", "retreat"]                    safe -> retreat (1.00)   dying -> attack (1.00)
#   ["keep attacking", "break off and escape"]  safe -> break off (0.91)  dying -> keep attacking (1.00)
#   ["press the attack on the colony", "flee to survive"]  safe -> flee (1.00)  dying -> press (0.99)
#
# Consistent inversion at high confidence, not noise: "attack" matches a text full of saucers and
# damage, which is precisely when the ship should be leaving. Inverting it would be cargo-culting a
# result we do not understand, so posture is derived from the model's own `threat` instead -- that one
# is monotonic and correct.
BREAK_OFF_AT = 3.0

# Targets are classified by KIND, not by instance.
#
# Asking it to pick between "destroy the colony building 137 metres away" and "... 174 metres away" is
# asking a text classifier to separate two labels that differ by two digits, which it cannot do and
# should not be asked to do. It classifies what to attack; the code picks the nearest one of that kind,
# which it already has sorted. Geometry stays in the code, judgement stays in the model -- the same
# split the whole harness is built on.
KINDS = {"building": "colony building", "saucer": "alien saucer",
         "boss": "the giant scorpion", "other": "nothing"}


def describe(state):
    """Render the game state as language, because that is what this model can actually read.

    Measured, with `posture` over ["attack", "retreat"]:

        "Hull 15 percent. lost 25 hull in the last ten seconds."   -> attack  (0.99)  WRONG
        "critically damaged and almost destroyed, must escape now" -> retreat (1.00)  right

    A DeBERTa encoder scoring labels has no arithmetic: `15 percent` is a token, not a quantity, so it
    cannot tell it apart from `95 percent`. Given the same situation in words it is decisive and
    correct. So the numeric bands below are crossed in code and the model judges what to do about the
    result -- which is the same division this harness makes for geometry, one level up. It is worth
    being plain about it: the thresholds are ours, the decision is the model's.

    One phrase mattered more than any threshold. An earlier version ended every state with the constant
    sentence "The ship dies at zero hull." -- true, harmless-looking, and present on every tick. It
    pinned `threat` to "critical" at 100 percent hull with an empty sky. Fourth branch, fourth time a
    sentence that is always true has broken a decision.
    """
    s = state
    hull = s.get("hull_percent", 0)
    lost = s.get("hull_lost_last_10s", 0)
    saucers = s.get("alien_saucers_airborne", 0)
    buildings = s.get("colony_buildings_standing", 0)

    if hull >= 80:      condition = "The gunship is barely scratched"
    elif hull >= 55:    condition = "The gunship is damaged but holding together"
    elif hull >= 30:    condition = "The gunship is badly damaged"
    else:               condition = "The gunship is critically damaged and almost destroyed"

    if lost >= 20:      taking = "It is being hit hard right now"
    elif lost >= 8:     taking = "It is taking steady hits"
    elif lost > 0:      taking = "It is taking the occasional hit"
    else:               taking = "Nothing is hitting it"

    bits = [condition, taking]
    bits.append(f"{saucers} alien saucers are in the air shooting at it" if saucers
                else "the sky is clear of saucers")
    bits.append(f"{buildings} colony buildings are still standing" if buildings
                else "the colony is flattened")
    sc = s.get("scorpion") or {}
    status = sc.get("status", "")
    if "dormant" in status or "buried" in status:
        bits.append("the giant scorpion is still buried and asleep")
    elif sc.get("vulnerable_parts"):
        bits.append("the giant scorpion is awake, and only " +
                    " and ".join(f"its {p['part']}" for p in sc["vulnerable_parts"]) + " can be hurt")
    else:
        bits.append("the giant scorpion is awake")
    return ". ".join(bits) + "."


class Decider:
    def __init__(self, model_id, threads):
        import torch
        torch.set_num_threads(threads)
        from gliner2 import AutoExtractor
        t0 = time.perf_counter()
        self.model = AutoExtractor.from_pretrained(model_id)
        self.model_id = model_id
        print(f"loaded {model_id} on CPU ({threads} threads) in {time.perf_counter() - t0:.1f}s", flush=True)
        # One model, and the forward pass is short. Serializing keeps the thread pool undivided, which
        # is faster here than letting two calls split 12 cores.
        self.lock = __import__("threading").Lock()

    def warm(self):
        self.decide({"hull_percent": 100}, [{"uid": "building:0", "description": "a colony building"},
                                            {"uid": "saucer:0", "description": "an alien saucer"}], False)

    def decide(self, state, candidates, ask_wake):
        text = describe(state)
        # Candidates arrive sorted by distance, so the first of a kind is the nearest of that kind.
        order = []
        first_of = {}
        for c in candidates:
            kind = c["uid"].split(":")[0]
            if kind not in first_of:
                first_of[kind] = c["uid"]
                order.append(kind)
        by_label = {KINDS.get(k, k): first_of[k] for k in order}
        tasks = {
            "target": list(by_label),
            "threat": THREAT,
        }
        if ask_wake:
            tasks["wake"] = list(WAKE.values())

        t0 = time.perf_counter()
        with self.lock:
            raw = self.model.classify_text(text, tasks, include_confidence=True)
        ms = (time.perf_counter() - t0) * 1000

        def got(key):
            v = raw.get(key)
            if isinstance(v, list):
                v = v[0] if v else None
            if isinstance(v, dict):
                return v.get("label"), float(v.get("confidence") or 0.0)
            return (v, 1.0) if isinstance(v, str) else (None, 0.0)

        tgt_label, tgt_conf = got("target")
        target = by_label.get(tgt_label, candidates[0]["uid"])
        kind_of = {v: k for k, v in by_label.items()}
        thr_label, thr_conf = got("threat")
        threat = float(THREAT.index(thr_label)) if thr_label in THREAT else 0.0
        # Derived from the model's threat reading, not asked of the model. See BREAK_OFF_AT.
        posture = "break_off" if threat >= BREAK_OFF_AT else "press"

        answers = {
            "target": {
                "type": "choice", "choice": target, "confidence": tgt_conf,
                # Winner only; see the module docstring.
                "probabilities": {c["uid"]: (tgt_conf if c["uid"] == target else 0.0) for c in candidates},
            },
            "posture": {"type": "derived", "choice": posture, "confidence": thr_conf},
            "threat": {"type": "score", "score": threat},
        }
        if ask_wake:
            wk_label, wk_conf = got("wake")
            answers["wake"] = {"type": "choice",
                               "choice": next((k for k, v in WAKE.items() if v == wk_label), "no"),
                               "confidence": wk_conf}
        return {
            "model": self.model_id,
            "answers": answers,
            "usage": {"input_tokens": len(text.split()), "output_tokens": 0},
            "timing": {
                "total_ms": round(ms, 1),
                "questions": len(tasks),      # all answered in one forward pass
                "labels": sum(len(v) for v in tasks.values()),
                "cached_tokens": 0,
                "raw": text[:200],
            },
        }


class Handler(BaseHTTPRequestHandler):
    decider = None
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
            return self._send({"ok": True, "backend": "gliner2.5-cpu", "model": self.decider.model_id})
        self._send({"error": "not found"}, 404)

    def do_POST(self):
        if not self.path.startswith("/v1/decide"):
            return self._send({"error": "not found"}, 404)
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            cands = req.get("candidates") or []
            if not cands:
                return self._send({"error": "candidates required"}, 400)
            self._send(self.decider.decide(req.get("state", {}), cands, bool(req.get("ask_wake"))))
        except Exception as e:
            self._send({"error": f"{type(e).__name__}: {e}"}, 500)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--port", type=int, default=8780)
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()

    Handler.decider = Decider(args.model, args.threads)
    Handler.decider.warm()
    print(f"ready  ->  http://127.0.0.1:{args.port}/v1/decide", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
