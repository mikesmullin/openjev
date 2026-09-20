#!/usr/bin/env python
"""openJev-verdict-2.0 serving the simple-jev v1 classifier contract.

A ~151M encoder that scores a whole ballot in one non-autoregressive forward pass. The options are not
scored one at a time and there is no generation: every candidate description is laid into the same
sequence as a label, and the model emits one logit per label.

    <<LABEL>>desc0<<LABEL>>desc1...<<SEP>>Question: {instructions}\\n\\nContext:\\n{state}

Every question in a decision goes through as one padded batch, so `forward_call_count` is 1 no matter
how many questions were asked. The `nimble-mars` branch had to rebuild upstream's CUDA scorer to get
four questions into one pass; here that is just what the architecture does.

    POST /v1/classifier   {"model","state","questions":{...}}  -> v1 answers + `timing`
    GET  /health          {"ok":true,"backend":"verdict",...}
    GET  /v1/models

Run:
    .venv/bin/python server/verdict_server.py --port 8750

## Which model this actually is

Read this before trusting any number in the upstream README against this branch.

The project contains two different models, and only one of them can be downloaded:

  core/engine_encoder.py + heman10x/rlcd-modernbert-151m
      A GLiClass fine-tune of `knowledgator/gliclass-modern-base-v2.0` (~151M, ModernBERT-based). The
      weights are on Hugging Face, checksummed in artifacts/ARTIFACTS.json, and this is what runs here.

  verdict2/model.py + artifacts/verdict2-base/model.pt
      The marker-pointer architecture with the dual-channel correctness head -- the thing the README's
      headline numbers describe (77.10% top-1, 1.44% correctness ECE, 0.7664 AUROC). Its checkpoint is
      a git-LFS pointer whose object is NOT on GitHub's LFS server: the batch API answers
      `404 Object does not exist`. The advertised HF repo `heman10x/openJev-verdict-2.0` holds only a
      config, a tokenizer and two PNGs -- no weights at all.

So the second channel is not available, and this branch does not use or claim it. What is served is the
shipped bundle, with the shipped scalar temperature.

## Calibration scope is narrower than it looks

`calibrator.json` records `"scope": "restricted_5_candidate_selection"` and a single temperature of
1.4265. Upstream's own engine only reports `calibration_status: "calibrated_for_scope"` when a query has
exactly five candidates, and `"unvalidated_scope"` otherwise -- the temperature was fit for five-way
selection and is applied to everything else unvalidated. The ballot here is rarely exactly five, so most
answers come back `unvalidated_scope`. That status is passed through to the page rather than hidden,
because a calibrated-looking probability from outside its fitted scope is exactly the thing that should
not be quietly trusted.

## What v1 maps onto

Verdict's own primitives are Choice / Score / Noul, which is v1's vocabulary exactly -- this is the first
model through this harness that needs no invented mapping.

    v1 `choice` {criteria: {id: why}}   -> Choice(options=[Option(id, description=why)])
    v1 `score`  {criteria: [rubric]}    -> Score(levels=[Level(id=str(i), description, value=i)])
    v1 `noul`   {instructions}          -> Noul(proposition=instructions)

`noul` runs for the first time in this repo. `simplejev-mars` had to abandon it because a Qwen asked for
a bare digit reached for `0` and every permitted label sat 8+ nats down. That was a property of making a
generative model emit a digit; here noul is just a two-outcome query scored like any other.

## Abstention is a real outcome, not a candidate

Every query gets `__insufficient_evidence__` appended by upstream's formatter, and it is *reserved* --
`Option(id="__insufficient_evidence__")` raises. So "none of these" is a first-class answer the model
can return rather than a hand-written option competing on how agreeable its prose is. That matters here:
`simplejev-mars` documented `other:hold` ("Hold fire and attack nothing right now") winning at 0.98 on a
9B and flying a whole game with `shots: 0`, precisely because it read well. This branch removes
`other:hold` from the ballot entirely and uses abstention instead; see web/app.js.

The v1 response carries it as `abstained` plus `p_abstain`, alongside the substantive distribution
renormalised over the real options, so the page can rank the ballot and still see the escape hatch.
"""

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "vendor" / "verdict"
if not (VENDOR / "core" / "engine_encoder.py").exists():
    sys.exit(f"missing submodule at {VENDOR}\n  git submodule update --init --recursive")
sys.path.insert(0, str(VENDOR))

from core.engine_encoder import DecisionEngine  # noqa: E402
from core.formatting import MAX_SUBSTANTIVE_CANDIDATES  # noqa: E402
from core.primitives import (  # noqa: E402
    INSUFFICIENT_EVIDENCE_ID,
    Choice,
    Level,
    Noul,
    Option,
    Score,
)


class BadRequest(ValueError):
    """A malformed classifier request, as opposed to a model failure."""


def to_queries(questions):
    """v1 `questions` -> upstream's typed Query objects, in request order."""
    if not isinstance(questions, dict) or not questions:
        raise BadRequest("questions must be a nonempty object")
    queries = []
    for name, q in questions.items():
        if not isinstance(q, dict):
            raise BadRequest(f"{name}: question must be an object")
        kind, criteria = q.get("type"), q.get("criteria")
        instructions = str(q.get("instructions") or "").strip()
        if not instructions:
            raise BadRequest(f"{name}: instructions are required")
        try:
            if kind == "choice":
                if not isinstance(criteria, dict) or len(criteria) < 2:
                    raise BadRequest(f"{name}: a choice needs a criteria object with 2 or more candidates")
                if len(criteria) > MAX_SUBSTANTIVE_CANDIDATES:
                    raise BadRequest(f"{name}: {len(criteria)} candidates exceeds the model's "
                                     f"{MAX_SUBSTANTIVE_CANDIDATES} (abstention takes the last slot)")
                queries.append(Choice(id=name, question=instructions,
                                      options=[Option(id=k, description=str(v)) for k, v in criteria.items()]))
            elif kind == "score":
                if not isinstance(criteria, list) or len(criteria) < 2:
                    raise BadRequest(f"{name}: a score rubric must be a list of 2 or more ordered strings")
                # Level values are the rubric indices, which is what v1's `score` means and what keeps
                # the expected value on the same 0..N-1 scale the page already plots.
                queries.append(Score(id=name, question=instructions,
                                     levels=[Level(id=str(i), description=str(c), value=float(i))
                                             for i, c in enumerate(criteria)]))
            elif kind == "noul":
                queries.append(Noul(id=name, proposition=instructions,
                                    semantics="conditional_on_sufficient_evidence_v2"))
            else:
                raise BadRequest(f"{name}: type must be choice, score or noul (got {kind!r})")
        except BadRequest:
            raise
        except Exception as e:  # pydantic validation -> a 400 with the real reason
            raise BadRequest(f"{name}: {e}") from e
    return queries


def to_answers(batch):
    """Upstream's typed results -> v1 `answers`, with abstention surfaced separately."""
    answers = {}
    for r in batch.results:
        probs = dict(r.probabilities)
        p_abstain = float(probs.pop(INSUFFICIENT_EVIDENCE_ID, 0.0))
        # Renormalise over the real options so the ballot still sums to 1 and stays comparable with
        # the other branches; the abstention mass is reported on its own rather than folded in.
        total = sum(probs.values())
        substantive = {k: (v / total if total > 0 else 0.0) for k, v in probs.items()}
        answer = {"probabilities": substantive, "abstained": bool(r.is_abstention),
                  "p_abstain": p_abstain, "calibration": r.calibration_status}
        if r.kind == "score":
            # expected_score is None when the model abstained; hold the reading rather than invent one.
            answer["score"] = None if r.expected_score is None else float(r.expected_score)
        elif r.kind == "noul":
            answer["p_true"] = r.p_true_given_sufficient_evidence
            answer["choice"] = None if r.is_abstention else str(r.selected_outcome)
            answer["confidence"] = substantive.get(answer["choice"], 0.0) if answer["choice"] else 0.0
        else:
            answer["choice"] = None if r.is_abstention else r.selected_id
            # Taken from the renormalised distribution, not from r.selected_probability, so that
            # `confidence` is always the value sitting beside it in `probabilities`. The mass that
            # went to abstention is reported once, as p_abstain, rather than twice.
            answer["confidence"] = substantive.get(answer["choice"], 0.0) if answer["choice"] else 0.0
            answer["concentration"] = float(r.concentration)
        answers[r.id] = answer
    return answers


class Handler(BaseHTTPRequestHandler):
    engine = None
    lock = threading.Lock()

    def log_message(self, *a):
        pass

    def _send(self, code, body):
        payload = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"ok": True, "backend": "verdict",
                                    "model": self.engine.model_name_or_path,
                                    "device": self.engine.device,
                                    "calibrated": self.engine.calibrator is not None})
        if self.path == "/v1/models":
            return self._send(200, {"data": [{"id": "openJev-verdict-2.0", "object": "model"}]})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/classifier":
            return self._send(404, {"error": "not found"})
        t0 = time.perf_counter()
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        except json.JSONDecodeError as e:
            return self._send(400, {"error": f"invalid JSON: {e}"})
        try:
            state, questions = body.get("state"), body.get("questions")
            if state is None:
                raise BadRequest("state is required")
            queries = to_queries(questions)
            # The encoder reads the state as text. Pretty-printed JSON rather than the compact form:
            # this is a 151M encoder with a 1024-token window and no instruction tuning, and the
            # newline-per-field layout is closer to what it was trained to read than one long line.
            context = json.dumps(state, sort_keys=True, ensure_ascii=False, indent=1)
            with self.lock:
                batch = self.engine.evaluate(context, queries)
        except BadRequest as e:
            return self._send(400, {"error": str(e)})
        except Exception as e:  # noqa: BLE001 - the agent loop needs JSON, not a dropped socket
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})

        self._send(200, {
            "model": "openJev-verdict-2.0",
            "answers": to_answers(batch),
            "usage": {"input_tokens": 0},
            "timing": {"total_ms": round(1000 * (time.perf_counter() - t0), 1),
                       "model_ms": round(batch.total_latency_ms, 1),
                       "questions": len(queries),
                       "labels": sum(len(r.probabilities) for r in batch.results),
                       "forward_passes": batch.forward_call_count,
                       "cached_tokens": 0},
        })


SAMPLE = {
    "state": {"mission": "Raid a Mars colony from a gunship.", "hull_percent": 46,
              "hull_lost_last_10s": 18, "colony_buildings_standing": 5, "alien_saucers_airborne": 3,
              "evasion_working": False},
    "questions": {
        "target": {"type": "choice", "instructions": "What should the gunship attack right now?",
                   "criteria": {
                       "saucer:1": "Shoot down the alien saucer 49 metres away. The ship is at 46 percent"
                                   " hull and cannot finish the mission if it is destroyed first.",
                       "building:2": "Destroy the colony building 656 metres away. Flattening the colony"
                                     " is the mission and 5 still stand."}},
        "posture": {"type": "choice", "instructions": "Keep attacking, or break off and climb away?",
                    "criteria": {"press": "Keep attacking. The ship can survive the damage it is taking.",
                                 "break_off": "Break off and climb away, or the ship will be destroyed."}},
        "threat": {"type": "score", "instructions": "How much danger is the gunship in right now?",
                   "criteria": ["No danger: almost nothing is getting through, and the hull has plenty left.",
                                "Occasional hits, and the hull is healthy enough to absorb them.",
                                "Being worn down: losing hull steadily, or the hull is already low.",
                                "About to be destroyed: the hull is nearly gone and any fire finishes it."]},
        "wake": {"type": "noul",
                 "instructions": "the colony is finished and the buried scorpion should be woken now"},
    },
}


def selftest(engine, repeats=20):
    context = json.dumps(SAMPLE["state"], sort_keys=True, ensure_ascii=False, indent=1)
    queries = to_queries(SAMPLE["questions"])
    batch = engine.evaluate(context, queries)
    print(f"device {engine.device}   {len(queries)} questions in "
          f"{batch.forward_call_count} forward pass, {batch.execution_mode}\n")
    print(json.dumps(to_answers(batch), indent=2)[:1400])
    print()
    for n in (1, 2, 3, 4):
        qs = to_queries(dict(list(SAMPLE["questions"].items())[:n]))
        engine.evaluate(context, qs)
        ts = []
        for _ in range(repeats):
            t = time.perf_counter(); engine.evaluate(context, qs); ts.append(1000 * (time.perf_counter() - t))
        ts.sort()
        print(f"  {n} question(s): median {ts[len(ts)//2]:6.1f} ms   min {ts[0]:6.1f} ms")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / ".cache" / "verdict"),
                    help="written by scripts/download-verdict.sh")
    ap.add_argument("--port", type=int, default=8750)
    ap.add_argument("--device", default=None, help="cuda, cpu (default: cuda if available)")
    ap.add_argument("--no-calibrator", action="store_true",
                    help="serve raw logits instead of the shipped temperature")
    ap.add_argument("--selftest", action="store_true", help="one decision plus latency, then exit")
    args = ap.parse_args()

    weights = Path(args.weights)
    if not (weights / "model.safetensors").exists():
        sys.exit(f"missing weights in {weights}\n  bun run model:prepare")
    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    calibrator = None
    if not args.no_calibrator:
        from core.calibration import TemperatureCalibrator
        # Upstream's own loader, so model_id / scope / artifact_hash come across intact -- the engine
        # reads .scope to decide whether an answer counts as calibrated at all.
        calibrator = TemperatureCalibrator.load(weights / "calibrator.json")
        print(f"calibrator: T={calibrator.temperature:.4f}  scope={calibrator.scope!r}"
              f"  model_id={calibrator.model_id!r}")

    print(f"loading openJev-verdict-2.0 from {weights} on {device}", flush=True)
    t0 = time.perf_counter()
    engine = DecisionEngine(model_name_or_path=str(weights), calibrator=calibrator, device=device)
    params = sum(p.numel() for p in engine.model.parameters())
    print(f"loaded in {time.perf_counter() - t0:.1f}s  {params/1e6:.1f}M params", flush=True)

    if args.selftest:
        sys.exit(0 if selftest(engine) else 1)

    Handler.engine = engine
    print(f"\n  verdict classifier on http://127.0.0.1:{args.port}/v1/classifier\n", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
