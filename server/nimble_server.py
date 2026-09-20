#!/usr/bin/env python
"""Bespoke Nimble 9B, serving the simple-jev v1 classifier contract.

This process holds the weights. That is the whole difference from the `simplejev-mars` branch, where the
model lived behind llama-server's HTTP surface and every question cost its own prefill and its own ~100 ms
request floor. Nimble is a Transformers checkpoint we load ourselves, so the questions in one decision can
share a forward pass -- which is what the `--mode` flag below is about.

    POST /v1/classifier   {"model","state","questions":{...}}  -> v1 answers + `timing`
    GET  /health          {"ok":true,"backend":"nimble","model":...,"mode":...}
    GET  /v1/models       the merged checkpoint this process loaded

Run:
    .venv/bin/python server/nimble_server.py --port 8750

## Why the wire contract is still simple-jev v1

Nothing here needs it to be. It is v1 because `simplejev-mars` is v1, and keeping the shape identical is
what makes the two branches comparable: the same `web/app.js`, the same questions, the same state, the
same ballot. Swapping this branch's model for that one's is a `MODEL_URL` change and nothing else. The
translation from v1's vocabulary to Nimble's happens here, in `to_schema()`:

    v1 `choice`  {criteria: {label: why}}   -> enum field, choices = labels, choice_descriptions = why
    v1 `choice`  over exactly yes/no        -> boolean field  (Nimble's own second type; mapped back)
    v1 `score`   {criteria: [rubric, ...]}  -> enum over ordered levels, answered as an expected value

The `score` mapping is the one upstream explicitly sanctions: "If a field is an ordered rating scale,
your application can use the probabilities to calculate an expected level." Nimble returns a probability
per level, so the rubric index expectation sum(i * p_i) is continuous in exactly the way v1's `score`
promises -- 2.4 is a real reading, not a rounded 2.

v1's third type, `noul`, is not mapped. It encodes a probability as one of the digits 1-9, and the
`simplejev-mars` branch already measured that failing on a Qwen: the model wants to write `0.9`, so it
reaches for `0`, every permitted label sits 8+ nats down and the softmax runs on noise. Nimble's boolean
is the type that question actually wants, and the harness asks yes/no questions that way.

## The prompt budget is 2048, not 8192

Upstream's serving path now permits 8192 tokens (`NIMBLE_MAX_PROMPT_TOKENS`), but `schema_config.json`
records `max_length: 2048` and that is what the adapter was trained at. Prompts past it are outside the
training distribution rather than rejected, which is the worse failure -- it answers, just less well. The
budget here comes from the contract, and going over is an error with the token count in it, so the page
can shrink the ballot instead of silently degrading.

## Nimble renders the schema once, and that changes where the time goes

v1 puts the selected question *after* the context and renders its options twice, so each question carries
a large tail of its own and prompts share only their head. Nimble's prompt is the opposite shape: the
whole schema -- every field, every choice, every description -- is rendered once inside the user turn, and
the only thing that varies per field is the trailing `Requested field: "name"`. So the per-field suffix is
a handful of tokens against a prefix of a thousand, and `prepare_prompts()` hands back exactly that split
(`prefix_ids`, `suffix_ids`) already computed.

`CudaCandidateScorer` does not use it. It loops the fields, runs each one's `full_ids` with
`use_cache=False`, and pays for the shared prefix once per field -- upstream says so plainly: "The CUDA
scorer scores each field on its own, with the full prompt each time." Only the Mac/MLX `ParallelScorer`
prefills the context once. On CUDA that leaves the same N-prefills shape that cost `simplejev-mars` its
latency, in a process that -- unlike llama-server over HTTP -- is perfectly able to avoid it.

So this file adds two modes and measures them against upstream's:

    independent  upstream's CudaCandidateScorer, unchanged. The reference for correctness.
    batched      one padded batch of all N full prompts. One forward pass; still N prefixes of compute.
    prefix       prefill the shared prefix once with a KV cache, then one batched step over the
                 per-field suffixes against it. One prefix of compute, N answers.

`--verify` checks the fast modes against `independent` on a real payload and reports the largest logit
disagreement, because a batched attention mask or an off-by-one position id is exactly the kind of bug
that produces plausible numbers rather than obvious ones.
"""

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "vendor" / "nimble"
if not (VENDOR / "nimble" / "scoring" / "parallel_schema.py").exists():
    sys.exit(f"missing submodule at {VENDOR}\n  git submodule update --init --recursive")
sys.path.insert(0, str(VENDOR))

import torch  # noqa: E402
from nimble.scoring.parallel_schema import choice_key, prepare_prompts  # noqa: E402
from nimble.scoring.cuda_scorer import CudaCandidateScorer, candidate_projection  # noqa: E402

# Ordered level names for a v1 `score` rubric. Nimble needs string choices; these carry no meaning of
# their own -- the rubric text goes in choice_descriptions and the answer is the index expectation.
LEVELS = ["LEVEL_0", "LEVEL_1", "LEVEL_2", "LEVEL_3", "LEVEL_4", "LEVEL_5",
          "LEVEL_6", "LEVEL_7", "LEVEL_8", "LEVEL_9"]
YES_NO = ({"yes", "no"}, {"true", "false"})


class BadRequest(ValueError):
    """A malformed classifier request, as opposed to a model failure."""


# ---------------------------------------------------------------------------- v1 <-> nimble schema

def to_schema(questions):
    """v1 `questions` -> a flat Nimble schema, plus how to read each field's answer back.

    Returns (schema, plan) where plan[name] is ('choice', labels) | ('bool', [no, yes]) | ('score', n).
    """
    if not isinstance(questions, dict) or not questions:
        raise BadRequest("questions must be a nonempty object")
    schema, plan = {}, {}
    for name, q in questions.items():
        if not isinstance(q, dict):
            raise BadRequest(f"{name}: question must be an object")
        kind, criteria = q.get("type"), q.get("criteria")
        instructions = (q.get("instructions") or name).strip()

        if kind == "score":
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= len(LEVELS):
                raise BadRequest(f"{name}: a score rubric must be 2-{len(LEVELS)} ordered strings")
            levels = LEVELS[:len(criteria)]
            schema[name] = {"type": "enum", "choices": levels, "description": instructions,
                            "choice_descriptions": dict(zip(levels, criteria))}
            plan[name] = ("score", len(criteria))
            continue

        if kind != "choice":
            raise BadRequest(f"{name}: supported types are choice and score (got {kind!r})")
        if not isinstance(criteria, dict) or not criteria:
            raise BadRequest(f"{name}: a choice needs a criteria object")
        labels = list(criteria)

        # A yes/no choice is a boolean question wearing a choice's clothes. Nimble has a boolean type and
        # was trained on it, so ask it that way rather than as a two-item enum, and map the answer back.
        if set(map(str.lower, labels)) in YES_NO and len(labels) == 2:
            truthy = next(l for l in labels if l.lower() in ("yes", "true"))
            falsy = next(l for l in labels if l is not truthy)
            schema[name] = {"type": "boolean", "description": instructions,
                            "choice_descriptions": {"true": criteria[truthy], "false": criteria[falsy]}}
            plan[name] = ("bool", [falsy, truthy])
            continue

        if not 2 <= len(labels) <= 26:
            raise BadRequest(f"{name}: a choice needs 2-26 candidates, got {len(labels)}"
                             " (Nimble's codes are single letters A-Z)")
        schema[name] = {"type": "enum", "choices": labels, "description": instructions,
                        "choice_descriptions": {l: str(criteria[l]) for l in labels}}
        plan[name] = ("choice", labels)
    return schema, plan


def to_answers(result, plan):
    """Nimble's per-field probabilities -> v1 `answers`."""
    answers = {}
    for name, (kind, meta) in plan.items():
        field = result["fields"][name]
        scores = field["scores"]                       # keyed by choice_key(value)
        if kind == "score":
            # The expected rubric index. Upstream sanctions exactly this for ordered scales, and it is
            # what makes the reading continuous: a model split between "worn down" and "about to be
            # destroyed" reads 2.5, which is the honest answer, not either level on its own.
            probs = [scores[LEVELS[i]] for i in range(meta)]
            answers[name] = {"score": sum(i * p for i, p in enumerate(probs)),
                             "probabilities": {str(i): p for i, p in enumerate(probs)}}
            continue
        if kind == "bool":
            falsy, truthy = meta
            p = {falsy: scores["false"], truthy: scores["true"]}
        else:
            p = {label: scores[choice_key(label)] for label in meta}
        pick = max(p, key=p.get)
        answers[name] = {"choice": pick, "confidence": p[pick], "probabilities": p}
    return answers


# ---------------------------------------------------------------------------- faster CUDA scoring

class Scorer(CudaCandidateScorer):
    """Upstream's CUDA scorer plus two modes that let the fields of one decision share a forward pass.

    Both extra modes reuse upstream's `prepare_prompts()` verbatim, so the prompt text, the candidate
    token ids and the label-boundary checks are all still upstream's. The only thing replaced is how the
    hidden state at the answer position is obtained.
    """

    @torch.inference_mode()
    def score(self, context, schema, mode="prefix"):
        if mode == "independent":
            return super().score(context, schema, mode="independent")
        if mode not in ("batched", "prefix"):
            raise ValueError("mode must be independent, batched or prefix")

        prepared = self.prepare(context, schema)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        hidden = (self._batched(prepared) if mode == "batched" else self._prefix(prepared))
        out = self._read_out(prepared, hidden, context, mode, started)
        torch.cuda.synchronize()
        return out

    def _pad(self, rows):
        """Right-pad token id rows into [N, L] plus a [N, L] mask and each row's true length."""
        lengths = [len(r) for r in rows]
        width = max(lengths)
        ids = torch.zeros(len(rows), width, dtype=torch.long, device=self.device)
        mask = torch.zeros(len(rows), width, dtype=torch.long, device=self.device)
        for i, row in enumerate(rows):
            ids[i, :len(row)] = torch.tensor(row, device=self.device)
            mask[i, :len(row)] = 1
        return ids, mask, lengths

    def _batched(self, prepared):
        """One padded batch over the full prompts. One forward pass, N prefixes of compute."""
        ids, mask, lengths = self._pad(prepared.full_ids)
        states = self.backbone(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        # Right padding, so each row's answer position is its own last real token, not the last column.
        index = torch.tensor([n - 1 for n in lengths], device=states.device)
        return states[torch.arange(len(lengths), device=states.device), index, :]

    def _prefix(self, prepared):
        """Prefill the shared prefix once, then one batched step over the per-field suffixes.

        This is the shape the prompt was built for: `prepare_prompts()` already returns the exact common
        token prefix and each field's suffix, and on this schema the suffix is only the field name. The
        prefix is prefilled with batch 1 and its KV cache is then widened to N rows, so the thousand-token
        context is evaluated once per decision instead of once per field.
        """
        prefix = torch.tensor([prepared.prefix_ids], device=self.device)
        cache = self.backbone(input_ids=prefix, use_cache=True).past_key_values
        n = len(prepared.suffix_ids)
        _widen(cache, n)

        ids, suffix_mask, lengths = self._pad(prepared.suffix_ids)
        p = len(prepared.prefix_ids)
        # The cache holds the prefix for every row, so the mask must cover prefix + suffix.
        mask = torch.cat([torch.ones(n, p, dtype=torch.long, device=self.device), suffix_mask], dim=1)
        # Positions continue from the prefix. Pads get positions too; the mask is what excludes them.
        positions = p + torch.arange(ids.shape[1], device=self.device).unsqueeze(0).expand(n, -1)
        states = self.backbone(input_ids=ids, attention_mask=mask, position_ids=positions,
                               past_key_values=cache, use_cache=False).last_hidden_state
        index = torch.tensor([n_ - 1 for n_ in lengths], device=states.device)
        return states[torch.arange(n, device=states.device), index, :]

    def _read_out(self, prepared, hidden, context, mode, started):
        """Candidate projection + softmax per field, in upstream's response shape."""
        fields, output = {}, {}
        for i, (name, choices, ids, candidates) in enumerate(zip(
                prepared.names, prepared.choices, prepared.full_ids, prepared.candidate_ids)):
            logits = candidate_projection(hidden[i:i + 1], self.head_weight, candidates)[0]
            if not torch.isfinite(logits).all():
                raise ValueError(f"{name}: model produced non-finite candidate logits")
            probabilities = torch.softmax(logits / self.temperature, dim=-1)
            keys = [choice_key(v) for v in choices]
            output[name] = choices[logits.argmax().item()]
            fields[name] = {"value": output[name],
                            "scores": dict(zip(keys, probabilities.tolist())),
                            "logits": dict(zip(keys, logits.tolist())),
                            "prompt_token_count": len(ids)}
        return {"model": self.model_id, "revision": self.revision, "backend": "cuda",
                "temperature": self.temperature, "runtime": self.runtime,
                "context": context, "output": output, "fields": fields,
                "metrics": {"mode": mode, "fields": len(fields),
                            "total_seconds": time.perf_counter() - started,
                            "prefix_tokens": len(prepared.prefix_ids),
                            "suffix_tokens": [len(s) for s in prepared.suffix_ids],
                            "cuda_peak_active_gib": torch.cuda.max_memory_allocated() / 2**30}}


def _widen(cache, n):
    """Repeat a batch-1 KV cache across n rows, in place.

    Transformers has moved this around between versions (legacy tuples, `key_cache`/`value_cache` lists,
    and now per-layer objects), and getting it wrong is silent -- the rows still run, they just attend to
    the wrong thing. So handle the shapes explicitly and refuse anything unrecognised.
    """
    layers = getattr(cache, "layers", None)
    if layers is not None:
        for layer in layers:
            for attr in ("keys", "values"):
                t = getattr(layer, attr, None)
                if t is not None:
                    setattr(layer, attr, t.expand(n, *t.shape[1:]).contiguous())
        return
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        for store in (cache.key_cache, cache.value_cache):
            for i, t in enumerate(store):
                store[i] = t.expand(n, *t.shape[1:]).contiguous()
        return
    raise RuntimeError(f"unrecognised KV cache type {type(cache).__name__}; cannot widen for prefix mode")


# ---------------------------------------------------------------------------- http

class Handler(BaseHTTPRequestHandler):
    scorer = None
    mode = "prefix"
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
            return self._send(200, {"ok": True, "backend": "nimble", "mode": self.mode,
                                    "model": self.scorer.model_id, "revision": self.scorer.revision,
                                    "max_input_tokens": self.scorer.max_input_tokens})
        if self.path == "/v1/models":
            return self._send(200, {"data": [{"id": self.scorer.model_id, "object": "model"}]})
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
            schema, plan = to_schema(questions)
            # v1 sends the situation as an object and treats it as data, never instructions; Nimble wants
            # one string. Canonical JSON keeps it unambiguous and keeps key order stable across ticks,
            # which is also what keeps the prompt prefix worth caching.
            context = json.dumps(state, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            # One model, one GPU: serialise. Concurrent decisions would only queue on the device anyway,
            # and interleaved CUDA peak-memory stats would make the timings meaningless.
            with self.lock:
                result = self.scorer.score(context, schema, mode=self.mode)
        except BadRequest as e:
            return self._send(400, {"error": str(e)})
        except ValueError as e:
            return self._send(400, {"error": str(e)})
        except Exception as e:  # noqa: BLE001 - the agent loop needs a JSON error, not a dropped socket
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})

        metrics = result["metrics"]
        prompt_tokens = max(f["prompt_token_count"] for f in result["fields"].values())
        self._send(200, {
            "model": result["model"],
            "answers": to_answers(result, plan),
            "usage": {"input_tokens": prompt_tokens},
            "timing": {
                "total_ms": round(1000 * (time.perf_counter() - t0), 1),
                "model_ms": round(1000 * metrics["total_seconds"], 1),
                "questions": len(plan),
                "labels": sum(len(f["scores"]) for f in result["fields"].values()),
                # What the shared prefix actually saved: tokens NOT re-evaluated versus one full prompt
                # per field. Zero in independent and batched mode, which is the point of reporting it.
                "cached_tokens": (metrics["prefix_tokens"] * (len(plan) - 1)
                                  if metrics["mode"] == "prefix" else 0),
                "mode": metrics["mode"],
            },
        })


# ---------------------------------------------------------------------------- verification

SAMPLE = {
    "state": {"mission": "Raid a Mars colony from a gunship.", "hull_percent": 46,
              "hull_lost_last_10s": 18, "altitude_metres": 120,
              "colony_buildings_standing": 5, "alien_saucers_airborne": 3,
              "evasion_working": False},
    "questions": {
        "target": {"type": "choice", "instructions": "What should the gunship attack right now?",
                   "criteria": {
                       "saucer:1": "Shoot down the alien saucer 49 metres away. The ship is at 46 percent"
                                   " hull and cannot finish the mission if it is destroyed first.",
                       "building:2": "Destroy the colony building 656 metres away. Flattening the colony"
                                     " is the mission and 5 still stand.",
                       "building:3": "Destroy the colony building 812 metres away. Flattening the colony"
                                     " is the mission and 5 still stand."}},
        "posture": {"type": "choice", "instructions": "Keep attacking, or break off and climb away?",
                    "criteria": {"press": "Keep attacking. The ship can survive the damage it is taking.",
                                 "break_off": "Break off and climb away, or the ship will be destroyed."}},
        "threat": {"type": "score", "instructions": "How much danger is the gunship in right now?",
                   "criteria": ["No danger: almost nothing is getting through, and the hull has plenty left.",
                                "Occasional hits, and the hull is healthy enough to absorb them.",
                                "Being worn down: losing hull steadily, or the hull is already low.",
                                "About to be destroyed: the hull is nearly gone and any fire finishes it."]},
        "wake": {"type": "choice", "instructions": "Wake the buried scorpion now?",
                 "criteria": {"yes": "Wake it: the colony is finished and it is the last enemy worth attacking.",
                              "no": "Leave it buried: there are still colony buildings to destroy."}},
    },
}


def verify(scorer, repeats=5):
    """Check `batched` and `prefix` against upstream's `independent`, then time all three."""
    schema, plan = to_schema(SAMPLE["questions"])
    context = json.dumps(SAMPLE["state"], sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    reference = scorer.score(context, schema, mode="independent")
    print(f"prompt: {max(f['prompt_token_count'] for f in reference['fields'].values())} tokens"
          f"  (budget {scorer.max_input_tokens})")
    print(f"prefix shared by all {len(plan)} fields: "
          f"{len(scorer.prepare(context, schema).prefix_ids)} tokens\n")

    ok = True
    for mode in ("batched", "prefix"):
        got = scorer.score(context, schema, mode=mode)
        worst, where = 0.0, ""
        for name in plan:
            for key, value in reference["fields"][name]["logits"].items():
                d = abs(value - got["fields"][name]["logits"][key])
                if d > worst:
                    worst, where = d, f"{name}.{key}"
        agree = got["output"] == reference["output"]
        # BF16 matmul is not associative, so a different batch shape moves the last bits. A disagreement
        # that changes an argmax is a bug; one in the third decimal of a logit is arithmetic.
        flag = "ok " if agree and worst < 0.05 else "BAD"
        ok &= agree and worst < 0.05
        print(f"  {flag} {mode:11s} max |dlogit| {worst:.4f} at {where:24s} argmax {'agrees' if agree else 'DIFFERS'}")
    print()

    for mode in ("independent", "batched", "prefix"):
        scorer.score(context, schema, mode=mode)          # warm
        times = []
        for _ in range(repeats):
            t = time.perf_counter()
            scorer.score(context, schema, mode=mode)
            times.append(1000 * (time.perf_counter() - t))
        times.sort()
        print(f"  {mode:11s} {len(plan)} questions: median {times[len(times)//2]:7.1f} ms"
              f"   min {times[0]:7.1f} ms   max {times[-1]:7.1f} ms")

    print("\noutput:", json.dumps(reference["output"]))
    print("answers:", json.dumps(to_answers(reference, plan), indent=2)[:700])
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / ".cache" / "nimble-model.json"),
                    help="written by scripts/prepare-model.py")
    ap.add_argument("--port", type=int, default=8750)
    ap.add_argument("--mode", default="prefix", choices=("independent", "batched", "prefix"))
    ap.add_argument("--verify", action="store_true", help="check the fast modes against upstream's, then exit")
    args = ap.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        sys.exit(f"missing {config_path}\n  .venv/bin/python scripts/prepare-model.py")
    config = json.loads(config_path.read_text())
    print(f"loading {config['model_id']} from {config['model_path']}", flush=True)
    t0 = time.perf_counter()
    scorer = Scorer(**config)
    print(f"loaded in {time.perf_counter() - t0:.1f}s  {scorer.runtime['gpus']}  "
          f"budget {scorer.max_input_tokens} tokens", flush=True)

    if args.verify:
        sys.exit(0 if verify(scorer) else 1)

    Handler.scorer, Handler.mode = scorer, args.mode
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"\n  nimble classifier on http://127.0.0.1:{args.port}/v1/classifier  (mode: {args.mode})\n",
          flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
