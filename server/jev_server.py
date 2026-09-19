#!/usr/bin/env python
"""simple-jev v1, served from a local llama.cpp `llama-server`.

The upstream reference server (`vendor/simple-jev/hf-server`) holds the weights itself, in
Transformers, and reads next-token logits straight out of the forward pass. That is not an option here:
the model is `qwen3.8-27b-nvfp4-mtp-q8attn`, an NVFP4 GGUF that only llama.cpp can load, and it is already
resident on the GPU under `~/inference.mjs`. So this process holds no weights at all. It is an adapter:

    vendor/simple-jev/common/   the v1 contract -- prompt text, label assignment, softmax, response shape
    this file                   the engine boundary -- chat template, tokenizer checks, logit retrieval

`common/` is imported from the submodule, never copied. Everything that decides an answer lives there;
everything here is about talking to llama.cpp. That is exactly the seam the upstream project documents:
"Shared Python rules in `common/` so other inference implementations can use the same rules."

    POST /v1/classifier   {"model","state"|"messages","questions":{...}}  -> v1 response + `timing`
    GET  /v1/models       the alias llama-server reports
    GET  /health          {"ok":true,...}

Run:
    .venv/bin/python server/jev_server.py --llama http://127.0.0.1:1234 --port 8750

## Getting label logits out of llama.cpp

The v1 contract needs the raw next-token logits for a handful of permitted labels, at a position where
the assistant turn is deliberately unfinished (`{"answer": "`). Three llama-server endpoints get us there:

  /apply-template   renders messages with the model's own jinja template (`--jinja`) and an open
                    assistant turn. `enable_thinking: false` matters -- Qwen3.8 otherwise opens a
                    <think> block and the prefill lands inside it.
  /tokenize         proves each label is ONE token at the real rendered boundary (v1 section 9).
  /completion       `n_probs` returns `top_logprobs`, the raw log-softmax over the vocabulary.

Two approaches that look right and are not:

  * A GBNF grammar restricting output to the labels, with `post_sampling_probs`. llama.cpp computes
    those probabilities before the grammar sampler runs, so unconstrained tokens keep their mass and
    nothing is renormalised over the labels. Measured, not assumed.
  * `logit_bias` to force the labels into the top-N. The returned `logprob` values are the raw ones;
    bias never reaches them, so it changes neither the values nor which tokens are listed.

What does work is `n_probs` itself, with escalation. `top_logprobs` is the top-N of the true vocabulary
log-softmax, so a label that appears carries its exact value. log-softmax differs from the raw logit by
log Z, a single constant shared by every token at that position -- and v1 softmaxes over the permitted
labels alone, where any shared constant cancels. So the numbers below are the v1 logits, exactly.

Escalation covers the rest: ask for a small N, and if a permitted label did not make the cut, ask again
with a larger one. The `{"answer": "` prefill makes label tokens overwhelmingly likely, so N=64 answers
in practice; 4096 costs about the same on a warm prefix and is the backstop.
"""

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# The submodule is the source of truth for v1. Import it in place rather than vendoring a copy, so
# `git submodule update --remote` is the whole upgrade path and there is no second copy to drift.
VENDOR = Path(__file__).resolve().parent.parent / "vendor" / "simple-jev"
if not (VENDOR / "common" / "__init__.py").exists():
    sys.exit(f"missing submodule at {VENDOR}\n  git submodule update --init --recursive")
sys.path.insert(0, str(VENDOR))

from common import ClassifierRequest, build_response, prepare_prompt  # noqa: E402
from common.prompt_builder import canonical  # noqa: E402

# n_probs ladder. The first rung answers essentially every real request; the rest exist so a permitted
# label can never be silently dropped just because the model found it unlikely.
N_PROBS_LADDER = (64, 512, 4096)


class LlamaError(RuntimeError):
    """A llama-server call failed, as opposed to a bad classifier request."""


class Llama:
    """The llama-server HTTP surface this adapter needs, plus the label-boundary cache."""

    def __init__(self, base, timeout=120):
        self.base = base.rstrip("/")
        self.timeout = timeout
        # v1 section 9 wants every label checked as a single token at the *rendered* boundary, and that
        # check costs one /tokenize round trip per label. Re-running ~30 of them every tick would eat the
        # agent's budget, so cache on what actually determines the boundary: the tail of the rendered
        # text the label follows, and the label itself. The tail is the prefill, identical every tick, so
        # the first decision pays for the checks and the rest are free. Keyed on real text, not on an
        # assumption that the prefill alone fixes the tokenization.
        self._boundary = {}
        self._lock = threading.Lock()

    def _post(self, path, body):
        req = urllib.request.Request(
            self.base + path, json.dumps(body).encode(), {"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise LlamaError(f"{path} -> {e.code}: {e.read()[:200].decode('utf8', 'replace')}") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise LlamaError(f"{path} -> {type(e).__name__}: {e}") from e

    def render(self, messages):
        """Messages -> the model's own prompt string, ending at an open assistant turn."""
        return self._post(
            "/apply-template",
            {"messages": messages, "chat_template_kwargs": {"enable_thinking": False}},
        )["prompt"]

    def tokenize(self, text):
        return self._post("/tokenize", {"content": text})["tokens"]

    def label_token(self, text, label):
        """Token id for `label` at the end of `text`, or None if it is not single-token there.

        Checking the label in isolation is not enough: BPE merges across the boundary, so the only
        meaningful question is whether appending it to this exact text adds exactly one token.
        """
        key = (text[-24:], label)
        with self._lock:
            if key in self._boundary:
                return self._boundary[key]
        base = self.tokenize(text)
        extended = self.tokenize(text + label)
        ok = len(extended) == len(base) + 1 and extended[:-1] == base
        tid = extended[-1] if ok else None
        with self._lock:
            self._boundary[key] = tid
        return tid

    def label_logprobs(self, prompt, wanted):
        """Raw next-token log-softmax for `wanted` token ids, escalating n_probs until all are covered.

        Returns (logprobs_by_id, n_probs_used, escalations, prompt_tokens, cache_tokens).
        """
        want = set(wanted)
        escalations = 0
        for rung, n_probs in enumerate(N_PROBS_LADDER):
            r = self._post(
                "/completion",
                {
                    "prompt": prompt,
                    "n_predict": 1,
                    "n_probs": n_probs,
                    "temperature": 0,
                    # Raw vocabulary log-softmax, not the post-sampler distribution.
                    "post_sampling_probs": False,
                    # The shared prefix across a request's questions is the whole point of simple-jev:
                    # llama-server keeps the previous prompt in the slot and re-uses the longest matching
                    # token prefix, so only the selected-question tail is actually evaluated.
                    "cache_prompt": True,
                },
            )
            probs = r.get("completion_probabilities") or []
            if not probs:
                raise LlamaError("completion returned no token probabilities")
            found = {e["id"]: e["logprob"] for e in probs[0]["top_logprobs"] if e["id"] in want}
            timings = r.get("timings") or {}
            if len(found) == len(want) or rung == len(N_PROBS_LADDER) - 1:
                return (
                    found,
                    n_probs,
                    escalations,
                    int(timings.get("prompt_n", 0) or 0),
                    int(r.get("tokens_cached", 0) or 0),
                )
            escalations += 1
        raise AssertionError("unreachable")

    def health(self):
        with urllib.request.urlopen(self.base + "/health", timeout=10) as r:
            return json.loads(r.read())

    def model_alias(self):
        with urllib.request.urlopen(self.base + "/v1/models", timeout=10) as r:
            data = json.loads(r.read()).get("data") or []
        return data[0]["id"] if data else "unknown"


def build_messages(request, plan, question):
    """v1 section 6: the message sequence for one selected question.

    Identical to the reference adapter's assembly. Only the selected-question tail differs between a
    request's questions; system and context are shared, which is what makes the prefix cacheable.
    """
    system = plan.system_prompt_prefix + plan.prefix_instruction
    content = plan.suffix_instruction + question.instruction
    if request.messages is None:
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": f"State:\n{canonical(request.state)}\n\n" + content},
        ]
    messages = [m.model_dump(exclude_none=True) for m in request.messages]
    if messages[0]["role"] == "system":
        messages[0]["content"] = system + "\n" + messages[0]["content"]
    else:
        messages.insert(0, {"role": "system", "content": system})
    messages.append({"role": "user", "content": content})
    return messages


class Classifier:
    """Score every question in one request against the shared context."""

    def __init__(self, llama, model_alias, max_tokens=32768):
        self.llama = llama
        self.model_alias = model_alias
        self.max_tokens = max_tokens
        # llama-server is configured with a single slot (-np 1), so concurrent callers would evict each
        # other's prefix from the KV cache and each pay full prompt evaluation. Serialize instead: the
        # questions of one request then run back to back against a warm prefix.
        self.lock = threading.Lock()

    def classify(self, body):
        request = ClassifierRequest.model_validate(body)
        if request.tools or request.mm_processor_kwargs or request.media_io_kwargs:
            raise ValueError("this adapter supports text classification only")
        if request.messages and any(
            not isinstance(m.content, str) or m.model_extra or m.role in {"tool", "function"}
            for m in request.messages
        ):
            raise ValueError("this adapter accepts plain text chat only")

        plan = prepare_prompt(request)
        logits, token_ids, per_question = {}, {}, []
        input_tokens = 0
        t_all = time.perf_counter()

        with self.lock:
            for question in plan.questions:
                t0 = time.perf_counter()
                messages = build_messages(request, plan, question)
                # The template supplies the chat markers and the open assistant turn; the prefill is
                # appended to the rendered string afterwards, never as a completed assistant message.
                prompt = self.llama.render(messages) + question.answer_prefix

                ids = {}
                for label in question.output_labels:
                    tid = self.llama.label_token(prompt, label)
                    if tid is None:
                        raise ValueError(f"answer label {label!r} is not single-token stable")
                    ids[label] = tid
                if len(set(ids.values())) != len(ids):
                    raise ValueError("output labels must map to distinct token ids")

                found, n_probs, escalations, prompt_n, cached = self.llama.label_logprobs(
                    prompt, ids.values()
                )
                # A label still missing after the last rung sits below the 4096th most likely token.
                # Floor it just under everything observed rather than failing the whole decision: the
                # subsequent softmax makes such a label negligible, which is the truth about it.
                floor = min(found.values()) - 10.0 if found else -30.0
                row = {label: found.get(tid, floor) for label, tid in ids.items()}

                logits[question.branch_id] = row
                token_ids[question.branch_id] = ids
                input_tokens += prompt_n
                per_question.append(
                    {
                        "question": question.question_id,
                        "ms": round((time.perf_counter() - t0) * 1000, 2),
                        "labels": len(ids),
                        "prompt_tokens": prompt_n,
                        "cached_tokens": cached,
                        "n_probs": n_probs,
                        "escalations": escalations,
                        "missing_labels": len(ids) - len(found),
                    }
                )

        response = build_response(plan, logits, input_tokens=input_tokens)
        # The metric the harness plots. `questions` is the count answered in this one call -- the whole
        # point of the shared-prefix design, and the number that makes RTT-per-decision comparable.
        response["timing"] = {
            "total_ms": round((time.perf_counter() - t_all) * 1000, 2),
            "questions": len(per_question),
            "labels": sum(q["labels"] for q in per_question),
            "cached_tokens": max((q["cached_tokens"] for q in per_question), default=0),
            "escalations": sum(q["escalations"] for q in per_question),
            "per_question": per_question,
        }
        return response


class Handler(BaseHTTPRequestHandler):
    service = None
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
        s = self.service
        if self.path.startswith("/health"):
            try:
                return self._send(
                    {"ok": True, "backend": "llama.cpp", "llama": s.llama.base,
                     "model": s.model_alias, "upstream": s.llama.health()}
                )
            except Exception as e:
                return self._send({"ok": False, "error": f"{type(e).__name__}: {e}"}, 503)
        if self.path.startswith("/v1/models"):
            return self._send(
                {"object": "list", "data": [{"id": s.model_alias, "object": "model",
                                             "owned_by": "llama.cpp"}]}
            )
        self._send({"error": "not found"}, 404)

    def do_POST(self):
        if not self.path.startswith("/v1/classifier"):
            return self._send({"error": "not found"}, 404)
        try:
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}"
            self._send(self.service.classify(json.loads(raw)))
        except LlamaError as e:
            self._send({"error": str(e)}, 502)
        except Exception as e:
            # A bad question set must not take the server down mid-game.
            self._send({"error": f"{type(e).__name__}: {e}"}, 400)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama", default="http://127.0.0.1:1234", help="llama-server base URL")
    ap.add_argument("--port", type=int, default=8750)
    ap.add_argument("--max-tokens", type=int, default=32768)
    args = ap.parse_args()

    llama = Llama(args.llama)
    try:
        alias = llama.model_alias()
    except Exception as e:
        sys.exit(
            f"cannot reach llama-server at {args.llama}: {e}\n"
            f"  start it with:  ~/inference.mjs qwen3.8-27b-nvfp4-mtp-q8attn"
        )
    Handler.service = Classifier(llama, alias, max_tokens=args.max_tokens)
    print(f"simple-jev v1  ->  {alias}  via {args.llama}", flush=True)
    print(f"ready  ->  http://127.0.0.1:{args.port}/v1/classifier", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
