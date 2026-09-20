"""Serve a small browser UI for the local Laya Snake demo."""

import argparse
import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .game import SnakeGame
from .policy import LayaPolicy

WEB_PAGE = Path(__file__).resolve().parents[2] / "web" / "index.html"


class SnakeWebApp:
    """Own the model and one serialized Snake game session."""

    def __init__(
        self,
        model=None,
        *,
        width=24,
        height=16,
        seed=7,
        initial_length=6,
        prompt="compact",
        optimize=False,
        guarded=True,
    ):
        self.width = width
        self.height = height
        self.base_seed = seed
        self.initial_length = initial_length
        self.round = 1
        self.lock = threading.Lock()
        print("Loading local FP16 weights; browser demo uses no network requests...", flush=True)
        self.policy = LayaPolicy(
            model,
            guarded=guarded,
            prompt=prompt,
            optimize=optimize,
        )
        # Pay the first-use compilation/kernel cost before the browser starts animating.
        warm = SnakeGame(width, height, seed + 10000, initial_length)
        for _ in range(6):
            decision = self.policy.decide(warm)
            warm.step(decision.executed)
            if not warm.alive:
                break
        self.timestamps = deque(maxlen=60)
        self.interventions = 0
        self.best_score = 0
        self.steps = 0
        self.started = time.perf_counter()
        self.game = SnakeGame(width, height, seed, initial_length)

    def _stats(self):
        now = time.perf_counter()
        rate = 0.0
        if len(self.timestamps) > 1:
            rate = (len(self.timestamps) - 1) / (self.timestamps[-1] - self.timestamps[0])
        return {
            "round": self.round,
            "steps": self.steps,
            "interventions": self.interventions,
            "best_score": self.best_score,
            "decisions_per_second": rate,
            "elapsed_seconds": now - self.started,
            "guarded": self.policy.guarded,
            "network": "offline",
        }

    def status(self):
        with self.lock:
            return {
                "game": self.game.snapshot(),
                "stats": self._stats(),
                "model": self.policy.metadata,
            }

    def reset(self, seed=None):
        with self.lock:
            self.round += 1
            actual_seed = self.base_seed + self.round - 1 if seed is None else int(seed)
            self.game = SnakeGame(self.width, self.height, actual_seed, self.initial_length)
            self.timestamps.clear()
            self.interventions = 0
            self.steps = 0
            self.best_score = 0
            self.started = time.perf_counter()
            return self.status_unlocked()

    def status_unlocked(self):
        return {
            "game": self.game.snapshot(),
            "stats": self._stats(),
            "model": self.policy.metadata,
        }

    def step(self):
        with self.lock:
            if not self.game.alive or self.game.won:
                return {
                    "finished": True,
                    "before": self.game.snapshot(),
                    "after": self.game.snapshot(),
                    "decision": None,
                    "stats": self._stats(),
                    "model": self.policy.metadata,
                }
            before = self.game.snapshot()
            decision = self.policy.decide(self.game)
            self.timestamps.append(time.perf_counter())
            self.interventions += int(decision.intervened)
            self.steps += 1
            self.best_score = max(self.best_score, self.game.score)
            self.game.step(decision.executed)
            after = self.game.snapshot()
            self.best_score = max(self.best_score, after["score"])
            return {
                "finished": False,
                "before": before,
                "after": after,
                "ate": after["score"] > before["score"],
                "decision": decision.to_dict(),
                "stats": self._stats(),
                "model": self.policy.metadata,
            }


class SnakeRequestHandler(BaseHTTPRequestHandler):
    server_version = "LayaSnakeWeb/1.0"

    @property
    def app(self):
        return self.server.app

    def _send_bytes(self, payload, content_type, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, value, status=200):
        payload = json.dumps(value, separators=(",", ":")).encode("utf-8")
        self._send_bytes(payload, "application/json; charset=utf-8", status)

    def do_GET(self):  # noqa: N802 - http.server API
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            try:
                payload = WEB_PAGE.read_bytes()
            except FileNotFoundError:
                self._send_json({"error": f"Browser UI not found: {WEB_PAGE}"}, 500)
                return
            self._send_bytes(payload, "text/html; charset=utf-8")
        elif path in ("/api/status", "/health"):
            self._send_json(self.app.status())
        else:
            self._send_json({"error": "Not found"}, 404)

    def do_POST(self):  # noqa: N802 - http.server API
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", "0"))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except (json.JSONDecodeError, ValueError):
            self._send_json({"error": "Request body must be JSON"}, 400)
            return
        try:
            if path == "/api/step":
                self._send_json(self.app.step())
            elif path == "/api/reset":
                self._send_json(self.app.reset(body.get("seed")))
            else:
                self._send_json({"error": "Not found"}, 404)
        except Exception as error:  # Return useful errors to the browser during local setup.
            self._send_json({"error": f"{type(error).__name__}: {error}"}, 500)

    def log_message(self, format, *args):
        # Keep the terminal useful for startup/errors without logging every animation request.
        if self.path not in ("/api/step", "/api/status"):
            super().log_message(format, *args)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Local model directory or an already cached Hub ID")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--initial-length", type=int, default=6)
    parser.add_argument("--prompt", choices=("compact", "detailed"), default="compact")
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--unassisted", action="store_true")
    args = parser.parse_args(argv)
    try:
        app = SnakeWebApp(
            args.model,
            width=args.width,
            height=args.height,
            seed=args.seed,
            initial_length=args.initial_length,
            prompt=args.prompt,
            optimize=args.optimize,
            guarded=not args.unassisted,
        )
        server = ThreadingHTTPServer((args.host, args.port), SnakeRequestHandler)
        server.app = app
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Laya Snake browser demo: http://{args.host}:{args.port}", flush=True)
    print("The model is local/offline; leave this process running while the browser is open.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Laya Snake browser demo.", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
