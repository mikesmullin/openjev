#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${LAYA_MODEL_DIR:-$HOME/Library/Application Support/FluidUse/Models/laya-coreml}"
LAYA_PORT="${LAYA_PORT:-8787}"
WEB_PORT="${PORT:-3000}"

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "Model directory does not exist: $MODEL_DIR" >&2
  echo "Set LAYA_MODEL_DIR to a directory containing tokenizer.json and the L128 model bundle." >&2
  exit 1
fi

cd "$ROOT_DIR"
swift build -c release --product LayaServer
LAYA_BIN="$(swift build -c release --show-bin-path)/LayaServer"

cleanup() {
  if [[ -n "${LAYA_PID:-}" ]]; then
    kill "$LAYA_PID" 2>/dev/null || true
    wait "$LAYA_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

LAYA_MODEL_DIR="$MODEL_DIR" LAYA_PORT="$LAYA_PORT" "$LAYA_BIN" &
LAYA_PID=$!

for _ in {1..120}; do
  if curl --silent --fail "http://127.0.0.1:${LAYA_PORT}/healthz" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$LAYA_PID" 2>/dev/null; then
    echo "LayaServer exited before becoming ready." >&2
    exit 1
  fi
  sleep 0.25
done

if ! curl --silent --fail "http://127.0.0.1:${LAYA_PORT}/healthz" >/dev/null 2>&1; then
  echo "LayaServer did not become ready on port ${LAYA_PORT}." >&2
  exit 1
fi

cd "$ROOT_DIR/web"
LAYA_URL="http://127.0.0.1:${LAYA_PORT}" PORT="$WEB_PORT" bun run start
