#!/usr/bin/env bash
# Fetch the openJev-verdict-2.0 weights into .cache/verdict/.
#
#   bun run model:prepare      # or: bash scripts/download-verdict.sh
#
# This calls the submodule's OWN downloader rather than reimplementing it, because that script verifies
# every file against the SHA-256 and byte count recorded in artifacts/ARTIFACTS.json and exits non-zero
# on a mismatch. Provenance is the point: see the README on where these weights actually live.
#
# --slim skips the two ONNX exports (~900 MB) that the PyTorch path does not need.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR="$ROOT/vendor/verdict"
OUT="$ROOT/.cache/verdict"
PY="$ROOT/.venv/bin/python"

[ -f "$VENDOR/scripts/download_artifacts.py" ] || {
  echo "missing submodule at $VENDOR" >&2
  echo "  git submodule update --init --recursive" >&2
  exit 1
}

MANIFEST="$VENDOR/artifacts/ARTIFACTS.json"
if [ "${1:-}" = "--slim" ]; then
  MANIFEST="$(mktemp -t verdict-manifest-XXXXXX.json)"
  trap 'rm -f "$MANIFEST"' EXIT
  "$PY" - "$VENDOR/artifacts/ARTIFACTS.json" "$MANIFEST" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
m["files"] = {k: v for k, v in m["files"].items() if not k.endswith(".onnx")}
json.dump(m, open(sys.argv[2], "w"))
print("slim: skipping the ONNX exports")
PY
fi

mkdir -p "$OUT"
cd "$VENDOR"
exec "$PY" scripts/download_artifacts.py --manifest "$MANIFEST" --output_dir "$OUT"
