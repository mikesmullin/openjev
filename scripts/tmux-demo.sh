#!/usr/bin/env bash
# Show openjev playing tetris in tmux: the live board on the left, the model's reasoning on the right.
#
# Reuses an existing session rather than creating one (pass TMUX_SESSION to pick a specific one).
#
#   ./scripts/tmux-demo.sh [pieces]
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"

PIECES="${1:-60}"
TETRIS_BIN="${TETRIS_BIN:-/workspace/g4a/tetris/tetris}"
STATE="${TETRIS_STATE_FILE:-/tmp/openjev-tetris.dat}"
LOG="${TETRIS_TELEMETRY:-/tmp/openjev-tetris.log}"
MODEL_URL="${MODEL_URL:-http://127.0.0.1:8750}"

[ -x "$TETRIS_BIN" ] || { echo "no tetris binary at $TETRIS_BIN (set TETRIS_BIN)"; exit 1; }
curl -sf "$MODEL_URL/health" >/dev/null || { echo "model server down at $MODEL_URL — run: bun run model"; exit 1; }

SESSION="${TMUX_SESSION:-$(tmux list-sessions -F '#{session_name}' 2>/dev/null | head -1)}"
[ -n "$SESSION" ] || { echo "no tmux session to reuse; start one with: tmux new -s openjev"; exit 1; }

WINDOW="$(tmux list-windows -t "$SESSION" -F '#{window_index}' | head -1)"
# Do not assume pane 0 -- pane-base-index is commonly 1, and hardcoding 0 fails with "can't find pane".
PANE="$(tmux list-panes -t "$SESSION:$WINDOW" -F '#{pane_index}' | head -1)"
TARGET="$SESSION:$WINDOW"
LEFT="$TARGET.$PANE"

# Reuse the first pane, close any others we may have left behind on a previous run.
tmux kill-pane -a -t "$LEFT" 2>/dev/null || true
tmux send-keys -t "$LEFT" C-c 2>/dev/null || true

: > "$LOG"
"$TETRIS_BIN" press q >/dev/null 2>&1 || true   # fresh game; TETRIS_STATE_FILE is exported below

# Left: the game as a human would watch it. Right: what the model was asked and what it answered.
tmux split-window -h -t "$LEFT" -c "$ROOT"
RIGHT="$(tmux list-panes -t "$TARGET" -F '#{pane_index}' | tail -1)"
tmux resize-pane  -t "$LEFT" -x 44

tmux send-keys -t "$LEFT" \
  "clear; export TETRIS_STATE_FILE='$STATE'; while true; do printf '\\033[H\\033[2J'; '$TETRIS_BIN' show; sleep 0.3; done" C-m

tmux send-keys -t "$TARGET.$RIGHT" \
  "clear; export TETRIS_STATE_FILE='$STATE' TETRIS_BIN='$TETRIS_BIN' TETRIS_TELEMETRY='$LOG' MODEL_URL='$MODEL_URL'; \
   bun agent/tetris.js --reset --pieces $PIECES" C-m

echo "running in tmux session '$SESSION' window $WINDOW — board left, model right"
