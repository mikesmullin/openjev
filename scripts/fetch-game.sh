#!/usr/bin/env bash
# Fetch vibe-arcade's MARS RAID and apply the two local patches it needs.
#
# web/game/mars.html is gitignored: it is someone else's game, not ours, and it is fetched rather than
# vendored. Everything this repo changes about it is in this script, so the file is reproducible and the
# diff against upstream is reviewable.
#
#   bun run game        # or: bash scripts/fetch-game.sh
#
# Deliberately do NOT fetch playroomkit.js alongside it. Without that dependency the game takes its own
# "SOLO MODE" path, which is the one we want -- and the one with the damage bug patched below.
set -euo pipefail

COMMIT=0cc97efd3fb84ce67d8b19370fd9a444e21b3ee2   # pinned; the patch below is written against this
URL="https://raw.githubusercontent.com/mikesmullin/vibe-arcade/${COMMIT}/mars.html"
OUT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/web/game/mars.html"

mkdir -p "$(dirname "$OUT")"
echo "fetching mars.html @ ${COMMIT:0:8}"
curl -sSL --fail -o "$OUT" "$URL"

python3 - "$OUT" <<'PY'
import sys
path = sys.argv[1]
src = open(path, encoding="utf8").read()

# --- patch 1: the solo-mode damage bug -------------------------------------------------------------
# fireTwinLaser stamps `const owner = myId || 'me'`, but the three damage handlers guard with
# `if (ownerId !== myId) return`. Solo, myId is null and owner is 'me', so every hit a solo player lands
# is silently discarded: buildings, rocks and the boss never take damage, and stats.buildings /
# stats.saucers never increment, which is why the in-game KILLS counter reads 0 with the colony flattened.
# The game already has the correct idiom elsewhere: (owner === myId) || (!mpReady && owner === 'me').
OLD = "  if (ownerId !== myId) return;"
NEW = "  if (ownerId !== myId && (mpReady || ownerId !== 'me')) return;  // solo-mode fix: local hits have owner 'me' while myId is null"
n = src.count(OLD)
if n != 3:
    sys.exit(f"expected 3 solo-mode guards to patch, found {n}; upstream changed -- re-check the pin")
src = src.replace(OLD, NEW)

# --- patch 2: the agent bridge ----------------------------------------------------------------------
# One injected import. mars-hook.js owns state -> classifier state, targets -> choice candidates, and
# the autopilot; the agent loop in ../app.js only asks the questions and applies the answers.
ANCHOR = "modelScene.background = new THREE.Color(0xffffff);\n"
BRIDGE = ANCHOR + """/* simple-jev bridge (local injection, see scripts/fetch-game.sh) */
import { bind } from "./mars-hook.js";
bind({
  THREE, ship, aliens, buildings, keys, camera, bossHP, bossMax, bossClaws, stats, bullets,
  groundHeight, getBossPartWorldPos, startGame, startEmerge,
  get state() { return state; }, get hp() { return hp; },
  get firing() { return firing; }, set firing(v) { firing = v; },
  get yaw() { return yaw; }, set yaw(v) { yaw = v; },
  get pitch() { return pitch; }, set pitch(v) { pitch = v; },
  get bossAlive() { return bossAlive; }, get bossState() { return bossState; }, get bossPhase() { return bossPhase; },
});
"""
if src.count(ANCHOR) != 1:
    sys.exit("bridge anchor not found exactly once; upstream changed -- re-check the pin")
src = src.replace(ANCHOR, BRIDGE)

open(path, "w", encoding="utf8").write(src)
print(f"patched {path}")
PY
