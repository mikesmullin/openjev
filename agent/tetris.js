#!/usr/bin/env bun
/* openjev plays mikesmullin/tetris.
 *
 * The game is a turn-based C binary that persists state to $TETRIS_STATE_FILE and speaks JSON via `dump`.
 * That env var is the whole trick: every candidate placement is evaluated by copying the live state to a
 * scratch file, pressing the keys against the copy and dumping the result. The simulator is the real game,
 * so nothing here reimplements gravity, wall kicks, locking or line clears.
 *
 *   bun agent/tetris.js --pieces 40
 */
import { spawnSync } from 'node:child_process';
import { copyFileSync, mkdtempSync, writeFileSync, appendFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const BIN = process.env.TETRIS_BIN ?? '/workspace/g4a/tetris/tetris';
const LIVE = process.env.TETRIS_STATE_FILE ?? join(tmpdir(), 'openjev-tetris.dat');
const MODEL = process.env.MODEL_URL ?? 'http://127.0.0.1:8750';
const TELEMETRY = process.env.TETRIS_TELEMETRY ?? join(tmpdir(), 'openjev-tetris.log');
const SCRATCH = join(mkdtempSync(join(tmpdir(), 'openjev-sim-')), 'sim.dat');

const KIND = ['I', 'O', 'T', 'S', 'Z', 'J', 'L'];
const WIDTH = 10, HEIGHT = 20;

const run = (stateFile, args) =>
  spawnSync(BIN, args, { env: { ...process.env, TETRIS_STATE_FILE: stateFile }, encoding: 'utf8' });

const dump = (stateFile) => JSON.parse(run(stateFile, ['dump']).stdout);

// ------------------------------------------------------------------ board features
/** Column heights, holes (empty cells with a filled cell somewhere above), and surface roughness. */
function features(board) {
  const heights = Array(WIDTH).fill(0);
  let holes = 0;
  for (let c = 0; c < WIDTH; c++) {
    let seen = false;
    for (let r = 0; r < HEIGHT; r++) {
      if (board[r][c]) { if (!seen) { heights[c] = HEIGHT - r; seen = true; } }
      else if (seen) holes++;
    }
  }
  let bumpiness = 0;
  for (let c = 0; c + 1 < WIDTH; c++) bumpiness += Math.abs(heights[c] - heights[c + 1]);
  return { heights, holes, bumpiness, maxHeight: Math.max(...heights) };
}

const filledRows = (board) => board.reduce((n, row) => n + (row.every(Boolean) ? 1 : 0), 0);

// ------------------------------------------------------------------ candidate moves
/** Every (rotation, horizontal offset) the piece can reach, simulated on a forked state file. */
function candidates(live) {
  const before = dump(live);
  const base = features(before.board);
  const out = [];
  const seen = new Set();

  for (let rot = 0; rot < 4; rot++) {
    for (let dx = -6; dx <= 6; dx++) {
      const keys = [...Array(rot).fill('w'), ...Array(Math.abs(dx)).fill(dx < 0 ? 'a' : 'd'), 'space'];
      copyFileSync(live, SCRATCH);
      run(SCRATCH, ['press', ...keys]);
      const after = dump(SCRATCH);

      // The game clamps illegal moves rather than rejecting them, so different key strings can land the
      // piece in the same place. Dedupe on the resulting board, keeping the shortest key sequence.
      const sig = JSON.stringify(after.board);
      if (seen.has(sig)) continue;
      seen.add(sig);

      const f = features(after.board);
      // Lines already vanished by the time we dump, so recover the count from the height drop.
      const cleared = Math.max(0, (after.lines ?? 0) - (before.lines ?? 0));
      out.push({
        keys, rot, dx, cleared,
        gameOver: !!after.game_over,
        holesAdded: f.holes - base.holes,
        maxHeight: f.maxHeight,
        flatter: f.bumpiness <= base.bumpiness,
        bumpiness: f.bumpiness,
      });
    }
  }
  return { before, base, out };
}

// ------------------------------------------------------------------ language
function premise(state, base) {
  const cols = base.heights.map((h, i) => `${i + 1}:${h}`).join(' ');
  return [
    `A game of Tetris on a board 10 columns wide and 20 rows tall.`,
    `Score ${state.score}, level ${state.level}, ${state.lines} lines cleared so far.`,
    `The falling piece is an ${KIND[state.current.kind]} piece and the next piece is an ${KIND[state.next_piece]}.`,
    `Column heights left to right are ${cols}.`,
    `The stack is ${base.maxHeight} rows tall at its highest and has ${base.holes} buried empty cell${base.holes === 1 ? '' : 's'} under it.`,
    `A good move clears lines, buries no empty cells, keeps the stack low, and leaves the surface flat.`,
    `Burying an empty cell is bad because it cannot be filled until every row above it is cleared.`,
  ].join(' ');
}

/** Describe the board this move would produce. Every description is factually true -- the model is being
 *  used as a reranker over outcomes (his `rerank` protocol: premise states the goal, options compete on
 *  P(entailment)), not as a truth test. */
function hypothesis(c) {
  if (c.gameOver) return 'This move stacks the pieces over the top of the board and ends the game.';
  const parts = [];
  parts.push(c.cleared > 0
    ? `This move completes and clears ${c.cleared} line${c.cleared === 1 ? '' : 's'}`
    : 'This move clears no lines');
  parts.push(c.holesAdded > 0
    ? `buries ${c.holesAdded} new empty cell${c.holesAdded === 1 ? '' : 's'} that cannot be filled`
    : 'buries no new empty cells');
  parts.push(`leaves the tallest column ${c.maxHeight} rows high`);
  // A number the premise can be compared against. Without it every quiet placement produced the same
  // sentence, every probability tied, and the argmax collapsed to whichever candidate came first.
  parts.push(`and leaves the surface with a roughness of ${c.bumpiness}, where 0 is perfectly flat`);
  return parts.join(', ') + '.';
}

// ------------------------------------------------------------------ model
async function score(prem, hyps) {
  const r = await fetch(`${MODEL}/score`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ premise: prem, hypotheses: hyps }),
  });
  if (!r.ok) throw new Error(`model HTTP ${r.status}: ${(await r.text()).slice(0, 200)}`);
  return r.json();
}

// ------------------------------------------------------------------ loop
const argv = process.argv.slice(2);
const opt = (k, d) => { const i = argv.indexOf(k); return i >= 0 ? argv[i + 1] : d; };
const maxPieces = Number(opt('--pieces', 40));

if (argv.includes('--reset')) run(LIVE, ['press', 'q']);
writeFileSync(TELEMETRY, '');

const log = (s) => { appendFileSync(TELEMETRY, s + '\n'); if (!argv.includes('--quiet')) console.log(s); };

log(`openjev tetris  ·  bin ${BIN}`);
log(`state ${LIVE}`);
log('');

let placed = 0;
while (placed < maxPieces) {
  const { before, base, out } = candidates(LIVE);
  if (before.game_over) { log('game over.'); break; }
  if (!out.length) { log('no legal placement.'); break; }

  const prem = premise(before, base);
  // Distinct boards can still produce identical sentences; the model cannot tell those apart, so collapse
  // them and keep the shortest key sequence as the representative.
  const byText = new Map();
  for (const c of out) {
    const t = hypothesis(c);
    const prev = byText.get(t);
    if (!prev || c.keys.length < prev.keys.length) byText.set(t, c);
  }
  const uniq = [...byText.values()];
  const hyps = uniq.map(hypothesis);
  const t0 = performance.now();
  let res;
  try { res = await score(prem, hyps); }
  catch (e) { log(`model unreachable: ${e.message}`); break; }
  const ms = performance.now() - t0;

  const pick = uniq[res.argmax];
  run(LIVE, ['press', ...pick.keys]);
  placed++;

  const after = dump(LIVE);
  const ranked = uniq
    .map((c, i) => ({ c, p: res.probs[i] }))
    .sort((a, b) => b.p - a.p)
    .slice(0, 5);

  const af = features(after.board);
  log(`── piece ${placed}  ${KIND[before.current.kind]}→${KIND[before.next_piece]}  ` +
      `score ${after.score}  lines ${after.lines}  top ${af.maxHeight}  holes ${af.holes}`);
  log(`   ${uniq.length} outcomes · ${ms.toFixed(0)} ms`);
  // Compact columns rather than the full sentence: the tmux pane is ~54 wide and the sentences wrap into
  // unreadable mush. The sentence the model actually scored is printed once, for the move it chose.
  for (const { c, p } of ranked) {
    const keys = c.keys.join('').replace('space', '');
    log(`   ${c === pick ? '▸' : ' '} ${p.toFixed(3)} ${keys.padEnd(7)}` +
        (c.gameOver ? ' ends the game'
         : ` clr ${c.cleared}  holes ${c.holesAdded >= 0 ? '+' : ''}${c.holesAdded}  top ${String(c.maxHeight).padStart(2)}  rough ${c.bumpiness}`));
  }
  log(`   "${hypothesis(pick)}"`.replace(/(.{1,50})(\s|$)/g, (m) => '   ' + m.trim() + '\n').trimEnd());
  log('');
}

const end = dump(LIVE);
log(`finished: score ${end.score}, ${end.lines} lines, ${placed} pieces placed${end.game_over ? ' (game over)' : ''}`);
