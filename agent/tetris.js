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
const SIMDIR = mkdtempSync(join(tmpdir(), 'openjev-sim-'));
const SCRATCH = join(SIMDIR, 'sim.dat');
const SCRATCH2 = join(SIMDIR, 'sim2.dat');   // second ply
const SCRATCH3 = join(SIMDIR, 'sim3.dat');

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
  // How close is the board to actually cashing in? Only rows that have been started count -- an untouched
  // row "needs 10" and is not near completion. Without this the agent had no signal for line-clear
  // progress at all: it optimised flat-and-low forever and never aimed at finishing a row.
  let need = WIDTH;
  for (const row of board) {
    const filled = row.reduce((n, v) => n + (v ? 1 : 0), 0);
    if (filled > 0 && filled < WIDTH) need = Math.min(need, WIDTH - filled);
  }
  return { heights, holes, bumpiness, need, maxHeight: Math.max(...heights) };
}

const filledRows = (board) => board.reduce((n, row) => n + (row.every(Boolean) ? 1 : 0), 0);

/** Every distinct placement of the piece currently falling in `stateFile`, as {keys, after}. */
function placements(stateFile, scratch) {
  const out = [];
  const seen = new Set();
  for (let rot = 0; rot < 4; rot++) {
    for (let dx = -6; dx <= 6; dx++) {
      const keys = [...Array(rot).fill('w'), ...Array(Math.abs(dx)).fill(dx < 0 ? 'a' : 'd'), 'space'];
      copyFileSync(stateFile, scratch);
      run(scratch, ['press', ...keys]);
      const after = dump(scratch);
      const sig = JSON.stringify(after.board);
      if (seen.has(sig)) continue;
      seen.add(sig);
      out.push({ keys, after });
    }
  }
  return out;
}

/** Second ply: with the board this move produces, what is the best the *next* piece could then do?
 *  The lookahead is search, but the judgement stays with the model -- this only adds a clause to the
 *  description, it does not pick the move. */
function followup(stateFile, baseHoles, baseLines) {
  let bestCleared = 0, bestHoles = Infinity, any = false;
  for (const { after } of placements(stateFile, SCRATCH3)) {
    if (after.game_over) continue;
    any = true;
    bestCleared = Math.max(bestCleared, (after.lines ?? 0) - baseLines);
    bestHoles = Math.min(bestHoles, features(after.board).holes - baseHoles);
  }
  return { any, bestCleared, bestHoles: bestHoles === Infinity ? 0 : bestHoles };
}

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
        need: f.need,
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
    `The nearest unfinished row needs ${base.need} more cell${base.need === 1 ? '' : 's'} to complete and clear.`,
    `The single most important rule is to never bury an empty cell under a block: a buried cell cannot be`,
    `filled, so its row can never be completed until every row above it clears first.`,
    `After that, completing rows matters most, then keeping the stack low and the surface flat.`,
  ].join(' ');
}

/** Describe the board this move would produce. Every description is factually true -- the model is being
 *  used as a reranker over outcomes (his `rerank` protocol: premise states the goal, options compete on
 *  P(entailment)), not as a truth test. */
function hypothesis(c) {
  if (c.gameOver) return 'This move stacks the pieces over the top of the board and ends the game.';
  // Lead with the decisive fact and say it in words. The model is a language model: "buries a cell that can
  // never be filled" carries far more weight than the difference between "roughness 9" and "roughness 13".
  const lead = c.cleared > 0
    ? `This move completes and clears ${c.cleared} line${c.cleared === 1 ? '' : 's'}.`
    : c.holesAdded > 0
      ? `This move traps ${c.holesAdded} empty cell${c.holesAdded === 1 ? '' : 's'} under the blocks, ruining those rows.`
      : 'This move is clean and traps no empty cells.';

  // Words carry the judgement, numbers keep the options apart. Qualitative buckets alone collapsed most
  // placements to an identical sentence, dedupe left a single option, and the model had nothing to choose
  // between -- the same tie that made the first numeric version drop everything down the left wall.
  const tall = c.maxHeight >= 15 ? `The stack is dangerously close to the top at ${c.maxHeight} rows`
             : c.maxHeight >= 9 ? `The stack is getting high at ${c.maxHeight} rows`
             : `The stack stays low at ${c.maxHeight} rows`;
  const flat = c.bumpiness <= 4 ? `and the surface is left flat and easy to build on, roughness ${c.bumpiness}`
             : c.bumpiness <= 9 ? `and the surface is left a little uneven, roughness ${c.bumpiness}`
             : `and the surface is left jagged and full of gaps, roughness ${c.bumpiness}`;
  const close = c.need <= 2 ? ` A row is left needing only ${c.need} more cell${c.need === 1 ? '' : 's'} to clear.` : '';
  // What the piece after this one could then do. The search finds it; the model still decides whether it
  // is worth having.
  const n = c.next;
  const ahead = !n ? ''
    : !n.any ? ' After it there is no room left for the next piece at all.'
    : n.bestCleared > 0 ? ` After it the next piece could immediately clear ${n.bestCleared} more line${n.bestCleared === 1 ? '' : 's'}.`
    : n.bestHoles > 0 ? ' After it the next piece has nowhere clean to go and would have to trap more cells.'
    : ' After it the next piece still has a clean place to go.';
  return `${lead} ${tall}, ${flat}.${close}${ahead}`;
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
// Ablation: is the model choosing, or is the Pareto filter? 'random' picks uniformly from the same
// ballot, 'first' always takes the first -- both skip the model entirely.
const POLICY = opt('--policy', 'model');

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
  // Trim the ballot to moves that are not strictly worse than some other move on every axis at once
  // (lines cleared, cells buried, stack height, surface roughness). This removes dominated options without
  // taking a position on the trade-offs between them -- the model still chooses.
  //
  // An earlier attempt simply dropped every hole-creating move when a clean one existed. That backfired:
  // flat-topping the stack never buries anything, while filling a gap beside it usually does, so the
  // filter left only tower-building moves. The stack went 2 -> 5 -> 7 rows in three pieces, and by piece 4
  // there was no clean move left and it took seven holes at once.
  const alive = out.filter(c => !c.gameOver);
  const pool = alive.length ? alive : out;
  const better = (a, b) =>   // a is at least as good as b on every axis
    a.cleared >= b.cleared && a.holesAdded <= b.holesAdded &&
    a.maxHeight <= b.maxHeight && a.bumpiness <= b.bumpiness;
  const dominated = (x) => pool.some(y => y !== x && better(y, x) && !better(x, y));
  const front = pool.filter(c => !dominated(c));
  const ballot = front.length ? front : pool;

  // Second ply, over the trimmed ballot only. Doing it for every placement would be ~19x19 simulations a
  // piece; over the front it is closer to 6x19 and stays well under a second.
  // Off by default: measured at 2.8 lines/game with it against 10.8 without. The extra clause appears to
  // crowd out the facts that decide the move. Kept behind a flag because the simulation is sound -- it is
  // the wording that hurts, and that is worth another attempt.
  if (argv.includes('--lookahead')) for (const c of ballot) {
    copyFileSync(LIVE, SCRATCH2);
    run(SCRATCH2, ['press', ...c.keys]);
    const a = dump(SCRATCH2);
    c.next = followup(SCRATCH2, features(a.board).holes, a.lines ?? 0);
  }

  // If two placements really are indistinguishable in words, keep the better one by the objective rather
  // than the one with the fewest keypresses -- "fewest keys" means "furthest left", which is not a policy.
  const rank = (c) => [c.holesAdded, -c.cleared, c.maxHeight, c.bumpiness, c.keys.length];
  const byText = new Map();
  for (const c of ballot) {
    const t = hypothesis(c);
    const prev = byText.get(t);
    if (!prev || rank(c) < rank(prev)) byText.set(t, c);
  }
  const uniq = [...byText.values()];
  const hyps = uniq.map(hypothesis);
  const t0 = performance.now();
  let res;
  if (POLICY === 'model') {
    try { res = await score(prem, hyps); }
    catch (e) { log(`model unreachable: ${e.message}`); break; }
  } else {
    const i = POLICY === 'random' ? Math.floor(Math.random() * uniq.length) : 0;
    res = { probs: uniq.map((_, k) => (k === i ? 1 : 0)), argmax: i };
  }
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
  log(`   ${uniq.length} outcomes · ${ms.toFixed(0)} ms` + ` of ${pool.length}`);
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
