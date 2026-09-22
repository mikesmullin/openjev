const WIDTH = 10;
const HEIGHT = 20;
const MASK64 = (1n << 64n) - 1n;

const PIECES = [
  { name: "I", rotations: [[[0, 0], [1, 0], [2, 0], [3, 0]], [[0, 0], [0, 1], [0, 2], [0, 3]]] },
  { name: "O", rotations: [[[0, 0], [1, 0], [0, 1], [1, 1]]] },
  { name: "T", rotations: [[[0, 0], [1, 0], [2, 0], [1, 1]], [[0, 0], [0, 1], [0, 2], [1, 1]], [[1, 0], [0, 1], [1, 1], [2, 1]], [[1, 0], [0, 1], [1, 1], [1, 2]]] },
  { name: "S", rotations: [[[1, 0], [2, 0], [0, 1], [1, 1]], [[0, 0], [0, 1], [1, 1], [1, 2]]] },
  { name: "Z", rotations: [[[0, 0], [1, 0], [1, 1], [2, 1]], [[1, 0], [0, 1], [1, 1], [0, 2]]] },
  { name: "J", rotations: [[[0, 0], [0, 1], [1, 1], [2, 1]], [[0, 0], [1, 0], [0, 1], [0, 2]], [[0, 0], [1, 0], [2, 0], [2, 1]], [[1, 0], [1, 1], [0, 2], [1, 2]]] },
  { name: "L", rotations: [[[2, 0], [0, 1], [1, 1], [2, 1]], [[0, 0], [0, 1], [0, 2], [1, 2]], [[0, 0], [1, 0], [2, 0], [0, 1]], [[0, 0], [1, 0], [1, 1], [1, 2]]] },
];

class SplitMix64 {
  constructor(seed) { this.state = BigInt.asUintN(64, BigInt(seed)); }
  next() {
    this.state = (this.state + 0x9E3779B97F4A7C15n) & MASK64;
    let z = this.state;
    z = ((z ^ (z >> 30n)) * 0xBF58476D1CE4E5B9n) & MASK64;
    z = ((z ^ (z >> 27n)) * 0x94D049BB133111EBn) & MASK64;
    return (z ^ (z >> 31n)) & MASK64;
  }
}

const emptyRow = () => Array(WIDTH).fill(false);
const copyBoard = (board) => board.map((row) => row.slice());

class TetrisGame {
  constructor(seed = 7) {
    this.board = Array.from({ length: HEIGHT }, emptyRow);
    this.linesCleared = 0;
    this.isOver = false;
    this.bag = [];
    this.rng = new SplitMix64(seed);
  }

  spawn() {
    if (this.isOver) return null;
    if (this.bag.length === 0) {
      this.bag = Array.from({ length: PIECES.length }, (_, index) => index);
      for (let index = this.bag.length - 1; index > 0; index -= 1) {
        const other = Number(this.rng.next() % BigInt(index + 1));
        [this.bag[index], this.bag[other]] = [this.bag[other], this.bag[index]];
      }
    }
    return PIECES[this.bag.pop()];
  }

  fits(cells, column, row) {
    for (const [dx, dy] of cells) {
      const x = column + dx;
      const y = row + dy;
      if (x < 0 || x >= WIDTH || y >= HEIGHT) return false;
      if (y >= 0 && this.board[y][x]) return false;
    }
    return true;
  }

  candidates(piece) {
    const result = [];
    const heightsBefore = this.columnHeights(this.board);
    const bumpinessBefore = this.bumpiness(heightsBefore);
    for (let rotation = 0; rotation < piece.rotations.length; rotation += 1) {
      const cells = piece.rotations[rotation];
      const pieceWidth = Math.max(...cells.map(([x]) => x)) + 1;
      const pieceHeight = Math.max(...cells.map(([, y]) => y)) + 1;
      for (let column = 0; column <= WIDTH - pieceWidth; column += 1) {
        let row = -pieceHeight;
        while (this.fits(cells, column, row + 1)) row += 1;
        if (row < 0) continue;

        const next = copyBoard(this.board);
        const occupied = cells.map(([dx, dy]) => [column + dx, row + dy]);
        for (const [x, y] of occupied) next[y][x] = true;

        const landingHeight = HEIGHT - row - Math.floor(pieceHeight / 2);
        const cleared = this.clearLines(next);
        const heightsAfter = this.columnHeights(next);
        const holesBefore = this.holes(this.board);
        const holesAfter = this.holes(next);
        const bump = this.bumpiness(heightsAfter);
        const features = {
          linesCleared: cleared,
          newHoles: Math.max(0, holesAfter - holesBefore),
          landingHeight,
          maxHeight: Math.max(...heightsAfter, 0),
          bumpiness: bump,
          bumpinessDelta: bump - bumpinessBefore,
          wellDepth: this.deepestWell(heightsAfter),
          flushSides: this.flushSides(cells, column, row),
        };
        result.push({
          id: result.length,
          rotation,
          column,
          board: next,
          cells: occupied,
          features,
        });
      }
    }
    return result;
  }

  apply(candidate) {
    this.board = copyBoard(candidate.board);
    this.linesCleared += candidate.features.linesCleared;
    this.isOver = this.board[0].some(Boolean) || this.board[1].some(Boolean);
  }

  describe(candidate, piece) {
    const f = candidate.features;
    const clauses = [];
    clauses.push(f.newHoles > 0
      ? `leaves ${words(f.newHoles)} hole${f.newHoles === 1 ? "" : "s"} under it`
      : "leaves no holes");
    if (f.bumpinessDelta > 2) clauses.push("makes the surface much bumpier");
    else if (f.bumpinessDelta > 0) clauses.push("makes the surface bumpier");
    else if (f.bumpinessDelta < 0) clauses.push("makes the surface flatter");
    else clauses.push("keeps the surface flat");
    if (f.maxHeight >= 15) clauses.push("the stack is getting dangerously tall");
    else if (f.landingHeight > 8) clauses.push("makes the stack taller");
    else clauses.push("keeps the stack low");
    if (f.wellDepth >= 3) clauses.push("leaves a deep well");
    if (f.linesCleared > 0) clauses.push(`clears ${words(f.linesCleared)} line${f.linesCleared === 1 ? "" : "s"}`);
    const body = clauses.length > 1
      ? `${clauses.slice(0, -1).join(", ")}, and ${clauses.at(-1)}`
      : clauses[0];
    return `The ${piece.name} piece dropped at column ${candidate.column} ${body}.`;
  }

  clearLines(grid) {
    const kept = grid.filter((row) => !row.every(Boolean));
    const cleared = grid.length - kept.length;
    grid.splice(0, grid.length, ...Array.from({ length: cleared }, emptyRow), ...kept);
    return cleared;
  }

  columnHeights(grid) {
    return Array.from({ length: WIDTH }, (_, x) => {
      for (let y = 0; y < HEIGHT; y += 1) if (grid[y][x]) return HEIGHT - y;
      return 0;
    });
  }

  holes(grid) {
    let count = 0;
    for (let x = 0; x < WIDTH; x += 1) {
      let covered = false;
      for (let y = 0; y < HEIGHT; y += 1) {
        if (grid[y][x]) covered = true;
        else if (covered) count += 1;
      }
    }
    return count;
  }

  bumpiness(heights) {
    return heights.slice(0, -1).reduce((sum, height, index) => sum + Math.abs(height - heights[index + 1]), 0);
  }

  deepestWell(heights) {
    let deepest = 0;
    for (let x = 0; x < WIDTH; x += 1) {
      const left = x === 0 ? HEIGHT : heights[x - 1];
      const right = x === WIDTH - 1 ? HEIGHT : heights[x + 1];
      deepest = Math.max(deepest, Math.min(left, right) - heights[x]);
    }
    return deepest;
  }

  flushSides(cells, column, row) {
    let count = 0;
    for (const [dx, dy] of cells) {
      for (const [nx, ny] of [[column + dx - 1, row + dy], [column + dx + 1, row + dy]]) {
        if (nx < 0 || nx >= WIDTH) count += 1;
        else if (ny >= 0 && this.board[ny][nx]) count += 1;
      }
    }
    return count;
  }
}

function words(value) {
  const names = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"];
  return value < names.length ? names[value] : String(value);
}

function heuristic(features) {
  return features.linesCleared * 3.4 - features.newHoles * 7.9 - features.landingHeight * 4.5
    - features.bumpiness * 1.2 - features.wellDepth * 3.4;
}

const elements = {
  board: document.querySelector("#board"),
  connection: document.querySelector("#connection"),
  modelStatus: document.querySelector("#model-status"),
  play: document.querySelector("#play"),
  reset: document.querySelector("#reset"),
  connect: document.querySelector("#connect"),
  policy: document.querySelector("#policy"),
  seed: document.querySelector("#seed"),
  evaluationDelay: document.querySelector("#evaluation-delay"),
  evaluationDelayValue: document.querySelector("#evaluation-delay-value"),
  pieceDelay: document.querySelector("#piece-delay"),
  pieceDelayValue: document.querySelector("#piece-delay-value"),
  pieceLabel: document.querySelector("#piece-label"),
  runLabel: document.querySelector("#run-label"),
  pieces: document.querySelector("#pieces"),
  lines: document.querySelector("#lines"),
  decisions: document.querySelector("#decisions"),
  median: document.querySelector("#median"),
  batch: document.querySelector("#batch"),
  bucket: document.querySelector("#bucket"),
  candidateCount: document.querySelector("#candidate-count"),
  rankings: document.querySelector("#rankings"),
  log: document.querySelector("#log"),
};

const context = elements.board.getContext("2d");
const state = {
  game: new TetrisGame(7),
  running: false,
  generation: 0,
  piece: null,
  candidates: [],
  scores: [],
  evaluating: null,
  chosen: null,
  pieces: 0,
  decisions: 0,
  latencies: [],
  lastBatchMs: null,
  bucket: null,
  log: [],
  apiOnline: false,
};

function sleep(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

function draw() {
  const scale = window.devicePixelRatio || 1;
  const logicalWidth = 300;
  const logicalHeight = 600;
  if (elements.board.width !== logicalWidth * scale) {
    elements.board.width = logicalWidth * scale;
    elements.board.height = logicalHeight * scale;
  }
  context.setTransform(scale, 0, 0, scale, 0, 0);
  context.clearRect(0, 0, logicalWidth, logicalHeight);
  const cell = Math.min(logicalWidth / WIDTH, logicalHeight / HEIGHT);
  const offsetX = (logicalWidth - cell * WIDTH) / 2;
  const offsetY = (logicalHeight - cell * HEIGHT) / 2;

  context.fillStyle = "#05070b";
  context.fillRect(offsetX, offsetY, cell * WIDTH, cell * HEIGHT);
  context.strokeStyle = "#172033";
  context.lineWidth = 1;
  for (let x = 0; x <= WIDTH; x += 1) {
    context.beginPath(); context.moveTo(offsetX + x * cell, offsetY); context.lineTo(offsetX + x * cell, offsetY + HEIGHT * cell); context.stroke();
  }
  for (let y = 0; y <= HEIGHT; y += 1) {
    context.beginPath(); context.moveTo(offsetX, offsetY + y * cell); context.lineTo(offsetX + WIDTH * cell, offsetY + y * cell); context.stroke();
  }

  for (let y = 0; y < HEIGHT; y += 1) {
    for (let x = 0; x < WIDTH; x += 1) if (state.game.board[y][x]) drawCell(x, y, "#47d8e8", offsetX, offsetY, cell);
  }
  if (state.evaluating != null) {
    for (const [x, y] of state.candidates[state.evaluating]?.cells ?? []) {
      if (y >= 0) drawOutline(x, y, "#f6a84b", offsetX, offsetY, cell);
    }
  }
  if (state.chosen) {
    for (const [x, y] of state.chosen.cells) {
      if (y >= 0) drawCell(x, y, "#5bdd8c", offsetX, offsetY, cell);
    }
  }

  elements.pieceLabel.textContent = `Piece ${state.piece?.name ?? "—"}`;
  elements.runLabel.textContent = state.running ? "Playing" : (state.game.isOver ? "Topped out" : "Paused");
  elements.play.textContent = state.running ? "Pause" : (state.game.isOver ? "Restart" : "Play");
  elements.pieces.textContent = String(state.pieces);
  elements.lines.textContent = String(state.game.linesCleared);
  elements.decisions.textContent = String(state.decisions);
  elements.median.textContent = state.latencies.length ? `${median(state.latencies).toFixed(2)} ms` : "—";
  elements.batch.textContent = state.lastBatchMs == null ? "—" : `${state.lastBatchMs.toFixed(1)} ms`;
  elements.bucket.textContent = state.bucket ? `L${state.bucket}` : "—";
  elements.candidateCount.textContent = state.candidates.length ? `${state.candidates.length} landings` : "No piece active";
  renderRankings();
  renderLog();
}

function drawCell(x, y, color, offsetX, offsetY, cell) {
  context.fillStyle = color;
  context.fillRect(offsetX + x * cell + 2, offsetY + y * cell + 2, cell - 4, cell - 4);
}

function drawOutline(x, y, color, offsetX, offsetY, cell) {
  context.strokeStyle = color;
  context.lineWidth = 2;
  context.strokeRect(offsetX + x * cell + 2, offsetY + y * cell + 2, cell - 4, cell - 4);
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

function renderRankings() {
  if (!state.scores.length) {
    elements.rankings.innerHTML = '<li class="muted">The current piece has not been scored yet.</li>';
    return;
  }
  const ranked = state.scores.map((score, index) => ({ ...score, index }))
    .sort((a, b) => b.pTrue - a.pTrue).slice(0, 8);
  elements.rankings.innerHTML = ranked.map((item) => {
    const selected = state.chosen?.id === state.candidates[item.index]?.id;
    return `<li class="${selected ? "selected" : ""}">
      <span class="score">${(item.pTrue * 100).toFixed(1)}%</span>
      <span class="sentence">${escapeHTML(item.sentence)}</span>
    </li>`;
  }).join("");
}

function renderLog() {
  elements.log.innerHTML = state.log.length
    ? state.log.map((entry) => `<li>${escapeHTML(entry)}</li>`).join("")
    : '<li class="muted">No pieces placed yet.</li>';
}

function escapeHTML(value) {
  return value.replace(/[&<>'"]/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[character]));
}

async function checkAPI() {
  elements.connection.textContent = "Laya API: checking…";
  elements.connection.className = "status-pill offline";
  try {
    const response = await fetch("/api/health", { cache: "no-store" });
    const body = await response.json();
    if (!response.ok || body.status !== "ok") throw new Error(body.error ?? "unhealthy");
    state.apiOnline = true;
    elements.connection.textContent = `Laya API: online · L${body.bucket}`;
    elements.connection.className = "status-pill online";
    elements.modelStatus.textContent = "Model ready";
  } catch (error) {
    state.apiOnline = false;
    elements.connection.textContent = "Laya API: offline";
    elements.connection.className = "status-pill error";
    elements.modelStatus.textContent = "Start LayaServer first";
    if (elements.policy.value === "laya") elements.play.title = String(error);
  }
}

async function scoreStates(states) {
  const started = performance.now();
  const response = await fetch("/api/laya/tetris/score", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ states }),
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error ?? "Laya scoring failed");
  state.lastBatchMs = body.elapsedMs ?? (performance.now() - started);
  state.bucket = body.bucket;
  return body.results;
}

async function playLoop(run) {
  while (state.running && state.generation === run && !state.game.isOver) {
    const piece = state.game.spawn();
    if (!piece) break;
    state.piece = piece;
    state.candidates = state.game.candidates(piece);
    state.scores = [];
    state.evaluating = null;
    state.chosen = null;
    draw();
    if (!state.candidates.length) break;

    let chosenIndex = 0;
    if (elements.policy.value === "laya") {
      if (!state.apiOnline) await checkAPI();
      if (!state.apiOnline) break;
      try {
        const sentences = state.candidates.map((candidate) => state.game.describe(candidate, piece));
        const results = await scoreStates(sentences);
        if (state.generation !== run) return;
        state.scores = results.map((result, index) => ({
          pTrue: result.pTrue,
          sentence: sentences[index],
          milliseconds: result.latencyMs ?? 0,
        }));
        state.latencies.push(...results.map((result) => result.latencyMs ?? 0));
        state.decisions += results.length;
        chosenIndex = state.scores.reduce((best, score, index, scores) => score.pTrue > scores[best].pTrue ? index : best, 0);
        for (let index = 0; index < state.candidates.length; index += 1) {
          if (!state.running || state.generation !== run) return;
          state.evaluating = index;
          draw();
          await sleep(Number(elements.evaluationDelay.value));
        }
      } catch (error) {
        elements.modelStatus.textContent = String(error);
        state.running = false;
        draw();
        return;
      }
    } else if (elements.policy.value === "heuristic") {
      state.scores = state.candidates.map((candidate) => ({
        pTrue: heuristic(candidate.features),
        sentence: state.game.describe(candidate, piece),
        milliseconds: 0,
      }));
      chosenIndex = state.scores.reduce((best, score, index, scores) => score.pTrue > scores[best].pTrue ? index : best, 0);
      state.evaluating = chosenIndex;
      draw();
      await sleep(Number(elements.evaluationDelay.value));
    } else {
      chosenIndex = Math.floor(Math.random() * state.candidates.length);
      state.evaluating = chosenIndex;
      draw();
      await sleep(Number(elements.evaluationDelay.value));
    }

    if (!state.running || state.generation !== run) return;
    state.evaluating = null;
    state.chosen = state.candidates[chosenIndex];
    const selectedScore = state.scores[chosenIndex]?.pTrue;
    state.log.unshift(`${piece.name} → column ${state.chosen.column} rot ${state.chosen.rotation} · ${elements.policy.value === "laya" ? "P(clean)" : "score"} ${Number(selectedScore ?? 0).toFixed(3)}`);
    state.log = state.log.slice(0, 12);
    draw();
    await sleep(Number(elements.pieceDelay.value));
    if (!state.running || state.generation !== run) return;
    state.game.apply(state.chosen);
    state.pieces += 1;
    state.chosen = null;
    draw();
  }
  if (state.generation === run) {
    state.running = false;
    draw();
  }
}

function reset() {
  state.generation += 1;
  state.running = false;
  state.game = new TetrisGame(BigInt(Math.max(0, Number(elements.seed.value) || 0)));
  state.piece = null;
  state.candidates = [];
  state.scores = [];
  state.evaluating = null;
  state.chosen = null;
  state.pieces = 0;
  state.decisions = 0;
  state.latencies = [];
  state.lastBatchMs = null;
  state.bucket = null;
  state.log = [];
  draw();
}

function togglePlay() {
  if (state.running) {
    state.running = false;
    state.generation += 1;
    draw();
    return;
  }
  if (elements.policy.value === "laya" && !state.apiOnline) {
    checkAPI().then(() => { if (state.apiOnline) togglePlay(); });
    return;
  }
  if (state.game.isOver) reset();
  state.running = true;
  const run = ++state.generation;
  draw();
  playLoop(run);
}

elements.play.addEventListener("click", togglePlay);
elements.reset.addEventListener("click", reset);
elements.connect.addEventListener("click", checkAPI);
elements.seed.addEventListener("change", () => { if (!state.running) reset(); });
elements.evaluationDelay.addEventListener("input", () => { elements.evaluationDelayValue.textContent = elements.evaluationDelay.value; });
elements.pieceDelay.addEventListener("input", () => { elements.pieceDelayValue.textContent = elements.pieceDelay.value; });
elements.policy.addEventListener("change", () => {
  if (elements.policy.value === "laya") checkAPI();
  elements.modelStatus.textContent = elements.policy.value === "laya" ? "Checking model…" : `${elements.policy.value} policy`;
});

window.addEventListener("resize", draw);
reset();
checkAPI();
