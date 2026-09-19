/* The agent loop for MARS RAID, as an m.js component.
 *
 * Third model to drive this harness, and the first generative one. openjev scored hypotheses with an
 * NLI cross-encoder; simple-jev read next-token logits for a fixed label set. Neither generated a
 * single token. DiffusionGemma does the opposite -- it writes the decision out as a small JSON object
 * and the harness parses it -- which is only reasonable because diffusion generation is fast: the whole
 * answer lands in a couple of denoising passes rather than token by token.
 *
 * What that costs is the probability distribution. A classifier ranks every candidate; a generative
 * model picks one. The ballot below therefore shows a single full bar rather than a spread, and the
 * decision-history graph degenerates accordingly. That is a real difference between the approaches,
 * not a rendering bug.
 *
 * The code still flies and aims (see web/game/mars-hook.js). Geometry is the part a hand-written
 * controller does better; the model answers the judgement calls.
 */

/* The decision rate is set by the GAME, not by the model.
 *
 * GLiNER answers in ~70 ms, so for the first time in this repo the model is not the bottleneck --
 * openjev needed 35 ms, simple-jev 430, DiffusionGemma 3700. Running the loop as fast as the model
 * allows turned out to be actively harmful: at ~4 decisions/second the autopilot never held an aim
 * long enough to reach the 4.5 degree firing threshold, and the ship fired 2 shots in 28 seconds
 * while eleven buildings stood untouched.
 *
 * 900 ms is roughly what the earlier branches used, and it is about how long the aim servo needs.
 * The model finishing in 70 ms of that budget is the actual result; spending the other 830 ms letting
 * the controller execute is not a compromise. */
const TICK = 900;
const HISTORY = 120;       // samples kept for the sparklines

/* No cadence on this branch.
 *
 * simple-jev needed one forward pass per question, so asking four of them every tick was four times the
 * work and the slow ones were held between ticks. DiffusionGemma writes all four fields into a single
 * JSON object in one generation, so there is nothing to stagger -- every tick asks everything. */

/* One line per *candidate*, the way the openjev branch plotted one line per option.
 *
 * The ballot churns -- saucers spawn and die, buildings fall -- so there is no fixed row to hang a line
 * on. Each candidate is content-addressed instead: hash the stable identity of the thing being shot at
 * ("saucer:17", "building:3", "boss:clawL") down to six hex digits, and that is the series id.
 */
function hash6(str) {
  let h = 0x811c9dc5;                         // FNV-1a, 32-bit
  for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 0x01000193); }
  return ((h >>> 8) & 0xffffff).toString(16).padStart(6, '0');
}
/* Colour straight off the hash, so a series keeps its colour for life without a lookup table. Kind sets
   the hue family and the hash varies it within that family, so saucers stay reddish and the boss
   greenish -- twelve saucers read as one colour, not twelve. */
const HUE = { saucer: 8, building: 40, boss: 145, other: 205 };
function colorOf(uid, kind) {
  const n = parseInt(hash6(uid), 16);
  return `hsl(${(HUE[kind] ?? 280) + (n % 40) - 20} 72% ${52 + (n >> 8) % 18}%)`;
}

const pct = (x) => Math.round(100 * x);
function quantile(sorted, q) {
  if (!sorted.length) return 0;
  const i = (sorted.length - 1) * q;
  const lo = Math.floor(i), hi = Math.ceil(i);
  return lo === hi ? sorted[lo] : sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo);
}

/* How much danger the ship is in. A `score` question returns the expected index over this rubric, so the
   answer is continuous -- 2.4 is a real reading, not a rounding of 2.
 *
 * The first version of this rubric described the rate of damage only ("taking occasional hits", "being
 * hit steadily"). It read 1.8 with fifteen percent hull left, which is not wrong -- nothing was getting
 * through at that instant -- but it is not the question either. Danger is the rate AND the margin left
 * to absorb it, so every level now names both. */
const THREAT_RUBRIC = [
  'No danger: almost nothing is getting through, and the hull has plenty left.',
  'Occasional hits, and the hull is healthy enough to absorb them.',
  'Being worn down: losing hull steadily, or the hull is already low.',
  'About to be destroyed: the hull is nearly gone and any sustained fire finishes it.',
];

export function Agent(M) {
  return {
    mars: null, booted: false, error: '', modelInfo: '', modelName: '',
    running: false, arming: false, decisions: 0,
    premise: '', options: [], chosen: null, history: [], game: {},
    posture: null, postureP: 0, threat: null, wake: null, committed: null,
    // Latency bookkeeping. rtt is measured in the browser around the fetch, so it is what the agent
    // actually waits for; serverMs is what the adapter spent talking to llama.cpp. The gap between them
    // is HTTP and proxy overhead, and it is worth being able to see it.
    lastRtt: 0, lastServerMs: null, lastQuestions: 0, lastLabels: 0, lastCached: null, lastInputTokens: 0,
    lastTpf: null, lastRaw: '',
    rtts: [], totalQuestions: 0, totalRtt: 0,
    gameSrc: './game/mars.html?v=' + Date.now(),

    template: `
<div>
<header>
  <h1>GLiNER 2.5 &mdash; <span x-text="modelName || 'local model'"></span> plays MARS RAID</h1>
  <span class="sub" x-text="status()"></span>
  <span class="grow"></span>
  <button @click="toggle()" :class="running ? 'on' : ''" :disabled="!booted"
          x-text="arming ? 'waking model…' : running ? 'pause agent' : 'run agent'"></button>
  <button @click="stepOnce()" :disabled="!booted || running || arming">single step</button>
  <button @click="wakeNow()" :disabled="!booted">wake the scorpion</button>
</header>

<main>
  <div id="stage"><iframe :src="gameSrc" title="MARS RAID"></iframe></div>

  <aside>
    <div class="card" x-show="error"><div class="k">problem</div><div class="warn" x-text="error"></div></div>

    <div class="card">
      <div class="k">situation</div>
      <div class="stats">
        <div class="stat"><b x-text="game.hull ?? 0"></b><span>hull</span></div>
        <div class="stat"><b x-text="game.buildings ?? 0"></b><span>colony left</span></div>
        <div class="stat"><b x-text="game.saucers ?? 0"></b><span>saucers</span></div>
        <div class="stat"><b x-text="game.bossState || '—'"></b><span x-text="'scorpion' + (game.bossPhase ? ' · ' + game.bossPhase : '')"></span></div>
      </div>
    </div>

    <!-- The headline metric: one HTTP call, several questions answered inside it. -->
    <div class="card">
      <div class="k">inference</div>
      <div class="stats">
        <div class="stat"><b x-text="Math.round(lastRtt)"></b><span>ms RTT</span></div>
        <div class="stat"><b x-text="lastQuestions"></b><span>questions / call</span></div>
        <div class="stat"><b x-text="lastQuestions ? Math.round(lastRtt / lastQuestions) : 0"></b><span>ms / question</span></div>
        <div class="stat"><b x-text="decisions"></b><span>decisions</span></div>
      </div>
      <div class="stats" style="margin-top:10px">
        <div class="stat"><b x-text="Math.round(rttStat('mean'))"></b><span>mean RTT</span></div>
        <div class="stat"><b x-text="Math.round(rttStat('p50'))"></b><span>p50</span></div>
        <div class="stat"><b x-text="Math.round(rttStat('p95'))"></b><span>p95</span></div>
        <div class="stat"><b x-text="totalQuestions"></b><span>questions total</span></div>
      </div>
      <div class="legend" style="margin-top:9px">
        <span x-text="lastServerMs == null ? 'server n/a' : 'server ' + Math.round(lastServerMs) + ' ms'"></span>
        <span x-text="lastServerMs == null ? '' : 'overhead ' + Math.max(0, Math.round(lastRtt - lastServerMs)) + ' ms'"></span>
        <span x-text="lastLabels + ' labels scored'"></span>
        <span x-text="lastTpf ? lastTpf.toFixed(1) + ' tok/forward' : ''"></span>
      </div>
      <div class="legend" style="margin-top:4px">
        <span class="muted" x-text="lastRaw ? 'generated: ' + lastRaw : ''"></span>
      </div>
      <svg viewBox="-16 -6 500 94" preserveAspectRatio="none" style="height:92px">
        <line x1="0" y1="1"  x2="480" y2="1"  stroke="#43261f"></line>
        <line x1="0" y1="79" x2="480" y2="79" stroke="#43261f"></line>
        <polyline fill="none" stroke="#4aa3df" stroke-width="1.6" :points="rttLine()"></polyline>
        <text x="-4" y="5"  fill="#b08c80" font-size="9" text-anchor="end" x-text="Math.round(rttStat('max'))"></text>
        <text x="-4" y="82" fill="#b08c80" font-size="9" text-anchor="end">0</text>
      </svg>
      <div class="legend"><span class="muted">RTT per decision, last 120</span></div>
    </div>

    <div class="card">
      <div class="k">judgement</div>
      <div class="row">
        <span class="val" x-text="postureP ? postureP.toFixed(2) : '—'"></span>
        <span class="bar"><i :style="'width:' + (postureP*100).toFixed(1) + '%;background:var(--amber)'"></i></span>
        <span class="txt"><span x-text="posture || 'no posture yet'"></span><em>derived from threat &middot; not asked of the model</em></span>
      </div>
      <div class="row">
        <span class="val" x-text="threat == null ? '—' : threat.toFixed(2)"></span>
        <span class="bar"><i :style="'width:' + (threat == null ? 0 : threat/3*100).toFixed(1) + '%;background:var(--red)'"></i></span>
        <span class="txt"><span x-text="threatLabel()"></span><em>score &middot; 0&ndash;3 danger rubric</em></span>
      </div>
      <template x-if="wake">
        <div class="row">
          <span class="val" x-text="wake.p.toFixed(2)"></span>
          <span class="bar"><i :style="'width:' + (wake.p*100).toFixed(1) + '%;background:var(--violet)'"></i></span>
          <span class="txt"><span x-text="'wake the scorpion: ' + wake.choice"></span><em>choice &middot; only asked while it is dormant</em></span>
        </div>
      </template>
    </div>

    <div class="card">
      <div class="k" x-text="'P(target) &mdash; ' + options.length + ' candidates on the ballot'"></div>
      <template x-if="!options.length"><div class="muted">no target right now</div></template>
      <template x-for="o in options" :key="o.uid">
        <div class="row" :class="o.uid === chosen ? 'win' : ''">
          <span class="val" x-text="o.p == null ? '&mdash;' : o.p.toFixed(2)"></span>
          <span class="bar"><i :style="'width:' + ((o.p||0)*100).toFixed(1) + '%;background:' + o.color"></i></span>
          <span class="txt"><span x-text="o.description"></span><em x-text="'&rarr; ' + o.uid"></em></span>
        </div>
      </template>
    </div>

    <div class="card">
      <div class="k">last 120 decisions</div>
      <svg viewBox="-16 -6 500 174" preserveAspectRatio="none">
        <text x="-4" y="4"   fill="#b08c80" font-size="9" text-anchor="end">1</text>
        <text x="-4" y="78"  fill="#b08c80" font-size="9" text-anchor="end">0.5</text>
        <text x="-4" y="152" fill="#b08c80" font-size="9" text-anchor="end">0</text>
        <line x1="0" y1="1"   x2="480" y2="1"   stroke="#43261f"></line>
        <line x1="0" y1="75"  x2="480" y2="75"  stroke="#43261f" stroke-dasharray="3 4"></line>
        <line x1="0" y1="149" x2="480" y2="149" stroke="#43261f"></line>
        <template x-for="s in lines()" :key="s.key">
          <polyline fill="none" stroke-width="1.6" :stroke="s.color" :points="s.points"></polyline>
        </template>
        <template x-for="t in ticks()" :key="t.key">
          <rect :x="t.x" y="152" width="2.4" height="9" :fill="t.color"></rect>
        </template>
      </svg>
      <div class="legend">
        <template x-for="s in legend()" :key="s.key">
          <span><i :style="'background:' + s.color"></i><span x-text="s.label"></span></span>
        </template>
        <span class="muted">ticks = chosen</span>
      </div>
    </div>

    <div class="card">
      <div class="k">state sent to the model</div>
      <div id="premise" x-text="premise || '(waiting for the game)'"></div>
    </div>
  </aside>
</main>
</div>`,

    /* ------------------------------------------------------------------ telemetry */

    rttStat(which) {
      const n = this.rtts.length;
      if (!n) return 0;
      if (which === 'mean') return this.totalRtt / n;
      if (which === 'max') return Math.max(...this.rtts, 1);
      const sorted = [...this.rtts].sort((a, b) => a - b);
      return quantile(sorted, which === 'p50' ? 0.5 : 0.95);
    },
    rttLine() {
      const w = this.rtts.slice(-HISTORY);
      if (w.length < 2) return '';
      const max = Math.max(...w, 1);
      const step = 480 / Math.max(1, HISTORY - 1);
      return w.map((v, i) => `${(i * step).toFixed(1)},${(79 - (v / max) * 78).toFixed(1)}`).join(' ');
    },
    threatLabel() {
      if (this.threat == null) return 'no reading yet';
      return THREAT_RUBRIC[Math.round(this.threat)] || '—';
    },

    /** Legend: one entry per candidate kind on the ballot, not per candidate. */
    legend() {
      const m = new Map();
      for (const o of this.options) if (!m.has(o.kind)) m.set(o.kind, { key: o.kind, color: o.color, label: o.kind });
      return [...m.values()];
    },

    /** One polyline per candidate. Gaps (the candidate was not on that tick's ballot) break the line
     *  rather than being drawn through, so a saucer that dies and a new one that spawns read as separate. */
    lines() {
      const h = this.history;
      if (h.length < 2) return [];
      const win = h.slice(Math.max(0, h.length - HISTORY));
      const step = 480 / Math.max(1, HISTORY - 1);
      const seen = new Map();
      win.forEach(row => Object.entries(row.probs).forEach(([sid, v]) => seen.set(sid, v.color)));

      const out = [];
      for (const [sid, color] of seen) {
        let seg = [];
        win.forEach((row, i) => {
          const e = row.probs[sid];
          if (e) seg.push(`${(i * step).toFixed(1)},${(149 - e.p * 148).toFixed(1)}`);
          else if (seg.length) { if (seg.length > 1) out.push({ key: sid + ':' + out.length, color, points: seg.join(' ') }); seg = []; }
        });
        if (seg.length > 1) out.push({ key: sid + ':' + out.length, color, points: seg.join(' ') });
      }
      return out;
    },

    /** The strip along the baseline: one tick per decision, coloured by what was actually chosen. */
    ticks() {
      const h = this.history;
      const step = 480 / Math.max(1, HISTORY - 1);
      return h.slice(Math.max(0, h.length - HISTORY)).map((row, i) => ({ key: i, x: (i * step).toFixed(1), color: row.pickColor }));
    },

    /* ------------------------------------------------------------------ lifecycle */

    async init() {
      try {
        const h = await (await fetch('/api/health')).json();
        if (h.model && h.model.ok) {
          this.modelName = h.model.model || 'local model';
          this.modelInfo = `${this.modelName} via ${h.model.backend}`;
        } else {
          this.modelInfo = 'model offline';
          this.error = 'simple-jev adapter is not up. Run:  bun run model  (and ~/inference.mjs first)';
        }
      } catch { this.error = 'Cannot reach the web server API.'; }
      M.redraw();

      const wait = setInterval(() => {
        const w = (document.querySelector('#stage iframe') || {}).contentWindow;
        if (w && w.__mars) {
          clearInterval(wait);
          this.mars = w.__mars;
          // Deliberately do NOT start the game here. mars.html has no pause state, but its menu is one:
          // nothing updates until startGame(). Leaving it there means the sim does not begin -- and the
          // ship does not start taking fire -- until the model is hot and has already chosen a target.
          this.booted = true;
          this.refresh();
          M.redraw();
        }
      }, 250);
    },

    status() {
      if (!this.booted) return 'loading the game…';
      if (this.arming) return 'waking the model — game held at the menu…';
      if (this.mars && this.mars.mode() !== 'playing') return `${this.modelInfo} · paused, press run agent`;
      return `${this.modelInfo} · ${this.game.mode || '?'}`;
    },
    wakeNow() { this.mars.wakeBoss(); this.refresh(); M.redraw(); },

    refresh() {
      if (!this.mars) return;
      this.game = this.mars.state();
      const s = this.situation();
      this.premise = JSON.stringify(s.state, null, 1);
      this.options = s.candidates.map(o => ({ ...o, p: null }));
      this.chosen = null;
    },

    /* ------------------------------------------------------------------ the request */

    /** The shared state and the questions asked about it.
     *
     *  simple-jev renders `state` as canonical JSON and treats it as data, not instructions, so the
     *  situation goes over structured rather than as prose. The *candidates* stay prose: a candidate
     *  description is the argument for attacking that thing, and that is where the judgement lives.
     */
    situation() {
      const s = this.mars.state();
      const t = this.mars.targets();
      const near = (k, n) => t.filter(x => x.kind === k).slice(0, n);

      const candidates = [];
      const add = (target, kind, uid, description) =>
        candidates.push({ uid, kind, target, description, color: colorOf(uid, kind), sid: hash6(uid) });

      /* v1 renders the candidate list into the prompt TWICE (section 5: the selected question is asked,
         then asked again verbatim), so every extra candidate and every extra clause is paid for twice in
         prompt evaluation. Two per kind is enough to express a preference -- the list is sorted by
         distance, so the third-nearest building is never the interesting answer -- and it keeps the
         choice question's tail short enough to decide inside a second. */
      const bleeding = s.recentDamage >= 12 || s.hull < 40;

      for (const b of near('boss', 2))
        add(b, 'boss', b.uid || `boss:${b.label}`,
            `Attack ${b.label}: the only part of the scorpion that can be hurt in this phase, down to ${Math.round(100 * b.hp / b.max)} percent.`);
      /* A candidate description has to argue FOR its action. The first version of the saucer sentence
         ended "...more saucers keep spawning, so clearing them is endless", which is true, is useful
         context, and is an argument against picking it -- so saucers scored ~0.01 while the ship was
         being shot down. That caveat belongs in the shared state, where it informs every question
         equally. What belongs here is the reason to shoot this saucer now, escalating as the hull drops. */
      for (const a of near('saucer', 2))
        add(a, 'saucer', a.uid, bleeding
          ? `Shoot down the alien saucer ${Math.round(a.dist)} metres away. The ship is at ${s.hull} percent hull and cannot finish the mission if it is destroyed first.`
          : `Shoot down the alien saucer ${Math.round(a.dist)} metres away. It is firing on the ship.`);
      for (const b of near('building', 2))
        add(b, 'building', b.uid,
            `Destroy the colony building ${Math.round(b.dist)} metres away. Flattening the colony is the mission and ${s.buildings} still stand${s.buildings === 1 ? 's' : ''}.`);
      /* "Hold fire" is only on the ballot when there is nothing real to shoot at.
       *
       * It used to be offered unconditionally, as a legitimate answer rather than filler. On the 27B that
       * was harmless -- it almost never won. On Qwen3.5-9B it won at 0.98 essentially every tick, and the
       * agent flew a whole game with `shots: 0` and eleven buildings standing. Same trap as the saucer
       * sentence, one level up: "attack nothing right now" is a safe, agreeable, always-defensible
       * statement, and a choice question rewards the option that reads best, not the one that does best.
       * Smaller models are far more susceptible to it. So it only appears when it is the honest answer --
       * which also satisfies the 2-candidate minimum when the sky is empty. */
      if (candidates.length < 2)
        add({ hold: true }, 'other', 'other:hold', 'There is nothing in range worth attacking, so hold fire.');

      const state = {
        mission: 'Raid a Mars colony from a gunship. Destroy the colony, then kill the giant scorpion.',
        hull_percent: s.hull,
        hull_lost_last_10s: s.recentDamage,
        altitude_metres: s.altitude,
        colony_buildings_standing: s.buildings,
        alien_saucers_airborne: s.saucers,
        // A snapshot ("a saucer is 49 m away") is always true and therefore always urgent. A trend
        // ("no damage in ten seconds") is what should actually decide whether to break off, so the
        // trend is what goes in the state.
        evasion_working: s.recentDamage < 12,
        scorpion: s.bossState === 'dormant'
          ? { status: 'buried and dormant', note: 'it can be woken, and it is the last enemy worth attacking' }
          : { status: s.bossState, phase: s.bossPhase || null,
              vulnerable_parts: s.boss.map(b => ({ part: b.part, percent: b.pct })),
              note: 'every part not listed here is armoured and cannot be hurt' },
        notes: [
          'Saucers have poor aim against a ship that keeps moving, and more of them keep spawning, so clearing them all is not possible.',
          'The ship is destroyed if the hull reaches zero, and the mission fails with it.',
        ],
      };

      /* The server turns these into one instruction and one JSON answer. posture, threat and wake are
         fixed fields of that object rather than separate questions, so the only thing that varies is
         whether wake is worth asking -- which is what makes questions-per-inference read 4 while the
         scorpion is buried and 3 once it is awake. */
      return { state, candidates, askWake: s.bossState === 'dormant' };
    },

    async ask(state, candidates, askWake) {
      const t0 = performance.now();
      const r = await fetch('/api/decide', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ state, candidates, ask_wake: askWake }),
      });
      const body = await r.json();
      body.__rtt = performance.now() - t0;      // measured around the fetch: what the agent really waits for
      return body;
    },

    /* ------------------------------------------------------------------ the loop */

    /** Warm the model with the real first question while the game is still paused on the menu, then
     *  start the sim and fly immediately. The first call after a page load pays for prompt evaluation
     *  with a cold KV cache, and that cost used to be paid with the ship already airborne and taking fire. */
    async arm() {
      this.arming = true; M.redraw();
      try {
        const { state, candidates, askWake } = this.situation();
        if (candidates.length) await this.ask(state, candidates, askWake);
      } catch (e) { this.error = 'model warm-up failed: ' + e.message; }
      this.arming = false;
    },

    async stepOnce() {
      if (this.mars.mode() !== 'playing') { await this.arm(); this.mars.start(); }
      await this.step();
    },

    async toggle() {
      this.running = !this.running;
      if (!this.running) { this.mars.release(); M.redraw(); return; }
      if (this.mars.mode() !== 'playing') {
        await this.arm();                 // model hot first
        if (this.error) { this.running = false; M.redraw(); return; }
        this.mars.start();                // then unpause
        await this.step();                // and have a target before the first danger arrives
      }
      this.loop();
    },

    async loop() {
      // A throw inside step() used to reject here, stop the loop, and show nothing: the page kept
      // rendering the last good decision while the agent was dead. Surface it instead.
      try {
        while (this.running) {
          const took = await this.step();
          await new Promise(r => setTimeout(r, Math.max(0, TICK - took)));
        }
      } catch (e) {
        this.error = 'agent loop crashed: ' + (e && e.message ? e.message : e);
        this.running = false;
        M.redraw();
        throw e;
      }
    },

    async step() {
      const t0 = performance.now();
      if (!this.mars) return 0;
      this.game = this.mars.state();
      if (this.game.mode !== 'playing') { this.refresh(); M.redraw(); return performance.now() - t0; }

      const { state, candidates, askWake } = this.situation();
      this.premise = JSON.stringify(state, null, 1);
      // Nothing in range: do not spend a 3-second generation proving it.
      if (!candidates.length) {
        this.mars.release();
        this.options = []; this.chosen = null; M.redraw();
        return performance.now() - t0;
      }

      let r;
      try {
        r = await this.ask(state, candidates, askWake);
      } catch (e) { this.error = 'decide failed: ' + e.message; this.running = false; return performance.now() - t0; }
      if (r.error) { this.error = r.error; this.running = false; return performance.now() - t0; }

      const a = r.answers;
      const probs = (a.target && a.target.probabilities) || {};
      this.options = candidates.map(c => ({ ...c, p: probs[c.uid] ?? 0 }));
      this.chosen = a.target ? a.target.choice : null;
      // Every tick asks everything on this branch, so take whatever came back and clear what did not.
      if (a.posture) { this.posture = a.posture.choice; this.postureP = a.posture.confidence; }
      if (a.threat) this.threat = a.threat.score;
      this.wake = a.wake ? { choice: a.wake.choice, p: a.wake.confidence } : null;

      // Latency and question accounting. `questions` is what the adapter actually answered inside this
      // one call, which is the number that makes RTT comparable between ticks.
      // `timing` is our llama.cpp adapter's extension, not part of the v1 response. Upstream's
      // hf-server returns model/answers/usage only, so treat it as absent rather than zero.
      const timing = r.timing || null;
      this.lastRtt = r.__rtt;
      this.lastServerMs = timing ? timing.total_ms : null;
      this.lastQuestions = (timing && timing.questions) || 0;
      this.lastLabels = timing ? timing.labels : candidates.length;
      this.lastTpf = timing && timing.tokens_per_forward;
      this.lastRaw = (timing && timing.raw) || '';
      this.lastCached = timing ? timing.cached_tokens : null;
      this.lastInputTokens = (r.usage && r.usage.input_tokens) || 0;
      this.rtts.push(r.__rtt);
      if (this.rtts.length > HISTORY * 2) this.rtts = this.rtts.slice(-HISTORY);
      this.totalRtt += r.__rtt;
      this.totalQuestions += this.lastQuestions;
      this.decisions++;

      /* Act, exactly as the earlier branches did: re-issue the aim every tick.
       *
       * Two "improvements" were tried here and both made it worse, so they are gone: committing to a
       * target until its kind changes, and only calling aim() when the chosen uid changes. With either
       * in place the ship fired 1-2 shots in 28 seconds. Re-seating the target every tick is what the
       * openjev and simple-jev branches did, and what the autopilot is written to expect -- it reads
       * target.live() each frame, so a fresh object per tick is the normal case, not a disturbance.
       */
      /* Commit to an instance, but keep re-issuing the aim.
       *
       * The server classifies a KIND ("colony building") and resolves it to the nearest instance of
       * that kind. "Nearest" changes as the ship flies, so the target swapped between neighbouring
       * buildings every tick and the ship orbited the colony instead of closing on anything -- one
       * building destroyed per ~40 seconds. Hold the instance while it is alive and while the model
       * still wants that kind; switch freely when the model changes its mind.
       *
       * Note this is hysteresis on WHICH instance, not on whether to re-aim. Suppressing the aim()
       * call itself was tried and broke convergence badly; the autopilot expects to be told every
       * tick. Two separate ideas, only one of them correct.
       */
      const kindOf = (uid) => String(uid).split(':')[0];
      if (this.committed && this.chosen && kindOf(this.committed) === kindOf(this.chosen)
          && candidates.some(c => c.uid === this.committed)) {
        this.chosen = this.committed;
      }
      const pick = candidates.find(c => c.uid === this.chosen) || candidates[0];
      this.committed = pick ? pick.uid : null;
      if (this.posture === 'break_off') this.mars.evade();
      else if (this.wake && this.wake.choice === 'yes') this.mars.wakeBoss();
      else if (!pick || pick.target.hold) this.mars.release();
      else this.mars.aim(pick.target);

      // One sample per candidate per tick, keyed by its identity hash, plus what was actually chosen.
      const row = {};
      for (const o of this.options) row[o.sid] = { p: o.p, color: o.color };
      this.history.push({ probs: row, pickColor: pick ? pick.color : '#5c3a30' });
      if (this.history.length > HISTORY * 2) this.history = this.history.slice(-HISTORY);

      this.game = this.mars.state();
      M.redraw();
      return performance.now() - t0;
    },
  };
}
