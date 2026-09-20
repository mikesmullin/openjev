/* The agent loop for MARS RAID, as an m.js component.
 *
 * Ported from the openjev branch of this repo, with the model swapped underneath. openjev was an NLI
 * cross-encoder: one premise, a list of hypotheses, one P(entailment) per hypothesis, argmax wins. Every
 * decision had to be smuggled into the shape of "how true is this sentence", which is why that branch
 * ended up fighting its own wording -- a sentence can be perfectly true and still be a terrible reason to
 * act on it.
 *
 * Bespoke Nimble asks instead. One shared context, a flat schema of named fields, each answered from
 * the logits of its own permitted answer codes. Target choice is an enum over live candidates; whether
 * to keep pressing is its own enum; whether to wake the scorpion is a boolean; how much danger the ship
 * is in is an ordered enum read back as an expected level. The ballot no longer has to double as the
 * argument.
 *
 * The wire format is still simple-jev v1, so this file is the simplejev-mars file and the two branches
 * are directly comparable; server/nimble_server.py translates v1's vocabulary into Nimble's schema.
 *
 * The code still flies and aims (see web/game/mars-hook.js). Geometry is the part a hand-written
 * controller does better; the model answers the judgement calls.
 */

const TICK = 400;          // ms floor between decisions; in practice the model's RTT sets the pace
const HISTORY = 120;       // samples kept for the sparklines

/* Every question, every tick.
 *
 * simplejev-mars asked the slow questions only every third decision, because there each one cost its
 * own prefill and its own ~100 ms request floor, so four questions was four times the latency. That
 * is not the shape here. The fields of one decision share a single prefill (server/nimble_server.py,
 * `prefix` mode), and the whole schema is rendered once rather than per question, so on this payload:
 *
 *     target only  76.5 ms      +posture  83.5 ms      +threat  110.8 ms      +wake  115.1 ms
 *
 * The fourth question costs 4.5 ms. Staleness costs more than that -- holding a posture or threat
 * reading for three ticks means breaking off up to a second late -- so the cadence is gone and every
 * question is asked every tick. questions-per-inference is now 3 or 4 depending only on whether the
 * scorpion is still buried, not on a counter. */
const CADENCE = 1;         // every question, every decision; see above
const BREAK_OFF = 0.75;    // how sure `posture` must be before it vetoes shooting; see the loop below
/* Breaking off has to be bounded, not just confident.
 *
 * evade() flies 600 m away from the nearest saucer. Saucers respawn without limit, so once the colony
 * is gone and twelve of them are up, "break off" is defensible on almost every tick -- and each one
 * restarts the run at the boss from 600 m. Measured: the scorpion's tail sat at 73 percent for four
 * minutes while `target` correctly picked it at 0.63 every tick and the ship never got inside firing
 * range. The veto was not wrong about the danger; it was just never allowed to end.
 *
 * So a break-off lasts at most BREAK_MAX consecutive decisions, and after that the agent must press for
 * at least BREAK_COOLDOWN before it may break off again. The model still decides whether the fight is
 * going badly; the harness decides that disengaging forever is not a strategy. */
const BREAK_MAX = 6;       // consecutive evade decisions before the agent must re-engage
const BREAK_COOLDOWN = 8;  // decisions of pressing required before breaking off again

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
    posture: null, postureP: 0, threat: null, wake: null,
    breakRun: 0, pressRun: BREAK_COOLDOWN,   // bounded break-off; starts off cooldown

    // Latency bookkeeping. rtt is measured in the browser around the fetch, so it is what the agent
    // actually waits for; serverMs is what the adapter spent talking to llama.cpp. The gap between them
    // is HTTP and proxy overhead, and it is worth being able to see it.
    lastRtt: 0, lastServerMs: null, lastQuestions: 0, lastLabels: 0, lastCached: null, lastInputTokens: 0,
    rtts: [], totalQuestions: 0, totalRtt: 0,
    gameSrc: './game/mars.html?v=' + Date.now(),

    template: `
<div>
<header>
  <h1>Bespoke Nimble &mdash; <span x-text="modelName || 'local model'"></span> plays MARS RAID</h1>
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
        <span x-text="lastCached == null ? lastInputTokens + ' input tok' : 'prefix cache ' + lastCached + ' tok'"></span>
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
        <span class="txt"><span x-text="posture || 'no posture yet'"></span><em>choice &middot; press the attack or break off</em></span>
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
          this.error = 'nimble server is not up. Run:  bun run model';
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
    situation(full = true) {
      const s = this.mars.state();
      const t = this.mars.targets();
      const near = (k, n) => t.filter(x => x.kind === k).slice(0, n);

      const candidates = [];
      const add = (target, kind, uid, description) =>
        candidates.push({ uid, kind, target, description, color: colorOf(uid, kind), sid: hash6(uid) });

      /* Three per kind, where simplejev-mars could only afford two.
         v1 rendered the candidate list into the prompt twice, so every extra candidate was paid for
         twice in prompt evaluation on a backend where that was the whole cost. Nimble renders the
         schema once and the candidates ride inside the shared prefix, so they are prefilled once per
         decision rather than once per question. Measured: 4 candidates 117 ms, 8 candidates 134 ms,
         12 candidates 158 ms, and even 12 leaves the prompt at 1102 tokens against a 2048 budget.
         The list is sorted by distance, so this mostly buys a third building to choose between. */
      const bleeding = s.recentDamage >= 12 || s.hull < 40;

      for (const b of near('boss', 3))
        add(b, 'boss', b.uid || `boss:${b.label}`,
            `Attack ${b.label}: the only part of the scorpion that can be hurt in this phase, down to ${Math.round(100 * b.hp / b.max)} percent.`);
      /* A candidate description has to argue FOR its action. The first version of the saucer sentence
         ended "...more saucers keep spawning, so clearing them is endless", which is true, is useful
         context, and is an argument against picking it -- so saucers scored ~0.01 while the ship was
         being shot down. That caveat belongs in the shared state, where it informs every question
         equally. What belongs here is the reason to shoot this saucer now, escalating as the hull drops. */
      for (const a of near('saucer', 3))
        add(a, 'saucer', a.uid, bleeding
          ? `Shoot down the alien saucer ${Math.round(a.dist)} metres away. The ship is at ${s.hull} percent hull and cannot finish the mission if it is destroyed first.`
          : `Shoot down the alien saucer ${Math.round(a.dist)} metres away. It is firing on the ship.`);
      for (const b of near('building', 3))
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

      // Field order and candidate order are both preserved end to end: JS object enumeration keeps
      // insertion order for these non-numeric-looking uid keys, and Nimble assigns answer codes A, B,
      // C... in that same order. The uid keys are not numeric-looking, so this is stable.
      const criteria = {};
      for (const c of candidates) criteria[c.uid] = c.description;

      // Nimble's answer codes are single letters, so an enum takes 2-26 candidates. With an empty sky
      // and no colony left there may be fewer than two even after the hold option, and then there is
      // simply no target question to ask.
      const questions = {};
      if (candidates.length >= 2)
        questions.target = { type: 'choice', instructions: 'What should the gunship attack right now?', criteria };
      if (!full) return { state, questions, candidates };

      Object.assign(questions, {
        posture: {
          type: 'choice',
          instructions: 'Keep attacking, or break off and climb away to survive?',
          criteria: {
            press: 'Keep attacking. The ship can survive the damage it is taking.',
            break_off: 'Break off and climb away, or the ship will be destroyed.',
          },
        },
        threat: { type: 'score', instructions: 'How much danger is the gunship in right now?', criteria: THREAT_RUBRIC },
      });
      // Only asked while there is something to answer. This is why questions-per-inference moves: it is
      // 3 for most of the raid and 4 while the scorpion is still buried.
      if (s.bossState === 'dormant') {
        questions.wake = {
          type: 'choice',
          instructions: 'Wake the buried scorpion now?',
          criteria: {
            yes: 'Wake it: the colony is finished and it is the last enemy worth attacking.',
            no: 'Leave it buried: there are still colony buildings to destroy.',
          },
        };
      }
      return { state, questions, candidates };
    },

    async ask(state, questions) {
      const t0 = performance.now();
      const r = await fetch('/api/decide', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: this.modelName || 'local', state, questions }),
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
        const { state, questions } = this.situation();
        if (Object.keys(questions).length) await this.ask(state, questions);
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
      while (this.running) {
        const took = await this.step();
        await new Promise(r => setTimeout(r, Math.max(0, TICK - took)));
      }
    },

    async step() {
      const t0 = performance.now();
      if (!this.mars) return 0;
      this.game = this.mars.state();
      if (this.game.mode !== 'playing') { this.refresh(); M.redraw(); return performance.now() - t0; }

      // Slow questions ride along on every CADENCE-th decision; the rest are target-only.
      const full = this.decisions % CADENCE === 0;
      const { state, questions, candidates } = this.situation(full);
      this.premise = JSON.stringify(state, null, 1);
      // v1 requires 1-256 questions. On a target-only tick with an empty sky there is nothing to ask,
      // so do not spend a round trip proving it.
      if (!Object.keys(questions).length) {
        this.mars.release();
        this.options = []; this.chosen = null; M.redraw();
        return performance.now() - t0;
      }

      let r;
      try {
        r = await this.ask(state, questions);
      } catch (e) { this.error = 'decide failed: ' + e.message; this.running = false; return performance.now() - t0; }
      if (r.error) { this.error = r.error; this.running = false; return performance.now() - t0; }

      const a = r.answers;
      const probs = (a.target && a.target.probabilities) || {};
      this.options = candidates.map(c => ({ ...c, p: probs[c.uid] ?? 0 }));
      this.chosen = a.target ? a.target.choice : null;
      // On a target-only tick the slow answers are simply the previous ones, held rather than re-asked.
      if (a.posture) { this.posture = a.posture.choice; this.postureP = a.posture.confidence; }
      if (a.threat) this.threat = a.threat.score;
      if (full) this.wake = a.wake ? { choice: a.wake.choice, p: a.wake.confidence } : null;

      // Latency and question accounting. `questions` is what the adapter actually answered inside this
      // one call, which is the number that makes RTT comparable between ticks.
      // `timing` is our llama.cpp adapter's extension, not part of the v1 response. Upstream's
      // hf-server returns model/answers/usage only, so treat it as absent rather than zero.
      const timing = r.timing || null;
      this.lastRtt = r.__rtt;
      this.lastServerMs = timing ? timing.total_ms : null;
      this.lastQuestions = (timing && timing.questions) || Object.keys(questions).length;
      this.lastLabels = timing ? timing.labels : Object.values(questions)
        .reduce((n, q) => n + (q.type === 'noul' ? 9 : Object.keys(q.criteria).length), 0);
      this.lastCached = timing ? timing.cached_tokens : null;
      this.lastInputTokens = (r.usage && r.usage.input_tokens) || 0;
      this.rtts.push(r.__rtt);
      if (this.rtts.length > HISTORY * 2) this.rtts = this.rtts.slice(-HISTORY);
      this.totalRtt += r.__rtt;
      this.totalQuestions += this.lastQuestions;
      this.decisions++;

      /* Act. posture is a veto over target: the model can decide the fight is lost before it decides
         what to shoot, and breaking off has to win when it does.
       *
       * But a veto needs conviction, not a plurality. Breaking off stops the agent shooting at all, so
       * a bare majority for `break_off` halts the mission: measured mid-game at hull 82, posture read
       * break_off 0.58 while target read `destroy the colony building` 0.75, and the raid sat at 7 of
       * 11 buildings for ninety seconds -- evading, never firing. The target question was right and the
       * veto was overriding it on a coin flip.
       *
       * This is what the probabilities are for. An argmax-only interface would have to take 0.58 as a
       * decision; Nimble returns a distribution, so the agent can require the veto to be decisive and
       * treat "narrowly break off" as "keep fighting, but the fight is getting bad". Not calibration --
       * upstream is explicit that 0.9 does not mean right 90% of the time -- just a threshold tested on
       * this task, which is exactly what they recommend doing with it. */
      const pick = candidates.find(c => c.uid === this.chosen);
      // Bounded break-off: confident enough, not already over the limit, and off cooldown.
      const wantsBreak = this.posture === 'break_off' && this.postureP >= BREAK_OFF;
      const mayBreak = wantsBreak && this.breakRun < BREAK_MAX && this.pressRun >= BREAK_COOLDOWN;
      if (mayBreak) { this.breakRun++; this.pressRun = 0; }
      else { this.pressRun++; if (!wantsBreak) this.breakRun = 0; }
      if (mayBreak) this.mars.evade();
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
