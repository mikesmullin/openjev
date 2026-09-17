/* The agent loop for MARS RAID, as an m.js component.
 *
 * The code flies and aims (see web/game/mars-hook.js); the model only chooses WHAT to attack. In a
 * continuous 3D flight sim the geometry is exactly the part a hand-written controller does better, and
 * target priority under a described situation is the part that is actually about judgement.
 */

const TICK = 900;          // ms between decisions; target choice is a tactical call, not a per-frame one
const HISTORY = 120;       // samples kept for the sparkline

/* Series are by target *kind*, not by candidate: the candidate list churns every tick as things die,
   but "saucer / building / scorpion" are stable enough to draw a line through. */
const SERIES = [
  { key: 'saucer',   label: 'saucer',   color: '#e5533d' },
  { key: 'building', label: 'building', color: '#f2b134' },
  { key: 'boss',     label: 'scorpion', color: '#3fbf7f' },
  { key: 'other',    label: 'other',    color: '#4aa3df' },
];

export function Agent(M) {
  return {
    mars: null, booted: false, error: '', modelInfo: '',
    running: false, lastMs: 0, decisions: 0, startedAt: 0,
    premise: '', options: [], chosen: null, history: [], game: {},
    gameSrc: './game/mars.html?v=' + Date.now(),

    template: `
<div>
<header>
  <h1>openjev &mdash; Qwen3.5-4B NLI cross-encoder plays MARS RAID</h1>
  <span class="sub" x-text="status()"></span>
  <span class="grow"></span>
  <button @click="toggle()" :class="running ? 'on' : ''" :disabled="!booted"
          x-text="running ? 'pause agent' : 'run agent'"></button>
  <button @click="step()" :disabled="!booted || running">single step</button>
  <button @click="wake()" :disabled="!booted">wake the scorpion</button>
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

    <div class="card">
      <div class="k" x-text="'P(entailment) per target &mdash; ' + options.length + ' scored'"></div>
      <template x-if="!options.length"><div class="muted">no target right now</div></template>
      <template x-for="o in options" :key="o.id">
        <div class="row" :class="o.id === chosen ? 'win' : ''">
          <span class="val" x-text="o.p == null ? '&mdash;' : o.p.toFixed(2)"></span>
          <span class="bar"><i :style="'width:' + ((o.p||0)*100).toFixed(1) + '%;background:' + o.color"></i></span>
          <span class="txt"><span x-text="o.hypothesis"></span><em x-text="'&rarr; ' + o.label"></em></span>
        </div>
      </template>
    </div>

    <div class="card">
      <div class="k">last 120 decisions</div>
      <svg viewBox="0 0 480 150" preserveAspectRatio="none">
        <line x1="0" y1="149" x2="480" y2="149" stroke="#43261f"></line>
        <line x1="0" y1="75"  x2="480" y2="75"  stroke="#43261f" stroke-dasharray="3 4"></line>
        <line x1="0" y1="1"   x2="480" y2="1"   stroke="#43261f"></line>
        <template x-for="s in lines()" :key="s.key">
          <polyline fill="none" stroke-width="2" :stroke="s.color" :points="s.points"></polyline>
        </template>
      </svg>
      <div class="legend">
        <template x-for="s in series" :key="s.key">
          <span><i :style="'background:' + s.color"></i><span x-text="s.label"></span></span>
        </template>
      </div>
    </div>

    <div class="card">
      <div class="k">timing</div>
      <div class="stats">
        <div class="stat"><b x-text="Math.round(lastMs)"></b><span>ms / decision</span></div>
        <div class="stat"><b x-text="decisions"></b><span>decisions</span></div>
        <div class="stat"><b x-text="game.shots ?? 0"></b><span>shots</span></div>
        <div class="stat"><b x-text="game.altitude ?? 0"></b><span>altitude</span></div>
      </div>
    </div>

    <div class="card">
      <div class="k">premise sent to the model</div>
      <div id="premise" x-text="premise || '(waiting for the game)'"></div>
    </div>
  </aside>
</main>
</div>`,

    series: SERIES,

    /** One polyline per kind: highest probability that kind attracted at each tick. */
    lines() {
      const h = this.history;
      if (h.length < 2) return [];
      const step = 480 / Math.max(1, HISTORY - 1);
      const start = Math.max(0, h.length - HISTORY);
      return SERIES.map(s => ({
        ...s,
        points: h.slice(start).map((row, i) =>
          `${(i * step).toFixed(1)},${(149 - (row[s.key] || 0) * 148).toFixed(1)}`).join(' '),
      })).filter(s => s.points);
    },

    async init() {
      try {
        const h = await (await fetch('/api/health')).json();
        this.modelInfo = h.model ? `model on ${h.model.device}` : 'model offline';
        if (!h.model) this.error = 'Model server is not up. Run:  bun run model';
      } catch { this.error = 'Cannot reach the web server API.'; }
      M.redraw();

      const wait = setInterval(() => {
        const w = (document.querySelector('#stage iframe') || {}).contentWindow;
        if (w && w.__mars) {
          clearInterval(wait);
          this.mars = w.__mars;
          if (!this.mars.ready()) this.mars.start();
          this.booted = true;
          this.refresh();
          M.redraw();
        }
      }, 250);
    },

    status() {
      if (!this.booted) return 'loading the game…';
      return `${this.modelInfo} · ${this.game.mode || '?'}`;
    },
    wake() { this.mars.wakeBoss(); this.refresh(); M.redraw(); },

    refresh() {
      if (!this.mars) return;
      this.game = this.mars.state();
      const c = this.candidates();
      this.premise = c.premise;
      this.options = c.list.map(o => ({ ...o, p: null }));
      this.chosen = null;
    },

    /** Candidate targets, each with the statement about the world that would justify attacking it. */
    candidates() {
      const s = this.mars.state();
      const t = this.mars.targets();
      const near = (k, n) => t.filter(x => x.kind === k).slice(0, n);
      const list = [];
      const add = (target, label, hypothesis, kind) =>
        list.push({ id: list.length, label, hypothesis, kind, target,
                    color: (SERIES.find(x => x.key === kind) || SERIES[3]).color });

      for (const b of near('boss', 2))
        add(b, `attack ${b.label}`,
            `${b.label[0].toUpperCase()}${b.label.slice(1)} is the only part of the scorpion that can be hurt right now, and it is down to ${Math.round(100 * b.hp / b.max)} percent.`,
            'boss');
      // A generic "a saucer is 49 metres away" scored 0.01 against the boss clause at 0.80, so the ship
      // flew into a swarm of twelve and was shot down with the scorpion at 23 percent. The danger has to
      // be in the sentence, and it has to escalate as the hull goes.
      for (const a of near('saucer', 3)) {
        const hurt = s.hull < 40 ? 'The ship is about to be destroyed'
                   : s.hull < 70 ? 'The ship is badly damaged'
                   : 'The ship is under fire';
        add(a, 'attack an alien saucer',
            `${hurt} at ${s.hull} percent hull, and an alien saucer only ${Math.round(a.dist)} metres away is shooting at it. Nothing else matters if the ship is destroyed.`,
            'saucer');
      }
      for (const b of near('building', 3))
        add(b, 'attack a colony building',
            `A colony building is standing ${Math.round(b.dist)} metres away and destroying it damages the colony.`,
            'building');
      if (s.bossState === 'dormant')
        add({ wake: true }, 'wake the scorpion',
            'The colony is already in ruins and the buried scorpion is the only enemy left worth attacking.',
            'other');
      if (s.hull < 45)
        add({ retreat: true }, 'break off and climb',
            `The ship is badly damaged at ${s.hull} percent hull and needs to break off before it is destroyed.`,
            'other');

      const L = [];
      L.push(`A raid on a Mars colony, flying a gunship. Hull is at ${s.hull} percent and the ship is ${s.altitude} metres up.`);
      L.push(s.buildings ? `${s.buildings} colony building${s.buildings === 1 ? ' is' : 's are'} still standing.` : 'Every colony building has been destroyed.');
      L.push(s.saucers ? `${s.saucers} alien saucer${s.saucers === 1 ? ' is' : 's are'} in the air.` : 'No alien saucers are in the air.');
      if (s.bossState === 'dormant') L.push('A giant scorpion lies buried and dormant; it can be woken.');
      else if (s.boss.length) L.push('The scorpion is awake. ' + s.boss.map(b => `${b.part} is at ${b.pct} percent`).join('; ') + '. Every other part of it is armoured and cannot be hurt.');
      L.push('Saucers shoot back and will destroy the ship if ignored. Destroying the colony is the objective.');
      return { premise: L.join(' '), list };
    },

    toggle() {
      this.running = !this.running;
      if (this.running) { this.startedAt ||= Date.now(); this.loop(); }
      else this.mars.release();
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

      const { premise, list } = this.candidates();
      this.premise = premise;
      if (!list.length) { this.options = []; M.redraw(); return performance.now() - t0; }

      let r;
      try {
        r = await (await fetch('/api/decide', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ premise, hypotheses: list.map(o => o.hypothesis) }),
        })).json();
      } catch (e) { this.error = 'decide failed: ' + e.message; this.running = false; return performance.now() - t0; }
      if (r.error) { this.error = r.error; this.running = false; return performance.now() - t0; }

      this.options = list.map((o, i) => ({ ...o, p: r.probs[i] }));
      this.chosen = list[r.argmax].id;
      this.lastMs = r.ms;
      this.decisions++;

      const pick = list[r.argmax];
      if (pick.target.wake) this.mars.wakeBoss();
      else if (pick.target.retreat) this.mars.release();
      else this.mars.aim(pick.target);

      // One sample per kind per tick: the best probability anything of that kind attracted.
      const row = {};
      for (const s of SERIES) row[s.key] = Math.max(0, ...this.options.filter(o => o.kind === s.key).map(o => o.p));
      this.history.push(row);
      if (this.history.length > HISTORY * 2) this.history = this.history.slice(-HISTORY);

      this.game = this.mars.state();
      M.redraw();
      return performance.now() - t0;
    },
  };
}
