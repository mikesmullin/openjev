/* The agent loop for MARS RAID, as an m.js component.
 *
 * The code flies and aims (see web/game/mars-hook.js); the model only chooses WHAT to attack. In a
 * continuous 3D flight sim the geometry is exactly the part a hand-written controller does better, and
 * target priority under a described situation is the part that is actually about judgement.
 */

const TICK = 900;          // ms between decisions; target choice is a tactical call, not a per-frame one
const HISTORY = 120;       // samples kept for the sparkline

/* One line per *option*, the way the Doom HUD plotted one line per action.
 *
 * The candidate list churns -- saucers spawn and die, buildings fall -- so there is no fixed row to hang a
 * line on. Instead each option is content-addressed: hash a stable identity for the thing being shot at
 * ("saucer:17", "building:3", "boss:clawL") down to six hex digits, and that is the series id.
 *
 * Hashing the rendered hypothesis would NOT work: the sentences carry live numbers ("49 metres away",
 * "67 percent"), so the text changes every tick and every tick would mint a new series.
 */
function hash6(str) {
  let h = 0x811c9dc5;                         // FNV-1a, 32-bit
  for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 0x01000193); }
  return ((h >>> 8) & 0xffffff).toString(16).padStart(6, '0');
}
/* Colour straight off the hash, so a series keeps its colour for life without a lookup table. Kind sets
   the hue family, the hash varies it within that family, so saucers stay reddish and the boss greenish. */
/* The other half of the idea: hash the *template* rather than the rendered sentence -- same string with the
   live numbers blanked out. Identity decides which line a point belongs to (no averaging); the template
   decides its colour family and its legend entry, so twelve saucers read as one colour, not twelve. */
const templateOf = (hyp) => hyp.replace(/\d+(\.\d+)?/g, '{{n}}');

const HUE = { saucer: 8, building: 40, boss: 145, other: 205 };
function colorOf(uid, kind) {
  const n = parseInt(hash6(uid), 16);
  return `hsl(${(HUE[kind] ?? 280) + (n % 40) - 20} 72% ${52 + (n >> 8) % 18}%)`;
}

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

    /** Legend: one entry per template on the current ballot, not per option. */
    legend() {
      const m = new Map();
      for (const o of this.options) {
        const t = templateOf(o.hypothesis);
        if (!m.has(t)) m.set(t, { key: hash6(t), color: o.color, label: o.label });
      }
      return [...m.values()];
    },

    /** One polyline per option id. Gaps (the option did not exist that tick) break the line rather than
     *  being drawn through, so a saucer that dies and a new one that spawns are visibly separate. */
    lines() {
      const h = this.history;
      if (h.length < 2) return [];
      const start = Math.max(0, h.length - HISTORY);
      const win = h.slice(start);
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
      const start = Math.max(0, h.length - HISTORY);
      const step = 480 / Math.max(1, HISTORY - 1);
      return h.slice(start).map((row, i) => ({ key: i, x: (i * step).toFixed(1), color: row.pickColor }));
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
      const add = (target, label, hypothesis, kind, uid) => {
        const u = uid || target.uid || `${kind}:${list.length}`;
        list.push({ id: list.length, label, hypothesis, kind, target, uid: u,
                    sid: hash6(u),                                  // identity -> which line
                    tid: hash6(templateOf(hypothesis)),             // template -> colour family + legend
                    color: colorOf(u, kind) });
      };

      for (const b of near('boss', 2))
        add(b, `attack ${b.label}`,
            `${b.label[0].toUpperCase()}${b.label.slice(1)} is the only part of the scorpion that can be hurt right now, and it is down to ${Math.round(100 * b.hp / b.max)} percent.`,
            'boss');
      /* Saucers are only worth turning on when evasion is FAILING. The previous wording stated a
         distance and asserted "nothing else matters", which is true at every instant, so it always won and
         the ship spent the game swatting an endless spawn instead of flattening the colony. The ship now
         jinks continuously (see mars-hook.js), so the deciding fact is whether it is still being hit. */
      /* Only offer saucers when evasion is actually failing.
         The previous version always offered one, with a sentence saying the saucers were missing and could
         be ignored -- and it still won. That is a reranker-specific trap: P(entailment) measures whether the
         sentence is TRUE given the premise, not whether it argues for the action it is bound to. "The
         saucers are missing and can be ignored for now" is extremely true, so it scored high and dragged
         the ship onto a target its own sentence said to skip. A hypothesis must argue FOR its action; if it
         cannot, the option does not belong on the ballot at all. */
      const bleeding = s.recentDamage >= 12;
      if (bleeding) for (const a of near('saucer', 3)) {
        add(a, 'attack an alien saucer',
            `The evasive flying is not working: the ship has lost ${s.recentDamage} hull in the last ten seconds and is down to ${s.hull} percent, so the saucer ${Math.round(a.dist)} metres away has to be shot down before anything else.`,
            'saucer');
      }
      for (const b of near('building', 3))
        add(b, 'attack a colony building',
            `Destroying the colony is the mission and ${s.buildings} building${s.buildings === 1 ? ' is' : 's are'} still standing; this one is only ${Math.round(b.dist)} metres away and the ship is not being hit hard right now.`,
            'building');
      if (s.bossState === 'dormant')
        add({ wake: true }, 'wake the scorpion',
            'The colony is already in ruins and the buried scorpion is the only enemy left worth attacking.',
            'other', 'other:wake');
      if (s.hull < 45)
        add({ retreat: true }, 'break off and climb',
            `The ship is down to ${s.hull} percent hull and has lost ${s.recentDamage} in the last ten seconds; it has to break off and climb away to survive.`,
            'other', 'other:retreat');

      const L = [];
      L.push(`A raid on a Mars colony, flying a gunship. Hull is at ${s.hull} percent and the ship is ${s.altitude} metres up.`);
      L.push(s.recentDamage >= 12
        ? `The ship has lost ${s.recentDamage} hull in the last ten seconds; it is being hit badly and the evasive flying is not keeping it safe.`
        : `The ship is flying evasively and has lost only ${s.recentDamage} hull in the last ten seconds, so the saucers are mostly missing.`);
      L.push(s.buildings ? `${s.buildings} colony building${s.buildings === 1 ? ' is' : 's are'} still standing.` : 'Every colony building has been destroyed.');
      L.push(s.saucers ? `${s.saucers} alien saucer${s.saucers === 1 ? ' is' : 's are'} in the air.` : 'No alien saucers are in the air.');
      if (s.bossState === 'dormant') L.push('A giant scorpion lies buried and dormant; it can be woken.');
      else if (s.boss.length) L.push('The scorpion is awake. ' + s.boss.map(b => `${b.part} is at ${b.pct} percent`).join('; ') + '. Every other part of it is armoured and cannot be hurt.');
      L.push('Saucers shoot back, but they have poor aim against a ship that keeps moving, and more of them keep spawning, so clearing them is endless. Destroying the colony and then the scorpion is the objective.');
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
      else if (pick.target.retreat) this.mars.evade();
      else this.mars.aim(pick.target);

      // One sample per *option* per tick, keyed by its identity hash, plus what was actually chosen.
      const probs = {};
      for (const o of this.options) probs[o.sid] = { p: o.p, color: o.color };
      this.history.push({ probs, pickColor: pick.color, pickLabel: pick.label });
      if (this.history.length > HISTORY * 2) this.history = this.history.slice(-HISTORY);

      this.game = this.mars.state();
      M.redraw();
      return performance.now() - t0;
    },
  };
}
