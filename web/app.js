/* The agent loop, as an m.js component.
 *
 * Each tick: read the game's state through window.__agent (same origin, so the iframe's scope is directly
 * reachable), render it as a premise, send the premise plus one justifying statement per legal action to
 * the model, and apply the argmax. The model never sees an action name -- only statements about the world.
 */

const LATENCY_BUDGET = 400;   // ms; a cooking game is turn-ish, so this is comfort not hard real-time

export function Agent(M) {
  return {
    // ---- wiring
    agent: null,
    booted: false,
    error: '',
    modelInfo: '',

    // ---- loop
    running: false,
    tick: 0,
    lastMs: 0,
    decisions: 0,
    startedAt: 0,

    // ---- latest decision
    premise: '',
    options: [],        // [{id, label, hypothesis, p}]
    chosen: null,
    history: [],        // recent {label, p}
    game: {},
    gameSrc: './game/cook2.html?v=' + Date.now(),   // dev cache-bust; see index.html

    template: `
<div>
<header>
  <h1>openjev &mdash; Qwen3.5-4B NLI cross-encoder plays Cook Fever</h1>
  <span class="sub" x-text="status()"></span>
  <span class="grow"></span>
  <button @click="toggle()" :class="running ? 'on' : ''" :disabled="!booted"
          x-text="running ? 'pause agent' : 'run agent'"></button>
  <button @click="step()" :disabled="!booted || running">single step</button>
  <button @click="startLevel()" :disabled="!booted">restart level</button>
</header>

<main>
  <div id="stage">
    <iframe :src="gameSrc" title="Cook Fever"></iframe>
  </div>

  <aside>
    <div class="card" x-show="error">
      <div class="k">problem</div>
      <div class="warn" x-text="error"></div>
    </div>

    <div class="card">
      <div class="k">game state</div>
      <div class="stats">
        <div class="stat"><b x-text="game.served ?? 0"></b><span x-text="'served / ' + (game.goal ?? '?')"></span></div>
        <div class="stat"><b x-text="game.coins ?? 0"></b><span>coins</span></div>
        <div class="stat"><b x-text="game.timeLeft ?? 0"></b><span>seconds left</span></div>
        <div class="stat"><b x-text="(game.customers || []).length"></b><span>waiting</span></div>
      </div>
    </div>

    <div class="card">
      <div class="k" x-text="'P(entailment) per candidate action &mdash; ' + options.length + ' scored'"></div>
      <template x-if="!options.length">
        <div class="muted">no legal action right now</div>
      </template>
      <template x-for="o in options" :key="o.id">
        <div class="row" :class="o.id === chosen ? 'win' : ''">
          <span class="val" x-text="o.p == null ? '&mdash;' : o.p.toFixed(2)"></span>
          <span class="bar"><i :style="'width:' + ((o.p || 0) * 100).toFixed(1) + '%'"></i></span>
          <span class="txt"><span x-text="o.hypothesis"></span><em x-text="'&rarr; ' + o.label"></em></span>
        </div>
      </template>
    </div>

    <div class="card">
      <div class="k">timing</div>
      <div class="stats">
        <div class="stat"><b x-text="Math.round(lastMs)"></b><span>ms / decision</span></div>
        <div class="stat"><b x-text="rate()"></b><span>decisions / s</span></div>
        <div class="stat"><b x-text="decisions"></b><span>decisions</span></div>
        <div class="stat"><b x-text="options.length"></b><span>hypotheses</span></div>
      </div>
    </div>

    <div class="card">
      <div class="k">last actions taken</div>
      <ol>
        <template x-for="h in history" :key="h.n">
          <li x-text="h.label + '  (' + h.p.toFixed(2) + ')'"></li>
        </template>
      </ol>
    </div>

    <div class="card">
      <div class="k">premise sent to the model</div>
      <div id="premise" x-text="premise || '(waiting for the game to start)'"></div>
    </div>
  </aside>
</main>
</div>`,

    // ------------------------------------------------------------------ init
    async init() {
      try {
        const h = await (await fetch('/api/health')).json();
        this.modelInfo = h.model ? `model on ${h.model.device}` : (h.error || 'model offline');
        if (!h.model) this.error = 'Model server is not up. Run:  bun run model';
      } catch (e) { this.error = 'Cannot reach the web server API.'; }
      M.redraw();

      // Query the DOM rather than $refs: m.js consumes the x-ref attribute, but $refs is not reliably
      // reachable from a method closure here, and the iframe only gains __agent once its module has run.
      const wait = setInterval(() => {
        const f = document.querySelector('#stage iframe');
        const w = f && f.contentWindow;
        if (w && w.__agent) {
          clearInterval(wait);
          this.agent = w.__agent;
          this.booted = true;
          this.refresh();
          M.redraw();
        }
      }, 250);
    },

    status() {
      if (this.error) return this.modelInfo;
      if (!this.booted) return 'loading the game…';
      return `${this.modelInfo} · game ${this.game.mode || '?'}`;
    },
    rate() {
      const s = (Date.now() - this.startedAt) / 1000;
      return this.startedAt && s > 0 ? (this.decisions / s).toFixed(1) : '0.0';
    },

    startLevel() {
      this.agent.start(this.game.level || 1);
      this.history = [];
      this.refresh();
    },

    refresh() {
      if (!this.agent) return;
      this.game = this.agent.state();
      this.premise = this.agent.premise();
      this.options = this.agent.actions().map(o => ({ ...o, p: null }));
      this.chosen = null;
    },

    toggle() {
      this.running = !this.running;
      if (this.running) {
        if (!this.startedAt) this.startedAt = Date.now();
        if (this.agent.mode() !== 'playing') this.agent.start(this.game.level || 1);
        this.loop();
      }
    },

    async loop() {
      while (this.running) {
        const waited = await this.step();
        await new Promise(r => setTimeout(r, Math.max(0, LATENCY_BUDGET - waited)));
      }
    },

    /** One decision: state -> premise + hypotheses -> model -> argmax -> act. Returns elapsed ms. */
    async step() {
      const t0 = performance.now();
      if (!this.agent) return 0;

      this.game = this.agent.state();
      if (this.game.mode !== 'playing') { this.refresh(); return performance.now() - t0; }

      const acts = this.agent.actions();
      this.premise = this.agent.premise();
      if (!acts.length) { this.options = []; M.redraw(); return performance.now() - t0; }

      let r;
      try {
        r = await (await fetch('/api/decide', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ premise: this.premise, hypotheses: acts.map(a => a.hypothesis) }),
        })).json();
      } catch (e) { this.error = 'decide failed: ' + e.message; this.running = false; return performance.now() - t0; }
      if (r.error) { this.error = r.error; this.running = false; return performance.now() - t0; }

      this.options = acts.map((a, i) => ({ ...a, p: r.probs[i] }));
      this.chosen = acts[r.argmax].id;
      this.lastMs = r.ms;
      this.decisions++;

      // The action list is rebuilt every tick, so ids are only valid against the list we just scored.
      this.agent.act(this.chosen);
      this.history.unshift({ n: ++this.tick, label: acts[r.argmax].label, p: r.probs[r.argmax] });
      if (this.history.length > 12) this.history.pop();

      this.game = this.agent.state();
      M.redraw();
      return performance.now() - t0;
    },
  };
}
