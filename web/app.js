/* Inbox × GLiNER 2.5 — recommend an operation for every email, and score how spammy it is.
 *
 * Replaces the generative LLM pass in the operator's personal-email agent. Nothing is generated: the
 * recommendation is an argmax over sentences describing each operation, and the spam figure is one
 * sentence's entailment probability read off directly.
 */
import { ACTIONS, FOLDERS, SPAM_SIGNALS, premiseFor, groupsFor, loadConfig, OP_COLORS, colorForAction } from './questions.js';

const WORKERS = 3;        // in-flight requests; the GPU serialises, so this only hides round-trip time

const pct = (n, d) => (d ? Math.round((100 * n) / d) : 0);
const fmtWhen = (iso) => iso
  ? new Date(iso).toLocaleString('en-US', { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' })
  : '—';

export function Inbox(M) {
  return {
    emails: [], results: new Map(), search: '',
    running: false, done: 0, errors: 0, startedAt: 0, elapsed: 0,
    latencies: [], tokensIn: 0, tokensOut: 0, confSum: 0, spamSum: 0, statements: 0,
    error: '', booted: false, modelName: '',

    template: `
<div>
<header>
  <div class="logo"></div>
  <div>
    <h1>Inbox &times; GLiNER 2.5</h1>
    <div class="meta"><span x-text="emails.length"></span> emails &middot; <span x-text="ACTIONS.length"></span> operations &middot; model
      <code x-text="modelName || 'gliner2.5-base-v1'"></code></div>
  </div>
  <span class="grow"></span>
  <span class="pill"><i class="dot" :class="running ? 'busy' : (done ? '' : 'idle')"></i><span x-text="statusText()"></span></span>
  <button class="primary" @click="run()" :disabled="!booted || running" x-text="running ? 'Running…' : 'Run'"></button>
  <button @click="clear()" :disabled="running || !done">Clear</button>
</header>

<div class="cards">
  <div class="card">
    <div class="k">Progress</div>
    <div class="big"><span x-text="done"></span> <small x-text="'/ ' + emails.length"></small></div>
    <div class="bar"><i :style="'width:' + pctDone() + '%'"></i></div>
    <div class="triple">
      <div><b x-text="rate()"></b><span>per second</span></div>
      <div><b x-text="emails.length - done"></b><span>remaining</span></div>
      <div><b :class="errors ? 'err' : ''" x-text="errors"></b><span>errors</span></div>
    </div>
  </div>

  <div class="card">
    <div class="k">Recommended operations</div>
    <div class="donutwrap">
      <svg width="104" height="104" viewBox="0 0 42 42">
        <circle cx="21" cy="21" r="15.9" fill="none" stroke="#0f151c" stroke-width="6"></circle>
        <template x-for="a in donut()" :key="a.name">
          <circle cx="21" cy="21" r="15.9" fill="none" :stroke="a.color" stroke-width="6"
                  :stroke-dasharray="a.dash" :stroke-dashoffset="a.offset" transform="rotate(-90 21 21)"></circle>
        </template>
        <text x="21" y="20.5" text-anchor="middle" fill="#e6edf3" font-size="7" font-weight="600" x-text="done"></text>
        <text x="21" y="26" text-anchor="middle" fill="#5d6b7a" font-size="3.2">triaged</text>
      </svg>
      <div class="cats">
        <template x-for="c in topActions()" :key="c.name">
          <div class="cat">
            <i :style="'background:' + c.color"></i>
            <span class="nm" x-text="c.name"></span>
            <span class="n" x-text="c.n"></span>
            <span class="pc" x-text="c.pc + '%'"></span>
          </div>
        </template>
      </div>
    </div>
  </div>

  <div class="card">
    <div class="k">Signals</div>
    <div class="sig">
      <div><b :style="'color:' + OP_COLORS.delete" x-text="opCount('delete')"></b>
        <span>to delete</span><span x-text="pct(opCount('delete'), done) + '% of triaged'"></span></div>
      <div><b :style="'color:' + OP_COLORS.move" x-text="opCount('move')"></b>
        <span>filed to folders</span><span x-text="pct(opCount('move'), done) + '% of triaged'"></span></div>
      <div><b :style="'color:' + OP_COLORS.skip" x-text="opCount('skip')"></b>
        <span>needs you</span><span x-text="pct(opCount('skip'), done) + '% of triaged'"></span></div>
    </div>
    <div class="ubar"><span class="lbl">likely spam</span>
      <span class="track"><i :style="'width:' + pct(spamOver(50), Math.max(1,done)) + '%;background:' + OP_COLORS.delete"></i></span>
      <span class="n" x-text="spamOver(50)"></span></div>
    <div class="ubar"><span class="lbl">borderline</span>
      <span class="track"><i :style="'width:' + pct(spamBetween(20,50), Math.max(1,done)) + '%;background:#e8b23a'"></i></span>
      <span class="n" x-text="spamBetween(20,50)"></span></div>
    <div class="ubar"><span class="lbl">clean</span>
      <span class="track"><i :style="'width:' + pct(done - spamOver(20), Math.max(1,done)) + '%;background:#3fbf7f'"></i></span>
      <span class="n" x-text="done - spamOver(20)"></span></div>
  </div>

  <div class="card">
    <div class="k">Model</div>
    <div class="sig" style="grid-template-columns:repeat(2,1fr)">
      <div><b x-text="avgConf() + '%'"></b><span>avg confidence</span></div>
      <div><b x-text="avgLatency() + ' ms'"></b><span>avg latency</span></div>
      <div><b x-text="tokensIn.toLocaleString()"></b><span>tokens in</span></div>
      <div><b x-text="tokensOut.toLocaleString()"></b><span>tokens out</span></div>
    </div>
    <div class="note" x-text="note()"></div>
  </div>
</div>

<div class="tablewrap">
  <div class="thead">
    <h2>Emails</h2><span class="count" x-text="visible().length"></span>
    <span class="grow"></span>
    <input type="search" placeholder="Search subject or sender" x-model="search">
  </div>
  <div class="scroll">
    <table>
      <thead><tr>
        <th class="num">#</th><th>Received</th><th>From</th><th>Subject</th>
        <th>Recommended action</th><th>Spam</th><th style="text-align:right">Conf</th>
      </tr></thead>
      <tbody>
        <template x-for="row in visible()" :key="row.e.id">
          <tr>
            <td class="num" x-text="row.i"></td>
            <td class="when" x-text="row.when"></td>
            <td class="from" x-text="row.from"></td>
            <td class="subj"><b x-text="row.e.subject"></b></td>
            <td>
              <template x-if="row.r">
                <span class="tag" :style="row.tagStyle"><i :style="'background:' + row.color"></i><span x-text="row.action"></span></span>
              </template>
              <template x-if="!row.r"><span class="muted">—</span></template>
            </td>
            <td>
              <template x-if="row.r">
                <span class="ans"><span class="spambar"><i :style="'width:' + row.spam + '%;background:' + row.spamColor"></i></span><span class="c" x-text="row.spam + '%'"></span></span>
              </template>
              <template x-if="!row.r"><span class="muted">—</span></template>
            </td>
            <td class="conf" x-text="row.r ? row.conf : ''"></td>
          </tr>
        </template>
      </tbody>
    </table>
  </div>
</div>
</div>`,

    pct, ACTIONS, OP_COLORS,

    // ---------------------------------------------------------------- lifecycle
    async init() {
      try { await loadConfig(); } catch {} // personal taxonomy from /api/config; falls back to example defaults
      this.ACTIONS = ACTIONS;
      try {
        const h = await (await fetch('/api/health')).json();
        if (!h.model) this.error = 'model offline — run: bun run model';
        else this.modelName = (h.model.model || '').split('/').pop() + ' · ' + (h.model.device || '');
      } catch { this.error = 'cannot reach the server'; }
      try {
        const r = await (await fetch('/api/emails?limit=500')).json();
        this.emails = r.emails;
        this.booted = !this.error;
      } catch (e) { this.error = 'failed to load emails: ' + e.message; }
      M.redraw();
    },

    note() {
      return `One flat choice over ${ACTIONS.length} operations — delete, skip, archive and ${FOLDERS.length} ` +
             `folders — plus ${SPAM_SIGNALS.length} spam signals, all scored in a single forward pass. Confidence is a ` +
             `by-product: the chosen operation's share of the probability mass across all ${ACTIONS.length}. ` +
             `The model generates nothing, so tokens out is structurally zero.`;
    },
    statusText() {
      if (this.error) return this.error;
      if (this.running) return `Running ${this.fmtElapsed()}`;
      if (this.done) return `Done ${this.fmtElapsed()}`;
      return 'Ready';
    },
    fmtElapsed() {
      const s = Math.round(this.elapsed / 1000);
      return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
    },

    // ---------------------------------------------------------------- derived
    pctDone() { return this.emails.length ? (100 * this.done) / this.emails.length : 0; },
    rate() { const s = this.elapsed / 1000; return s > 0.3 ? (this.done / s).toFixed(1) : '0.0'; },
    avgLatency() { return this.latencies.length ? Math.round(this.latencies.reduce((a, b) => a + b, 0) / this.latencies.length) : 0; },
    avgConf() { return this.done ? Math.round((100 * this.confSum) / this.done) : 0; },

    opCount(op) { let n = 0; for (const r of this.results.values()) if (r.action.op === op) n++; return n; },
    spamOver(t) { let n = 0; for (const r of this.results.values()) if (r.spam * 100 >= t) n++; return n; },
    spamBetween(a, b) { let n = 0; for (const r of this.results.values()) { const s = r.spam * 100; if (s >= a && s < b) n++; } return n; },

    actionCounts() {
      const m = new Map();
      for (const r of this.results.values()) {
        const k = r.action.label;
        if (!m.has(k)) m.set(k, { name: k, color: colorForAction(r.action), n: 0 });
        m.get(k).n++;
      }
      return [...m.values()].sort((a, b) => b.n - a.n);
    },
    topActions() {
      const all = this.actionCounts();
      const head = all.slice(0, 8).map(a => ({ ...a, pc: pct(a.n, this.done) }));
      const restN = all.slice(8).reduce((s, a) => s + a.n, 0);
      if (restN) head.push({ name: `${all.length - 8} more folders`, color: '#3d4b5c', n: restN, pc: pct(restN, this.done) });
      return head;
    },
    donut() {
      const rows = this.actionCounts();
      const total = rows.reduce((a, r) => a + r.n, 0) || 1;
      let acc = 0;
      return rows.map(r => {
        const len = (r.n / total) * 100;
        const seg = { name: r.name, color: r.color, dash: `${len.toFixed(2)} ${(100 - len).toFixed(2)}`, offset: (-acc).toFixed(2) };
        acc += len;
        return seg;
      });
    },

    visible() {
      const q = this.search.trim().toLowerCase();
      const out = [];
      this.emails.forEach((e, idx) => {
        if (q && !`${e.subject} ${e.fromName} ${e.fromAddress}`.toLowerCase().includes(q)) return;
        const r = this.results.get(e.id);
        const row = { i: idx + 1, e, r, when: fmtWhen(e.received), from: e.fromName || e.fromAddress };
        if (r) {
          const c = colorForAction(r.action);
          row.action = r.action.label; row.color = c;
          row.tagStyle = `background:${c}1f;border-color:${c}55;color:#e6edf3`;
          row.spam = Math.round(r.spam * 100);
          row.spamColor = row.spam >= 50 ? OP_COLORS.delete : row.spam >= 20 ? '#e8b23a' : '#3fbf7f';
          row.conf = Math.round(r.confidence * 100);
        }
        out.push(row);
      });
      return out;
    },

    // ---------------------------------------------------------------- the run
    clear() {
      this.results = new Map(); this.done = 0; this.errors = 0; this.elapsed = 0;
      this.latencies = []; this.tokensIn = 0; this.tokensOut = 0; this.confSum = 0; this.spamSum = 0;
      M.redraw();
    },

    async run() {
      if (this.running) return;
      this.clear();
      this.running = true;
      this.startedAt = Date.now();
      const timer = setInterval(() => { this.elapsed = Date.now() - this.startedAt; M.redraw(); }, 250);

      let cursor = 0;
      const groups = groupsFor();
      const worker = async () => {
        while (cursor < this.emails.length && this.running) {
          const e = this.emails[cursor++];
          try {
            const res = await (await fetch('/api/classify', {
              method: 'POST', headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ premise: premiseFor(e), groups }),
            })).json();
            if (res.error) throw new Error(res.error);

            const a = res.answers.action;
            this.results.set(e.id, {
              action: ACTIONS[a.index],
              confidence: a.confidence,          // share of the mass across every operation
              // Strongest of the observable spam signals, not an argmax over classes.
              spam: Math.max(...res.answers.spam.probs),
            });
            this.confSum += a.confidence;
            this.latencies.push(res.ms);
            this.tokensIn += res.tokensIn || 0; this.tokensOut += res.tokensOut || 0;
          } catch { this.errors++; }
          this.done++;
        }
      };
      await Promise.all(Array.from({ length: WORKERS }, worker));

      clearInterval(timer);
      this.elapsed = Date.now() - this.startedAt;
      this.running = false;
      M.redraw();
    },
  };
}
