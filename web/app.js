/* Inbox × openjev — four classification questions per email, answered by the NLI cross-encoder.
 *
 * Replaces the generative LLM pass in agl-agents/personal-email. No tokens are generated: each question is
 * a group of mutually exclusive statements, and the answer is whichever the model judges most entailed by
 * the email. Four questions ride in one batched forward pass.
 */
import { QUESTIONS, KEYS, premiseFor, groupsFor, CATEGORY_COLORS, URGENCY_COLORS } from './questions.js';

const WORKERS = 3;        // in-flight requests; the GPU serialises, so this only hides round-trip time

const pct = (n, d) => (d ? Math.round((100 * n) / d) : 0);
const fmtWhen = (iso) => {
  if (!iso) return '—';
  const d = new Date(iso);
  return d.toLocaleString('en-US', { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' });
};

export function Inbox(M) {
  return {
    emails: [], results: new Map(), search: '',
    running: false, done: 0, errors: 0, startedAt: 0, elapsed: 0,
    latencies: [], tokensIn: 0, tokensOut: 0, confSum: 0, confN: 0,
    modelInfo: '', error: '', booted: false,

    template: `
<div>
<header>
  <div class="logo"></div>
  <div>
    <h1>Inbox &times; openjev</h1>
    <div class="meta"><span x-text="emails.length"></span> emails loaded &middot; model
      <code x-text="modelName()"></code></div>
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
    <div class="k">Categories</div>
    <div class="donutwrap">
      <svg width="104" height="104" viewBox="0 0 42 42">
        <circle cx="21" cy="21" r="15.9" fill="none" stroke="#0f151c" stroke-width="6"></circle>
        <template x-for="a in donut()" :key="a.name">
          <circle cx="21" cy="21" r="15.9" fill="none" :stroke="a.color" stroke-width="6"
                  :stroke-dasharray="a.dash" :stroke-dashoffset="a.offset" transform="rotate(-90 21 21)"></circle>
        </template>
        <text x="21" y="20.5" text-anchor="middle" fill="#e6edf3" font-size="7" font-weight="600"
              x-text="done"></text>
        <text x="21" y="26" text-anchor="middle" fill="#5d6b7a" font-size="3.2">classified</text>
      </svg>
      <div class="cats">
        <template x-for="c in categoryRows()" :key="c.name">
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
      <div><b style="color:var(--green)" x-text="signal('reply','Yes')"></b>
        <span>needs reply</span><span x-text="pct(signal('reply','Yes'), done) + '% of classified'"></span></div>
      <div><b style="color:var(--blue)" x-text="signal('sender','Human')"></b>
        <span>written by a human</span><span x-text="pct(signal('sender','Human'), done) + '% of classified'"></span></div>
      <div><b style="color:var(--red)" x-text="signal('urgency','Today')"></b>
        <span>needs action today</span><span x-text="pct(signal('urgency','Today'), done) + '% of classified'"></span></div>
    </div>
    <template x-for="u in urgencyRows()" :key="u.name">
      <div class="ubar">
        <span class="lbl" x-text="u.name"></span>
        <span class="track"><i :style="'width:' + u.pc + '%;background:' + u.color"></i></span>
        <span class="n" x-text="u.n"></span>
      </div>
    </template>
  </div>

  <div class="card">
    <div class="k">Model</div>
    <div class="sig" style="grid-template-columns:repeat(2,1fr)">
      <div><b x-text="avgConf() + '%'"></b><span>avg confidence</span></div>
      <div><b x-text="avgLatency() + ' ms'"></b><span>avg latency</span></div>
      <div><b x-text="tokensIn.toLocaleString()"></b><span>tokens in</span></div>
      <div><b x-text="tokensOut.toLocaleString()"></b><span>tokens out</span></div>
    </div>
    <div class="note" x-text="modelNote()"></div>
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
        <th>Category</th><th>Reply</th><th>Sender</th><th>Urgency</th><th style="text-align:right">Conf</th>
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
                <span class="tag" :style="row.catStyle"><i :style="'background:' + row.catColor"></i><span x-text="row.cat"></span></span>
              </template>
              <template x-if="!row.r"><span class="muted">—</span></template>
            </td>
            <td>
              <template x-if="row.r"><span class="ans"><span x-text="row.reply"></span><span class="c" x-text="row.replyC"></span></span></template>
              <template x-if="!row.r"><span class="muted">—</span></template>
            </td>
            <td>
              <template x-if="row.r"><span class="ans"><span x-text="row.sender"></span><span class="c" x-text="row.senderC"></span></span></template>
              <template x-if="!row.r"><span class="muted">—</span></template>
            </td>
            <td>
              <template x-if="row.r">
                <span class="ans"><span class="ubars"><template x-for="b in row.ubars" :key="b.i"><i :style="b.style"></i></template></span><span :style="'color:' + row.urgColor" x-text="row.urgency"></span></span>
              </template>
              <template x-if="!row.r"><span class="muted">—</span></template>
            </td>
            <td class="conf" x-text="row.conf"></td>
          </tr>
        </template>
      </tbody>
    </table>
  </div>
</div>
</div>`,

    pct,

    // ---------------------------------------------------------------- lifecycle
    async init() {
      try {
        const h = await (await fetch('/api/health')).json();
        this.modelInfo = h.model ? h.model.device : '';
        if (!h.model) this.error = 'model offline — run: bun run model';
      } catch { this.error = 'cannot reach the server'; }

      try {
        const r = await (await fetch('/api/emails?limit=500')).json();
        this.emails = r.emails;
        this.booted = !this.error;
      } catch (e) { this.error = 'failed to load emails: ' + e.message; }
      M.redraw();
    },

    modelName: () => 'qwen3.5-4b-nli',
    modelNote() {
      return `Four questions per email in one request: ${KEYS.map(k => QUESTIONS[k].label.toLowerCase()).join(', ')}. ` +
             `The model generates nothing — it scores ${Object.values(groupsFor()).flat().length} statements ` +
             `per email and takes the argmax of each group, so tokens out is structurally zero.`;
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
    rate() {
      const s = this.elapsed / 1000;
      return s > 0.3 ? (this.done / s).toFixed(1) : '0.0';
    },
    avgLatency() { return this.latencies.length ? Math.round(this.latencies.reduce((a, b) => a + b, 0) / this.latencies.length) : 0; },
    avgConf() { return this.confN ? Math.round((100 * this.confSum) / this.confN) : 0; },

    counts(key) {
      const m = new Map();
      for (const r of this.results.values()) m.set(r[key].label, (m.get(r[key].label) || 0) + 1);
      return m;
    },
    signal(key, label) { return this.counts(key).get(label) || 0; },

    categoryRows() {
      const m = this.counts('category');
      return QUESTIONS.category.options.map(([name]) => ({
        name, color: CATEGORY_COLORS[name], n: m.get(name) || 0, pc: pct(m.get(name) || 0, this.done),
      }));
    },
    donut() {
      const rows = this.categoryRows().filter(r => r.n);
      const total = rows.reduce((a, r) => a + r.n, 0) || 1;
      let acc = 0;
      return rows.map(r => {
        const len = (r.n / total) * 100;
        const seg = { name: r.name, color: r.color, dash: `${len.toFixed(2)} ${(100 - len).toFixed(2)}`, offset: (-acc).toFixed(2) };
        acc += len;
        return seg;
      });
    },
    urgencyRows() {
      const m = this.counts('urgency');
      const max = Math.max(1, ...[...m.values()]);
      return QUESTIONS.urgency.options.map(([name]) => ({
        name, color: URGENCY_COLORS[name], n: m.get(name) || 0, pc: ((m.get(name) || 0) / max) * 100,
      }));
    },

    visible() {
      const q = this.search.trim().toLowerCase();
      const out = [];
      this.emails.forEach((e, idx) => {
        if (q && !(`${e.subject} ${e.fromName} ${e.fromAddress}`.toLowerCase().includes(q))) return;
        const r = this.results.get(e.id);
        const row = { i: idx + 1, e, r, when: fmtWhen(e.received), from: e.fromName || e.fromAddress };
        if (r) {
          const c = r.category.label, u = r.urgency.label;
          row.cat = c; row.catColor = CATEGORY_COLORS[c];
          row.catStyle = `background:${CATEGORY_COLORS[c]}1f;border-color:${CATEGORY_COLORS[c]}55;color:#e6edf3`;
          row.reply = r.reply.label; row.replyC = Math.round(r.reply.confidence * 100);
          row.sender = r.sender.label; row.senderC = Math.round(r.sender.confidence * 100);
          row.urgency = u; row.urgColor = URGENCY_COLORS[u];
          const lit = u === 'Today' ? 3 : u === 'This week' ? 2 : 1;
          row.ubars = [0, 1, 2].map(i => ({ i, style: i < lit ? `background:${URGENCY_COLORS[u]}` : '' }));
          row.conf = Math.round(r.mean * 100);
        }
        out.push(row);
      });
      return out;
    },

    // ---------------------------------------------------------------- the run
    clear() {
      this.results = new Map(); this.done = 0; this.errors = 0; this.elapsed = 0;
      this.latencies = []; this.tokensIn = 0; this.tokensOut = 0; this.confSum = 0; this.confN = 0;
      M.redraw();
    },

    async run() {
      if (this.running) return;
      this.clear();
      this.running = true;
      this.startedAt = Date.now();
      const timer = setInterval(() => { this.elapsed = Date.now() - this.startedAt; M.redraw(); }, 250);

      // A simple worker pool over a shared cursor: the GPU serialises anyway, so this exists to keep a
      // request in flight while another is being decoded, not to get real parallelism.
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

            const out = { mean: 0 };
            let sum = 0;
            for (const k of KEYS) {
              const a = res.answers[k];
              out[k] = { label: QUESTIONS[k].options[a.index][0], confidence: a.confidence, p: a.p };
              sum += a.confidence;
            }
            out.mean = sum / KEYS.length;
            this.results.set(e.id, out);
            this.confSum += out.mean; this.confN++;
            this.latencies.push(res.ms);
            this.tokensIn += res.tokensIn || 0; this.tokensOut += res.tokensOut || 0;
          } catch (err) {
            this.errors++;
          }
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
