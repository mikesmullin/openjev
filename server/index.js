#!/usr/bin/env bun
/* Bun + express. Serves the page and the game, and proxies scoring to the one Python process.
 *
 * The agent loop itself lives in the browser (web/app.js): the game is there, so reading its state and
 * applying an action is a direct call rather than a round trip. This server only has to hand the model a
 * premise and a list of hypotheses.
 */
import express from 'express';
import { join } from 'node:path';

const PORT = Number(process.env.PORT ?? 8733);
const MODEL = process.env.MODEL_URL ?? 'http://127.0.0.1:8750';
const WEB = join(import.meta.dir, '..', 'web');

const app = express();
app.use(express.json({ limit: '1mb' }));
app.use(express.static(WEB, { etag: true, setHeaders: (r, p) => r.setHeader('Cache-Control', p.endsWith('.png') ? 'max-age=3600' : 'no-store') }));

/** Score one premise against N hypotheses. Returns the winning index plus every probability, so the page
 *  can show the full distribution the way the recorded Doom HUD did. */
app.post('/api/decide', async (req, res) => {
  const { premise, hypotheses } = req.body ?? {};
  if (!premise || !Array.isArray(hypotheses) || !hypotheses.length) {
    return res.status(400).json({ error: 'premise and non-empty hypotheses required' });
  }
  const t0 = performance.now();
  try {
    const r = await fetch(`${MODEL}/score`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ premise, hypotheses }),
    });
    const j = await r.json();
    if (!r.ok) return res.status(502).json({ error: j.error ?? `model HTTP ${r.status}` });
    res.json({ ...j, roundTripMs: performance.now() - t0 });
  } catch (e) {
    res.status(503).json({ error: `model server unreachable at ${MODEL} — is \`bun run model\` up? (${e.message})` });
  }
});

app.get('/api/health', async (_req, res) => {
  try {
    const r = await fetch(`${MODEL}/health`);
    res.json({ web: true, model: await r.json() });
  } catch (e) {
    res.status(503).json({ web: true, model: null, error: e.message });
  }
});

app.listen(PORT, '127.0.0.1', () => {
  console.log(`\n  openjev cook  ->  http://127.0.0.1:${PORT}/`);
  console.log(`  model proxy   ->  ${MODEL}\n`);
});
