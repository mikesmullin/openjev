#!/usr/bin/env bun
/* Static files, the email corpus, and a proxy to the one Python process. No classification logic here --
   the agent loop runs in the browser so the table can fill in as answers arrive. */
import { loadEmails } from './emails.js';

const ROOT = new URL('../web/', import.meta.url).pathname;
const MODEL = process.env.MODEL_URL ?? 'http://127.0.0.1:8750';
const PORT = Number(process.env.PORT ?? 8735);

Bun.serve({
  port: PORT, hostname: '127.0.0.1',
  async fetch(req) {
    const url = new URL(req.url);

    if (url.pathname === '/api/emails') {
      const limit = Number(url.searchParams.get('limit') ?? 500);
      const t0 = performance.now();
      const emails = loadEmails(limit);
      return Response.json({ emails, ms: performance.now() - t0 });
    }
    if (url.pathname === '/api/classify' && req.method === 'POST') {
      const r = await fetch(`${MODEL}/classify`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: await req.text(),
      });
      return new Response(await r.text(), { status: r.status, headers: { 'Content-Type': 'application/json' } });
    }
    if (url.pathname === '/api/health') {
      try { return Response.json({ model: await (await fetch(`${MODEL}/health`)).json() }); }
      catch (e) { return Response.json({ model: null, error: String(e) }, { status: 503 }); }
    }

    const p = url.pathname === '/' ? '/index.html' : url.pathname;
    const f = Bun.file(ROOT + p.replace(/^\/+/, ''));
    return (await f.exists())
      ? new Response(f, { headers: { 'Cache-Control': 'no-store' } })
      : new Response('not found', { status: 404 });
  },
});
console.log(`\n  inbox x openjev  ->  http://127.0.0.1:${PORT}/\n  model proxy      ->  ${MODEL}\n`);
