#!/usr/bin/env bun
// Minimal static server for the game + agent page. No framework: the agent loop runs in the browser and
// talks to the simple-jev adapter through this proxy, so there is nothing for a backend to do but serve
// files and forward. Game logic lives in the page; classifier logic lives in server/jev_server.py.
const ROOT = new URL('../web/', import.meta.url).pathname;
const MODEL = process.env.MODEL_URL ?? 'http://127.0.0.1:8750';
const PORT = Number(process.env.PORT ?? 8734);

Bun.serve({
  port: PORT, hostname: '127.0.0.1',
  // A decision is several questions against a 27B model; the default 10 s would abort a cold first call.
  idleTimeout: 120,
  async fetch(req) {
    const url = new URL(req.url);
    if (url.pathname === '/api/decide' && req.method === 'POST') {
      const r = await fetch(`${MODEL}/v1/classifier`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: await req.text() });
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
console.log(`\n  http://127.0.0.1:${PORT}/\n  classifier proxy -> ${MODEL}\n`);
