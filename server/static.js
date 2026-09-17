#!/usr/bin/env bun
// Minimal static server for the game + agent page. No framework: the agent loop runs in the browser and
// talks to the Python scorer directly, so there is nothing for a backend to do but serve files and proxy.
const ROOT = new URL('../web/', import.meta.url).pathname;
const MODEL = process.env.MODEL_URL ?? 'http://127.0.0.1:8750';
const PORT = Number(process.env.PORT ?? 8734);

Bun.serve({
  port: PORT, hostname: '127.0.0.1',
  async fetch(req) {
    const url = new URL(req.url);
    if (url.pathname === '/api/decide' && req.method === 'POST') {
      const r = await fetch(`${MODEL}/score`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: await req.text() });
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
console.log(`\n  http://127.0.0.1:${PORT}/game/mars.html\n  model proxy -> ${MODEL}\n`);
