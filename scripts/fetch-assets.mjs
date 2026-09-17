#!/usr/bin/env bun
/* Pull cook2.html + its art from mikesmullin/vibe-arcade into web/game/, and inject the agent bridge.
 *
 * The art is ~16 MB of atlas pages, so it is fetched at setup rather than committed -- same treatment as
 * the 8.5 GB checkpoint. Only our own agent-hook.js lives in git.
 *
 *   bun run fetch-assets
 */
import { mkdir, writeFile, readFile, stat } from 'node:fs/promises';
import { dirname, join } from 'node:path';

const RAW = 'https://raw.githubusercontent.com/mikesmullin/vibe-arcade/main';
const OUT = join(import.meta.dir, '..', 'web', 'game');

// cook2.html loads the atlas pages by preference and falls back to per-item PNGs; the atlas covers all
// 80 frames the game uses, so the pages plus the two manifests are enough.
const FILES = [
  'cook2.html',
  'assets/assets.json',
  'assets/daw.mjs',
  'assets/art/atlas.json',
  ...Array.from({ length: 6 }, (_, i) => `assets/art/atlas-${i}.png`),
];

const exists = (p) => stat(p).then(() => true, () => false);

async function grab(path) {
  const dest = join(OUT, path);
  if (await exists(dest)) return { path, skipped: true };
  const res = await fetch(`${RAW}/${path}`);
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  const buf = Buffer.from(await res.arrayBuffer());
  await mkdir(dirname(dest), { recursive: true });
  await writeFile(dest, buf);
  return { path, bytes: buf.length };
}

/** Inject our hook script and one bridge line that hands the game's module scope to it. */
async function patch() {
  const p = join(OUT, 'cook2.html');
  let html = await readFile(p, 'utf8');
  const MARK = '/* ---- openjev agent bridge';
  const at = html.indexOf(MARK);
  if (at >= 0) html = html.slice(0, at) + html.slice(html.indexOf('</script>', at));   // re-patch cleanly

  // Hand the module's private scope to the hook on the last line of that module. `selected` is a `let`,
  // so it is exposed as a getter -- a plain copy would freeze at whatever was held when this ran.
  // The hook is imported dynamically with a version query: browsers hold classic scripts and ES modules
  // across reloads even under no-store, and a stale hook is invisible (it just keeps the old behaviour).
  const bridge = `
/* ---- openjev agent bridge (injected by scripts/fetch-assets.mjs) ---- */
try {
  const { bind } = await import('./agent-hook.js?v=' + Date.now());
  bind({
    G, interactives, stations, customers, piles, DISH,
    select, deselect, tryDrop, startLevel, showMenu,
    Item, Grill, Tray, PlateRack, SodaMachine, Fryer, Trash, Customer,
    get selected() { return selected; },
  });
} catch (e) { console.error('[openjev] bridge failed', e); }
`;
  const i = html.lastIndexOf('</script>');
  html = html.slice(0, i) + bridge + html.slice(i);
  await writeFile(p, html);
  return 'patched';
}

const results = [];
for (const f of FILES) {
  try { results.push(await grab(f)); }
  catch (e) { console.error(`  ! ${e.message}`); process.exitCode = 1; }
}
const got = results.filter(r => !r.skipped);
console.log(`fetched ${got.length} file(s), ${(got.reduce((a, r) => a + r.bytes, 0) / 1e6).toFixed(1)} MB` +
  (results.length - got.length ? `, ${results.length - got.length} already present` : ''));
console.log('cook2.html:', await patch());
