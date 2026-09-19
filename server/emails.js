/* Load the personal-email corpus.
 *
 * Entities are YAML files whose `origin.raw` is itself a YAML document (the stored Gmail payload), so this
 * parses twice. The body is usually HTML; it is flattened to text here rather than in the browser, because
 * the model wants prose and the table wants a snippet.
 */
import { readdirSync, readFileSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import * as yaml from 'js-yaml';

// Point EMAIL_DB at a directory containing `entities/` and/or `_archive/` of entity YAML files.
// No corpus ships with this repo -- it is someone's mail.
const ROOT = process.env.EMAIL_DB ?? './data/emails';

const stripHtml = (html) => html
  .replace(/<style[\s\S]*?<\/style>/gi, ' ')
  .replace(/<script[\s\S]*?<\/script>/gi, ' ')
  .replace(/<[^>]+>/g, ' ')
  .replace(/&nbsp;/g, ' ').replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>')
  .replace(/&#\d+;/g, ' ')
  .replace(/\s+/g, ' ')
  .trim();

function parseEntity(path, id) {
  let doc;
  try { doc = yaml.load(readFileSync(path, 'utf8')); } catch { return null; }
  const raw = doc?.origin?.raw;
  if (!raw) return null;

  let m;
  try { m = typeof raw === 'string' ? yaml.load(raw) : raw; } catch { return null; }
  if (!m || !m.subject) return null;

  const body = m.body?.content ?? '';
  const text = m.body?.contentType === 'html' ? stripHtml(body) : String(body).replace(/\s+/g, ' ').trim();

  return {
    id,
    subject: String(m.subject ?? '').trim(),
    fromName: String(m.from?.name ?? '').trim(),
    fromAddress: String(m.from?.address ?? '').trim(),
    received: m.receivedDateTime ?? null,
    snippet: String(m.snippet ?? '').replace(/\s+/g, ' ').trim(),
    // Enough for the model to judge tone and intent without paying for a whole marketing email.
    body: (text || String(m.snippet ?? '')).slice(0, 1200),
    labels: m.labelIds ?? [],
  };
}

let CACHE = null;

export function loadEmails(limit = 2000) {
  if (CACHE) return CACHE.slice(0, limit);
  const out = [];
  for (const dir of [join(ROOT, 'entities'), join(ROOT, '_archive')]) {
    if (!existsSync(dir)) continue;
    for (const f of readdirSync(dir)) {
      if (!f.endsWith('.yaml')) continue;
      const e = parseEntity(join(dir, f), f.replace(/\.yaml$/, ''));
      if (e) out.push(e);
    }
  }
  // Newest first, which is how an inbox reads.
  out.sort((a, b) => String(b.received ?? '').localeCompare(String(a.received ?? '')));
  CACHE = out;
  return out.slice(0, limit);
}
