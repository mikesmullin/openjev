/* What the model is asked about each email.
 *
 * Two groups, both answered in ONE forward pass:
 *
 *   action  — a flat single-label choice over every operation, including one per Gmail folder.
 *   spam    — a two-class question, so P(spam) is a real probability rather than a spread.
 *
 * The flat-not-two-stage argument from the openjev branch still holds: an intermediate option meaning
 * "belongs in SOME folder" cannot be phrased without being vague, and vague options win. What changed
 * is the *form* of each option.
 *
 * openjev scored checkable assertions ("This is a receipt, invoice, or order confirmation for something
 * the recipient bought, with an amount stated."). GLiNER is a classification encoder, and measured on
 * this inbox those sentences are actively worse than useless:
 *
 *              labels as descriptions   labels as class names   labels as short phrases
 *   speed          236 ms/email              86 ms/email             89 ms/email
 *   behaviour      collapses to "skip"       discriminates           discriminates, fewer errors
 *                  on 7 of 8 emails
 *
 * So each folder and operation carries a short `label` -- two or three words naming the class, which is
 * what a classifier matches against. Bare folder names work but confuse neighbours: a Chase card charge
 * filed as `Income` until "Expenses" became "purchase receipt". Descriptions are kept in config.yaml
 * because they document the taxonomy for a human, and because the openjev branch still needs them.
 */

/* Generic placeholder folders. Replace via config.yaml with your own Gmail labels + one
   positive, checkable sentence per folder describing what belongs there. */
/* [name, label] -- label is what the model sees. Replaced from config.yaml via /api/config. */
export let FOLDERS = [
  ['Receipts',    'purchase receipt'],
  ['Statements',  'bank statement'],
  ['Travel',      'trip booking'],
  ['Newsletters', 'newsletter'],
  ['Work',        'internal company mail'],
];

/* Operation set: delete, archive, skip, move to <Folder>. */
let OPERATIONS = [
  ['delete',  'advertisement'],
  ['skip',    'needs a personal decision'],
  ['archive', 'routine correspondence'],
];

function buildActions() {
  return [
    ...OPERATIONS.map(([op, cls]) => ({ op, folder: null, label: op, hypothesis: cls })),
    ...FOLDERS.map(([folder, cls]) => ({ op: 'move', folder, label: `move to ${folder}`, hypothesis: cls })),
  ];
}

export let ACTIONS = buildActions();

/* Not a classification: several signals are scored and the highest is reported as the percentage.
 *
 * The big lesson here. Asking about INTENT does not work at all -- "this is a scam trying to trick the
 * reader" scored 0.060 on a blatant prize scam, and no rephrasing of the intent helped ("this email is
 * spam" 0.069, "the sender is trying to defraud the reader" 0.075). An NLI head judges whether a claim is
 * supported by the text, and a scam email is precisely a text that conceals its intent. Nothing in it
 * supports the sentence "this is a scam".
 *
 * Describing what is OBSERVABLE on the page instead took the same email from 0.060 to 0.975. Scams differ
 * in shape, though -- the prize signal misses phishing entirely -- so a few observable patterns are scored
 * and the strongest wins:
 *
 *              prize scam   receipt   newsletter   phishing
 *   intent          0.060     0.006        0.004      0.008
 *   windfall        0.975     0.012        0.003      0.005
 *   urgency         0.758     0.005        0.003      0.005
 *   credentials     0.483     0.050        0.001      0.238
 */
/* Two classes, not a bank of observable signals.
 *
 * openjev could not ask about intent -- "this is a scam" scored 0.060 on a blatant prize scam, because
 * an NLI head judges whether a claim is supported by the text and a scam conceals its intent. It had to
 * score observable patterns (windfall, urgency, credentials) and take the strongest.
 *
 * A classification encoder has no such problem: spam is a canonical text class, and asking for it
 * directly works. Measured on this inbox: a marketing blast 0.98, a Chase charge alert 0.25, a
 * community newsletter 0.00. Two classes also means the confidence IS the probability. */
export let SPAM_SIGNALS = ['spam', 'legitimate email'];

/* Fetch the operator's real taxonomy from the server (reads config.yaml, falls back to
   config.yaml.example). Rebuilds FOLDERS / ACTIONS / SPAM_SIGNALS in place so existing
   imports keep working; call once at startup before scoring. */
export async function loadConfig() {
  let cfg;
  try {
    const r = await fetch('/api/config');
    if (!r.ok) return;
    cfg = await r.json();
  } catch { return; }
  if (Array.isArray(cfg.folders) && cfg.folders.length) {
    FOLDERS = cfg.folders
      .filter(f => f && f.name)
      .map(f => [String(f.name), String(f.label || f.name)]);
  }
  if (Array.isArray(cfg.operations) && cfg.operations.length) {
    OPERATIONS = cfg.operations
      .filter(o => o && o.op)
      .map(o => [String(o.op), String(o.label || o.op)]);
  }
  // spam stays a fixed two-class question; config.yaml's openjev-era signal list is ignored.
  ACTIONS = buildActions();
}

export const groupsFor = () => ({
  action: ACTIONS.map(a => a.hypothesis),
  spam: SPAM_SIGNALS,
});

/** The email as prose, with nothing in it that is the same on every email.
 *
 * The received date used to lead this string ("An email received on Thu, 19 Jun 2026 18:28:00."). It
 * is not signal -- the table already shows the date, and no folder in the taxonomy is about *when*
 * something arrived -- and measured on four Chase card alerts it flipped every one of them:
 *
 *   From/Subject/Body only          -> Expenses  (0.45, 0.35, 0.34, 0.30)
 *   with the date line prepended    -> archive   (0.39, 0.38, 0.36, 0.47)
 *
 * A date stamp reads as routine correspondence, and it is present on all 500 emails, so it dragged the
 * whole inbox toward the catch-all: `archive` took 61 percent of the run that included it. This is the
 * same failure the other branches kept hitting -- text that is always there quietly deciding the
 * answer -- and it is the reason `archive` looked like a bad label when the label was fine.
 *
 * Body is capped at 240 characters: subject and sender carry nearly all the signal, and unlike the NLI
 * branch (where the premise was re-encoded once per hypothesis) the cost here is one encode either way.
 */
export function premiseFor(e) {
  return [
    `From: ${e.fromName ? `${e.fromName} <${e.fromAddress}>` : e.fromAddress}.`,
    `Subject: ${e.subject}.`,
    `Body: ${(e.body || e.snippet || '').slice(0, 240)}`,
  ].join('\n');
}


export const OP_COLORS = { delete: '#ef4470', skip: '#e8b23a', archive: '#8593a6', move: '#4aa3df' };

/* A stable colour per destination, hue derived from the name so it never needs a lookup table. */
export function colorForAction(a) {
  if (!a) return OP_COLORS.archive;
  if (a.op !== 'move') return OP_COLORS[a.op];
  let h = 0;
  for (let i = 0; i < a.folder.length; i++) h = (h * 31 + a.folder.charCodeAt(i)) >>> 0;
  return `hsl(${h % 360} 62% 62%)`;
}
