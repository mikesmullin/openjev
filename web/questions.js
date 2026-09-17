/* What the model is asked about each email.
 *
 * Two questions, both scored in one batched forward pass:
 *
 *   action  — one flat choice over every operation the operator can take, including one option per Gmail
 *             folder. Mutually exclusive, so the argmax is the recommendation and its share of the group's
 *             mass is the confidence.
 *   spam    — a single statement whose P(entailment) is reported directly as a percentage. It is not a
 *             classification: nothing is chosen, the probability itself is the answer.
 *
 * Why the action question is flat rather than "operation first, then folder":
 *
 * A two-stage design needs an intermediate option meaning "this belongs in SOME folder", and there is no
 * way to phrase that which is not vague. Vague hypotheses have beaten specific ones repeatedly here -- a
 * catch-all category sentence once took 63% of the inbox -- because an NLI head scores whether a sentence
 * is supported by the text, and a loose sentence is supported by almost anything. A flat list has no such
 * option: every choice names something concrete. It also costs less. The premise is re-encoded once per
 * statement and dominates the bill, so two stages would pay for it twice.
 *
 * Every sentence is a POSITIVE, checkable assertion about the email. Measured on this model, statements
 * built on "nothing", "nobody" or "never" score near zero and can never win.
 *
 * Personal taxonomy lives in config.yaml (gitignored) — copy config.yaml.example there and edit.
 * The server exposes it at /api/config; the defaults below mirror the example so the page
 * still runs before the fetch resolves.
 */

/* Generic placeholder folders. Replace via config.yaml with your own Gmail labels + one
   positive, checkable sentence per folder describing what belongs there. */
export let FOLDERS = [
  ['Receipts',    'This is a receipt, invoice, or order confirmation for something the recipient bought, with an amount stated.'],
  ['Statements',  'This is a periodic account statement from a bank or credit card.'],
  ['Travel',      'This is about a trip: a flight, hotel, reservation, or ticket.'],
  ['Newsletters', 'This is a periodical the recipient subscribed to and reads at leisure: a digest or roundup.'],
  ['Work',        'This addresses the recipient as an employee, about internal company matters.'],
];

/* Operation set: delete, archive, skip, move to <Folder>. */
let OPERATIONS = [
  ['delete',  'This is a promotional advertisement from a shop, offering a sale, a discount, a deal or reward points.'],
  ['skip',    'This needs the operator to read it and decide personally, because it concerns something only they can judge.'],
  ['archive', 'This is ordinary correspondence that has served its purpose and just needs filing away out of the inbox.'],
];

function buildActions() {
  return [
    ...OPERATIONS.map(([op, hyp]) => ({ op, folder: null, label: op, hypothesis: hyp })),
    ...FOLDERS.map(([folder, hyp]) => ({ op: 'move', folder, label: `move to ${folder}`, hypothesis: hyp })),
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
export let SPAM_SIGNALS = [
  'The message promises the reader a large sum of money and asks them to click a link or confirm personal details.',
  'The message uses urgent capitalised excitement and asks the reader to act immediately on an unexpected windfall.',
  'The sender is unknown to the recipient and the message asks for bank, password or payment information.',
];

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
      .filter(f => f && f.name && f.description)
      .map(f => [String(f.name), String(f.description)]);
  }
  if (Array.isArray(cfg.operations) && cfg.operations.length) {
    OPERATIONS = cfg.operations
      .filter(o => o && o.op && o.description)
      .map(o => [String(o.op), String(o.description)]);
  }
  if (Array.isArray(cfg.spam_signals) && cfg.spam_signals.length) {
    SPAM_SIGNALS = cfg.spam_signals.map(String);
  }
  ACTIONS = buildActions();
}

export const groupsFor = () => ({
  action: ACTIONS.map(a => a.hypothesis),
  spam: SPAM_SIGNALS,
});

/** The email as prose. The model was trained on sentence pairs, so it gets a readable message, not JSON.
 *
 * The body is capped hard: the premise is re-encoded once per statement and cost is linear in tokens.
 * Measured for a 16-statement request -- 0 chars 61 ms, 240 chars 110 ms, 900 chars 230 ms. Subject, sender
 * and opening lines carry nearly all the signal, so a long marketing footer is paid for many times over
 * and buys almost nothing. */
export function premiseFor(e) {
  const when = e.received ? new Date(e.received).toUTCString().replace(/ GMT$/, '') : 'an unknown date';
  return [
    `An email received on ${when}.`,
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
