/* The questions asked of every email, and the sentences that stand for each answer.
 *
 * Each question is a group of mutually exclusive hypotheses. The model scores all of them against the same
 * premise in one batched forward pass; the argmax within a group is the answer, and its share of the
 * group's mass is the confidence.
 *
 * The wording follows what the games taught: a hypothesis is a STATEMENT ABOUT THE EMAIL that would be
 * true if that answer were right -- never a bare label. "Newsletter" is a word; "This is a newsletter or
 * digest the recipient subscribed to, sent on a regular schedule" is something that can be judged true or
 * false against the text.
 */

export const QUESTIONS = {
  /* Every sentence has to be true ONLY when its answer is right. The first draft failed on exactly the
     trap this project keeps hitting: "This does not fit any usual category of email" is vaguely true of
     almost anything, so Other swallowed 63% of the inbox. Catch-alls must describe something concrete,
     and the mundane categories have to name the artefacts that actually appear in the text. */
  category: {
    label: 'Category',
    options: [
      ['Brand deal',    'The sender offers to pay the recipient, or to send free product, in exchange for promotion or a collaboration.'],
      ['Cold pitch',    'A salesperson the recipient has never met is introducing a product and asking for a meeting or a call.'],
      ['Personal',      'A private message from one individual to another, about their own lives, signed by the person who wrote it.'],
      ['Newsletter',    'A regular publication sent to subscribers: articles, a digest, or a roundup of stories.'],
      ['Marketing',     'A shop or brand advertising a sale, discount, coupon or new product, addressed to customers generally.'],
      ['Receipt',       'A record of a specific transaction: an amount of money charged, paid, ordered or refunded, with the figure stated.'],
      ['Service alert', 'An automated warning about an account: a security event, a login, an expiry, a failure or a status change that may need attention.'],
      ['Notification',  'An automated note that routine activity happened in an app: a comment, a reminder, an update, a calendar event.'],
      ['Other',         'A bare administrative message such as a delivery failure, a test message, or an automated bounce.'],
    ],
  },
  /* Phrase every option as a POSITIVE, checkable assertion. Measured: sentences built on "nothing",
     "nobody" or "no one" barely entail at all -- "Nobody is waiting for an answer" scored 0.001-0.085
     across a personal note, a receipt, an ad and a security alert, while the positively-phrased "There is
     a deadline..." hit 0.669 on the alert it actually described. An NLI head judges whether a claim is
     supported by the text, and a universal negative has nothing in the text to support it. */
  reply: {
    label: 'Reply',
    options: [
      // Picked by A/B against five hand-labelled emails (marketing, receipt, security alert, personal
      // note, cold pitch). This pair got 5/5; two other phrasings got 3/5 and 4/5, both by calling the
      // security alert and the ad "needs a reply".
      ['Yes', 'The sender personally addresses the recipient and asks them to respond.'],
      ['No',  'This message is broadcast from a brand or a system, and the recipient is simply a reader of it.'],
    ],
  },
  sender: {
    label: 'Sender',
    options: [
      ['Human', 'One person typed this out for this recipient in particular, in their own voice, and signed it themselves.'],
      ['Auto',  'A template filled in by software and sent to many recipients at once, with no person composing it.'],
    ],
  },
  /* Same fix here: "Nothing is asked of the recipient at all" scored ~0.00 on every email, so the hedged
     middle won 100% of the inbox by default. */
  urgency: {
    label: 'Urgency',
    options: [
      ['Today',     'There is a deadline, an expiry or a security problem that costs the recipient something real if it is left until tomorrow.'],
      ['This week', 'The message names a specific appointment, due date or event that falls within the next several days.'],
      ['Whenever',  'This is an advertisement, a digest or a record of something already done, which the recipient can read at their leisure.'],
    ],
  },
};

export const KEYS = Object.keys(QUESTIONS);

/** The email as prose. The model was trained on sentence pairs, so it gets a readable message, not JSON.
 *
 * The body is capped hard, because the premise is re-encoded once per statement and cost is linear in
 * tokens. Measured on this box, for 16 statements:
 *
 *     body chars     tokens   ms/email
 *              0       1014         61
 *            240       1846        110
 *            900       4086        230
 *
 * Subject, sender and the opening lines carry nearly all the signal for these four questions, so paying
 * for 900 characters of marketing footer costs 2x latency and buys almost nothing. */
export function premiseFor(e) {
  const when = e.received ? new Date(e.received).toUTCString().replace(/ GMT$/, '') : 'an unknown date';
  return [
    `An email received on ${when}.`,
    `From: ${e.fromName ? `${e.fromName} <${e.fromAddress}>` : e.fromAddress}.`,
    `Subject: ${e.subject}.`,
    `Body: ${(e.body || e.snippet || '').slice(0, 240)}`,
  ].join('\n');
}

export const groupsFor = () =>
  Object.fromEntries(KEYS.map(k => [k, QUESTIONS[k].options.map(o => o[1])]));

export const CATEGORY_COLORS = {
  'Brand deal': '#e8b23a', 'Cold pitch': '#e5533d', 'Personal': '#3fbf7f', 'Newsletter': '#4aa3df',
  'Marketing': '#a77bf0', 'Receipt': '#2fb8a6', 'Service alert': '#ef4470', 'Notification': '#8593a6',
  'Other': '#5c6b7a',
};
export const URGENCY_COLORS = { 'Today': '#ef4470', 'This week': '#e8b23a', 'Whenever': '#5c6b7a' };
