/* openjev agent bridge for vibe-arcade's cook2.html.
 *
 * Loaded before the game's module; the fetch script appends one line inside that module which calls
 * bind() with its private scope. From there we drive the game through its own affordances --
 * select() / tryDrop() / onTap() -- rather than synthesising pointer events at guessed coordinates.
 *
 * The hypothesis wording follows the Doom result: every candidate action is offered to the model as a
 * STATEMENT ABOUT THE WORLD that would justify it ("The patty on the grill is cooked"), never as the name
 * of the action ("pick up the patty"). Naming the action scores at chance; verifying a statement works.
 */
const pct = (v) => Math.round(v * 100);

export function bind(ctx) {
    const { G, interactives, customers, piles, DISH } = ctx;

    // ---------------------------------------------------------------- state
    const held = () => ctx.selected;

    const itemName = (it) => {
      if (!it) return null;
      if (it.kind === 'plate') {
        const s = it.stack || [];
        return s.length ? `plate holding ${s.join(' + ')}` : 'empty plate';
      }
      return it.state && it.state !== 'whole' ? `${it.state} ${it.kind}` : it.kind;
    };

    function grillSlots() {
      const out = [];
      for (const st of ctx.stations) {
        if (!(st instanceof ctx.Grill)) continue;
        st.slots.forEach((s, i) =>
          out.push({
            i, item: s.item, name: itemName(s.item),
            progress: s.item ? Math.min(1, s.t / st.cookT) : 0,
            burning: s.item ? s.t > st.cookT : false,
          }));
      }
      return out;
    }

    function plateSlots() {
      const out = [];
      for (const st of ctx.stations) {
        if (!(st instanceof ctx.PlateRack)) continue;
        st.slots.forEach((s, i) =>
          out.push({ i, stack: (s.item && s.item.stack) || [], dish: s.item && s.item.dish ? s.item.dish() : null }));
      }
      return out;
    }

    /** Machines that work on their own over time. If the premise omits these, the model cannot tell that
     *  tapping the soda machine did anything, re-picks it forever, and the game never advances. */
    function machines() {
      const out = { soda: null, fryer: null };
      for (const st of ctx.stations) {
        if (st instanceof ctx.SodaMachine) {
          out.soda = st.cup
            ? { busy: true, full: st.cup.state === 'full', done: pct(Math.min(1, st.t / st.fillT)) }
            : { busy: false };
        } else if (st instanceof ctx.Fryer) {
          out.fryer = st.item
            ? { busy: true, ready: st.item.state === 'ready', done: pct(Math.min(1, st.t / st.cookT)) }
            : { busy: false };
        }
      }
      return out;
    }

    function state() {
      return {
        ...machines(),
        piles: (piles || []).length,
        mode: G.mode,
        level: G.level,
        coins: G.coins,
        served: G.served,
        goal: G.cfg ? G.cfg.goal : null,
        timeLeft: Math.max(0, Math.round(G.time || 0)),
        holding: itemName(held()),
        customers: customers.filter(c => c.state === 'waiting').map((c, i) => ({
          i,
          wants: (c.remaining || []).map(d => (DISH[d] && DISH[d].label) || d),
          patience: c.bar && typeof c.bar.v === 'number' ? pct(c.bar.v) : null,
        })),
        grill: grillSlots().map(g => ({ slot: g.i, has: g.name, done: pct(g.progress), burning: g.burning })),
        plates: plateSlots().map(p => ({ slot: p.i, stack: p.stack, dish: p.dish })),
      };
    }

    // ------------------------------------------------------------- premise
    function premise() {
      const s = state();
      const L = [];
      L.push(`A diner kitchen, level ${s.level}. You have served ${s.served} of ${s.goal ?? '?'} orders with ${s.timeLeft} seconds left.`);
      L.push(s.holding ? `In your hands: a ${s.holding}.` : 'Your hands are empty.');

      if (s.customers.length) {
        L.push('Waiting customers: ' + s.customers.map((c, i) =>
          `customer ${i + 1} still wants ${c.wants.join(' and ')}${c.patience != null ? ` (patience ${c.patience}%)` : ''}`
        ).join('; ') + '.');
      } else L.push('No customer is waiting at the counter right now.');

      const g = s.grill.filter(x => x.has);
      L.push(g.length
        ? 'On the grill: ' + g.map(x => `a ${x.has} in pan ${x.slot + 1}, ${x.burning ? 'already burning' : `${x.done}% cooked`}`).join('; ') + '.'
        : 'The grill is empty.');

      if (s.soda) L.push(s.soda.busy
        ? (s.soda.full ? 'A full cup of soda is sitting on the soda machine, poured and ready to pick up.'
                       : `The soda machine is busy pouring a cup, ${s.soda.done}% full.`)
        : 'The soda machine is idle with no cup under it.');
      if (s.fryer) L.push(s.fryer.busy
        ? (s.fryer.ready ? 'A basket of fries has finished frying and is ready to pick up.'
                         : `The fryer is busy cooking fries, ${s.fryer.done}% done.`)
        : 'The fryer is empty.');

      const p = s.plates.filter(x => x.stack.length);
      if (p.length) L.push('Plates being assembled: ' + p.map(x => `plate ${x.slot + 1} has ${x.stack.join(' + ')}${x.dish ? `, which is a finished ${x.dish}` : ''}`).join('; ') + '.');
      else L.push('Every plate on the rack is still empty.');

      if (s.piles) L.push(`There ${s.piles === 1 ? 'is 1 pile' : `are ${s.piles} piles`} of payment left on the counter. ` +
        'A seat stays occupied until its payment is picked up, so no new customer can sit down there.');

      L.push('Food left on the grill too long burns and becomes worthless. A customer whose patience runs out leaves without paying.');
      return L.join(' ');
    }

    // ------------------------------------------------------------- actions
    // Each candidate carries the statement about the world that would justify taking it.
    let table = [];

    function push(out, label, hypothesis, run) {
      out.push({ id: out.length, label, hypothesis, run });
    }

    function actions() {
      const out = [];
      const sel = held();
      const wanted = new Set(customers.filter(c => c.state === 'waiting').flatMap(c => c.remaining || []));

      // Collecting payment works with or without something in hand, so it is offered in both branches.
      // It is not just income: freeSlot() skips any seat that still has a pile on it, so leaving money on
      // the counter stops the next customer from ever arriving.
      for (const pile of piles || []) {
        push(out, 'collect the payment from the counter',
          'There is payment sitting on the counter, and the seat it is on stays blocked until it is picked up.',
          () => pile.onTap());
        break;   // one hypothesis covers all piles; collectPile() takes the whole seat's stack anyway
      }

      if (!sel) {
        for (const o of interactives) {
          const w = o.owner;
          if (!w) continue;
          // pick an item up off a station or a plate rack
          if (o.drag && w instanceof ctx.Item) {
            const n = itemName(w);
            const onGrill = w.station instanceof ctx.Grill;
            const dish = w.dish && w.dish();
            // A finished dish nobody ordered is a trap: picking it up leads to a state whose only move is
            // to put it back down, and the pair alternates forever. Leave it where it is until it is wanted.
            if (dish && !wanted.has(dish)) continue;
            // Plates are assembled in place on the rack -- you drop ingredients onto them. Lifting a plate
            // is only useful once it is a finished dish, and offering it otherwise creates the same
            // pick-up / put-down alternation.
            if (w.kind === 'plate' && !dish) continue;
            // Food still on the heat: lifting it undoes progress, so the only follow-up is to put it back.
            // Offer it only once it is actually done (or ruined, so it can be binned).
            if (onGrill && (w.state === 'raw' || w.state === 'cooking' || w.state === 'fresh')) continue;
            let hyp;
            if (onGrill) hyp = `The ${w.kind} on the grill is ${w.state} and should come off the heat now.`;
            else if (dish && wanted.has(dish)) hyp = `A finished ${DISH[dish] ? DISH[dish].label.toLowerCase() : dish} is ready and a customer is waiting for exactly that.`;
            else hyp = `The ${n} should be picked up now.`;
            push(out, `pick up the ${n}`, hyp, () => ctx.select(w));
          } else if (w instanceof ctx.Tray || w instanceof ctx.SodaMachine || w instanceof ctx.Fryer) {
            // Only the stations that actually produce an item. Station.onTap() is a no-op on the base
            // class, so offering it for every station yields identical dead hypotheses (one per grill pan)
            // that cost a forward pass each and can out-vote a real action by sheer duplication.
            // Guard each tap by the station's own precondition. onTap() returns false when it cannot act
            // (SodaMachine: `if (selected || this.cup) return false`), and offering an action that is a
            // no-op lets it win the argmax again on an unchanged state -- a livelock.
            if (w instanceof ctx.Tray) {
              push(out, `take a ${w.kind} from the tray`,
                `Another ${w.kind} is needed to fill an order that is still outstanding.`, () => w.onTap());
            } else if (w instanceof ctx.SodaMachine) {
              if (!w.cup) push(out, 'start pouring a soda',
                'A customer has ordered a soda and none has been poured yet.', () => w.onTap());
            } else if (!w.item) {
              push(out, 'start the fryer', 'A customer has ordered fries and none are in the fryer yet.', () => w.onTap());
            }
          }
        }
      } else {
        const dish = sel.dish && sel.dish();
        for (const o of interactives) {
          const w = o.owner;
          if (!w || w === sel) continue;
          let ok = false;
          try { ok = o.customer ? w.accepts(sel) : (w.accepts ? w.accepts(sel, o.slot) : false); } catch (e) { ok = false; }
          if (!ok) continue;

          if (o.customer) {
            const i = customers.filter(c => c.state === 'waiting').indexOf(w) + 1;
            push(out, `serve the ${itemName(sel)} to customer ${i}`,
              `The ${dish ? (DISH[dish] ? DISH[dish].label.toLowerCase() : dish) : itemName(sel)} in your hands is exactly what a waiting customer ordered.`,
              () => ctx.tryDrop(sel, w));
          } else if (w instanceof ctx.Grill) {
            push(out, `put the ${itemName(sel)} on the grill`,
              `The ${sel.kind} in your hands is raw and needs to be cooked on the grill.`,
              () => ctx.tryDrop(sel, w, o.slot));
          } else if (w instanceof ctx.Trash) {
            push(out, `throw the ${itemName(sel)} away`,
              `The ${itemName(sel)} in your hands is burnt or useless and should be thrown away.`,
              () => ctx.tryDrop(sel, w, o.slot));
          } else if (w instanceof ctx.PlateRack) {
            push(out, `add the ${itemName(sel)} to plate ${(o.slot && o.slot.i != null ? o.slot.i : 0) + 1}`,
              `The ${sel.kind} in your hands is cooked and belongs on a plate with the rest of the order.`,
              () => ctx.tryDrop(sel, w, o.slot));
          } else {
            push(out, `put the ${itemName(sel)} on the ${w.constructor.name.toLowerCase()}`,
              `The ${itemName(sel)} in your hands belongs on the ${w.constructor.name.toLowerCase()}.`,
              () => ctx.tryDrop(sel, w, o.slot));
          }
        }
        push(out, `put the ${itemName(sel)} back down`,
          `Nothing useful can be done with the ${itemName(sel)} in your hands right now.`,
          () => ctx.deselect());
      }

      // Always leave one legal move. Food cooks and customers arrive on their own, so waiting is a real
      // strategy, and an empty action list would otherwise stall the loop.
      push(out, 'wait', 'Everything that can be started right now is already underway, so the best move is to wait.', () => {});
      // Several affordances can justify themselves with the same sentence (three empty grill pans, two
      // identical trays). Scoring a duplicate costs a forward pass and tells us nothing new, so collapse
      // them: the first action wins the id, the rest are dropped.
      const seen = new Map();
      table = out.filter(a => (seen.has(a.hypothesis) ? false : (seen.set(a.hypothesis, a.id), true)));
      table.forEach((a, i) => { a.id = i; });
      return table.map(({ id, label, hypothesis }) => ({ id, label, hypothesis }));
    }

    function act(id) {
      const a = table[id];
      if (!a) return false;
      try { a.run(); return true; } catch (e) { console.warn('[agent] action failed', a.label, e); return false; }
    }

    window.__agent = {
      ready: () => G.mode === 'playing',
      mode: () => G.mode,
      start: (n = 1) => ctx.startLevel(n),
      menu: () => ctx.showMenu(),
      state, premise, actions, act,
    };
    window.dispatchEvent(new Event('agent-ready'));
}
