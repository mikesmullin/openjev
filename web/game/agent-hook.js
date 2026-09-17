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
        // The slot timer is reused: it counts up to cookT while raw, is reset to 0 the instant the item is
        // done, then counts up to burnT. Reporting t/cookT throughout tells the model a finished patty is
        // "69% cooked" when it is really 69% of the way to being ruined.
        st.slots.forEach((s, i) => {
          const it = s.item;
          const done = it && (it.state === 'cooked' || it.state === 'melted' || it.state === 'ready');
          out.push({
            i, item: it, name: itemName(it),
            phase: !it ? 'empty' : it.state === 'burnt' ? 'burnt' : done ? 'done' : 'cooking',
            progress: it ? Math.min(1, s.t / (done ? st.burnT : st.cookT)) : 0,
          });
        });
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
        grill: grillSlots().map(g => ({ slot: g.i, has: g.name, phase: g.phase, progress: pct(g.progress) })),
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
      const phrase = (x) =>
        x.phase === 'burnt' ? `a burnt ${x.has.replace(/^burnt /, '')} in pan ${x.slot + 1}, ruined and only fit for the bin`
      : x.phase === 'done' ? `a finished ${x.has} in pan ${x.slot + 1}, cooked through and ${x.progress}% of the way to burning if it is left there`
      : `a ${x.has} in pan ${x.slot + 1}, still raw at ${x.progress}% cooked`;
      L.push(g.length ? 'On the grill: ' + g.map(phrase).join('; ') + '.' : 'The grill is empty.');

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

    /** Would anything accept what this tray produces? Taking an ingredient with nowhere to put it leads
     *  straight back to putting it down, which is the same alternation as every other livelock here. */
    function trayUseful(kind) {
      const free = (Cls, test) => ctx.stations.some(st => st instanceof Cls && test(st));
      switch (kind) {
        case 'patty': case 'sausage': case 'cheese':
          return free(ctx.Grill, st => st.slots.some(sl => !sl.item));
        case 'fries':
          return free(ctx.Fryer, st => !st.item);
        case 'bun': case 'hotdog_bun': case 'carton_empty':
          return free(ctx.PlateRack, st => st.slots.some(sl => sl.item && !(sl.item.stack || []).length));
        default:
          return true;
      }
    }

    /** Would anything take this item if we picked it up? Trash is excluded -- it accepts everything, which
     *  would make the test vacuous. */
    function hasDestination(item) {
      return interactives.some(o => {
        const w = o.owner;
        if (!w || w === item || w instanceof ctx.Trash) return false;
        try { return o.customer ? w.accepts(item) : (w.accepts ? w.accepts(item, o.slot) : false); }
        catch (e) { return false; }
      });
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
            // The invariant behind every livelock in this game: if nothing will accept an item, picking it
            // up leads to a state whose only move is to put it back down, and the two alternate forever.
            // This one test covers an unordered finished dish, an empty plate, food still on the heat, and
            // a cooked patty with no bun waiting for it. Burnt food is exempt so it can still be binned.
            if (w.state !== 'burnt' && !hasDestination(w)) continue;
            // Food on the heat is never worth lifting early, even if a plate would take it.
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
              if (!trayUseful(w.kind)) continue;
              // A generic "another X is needed" loses to `wait` on a quiet board. Name the specific fact in
              // the premise that makes this the right move -- the same reason statements beat action names.
              const waitingOn = grillSlots().find(x => x.phase === 'done' && x.item &&
                ((w.kind === 'bun' && x.item.kind === 'patty') || (w.kind === 'hotdog_bun' && x.item.kind === 'sausage')));
              push(out, `take a ${w.kind} from the tray`,
                waitingOn
                  ? `A ${waitingOn.item.kind} is already cooked and burning on the grill, but no plate has a ${w.kind} on it to put it on.`
                  : `Another ${w.kind} is needed to fill an order that is still outstanding.`,
                () => w.onTap());
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
