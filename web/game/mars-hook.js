/* openjev bridge for vibe-arcade's mars.html.
 *
 * Division of labour, and it is deliberate: MARS RAID is a continuous 3D flight sim, which is exactly the
 * geometric case where a hand-written controller wins. So the code flies and aims, and the model only
 * chooses *what to attack next*. Target priority under a described situation is the part that is actually
 * about judgement rather than trigonometry.
 *
 * Aiming closes the loop on camera.getWorldDirection() -- the same vector the game raycasts shots along --
 * instead of assuming an Euler convention that could silently be off by a sign.
 */
export function bind(ctx) {
  const { THREE, ship, aliens, buildings, keys, camera } = ctx;
  const V = (x, y, z) => new THREE.Vector3(x, y, z);

  const worldPos = (o, out = V()) => (o.getWorldPosition ? o.getWorldPosition(out) : out.copy(o.position));

  // ------------------------------------------------------------------ targets
  // The scorpion fights in phases -- claws, then tail, then head -- and anything outside the current phase
  // reports back "armored" and takes no damage at all. Offering an invulnerable part as a target would be
  // the Cook Fever livelock again: a move that cannot change the state, chosen forever.
  const LABEL = { clawL: "the scorpion's left claw", clawR: "the scorpion's right claw",
                  tail: "the scorpion's tail", head: "the scorpion's head" };
  const PHASE_PARTS = { claws: ['clawL', 'clawR'], tail: ['tail'], head: ['head'] };

  const bossParts = () => {
    if (!ctx.bossAlive || ctx.bossState === 'dormant' || ctx.bossState === 'emerging' || ctx.bossState === 'dead') return [];
    const out = [];
    for (const key of PHASE_PARTS[ctx.bossPhase] || []) {
      const hp = ctx.bossHP[key], max = ctx.bossMax[key];
      if (!(hp > 0)) continue;
      if (key === 'clawL' && !(ctx.bossClaws.L && ctx.bossClaws.L.visible)) continue;
      if (key === 'clawR' && !(ctx.bossClaws.R && ctx.bossClaws.R.visible)) continue;
      // live() is the liveness test as well as the tracker: it returns null once the target is gone, and
      // the autopilot releases on null. Without it the ship keeps shooting a building it already destroyed.
      out.push({ kind: 'boss', key, uid: `boss:${key}`, label: LABEL[key], hp, max,
                 radius: ({ clawL: 6.5, clawR: 6.5, tail: 4.2, head: 5.6 })[key],
                 pos: ctx.getBossPartWorldPos(key).clone(),
                 live: () => (ctx.bossHP[key] > 0 && (PHASE_PARTS[ctx.bossPhase] || []).includes(key)
                              ? ctx.getBossPartWorldPos(key) : null) });
    }
    return out;
  };

  function targets() {
    const from = ship.position;
    const out = [];

    for (const a of aliens) {
      if (!a || !a.pos) continue;
      out.push({ kind: 'saucer', label: 'an alien saucer', radius: 3.2, pos: a.pos.clone(), dist: a.pos.distanceTo(from),
                 uid: `saucer:${a.uid}`,
                 live: () => (aliens.includes(a) ? a.pos : null) });
    }
    for (const b of buildings) {
      if (!b || !b.alive) continue;
      out.push({ kind: 'building', label: 'a colony building', bd: b, radius: b.radius, pos: b.center.clone(),
                 hp: b.hp, max: b.maxHp, dist: b.center.distanceTo(from),
                 uid: `building:${buildings.indexOf(b)}`,
                 live: () => (b.alive ? b.center : null) });
    }
    for (const p of bossParts()) out.push({ ...p, dist: p.pos.distanceTo(from) });

    out.sort((x, y) => x.dist - y.dist);
    return out;
  }

  function state() {
    const t = targets();
    return {
      mode: ctx.state,
      hull: Math.round(ctx.hp),
      altitude: Math.round(ship.position.y),
      shots: ctx.stats.shots, saucersKilled: ctx.stats.saucers, buildingsKilled: ctx.stats.buildings,
      bossState: ctx.bossState,
      bossPhase: ctx.bossPhase,
      bossAlive: !!ctx.bossAlive,
      boss: bossParts().map(p => ({ part: p.label, hp: p.hp, max: p.max, pct: Math.round(100 * p.hp / p.max) })),
      recentDamage: Math.round(recentDamage()),
      saucers: t.filter(x => x.kind === 'saucer').length,
      buildings: t.filter(x => x.kind === 'building').length,
      nearest: t[0] ? { what: t[0].label, dist: Math.round(t[0].dist) } : null,
    };
  }

  // ------------------------------------------------------------------ autopilot
  let target = null, fireHeld = false;

  /* The model is stateless between calls, so anything like "is this working?" has to be computed here and
     put in the premise. A snapshot ("a saucer is 49 m away") is always true and therefore always urgent;
     a trend ("no damage taken in the last 10 seconds") is what should actually decide whether to break off. */
  const hullLog = [];
  setInterval(() => {
    hullLog.push({ t: performance.now(), hp: ctx.hp });
    while (hullLog.length && hullLog[0].t < performance.now() - 10000) hullLog.shift();
  }, 250);
  const recentDamage = () => (hullLog.length ? Math.max(0, hullLog[0].hp - ctx.hp) : 0);

  const aimError = (pos) => {
    const d = camera.getWorldDirection(V());
    const want = pos.clone().sub(camera.getWorldPosition(V())).normalize();
    // Horizontal error signed by the cross product about world up, vertical error from the pitch angle.
    const yawErr = Math.atan2(d.x * want.z - d.z * want.x, d.x * want.x + d.z * want.z);
    const pitchErr = Math.asin(THREE.MathUtils.clamp(want.y, -1, 1)) - Math.asin(THREE.MathUtils.clamp(d.y, -1, 1));
    return { yawErr, pitchErr, angle: Math.acos(THREE.MathUtils.clamp(d.dot(want), -1, 1)) };
  };

  // One-time calibration: nudge yaw and see which way the camera actually turns, rather than trusting a
  // sign convention. Cheap, and it cannot be wrong.
  let yawSign = 0;
  function calibrate() {
    const before = aimError(ship.position.clone().add(V(0, 0, -100))).yawErr;
    ctx.yaw += 0.02;
    const after = aimError(ship.position.clone().add(V(0, 0, -100))).yawErr;
    ctx.yaw -= 0.02;
    yawSign = after < before ? 1 : -1;
  }

  function fly(dt) {
    if (ctx.state !== 'playing') { release(); return; }
    if (!yawSign) calibrate();

    if (!target || target.gone) { release(); return; }
    const pos = target.live ? target.live() : target.pos;
    if (!pos) { release(); return; }

    // Solve yaw/pitch directly from the target direction rather than servoing on the camera. The chase
    // camera springs behind the ship and the ship itself lags yaw/pitch, so a rate-limited loop closed on
    // camera direction oscillated around ~20 degrees and never reached the firing threshold.
    const dir = pos.clone().sub(ship.position).normalize();
    ctx.yaw = Math.atan2(-dir.x, -dir.z);
    ctx.pitch = THREE.MathUtils.clamp(Math.asin(THREE.MathUtils.clamp(dir.y, -1, 1)), -1.15, 1.15);
    const { angle } = aimError(pos);

    const dist = pos.distanceTo(ship.position);

    /* Height above the terrain, not above zero: the map is not flat, and the ship was flying into hillsides
       while its altitude reading still looked healthy. */
    const agl = ship.position.y - (ctx.groundHeight ? ctx.groundHeight(ship.position.x, ship.position.z) : 0);

    /* Close right in, the way it did before. The damage was never a nosedive -- it was the belly
       occasionally clipping the ground while manoeuvring at low level -- so the fix is a light floor that
       nudges the nose up, NOT a refusal to approach. Standing off at 400 m just meant missing everything. */
    const wantClose = dist > 70;
    if (wantClose && angle < 1.0) keys.add('KeyW'); else keys.delete('KeyW');
    if (dist < 45) keys.add('KeyS'); else keys.delete('KeyS');

    keys.delete('KeyQ'); keys.delete('KeyE');
    if (agl < 22) {
      keys.add('KeyQ');                       // climb out of a scrape, but keep flying at the target
      if (agl < 12 && ctx.pitch < 0) ctx.pitch = 0;   // only fight the aim when it is genuinely about to hit
    } else if (agl > 160) {
      keys.add('KeyE');
    } else if (agl > 45 && Math.floor(performance.now() / 2100) % 2 === 0) {
      keys.add('KeyE');                       // vertical jink only with room beneath to spend
    } else if (agl < 45) {
      keys.add('KeyQ');                       // low but not scraping: bias upward rather than jinking down
    }

    /* Horizontal jink is the part that actually makes saucers miss, and it costs no altitude. */
    const left = Math.floor(performance.now() / 1300) % 2 === 0;
    keys.delete(left ? 'KeyD' : 'KeyA');
    keys.add(left ? 'KeyA' : 'KeyD');

    // Speed is the best defence while hurt, and the booster is free to hold.
    if (recentDamage() >= 12 && agl > 30) keys.add('ShiftLeft'); else keys.delete('ShiftLeft');

    /* Gate on the actual miss distance rather than a fixed angle. A 9-degree cone is +/-63 m of lateral
       error at 400 m, against a building 4.5 m across -- it was firing constantly and hitting nothing
       (229 shots for 3 buildings). sin(angle) * dist is how far the beam passes from the target centre. */
    const miss = Math.sin(angle) * dist;
    // Exactly the game's own tolerance: it tests `distance < radius + 0.4`. Anything stricter just wastes
    // firing opportunities, and the jink is already pulling the aim around.
    fireHeld = !target.noFire && dist < 900 && miss < (target.radius ?? 4) + 0.4;
    ctx.firing = fireHeld;
  }

  function release() {
    for (const k of ['KeyW', 'KeyS', 'KeyA', 'KeyD', 'KeyQ', 'KeyE', 'ShiftLeft']) keys.delete(k);
    ctx.firing = false; fireHeld = false;
  }

  /** Break off: climb away from the nearest threat under boost. "Stop flying" was the old behaviour and it
   *  is the worst possible answer -- a stationary ship is exactly what the saucers can actually hit. */
  function evade() {
    const from = ship.position;
    const threat = targets().find(t => t.kind === 'saucer');
    const away = threat ? from.clone().sub(threat.pos).setY(0).normalize() : V(0, 0, -1);
    target = { pos: from.clone().addScaledVector(away, 600).setY(from.y + 120), live() { return this.pos; }, noFire: true };
  }

  // Drive the autopilot on its own rAF loop; the game keeps its own.
  let last = performance.now();
  (function tick(now) {
    const dt = Math.min(0.05, (now - last) / 1000); last = now;
    try { fly(dt); } catch (e) { /* never break the game's loop */ }
    requestAnimationFrame(tick);
  })(last);

  window.__mars = {
    state, targets,
    raw: ctx,
    debug: () => {
      const t = target && (target.live ? target.live() : target.pos);
      const e = t ? aimError(t) : null;
      return {
        yawSign, hasTarget: !!target,
        angleDeg: e ? +(e.angle * 180 / Math.PI).toFixed(2) : null,
        dist: t ? Math.round(t.distanceTo(ship.position)) : null,
        firing: fireHeld, gameFiring: ctx.firing,
        bullets: ctx.bullets ? ctx.bullets.length : null,
        sample: ctx.bullets && ctx.bullets[0]
          ? { isPlayer: ctx.bullets[0].isPlayer, ghost: ctx.bullets[0].ghost, dmg: ctx.bullets[0].dmg, owner: String(ctx.bullets[0].owner) }
          : null,
      };
    },
    ready: () => ctx.state === 'playing',
    mode: () => ctx.state,
    start: () => ctx.startGame(true),
    wakeBoss: () => { if (ctx.bossState === 'dormant') ctx.startEmerge(); },
    aim: (t) => { target = t; },
    evade, release,
    firing: () => fireHeld,
  };
  window.dispatchEvent(new Event('mars-ready'));
}
