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
      out.push({ kind: 'boss', key, label: LABEL[key], hp, max,
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
      out.push({ kind: 'saucer', label: 'an alien saucer', pos: a.pos.clone(), dist: a.pos.distanceTo(from),
                 live: () => (aliens.includes(a) ? a.pos : null) });
    }
    for (const b of buildings) {
      if (!b || !b.alive) continue;
      out.push({ kind: 'building', label: 'a colony building', bd: b, pos: b.center.clone(),
                 hp: b.hp, max: b.maxHp, dist: b.center.distanceTo(from),
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
      saucers: t.filter(x => x.kind === 'saucer').length,
      buildings: t.filter(x => x.kind === 'building').length,
      nearest: t[0] ? { what: t[0].label, dist: Math.round(t[0].dist) } : null,
    };
  }

  // ------------------------------------------------------------------ autopilot
  let target = null, fireHeld = false;

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
    if (dist > 70 && angle < 1.0) keys.add('KeyW'); else keys.delete('KeyW');
    if (ship.position.y < 18) keys.add('KeyQ'); else keys.delete('KeyQ');

    fireHeld = angle < 0.16 && dist < 900;
    ctx.firing = fireHeld;
  }

  function release() {
    keys.delete('KeyW'); keys.delete('KeyQ');
    ctx.firing = false; fireHeld = false;
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
    start: () => ctx.startGame(true),
    wakeBoss: () => { if (ctx.bossState === 'dormant') ctx.startEmerge(); },
    aim: (t) => { target = t; },
    release,
    firing: () => fireHeld,
  };
  window.dispatchEvent(new Event('mars-ready'));
}
