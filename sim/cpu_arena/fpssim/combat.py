"""Combat: hit volumes, damage/armor, shot direction sampling, bullet tracing.

Damage (EXACT vs recordings, 522/522 hit events in the probe game reproduce hp_damage/armor_absorbed):
    amount   = base_damage * range_modifier ** (distance_m / 12.7) * part_mult
    part_mult: head -> headMultiplier, upper (chest/abdomen/arms) -> 1.0, lower (legs) -> 0.75
    armor applies to 'upper' with armor>0, and to 'head' only with helmet and armor>0; legs never.
    hp_damage = amount * armorRatio * 0.5 ; armor_absorbed = (amount - hp_damage) * 0.5   (if armor applies)
    if armor_absorbed > armor: armor_absorbed = armor; hp_damage = amount - armor_absorbed / 0.5
(12.7 m == 500 Source units; matches CS rangeModifier^(dist/500u).)
"""
import math, os, json
import numpy as np
from .common import WEAPONS, KNIFE
from . import weapons as W
from . import accuracy as ACCM

RANGE_UNIT = 12.7
LOWER_FRAC = 0.75 / 1.8      # legs below this fraction of body height
HEAD_FRAC = 1.55 / 1.8       # head above this fraction
STAND_H = 1.8; CROUCH_H = 1.1; EYE_STAND = 1.62; EYE_CROUCH = 0.95
PART_MULT = {'upper': 1.0, 'lower': 0.75}
# hit-volume radii (fitted upper bounds from hit points; see docs/combat.md)
R_HEAD, R_UPPER, R_LOWER = 0.25, 0.42, 0.34


def damage(wid, part, dist, armor=0.0, helmet=False, backstab=False):
    """Returns (amount, hp_damage, armor_absorbed) for one bullet/pellet."""
    w = WEAPONS[wid]
    mult = w['headMultiplier'] if part == 'head' else PART_MULT[part]
    d = min(dist, w['range'])
    amount = w['damage'] * (w['rangeModifier'] ** (d / RANGE_UNIT)) * mult
    protected = armor > 0 and (part == 'upper' or (part == 'head' and helmet))
    if not protected:
        return amount, amount, 0.0
    hp = amount * w['armorRatio'] * 0.5
    ab = (amount - hp) * 0.5
    if ab > armor:
        ab = armor; hp = amount - ab / 0.5
    return amount, hp, ab


def knife_damage(kind, armor=0.0, helmet=False, back=False, first=True):
    """LEGACY (superseded by fpssim.melee.knife_damage, the cs2-web port used by knife_attack)."""
    if kind == 'stab': amt = KNIFE['back_stab'] if back else KNIFE['stab_damage']
    else: amt = KNIFE['back_slash'] if back else (KNIFE['first_slash'] if first else KNIFE['repeat_slash'])
    if armor > 0:
        hp = amt * KNIFE['armor_scale']; ab = (amt - hp) * 0.5
        if ab > armor: ab = armor; hp = amt - ab / 0.5
        return amt, hp, ab
    return amt, amt, 0.0


def body_height(crouch_height):
    return crouch_height


def part_from_z(z_rel, h):
    f = z_rel / h
    if f >= HEAD_FRAC: return 'head'
    if f < LOWER_FRAC: return 'lower'
    return 'upper'


def ray_vs_player(o, d, pos, h, tmax):
    """LEGACY cylinder model (superseded by ray_hit_player).  Ray o + t*d (d unit) against the vertical stack of cylinders for a player standing at pos (feet).
    Returns (t, part) of nearest hit with t<=tmax or None."""
    best = None
    z0 = pos[2]
    vols = ((0.0, LOWER_FRAC * h, R_LOWER, 'lower'), (LOWER_FRAC * h, HEAD_FRAC * h, R_UPPER, 'upper'), (HEAD_FRAC * h, h, R_HEAD, 'head'))
    ox = o[0] - pos[0]; oy = o[1] - pos[1]
    a = d[0] * d[0] + d[1] * d[1]
    for lo, hi, r, name in vols:
        zlo = z0 + lo; zhi = z0 + hi
        cand = []
        if a > 1e-12:
            b = ox * d[0] + oy * d[1]; c = ox * ox + oy * oy - r * r
            disc = b * b - a * c
            if disc >= 0:
                sq = math.sqrt(disc)
                for t in ((-b - sq) / a, (-b + sq) / a):
                    if 0 <= t <= tmax and zlo <= o[2] + t * d[2] <= zhi: cand.append(t)
        if abs(d[2]) > 1e-12:
            for zc in (zlo, zhi):
                t = (zc - o[2]) / d[2]
                if 0 <= t <= tmax:
                    x = ox + t * d[0]; y = oy + t * d[1]
                    if x * x + y * y <= r * r: cand.append(t)
        if cand:
            t = min(cand)
            if best is None or t < best[0]: best = (t, name)
    return best


def shot_direction(yaw, pitch, ap_pitch, ap_yaw, radius, rng):
    """Bullet direction = view + 2*aim_punch + uniform-disc deviation of given radius (radians).
    Returns unit vector (forward=(cos yaw,-sin yaw))."""
    if radius > 0:      # `radius` is the per-axis gaussian sigma (fitted, see docs/combat.md)
        dy = rng.gauss(0.0, radius); dp = rng.gauss(0.0, radius)
    else:
        dy = dp = 0.0
    y = yaw + 2 * ap_yaw + dy; p = pitch + 2 * ap_pitch + dp
    cp = math.cos(p)
    return (cp * math.cos(y), -cp * math.sin(y), math.sin(p))


def _spread_radius(wid, inaccuracy, crouch, speed, grounded, scoped):
    """Per-axis gaussian sigma (rad) of bullet deviation: base_scale*stance spread + inacc_scale*inaccuracy
    + move_scale*movingSpread*min(1, speed/maxSpeed) (also applied at full strength while airborne). Fitted: still shots
    n=625 (sigma = 0.4*base + 0.1*inacc ML optimum), moving shots n=99 (move_scale 0.25, weakly constrained)."""
    sm = W.FIT['spread_model']; w = WEAPONS[wid]
    base = W.spread_for(wid, crouch, speed, grounded, scoped)
    if not grounded: frac = 1.0
    else: frac = min(1.0, speed / max(w['maxSpeed'], 1e-6)) if speed > sm['move_speed_threshold'] else 0.0
    return sm['base_scale'] * base + sm['inacc_scale'] * inaccuracy + sm['move_scale'] * w['movingSpread'] * frac


def trace_shot(o, d, shooter, targets, geom=None, maxdist=None):
    """First-hit bullet trace.  targets: list of dict(seat, pos(x,y,z feet), h, alive).
    geom: object with raycast(p0,p1)->(hit,t,point,material) for static world.  Returns dict or None."""
    L = maxdist
    if geom is not None:
        p1 = (o[0] + d[0] * L, o[1] + d[1] * L, o[2] + d[2] * L)
        hit, t, pt, mat = geom.raycast(o, p1)
        wall_t = t * L if hit else None
    else:
        wall_t = None
    tmax = wall_t if wall_t is not None else L
    best = None
    for tg in targets:
        if tg['seat'] == shooter or not tg['alive']: continue
        r = ray_hit_player(o, d, tg['pos'], tg.get('yaw', 0.0), _target_crouch(tg), tmax)
        if r and (best is None or r[0] < best[0]): best = (r[0], r[1], tg)
    if best:
        t, part, tg = best
        return dict(kind='player', t=t, part=part, target=tg['seat'], point=tuple(o[i] + d[i] * t for i in range(3)))
    if wall_t is not None:
        return dict(kind='wall', t=wall_t, point=tuple(o[i] + d[i] * wall_t for i in range(3)))
    return None


def damage_validation(dirs, load):
    """Replays every 'hit' event in the recordings: (a) amount/hp/armor from implied distance (formula exactness),
    (b) amount from our own ray-vs-hitbox geometry (distance error)."""
    import collections
    stats = collections.Counter(); relerr = []; part_ok = collections.Counter(); implied_gap = []
    for d in dirs:
        P, T, ci, inv, ev = load(d)
        lastfire = {}
        for e in ev:
            if e['type'] == 'fire': lastfire.setdefault((e['match_tick'], e['player']), []).append(e)
            if e['type'] != 'hit' or e['kind'] != 'bullet': continue
            fl = lastfire.get((e['match_tick'], e['player']))
            if not fl: stats['no_fire_match'] += 1; continue
            # shotgun pellets: pick matching pellet by order of hits at same tick
            f = fl[min(len(fl) - 1, 0)]
            wid = f['weapon']
            t = e['match_tick']
            fi = int(np.searchsorted(T, t))
            fi = min(fi, len(T) - 1)
            prev = max(fi - 1, 0)
            tg = e['target']
            armor = float(P[prev, tg, ci['armor']]); helmet = bool(P[prev, tg, ci['helmet']])
            amt, hp, ab = damage(wid, e['part'], 0.0, armor, helmet)  # distance 0 only to test armor split below
            # armor split exactness given the reported amount
            amount = e['amount']
            protected = armor > 0 and (e['part'] == 'upper' or (e['part'] == 'head' and helmet))
            w = WEAPONS[wid]
            if protected:
                hp2 = amount * w['armorRatio'] * 0.5; ab2 = (amount - hp2) * 0.5
                if ab2 > armor: ab2 = armor; hp2 = amount - ab2 / 0.5
            else: hp2, ab2 = amount, 0.0
            ok = abs(hp2 - e['hp_damage']) < 1e-6 and abs(ab2 - e['armor_absorbed']) < 1e-6
            stats['armor_split_exact' if ok else 'armor_split_mismatch'] += 1
            if w['pellets'] > 1: continue
            # geometry: implied distance from amount vs ray-hitbox distance
            mult = w['headMultiplier'] if e['part'] == 'head' else PART_MULT[e['part']]
            if amount <= 0: continue
            de = RANGE_UNIT * math.log(amount / (w['damage'] * mult)) / math.log(w['rangeModifier'])
            # target pose at tick (linear interp between frames)
            f0 = int(np.searchsorted(T, t, side='right') - 1); f1 = min(f0 + 1, len(T) - 1)
            wgt = (t - T[f0]) / max(T[f1] - T[f0], 1)
            pos = [(1 - wgt) * P[f0, tg, ci[k]] + wgt * P[f1, tg, ci[k]] for k in ('x', 'y', 'z')]
            h = float(P[f0, tg, ci['height']])
            o = np.array(f['from']); dv = np.array(f['to']) - o; L = np.linalg.norm(dv); dv /= L
            r = ray_vs_player(o, dv, pos, h, L)
            if r is None: stats['ray_miss_our_hitbox'] += 1; continue
            part_ok[(r[1] == e['part'])] += 1
            implied_gap.append(de - r[0])
            relerr.append((w['rangeModifier'] ** (r[0] / RANGE_UNIT)) / (w['rangeModifier'] ** (de / RANGE_UNIT)) - 1.0)
    out = dict(stats)
    out['part_agree'] = dict(part_ok)
    if relerr:
        a = np.abs(relerr)
        out['amount_rel_err_from_geometry'] = dict(n=len(a), p50=float(np.median(a)), p90=float(np.percentile(a, 90)), p99=float(np.percentile(a, 99)))
        g = np.array(implied_gap); out['implied_distance_minus_ray_distance_m'] = dict(p5=float(np.percentile(g, 5)), p50=float(np.median(g)), p95=float(np.percentile(g, 95)))
    return out


def shot_rng(shot_id, pellet=0):
    """cs2-web shared/aim.js shotRng: per-(shot, pellet) LCG stream returning floats in [0, 1).
    The platform's shotId/seed is not public; Arena draws shot_id from its own seeded RNG (see new_shot_id)."""
    s = (((shot_id + pellet * 10007) * 1664525 + 1013904223) & 0xFFFFFFFF)

    def nxt():
        nonlocal s
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        return s / 4294967296.0
    return nxt


def new_shot_id(rng):
    """shot id in [1, 2^31-1] from a python random.Random (our own deterministic stream)."""
    return rng.randrange(1, 2147483647)


def fire_angles(yaw, pitch, ap_pitch, ap_yaw):
    """cs2-web bulletFireAngles: view + RECOIL_SCALE(2) * aim punch, pitch clamped to +-1.48."""
    return yaw + 2.0 * ap_yaw, max(-1.48, min(1.48, pitch + 2.0 * ap_pitch))


def pellet_directions(wid, yaw, pitch, ap_pitch, ap_yaw, penalty, speed, crouch, grounded, vz, zoom_level, shot_id):
    """Directions of every pellet of one trigger pull (cs2-web fire(): accuracyForShot once, then per pellet
    sampleShotDirection with shotRng(shotId, pellet)).  Scoped weapons use accuracy index 1 when zoom_level > 0.
    Multi-pellet (shotgun) behaviour is a literal port, UNVERIFIED (no shotgun shots in the recordings)."""
    meta = WEAPONS[wid]
    acc = ACCM.accuracy(meta, speed, crouch, grounded, vz, zoom_level, penalty)
    fy, fp = fire_angles(yaw, pitch, ap_pitch, ap_yaw)
    return [ACCM.sample_direction(fy, fp, acc, shot_rng(shot_id, k)) for k in range(meta.get('pellets', 1) or 1)], acc


def _target_crouch(tg):
    if 'crouch' in tg: return bool(tg['crouch'])
    return tg.get('h', STAND_H) < 0.5 * (STAND_H + CROUCH_H)


def shoot(w, st, targets, geom, rng, mesh=None):
    """Resolve one trigger pull of Weapon `w` (after WeaponSystem emitted ('shot', w)).
    st: dict(seat, eye(x,y,z), yaw, pitch, crouch, speed, grounded[, vz], zoom_level[, team]).
    targets: list of dict(seat, pos, alive, armor, helmet, team, yaw, crouch | h).
    geom: object with raycast(p0,p1)->(hit,t,point,material) (used when mesh is None).
    mesh: optional TriMesh -> cs2-web penetration trace (fpssim.penetration) incl. multi-player 0.55 scaling.
    Uses the pre-kick punch / penalty stored in w.punch_at_shot (validated in validate/spread.py), cs2-web accuracy,
    shotRng pellets, cs2-web hit volumes (ray_hit_player) and the exact damage formula.
    Returns list of per-pellet dict(pellet, from, to, kind('player'|'wall'|None), part, target, dist, amount,
    hp_damage, armor_absorbed, materials, hits=[(seat, dist, part, scale, wallbang, pens)])."""
    from . import penetration as PEN
    meta = WEAPONS[w.id]
    ap_p, ap_y, pen = getattr(w, 'punch_at_shot', None) or (w.ap_pitch, w.ap_yaw, w.inaccuracy)
    o = tuple(st['eye'])
    dirs, _ = pellet_directions(w.id, st['yaw'], st['pitch'], ap_p, ap_y, pen, st.get('speed', 0.0), st.get('crouch', False),
                                st.get('grounded', True), st.get('vz', 0.0), st.get('zoom_level', 0), new_shot_id(rng))
    L = meta['range']
    armor_state = {}
    out = []
    for pel, d in enumerate(dirs):
        phits = []
        for tg in targets:
            if tg['seat'] == st['seat'] or not tg['alive']: continue
            if st.get('team') is not None and tg.get('team') == st.get('team'): continue      # cs2-web: enemies only
            r = ray_hit_player(o, d, tg['pos'], tg.get('yaw', 0.0), _target_crouch(tg), L)
            if r is not None: phits.append((tg['seat'], r[0], r[1]))
        if mesh is not None:
            tr = PEN.trace_bullet(o, d, meta, phits, PEN.surfaces(mesh, o, d, L))
        else:
            wall = None
            if geom is not None:
                hit, t, _, _ = geom.raycast(o, tuple(o[k] + d[k] * L for k in range(3)))
                wall = t * L if hit else None
            lim = wall if wall is not None else L
            first = min((h for h in phits if h[1] < lim), key=lambda h: h[1], default=None)
            tr = dict(end_distance=first[1] if first else lim, hit_world=wall is not None and first is None,
                      hits=[(first[0], first[1], first[2], 1.0, False, 0)] if first else [], segments=[])
        to = tuple(o[k] + d[k] * tr['end_distance'] for k in range(3))
        res = {'pellet': pel, 'from': o, 'to': to, 'kind': 'wall' if tr['hit_world'] else None, 'hits': tr['hits'],
               'materials': [dict(entry=sg['entry'], exit=sg['exit'], material=sg['material'], thickness=sg['thickness'])
                             for sg in tr['segments']]}
        if tr['hits']:
            seat, dist, part, scale, wb, npen = tr['hits'][0]
            tg = next(t for t in targets if t['seat'] == seat)
            arm = armor_state.get(seat, tg.get('armor', 0.0))
            amt, hp, ab = damage(w.id, part, dist, arm, tg.get('helmet', False))
            if scale != 1.0: amt, hp, ab = amt * scale, hp * scale, ab * scale
            armor_state[seat] = max(0.0, arm - ab)          # cs2-web: pellets of one shot share a temporary armor state
            res.update(kind='player', part=part, target=seat, dist=dist, amount=amt, hp_damage=hp, armor_absorbed=ab)
        out.append(res)
    return out


def knife_attack(st, targets, secondary=False, first=True, raycast=None):
    """cs2-web swingKnife / traceKnife (fpssim.melee).  UNVERIFIED (no knife hits in recordings).
    st: dict(seat, eye, yaw[, pitch=0][, team][, x, y]); targets: Arena _targets() dicts (pos, yaw, crouch | h, ...).
    first: True if > 60 ticks since this attacker's previous swing (WeaponSystem.knife_first).
    raycast(origin, dir, max_dist) -> distance or None (static world), optional.
    Returns None or dict(kind, target, part='upper', dist, amount, hp_damage, armor_absorbed=0, back, heavy).
    Knife damage = floor(raw * 0.85 if armor > 0); armor is not reduced (cs2-web)."""
    from . import melee as M
    o = tuple(st['eye'])
    dv = M.direction(st['yaw'], st.get('pitch', 0.0))
    sh = dict(seat=st['seat'], team=st.get('team'), x=st.get('x', o[0]), y=st.get('y', o[1]))
    players = [dict(seat=tg['seat'], team=tg.get('team'), alive=tg['alive'], x=tg['pos'][0], y=tg['pos'][1], z=tg['pos'][2],
                    yaw=tg.get('yaw', 0.0), crouch=_target_crouch(tg)) for tg in targets]
    if st.get('team') is None:                       # Arena's st has no team: take it from the shooter's own target entry
        sh['team'] = next((tg.get('team') for tg in targets if tg['seat'] == st['seat']), None)
    r = M.trace_knife(o, dv, sh, players, heavy=secondary, raycast=raycast)
    if r['hit'] is None: return None
    tg = next(t for t in targets if t['seat'] == r['hit'])
    hp = M.knife_damage(heavy=secondary, backstab=r['backstab'], first=first, armor=tg.get('armor', 0.0))
    raw = M.knife_damage(heavy=secondary, backstab=r['backstab'], first=first, armor=0.0)
    return dict(kind='player', target=r['hit'], part='upper', dist=r['distance'], amount=float(raw), hp_damage=float(hp),
                armor_absorbed=0.0, back=r['backstab'], heavy=bool(secondary), point=r['end'])


# ---------------------------------------------------------------------------------------------------------------
# cs2-web hit volumes (server/game.js rayHitPlayer, commit e0e7e858): head sphere + yaw-rotated upper / lower boxes.
# Platform parts: head / upper / lower.  Local frame: lateral axis = right (-sin y, -cos y), depth axis = forward.
def _sphere_hit(o, d, c, r):
    oc = (o[0] - c[0], o[1] - c[1], o[2] - c[2])
    b = oc[0] * d[0] + oc[1] * d[1] + oc[2] * d[2]
    cc = oc[0] ** 2 + oc[1] ** 2 + oc[2] ** 2 - r * r
    h = b * b - cc
    if h < 0: return None
    s = math.sqrt(h); t = -b - s
    if t < 0: t = -b + s
    return t if t >= 0 else None


def _box_hit(o, d, lo, hi):
    tn, tf = -math.inf, math.inf
    for k in range(3):
        if abs(d[k]) < 1e-12:
            if o[k] < lo[k] or o[k] > hi[k]: return None
            continue
        a = (lo[k] - o[k]) / d[k]; b = (hi[k] - o[k]) / d[k]
        if a > b: a, b = b, a
        tn = max(tn, a); tf = min(tf, b)
        if tn > tf: return None
    if tf < 0: return None
    return tn if tn >= 0 else tf


def ray_hit_player(o, d, pos, yaw, crouch, tmax=math.inf):
    """-> (distance, part) or None.  o, d world (z up, d unit); pos = feet; yaw = target's platform yaw.

    Literal cs2-web rayHitPlayer.  Validated on 4047 recorded single-bullet hits without wall materials (25 games):
    0 misses, part 100% equal, implied damage distance equal to < 1e-4 m for 98.1% (the remaining 1.9% are exactly
    0.55^k scaled = bullets that passed through k other players first, handled by fpssim.penetration).
    The box frame is cs2-web's (which rotates by -yaw in its own axes); expressed in platform axes the local
    lateral / depth axes are (sin y, -cos y) / (cos y, sin y) and the head sphere centre sits 0.02 m along -(cos y, sin y).
    NOTE the previous version used the platform forward (cos y, -sin y): 1.3% of hits missed and 2% parts differed."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    f = (cy, sy); r = (sy, -cy)
    ox, oy = o[0] - pos[0], o[1] - pos[1]
    lo_ = (ox * r[0] + oy * r[1], o[2] - pos[2], ox * f[0] + oy * f[1])        # (lateral, up, depth)
    ld = (d[0] * r[0] + d[1] * r[1], d[2], d[0] * f[0] + d[1] * f[1])
    h = 1.10 if crouch else 1.80
    head = _sphere_hit(lo_, ld, (0.0, h - 0.16, -0.02), 0.23)
    up = _box_hit(lo_, ld, (-0.38, 0.45 if crouch else 0.75, -0.22), (0.38, h - 0.18, 0.22))
    low = _box_hit(lo_, ld, (-0.30, 0.0, -0.22), (0.30, 0.50 if crouch else 0.85, 0.22))
    body, part = None, None
    if up is not None and (low is None or up <= low): body, part = up, 'upper'
    elif low is not None: body, part = low, 'lower'
    if head is not None and head <= tmax and (body is None or head <= body): return head, 'head'
    return (body, part) if body is not None and body <= tmax else None
