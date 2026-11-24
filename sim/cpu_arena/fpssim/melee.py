"""Knife, ported from cs2-web shared/melee.js (commit e0e7e858): traceKnife, knifeDamage, knifeInterval.

UNVERIFIED against the platform: the public recordings contain no knife hit, so this is a literal port of the code.
Coordinates: platform metres, z up (cs2-web y up).  cs2-web's axis-aligned player box (x/z +-0.28, y +0.06 .. +1.1/1.8)
becomes x/y +-0.28, z +0.06 .. +1.1/1.8 here; the swept hull adds 0.4064 m on every axis.
Backstab: cs2-web tests (target forward) . (target - shooter)_horizontal / |..| > 0.475, i.e. the victim faces away from
the attacker; written here with the platform forward (cos yaw, -sin yaw).
"""
import math

KNIFE = dict(slash_range=48 * .0254, stab_range=32 * .0254, hull_radius=16 * .0254, slash_interval=.5, stab_interval=1.0,
             first_slash=40, repeat_slash=25, stab_damage=65, back_slash=90, back_stab=180, armor_scale=.85)
BOX_HALF = 0.28
BOX_BOTTOM = 0.06
STAND_H, CROUCH_H = 1.8, 1.1


def knife_interval(heavy):
    """seconds between swings (light 0.5 s -> 30 ticks, heavy 1.0 s -> 60 ticks)."""
    return KNIFE['stab_interval'] if heavy else KNIFE['slash_interval']


def knife_damage(heavy=False, backstab=False, first=True, armor=0.0):
    """cs2-web knifeDamage: floor(raw * (0.85 if armor > 0)).  Knife damage bypasses armor bookkeeping (armor value is
    not reduced in cs2-web swingKnife)."""
    if backstab: raw = KNIFE['back_stab'] if heavy else KNIFE['back_slash']
    else: raw = KNIFE['stab_damage'] if heavy else (KNIFE['first_slash'] if first else KNIFE['repeat_slash'])
    return int(math.floor(raw * (KNIFE['armor_scale'] if armor > 0 else 1.0)))


def _segment_box(o, d, lo, hi, rng):
    enter, exit_ = 0.0, rng
    for k in range(3):
        if abs(d[k]) < 1e-8:
            if o[k] < lo[k] or o[k] > hi[k]: return None
            continue
        a = (lo[k] - o[k]) / d[k]; b = (hi[k] - o[k]) / d[k]
        enter = max(enter, min(a, b)); exit_ = min(exit_, max(a, b))
        if enter > exit_: return None
    return enter


def direction(yaw, pitch):
    cp = math.cos(pitch)
    return (cp * math.cos(yaw), -cp * math.sin(yaw), math.sin(pitch))


def trace_knife(origin, dirv, shooter, players, heavy=False, raycast=None):
    """cs2-web traceKnife.
    origin: eye (x, y, z); dirv: unit view direction; shooter: dict(seat, team, x, y).
    players: iterable of dict(seat, team, alive, x, y, z, yaw, crouch[, protected]).
    raycast(origin, dir, max_dist) -> distance of first static-world hit or None.
    Returns dict(end, hit (seat or None), hit_world, backstab, distance)."""
    rng = KNIFE['stab_range'] if heavy else KNIFE['slash_range']
    rad = KNIFE['hull_radius']
    wall = raycast(origin, dirv, rng) if raycast else None
    nearest = wall if wall is not None else rng
    target = None; contact = None
    for p in players:
        if not p.get('alive', True) or p['seat'] == shooter['seat'] or p.get('team') == shooter.get('team') or p.get('protected'):
            continue
        # the sweep broadens aim forgiveness but cannot hit somebody behind us
        if (p['x'] - origin[0]) * dirv[0] + (p['y'] - origin[1]) * dirv[1] <= .02: continue
        h = CROUCH_H if p.get('crouch') else STAND_H
        lo = (p['x'] - BOX_HALF, p['y'] - BOX_HALF, p['z'] + BOX_BOTTOM)
        hi = (p['x'] + BOX_HALF, p['y'] + BOX_HALF, p['z'] + h)
        elo = tuple(v - rad for v in lo); ehi = tuple(v + rad for v in hi)
        dist = _segment_box(origin, dirv, elo, ehi, rng)
        if dist is None or dist > nearest: continue
        direct = _segment_box(origin, dirv, lo, hi, rng + rad * 2)
        if direct is None:
            pt = tuple(min(max(origin[k] + dirv[k] * dist, lo[k]), hi[k]) for k in range(3))
        else:
            pt = tuple(origin[k] + dirv[k] * direct for k in range(3))
        dd = math.dist(pt, origin)
        if dd > 1e-4 and raycast is not None:
            sight = tuple((pt[k] - origin[k]) / dd for k in range(3))
            ob = raycast(origin, sight, dd)
            if ob is not None and ob < dd - .002: continue
        nearest = dist; target = p; contact = pt
    end = contact or tuple(origin[k] + dirv[k] * nearest for k in range(3))
    if target is None:
        return dict(end=end, hit=None, hit_world=wall is not None, backstab=False, distance=nearest)
    dx = target['x'] - shooter['x']; dy = target['y'] - shooter['y']; L = math.hypot(dx, dy)
    ty = target.get('yaw', 0.0)
    back = L > 0 and (math.cos(ty) * dx - math.sin(ty) * dy) / L > .475
    return dict(end=end, hit=target['seat'], hit_world=False, backstab=back, distance=nearest, target=target)
