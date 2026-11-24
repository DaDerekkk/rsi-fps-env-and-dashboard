"""Enemy visibility for the legal observation (`visible` + `parts`).

Rule (inferred from the recorded per-seat visibility bitmasks of 25 games, see docs/visibility.md):
  observer alive and not blind (blind_start <= tick < blind_until  -> sees nothing);
  horizontal angle between the observer yaw and the direction to the target's feet (xy) <= 55 deg;
  sample points on the target (cs2-web hit volumes, H = 1.8 standing / 1.1 crouched):
      head    = feet + H - 0.16        (cs2-web head sphere centre: 1.64 / 0.94)
      chest   = feet + (lo + H-0.18)/2 (centre of the upper-body box, lo = 0.75 / 0.45: 1.185 / 0.685)
      abdomen = feet + 0.3 H           (0.54 / 0.33; fitted)
  a point counts if |elevation(eye->point) - pitch| <= 45 deg, the static world does not block eye->point and
  no smoke blocks it (cs2-web segmentIntersectsSmoke via smoke_fn);
  visible iff at least one point counts; `parts` = the counting points.  Eye = feet + 1.62 (0.95 crouched).
Fires do not block sight.  Teammates follow the same rule in the recordings (not part of the observation).
"""
import math

import numpy as np

from .grenades import _get, BoxContact

HALF_H = math.radians(55.0)
HALF_V = math.radians(45.0)
EYE = {False: 1.62, True: 0.95}
BODY_H = {False: 1.8, True: 1.1}
UPPER_LO = {False: 0.75, True: 0.45}


def part_heights(crouch):
    h = BODY_H[bool(crouch)]
    return (('head', h - 0.16), ('chest', (UPPER_LO[bool(crouch)] + h - 0.18) / 2), ('abdomen', 0.3 * h))


class LOS:
    """Static line of sight on a TriMesh (Dust2/Mirage) or the whitebox box model (static dict / MapGeom)."""

    def __init__(self, geom):
        self.mesh = self.box = None
        if hasattr(geom, 'tri') and hasattr(geom, 'intersect'):
            self.mesh = geom
        elif isinstance(geom, BoxContact):
            self.box = geom
        elif isinstance(geom, dict):
            self.box = BoxContact(geom.get('map', geom))
        elif isinstance(getattr(geom, 'static', None), dict) and geom.static.get('boxes'):
            self.box = BoxContact(geom.static)
        elif hasattr(geom, 'los'):
            self.fallback = geom
        else:
            raise ValueError('unsupported geometry for visibility: %r' % type(geom))

    def clear_many(self, eye, pts):
        if self.mesh is not None:
            idx = self.mesh.candidates(np.asarray(eye, float), np.asarray(pts[0], float))
            return [len(self.mesh.intersect(eye, p, idx=idx)[0]) == 0 for p in pts]
        if self.box is not None:
            out = []
            for p in pts:
                dv = np.asarray(p, float) - eye; L = float(np.linalg.norm(dv))
                out.append(L < 1e-9 or self.box.raycast_dist(eye, dv / L, L) is None)
            return out
        return [self.fallback.los(eye, p) for p in pts]


_LOS_CACHE = {}


def _los_for(geom):
    if isinstance(geom, LOS):
        return geom
    k = id(geom)
    if k not in _LOS_CACHE:
        _LOS_CACHE[k] = (geom, LOS(geom))
    return _LOS_CACHE[k][1]


def visible_parts_of(los, obs, tgt, smoke_fn=None, tick=None):
    ci = bool(_get(obs, 'crouch'))
    ex, ey, ez = _get(obs, 'x'), _get(obs, 'y'), _get(obs, 'z') + EYE[ci]
    yaw, pitch = _get(obs, 'yaw') or 0.0, _get(obs, 'pitch') or 0.0
    tx, ty, tz = _get(tgt, 'x'), _get(tgt, 'y'), _get(tgt, 'z')
    dx, dy = tx - ex, ty - ey
    dxy = math.hypot(dx, dy)
    az = math.atan2(-dy, dx)
    if abs(math.atan2(math.sin(az - yaw), math.cos(az - yaw))) > HALF_H:
        return []
    cand = []
    for part, hz in part_heights(_get(tgt, 'crouch')):
        p = (tx, ty, tz + hz)
        if abs(math.atan2(p[2] - ez, dxy) - pitch) <= HALF_V:
            cand.append((part, p))
    if not cand:
        return []
    eye = np.array([ex, ey, ez])
    ok = los.clear_many(eye, [p for _, p in cand])
    out = []
    for (part, p), c in zip(cand, ok):
        if c and not (smoke_fn is not None and smoke_fn((ex, ey, ez), p, tick)):
            out.append(part)
    return out


def is_blind(p, tick):
    if tick is None:
        return False
    return (_get(p, 'blind_start', 0) or 0) <= tick < (_get(p, 'blind_until', 0) or 0)


def visible_enemies(players, geom_or_mesh, smoke_fn=None, tick=None, teammates=False):
    """players: list of objects/dicts (seat, team, alive, x, y, z, yaw, pitch, crouch, blind_until, blind_start).
    geom_or_mesh: TriMesh, whitebox static dict / MapGeom / BoxContact, or a prepared LOS.
    smoke_fn(p0, p1, tick) -> bool, e.g. GrenadeSystem.smoke_blocks.
    Returns {seat: [(enemy_seat, parts)]} (parts subset of head/chest/abdomen, never empty)."""
    los = _los_for(geom_or_mesh)
    res = {}
    for o in players:
        s = _get(o, 'seat')
        res[s] = []
        if not _get(o, 'alive', True) or is_blind(o, tick):
            continue
        for t in players:
            if t is o or not _get(t, 'alive', True):
                continue
            if (_get(t, 'team') == _get(o, 'team')) and not teammates:
                continue
            parts = visible_parts_of(los, o, t, smoke_fn, tick)
            if parts:
                res[s].append((_get(t, 'seat'), parts))
    return res


# ---------------------------------------------------------------- compat with the old arena call
def visible_parts(geom, eye, yaw, pitch, foot, crouch, smokes=(), blind=False):
    """Old signature (eye given).  `smokes` is ignored here - pass smoke_fn through visible_enemies instead."""
    if blind:
        return []
    obs = dict(x=eye[0], y=eye[1], z=eye[2] - 1.62, yaw=yaw, pitch=pitch, crouch=False)
    tgt = dict(x=foot[0], y=foot[1], z=foot[2], crouch=crouch)
    return visible_parts_of(_los_for(geom), obs, tgt)
