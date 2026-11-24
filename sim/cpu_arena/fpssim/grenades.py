"""Utilities / grenades: throw state machine, projectile physics, HE, flash, smoke, fire, decoy.

Port of the frozen cs2-web (apich/cs2-web @ e0e7e858) `server/grenades.js` + the throw / damage / flash parts
of `server/game.js`, re-timed to the platform's 60 Hz tick.  Validated against the platform recordings by
`validate/grenades.py`; see docs/grenades.md for what is exact and what is approximate.

Coordinates
-----------
Platform world: metres, z up, forward = (cos p cos y, -cos p sin y, sin p).  The platform runs the cs2-web code with
the axis map  world = (-z_cs, x_cs, y_cs)  and no translation (also on Dust2); verified from the fire-cell RNG seed
(`g.x*1000+g.z`) and the cell offsets on all 15 recorded fires.  Everything here works directly in
world coordinates; the only places where the cs2 axes matter are the 7-ray sweep order and the fire-cell RNG,
which use `_cs()`.

Timing (60 Hz)
--------------
* A throw executes on the physics tick T in which the release is processed, *after* that tick's movement, with
  the thrower's post-movement position / velocity / view.  The new projectile is advanced in the same tick.
  `born = T`, `detonate = T + fuse_ticks`, the recorded path starts at tick T-1 (the spawn point).
* Each tick advances every projectile by two 1/120 s sub-steps (cs2-web: min(1/120, dt) sub-steps).
* After a throw the thrower's `throw_ready = T + 24`.
* Smoke / decoy only detonate once resting; any projectile still flying after 720 ticks is removed.
* Fires damage every 12 ticks starting the tick after ignition; smokes last 1080 ticks, bloom over 108 ticks.

Interface (what Arena calls)
----------------------------
    gs = GrenadeSystem(geom, mesh=None, damage_cb=None, item_id_fn=None)
        geom: a TriMesh (Dust2), a MapGeom of a whitebox map (its .static is used), a static map dict, an Arena
        (uses .mesh if present, else .geom), or any object with  contact(origin, dir, length)->(dist, normal)|None
        and  raycast_dist(origin, dir, length)->dist|None.
        damage_cb(victim, attacker_seat, hp_damage, armor_absorbed, weapon) is called for every HE/fire/decoy
        hit at the moment it happens (so later hits in the same tick see updated hp/armor); if None the
        system does not touch hp/armor and the caller applies the returned `hit` events.
        item_id_fn(player, kind) -> item id string used for the projectile (e.g. pop the player's throwable).
    gs.handle_throw(player, action, tick) -> [throw event] or []   (call every live tick, after movement, before
        step(); action fields throw / throw_state / throw_mode).  On a throw it decrements player.nades[kind] and
        sets player.throw_ready = tick+24; the caller emits the returned throw event and the economy `consume`.
        These throw events are NOT repeated by step().
    events = gs.step(tick, players)   # once per *live* physics tick, after movement and handle_throw; returns
        bounce / detonate / flash / hit / decoy_pulse events.  cs2-web only ticks utilities while the round is
        live: in `post` (and buy) do not call step() - projectiles, fires and smokes freeze; clear() at round start.
    gs.smoke_blocks(p0, p1, tick=None) -> bool          (bloom + carves, cs2-web segmentIntersectsSmoke)
    gs.carve_bullet(start, end, tick)                    (call for every bullet trace through the world)
    gs.blind_until(seat) / gs.blind_start(seat)          (also written to the player objects)
    gs.is_primed(seat) -> kind or None
    gs.clear()                                           (new round)
    gs.smoke_spheres(tick=None) -> [(x,y,z,r_bloom)]     (compat for fpssim.visibility)

Player objects may be dicts or objects; fields are read from the object, then from `.ps`
(seat, team, alive, x, y, z, yaw, pitch, vx, vy, vz, crouch, armor, helmet, nades, throw_ready, protection_until).
`team` may be 0/1 or 'T'/'CT'; only equality matters.
"""
import math
import numpy as np

# ----------------------------------------------------------------------------------------------- constants
TICK_HZ = 60
DT = 1.0 / TICK_HZ
SUBSTEP = 1.0 / 120
GRAVITY = 8.128                      # cs2-web EQUIPMENT gravity (not the player gravity 20.32)
BOUNCE = 0.45
THROW_SPEED = 750 * 0.0254           # 19.05 m/s, x0.9 x (0.3+0.7*strength)
RADIUS = 0.06                        # projectile sweep radius
EYE_STAND, EYE_CROUCH = 1.62, 0.95
THROW_READY_TICKS = 24               # recorded: throw_ready == throw tick + 24 in 126/126 throws
BOUNCE_EVENT_MS = 90                 # bounce events at most every >90 ms (=> >= 6 ticks)
MAX_FLIGHT_TICKS = 720               # catalog smokegrenade.airborne_clear_ticks (cs2-web 12000 ms)
MAX_PROJECTILES = 40
FIRE_DAMAGE_TICKS = 12
DECOY_PULSE_TICKS = 54
WALKABLE = 0.7
STRENGTH = {'full': 1.0, 'lob': 0.5, 'drop': 0.0}
HE_BYPASS_ARMOR = True               # platform: explosion hit with armor 91.7 absorbed 0 (cs2-web would use armor)
REQUIRE_PRIME = True                 # release only throws a kind primed in an earlier tick (assumption)
FLOAT32_GROUND = float(np.float32(-0.002))

UTIL = {   # cs2-web EQUIPMENT + platform catalog (fuse/duration in ticks)
    'hegrenade':    dict(fuse=90,  radius=8.9, damage=99.0, effect='explosion'),
    'flashbang':    dict(fuse=90,  radius=30.0, duration=270, effect='flash'),
    'smokegrenade': dict(fuse=120, radius=4.5, duration=1080, effect='smoke'),
    'molotov':      dict(fuse=210, radius=3.4, duration=420, damage=8.0, effect='fire'),
    'incgrenade':   dict(fuse=210, radius=2.8, duration=330, damage=8.0, effect='fire'),
    'decoy':        dict(fuse=120, radius=2.0, duration=900, damage=5.0, effect='decoy'),
}
KINDS = tuple(UTIL)


def load_catalog(catalog):
    """Overwrite UTIL fuse/radius/duration/damage from a catalog dict (fpssim/catalog.json 'utilities')."""
    for k, c in (catalog or {}).items():
        if k in UTIL:
            u = UTIL[k]
            u['fuse'] = int(c.get('fuse_ticks', u['fuse']))
            u['radius'] = float(c.get('radius', u['radius']))
            if c.get('duration_ticks'):
                u['duration'] = int(c['duration_ticks'])
            if 'damage' in c:
                u['damage'] = float(c['damage'])


try:
    from .common import UTIL as _CAT   # noqa
    load_catalog(_CAT)
except Exception:
    pass


# ----------------------------------------------------------------------------------------------- helpers
def _get(p, k, default=None):
    if isinstance(p, dict):
        return p.get(k, default)
    if hasattr(p, k):
        return getattr(p, k)
    ps = getattr(p, 'ps', None)
    if ps is not None and hasattr(ps, k):
        return getattr(ps, k)
    return default


def _set(p, k, v):
    if isinstance(p, dict):
        p[k] = v
    else:
        setattr(p, k, v)


def forward(yaw, pitch):
    c = math.cos(pitch)
    return (c * math.cos(yaw), -c * math.sin(yaw), math.sin(pitch))


def _dist(a, b):
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2)


def point_segment_distance(p, s1, s2):
    dx, dy, dz = s2[0] - s1[0], s2[1] - s1[1], s2[2] - s1[2]
    L = dx * dx + dy * dy + dz * dz
    t = max(0.0, min(1.0, ((p[0] - s1[0]) * dx + (p[1] - s1[1]) * dy + (p[2] - s1[2]) * dz) / L)) if L > 0 else 0.0
    return math.sqrt((p[0] - (s1[0] + t * dx)) ** 2 + (p[1] - (s1[1] + t * dy)) ** 2 + (p[2] - (s1[2] + t * dz)) ** 2)


# 7-ray sweep offsets in cs2 order (0, +x_cs, -x_cs, +y_cs, -y_cs, +z_cs, -z_cs) mapped to world (-z_cs, x_cs, y_cs)
SWEEP_OFFSETS = ((0.0, 0.0, 0.0), (0.0, RADIUS, 0.0), (0.0, -RADIUS, 0.0), (0.0, 0.0, RADIUS), (0.0, 0.0, -RADIUS),
                 (-RADIUS, 0.0, 0.0), (RADIUS, 0.0, 0.0))


# ----------------------------------------------------------------------------------------------- world contact
class MeshContact:
    """First-hit (double sided) ray contact on a fpssim.trimesh.TriMesh, normal flipped to face the ray."""

    def __init__(self, mesh, cs_offset=(0.0, 0.0)):
        self.mesh = mesh
        n = np.cross(mesh.e1, mesh.e2)
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        self.n = n / np.where(ln > 0, ln, 1.0)
        self.cs_offset = cs_offset     # fire RNG seed uses world coords without the Dust2 offset (verified)

    def _first(self, o, d, L, idx=None):
        t, ids = self.mesh.intersect(o, (o[0] + d[0] * L, o[1] + d[1] * L, o[2] + d[2] * L), idx=idx)
        if len(t) == 0:
            return None
        return float(t[0]) * L, int(ids[0])

    def contact(self, o, d, L, idx=None):
        r = self._first(o, d, L, idx)
        if r is None:
            return None
        dist, tid = r
        n = self.n[tid]
        if n[0] * d[0] + n[1] * d[1] + n[2] * d[2] > 0:
            n = -n
        return dist, (float(n[0]), float(n[1]), float(n[2]))

    def candidates(self, o, d, L):
        return self.mesh.candidates(np.asarray(o, float), np.asarray(o, float) + np.asarray(d, float) * L)

    def raycast_dist(self, o, d, L):
        r = self._first(o, d, L)
        return None if r is None else r[0]


class BoxContact:
    """Whitebox collision: static boxes + stepped ramps + ground plane (z=-0.002 float32) inside map bounds.

    Matches the authored triangle mesh where the recordings can check it (box tops at float32 heights, ground at
    -0.002, stair step tops at z0+(i+0.5)(z1-z0)/n); the real mesh has extra trim / facade triangles that are not
    public, see docs/grenades.md."""

    def __init__(self, static):
        m = static
        f32 = lambda v: float(np.float32(v))
        rows = []
        for b in m.get('boxes', []):
            rows.append((f32(b['x'] - b['w'] / 2), f32(b['y'] - b['d'] / 2), f32(b['z']),
                         f32(b['x'] + b['w'] / 2), f32(b['y'] + b['d'] / 2), f32(b['z'] + b['h'])))
        stairs = {s['ramp_id']: s for s in m.get('stairs', [])}
        gz = float((m.get('ground') or {}).get('z', 0.0))
        for r in m.get('ramps', []):
            st = stairs.get(r['id'])
            n = st['steps'] if st else 1
            ax = r['axis']
            lo, ln = (r['x'] - r['w'] / 2, r['w']) if ax == 'x' else (r['y'] - r['d'] / 2, r['d'])
            for i in range(n):
                top = r['z0'] + (i + 0.5) * (r['z1'] - r['z0']) / n
                a0, a1 = lo + ln * i / n, lo + ln * (i + 1) / n
                if ax == 'x':
                    row = (a0, r['y'] - r['d'] / 2, gz - 0.002, a1, r['y'] + r['d'] / 2, top)
                else:
                    row = (r['x'] - r['w'] / 2, a0, gz - 0.002, r['x'] + r['w'] / 2, a1, top)
                rows.append(tuple(f32(v) for v in row))
        B = np.array(rows, float).reshape(-1, 6)
        self.lo = B[:, :3]; self.hi = B[:, 3:]
        self.ground = f32(gz - 0.002)
        bd = m.get('bounds') or {}
        self.gmin = bd.get('min', [-1e9, -1e9]); self.gmax = bd.get('max', [1e9, 1e9])
        self.cs_offset = (0.0, 0.0)

    def candidates(self, o, d, L):
        return None

    def _hits(self, o, d, L):
        """(dist, normal) of the first surface crossed in [0, L] or None (double sided: exits count)."""
        o = np.asarray(o, float); d = np.asarray(d, float)
        with np.errstate(divide='ignore', invalid='ignore'):
            inv = 1.0 / np.where(d == 0, 1e-300, d)
        ta = (self.lo - o) * inv; tb = (self.hi - o) * inv
        tmin = np.minimum(ta, tb); tmax = np.maximum(ta, tb)
        par = d == 0
        for a in np.where(par)[0]:
            inside = (o[a] >= self.lo[:, a]) & (o[a] <= self.hi[:, a])
            tmin[:, a] = np.where(inside, -np.inf, np.inf)
            tmax[:, a] = np.where(inside, np.inf, -np.inf)
        t0 = tmin.max(1); t1 = tmax.min(1)
        ok = t0 <= t1
        best = None
        # entering hits
        e = ok & (t0 >= 0) & (t0 <= L)
        if e.any():
            j = np.where(e)[0][np.argmin(t0[e])]
            ax = int(np.argmax(tmin[j]))
            n = [0.0, 0.0, 0.0]; n[ax] = -1.0 if d[ax] > 0 else 1.0
            best = (float(t0[j]), tuple(n))
        # exiting hits (origin inside a box)
        x = ok & (t0 < 0) & (t1 >= 0) & (t1 <= L)
        if x.any():
            j = np.where(x)[0][np.argmin(t1[x])]
            if best is None or t1[j] < best[0]:
                ax = int(np.argmin(tmax[j]))
                n = [0.0, 0.0, 0.0]; n[ax] = -1.0 if d[ax] > 0 else 1.0
                best = (float(t1[j]), tuple(n))
        if d[2] != 0:
            tg = (self.ground - o[2]) / d[2]
            if 0 <= tg <= L and (best is None or tg < best[0]):
                px, py = o[0] + d[0] * tg, o[1] + d[1] * tg
                if self.gmin[0] <= px <= self.gmax[0] and self.gmin[1] <= py <= self.gmax[1]:
                    best = (float(tg), (0.0, 0.0, 1.0 if d[2] < 0 else -1.0))
        return best

    def contact(self, o, d, L, idx=None):
        return self._hits(o, d, L)

    def raycast_dist(self, o, d, L):
        h = self._hits(o, d, L)
        return None if h is None else h[0]


def make_contact(geom, mesh=None):
    if mesh is not None:
        return MeshContact(mesh)
    if hasattr(geom, 'contact') and hasattr(geom, 'raycast_dist'):
        return geom
    if hasattr(geom, 'tri') and hasattr(geom, 'intersect'):
        return MeshContact(geom)
    if isinstance(geom, dict):
        st = geom.get('map', geom)
        if st.get('boxes') is not None:
            return BoxContact(st)
        raise ValueError('static map without boxes: pass mesh=TriMesh(...)')
    for attr in ('mesh', 'trimesh'):
        m = getattr(geom, attr, None)
        if m is not None and hasattr(m, 'tri'):
            return MeshContact(m)
    st = getattr(geom, 'static', None)
    if isinstance(st, dict) and st.get('boxes'):
        return BoxContact(st)
    g = getattr(geom, 'geom', None)
    if g is not None and g is not geom:
        return make_contact(g)
    raise ValueError('cannot build grenade world contact from %r' % type(geom))


# ----------------------------------------------------------------------------------------------- smoke test
def segment_intersects_smoke(a, b, s, now):
    """cs2-web segmentIntersectsSmoke with ages in ticks (now, s['created'] ticks)."""
    dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    L = dx * dx + dy * dy + dz * dz
    if L < 1e-6:
        return False
    sx, sy, sz = s['position']
    tc = max(0.0, min(1.0, ((sx - a[0]) * dx + (sy - a[1]) * dy + (sz - a[2]) * dz) / L))
    cd = math.sqrt((a[0] + tc * dx - sx) ** 2 + (a[1] + tc * dy - sy) ** 2 + (a[2] + tc * dz - sz) ** 2)
    age = max(0.0, (now - s['created']) / TICK_HZ) if (now and s['created']) else 10.0
    prog = min(1.0, age / 1.8)
    br = s['radius'] * (0.15 + 0.85 * (1 - (1 - prog) ** 3))
    if cd >= br:
        return False
    carves = s['carves']
    if not carves or not now:
        return True
    fx, fy, fz = a[0] - sx, a[1] - sy, a[2] - sz
    pA = L
    pB = 2 * (fx * dx + fy * dy + fz * dz)
    pC = (fx * fx + fy * fy + fz * fz) - br * br
    disc = pB * pB - 4 * pA * pC
    if disc <= 0:
        return False
    sq = math.sqrt(disc)
    t0 = max(0.0, (-pB - sq) / (2 * pA)); t1 = min(1.0, (-pB + sq) / (2 * pA))
    if t1 <= t0:
        return False
    for step in range(5):
        st = t0 + step / 4 * (t1 - t0)
        p = (a[0] + st * dx, a[1] + st * dy, a[2] + st * dz)
        carved = False
        for c in carves:
            cage = (now - c['created']) / TICK_HZ
            if cage < 0 or cage >= c['duration']:
                continue
            if c['type'] == 'explosion':
                rec = c.get('recover_duration') or 0.6
                full = c['duration'] - rec
                eff = c['radius'] if cage < full else c['radius'] * max(0.0, 1 - ((cage - full) / max(0.01, rec)) ** 2)
                if eff > 0.001 and _dist(p, c['position']) <= eff:
                    carved = True; break
            elif c['type'] == 'bullet':
                rec = c.get('recover_duration') or 0.25
                full = c['duration'] - rec
                eff = c['radius'] if cage < full else c['radius'] * max(0.0, 1 - (cage - full) / max(0.01, rec))
                if eff > 0.001 and point_segment_distance(p, c['start'], c['end']) <= eff:
                    carved = True; break
        if not carved:
            return True
    return False


# ----------------------------------------------------------------------------------------------- system
class Projectile:
    __slots__ = ('id', 'kind', 'owner', 'team', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'mode', 'strength', 'born',
                 'detonate', 'resting', 'ignite', 'bounces', 'last_bounce', 'path')

    def snapshot(self):
        return dict(id=self.id, kind=self.kind, owner=self.owner, team=self.team, position=[self.x, self.y, self.z],
                    velocity=[self.vx, self.vy, self.vz], born=self.born, detonate=self.detonate,
                    resting=self.resting, bounces=self.bounces, throw_mode=self.mode, throw_strength=self.strength)


PRIME_MIN_TICKS = 12                 # 200 ms minimum hold between prime and throw (cs2-web readyAt)


class GrenadeSystem:
    def __init__(self, geom, mesh=None, damage_cb=None, item_id_fn=None, path_ticks=36):
        self.world = make_contact(geom, mesh)
        self.item_id_fn = item_id_fn
        self.damage_cb = damage_cb
        self.path_ticks = path_ticks
        self.serial = 0
        self.primed = {}            # seat -> dict(kind, mode, tick, release)
        self.blind = {}             # seat -> (start, until)
        self.throw_ready = {}       # seat -> tick
        self.pending = []           # throw events produced by handle_throw, emitted by the next step()
        self.clear()

    # --------------------------------------------------------------------------------- bookkeeping
    def clear(self):
        """New round: drop every projectile / smoke / fire / decoy, primes and blindness."""
        self.projectiles = []; self.smokes = []; self.fires = []; self.decoys = []
        self.primed = {}; self.blind = {}; self.pending = []; self.events = []
        self.tick = 0

    def is_primed(self, seat):
        p = self.primed.get(seat)
        return p['kind'] if p else None

    def blind_until(self, seat):
        return self.blind.get(seat, (0, 0))[1]

    def blind_start(self, seat):
        return self.blind.get(seat, (0, 0))[0]

    def smoke_spheres(self, tick=None):
        now = self.tick if tick is None else tick
        out = []
        for s in self.smokes:
            if s['expires'] <= now:
                continue
            age = max(0.0, (now - s['created']) / TICK_HZ)
            prog = min(1.0, age / 1.8)
            out.append((*s['position'], s['radius'] * (0.15 + 0.85 * (1 - (1 - prog) ** 3))))
        return out

    def snapshot(self):
        return dict(projectiles=[g.snapshot() for g in self.projectiles], smokes=[dict(s) for s in self.smokes],
                    fires=[dict(f) for f in self.fires], decoys=[dict(d) for d in self.decoys])

    def _emit(self, typ, **kw):
        e = dict(type=typ, match_tick=self.tick); e.update(kw); self.events.append(e); return e

    # --------------------------------------------------------------------------------- geometry
    def _sweep(self, o, d, L):
        idx = self.world.candidates(o, d, L + RADIUS)
        best = None
        for k, off in enumerate(SWEEP_OFFSETS):
            oo = (o[0] + off[0], o[1] + off[1], o[2] + off[2])
            h = self.world.contact(oo, d, L + RADIUS, idx)
            if h is None:
                continue
            dist, n = h
            closing = -(d[0] * n[0] + d[1] * n[1] + d[2] * n[2])
            if closing < 0.00001:
                continue
            dd = max(0.0, dist - (RADIUS / closing if k == 0 else 0.0))
            if dd <= L and (best is None or dd < best[0]):
                best = (dd, n)
        return best

    def clear_sight(self, a, b):
        L = _dist(a, b)
        if L < 0.01:
            return True
        d = ((b[0] - a[0]) / L, (b[1] - a[1]) / L, (b[2] - a[2]) / L)
        h = self.world.raycast_dist(a, d, L)
        return h is None or h >= L - 0.03

    def smoke_blocks(self, p0, p1, tick=None):
        now = self.tick if tick is None else tick
        return any(s['expires'] > now and segment_intersects_smoke(p0, p1, s, now) for s in self.smokes)

    # --------------------------------------------------------------------------------- throwing
    def handle_throw(self, p, action, tick):
        """Throw state machine for one player and one physics tick (call every tick with the held action).

        prime  : start / keep holding `throw` (needs count>0); re-priming another kind replaces it.
        release: throw the primed kind on the first tick >= max(prime tick + PRIME_MIN_TICKS, throw_ready) (pending until then).
                 cs2-web game.js: readyAt = max(primedAt + 200 ms, nextShotAt); the platform doc says utility rules copy game.js
                 and every deadline is rounded up to the first 60 Hz tick -> 12 ticks. Our own training.jsonl agrees
                 (112/112 throws execute on the first tick the release applies, all >= 18 ticks after the prime).
        cancel : drop the prime.  Death / not live -> call cancel(seat) (Arena).
        Returns the events produced (the throw event); throws mutate p.nades / p.throw_ready."""
        seat = _get(p, 'seat')
        if not _get(p, 'alive', True):
            self.primed.pop(seat, None)
            return []
        a = action or {}
        kind = a.get('throw', 'none') or 'none'
        state = a.get('throw_state', 'idle') or 'idle'
        mode = a.get('throw_mode', 'full') or 'full'
        nades = _get(p, 'nades') or {}
        if state == 'cancel':
            self.primed.pop(seat, None)
        elif state == 'prime' and kind in UTIL:
            cur = self.primed.get(seat)
            if nades.get(kind, 0) > 0:
                if cur is None or cur['kind'] != kind or cur['release']:
                    self.primed[seat] = dict(kind=kind, mode=mode, tick=tick, release=False)
                else:
                    cur['mode'] = mode
        elif state == 'release':
            cur = self.primed.get(seat)
            if cur is None and not REQUIRE_PRIME and kind in UTIL and nades.get(kind, 0) > 0:
                cur = self.primed[seat] = dict(kind=kind, mode=mode, tick=tick, release=False)
            if cur is not None and (kind == 'none' or kind == cur['kind']) and (cur['tick'] < tick or not REQUIRE_PRIME):
                if not cur['release']:
                    cur['release'] = True; cur['mode'] = mode
        cur = self.primed.get(seat)
        out = []
        if cur is not None and cur['release']:
            ready = _get(p, 'throw_ready', None)
            if ready is None:
                ready = self.throw_ready.get(seat, 0)
            if tick >= ready and tick >= cur['tick'] + PRIME_MIN_TICKS:
                kind = cur['kind']
                if nades.get(kind, 0) <= 0:
                    self.primed.pop(seat, None)
                    return []
                self.primed.pop(seat, None)
                iid = self.item_id_fn(p, kind) if self.item_id_fn else None
                e = self.throw(p, kind, cur['mode'], tick, item_id=iid, emit=False)
                if e is not None:
                    nades[kind] = nades.get(kind, 0) - 1
                    if nades[kind] <= 0:
                        nades.pop(kind, None)
                    self.throw_ready[seat] = tick + THROW_READY_TICKS
                    try:
                        _set(p, 'throw_ready', tick + THROW_READY_TICKS)
                    except AttributeError:
                        pass
                    out.append(e)
        return out

    def cancel(self, seat):
        self.primed.pop(seat, None)

    def throw(self, p, kind, mode, tick, yaw=None, pitch=None, item_id=None, emit=True):
        """cs2-web throwGrenade from the thrower's current (post-movement) state.  Returns the throw event
        (also emitted by the next step() when emit=True).  Does not touch inventory / throw_ready.
        The projectile is advanced by the next step(tick)."""
        if kind not in UTIL or len(self.projectiles) >= MAX_PROJECTILES:
            return None
        self.tick = tick
        yaw = _get(p, 'yaw') if yaw is None else yaw
        pitch = _get(p, 'pitch') if pitch is None else pitch
        raw = max(-1.48, min(1.48, pitch))
        pa = raw + (math.pi / 18) * (1 - abs(raw) / (math.pi / 2))
        d = forward(yaw, pa)
        mode = mode if mode in STRENGTH else 'full'
        strength = STRENGTH[mode]
        speed = THROW_SPEED * 0.9 * (0.3 + 0.7 * strength)
        x, y, z = _get(p, 'x'), _get(p, 'y'), _get(p, 'z')
        oz = z + (EYE_CROUCH if _get(p, 'crouch') else EYE_STAND) - (1 - strength) * 0.3
        o = (x, y, oz)
        blk = self._sweep(o, d, 0.4)
        off = max(0.0, blk[0] - 0.01) if blk else 0.35
        g = Projectile()
        self.serial += 1
        g.id = item_id or 'grenade_%d' % self.serial
        g.kind = kind; g.owner = _get(p, 'seat'); g.team = _get(p, 'team')
        g.x, g.y, g.z = o[0] + d[0] * off, o[1] + d[1] * off, o[2] + d[2] * off
        vx, vy, vz = (_get(p, 'vx') or 0.0), (_get(p, 'vy') or 0.0), (_get(p, 'vz') or 0.0)
        g.vx = d[0] * speed + vx * 1.25; g.vy = d[1] * speed + vy * 1.25; g.vz = d[2] * speed + vz * 1.25
        g.mode = mode; g.strength = strength; g.born = tick; g.detonate = tick + UTIL[kind]['fuse']
        g.resting = False; g.ignite = False; g.bounces = 0; g.last_bounce = None
        g.path = [(tick - 1, (g.x, g.y, g.z))]
        self.projectiles.append(g)
        e = dict(type='throw', match_tick=tick, player=g.owner, item_id=g.id, kind=kind,
                 position=[g.x, g.y, g.z], throw_mode=mode)
        if emit:
            self.pending.append(e)
        return e

    # --------------------------------------------------------------------------------- physics
    def _advance(self, g, dt, sub_tick):
        if g.resting:
            return
        g.vz -= GRAVITY * dt * 0.5
        remaining = dt
        fire = UTIL[g.kind]['effect'] == 'fire'
        for _ in range(4):
            if remaining <= 1e-6:
                break
            sp = math.sqrt(g.vx * g.vx + g.vy * g.vy + g.vz * g.vz)
            if sp < 1e-6:
                break
            d = (g.vx / sp, g.vy / sp, g.vz / sp)
            travel = sp * remaining
            hit = self._sweep((g.x, g.y, g.z), d, travel)
            if hit is None:
                g.x += d[0] * travel; g.y += d[1] * travel; g.z += d[2] * travel
                break
            dist, n = hit
            g.x += d[0] * dist + n[0] * 0.002; g.y += d[1] * dist + n[1] * 0.002; g.z += d[2] * dist + n[2] * 0.002
            if n[2] > WALKABLE and fire:
                g.ignite = True; g.vx = g.vy = g.vz = 0.0
                return
            dot = g.vx * n[0] + g.vy * n[1] + g.vz * n[2]
            g.vx = (g.vx - 2 * dot * n[0]) * BOUNCE
            g.vy = (g.vy - 2 * dot * n[1]) * BOUNCE
            g.vz = (g.vz - 2 * dot * n[2]) * BOUNCE
            g.bounces += 1
            if abs(dot) > 1 and (g.last_bounce is None or (self.tick - g.last_bounce) * 1000.0 / TICK_HZ > BOUNCE_EVENT_MS):
                g.last_bounce = self.tick
                self._emit('bounce', player=g.owner, item_id=g.id, kind=g.kind, position=[g.x, g.y, g.z],
                           effect='grenade_bounce')
            if n[2] > WALKABLE and math.sqrt(g.vx * g.vx + g.vy * g.vy + g.vz * g.vz) < 1.2:
                g.vx = g.vy = g.vz = 0.0; g.resting = True
                return
            remaining *= max(0.0, 1 - dist / max(travel, 0.001))
        g.vz -= GRAVITY * dt * 0.5

    # --------------------------------------------------------------------------------- step
    def step(self, tick, players=()):
        """Advance one 60 Hz tick.  `players`: all players (dicts/objects).  Returns the list of events."""
        self.tick = now = tick
        self.events = self.pending; self.pending = []
        self.players = list(players)
        self.smokes = [s for s in self.smokes if s['expires'] > now]
        for s in self.smokes:
            if s['carves']:
                s['carves'] = [c for c in s['carves'] if (now - c['created']) / TICK_HZ < c['duration']]
        self._extinguish()
        for f in self.fires:
            if now >= f['next_damage']:
                f['next_damage'] = now + FIRE_DAMAGE_TICKS
                self._burn(f)
        for i in range(len(self.decoys) - 1, -1, -1):
            dc = self.decoys[i]
            if now >= dc['expires']:
                self._emit('detonate', player=dc['owner'], item_id=dc['id'], kind='decoy', position=list(dc['position']),
                           effect='explosion')
                self._explode(dc['position'], dc['owner'], dc['team'], 'decoy')
                del self.decoys[i]
            elif now >= dc['next_sound']:
                dc['next_sound'] = now + DECOY_PULSE_TICKS
                self._emit('decoy_pulse', player=dc['owner'], item_id=dc['id'], position=list(dc['position']))
        for i in range(len(self.projectiles) - 1, -1, -1):
            g = self.projectiles[i]
            rem = DT
            k = 0
            while rem > 1e-8 and not g.ignite:
                stp = min(SUBSTEP, rem)
                self._advance(g, stp, k)
                rem -= stp
                k += 1
                g.path.append((now - 1 + 0.5 * k, (g.x, g.y, g.z)))
            if len(g.path) > 2 * self.path_ticks + 8:
                g.path = g.path[-(2 * self.path_ticks + 1):]
            if not math.isfinite(g.x + g.y + g.z) or g.z < -250:
                del self.projectiles[i]; continue
            delayed = g.kind in ('smokegrenade', 'decoy')
            if g.ignite or (now >= g.detonate and (not delayed or g.resting)):
                self._detonate(g)
                del self.projectiles[i]
            elif now - g.born > MAX_FLIGHT_TICKS:
                del self.projectiles[i]
        out = self.events; self.events = []
        return out

    # --------------------------------------------------------------------------------- effects
    def _detonate(self, g):
        u = UTIL[g.kind]; now = self.tick; pos = [g.x, g.y, g.z]
        path = [dict(tick=t, position=list(p)) for t, p in g.path if t >= now - self.path_ticks]
        self._emit('detonate', player=g.owner, item_id=g.id, kind=g.kind, position=pos, effect='detonate', path=path)
        eff = u['effect']
        if eff == 'smoke':
            s = dict(id=g.id, owner=g.owner, team=g.team, kind='smokegrenade', position=[g.x, g.y, g.z + 1.3],
                     created=now, expires=now + u['duration'], radius=u['radius'], carves=[])
            self.smokes.append(s)
            if len(self.smokes) > 40:
                self.smokes.pop(0)
            self._extinguish()
        elif eff == 'flash':
            self._flash(g)
        elif eff == 'fire':
            if g.ignite:
                self._ignite(g)
            else:
                self._emit('detonate', player=g.owner, item_id=g.id, kind=g.kind, position=pos, effect='fire_failed')
        elif eff == 'decoy':
            self.decoys.append(dict(id=g.id, owner=g.owner, team=g.team, position=pos, created=now,
                                    expires=now + u['duration'], next_sound=now))
            if len(self.decoys) > 20:
                self.decoys.pop(0)
        else:
            self._disperse(pos)
            self._explode(pos, g.owner, g.team, g.kind)

    def _damage(self, victim, attacker, amount, kind, weapon, bypass_armor):
        armor = float(_get(victim, 'armor', 0.0) or 0.0)
        if not bypass_armor and armor > 0:
            hp = amount * 0.5; ab = (amount - hp) * 0.5        # armorRatio 1, no rounding (platform)
            if ab > armor:
                ab = armor; hp = amount - ab / 0.5
        else:
            hp, ab = amount, 0.0
        e = self._emit('hit', player=attacker, target=_get(victim, 'seat'), part='chest', kind=kind, amount=amount,
                       hp_damage=hp, armor_absorbed=ab, weapon=weapon)
        if self.damage_cb is not None:
            self.damage_cb(victim, attacker, hp, ab, weapon)
        return e

    def _hurtable(self, v, owner, team):
        if not _get(v, 'alive', True):
            return False
        if self.tick < (_get(v, 'protection_until', 0) or 0):
            return False
        return not (_get(v, 'team') == team and _get(v, 'seat') != owner)

    def _explode(self, pos, owner, team, kind):
        u = UTIL[kind]
        for v in self.players:
            if not self._hurtable(v, owner, team):
                continue
            tgt = (_get(v, 'x'), _get(v, 'y'), _get(v, 'z') + 0.9)
            d = _dist(pos, tgt)
            if d >= u['radius'] or not self.clear_sight(pos, tgt):
                continue
            self._damage(v, owner, u['damage'] * max(0.0, 1 - d / u['radius']), 'explosion', kind, HE_BYPASS_ARMOR)

    def _burn(self, f):
        u = UTIL[f['kind']]
        for v in self.players:
            if not self._hurtable(v, f['owner'], f['team']):
                continue
            x, y, z = _get(v, 'x'), _get(v, 'y'), _get(v, 'z')
            for c in f['cells']:
                if math.hypot(x - c[0], y - c[1]) < 0.75 and z < c[2] + 1.3 and z > c[2] - 0.4 and \
                        self.clear_sight((c[0], c[1], c[2] + 0.2), (x, y, z + 0.5)):
                    self._damage(v, f['owner'], u['damage'], 'fire', f['kind'], True)
                    break

    def _flash(self, g):
        u = UTIL['flashbang']; now = self.tick; gp = (g.x, g.y, g.z)
        for v in self.players:
            if not _get(v, 'alive', True):
                continue
            eye = (_get(v, 'x'), _get(v, 'y'), _get(v, 'z') + (EYE_CROUCH if _get(v, 'crouch') else EYE_STAND))
            d = _dist(eye, gp)
            if d >= u['radius'] or not self.clear_sight(eye, gp):
                continue
            fw = forward(_get(v, 'yaw') or 0.0, _get(v, 'pitch') or 0.0)
            dot = 1.0 if d < 0.01 else (fw[0] * (gp[0] - eye[0]) + fw[1] * (gp[1] - eye[1]) + fw[2] * (gp[2] - eye[2])) / d
            df = max(0.0, 1 - d / u['radius'])
            if dot > 0.5:
                alpha, hold, fade = 1.0, 2.2 * df, 2.3 * df
            elif dot > -0.2:
                fct = (dot + 0.2) / 0.7
                alpha, hold, fade = 0.5 + 0.5 * fct, 0.8 * fct * df, 1.8 * df
            else:
                alpha, hold, fade = 0.4 * df, 0.0, 0.6 * df
            dur = hold + fade
            if dur > 0.05:
                ticks = int(math.ceil(dur * TICK_HZ - 1e-9))
                seat = _get(v, 'seat')
                st, un = self.blind.get(seat, (0, 0))
                if now + ticks > un:
                    self.blind[seat] = (now, now + ticks)
                    try:
                        _set(v, 'blind_until', now + ticks); _set(v, 'blind_start', now)
                    except AttributeError:
                        pass
                self._emit('flash', target=seat, duration=ticks, hold=hold, fade=fade, alpha=alpha, exposure=alpha,
                           player=g.owner, item_id=g.id)

    def _cs(self, x, y):
        ox, oy = self.world.cs_offset
        return y - oy, -(x - ox)     # (x_cs, z_cs)

    def _ignite(self, g):
        u = UTIL[g.kind]
        cells = []
        xc, zc = self._cs(g.x, g.y)
        v = abs(xc * 1000 + zc)
        seed = int(math.trunc(v)) & 0xffffffff
        if seed >= 2 ** 31:
            seed -= 2 ** 32
        seed = seed or 1
        state = [seed]

        def rnd():
            state[0] = (state[0] * 1664525 + 1013904223) % 4294967296
            return state[0] / 4294967296

        for rr, cnt in ((0.1, 1), (0.65, 6), (1.30, 11), (1.95, 15), (2.60, 17), (3.20, 12)):
            if rr > u['radius']:
                continue
            i = 0
            while i < cnt and len(cells) < 60:
                ang = (i / cnt) * math.pi * 2 + rnd() * 0.45
                r = rr + (rnd() - 0.5) * 0.32
                i += 1
                if r > u['radius']:
                    continue
                ox, oz = math.cos(ang) * r, math.sin(ang) * r       # cs2 x / z offsets
                fx, fy, fz = g.x - oz, g.y + ox, g.z + 0.45          # world = (-z_cs, x_cs, y_cs)
                fl = self.world.contact((fx, fy, fz), (0.0, 0.0, -1.0), 1.3)
                if fl is None or fl[1][2] < WALKABLE:
                    continue
                cell = (fx, fy, fz - fl[0] + 0.025)
                if self.clear_sight((g.x, g.y, g.z + 0.25), (cell[0], cell[1], cell[2] + 0.25)):
                    cells.append(cell)
        if not cells:
            return
        now = self.tick
        f = dict(id=g.id, owner=g.owner, team=g.team, kind=g.kind, position=[g.x, g.y, g.z], cells=cells,
                 created=now, expires=now + u['duration'], next_damage=now)
        self.fires.append(f)
        if len(self.fires) > 12:
            self.fires.pop(0)
        self._extinguish()
        if f['cells']:
            self._emit('detonate', player=g.owner, item_id=g.id, kind=g.kind, position=[g.x, g.y, g.z],
                       effect='fire_started', expires=f['expires'], cells=[list(c) for c in f['cells']])

    def _extinguish(self):
        now = self.tick
        for f in self.fires:
            before = len(f['cells'])
            f['cells'] = [c for c in f['cells'] if not any(
                s['expires'] > now and _dist((c[0], c[1], c[2] + 0.4), s['position']) < s['radius'] for s in self.smokes)]
            if before and not f['cells']:
                self._emit('detonate', player=f['owner'], item_id=f['id'], kind=f['kind'], position=list(f['position']),
                           effect='fire_extinguished')
        self.fires = [f for f in self.fires if f['cells'] and f['expires'] > now]

    def _disperse(self, pos, radius=3.5, duration=2.5, recover=0.6):
        now = self.tick
        for s in self.smokes:
            if s['expires'] <= now:
                continue
            d = _dist(s['position'], pos)
            if d < s['radius'] + radius:
                self.serial += 1
                s['carves'].append(dict(id=self.serial, type='explosion', position=list(pos),
                                        radius=max(radius, s['radius'] + 0.5) if d < 2.0 else radius,
                                        created=now, duration=duration, recover_duration=recover))

    def carve_bullet(self, start, end, tick=None, radius=0.24, duration=0.7, recover=0.25):
        now = self.tick if tick is None else tick
        for s in self.smokes:
            if s['expires'] <= now:
                continue
            if point_segment_distance(s['position'], start, end) < s['radius'] + radius:
                self.serial += 1
                s['carves'].append(dict(id=self.serial, type='bullet', start=list(start), end=list(end), radius=radius,
                                        created=now, duration=duration, recover_duration=recover))
                if len(s['carves']) > 30:
                    s['carves'].pop(0)

    # --------------------------------------------------------------------------------- arena compat
    def on_noise(self, p, kind):
        pass

    def drain_sounds(self):
        pass
