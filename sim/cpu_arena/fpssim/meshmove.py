"""Player movement on a triangle collision mesh (Dust2, Mirage): port of cs2-web shared/physics.js stepPlayer +
shared/hull-collision.js (HullContact SAT), with the platform deviations measured on recordings (docs/movement_mesh.md).

The physics runs in the mesh's own (cs2-web) coordinates: y is up, the hull is the axis-aligned box
[x-0.3, x+0.3] x [y, y+h] x [z-0.3, z+0.3].  Platform world = basis @ raw + offset (static map collision[0]);
for Dust2 basis = [[0,0,-1],[1,0,0],[0,1,0]] (det -1), so world (x,y,z) = (100-rz, 100+rx, 20+ry) and the
platform yaw equals the cs2-web yaw (forward raw (-sin,-cos) -> world (cos,-sin)).  Working in raw coordinates keeps
the SAT axis order, tie-breaks and triangle normals (three.js Triangle.getNormal = (c-b)x(a-b)) identical to cs2-web.

Sequential order of the triangle contacts matters in hullIntersect (each contact pushes the box before the next
triangle is tested).  cs2-web visits triangles in three-mesh-bvh shapecast order (CENTER split, left child first), which
is the leaf order of the BVH, so the mesh is re-ordered once by replicating the BVH build (build_bvh_order) and
every query visits candidates in that order.

Interface: MeshMover().step(state, held_action, M, world, others, tick, objective_locked), same as movement.Mover.
World: MeshWorld (make_mesh_world(static_map, assets_dir)).
"""
import hashlib, math, os
import numpy as np
from .movement import Mover, MoveParams, PlayerState, wrap

PLAYER_RADIUS = 0.30
STAND_HEIGHT = 1.80
CROUCH_HEIGHT = 1.10
GRAVITY = 800 * .0254
JUMP_SPEED = 301.993378 * .0254
STEP_HEIGHT = .46
JUMP_BUFFER_TIME = .05
CROUCH_JUMP_LIFT = 18 * .0254
SV_FRICTION = 5.2
SV_STOPSPEED = 100 * .0254
SV_ACCELERATE = 5.5
SV_AIRACCELERATE = 12.0
SV_AIR_WISHSPEED_CAP = 30 * .0254
FLOAT32_EPSILON = 2.0 ** -24
INF = float('inf')


# ----------------------------------------------------------------------------------------------------------------------
# three-mesh-bvh 0.9 build replication (CENTER strategy, non-indexed geometry -> identity index, Hoare partition)
def build_bvh_order(tri, leaf_size=10, max_depth=32):
    """tri: [T,3,3] float64 (from float32).  Returns (order[T] original triangle id per BVH slot, leaf_lo[T,3],
    leaf_hi[T,3] float32-rounded bounds of the leaf that holds each slot)."""
    T = len(tri)
    mn = np.minimum(np.minimum(tri[:, 0], tri[:, 1]), tri[:, 2])
    mx = np.maximum(np.maximum(tri[:, 0], tri[:, 1]), tri[:, 2])
    half = (mx - mn) / 2
    cen = (mn + half).astype(np.float32).astype(np.float64)
    hx = (half + (np.abs(mn) + half) * FLOAT32_EPSILON).astype(np.float32).astype(np.float64)
    order = np.arange(T)
    leaf_lo = np.zeros((T, 3)); leaf_hi = np.zeros((T, 3))
    lo_all = cen - hx; hi_all = cen + hx
    stack = [(0, T, 0)]
    while stack:
        off, cnt, depth = stack.pop()
        sl = order[off:off + cnt]
        c = cen[sl]
        lo = lo_all[sl].min(0).astype(np.float32).astype(np.float64)
        hi = hi_all[sl].max(0).astype(np.float32).astype(np.float64)
        cmin = c.min(0); cmax = c.max(0)
        leaf = cnt <= leaf_size or depth >= max_depth
        if not leaf:
            d = cmax - cmin
            axis = -1; best = -INF
            for i in range(3):
                if d[i] > best: best = d[i]; axis = i
            pos = (cmin[axis] + cmax[axis]) / 2
            isl = c[:, axis] < pos
            nl = int(isl.sum())
            if nl == 0 or nl == cnt:
                leaf = True
            else:
                # Hoare partition: the k-th misplaced slot on the left swaps with the k-th misplaced slot from the right
                wl = np.nonzero(~isl[:nl])[0]
                wr = np.nonzero(isl[nl:])[0][::-1] + nl
                new = sl.copy()
                new[wl], new[wr] = sl[wr], sl[wl]
                order[off:off + cnt] = new
                # push right first so that left is processed first (order of processing does not matter for the result)
                stack.append((off + nl, cnt - nl, depth + 1))
                stack.append((off, nl, depth + 1))
        if leaf:
            leaf_lo[off:off + cnt] = lo; leaf_hi[off:off + cnt] = hi
    return order, leaf_lo, leaf_hi


def _dot3(n, v):
    return n[..., 0] * v[..., 0] + n[..., 1] * v[..., 1] + n[..., 2] * v[..., 2]


class MeshWorld:
    """Static triangle world in raw (cs2-web) coordinates, triangles stored in BVH visit order."""

    def __init__(self, positions_path, basis, offset, ladders=(), vertical=None, leaf_size=16, max_depth=32, cell=2.0,
                 cache_dir=None, order_mode='bvh'):
        raw = np.fromfile(positions_path, dtype='<f4')
        self.B = np.asarray(basis, np.float64); self.o = np.asarray(offset, np.float64)
        self.Bt = self.B.T.copy()
        tri = raw.astype(np.float64).reshape(-1, 3, 3)
        key = hashlib.sha256(raw.tobytes()).hexdigest()[:16] + f'_{leaf_size}_{max_depth}'
        cache = os.path.join(cache_dir, f'meshbvh_{key}.npz') if cache_dir else None
        if order_mode in ('orig', 'rev'):
            T0 = len(tri); order = np.arange(T0) if order_mode == 'orig' else np.arange(T0)[::-1].copy()
            mn = tri.min(1); mx = tri.max(1)
            llo = mn; lhi = mx
        elif cache and os.path.exists(cache):
            z = np.load(cache); order, llo, lhi = z['order'], z['llo'], z['lhi']
        else:
            order, llo, lhi = build_bvh_order(tri, leaf_size, max_depth)
            if cache:
                try:
                    os.makedirs(cache_dir, exist_ok=True); np.savez(cache, order=order, llo=llo, lhi=lhi)
                except OSError:
                    pass
        tri = tri[order]
        self.order = order
        self.tri = tri
        self.llo, self.lhi = llo, lhi
        a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
        # three.js Triangle.getNormal: (c-b) x (a-b), normalised by multiplyScalar(1/len)
        s = np.cross(c - b, a - b)
        ln = np.sqrt(s[:, 0] * s[:, 0] + s[:, 1] * s[:, 1] + s[:, 2] * s[:, 2])
        s = s * np.where(ln > 0, 1.0 / np.where(ln > 0, ln, 1.0), 0.0)[:, None]
        self.surface = s
        T = len(tri)
        ax = np.zeros((T, 13, 3))
        ax[:, 0] = s; ax[:, 1] = (0, 1, 0); ax[:, 2] = (1, 0, 0); ax[:, 3] = (0, 0, 1)
        for i, e in enumerate((b - a, c - b, a - c)):
            ax[:, 4 + i * 3] = np.stack([np.zeros(T), e[:, 2], -e[:, 1]], 1)
            ax[:, 5 + i * 3] = np.stack([-e[:, 2], np.zeros(T), e[:, 0]], 1)
            ax[:, 6 + i * 3] = np.stack([e[:, 1], -e[:, 0], np.zeros(T)], 1)
        ln = np.sqrt(ax[..., 0] * ax[..., 0] + ax[..., 1] * ax[..., 1] + ax[..., 2] * ax[..., 2])
        self.valid = ln >= 1e-10
        ax = ax * np.where(self.valid, 1.0 / np.where(self.valid, ln, 1.0), 0.0)[..., None]
        self.axes = ax
        self.absax = np.abs(ax)
        pa = _dot3(ax, a[:, None]); pb = _dot3(ax, b[:, None]); pc = _dot3(ax, c[:, None])
        self.pmin = np.minimum(np.minimum(pa, pb), pc)
        self.pmax = np.maximum(np.maximum(pa, pb), pc)
        self.tlo = tri.min(1); self.thi = tri.max(1)
        # horizontal (raw x,z) grid; cells hold slot ids ascending (= BVH visit order)
        self.cell = cell
        self.gx0, self.gz0 = self.tlo[:, 0].min(), self.tlo[:, 2].min()
        ix0 = ((self.tlo[:, 0] - self.gx0) // cell).astype(np.int64); ix1 = ((self.thi[:, 0] - self.gx0) // cell).astype(np.int64)
        iz0 = ((self.tlo[:, 2] - self.gz0) // cell).astype(np.int64); iz1 = ((self.thi[:, 2] - self.gz0) // cell).astype(np.int64)
        self.nx = int(ix1.max()) + 1; self.nz = int(iz1.max()) + 1
        nxs = ix1 - ix0 + 1; nzs = iz1 - iz0 + 1; tot = nxs * nzs
        tid = np.repeat(np.arange(T), tot)
        k = np.arange(tot.sum()) - np.repeat(np.cumsum(tot) - tot, tot)
        cx = ix0[tid] + k // nzs[tid]; cz = iz0[tid] + k % nzs[tid]
        cid = cx * self.nz + cz
        o = np.lexsort((tid, cid))
        self._flat = tid[o]
        self._start = np.searchsorted(cid[o], np.arange(self.nx * self.nz + 1))
        self.ladders = [dict(l) for l in (ladders or [])]
        self.vertical = vertical or {}

    # world <-> raw
    def to_raw(self, p):
        return self.Bt @ (np.asarray(p, np.float64) - self.o)

    def to_world(self, r):
        return self.B @ np.asarray(r, np.float64) + self.o

    def gather(self, lo, hi):
        """Local candidate set (slots ascending) whose triangle AABB overlaps the raw box [lo,hi]."""
        c = self.cell
        ix0 = max(0, int((lo[0] - self.gx0) // c)); ix1 = min(self.nx - 1, int((hi[0] - self.gx0) // c))
        iz0 = max(0, int((lo[2] - self.gz0) // c)); iz1 = min(self.nz - 1, int((hi[2] - self.gz0) // c))
        parts = []
        for ix in range(ix0, ix1 + 1):
            a = self._start[ix * self.nz + iz0]; b = self._start[ix * self.nz + iz1 + 1]
            if b > a: parts.append(self._flat[a:b])
        if not parts:
            return Local(self, np.zeros(0, np.int64))
        ids = np.unique(np.concatenate(parts))
        m = ((self.tlo[ids] <= hi) & (self.thi[ids] >= lo)).all(1)
        return Local(self, ids[m])


class Contact:
    __slots__ = ('normal', 'surface', 'depth', 'walkable')

    def __init__(self, normal, surface, depth, walkable):
        self.normal = normal; self.surface = surface; self.depth = depth; self.walkable = walkable


class Local:
    """Candidate triangles for one tick, with vectorised HullContact.intersect / sweepDown."""

    def __init__(self, w, ids):
        self.ids = ids
        self.n = len(ids)
        self.axes = w.axes[ids]; self.absax = w.absax[ids]; self.valid = w.valid[ids]
        self.pmin = w.pmin[ids]; self.pmax = w.pmax[ids]
        self.surface = w.surface[ids]
        self.tlo = w.tlo[ids]; self.thi = w.thi[ids]
        self.llo = w.llo[ids]; self.lhi = w.lhi[ids]

    # --- SAT overlap test of candidates [i0:] against box -> bool mask
    def _proj(self, box, sl):
        mn, mx = box
        cen = np.array([(mn[0] + mx[0]) * .5, (mn[1] + mx[1]) * .5, (mn[2] + mx[2]) * .5])
        hf = np.array([(mx[0] - mn[0]) * .5, (mx[1] - mn[1]) * .5, (mx[2] - mn[2]) * .5])
        ax = self.axes[sl]; aa = self.absax[sl]
        c = ax[..., 0] * cen[0] + ax[..., 1] * cen[1] + ax[..., 2] * cen[2]
        r = aa[..., 0] * hf[0] + aa[..., 1] * hf[1] + aa[..., 2] * hf[2]
        pos = self.pmax[sl] - c + r
        neg = c + r - self.pmin[sl]
        return pos, neg

    def overlap_mask(self, box, sl=slice(None)):
        pos, neg = self._proj(box, sl)
        sep = ((pos <= 1e-8) | (neg <= 1e-8)) & self.valid[sl]
        return ~sep.any(1)

    def contact(self, j, box):
        """HullContact.intersect for candidate j (assumed overlapping)."""
        pos, neg = self._proj(box, slice(j, j + 1))
        pos = pos[0]; neg = neg[0]; v = self.valid[j]
        depth = INF; best = -1; sign = 1.0
        for k in range(13):
            if not v[k]: continue
            p, q = pos[k], neg[k]
            ov = p if p < q else q
            if ov < depth - 1e-7:
                depth = ov; best = k; sign = 1.0 if p < q else -1.0
        n = self.axes[j, best] * sign
        s = self.surface[j]
        if s[0] * n[0] + s[1] * n[1] + s[2] * n[2] < 0: s = -s
        return Contact(n, s, depth, bool(s[1] >= .7 and n[1] >= .7))

    def hull_intersect(self, box):
        """MapCollision.hullIntersect.  box = (min[3], max[3]) lists.  Returns None or (normal, depth, contacts, walkable)."""
        if self.n == 0: return None
        mn0 = list(box[0]); mx0 = list(box[1])
        rmn = list(mn0); rmx = list(mx0)
        contacts = []
        i = 0; N = self.n
        while i < N:
            m = self.overlap_mask((rmn, rmx), slice(i, N))
            hits = np.nonzero(m)[0]
            j = i + int(hits[0]) if len(hits) else N
            if contacts and j > i:
                # triangles visited between contacts: walkable support against the ORIGINAL box, if the BVH leaf is visited
                sl = slice(i, j)
                blo = np.array([rmn[0] - .002, rmn[1] - .002, rmn[2] - .002]); bhi = np.array([rmx[0] + .002, rmx[1] + .002, rmx[2] + .002])
                vis = ((self.llo[sl] <= bhi) & (self.lhi[sl] >= blo)).all(1)
                if vis.any():
                    om = self.overlap_mask((mn0, mx0), sl) & vis
                    for k in np.nonzero(om)[0]:
                        ct = self.contact(i + int(k), (mn0, mx0))
                        if ct.walkable:
                            contacts.append(Contact(ct.normal, ct.surface, 0.0, True))
            if j >= N: break
            ct = self.contact(j, (rmn, rmx))
            contacts.append(ct)
            d = ct.depth + .00001
            for k in range(3):
                t = ct.normal[k] * d
                rmn[k] += t; rmx[k] += t
            i = j + 1
        dn = [rmn[0] - mn0[0], rmn[1] - mn0[1], rmn[2] - mn0[2]]
        depth = math.sqrt(dn[0] * dn[0] + dn[1] * dn[1] + dn[2] * dn[2])
        if depth > 1e-7:
            inv = 1 / depth
            return ((dn[0] * inv, dn[1] * inv, dn[2] * inv), depth, contacts, any(c.walkable for c in contacts))
        return None

    def sweep_down(self, box, max_drop):
        """MapCollision.sweepDown: distance of the first walkable support below box within max_drop, or None."""
        if self.n == 0: return None
        mn, mx = box
        blo = np.array([mn[0], mn[1] - max_drop, mn[2]]); bhi = np.array(mx)
        sel = np.nonzero(((self.llo <= bhi) & (self.lhi >= blo)).all(1) & ((self.tlo <= bhi) & (self.thi >= blo)).all(1))[0]
        if not len(sel): return None
        cen = np.array([(mn[0] + mx[0]) * .5, (mn[1] + mx[1]) * .5, (mn[2] + mx[2]) * .5])
        hf = np.array([(mx[0] - mn[0]) * .5, (mx[1] - mn[1]) * .5, (mx[2] - mn[2]) * .5])
        ax = self.axes[sel]; aa = self.absax[sel]; val = self.valid[sel]
        c = ax[..., 0] * cen[0] + ax[..., 1] * cen[1] + ax[..., 2] * cen[2]
        r = aa[..., 0] * hf[0] + aa[..., 1] * hf[1] + aa[..., 2] * hf[2]
        lo = self.pmin[sel] - r - c
        hi = self.pmax[sel] + r - c
        speed = -max_drop * ax[..., 1]
        K = len(sel)
        enter = np.full(K, -INF); ext = np.ones(K); ny = np.zeros(K); alive = np.ones(K, bool)
        with np.errstate(divide='ignore', invalid='ignore'):
            for k in range(13):
                v = val[:, k] & alive
                sp = speed[:, k]; l = lo[:, k]; h = hi[:, k]
                still = np.abs(sp) < 1e-10
                kill = v & still & ((l >= -1e-8) | (h <= 1e-8))
                alive &= ~kill
                mv = v & ~still & alive
                a1 = l / sp; a2 = h / sp
                first = np.minimum(a1, a2); last = np.maximum(a1, a2)
                upd = mv & (first > enter)
                enter = np.where(upd, first, enter)
                ny = np.where(upd, np.where(sp < 0, ax[:, k, 1], -ax[:, k, 1]), ny)
                ext = np.where(mv, np.minimum(ext, last), ext)
                alive &= ~(mv & (enter > ext + 1e-8))
        ok = alive & ~((enter < -.00001) | (enter > 1) | (ext < 0))
        idx = np.nonzero(ok)[0]
        if not len(idx): return None
        dist = np.maximum(0, enter[idx]) * max_drop
        walk = (self.surface[sel[idx], 1] >= .7) & (ny[idx] >= .7)
        distance = INF; walkable = False
        for d, wk in zip(dist, walk):
            if d < distance - .00001: distance = d; walkable = bool(wk)
            elif abs(d - distance) <= .00001: walkable = walkable or bool(wk)
        return distance if (distance < INF and walkable) else None


# ----------------------------------------------------------------------------------------------------------------------
class _P:
    """raw-coordinate player state used inside one physics tick (mirrors cs2-web field names; y up)."""
    __slots__ = ('x', 'y', 'z', 'vx', 'vy', 'vz', 'yaw', 'grounded', 'crouch', 'height', 'last_jump', 'jbuf')


def _box(p, height=None):
    h = p.height if height is None else height
    R = PLAYER_RADIUS
    return [p.x - R, p.y, p.z - R], [p.x + R, p.y + h, p.z + R]


def _translate(box, d):
    mn, mx = box
    return [mn[0] + d[0], mn[1] + d[1], mn[2] + d[2]], [mx[0] + d[0], mx[1] + d[1], mx[2] + d[2]]


def _collide(L, box, p, land='cs2web'):
    grounded = False
    for _ in range(3):
        hit = L.hull_intersect(box)
        if hit is None: break
        normal, depth, contacts, walkable = hit
        if walkable and p.vy <= .1:
            grounded = True
            if land == 'cs2web': p.vy = 0.0
        for ct in contacts:
            if ct.walkable and p.vy <= .1 and land == 'cs2web':
                grounded = True; p.vy = 0.0
            else:
                n = ct.surface if ct.walkable else ct.normal
                dot = p.vx * n[0] + p.vy * n[1] + p.vz * n[2]
                if dot < 0:
                    p.vx -= n[0] * dot; p.vy -= n[1] * dot; p.vz -= n[2] * dot
        t = depth + 0.0001
        box = _translate(box, (normal[0] * t, normal[1] * t, normal[2] * t))
    return grounded, box


def _sweep_down(L, box, max_drop):
    d = L.sweep_down(box, max_drop)
    if d is None: return None
    return _translate(box, (0.0, -max(0.0, d - .0001), 0.0))


class MeshMover(Mover):
    """cs2-web stepPlayer on a MeshWorld, platform conventions (M-scaled accel, yaw/pitch rates, held actions).
    Platform deviations / knobs (see docs/movement_mesh.md):
      crouch_factor  cs2-web 2.15/6 (crouch wins over walk)
      air_accel_base 'wish' (platform, 12*wishspeed*dt, Source formula) or 'M' (cs2-web: 12*baseSpeed*dt)
    """

    def __init__(self, params=None, crouch_factor=2.15 / 6.0, walk_factor=2.7 / 6.0, crouch_walk_factor=None,
                 air_accel_base='wish', jump_buffer=True, lock_crouch=True, step_mode='cs2web',
                 land_mode='cs2web', clamp=True, jump_mode='cs2web'):
        super().__init__(params)
        self.crouch_factor = crouch_factor
        self.walk_factor = walk_factor
        self.crouch_walk_factor = crouch_walk_factor
        self.air_accel_base = air_accel_base
        self.jump_buffer = jump_buffer
        self.lock_crouch = lock_crouch
        self.step_mode = step_mode
        self.land_mode = land_mode
        self.clamp = clamp
        self.jump_mode = jump_mode
        self.trace = None            # debug: list of per-substep dicts when not None
        self.fixed_local = None      # optional precomputed Local (validation: one gather per 3-tick block)

    def local_for_block(self, world, s, ticks=3, margin=0.6):
        """Candidate set covering `ticks` ticks of motion from world-state s (for repeated evaluation of one block)."""
        r = world.to_raw((s.x, s.y, s.z))
        sp = math.hypot(s.vx, s.vy) + 6.5
        mh = sp * ticks * self.p.dt + margin
        vdn = max(0.0, -s.vz) * ticks * self.p.dt + 1.0 + margin
        R = PLAYER_RADIUS
        return world.gather(np.array([r[0] - R - mh, r[1] - vdn, r[2] - R - mh]),
                            np.array([r[0] + R + mh, r[1] + STAND_HEIGHT + 0.6 + margin, r[2] + R + mh]))

    def _update_crouch(self, L, p, wanted):
        if bool(wanted) == p.crouch: return False
        oh = CROUCH_HEIGHT if p.crouch else STAND_HEIGHT; nh = CROUCH_HEIGHT if wanted else STAND_HEIGHT
        d = oh - nh
        lift = math.copysign(1.0, d) * min(abs(d), CROUCH_JUMP_LIFT) if d != 0 else 0.0
        y = p.y + (0.0 if p.grounded else lift)
        R = PLAYER_RADIUS
        test = ([p.x - R, y, p.z - R], [p.x + R, y + nh, p.z + R])
        if not wanted:
            test = _translate(test, (0.0, .0002, 0.0))
            hit = L.hull_intersect(test)
            if hit is not None and any(c.depth > .0003 for c in hit[2]): return True
        p.y = y; p.crouch = bool(wanted); p.height = nh
        return False

    def step(self, s, a, M, world, others=(), tick=0, objective_locked=False):
        if not s.alive: return s
        prm = self.p; dt = prm.dt
        get = a.get
        fwd = max(-1.0, min(1.0, float(get('move_forward', 0.0) or 0.0)))
        rgt = max(-1.0, min(1.0, float(get('move_right', 0.0) or 0.0)))
        jump = bool(get('jump', False))
        want_crouch = bool(get('crouch', False) or get('duck', False))
        walk = bool(get('walk', False))
        s.yaw = wrap(s.yaw + float(get('yaw_rate', 0.0) or 0.0) * prm.yaw_rate * dt)
        s.pitch = max(-prm.pitch_limit, min(prm.pitch_limit, s.pitch + float(get('pitch_rate', 0.0) or 0.0) * prm.pitch_rate * dt))
        if s.movement_mode == 'ladder':
            return self._ladder_step(s, a, world, tick)
        if get('ladder', 'none') == 'attach' and tick >= s.reattach_until and world.ladders:
            if self._try_attach(s, world, tick): return s
        # ---- to raw coordinates
        r = world.Bt @ (np.array([s.x, s.y, s.z]) - world.o)
        v = world.Bt @ np.array([s.vx, s.vy, s.vz])
        p = _P()
        p.x, p.y, p.z = float(r[0]), float(r[1]), float(r[2])
        p.vx, p.vy, p.vz = float(v[0]), float(v[1]), float(v[2])
        p.yaw = s.yaw; p.grounded = bool(s.grounded); p.crouch = bool(s.crouch)
        p.height = CROUCH_HEIGHT if p.crouch else STAND_HEIGHT
        p.last_jump = bool(s.last_jump); p.jbuf = float(getattr(s, 'jump_buffer', 0.0) or 0.0)
        x0w, y0w = s.x, s.y
        # ---- candidate triangles for the whole tick
        sp = math.sqrt(p.vx * p.vx + p.vz * p.vz)
        mh = sp * dt + 0.45; vdn = max(0.0, -p.vy) * dt + 1.2; vup = max(0.0, p.vy) * dt + 0.7
        R = PLAYER_RADIUS
        L = self.fixed_local
        if L is None:
            L = world.gather(np.array([p.x - R - mh, p.y - vdn, p.z - R - mh]),
                             np.array([p.x + R + mh, p.y + STAND_HEIGHT + vup, p.z + R + mh]))
        if objective_locked:
            p.vx = p.vy = p.vz = 0.0
            if self.lock_crouch:
                p.crouch = True; p.height = CROUCH_HEIGHT
            p.jbuf = 0.0; p.last_jump = jump
            self._write(s, p, world, x0w, y0w)
            s.stand_blocked = False
            return s
        if p.vy > .1: p.grounded = False
        pressed = jump and not p.last_jump
        if pressed and self.jump_buffer: p.jbuf = JUMP_BUFFER_TIME
        p.last_jump = jump
        p.jbuf = max(0.0, p.jbuf)
        jumped = False

        def try_jump():
            nonlocal jumped
            if p.grounded and (p.jbuf > 0 if self.jump_buffer else (pressed and not jumped)):
                p.vy = JUMP_SPEED; p.grounded = False; p.jbuf = 0.0; jumped = True
                return True
            return False
        if self.jump_mode == 'cs2web': try_jump()
        blocked = self._update_crouch(L, p, want_crouch)
        s.stand_blocked = bool(blocked)
        if p.grounded and not jumped:
            speed = math.hypot(p.vx, p.vz)
            if speed < .00254:
                p.vx = 0.0; p.vz = 0.0
            else:
                control = SV_STOPSPEED if speed < SV_STOPSPEED else speed
                drop = control * SV_FRICTION * dt
                ns = max(0.0, speed - drop)
                if ns != speed:
                    f = ns / speed; p.vx *= f; p.vz *= f
        mag = math.hypot(fwd, rgt)
        if mag > 0:
            sy, cy = math.sin(p.yaw), math.cos(p.yaw)
            wdx = (-sy * fwd + cy * rgt) / mag
            wdz = (-cy * fwd - sy * rgt) / mag
            base = M
            if p.crouch and walk and self.crouch_walk_factor is not None: fac = self.crouch_walk_factor
            elif p.crouch: fac = self.crouch_factor
            elif walk: fac = self.walk_factor
            else: fac = 1.0
            wish = fac * base * min(1.0, mag)
            if p.grounded and not jumped:
                cur = p.vx * wdx + p.vz * wdz
                add = wish - cur
                if add > 0:
                    acc = min(add, SV_ACCELERATE * base * dt)
                    p.vx += acc * wdx; p.vz += acc * wdz
            else:
                aw = min(wish, SV_AIR_WISHSPEED_CAP)
                cur = p.vx * wdx + p.vz * wdz
                add = aw - cur
                if add > 0:
                    ab = base if self.air_accel_base == 'M' else wish
                    acc = min(add, SV_AIRACCELERATE * ab * dt)
                    p.vx += acc * wdx; p.vz += acc * wdz
        if self.jump_mode == 'after_accel': try_jump()
        # ---- player contact (fitted platform rule, see movement.Mover)
        touching = bool(others) and self._touching(s, others)
        pinned = touching and ((p.grounded and not jumped) or s.pinned)
        s.pinned = pinned
        # ---- substeps
        n = max(3, math.ceil(dt * 180)); h = dt / n
        incoming = math.hypot(p.vx, p.vz)
        for _ in range(n):
            try_jump()
            was_g = p.grounded
            hs = math.hypot(p.vx, p.vz)
            p.vy -= GRAVITY * h * .5
            c = _box(p)
            before = c
            mx_, mz_ = (0.0, 0.0) if pinned else (p.vx * h, p.vz * h)
            my_ = p.vy * h
            c = _translate(c, (mx_, my_, mz_))
            hit = L.hull_intersect(c)
            ground = False
            if hit is not None and hit[1] * math.hypot(hit[0][0], hit[0][2]) > .0001 and was_g and not jumped \
                    and math.hypot(mx_, mz_) > .001 and self._step_ok(hit):
                stepped = _translate(before, (mx_, STEP_HEIGHT, mz_))
                if L.hull_intersect(stepped) is None:
                    sup = _sweep_down(L, stepped, STEP_HEIGHT + .06)
                    if sup is not None and sup[0][1] >= before[0][1] - .001:
                        c = sup; p.vy = 0.0; ground = True
            g2, c = _collide(L, c, p, self.land_mode)
            ground = g2 or ground
            if not ground and was_g and not jumped and p.vy <= 0:
                sup = _sweep_down(L, c, STEP_HEIGHT)
                if sup is not None: c = sup; ground = True
            p.x = (c[0][0] + c[1][0]) / 2; p.y = c[0][1]; p.z = (c[0][2] + c[1][2]) / 2
            p.grounded = ground
            if ground:
                p.vy = 0.0
                a2 = math.hypot(p.vx, p.vz)
                if self.clamp and a2 > hs and a2 > 0:
                    k = hs / a2; p.vx *= k; p.vz *= k
            else:
                p.vy -= GRAVITY * h * .5
            p.jbuf = max(0.0, p.jbuf - h)
            if self.trace is not None:
                self.trace.append(dict(hit=None if hit is None else (hit[0], hit[1], [(tuple(np.round(q.normal, 4)), q.depth, q.walkable) for q in hit[2]]),
                                       pos=(p.x, p.y, p.z), v=(p.vx, p.vy, p.vz), ground=ground))
        if p.grounded and self.clamp:
            sp2 = math.hypot(p.vx, p.vz)
            if sp2 > incoming and sp2 > 0:
                k = incoming / sp2; p.vx *= k; p.vz *= k
        self._write(s, p, world, x0w, y0w)
        if others and not pinned:
            self._separate(s, others, x0w, y0w)
        s.movement_mode = 'ground' if s.grounded else 'air'
        return s

    def _step_ok(self, hit):
        m = self.step_mode
        if m == 'cs2web': return True
        if m == 'nonwalk': return not hit[3]
        if m == 'never': return False
        return True

    def _write(self, s, p, world, x0w, y0w):
        w = world.B @ np.array([p.x, p.y, p.z]) + world.o
        v = world.B @ np.array([p.vx, p.vy, p.vz])
        s.x, s.y, s.z = float(w[0]), float(w[1]), float(w[2])
        s.vx, s.vy, s.vz = float(v[0]), float(v[1]), float(v[2])
        s.grounded = p.grounded; s.crouch = p.crouch; s.last_jump = p.last_jump; s.jump_buffer = p.jbuf
        s.step_distance += math.hypot(s.x - x0w, s.y - y0w)

    def _air_z(self, s, world, h):      # ladder detach: leave in the air, the next mesh tick integrates the fall
        s.grounded = False


def make_mesh_world(map_dict, assets_dir, positions_name=None, cache_dir=None, **kw):
    """MeshWorld from a static map dict (collision[0]: positions/basis/offset).  The positions file is looked up as
    assets_dir/<positions_name or collision path>; for Dust2 the server copy is ref/assets/map_positions.f32."""
    col = map_dict['collision'][0]
    cands = [positions_name, col['positions']['path'], 'map_positions.f32']
    path = next((os.path.join(assets_dir, c) for c in cands if c and os.path.exists(os.path.join(assets_dir, c))), None)
    if path is None:
        raise FileNotFoundError(f'no collision mesh for {map_dict.get("id")} in {assets_dir}')
    return MeshWorld(path, col['basis'], col['offset'], ladders=map_dict.get('ladders', []),
                     vertical=map_dict.get('vertical_rules'), cache_dir=cache_dir, **kw)
