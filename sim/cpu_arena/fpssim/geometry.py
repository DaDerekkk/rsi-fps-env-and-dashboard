"""Static-world geometry: segment raycast / line of sight.

Whitebox maps (borgo/molo/nevaio): analytic boxes + wedge ramps + ground plane.
Dust2/Mirage: approximated from fps_platform/kit mapdata (wall/cover voxel contours + walkable cell
heights).  Both are exposed through MapGeom.

Coordinates: metres, z up.  A "solid" is an axis-aligned box or a wedge (box clipped by a sloped
plane).  Ray tests use the slab method, vectorised over candidate solids selected by a uniform
2D grid.
"""
import json, os
import numpy as np

EYE_STAND = 1.62      # refined against fire 'from' by validate/los.py
EYE_CROUCH = 0.95     # fire from-z minus foot z, crouched (measured 0.9501)
BODY_STAND_H, BODY_CROUCH_H = 1.8, 1.1


class MapGeom:
    def __init__(self, map_id, lo, hi, slope=None, material=None, ground_z=0.0, ground_material='concrete',
                 bounds=None, static=None, floor=None, cell=4.0):
        """lo,hi: [N,3] solid AABBs.  slope: [N,5] rows (axis(0=x,1=y), z0, z1, umin, umax) or NaN rows for plain boxes
        (wedge top surface z = z0 + (z1-z0)*(c-umin)/(umax-umin), c=coordinate along axis; the solid is
        the AABB cut by z <= surface)."""
        self.map_id = map_id
        self.lo = np.asarray(lo, float).reshape(-1, 3)
        self.hi = np.asarray(hi, float).reshape(-1, 3)
        n = len(self.lo)
        self.slope = np.full((n, 5), np.nan) if slope is None else np.asarray(slope, float).reshape(-1, 5)
        self.material = list(material) if material is not None else ['stone'] * n
        self.ground_z = ground_z
        self.ground_material = ground_material
        self.bounds = bounds
        self.static = static or {}
        self.floor = floor            # optional FloorField (dust2/mirage)
        self.is_wedge = ~np.isnan(self.slope[:, 0])
        self._build_grid(cell)

    # ---------------------------------------------------------------- construction
    @classmethod
    def from_static(cls, m, kit_dir=None):
        mid = m['id']
        b = m.get('bounds')
        if m.get('boxes') or m.get('ramps'):
            lo, hi, slope, mat = [], [], [], []
            for bx in m['boxes']:
                lo.append([bx['x'] - bx['w'] / 2, bx['y'] - bx['d'] / 2, bx['z']])
                hi.append([bx['x'] + bx['w'] / 2, bx['y'] + bx['d'] / 2, bx['z'] + bx['h']])
                slope.append([np.nan] * 5); mat.append(bx.get('material', 'stone'))
            for r in m['ramps']:
                ax = 0 if r['axis'] == 'x' else 1
                cx, cy, w, d = r['x'], r['y'], r['w'], r['d']
                l = [cx - w / 2, cy - d / 2, 0.0]
                h = [cx + w / 2, cy + d / 2, max(r['z0'], r['z1'])]
                l[2] = m['ground']['z'] if 'ground' in m else 0.0
                lo.append(l); hi.append(h)
                umin = l[ax]; umax = h[ax]
                slope.append([ax, r['z0'], r['z1'], umin, umax]); mat.append(r.get('material', 'stone'))
            gz = m.get('ground', {}).get('z', 0.0)
            return cls(mid, lo, hi, slope, mat, ground_z=gz,
                       ground_material=m.get('ground', {}).get('material', 'concrete'), bounds=b, static=m)
        # dust2 / mirage approximation
        from .geometry_voxel import build_voxel_geom
        return build_voxel_geom(cls, m, kit_dir)

    # ---------------------------------------------------------------- grid
    def _build_grid(self, cell):
        self.cell = cell
        if len(self.lo) == 0:
            self.gx0 = self.gy0 = 0; self.gnx = self.gny = 1; self.cells = {}; return
        self.gx0 = float(self.lo[:, 0].min()) - 1; self.gy0 = float(self.lo[:, 1].min()) - 1
        gx1 = float(self.hi[:, 0].max()) + 1; gy1 = float(self.hi[:, 1].max()) + 1
        self.gnx = int((gx1 - self.gx0) / cell) + 1; self.gny = int((gy1 - self.gy0) / cell) + 1
        cells = {}
        ix0 = np.floor((self.lo[:, 0] - self.gx0) / cell).astype(int); ix1 = np.floor((self.hi[:, 0] - self.gx0) / cell).astype(int)
        iy0 = np.floor((self.lo[:, 1] - self.gy0) / cell).astype(int); iy1 = np.floor((self.hi[:, 1] - self.gy0) / cell).astype(int)
        for i in range(len(self.lo)):
            for a in range(ix0[i], ix1[i] + 1):
                for b in range(iy0[i], iy1[i] + 1):
                    cells.setdefault(a * self.gny + b, []).append(i)
        self.cells = {k: np.array(v, np.int32) for k, v in cells.items()}

    def _candidates(self, p0, p1):
        """indices of solids whose grid cells are touched by the 2D segment (conservative supercover via stepping)."""
        if not self.cells:
            return np.zeros(0, np.int32)
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        L = max(abs(dx), abs(dy))
        n = int(L / (self.cell * 0.5)) + 2
        t = np.linspace(0, 1, n)
        xs = p0[0] + dx * t; ys = p0[1] + dy * t
        ix = np.floor((xs - self.gx0) / self.cell).astype(int); iy = np.floor((ys - self.gy0) / self.cell).astype(int)
        keys = set()
        for a in (-1, 0, 1):          # +-1 cell margin makes the sample-based supercover conservative
            for b in (-1, 0, 1):
                for k in np.unique((ix + a) * self.gny + (iy + b)):
                    keys.add(int(k))
        parts = [self.cells[k] for k in keys if k in self.cells]
        if not parts:
            return np.zeros(0, np.int32)
        return np.unique(np.concatenate(parts))

    # ---------------------------------------------------------------- ray tests
    def _solid_hits(self, p0, p1, idx=None):
        """returns arrays (t_enter, t_exit, idx) for every solid pierced by the segment (t in [0,1])."""
        p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
        if idx is None:
            idx = self._candidates(p0, p1)
        if len(idx) == 0:
            return np.zeros(0), np.zeros(0), idx
        d = p1 - p0
        lo = self.lo[idx]; hi = self.hi[idx]
        with np.errstate(divide='ignore', invalid='ignore'):
            inv = 1.0 / np.where(d == 0, 1e-300, d)
        ta = (lo - p0) * inv; tb = (hi - p0) * inv
        tmin = np.minimum(ta, tb); tmax = np.maximum(ta, tb)
        # parallel axes: inside slab -> unconstrained, outside -> miss
        par = d == 0
        if par.any():
            for a in np.where(par)[0]:
                inside = (p0[a] >= lo[:, a]) & (p0[a] <= hi[:, a])
                tmin[:, a] = np.where(inside, -np.inf, np.inf)
                tmax[:, a] = np.where(inside, np.inf, -np.inf)
        t0 = np.maximum(tmin.max(1), 0.0); t1 = np.minimum(tmax.min(1), 1.0)
        w = self.is_wedge[idx]
        if w.any():
            # clip by half-space  z - zs(c) <= 0 , zs linear in c on the slope axis
            for j in np.where(w)[0]:
                s = self.slope[idx[j]]; ax = int(s[0]); z0, z1, um, uM = s[1], s[2], s[3], s[4]
                k = (z1 - z0) / (uM - um)
                # f(t) = z(t) - (z0 + k*(c(t)-um)) ; solid where f<=0
                f0 = p0[2] - (z0 + k * (p0[ax] - um)); fd = d[2] - k * d[ax]
                if abs(fd) < 1e-12:
                    if f0 > 0: t1[j] = -1
                else:
                    tc = -f0 / fd
                    if fd > 0: t1[j] = min(t1[j], tc)
                    else: t0[j] = max(t0[j], tc)
        ok = t0 <= t1 + 1e-12
        ok &= t1 >= 0
        return t0[ok], t1[ok], idx[ok]

    def raycast(self, p0, p1, with_normal=False):
        """first static hit on the segment -> (hit, t, point, material)."""
        p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
        best_t = np.inf; mat = None
        t0, t1, idx = self._solid_hits(p0, p1)
        if len(t0):
            j = int(np.argmin(t0)); best_t = float(t0[j]); mat = self.material[idx[j]]
        # ground
        d = p1 - p0
        if self.floor is not None:
            tg = self.floor.raycast(p0, p1)
            if tg is not None and tg < best_t:
                best_t = tg; mat = self.ground_material
        elif d[2] != 0:
            tg = (self.ground_z - p0[2]) / d[2]
            if 0 <= tg <= 1 and tg < best_t:
                best_t = tg; mat = self.ground_material
        if best_t == np.inf:
            return False, 1.0, tuple(p1), None
        return True, best_t, tuple(p0 + d * best_t), mat

    def los(self, p0, p1):
        return not self.raycast(p0, p1)[0]

    def penetration(self, p0, p1, merge=0.03):
        """list of (t_enter, t_exit, material) of every solid region crossed by the segment, in order.
        Touching / overlapping solids (gap <= merge metres along the ray) are merged into one region."""
        t0, t1, idx = self._solid_hits(p0, p1)
        o = np.argsort(t0)
        L = float(np.linalg.norm(np.asarray(p1, float) - np.asarray(p0, float))) or 1.0
        out = []
        for i in o:
            a, b, m = float(t0[i]), float(t1[i]), self.material[idx[i]]
            if out and a * L <= out[-1][1] * L + merge:
                out[-1] = (out[-1][0], max(out[-1][1], b), out[-1][2])
            else:
                out.append((a, b, m))
        return out

    def trace(self, p0, p1, pen_limit=0.18):
        """bullet path: pierce regions of thickness <= pen_limit, stop at the first thicker one (or ground).
        returns (end_point, stop_t or None, [(entry_dist, exit_dist, material, thickness), ...])"""
        p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
        L = float(np.linalg.norm(p1 - p0))
        pens = []
        end_t = None
        tg = None
        if self.floor is None and (p1 - p0)[2] != 0:
            tt = (self.ground_z - p0[2]) / (p1 - p0)[2]
            if 0 <= tt <= 1: tg = tt
        for a, b, m in self.penetration(p0, p1):
            if tg is not None and a >= tg: break
            th = (b - a) * L
            if th <= pen_limit and (tg is None or b < tg):
                pens.append((a * L, b * L, m, th))
            else:
                end_t = a; break
        if end_t is None and tg is not None: end_t = tg
        end = p1 if end_t is None else p0 + (p1 - p0) * end_t
        return end, end_t, pens

    # ---------------------------------------------------------------- map metadata passthrough
    @property
    def sites(self):
        return self.static.get('sites', [])

    def spawns(self, side):
        pads = self.static.get('spawn_pads_by_side', {}).get(side)
        return [tuple(p) for p in pads] if pads else []

    def _in_poly(self, poly, x, y):
        pts = np.asarray(poly, float); inside = False
        j = len(pts) - 1
        for i in range(len(pts)):
            xi, yi = pts[i]; xj, yj = pts[j]
            if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-300) + xi:
                inside = not inside
            j = i
        return inside

    def in_plant_zone(self, x, y, z, site=None):
        """returns the site id ('A'/'B') whose plant zone contains the point, else None."""
        pz = self.static.get('plant_zones')
        if pz:
            for sid, zone in pz.items():
                if site and sid != site: continue
                if zone['min_z'] <= z <= zone['max_z'] and self._in_poly(zone['polygon'], x, y):
                    return sid
            return None
        for s in self.sites:
            if site and s['id'] != site: continue
            if np.hypot(x - s['center'][0], y - s['center'][1]) <= s['radius']:
                return s['id']
        return None

    def in_buy_zone(self, side, x, y, z):
        bz = self.static.get('buy_zones_by_side')
        if bz:
            for zone in bz.get(side, []):
                if zone.get('min_z', -1e9) <= z <= zone.get('max_z', 1e9) and self._in_poly(zone['polygon'], x, y):
                    return True
            return False
        b = self.static.get('buy_zone')
        if b:   # dust2/mirage: union of spawn circles, strict, vertical limit
            for sp in self.spawns(side):
                if np.hypot(x - sp[0], y - sp[1]) < b['radius'] and abs(z - sp[2]) < b['vertical_limit']:
                    return True
        return False
