"""Static-world collision for player movement (AABB hull vs axis-aligned boxes / stair steps / floor heightfield).

Hull: square, half-width HW=0.3 m versus world geometry (fitted: min Chebyshev clearance to every
whitebox box is 0.3000 in the recordings); standing height 1.8, crouch 1.1.
Player-vs-player separation is a separate circle rule (0.7 m centre distance), see movement.py.

World interface (used by movement.Mover):
    ground(x, y, zf, hw, step) -> top z of highest support under the hull footprint with top <= zf+step, or None
    blockers(x0,x1,y0,y1,zf,zh) -> (K,6) boxes overlapping the horizontal footprint and the z-range (zf,zh)
    ceiling(x0,x1,y0,y1,zh) -> lowest box bottom above zh over the footprint (or inf)
    ladders: list of dict(id, rail(x,y), normal(nx,ny,nz), width, bottom, top, bottom_exit, top_exit)
"""
import numpy as np

HW = 0.3
STAND_H = 1.8
CROUCH_H = 1.1
EPS = 1e-6
SKIN = 0.00011          # recorded floor contact heights are box-top + 0.00011 (ground plane at -0.002+0.00011)
GROUND_Z = -0.002 + SKIN


class BoxWorld:
    """Analytic whitebox world: boxes (centre x,y; base z; w along x, d along y; h) + stepped ramps."""

    def __init__(self, map_dict):
        m = map_dict
        self.map_id = m.get('id')
        self.ground_z = float((m.get('ground') or {}).get('z', 0.0)) + (GROUND_Z if m.get('boxes') else 0.0)
        rows = []
        for b in m.get('boxes', []):
            rows.append((b['x'] - b['w'] / 2, b['x'] + b['w'] / 2, b['y'] - b['d'] / 2, b['y'] + b['d'] / 2, b['z'], b['z'] + b['h'] + SKIN))
        stairs = {s['ramp_id']: s for s in m.get('stairs', [])}
        self.stair_boxes = []
        for r in m.get('ramps', []):
            st = stairs.get(r['id'])
            n = st['steps'] if st else 1
            ax = r['axis']
            lo, ln = (r['x'] - r['w'] / 2, r['w']) if ax == 'x' else (r['y'] - r['d'] / 2, r['d'])
            zlo = min(r['z0'], r['z1'])
            for i in range(n):
                top = r['z0'] + (i + 0.5) * (r['z1'] - r['z0']) / n + SKIN
                a0, a1 = lo + ln * i / n, lo + ln * (i + 1) / n
                if ax == 'x':
                    row = (a0, a1, r['y'] - r['d'] / 2, r['y'] + r['d'] / 2, zlo - 1.0, top)
                else:
                    row = (r['x'] - r['w'] / 2, r['x'] + r['w'] / 2, a0, a1, zlo - 1.0, top)
                rows.append(row)
                self.stair_boxes.append(row)
        self.B = np.array(rows, dtype=np.float64).reshape(-1, 6)
        self.ladders = []
        for l in m.get('ladders', []):
            self.ladders.append(dict(l))
        self.vertical = m.get('vertical_rules') or {}
        self.bounds = m.get('bounds')

    # -- queries ---------------------------------------------------------
    def _overlap_xy(self, x0, x1, y0, y1):
        B = self.B
        return (B[:, 0] < x1 - EPS) & (B[:, 1] > x0 + EPS) & (B[:, 2] < y1 - EPS) & (B[:, 3] > y0 + EPS)

    def ground(self, x, y, zf, hw=HW, step=0.45):
        B = self.B
        m = self._overlap_xy(x - hw, x + hw, y - hw, y + hw) & (B[:, 5] <= zf + step + 1e-4)
        best = self.ground_z
        if m.any():
            t = B[m, 5].max()
            if t > best: best = t
        return best

    def blockers(self, x0, x1, y0, y1, zlo, zhi):
        B = self.B
        m = self._overlap_xy(x0, x1, y0, y1) & (B[:, 4] < zhi - EPS) & (B[:, 5] > zlo + EPS)
        return B[m]

    def ceiling(self, x0, x1, y0, y1, zh):
        B = self.B
        m = self._overlap_xy(x0, x1, y0, y1) & (B[:, 4] >= zh - 1e-4)
        return B[m, 4].min() if m.any() else np.inf

    def find_ladder(self, x, y, z):
        return None


def sweep_axis(world, pos, axis, delta, zf, zh, step, hw=HW):
    """Move the hull footprint along `axis` (0=x,1=y) by delta, stopping at faces of blocking boxes.
    A box blocks if it overlaps the hull z-range and its top is above feet+step. Returns (newcoord, hit)."""
    x, y = pos
    if delta == 0.0: return (x if axis == 0 else y), False
    x0, x1, y0, y1 = x - hw, x + hw, y - hw, y + hw
    if axis == 0:
        nx0, nx1 = (x0, x1 + delta) if delta > 0 else (x0 + delta, x1)
        cand = world.blockers(nx0, nx1, y0, y1, zf, zh)
    else:
        ny0, ny1 = (y0, y1 + delta) if delta > 0 else (y0 + delta, y1)
        cand = world.blockers(x0, x1, ny0, ny1, zf, zh)
    cur = x if axis == 0 else y
    new = cur + delta
    hit = False
    for b in cand:
        if b[5] <= zf + step + 1e-4: continue          # steppable
        if axis == 0:
            if delta > 0 and b[0] >= x1 - EPS: lim = b[0] - hw
            elif delta < 0 and b[1] <= x0 + EPS: lim = b[1] + hw
            else: continue
        else:
            if delta > 0 and b[2] >= y1 - EPS: lim = b[2] - hw
            elif delta < 0 and b[3] <= y0 + EPS: lim = b[3] + hw
            else: continue
        if delta > 0 and lim < new: new = lim; hit = True
        if delta < 0 and lim > new: new = lim; hit = True
    return new, hit


class GridWorld:
    """Approximate world for the triangle-mesh maps (Dust2/Mirage) whose authoritative collision mesh is not public.

    walls : kit mapdata `wall_cover_contours` = 1 m cells with a height envelope (over-covers by <= sqrt2 m in XY, can merge
            disconnected surfaces vertically).  Treated as AABBs.
    floor : union of (a) kit `walkable_surface_candidates` cells (1 m), (b) static nav node positions, (c) optional recorded
            grounded player positions (0.5 m grid).  Support query = highest floor sample under the hull footprint with
            z <= feet + step.  Where no sample exists the world is 'void' (player falls to void_z).
    """
    RES = 0.5

    def __init__(self, map_dict, mapdata=None, traces=None):
        m = map_dict
        self.map_id = m.get('id')
        self.ladders = [dict(l) for l in m.get('ladders', [])]
        self.vertical = m.get('vertical_rules') or {}
        self.bounds = m.get('bounds')
        self.void_z = (self.bounds['min'][2] - 5.0) if self.bounds else -50.0
        self.floor = {}
        walls = {}
        if mapdata:
            wc = mapdata.get('wall_cover_contours', [])
            for c in wc:
                p = c['polygon_xy']
                xs = [q[0] for q in p]; ys = [q[1] for q in p]
                bb = (min(xs), max(xs), min(ys), max(ys), c['height_range_m'][0], c['height_range_m'][1])
                for ix in range(int(np.floor(bb[0] + 1e-6)), int(np.ceil(bb[1] - 1e-6))):
                    for iy in range(int(np.floor(bb[2] + 1e-6)), int(np.ceil(bb[3] - 1e-6))):
                        walls.setdefault((ix, iy), []).append(bb)
            w = mapdata.get('walkable_surface_candidates', {})
            cs = w.get('cell_size_m', 1); ox, oy = w.get('origin_xy', [0, 0])
            for ix, iy, z in w.get('cells', []):
                self._add_floor(ox + (ix + 0.5) * cs, oy + (iy + 0.5) * cs, z, spread=cs)
        for n in (m.get('nav', {}) or {}).get('nodes', []):
            pos = n['position']; self._add_floor(pos[0], pos[1], pos[2])
        if traces is not None:
            for x, y, z in traces: self._add_floor(x, y, z)
        self.walls = walls
        self._empty = np.zeros((0, 6))

    def _add_floor(self, x, y, z, spread=0.0):
        r = self.RES
        x0 = int(np.floor((x - spread / 2) / r)); x1 = int(np.floor((x + spread / 2) / r))
        y0 = int(np.floor((y - spread / 2) / r)); y1 = int(np.floor((y + spread / 2) / r))
        for i in range(x0, x1 + 1):
            for j in range(y0, y1 + 1):
                lst = self.floor.setdefault((i, j), [])
                z = round(float(z), 2)
                if z not in lst: lst.append(z)

    def ground(self, x, y, zf, hw=HW, step=0.45):
        r = self.RES
        best = None
        lim = zf + step + 1e-4
        for i in range(int(np.floor((x - hw) / r)), int(np.floor((x + hw) / r)) + 1):
            for j in range(int(np.floor((y - hw) / r)), int(np.floor((y + hw) / r)) + 1):
                for z in self.floor.get((i, j), ()):
                    if z <= lim and (best is None or z > best): best = z
        return self.void_z if best is None else best

    def blockers(self, x0, x1, y0, y1, zlo, zhi):
        out = []
        for i in range(int(np.floor(x0)) - 1, int(np.floor(x1)) + 1):
            for j in range(int(np.floor(y0)) - 1, int(np.floor(y1)) + 1):
                for b in self.walls.get((i, j), ()):
                    if b[0] < x1 - EPS and b[1] > x0 + EPS and b[2] < y1 - EPS and b[3] > y0 + EPS and b[4] < zhi - EPS and b[5] > zlo + EPS:
                        out.append(b)
        if not out: return self._empty
        return np.array(list(set(out)))

    def ceiling(self, x0, x1, y0, y1, zh):
        best = np.inf
        for i in range(int(np.floor(x0)) - 1, int(np.floor(x1)) + 1):
            for j in range(int(np.floor(y0)) - 1, int(np.floor(y1)) + 1):
                for b in self.walls.get((i, j), ()):
                    if b[0] < x1 - EPS and b[1] > x0 + EPS and b[2] < y1 - EPS and b[3] > y0 + EPS and b[4] >= zh - 1e-4:
                        best = min(best, b[4])
        return best


def make_world(map_dict, mapdata_dir=None, traces=None, floor_cache_dir=None):
    """BoxWorld for analytic whitebox maps, GridWorld for dust2/mirage.
    floor_cache_dir: directory with <map_id>.npz (pts[N,3]) from tools/build_floor.py, used when traces is None."""
    import json, os
    if map_dict.get('boxes'):
        return BoxWorld(map_dict)
    mapdata = None
    if mapdata_dir:
        short = map_dict['id'].split('-')[0]
        p = os.path.join(mapdata_dir, f'fps-{short}.mapdata.json')
        if os.path.exists(p): mapdata = json.load(open(p))
    if traces is None and floor_cache_dir:
        p = os.path.join(floor_cache_dir, map_dict['id'] + '.npz')
        if os.path.exists(p): traces = np.load(p)['pts']
    return GridWorld(map_dict, mapdata, traces)
