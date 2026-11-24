"""Approximate geometry for dust2 / mirage from the public kit mapdata (no triangle mesh is public).

wall_cover_contours: 1 m x 1 m XY cells, each with one vertical envelope [z0,z1]  (overcovers by up to sqrt2 m in XY,
the vertical envelope may merge disconnected surfaces).  walkable_surface_candidates: sparse 1 m cells with floor z.
We turn every wall cell into an AABB (runs of equal-envelope cells along x are merged) and keep the walkable
candidate cells (+ optional extra observed floor samples) as a multi-level floor field.
"""
import json, os
import numpy as np

_KIT_CANDIDATES = [
    os.environ.get('FPS_KIT', ''),
    './fps_data/fps_platform/kit/data',
    './fps_data/fps_platform/kit/data',
]


def _kit_file(map_id, kit_dir):
    name = 'fps-' + map_id.split('-')[0] + '.mapdata.json'
    for d in ([kit_dir] if kit_dir else []) + _KIT_CANDIDATES:
        if d and os.path.exists(os.path.join(d, name)):
            return os.path.join(d, name)
    raise FileNotFoundError(name)


class FloorField:
    """multi-level floor heights on a 1 m grid; used only for rays that hit the ground."""
    def __init__(self, cells, cell=1.0, ox=0.0, oy=0.0):
        self.cell = cell
        self.levels = {}
        for i, j, z in cells:
            self.levels.setdefault((int(i), int(j)), []).append(float(z))
        self.ox, self.oy = ox, oy

    def level_at(self, x, y, zref, tol=2.0):
        k = (int(np.floor(x)), int(np.floor(y)))
        lv = self.levels.get(k)
        if not lv: return None
        best = min(lv, key=lambda z: abs(z - zref))
        return best if abs(best - zref) <= tol else None

    def raycast(self, p0, p1, step=0.5):
        d = p1 - p0
        L = np.linalg.norm(d)
        n = max(int(L / step), 1)
        prev = None
        for k in range(n + 1):
            t = k / n
            p = p0 + d * t
            lv = self.level_at(p[0], p[1], p[2])
            if lv is not None and p[2] <= lv:
                return t if k > 0 else 0.0
        return None


def build_voxel_geom(cls, m, kit_dir=None):
    mp = json.load(open(_kit_file(m['id'], kit_dir)))
    cells = {}
    for w in mp['wall_cover_contours']:
        p = np.array(w['polygon_xy'], float)
        x0, y0 = p.min(0); x1, y1 = p.max(0)
        z0, z1 = w['height_range_m']
        cells[(int(round(x0)), int(round(y0)))] = (z0, z1, x1 - x0, y1 - y0)
    # merge runs along x
    lo, hi = [], []
    done = set()
    for (x, y) in sorted(cells):
        if (x, y) in done: continue
        z0, z1, w_, h_ = cells[(x, y)]
        n = 1
        while (x + n, y) in cells and cells[(x + n, y)][:2] == (z0, z1) and (x + n, y) not in done:
            done.add((x + n, y)); n += 1
        done.add((x, y))
        lo.append([x, y, z0]); hi.append([x + n * w_, y + h_, z1])
    fl = FloorField([(c[0], c[1], c[2]) for c in mp['walkable_surface_candidates']['cells']])
    g = cls(m['id'], lo, hi, None, ['concrete'] * len(lo), ground_z=float(m['bounds']['min'][2]) if m.get('bounds') else 0.0,
            bounds=m.get('bounds'), static=m, floor=fl, cell=4.0)
    g.approximate = True
    return g
