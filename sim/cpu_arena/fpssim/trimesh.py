"""Triangle-mesh world for the CS2 maps (Dust2 / Mirage): the exact collision/visibility mesh the platform uses.

positions.f32 = float32 xyz per vertex, 3 vertices per triangle, in cs2-web map coordinates; the platform static map gives
`basis` (3x3) and `offset` with world = basis @ p + offset (validated against recorded bounds and bullet impacts).
Segment queries use a uniform XY grid of triangle bins + vectorised Moller-Trumbore.
"""
import numpy as np


class TriMesh:
    def __init__(self, positions_path, basis, offset, materials_path=None, cell=2.0):
        v = np.fromfile(positions_path, dtype='<f4').astype(np.float64).reshape(-1, 3, 3)
        B = np.asarray(basis, np.float64); o = np.asarray(offset, np.float64)
        self.tri = np.einsum('ij,tkj->tki', B, v) + o            # [T,3,3] world (z up)
        self.mat = np.fromfile(materials_path, dtype=np.uint8) if materials_path else None
        self.v0 = self.tri[:, 0]; self.e1 = self.tri[:, 1] - self.v0; self.e2 = self.tri[:, 2] - self.v0
        lo = self.tri.min(1); hi = self.tri.max(1)
        self.lo, self.hi = lo.min(0), hi.max(0)
        self.cell = cell
        self.ox, self.oy = self.lo[0], self.lo[1]
        self.nx = int(np.ceil((self.hi[0] - self.ox) / cell)) + 1
        self.ny = int(np.ceil((self.hi[1] - self.oy) / cell)) + 1
        bins = [[] for _ in range(self.nx * self.ny)]
        ix0 = ((lo[:, 0] - self.ox) // cell).astype(int); ix1 = ((hi[:, 0] - self.ox) // cell).astype(int)
        iy0 = ((lo[:, 1] - self.oy) // cell).astype(int); iy1 = ((hi[:, 1] - self.oy) // cell).astype(int)
        for t in range(len(self.tri)):
            for ix in range(ix0[t], ix1[t] + 1):
                for iy in range(iy0[t], iy1[t] + 1):
                    bins[ix * self.ny + iy].append(t)
        self._start = np.zeros(len(bins) + 1, np.int64)
        self._start[1:] = np.cumsum([len(b) for b in bins])
        self._flat = np.array([t for b in bins for t in b], np.int64)
        g = np.array([-1, 0, 1]); self._dx = np.repeat(g, 3); self._dy = np.tile(g, 3)

    def _cells(self, p0, p1):
        """grid cells within one cell of the XY projection of the segment (dense sampling + 3x3 dilation)."""
        d = np.hypot(p1[0] - p0[0], p1[1] - p0[1])
        n = max(2, int(d / (self.cell * 0.5)) + 2)
        s = np.linspace(0, 1, n)
        ix = ((p0[0] + (p1[0] - p0[0]) * s - self.ox) // self.cell).astype(np.int64)
        iy = ((p0[1] + (p1[1] - p0[1]) * s - self.oy) // self.cell).astype(np.int64)
        ix = (ix[:, None] + self._dx[None, :]).ravel(); iy = (iy[:, None] + self._dy[None, :]).ravel()
        ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        return np.unique(ix[ok] * self.ny + iy[ok])

    def candidates(self, p0, p1):
        ks = self._cells(p0, p1)
        st, en = self._start[ks], self._start[ks + 1]
        if not len(ks) or (en - st).sum() == 0:
            return np.zeros(0, np.int64)
        idx = np.concatenate([self._flat[a:b] for a, b in zip(st, en) if b > a])
        return np.unique(idx)

    def intersect(self, p0, p1, idx=None, both=True):
        """all hits of segment p0->p1: (t sorted ascending, triangle ids)."""
        p0 = np.asarray(p0, np.float64); p1 = np.asarray(p1, np.float64)
        if idx is None:
            idx = self.candidates(p0, p1)
        if len(idx) == 0:
            return np.zeros(0), idx
        d = p1 - p0
        e1, e2, v0 = self.e1[idx], self.e2[idx], self.v0[idx]
        h = np.cross(d, e2)
        a = np.einsum('ij,ij->i', e1, h)
        ok = np.abs(a) > 1e-12
        f = np.where(ok, 1.0 / np.where(ok, a, 1.0), 0.0)
        s = p0 - v0
        u = f * np.einsum('ij,ij->i', s, h)
        q = np.cross(s, e1)
        v = f * (q @ d)
        t = f * np.einsum('ij,ij->i', e2, q)
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t >= 0) & (t <= 1)
        if not both:
            hit &= a > 0
        t, ids = t[hit], idx[hit]
        o = np.argsort(t)
        return t[o], ids[o]

    def raycast(self, p0, p1):
        t, ids = self.intersect(p0, p1)
        if len(t) == 0:
            return False, 1.0, None, None
        p0 = np.asarray(p0, np.float64); p1 = np.asarray(p1, np.float64)
        return True, float(t[0]), tuple(p0 + (p1 - p0) * t[0]), int(ids[0])

    def los(self, p0, p1):
        return len(self.intersect(p0, p1)[0]) == 0
