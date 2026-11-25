"""Exact 3-track geometry loader & Frenet projection for AGX Formula (rsi-racing-4.4).

Loads official public circuit JSONs:
  - parkland  (L ≈ 5795.84m, C3/C4/C5, override det=1000.0 / act=1172.0)
  - highland  (L ≈ 6999.48m, C2/C3/C4, override det=6606.51 / act=0.0)
  - crossover (L ≈ 5803.26m, C1/C2/C3, override det=5306.32 / act=0.0)
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

CIRCUITS_DIR = Path(__file__).resolve().parent / "circuits"

# Official sporting rule overrides (work/racing-bootstrap/docs/racing-sporting.zh.md §手动超越模式)
OFFICIAL_AID_ZONES: Dict[str, Tuple[float, float]] = {
    "parkland": (1000.0, 1172.0),
    "highland": (6606.51, 0.0),
    "crossover": (5306.32, 0.0),
}

TRACK_COMPOUND_MAP: Dict[str, Dict[str, str]] = {
    "crossover": {"hard": "C1", "medium": "C2", "soft": "C3"},
    "highland": {"hard": "C2", "medium": "C3", "soft": "C4"},
    "parkland": {"hard": "C3", "medium": "C4", "soft": "C5"},
}

ALL_TRACKS: Tuple[str, ...] = ("parkland", "highland", "crossover")


def _wrap_angle(a: np.ndarray | float) -> np.ndarray | float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _interval_contains_cyclic(s: np.ndarray, start_s: float, end_s: float, length: float) -> np.ndarray:
    s_mod = np.mod(s, length)
    start_m = start_s % length
    end_m = end_s % length
    if start_m <= end_m:
        return (s_mod >= start_m) & (s_mod <= end_m)
    return (s_mod >= start_m) | (s_mod <= end_m)


class CircuitGeometry:
    """Uniformly sampled closed-loop circuit with exact road, wall, kerb, aero, pit, and grid data."""

    def __init__(self, track_id: str, raw: Dict[str, Any], resample_ds: Optional[float] = None):
        self.track_id = track_id
        self.raw = raw
        cl = raw["centerline"]
        s_raw = np.array([p["s"] for p in cl], dtype=np.float64)
        x_raw = np.array([p["x"] for p in cl], dtype=np.float64)
        y_raw = np.array([p["y"] for p in cl], dtype=np.float64)
        z_raw = np.array([p["z"] for p in cl], dtype=np.float64)
        bank_raw = np.array([p.get("bank_deg", 0.0) for p in cl], dtype=np.float64)
        left_raw = np.array(raw["width"]["left"], dtype=np.float64)
        right_raw = np.array(raw["width"]["right"], dtype=np.float64)

        # Compute exact closed-loop arc length from geometry (matches track_length_m in replays!)
        dx_seg = np.diff(np.append(x_raw, x_raw[0]))
        dy_seg = np.diff(np.append(y_raw, y_raw[0]))
        dz_seg = np.diff(np.append(z_raw, z_raw[0]))
        seg_2d = np.hypot(dx_seg, dy_seg)
        # Note: last point in JSON is near s=L-2m; closing segment closes the loop cleanly.
        cum_s = np.concatenate(([0.0], np.cumsum(seg_2d)))
        self.length = float(cum_s[-1])
        # Official sector end_s is often the exact kernel track length; cross-check
        sectors = raw.get("sectors", [])
        if sectors:
            sec_end = float(sectors[-1]["end_s"])
            if abs(sec_end - self.length) < 15.0:
                self.length = sec_end

        if resample_ds is not None:
            n_nodes = max(500, int(round(self.length / resample_ds)))
            self.ds = self.length / n_nodes
            s_grid = np.arange(n_nodes, dtype=np.float64) * self.ds
            # Periodic interpolation on closed loop
            s_src = cum_s[:-1] * (self.length / float(cum_s[-1]))
            s_ext = np.append(s_src, self.length)
            self.x = np.interp(s_grid, s_ext, np.append(x_raw, x_raw[0]))
            self.y = np.interp(s_grid, s_ext, np.append(y_raw, y_raw[0]))
            self.z = np.interp(s_grid, s_ext, np.append(z_raw, z_raw[0]))
            self.bank_deg = np.interp(s_grid, s_ext, np.append(bank_raw, bank_raw[0]))
            self.left = np.interp(s_grid, s_ext, np.append(left_raw, left_raw[0]))
            self.right = np.interp(s_grid, s_ext, np.append(right_raw, right_raw[0]))
            self.s = s_grid
            self.n_nodes = n_nodes
        else:
            self.n_nodes = len(cl)
            self.ds = self.length / self.n_nodes
            self.s = np.arange(self.n_nodes, dtype=np.float64) * self.ds
            self.x, self.y, self.z = x_raw, y_raw, z_raw
            self.bank_deg = bank_raw
            self.left, self.right = left_raw, right_raw

        # Smooth tangent, normal, grade, and 10m-chord curvature
        tx = np.roll(self.x, -1) - np.roll(self.x, 1)
        ty = np.roll(self.y, -1) - np.roll(self.y, 1)
        tn = np.maximum(np.hypot(tx, ty), 1e-9)
        self.tx = tx / tn
        self.ty = ty / tn
        # Left unit normal (-ty, tx) because positive n is left of centerline
        self.nx = -self.ty
        self.ny = self.tx
        self.heading = np.arctan2(self.ty, self.tx)
        self.grade = (np.roll(self.z, -1) - np.roll(self.z, 1)) / (2.0 * self.ds)

        # 10m chord curvature (matches Driver.curvature anti-quantization chord)
        chord = max(2, int(round(10.0 / self.ds)))
        ax, ay = self.x - np.roll(self.x, chord), self.y - np.roll(self.y, chord)
        bx, by = np.roll(self.x, -chord) - self.x, np.roll(self.y, -chord) - self.y
        self.curvature = (
            2.0 * (ax * by - ay * bx)
            / np.maximum(np.hypot(ax, ay) * np.hypot(bx, by) * np.hypot(ax + bx, ay + by), 1e-8)
        )

        # Wall lateral clearances (default generous runoff where no collidable wall is defined)
        self.wall_left = self.left + 12.0
        self.wall_right = self.right + 12.0
        for w in raw.get("walls", []):
            if not w.get("collidable", True):
                continue
            pts = w.get("points", [])
            if len(pts) < 2:
                continue
            ws = np.array([float(p["s"]) for p in pts], dtype=np.float64)
            wn = np.array([float(p["n"]) for p in pts], dtype=np.float64)
            side = w.get("side")
            for k in range(len(pts) - 1):
                s0, s1 = ws[k], ws[k + 1]
                n0, n1 = wn[k], wn[k + 1]
                # Handle short segments robustly
                ds_seg = (s1 - s0 + self.length / 2.0) % self.length - self.length / 2.0
                if abs(ds_seg) > 600.0 or abs(ds_seg) < 1e-3:
                    continue
                n_steps = max(2, int(math.ceil(abs(ds_seg) / self.ds)) + 1)
                ts = np.linspace(0.0, 1.0, n_steps)
                ss = np.mod(s0 + ds_seg * ts, self.length)
                nn = n0 + (n1 - n0) * ts
                idxs = np.mod(np.round(ss / self.ds).astype(int), self.n_nodes)
                for idx, nv in zip(idxs, nn):
                    if side == "left" or (side is None and nv > 0):
                        if nv > 0:
                            self.wall_left[idx] = min(self.wall_left[idx], float(nv))
                    elif side == "right" or (side is None and nv < 0):
                        if nv < 0:
                            self.wall_right[idx] = min(self.wall_right[idx], float(-nv))

        # Ensure walls are never narrower than road width - 0.5m due to OSM artifacts
        self.wall_left = np.maximum(self.wall_left, self.left + 0.5)
        self.wall_right = np.maximum(self.wall_right, self.right + 0.5)

        # Active aero zones
        self.aero_zones: List[Tuple[float, float]] = []
        self.xmode = np.zeros(self.n_nodes, dtype=np.float64)
        self.aero_remaining = np.zeros(self.n_nodes, dtype=np.float64)
        for z in raw.get("active_aero_zones", []):
            st, en = float(z["start_s"]), float(z["end_s"])
            self.aero_zones.append((st, en))
            mask = _interval_contains_cyclic(self.s, st, en, self.length)
            self.xmode[mask] = 1.0
            rem = np.mod(en - self.s, self.length)
            self.aero_remaining[mask] = np.maximum(self.aero_remaining[mask], rem[mask])

        # Pit lane metadata
        pit = raw["pit"]
        self.pit_entry_s = float(pit["entry_s"])
        self.pit_exit_s = float(pit["exit_s"])
        self.pit_speed_mps = float(pit.get("speed_limit_kph", 80.0)) / 3.6
        self.pit_lane_length_m = float(pit.get("lane_length_m", (self.pit_exit_s - self.pit_entry_s) % self.length))

        # Manual Overtake Mode detection / activation
        det_s, act_s = OFFICIAL_AID_ZONES[track_id]
        self.aid_detection_s = det_s
        self.aid_activation_s = act_s

        # Start grid slots
        self.start_grid = raw.get("start_grid", [])
        self.compound_map = dict(TRACK_COMPOUND_MAP[track_id])
        self.sectors = [(float(sec["start_s"]), float(sec["end_s"])) for sec in raw.get("sectors", [])]

    def interp_scalar(self, arr: np.ndarray, s: float) -> float:
        u = (s % self.length) / self.ds
        i = int(u) % self.n_nodes
        f = u - int(u)
        return float(arr[i] + (arr[(i + 1) % self.n_nodes] - arr[i]) * f)

    def interp_vec(self, arr: np.ndarray, s: np.ndarray) -> np.ndarray:
        u = np.mod(s, self.length) / self.ds
        i = np.floor(u).astype(np.int64) % self.n_nodes
        f = u - np.floor(u)
        return arr[i] + (arr[(i + 1) % self.n_nodes] - arr[i]) * f

    def frenet_to_world(self, s: float, n: float) -> Tuple[float, float, float, float]:
        x0 = self.interp_scalar(self.x, s)
        y0 = self.interp_scalar(self.y, s)
        z0 = self.interp_scalar(self.z, s)
        nx = self.interp_scalar(self.nx, s)
        ny = self.interp_scalar(self.ny, s)
        tx = self.interp_scalar(self.tx, s)
        ty = self.interp_scalar(self.ty, s)
        yaw = math.atan2(ty, tx)
        return x0 + n * nx, y0 + n * ny, z0, yaw

    def project_xy_batch(
        self,
        x: np.ndarray,
        y: np.ndarray,
        s_hint: Optional[np.ndarray] = None,
        yaw: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fast vectorized projection from world (x, y) to Frenet (s, n, heading_error)."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if s_hint is not None:
            s_hint = np.mod(np.asarray(s_hint, dtype=np.float64), self.length)
            idx0 = np.floor(s_hint / self.ds).astype(np.int64) % self.n_nodes
            # Refine within +/- 8 nodes around s_hint
            offsets = np.arange(-8, 9, dtype=np.int64)
            cand = (idx0[:, None] + offsets[None, :]) % self.n_nodes
            dx = x[:, None] - self.x[cand]
            dy = y[:, None] - self.y[cand]
            best = np.argmin(dx * dx + dy * dy, axis=1)
            idx = cand[np.arange(len(x)), best]
        else:
            # Chunked exact nearest node
            idx = np.empty(len(x), dtype=np.int64)
            chunk = 2048
            for st in range(0, len(x), chunk):
                en = min(len(x), st + chunk)
                dx = x[st:en, None] - self.x[None, :]
                dy = y[st:en, None] - self.y[None, :]
                idx[st:en] = np.argmin(dx * dx + dy * dy, axis=1)

        dx0 = x - self.x[idx]
        dy0 = y - self.y[idx]
        along = dx0 * self.tx[idx] + dy0 * self.ty[idx]
        s_out = np.mod(self.s[idx] + along, self.length)
        n_out = dx0 * self.nx[idx] + dy0 * self.ny[idx]
        if yaw is not None:
            hdg = np.arctan2(self.interp_vec(self.ty, s_out), self.interp_vec(self.tx, s_out))
            herr = _wrap_angle(np.asarray(yaw, dtype=np.float64) - hdg)
        else:
            herr = np.zeros_like(s_out)
        return s_out, n_out, herr

    def preview_rows(self, s: float) -> List[List[float]]:
        """Generate exact 24-row 15m-spaced preview table matching official observation contract."""
        rows: List[List[float]] = []
        for k in range(1, 25):
            d = 15.0 * k
            sk = s + d
            curv = self.interp_scalar(self.curvature, sk)
            grd = self.interp_scalar(self.grade, sk)
            wl = self.interp_scalar(self.left, sk)
            wr = self.interp_scalar(self.right, sk)
            az = 1 if self.interp_scalar(self.xmode, sk) > 0.5 else 0
            rows.append([d, round(curv, 6), round(grd, 5), round(wl, 3), round(wr, 3), az])
        return rows


_CIRCUIT_CACHE: Dict[Tuple[str, Optional[float]], CircuitGeometry] = {}


def load_circuit(track_id: str, resample_ds: Optional[float] = 4.0, circuits_dir: Path = CIRCUITS_DIR) -> CircuitGeometry:
    key = (track_id, resample_ds)
    if key not in _CIRCUIT_CACHE:
        raw = json.loads((circuits_dir / f"{track_id}.json").read_text(encoding="utf-8"))
        _CIRCUIT_CACHE[key] = CircuitGeometry(track_id, raw, resample_ds=resample_ds)
    return _CIRCUIT_CACHE[key]


def identify_track_from_obs(obs: Dict[str, Any]) -> str:
    """Identify parkland / highland / crossover deterministically on tick 0 from legal obs."""
    alloc = (obs.get("tyres") or {}).get("allocation") or {}
    soft_c = alloc.get("soft")
    if soft_c == "C5":
        return "parkland"
    if soft_c == "C4":
        return "highland"
    if soft_c == "C3":
        return "crossover"
    length = float(obs.get("track_length_m", 5795.843))
    if length > 6400.0:
        return "highland"
    if abs(length - 5803.2) < abs(length - 5795.8):
        return "crossover"
    return "parkland"


if __name__ == "__main__":
    for tid in ALL_TRACKS:
        g = load_circuit(tid, resample_ds=4.0)
        print(
            f"{tid:10s}: L={g.length:.3f}m nodes={g.n_nodes} ds={g.ds:.4f} "
            f"max|k|={np.max(np.abs(g.curvature)):.5f} pit=({g.pit_entry_s:.1f}->{g.pit_exit_s:.1f}) "
            f"xmode_frac={np.mean(g.xmode):.3f} aid=({g.aid_detection_s},{g.aid_activation_s})"
        )
