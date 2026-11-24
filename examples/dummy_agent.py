"""Minimal placeholder agent & runnable CPU Arena smoke test (`rsi-fps-5.5`).

Contains ZERO proprietary strategy logic — purely demonstrates the legal
`act(obs)` contract and verifies the offline simulator + calibrated CS2 meshes
run end-to-end out of the box.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import sys
import tarfile

REPO_ROOT = Path(__file__).resolve().parents[1]


def ensure_extracted_data_dir() -> Path:
    """Extract bundled public static maps & calibrated CS2 TriMeshes into `.cache/fps_data`."""
    data_dir = REPO_ROOT / ".cache" / "fps_data"
    marker = data_dir / "static" / "static_borgo-r3-v54.json"
    if not marker.is_file():
        data_dir.mkdir(parents=True, exist_ok=True)
        mesh_tar = REPO_ROOT / "sim" / "meshes" / "exact_mesh_cache_v2_calibrated.tar.gz"
        with tarfile.open(mesh_tar, "r:gz") as tf:
            tf.extractall(data_dir)
    return data_dir


_DATA_DIR = ensure_extracted_data_dir()
os.environ.setdefault("FPS_DATA", str(_DATA_DIR))
os.environ.setdefault("FPS_STATIC_DIR", str(_DATA_DIR / "static"))
os.environ.setdefault("FPS_ASSET_DIR", str(_DATA_DIR / "assets"))
os.environ.setdefault("FPS_KIT_DIR", str(REPO_ROOT / "sim" / "meshes" / "mapdata"))
os.environ.setdefault("FPS_FLOOR_DIR", str(_DATA_DIR / "floor"))
sys.path.insert(0, str(REPO_ROOT / "sim/cpu_arena"))

from fpssim.arena import Arena  # noqa: E402


class DummyAgent:
    """Minimal legal agent returning valid neutral/slow-walk actions."""

    def __init__(self, seat_id: int = 0) -> None:
        self.seat_id = int(seat_id)

    def act(self, obs: dict) -> dict:
        me = obs.get("me") or {}
        if not me.get("alive") or obs.get("phase") != "live":
            return {}
        tick = int(obs.get("match_tick", 0))
        return {
            "move_forward": 0.25 if (tick % 60) < 30 else 0.0,
            "move_right": 0.0,
            "yaw_rate": 0.1 * math.sin(tick * 0.05),
            "pitch_rate": 0.0,
            "walk": True,
            "fire": False,
        }


def run_smoke(map_id: str = "borgo-r3-v54", steps: int = 40) -> dict:
    arena = Arena()
    obs_by_seat = arena.reset(map_id=map_id, seed=42)
    agents = {s: DummyAgent(s) for s in range(10)}
    for _ in range(steps):
        actions = [agents[s].act(obs_by_seat[s]) for s in range(10)]
        step_out = arena.step(actions)
        obs_by_seat = step_out[0] if isinstance(step_out, tuple) else step_out
        if arena.finished:
            break
    alive = sum(1 for s in range(10) if obs_by_seat[s]["me"]["alive"])
    return {
        "map_id": map_id,
        "steps_run": steps,
        "final_tick": int(obs_by_seat[0]["match_tick"]),
        "alive_seats": alive,
        "faults": sum(arena.faults),
    }


if __name__ == "__main__":
    for m in ("borgo-r3-v54", "dust2-cs2-v54"):
        print(run_smoke(map_id=m, steps=40))
