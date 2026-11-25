"""Minimal placeholder driver & runnable FormulaSim-v5 (rsi-racing-4.4) demo.

Demonstrates loading the 3 official circuits (`parkland`, `highland`, `crossover`),
validating a stock `formula-v4-design-1` envelope, and stepping the 120 Hz 4-wheel
CPU simulator (`sim/racing/formula_sim_v5.py`) with a simple centerline follower.

No proprietary strategy code, trained weights, or competition credentials are used.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
RACING_DIR = REPO_ROOT / "sim" / "racing"
if str(RACING_DIR) not in sys.path:
    sys.path.insert(0, str(RACING_DIR))

from design_validator import validate_design_envelope  # noqa: E402
from formula_sim_v5 import (  # noqa: E402
    SimV5ParamsCPU,
    run_weekend,
    standard_weather_scenarios,
)

STOCK_DESIGN: Dict[str, Any] = {
    "schema": "formula-v4-design-1",
    "aero": {"front_wing": 20.0, "rear_wing": 22.0},
    "ride_height_mm": {"front": 40.0, "rear": 75.0},
    "suspension": {
        "spring_front_n_per_mm": 420.0,
        "spring_rear_n_per_mm": 380.0,
        "arb_front": 5,
        "arb_rear": 5,
        "damping": "medium",
        "camber_front_deg": -2.75,
        "camber_rear_deg": -1.5,
        "toe_front_deg": -0.05,
        "toe_rear_deg": 0.15,
    },
    "mass": {"ballast_kg": 40.0, "weight_front_pct": 45.0},
    "gearbox": {"ratios": [4.1, 2.95, 2.3, 1.9, 1.62, 1.38, 1.18, 1.0], "final_drive": 5.4},
    "brakes": {"bias_front_pct": 59.0, "duct": 0.45},
    "cooling": {"radiator_opening": 0.70},
    "power_unit": {"fuel_load_kg": 8.5, "ers_harvest_level": 5},
    "tyres": {"start_compound": "soft", "pressure_front_psi": 28.0, "pressure_rear_psi": 28.0},
}


def centerline_dummy_act(obs: Dict[str, Any]) -> Dict[str, Any]:
    """Minimal centerline-following placeholder policy (strictly legal observation only)."""
    c = obs["self"]
    if int(c.get("pit_state", 0)) > 0:
        return {
            "throttle": 0.0, "brake": 0.0, "steer": 0.0,
            "ers_mode": 1, "aero_mode": 0, "override": False, "aid": False,
            "pit_request": False, "pit_compound": None,
        }

    v = float(c["speed_mps"])
    n = float(c["n"])
    h = float(c["heading_error"])
    wet = float(obs["weather"]["track_wetness"])
    rows = (obs.get("preview") or {}).get("rows") or []

    # Preview peak curvature ahead to choose a conservative safe corner speed
    k_now = abs(float(rows[0][1])) if rows else 0.0
    k_ahead = max((abs(float(r[1])) for r in rows[:6]), default=k_now)
    k_eff = max(k_now, k_ahead, 1e-4)
    ay_safe = 18.0 * (1.0 - 0.35 * wet)
    v_target = min(65.0, math.sqrt(ay_safe / k_eff))

    # Simple PD steering toward centerline (n -> 0, heading_error -> 0)
    k_signed = float(rows[0][1]) if rows else 0.0
    steer_rad = 3.4 * k_signed - 0.85 * h - 0.06 * n
    steer = max(-1.0, min(1.0, steer_rad / 0.30))

    dv = v_target - v
    throttle = max(0.0, min(1.0, 0.35 * dv)) if dv > 0.0 else 0.0
    brake = max(0.0, min(1.0, -0.18 * dv)) if dv < -1.0 else 0.0
    ers_mode = 0 if brake > 0.15 else (2 if throttle > 0.7 and float(c["ers_soc_mj"]) > 0.5 else 1)

    return {
        "throttle": float(throttle),
        "brake": float(brake),
        "steer": float(steer),
        "ers_mode": int(ers_mode),
        "aero_mode": 0,
        "override": False,
        "aid": False,
        "pit_request": False,
        "pit_compound": None,
    }


def main() -> None:
    env = {"design": STOCK_DESIGN}
    ok, errs, _ = validate_design_envelope(json.dumps(env).encode("utf-8"))
    if not ok:
        raise RuntimeError(f"Stock design failed validation: {errs}")

    params = SimV5ParamsCPU.load()
    weather = standard_weather_scenarios()["wet_p55"]
    print(f"Loaded FormulaSim-v5 params: mu_f={params.mu_f_base:.3f}, mu_r={params.mu_r_base:.3f}, ice={params.ice_power_scale:.3f}")

    for track_id in ("parkland", "highland", "crossover"):
        res = run_weekend(
            track_id=track_id,
            entries=[{"name": "dummy", "design_envelope": env, "act_fn": centerline_dummy_act}],
            weather_profile=weather,
            params=params,
            race_laps=1,
            run_qualifying=False,
        )
        row = res["classification"][0]
        print(
            f"  {track_id:10s} | status={row['status']:8s} | time={row.get('total_time_s')}s "
            f"| offtrack={row.get('offtrack_ticks')} | faults={row.get('faults')}"
        )


if __name__ == "__main__":
    main()
