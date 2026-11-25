"""Offline unit & physics-invariant smoke tests for FormulaSim-v5 (rsi-racing-4.4)."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import unittest

RACING_DIR = Path(__file__).resolve().parent
if str(RACING_DIR) not in sys.path:
    sys.path.insert(0, str(RACING_DIR))

from circuit_geom import ALL_TRACKS, load_circuit  # noqa: E402
from design_validator import validate_design_envelope  # noqa: E402
from formula_sim_v5 import (  # noqa: E402
    SimV5ParamsCPU,
    run_weekend,
    standard_weather_scenarios,
)
from formula_sim_v5_gpu import wheel_dynamics  # noqa: E402


STOCK_DESIGN = {
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


def _make_dummy_driver(lane_offset: float = 0.0):
    def _act(obs):
        c = obs["self"]
        rows = (obs.get("preview") or {}).get("rows") or []
        k = float(rows[0][1]) if rows else 0.0
        k_ahead = max((abs(float(r[1])) for r in rows[:6]), default=abs(k))
        v = float(c["speed_mps"])
        v_tgt = min(55.0, math.sqrt(14.0 / max(abs(k), k_ahead, 1e-4)))
        n_err = float(c["n"]) - lane_offset
        steer = max(-1.0, min(1.0, (3.4 * k - 0.85 * float(c["heading_error"]) - 0.06 * n_err) / 0.30))
        dv = v_tgt - v
        return {
            "throttle": max(0.0, min(1.0, 0.35 * dv)) if dv > 0 else 0.0,
            "brake": max(0.0, min(1.0, -0.18 * dv)) if dv < -1.0 else 0.0,
            "steer": steer,
            "ers_mode": 0 if dv < -1.5 else 1,
            "aero_mode": 0,
            "override": False,
            "aid": False,
            "pit_request": False,
            "pit_compound": None,
        }
    return _act


class TestRacingSimSmoke(unittest.TestCase):
    def test_circuits_and_configs_load(self) -> None:
        self.assertEqual(tuple(ALL_TRACKS), ("parkland", "highland", "crossover"))
        for tid in ALL_TRACKS:
            geom = load_circuit(tid, resample_ds=4.0)
            self.assertGreater(geom.length, 5000.0)
            self.assertGreater(geom.n_nodes, 1000)

        params = SimV5ParamsCPU.load()
        self.assertTrue(2.2 <= params.mu_f_base <= 2.8)
        self.assertTrue(2.2 <= params.mu_r_base <= 2.8)
        self.assertTrue(0.9 <= params.ice_power_scale <= 1.1)
        self.assertAlmostEqual(params.floor_f_share_rake_k, -0.0015, places=9)

        ens_path = RACING_DIR / "configs" / "sim_v5_ensemble_v3.json"
        ens = json.loads(ens_path.read_text(encoding="utf-8"))
        self.assertEqual(len(ens), 4)

    def test_design_validator(self) -> None:
        ok, errs, _ = validate_design_envelope(json.dumps({"design": STOCK_DESIGN}).encode("utf-8"))
        self.assertTrue(ok, msg=f"Unexpected design errors: {errs}")

    def test_wheel_dynamics_mirror_symmetry_and_lifted_wheel(self) -> None:
        p = SimV5ParamsCPU.load()
        common = dict(
            mass=795.0, wf=0.45, vx=55.0, dt=1.0 / 120.0,
            fz_f_ax=6700.0, fz_r_ax=7900.0, sf=420.0, sr=380.0,
            arb_front=5.0, arb_rear=5.0, damping_ratio=1.0,
            camber_front_deg=-2.75, camber_rear_deg=-1.5,
            pressure_front_psi=28.0, pressure_rear_psi=28.0,
            tyre_surf_c=[85.0, 85.0, 85.0, 85.0],
            tyre_core_c=[85.0, 85.0, 85.0, 85.0],
            tyre_wear=[0.1, 0.1, 0.1, 0.1],
            wet=0.0, off_track=0.0, comp_grip=1.0, comp_t_opt=85.0,
            fx_request=[0.0, 0.0, 4200.0, 4200.0],
            hydraulic_request=[0.0, 0.0, 0.0, 0.0],
            harvest_level=5.0,
        )
        w1 = wheel_dynamics(
            p, vy=1.2, yaw_rate=0.45, steer_rad=0.14, prev_glat=2.1,
            transfer_f=800.0, transfer_r=900.0,
            alpha_eff=[0.06, 0.06, 0.04, 0.04], **common,
        )
        w2 = wheel_dynamics(
            p, vy=-1.2, yaw_rate=-0.45, steer_rad=-0.14, prev_glat=-2.1,
            transfer_f=-800.0, transfer_r=-900.0,
            alpha_eff=[-0.06, -0.06, -0.04, -0.04], **common,
        )
        self.assertAlmostEqual(float(w1["mz_body"]), -float(w2["mz_body"]), places=5)
        self.assertAlmostEqual(float(w1["fx_body"]), float(w2["fx_body"]), places=5)
        self.assertAlmostEqual(float(w1["fy_body"]), -float(w2["fy_body"]), places=5)

        # Extreme roll transfer lifting inner wheel (fz == 0) must yield fx == fy == 0 on lifted wheel
        w_lift = wheel_dynamics(
            p, vy=2.5, yaw_rate=0.9, steer_rad=0.22, prev_glat=6.5,
            transfer_f=5000.0, transfer_r=5000.0,
            alpha_eff=[0.12, 0.12, 0.10, 0.10],
            **{**common, "fz_f_ax": 4000.0, "fz_r_ax": 4000.0},
        )
        for i in range(4):
            if float(w_lift["fz"][i]) <= 1e-8:
                self.assertEqual(float(w_lift["fx"][i]), 0.0)
                self.assertEqual(float(w_lift["fy"][i]), 0.0)

    def test_two_car_weekend_smoke(self) -> None:
        env = {"design": STOCK_DESIGN}
        params = SimV5ParamsCPU.load()
        res = run_weekend(
            track_id="parkland",
            entries=[
                {"name": "car_a", "design_envelope": env, "act_fn": _make_dummy_driver(-2.0)},
                {"name": "car_b", "design_envelope": env, "act_fn": _make_dummy_driver(+2.0)},
            ],
            weather_profile=standard_weather_scenarios()["wet_p55"],
            params=params,
            race_laps=1,
            run_qualifying=False,
        )
        self.assertEqual(len(res["classification"]), 2)
        for row in res["classification"]:
            self.assertEqual(row["faults"], 0)
            self.assertEqual(row["status"], "finished")


if __name__ == "__main__":
    unittest.main()
