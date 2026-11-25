"""Telemetry-Calibratable AGX Formula (rsi-racing-4.4 / rsi-formula-4.0) Physics Simulator.

Built directly from the published parameter registry in
work/racing-bootstrap/docs/racing-sporting.zh.md (lines 2177-2328) and the
3 official circuit geometries (parkland, highland, crossover).

Features:
  - 60 Hz physics (2 substeps -> 120 Hz integration) / 30 Hz agent callbacks
  - Complete aero (body, front/rear wing, floor/rake/stall, active X-mode,
    cooling/brake ducts, slipstream drag cut, dirty-air wake downforce loss)
  - 4-wheel Magic Formula tyres with load sensitivity, C1-C5 compounds,
    pressure/camber effects, 2-node thermal model, wear + 0.80 cliff, wet grip,
    friction ellipse, lockup/wheelspin flags
  - Powertrain: 3000 MJ/h ICE + turbo lag + 8-speed auto-shift + 350 kW MGU-K
    with official normal/override speed-derate curves, coolant thermal + 108C
    derate, 4 MJ battery, 8.5 MJ/lap recharge cap, Manual Overtake Mode
  - Sporting rules: 1-lap rolling qualifying + 4-lap standing-start race,
    pit lane (80 kph + 2.4s stationary), G11/G12 pit target latch, G13
    2-dry-compound on-track forward-progress ledger & track_wetness>=0.5 waiver,
    off-track lap invalidation, wall collision DNF, car-to-car contact penalty,
    30s stall retirement, strict 9-field action contract validation
  - Calibratable residual parameters (`SimCalibration`) fitted directly to
    24 Hz production telemetry replays from agx.cool!
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from circuit_geom import CircuitGeometry, load_circuit, OFFICIAL_AID_ZONES

# Official compound physics table (racing-sporting.zh.md line 2294)
COMPOUND_SPECS: Dict[str, Dict[str, float]] = {
    "C1": {"grip": 0.976, "t_opt": 106.0, "wear": 0.55, "heat": 0.90},
    "C2": {"grip": 0.988, "t_opt": 101.0, "wear": 0.70, "heat": 0.95},
    "C3": {"grip": 1.000, "t_opt": 96.0, "wear": 0.90, "heat": 1.00},
    "C4": {"grip": 1.012, "t_opt": 92.0, "wear": 1.25, "heat": 1.05},
    "C5": {"grip": 1.024, "t_opt": 88.0, "wear": 1.80, "heat": 1.10},
}

# Official MGU-K speed derate curves (matches official mechanics.npz!)
MGUK_SPEED_KPH = np.array([0.0, 290.0, 337.5, 340.0, 345.0, 355.0, 400.0], dtype=np.float64)
MGUK_NORMAL_W = np.array([350000.0, 350000.0, 112500.0, 100000.0, 0.0, 0.0, 0.0], dtype=np.float64)
MGUK_OVERRIDE_W = np.array([350000.0, 350000.0, 350000.0, 300000.0, 200000.0, 0.0, 0.0], dtype=np.float64)
WHEEL_EFFICIENCY = 0.93605  # 0.97 * 0.965

REQUIRED_ACTION_KEYS = ("throttle", "brake", "steer")
ALL_ACTION_KEYS = {
    "throttle",
    "brake",
    "steer",
    "ers_mode",
    "aero_mode",
    "override",
    "aid",
    "pit_request",
    "pit_compound",
}


@dataclass
class SimCalibration:
    """Parameters calibratable against real 24 Hz agx.cool replay telemetry."""

    grip_scale: float = 1.00
    drag_scale: float = 1.00
    downforce_scale: float = 1.00
    ice_power_scale: float = 1.00
    fuel_burn_scale: float = 1.00
    wear_scale: float = 1.00
    heat_scale: float = 1.00
    wet_grip_drop: float = 0.45  # grip factor = 1 - wet_grip_drop * wet^wet_grip_exp
    wet_grip_exp: float = 0.85
    brake_scale: float = 1.00

    @classmethod
    def load(cls, path: Path) -> "SimCalibration":
        if not path.is_file():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        valid = {k: float(v) for k, v in data.items() if hasattr(cls, k)}
        return cls(**valid)

    def to_dict(self) -> Dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")


@dataclass
class WeatherProfile:
    """Deterministic weather trajectory for a weekend (qualifying + race)."""

    condition: str = "dry"
    rain_probability: float = 0.15
    base_rain_intensity: float = 0.0
    base_wetness: float = 0.0
    peak_wetness: float = 0.0
    rain_start_race_s: float = 9999.0
    rain_ramp_s: float = 40.0
    air_temp_c: float = 24.0
    track_temp_c: float = 30.0

    def forecast(self) -> Dict[str, Any]:
        return {
            "condition": self.condition,
            "rain_probability": float(self.rain_probability),
            "rain_intensity": float(self.base_rain_intensity),
            "track_wetness": float(self.base_wetness),
            "air_temp_c": float(self.air_temp_c),
            "track_temp_c": float(self.track_temp_c),
        }

    def at(self, phase: str, phase_time_s: float) -> Dict[str, Any]:
        wet = self.base_wetness
        rain = self.base_rain_intensity
        if phase == "race" and self.peak_wetness > self.base_wetness and phase_time_s >= self.rain_start_race_s:
            frac = min(1.0, (phase_time_s - self.rain_start_race_s) / max(1.0, self.rain_ramp_s))
            wet = self.base_wetness + (self.peak_wetness - self.base_wetness) * frac
            rain = max(rain, 0.55 * frac if wet > 0.25 else 0.20 * frac)
        cond = "wet" if wet >= 0.5 else ("damp" if wet > 0.05 else "dry")
        return {
            "condition": cond,
            "rain_probability": float(self.rain_probability),
            "rain_intensity": float(rain),
            "track_wetness": float(wet),
            "air_temp_c": float(self.air_temp_c),
            "track_temp_c": float(self.track_temp_c - 6.0 * wet),
        }


def standard_weather_scenarios() -> Dict[str, WeatherProfile]:
    """Covers the live AGX weather regimes observed in competition replays (incl. monsoon 1.0)."""
    return {
        "dry_p15": WeatherProfile("dry", 0.15, 0.0, 0.0, 0.0, 9999.0, 40.0, 24.0, 30.0),
        "damp_to_wet_p35": WeatherProfile("dry", 0.35, 0.0, 0.0, 0.62, 70.0, 65.0, 21.0, 26.0),
        "wet_p55": WeatherProfile("dry", 0.55, 0.0, 0.08, 0.78, 10.0, 35.0, 19.0, 22.0),
        "monsoon_p55": WeatherProfile("wet", 0.55, 18.0, 0.65, 1.00, 0.0, 25.0, 17.0, 19.0),
    }


def _is_finite_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def validate_action(act: Any) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """Strict rsi-racing-4.4 9-field action contract validator."""
    fallback = {
        "throttle": 0.0,
        "brake": 1.0,
        "steer": 0.0,
        "ers_mode": 1,
        "aero_mode": 0,
        "override": False,
        "aid": False,
        "pit_request": False,
        "pit_compound": None,
    }
    if not isinstance(act, dict) or len(act) == 0:
        return False, "action must be a non-empty dict", fallback
    extra = set(act.keys()) - ALL_ACTION_KEYS
    if extra:
        return False, f"unexpected action keys: {sorted(extra)}", fallback
    for k in REQUIRED_ACTION_KEYS:
        if k not in act:
            return False, f"missing required key: {k}", fallback

    out = dict(fallback)
    for k, lo, hi in (("throttle", 0.0, 1.0), ("brake", 0.0, 1.0), ("steer", -1.0, 1.0)):
        v = act[k]
        if not _is_finite_num(v) or not (lo <= float(v) <= hi):
            return False, f"invalid {k}={v!r}", fallback
        out[k] = float(v)

    if "ers_mode" in act:
        v = act["ers_mode"]
        if not isinstance(v, int) or isinstance(v, bool) or v not in (0, 1, 2, 3):
            return False, f"invalid ers_mode={v!r}", fallback
        out["ers_mode"] = int(v)

    if "aero_mode" in act:
        v = act["aero_mode"]
        if not isinstance(v, int) or isinstance(v, bool) or v not in (0, 1):
            return False, f"invalid aero_mode={v!r}", fallback
        out["aero_mode"] = int(v)

    for bk in ("override", "aid", "pit_request"):
        if bk in act:
            v = act[bk]
            if not isinstance(v, bool):
                return False, f"invalid bool {bk}={v!r}", fallback
            out[bk] = v

    if "pit_compound" in act:
        v = act["pit_compound"]
        if v is not None and v not in ("soft", "medium", "hard"):
            return False, f"invalid pit_compound={v!r}", fallback
        out["pit_compound"] = v

    return True, None, out


def compute_aero_coeffs(
    design: Dict[str, Any],
    xmode_frac: float,
    rh_f_mm: float,
    rh_r_mm: float,
    cal: SimCalibration,
) -> Tuple[float, float, float]:
    """Return (CzA_total, CdA_total, front_downforce_share) from exact sporting table."""
    fw = float(design["aero"]["front_wing"])
    rw = float(design["aero"]["rear_wing"])
    rad = float(design["cooling"]["radiator_opening"])
    duct = float(design["brakes"]["duct"])

    # Wing Z-mode CzA peaking near stall angles (32/34 deg)
    fw_z = 0.582 * math.sin(min(1.0, fw / 32.0) * (math.pi / 2.0))
    rw_z = 0.747 * math.sin(min(1.0, rw / 34.0) * (math.pi / 2.0))
    fw_cza = fw_z * (1.0 - (1.0 - 0.30) * xmode_frac)
    rw_cza = rw_z * (1.0 - (1.0 - 0.13) * xmode_frac)

    # Floor ground effect + rake + stall (lines 2186-2202)
    rh_f_m = max(0.003, rh_f_mm * 1e-3)
    rh_r_m = max(0.010, rh_r_mm * 1e-3)
    rake_m = max(0.0, rh_r_m - rh_f_m)
    floor_gain = 1.0 + 0.01 * (28.0 - rh_f_mm) + 0.004 * (rake_m * 1000.0 - 34.0)
    floor_gain = max(0.45, min(1.35, floor_gain))
    if rh_f_m < 0.018:
        stall_f = (rh_f_m / 0.018) ** 2.5
        floor_gain *= stall_f
    if rake_m > 0.090:
        floor_gain *= max(0.4, 1.0 - 5.0 * (rake_m - 0.090))
    floor_cza = 0.96 * floor_gain
    body_cza = 0.10

    cza_tot = (fw_cza + rw_cza + floor_cza + body_cza) * cal.downforce_scale
    floor_f_share = max(0.28, min(0.56, 0.42 - 0.0015 * (rake_m * 1000.0 - 34.0)))
    front_cza = fw_cza + floor_cza * floor_f_share + 0.45 * body_cza
    front_share = front_cza / max(fw_cza + rw_cza + floor_cza + body_cza, 1e-6)

    # Drag CdA (lines 2181-2207)
    fw_prof = 0.040 * (1.0 - xmode_frac) + 0.035 * xmode_frac
    rw_prof = 0.050 * (1.0 - xmode_frac) + 0.035 * xmode_frac
    fw_ind = 0.14 * fw_cza * fw_cza
    rw_ind = 0.39 * rw_cza * rw_cza
    floor_ind = 0.04 * (floor_cza / 0.96) ** 2
    cool_cda = 0.145 * (rad ** 1.25)
    duct_cda = 0.020 * duct
    cda_tot = (0.478 + fw_prof + rw_prof + fw_ind + rw_ind + floor_ind + cool_cda + duct_cda) * cal.drag_scale

    return cza_tot, cda_tot, front_share


def compute_ice_power_w(rpm: float, coolant_c: float, cal: SimCalibration) -> float:
    """ICE wheel-input power from 3000 MJ/h fuel flow and efficiency curve (lines 2269-2276)."""
    # 3000 MJ/h = 833,333.33 W thermal; peak eta = 0.486 -> 405 kW at 12500 rpm
    x = (rpm - 12500.0) / 3500.0
    eta = max(0.22, 0.486 * (1.0 - 0.034 * x * x - 0.14 * max(0.0, -x) ** 2))
    # Rev limiter soft drop above 15000 rpm
    if rpm > 14800.0:
        eta *= max(0.2, 1.0 - (rpm - 14800.0) / 1500.0)
    p = (3000e6 / 3600.0) * eta * cal.ice_power_scale
    if coolant_c > 108.0:
        derate = min(0.35, 0.03 * (coolant_c - 108.0))
        p *= 1.0 - derate
    return max(0.0, p)


def compute_mguk_power_w(speed_mps: float, override_active: bool) -> float:
    kph = speed_mps * 3.6
    curve = MGUK_OVERRIDE_W if override_active else MGUK_NORMAL_W
    return float(np.interp(kph, MGUK_SPEED_KPH, curve))


def tyre_grip_mu(
    compound_spec: Dict[str, float],
    axle: str,
    fz_n: float,
    temp_c: float,
    wear: float,
    pressure_psi: float,
    camber_deg: float,
    wetness: float,
    cal: SimCalibration,
) -> float:
    """Effective peak tyre friction coefficient matching racing-sporting.zh.md."""
    if axle == "front":
        fz0, mu0, sens, g_opt, p_ref = 3800.0, 2.52, 0.26, -3.3, 29.0
    else:
        fz0, mu0, sens, g_opt, p_ref = 5200.0, 2.52, 0.20, -2.4, 29.0

    # Load sensitivity
    load_ratio = max(0.15, fz_n / fz0)
    mu = mu0 * (load_ratio ** (-sens)) * compound_spec["grip"]

    # Temperature window (45C cold / 40C hot, floor 0.45)
    t_opt = compound_spec["t_opt"]
    w_t = 45.0 if temp_c < t_opt else 40.0
    therm = max(0.45, 1.0 - 0.5 * ((temp_c - t_opt) / w_t) ** 2)
    mu *= therm

    # Wear linear + quadratic cliff above 0.80
    wear_c = max(0.0, min(1.0, wear))
    w_fac = max(0.45, 1.0 - 0.05 * wear_c - 0.40 * (max(0.0, wear_c - 0.80) / 0.20) ** 2)
    mu *= w_fac

    # Pressure & camber penalties (lines 2299-2300)
    mu *= max(0.85, 1.0 - 0.01 * (pressure_psi - p_ref))
    mu *= max(0.85, 1.0 - 0.0035 * ((camber_deg - g_opt) ** 2))

    # Wetness grip loss
    if wetness > 0.0:
        wet_fac = max(0.35, 1.0 - cal.wet_grip_drop * (wetness ** cal.wet_grip_exp))
        mu *= wet_fac

    return mu * cal.grip_scale


def effective_mass_for_gear(dry_mass: float, fuel_kg: float, total_ratio: float) -> float:
    """Effective longitudinal mass including wheel + crank rotational inertia."""
    r_wheel = 0.343
    m_wheels = 5.0 / (r_wheel * r_wheel)
    m_crank = 0.08 * (total_ratio / r_wheel) ** 2
    return dry_mass + fuel_kg + m_wheels + m_crank


def compute_ggv_capability_point(
    design: Dict[str, Any],
    compound_spec: Dict[str, float],
    speed_mps: float,
    temp_c: float,
    wetness: float,
    xmode: int,
    fuel_kg: float,
    cal: SimCalibration,
) -> Tuple[float, float, float, float]:
    """Compute (ay_max_mps2, fb_max_n, drag_n, roll_n) for offline GGV planning."""
    dry_mass = 748.0 + float(design["mass"]["ballast_kg"])
    mass = dry_mass + fuel_kg
    rh_f = float(design["ride_height_mm"]["front"])
    rh_r = float(design["ride_height_mm"]["rear"])
    cza, cda, f_share = compute_aero_coeffs(design, float(xmode), rh_f, rh_r, cal)
    q_dyn = 0.5 * 1.2 * speed_mps * speed_mps
    downforce = q_dyn * cza
    drag = q_dyn * cda
    fz_tot = mass * 9.80665 + downforce
    roll = 0.012 * fz_tot

    wf = float(design["mass"]["weight_front_pct"]) / 100.0
    fz_f = (mass * 9.80665 * wf + downforce * f_share) * 0.5
    fz_r = (mass * 9.80665 * (1.0 - wf) + downforce * (1.0 - f_share)) * 0.5

    pf = float(design["tyres"]["pressure_front_psi"])
    pr = float(design["tyres"]["pressure_rear_psi"])
    cf = float(design["suspension"]["camber_front_deg"])
    cr = float(design["suspension"]["camber_rear_deg"])

    mu_f = tyre_grip_mu(compound_spec, "front", fz_f, temp_c, 0.0, pf, cf, wetness, cal)
    mu_r = tyre_grip_mu(compound_spec, "rear", fz_r, temp_c, 0.0, pr, cr, wetness, cal)

    fy_max = 2.0 * fz_f * mu_f + 2.0 * fz_r * mu_r
    # Realistic steady-state cornering efficiency under lateral load transfer (~0.95)
    ay_raw = min(fy_max / mass, 2.0 * fz_f * mu_f / (mass * wf + 1e-6), 2.0 * fz_r * mu_r / (mass * (1.0 - wf) + 1e-6))
    ay_max = ay_raw * 0.95
    fb_tyre = (2.0 * fz_f * mu_f + 2.0 * fz_r * mu_r) * 1.02 * cal.brake_scale
    fb_max = min(56000.0, fb_tyre)
    return float(ay_max), float(fb_max), float(drag), float(roll)


class CarState:
    """Single car physics & sporting state inside FormulaSim."""

    def __init__(
        self,
        seat: int,
        design: Dict[str, Any],
        geom: CircuitGeometry,
        cal: SimCalibration,
    ):
        self.seat = seat
        self.design = design
        self.geom = geom
        self.cal = cal
        self.dry_mass = 748.0 + float(design["mass"]["ballast_kg"])
        self.wf = float(design["mass"]["weight_front_pct"]) / 100.0
        self.ratios = [float(r) for r in design["gearbox"]["ratios"]]
        self.final_drive = float(design["gearbox"]["final_drive"])
        self.ers_harvest_level = int(design["power_unit"]["ers_harvest_level"])
        self.reset_for_phase("qualifying", grid_slot=seat)

    def reset_for_phase(self, phase: str, grid_slot: int = 0) -> None:
        self.status = "racing"
        self.fault_reason: Optional[str] = None
        self.retire_reason: Optional[str] = None
        self.lap = 0
        self.pit_stops = 0
        self.pit_state = 0
        self.pit_timer = 0.0
        self.pit_lane_progress = 0.0
        self.pit_request = False
        self.pit_compound_target: Optional[str] = None
        self.pit_locked_compound: Optional[str] = None

        self.compound = str(self.design["tyres"]["start_compound"])
        self.fuel_kg = float(self.design["power_unit"]["fuel_load_kg"])
        self.ers_soc_mj = 3.0
        self.ers_deployed_mj = 0.0
        self.ers_harvested_mj = 0.0
        self.ers_lap_harvested_mj = 0.0
        self.ers_power_kw = 0.0
        self.pu_temp_c = 88.0
        self.turbo_boost = 0.0

        self.tyre_surf_c = np.array([84.0, 84.0, 84.0, 84.0], dtype=np.float64)
        self.tyre_core_c = np.array([80.0, 80.0, 80.0, 80.0], dtype=np.float64)
        self.tyre_wear = np.zeros(4, dtype=np.float64)
        self.brake_temp_c = np.array([500.0, 500.0, 500.0, 500.0], dtype=np.float64)

        self.gear = 1
        self.rpm = 7500.0
        self.shift_timer = 0.0
        self.shift_cooldown = 0.0
        self.steer_rad = 0.0
        self.alpha_f_eff = 0.0
        self.alpha_r_eff = 0.0
        self.xmode_frac = 0.0
        self.g_lat = 0.0
        self.g_long = 0.0
        self.drag_n = 0.0
        self.downforce_n = 0.0
        self.rolling_resistance_n = 0.0
        self.drive_force_n = [0.0, 0.0, 0.0, 0.0]
        self.front_sliding = False
        self.rear_sliding = False
        self.ride_height_mm = [
            float(self.design["ride_height_mm"]["front"]),
            float(self.design["ride_height_mm"]["rear"]),
        ]

        self.aid_pending = False
        self.aid_available = False
        self.aid_zone: Optional[int] = None
        self.aid_active = False
        self.override_available = False
        self.override_active = False
        self.aid_extra_recharge_mj = 0.0
        self.aid_extra_deployed_mj = 0.0

        # Sporting ledgers (G13 & lap validity & stall check)
        self.compounds_used: set[str] = set()
        self.wet_waiver = False
        self.prev_on_track = False
        self.prev_on_track_dist = 0.0
        self.current_lap_offtrack = False
        self.lap_start_time = 0.0
        self.lap_times: List[Tuple[float, bool]] = []  # (time_s, valid)
        self.finish_time_s: Optional[float] = None
        self.stints: List[Dict[str, Any]] = [{"compound": self.compound, "laps": 0}]
        self.stall_window_dist: List[Tuple[float, float]] = []
        self.offtrack_ticks = 0
        self.collision_count = 0

        self.last_action = {
            "throttle": 0.0,
            "brake": 0.0,
            "steer": 0.0,
            "ers_mode": 1,
            "aero_mode": 0,
            "override": False,
            "aid": False,
            "pit_request": False,
            "pit_compound": None,
        }

        if phase == "qualifying":
            self.distance = -420.0
            self.s = (self.geom.length - 420.0) % self.geom.length
            self.n = 0.0
            self.heading_error = 0.0
            self.vx = 22.0
            self.vy = 0.0
            self.yaw_rate = 0.0
        else:
            slot = self.geom.start_grid[min(grid_slot, len(self.geom.start_grid) - 1)]
            # Standing start behind start/finish line
            s_slot = float(slot["s"])
            back_from_line = (self.geom.length - s_slot) % self.geom.length
            if back_from_line > self.geom.length * 0.5:
                back_from_line = 8.0 * (grid_slot + 1)
            self.distance = -max(1.0, back_from_line)
            self.s = s_slot % self.geom.length
            self.n = float(slot.get("lateral_m", -3.5 if grid_slot % 2 == 0 else 3.5))
            self.heading_error = 0.0
            self.vx = 0.0
            self.vy = 0.0
            self.yaw_rate = 0.0

        self._sync_world_pose()

    def _sync_world_pose(self) -> None:
        x0, y0, z0, track_yaw = self.geom.frenet_to_world(self.s, self.n)
        self.x, self.y, self.z = x0, y0, z0
        self.yaw = (track_yaw + self.heading_error + math.pi) % (2.0 * math.pi) - math.pi

    def effective_mass(self) -> float:
        tot_ratio = self.ratios[self.gear - 1] * self.final_drive
        return effective_mass_for_gear(self.dry_mass, self.fuel_kg, tot_ratio)

    def update_drive_force_vector(self) -> None:
        v = max(self.vx, 1.0)
        tot_ratio = self.ratios[self.gear - 1] * self.final_drive
        rpm = max(7500.0, min(15500.0, (v / 0.343) * tot_ratio * 60.0 / (2.0 * math.pi)))
        ice_w = compute_ice_power_w(rpm, self.pu_temp_c, self.cal) * 0.965
        if self.fuel_kg <= 0.0:
            ice_w = 0.0
        f_ice = ice_w / max(v, 8.0)
        # Launch acceleration limit (1.2g below 30 m/s)
        if v < 30.0:
            f_ice = min(f_ice, (self.dry_mass + self.fuel_kg) * 1.2 * 9.80665)
        mguk_norm = compute_mguk_power_w(v, override_active=False) * WHEEL_EFFICIENCY
        f_m_full = (mguk_norm / max(v, 8.0)) if self.ers_soc_mj > 0.01 else 0.0
        regen_w = 70000.0 * self.ers_harvest_level * 0.965
        f_harvest = max(0.0, f_ice - regen_w / max(v, 10.0))
        self.drive_force_n = [
            float(f_harvest),
            float(f_ice),
            float(f_ice + 0.5 * f_m_full),
            float(f_ice + f_m_full),
        ]

    def step_sub(
        self,
        dt: float,
        phase: str,
        phase_time_s: float,
        weather: Dict[str, Any],
        slipstream_cut: float,
        wake_loss: float,
    ) -> None:
        if self.status != "racing":
            return

        act = self.last_action
        wet = float(weather["track_wetness"])
        air_t = float(weather["air_temp_c"])
        track_t = float(weather["track_temp_c"])
        mass = self.dry_mass + max(0.0, self.fuel_kg)
        comp_id = self.geom.compound_map[self.compound]
        comp_spec = COMPOUND_SPECS[comp_id]

        # Handle Pit Lane autonomous transit (G11/G12)
        if self.pit_state > 0:
            self._step_pit_lane(dt, phase_time_s)
            return

        # 1) Steering actuator rate limit (1.6 rad/s, max 0.3 rad)
        steer_target = act["steer"] * 0.3
        max_dsteer = 1.6 * dt
        self.steer_rad += max(-max_dsteer, min(max_dsteer, steer_target - self.steer_rad))

        # 2) Active aero X-mode eligibility (zone remaining > 45m, brake < 0.05)
        aero_rem = self.geom.interp_scalar(self.geom.aero_remaining, self.s)
        in_xzone = self.geom.interp_scalar(self.geom.xmode, self.s) > 0.5 and aero_rem > 45.0
        want_x = 1.0 if (act["aero_mode"] == 1 and in_xzone and act["brake"] < 0.05) else 0.0
        self.xmode_frac += max(-dt / 0.4, min(dt / 0.4, want_x - self.xmode_frac))

        # 3) Aero forces + realistic dynamic ride height compression (series spring+tyre + braking pitch + vertical curvature!)
        v = math.hypot(self.vx, self.vy)
        v_sq = v * v
        sf = float(self.design["suspension"]["spring_front_n_per_mm"])
        sr = float(self.design["suspension"]["spring_rear_n_per_mm"])
        # Official racing-sporting.zh.md line 2243-2244: tyre vertical stiffness 380/450 N/mm in series with suspension springs
        k_f = 2.0 * (sf * 380.0) / max(sf + 380.0, 1.0)  # effective axle N/mm
        k_r = 2.0 * (sr * 450.0) / max(sr + 450.0, 1.0)
        cza0, cda0, f_share0 = compute_aero_coeffs(
            self.design, self.xmode_frac, self.ride_height_mm[0], self.ride_height_mm[1], self.cal
        )
        df0 = 0.5 * 1.2 * v_sq * cza0 * (1.0 - wake_loss)
        # Vertical curvature compression (Eau Rouge up to +0.8g, crests -0.3g; line 2248-2249) + braking dive
        g_ahead = self.geom.interp_scalar(self.geom.grade, self.s + 6.0)
        g_behind = self.geom.interp_scalar(self.geom.grade, self.s - 6.0)
        a_vert = max(-3.0, min(8.0, v_sq * (g_ahead - g_behind) / 12.0))
        d_fz_pitch = mass * (self.g_long * 9.80665) * 0.272 / 3.4
        extra_f = df0 * f_share0 + mass * a_vert * self.wf - d_fz_pitch
        extra_r = df0 * (1.0 - f_share0) + mass * a_vert * (1.0 - self.wf) + d_fz_pitch
        rh_f = max(3.0, float(self.design["ride_height_mm"]["front"]) - max(-5.0, extra_f / max(k_f, 80.0)))
        rh_r = max(8.0, float(self.design["ride_height_mm"]["rear"]) - max(-5.0, extra_r / max(k_r, 80.0)))
        self.ride_height_mm = [rh_f, rh_r]

        cza, cda, f_share = compute_aero_coeffs(self.design, self.xmode_frac, rh_f, rh_r, self.cal)
        self.downforce_n = 0.5 * 1.2 * v_sq * cza * (1.0 - wake_loss)
        self.drag_n = 0.5 * 1.2 * v_sq * cda * (1.0 - slipstream_cut)

        # 4) Vertical axle loads with longitudinal transfer
        grade = self.geom.interp_scalar(self.geom.grade, self.s)
        fz_total = max(2000.0, mass * 9.80665 + self.downforce_n)
        self.rolling_resistance_n = 0.012 * fz_total
        d_fz_long = mass * (self.g_long * 9.80665) * 0.272 / 3.4
        fz_front_axle = max(800.0, mass * 9.80665 * self.wf + self.downforce_n * f_share - d_fz_long)
        fz_rear_axle = max(800.0, mass * 9.80665 * (1.0 - self.wf) + self.downforce_n * (1.0 - f_share) + d_fz_long)

        # 5) Automatic 8-speed gearbox + RPM + shift torque cut
        tot_ratio = self.ratios[self.gear - 1] * self.final_drive
        wheel_rpm = max(0.0, self.vx / 0.343) * 60.0 / (2.0 * math.pi)
        self.rpm = max(7500.0, wheel_rpm * tot_ratio)
        self.shift_cooldown = max(0.0, self.shift_cooldown - dt)
        self.shift_timer = max(0.0, self.shift_timer - dt)
        if self.shift_cooldown <= 0.0:
            if self.gear < 8 and self.rpm > 14200.0:
                self.gear += 1
                self.shift_timer = 0.05
                self.shift_cooldown = 0.25
            elif self.gear > 1 and self.rpm < 9200.0:
                next_ratio = self.ratios[self.gear - 2] * self.final_drive
                if wheel_rpm * next_ratio < 13600.0:
                    self.gear -= 1
                    self.shift_timer = 0.05
                    self.shift_cooldown = 0.25

        # 6) Powertrain (ICE + Turbo + MGU-K + Manual Overtake Mode)
        throttle = act["throttle"]
        brake = act["brake"]
        tau_turbo = 0.10 if throttle > self.turbo_boost else 0.35
        self.turbo_boost += (throttle - self.turbo_boost) * min(1.0, dt / tau_turbo)

        ice_w = 0.0
        if self.fuel_kg > 0.0 and self.shift_timer <= 0.0:
            ice_w = compute_ice_power_w(self.rpm, self.pu_temp_c, self.cal) * self.turbo_boost * 0.965
            # Fuel burn: 3000 MJ/h thermal at full throttle; ~44 MJ/kg fuel -> 0.01894 kg/s peak
            burn_kg_s = (3000e6 / 3600.0 / 44.0e6) * self.turbo_boost * self.cal.fuel_burn_scale
            self.fuel_kg = max(0.0, self.fuel_kg - burn_kg_s * dt)

        # Override / Aid eligibility gate (shared gate in race, brake < 0.05)
        ov_req = bool((act["override"] or act["aid"]) and self.aid_available and brake < 0.05 and phase == "race")
        self.override_active = bool(ov_req and self.ers_soc_mj > 0.01 and throttle > 0.1)
        self.aid_active = self.override_active

        ers_mode = act["ers_mode"]
        ers_w = 0.0  # positive = deploy to wheels, negative = harvest from wheels/ICE
        recharge_cap = 8.5 + self.aid_extra_recharge_mj
        if self.shift_timer <= 0.0:
            if ers_mode in (2, 3) and throttle > 0.05 and self.ers_soc_mj > 0.005:
                max_m = compute_mguk_power_w(self.vx, self.override_active)
                frac = 1.0 if (ers_mode == 3 or self.override_active) else 0.5
                req_dc = max_m * frac * throttle
                # Cap by available SOC
                max_dc_soc = max(0.0, (self.ers_soc_mj - 0.002) * 1e6 / dt)
                dc_w = min(req_dc, max_dc_soc)
                ers_w = dc_w * WHEEL_EFFICIENCY
                d_mj = (dc_w * dt) / 1e6
                self.ers_soc_mj = max(0.0, self.ers_soc_mj - d_mj)
                self.ers_deployed_mj += d_mj
                if self.override_active:
                    self.aid_extra_deployed_mj += d_mj
            elif (ers_mode == 0 or brake > 0.05 or throttle < 0.15) and self.ers_soc_mj < 4.0 and self.ers_lap_harvested_mj < recharge_cap:
                if self.vx > 12.0:
                    regen_cap = 70000.0 * self.ers_harvest_level
                    if brake > 0.05:
                        regen_w = regen_cap * min(1.0, brake * 1.4)
                    elif ers_mode == 0:
                        regen_w = regen_cap * (0.7 if throttle < 0.2 else 0.35)
                    else:
                        regen_w = regen_cap * 0.25 * (1.0 - throttle)
                    avail_headroom = min(4.0 - self.ers_soc_mj, recharge_cap - self.ers_lap_harvested_mj) * 1e6 / dt
                    actual_dc = min(regen_w * 0.97, max(0.0, avail_headroom))
                    d_mj = (actual_dc * dt) / 1e6
                    self.ers_soc_mj = min(4.0, self.ers_soc_mj + d_mj)
                    self.ers_harvested_mj += d_mj
                    self.ers_lap_harvested_mj += d_mj
                    ers_w = -(actual_dc / 0.97)
        self.ers_power_kw = ers_w / 1000.0

        # 7) Tyre grip, lateral/longitudinal forces, friction ellipse
        pf = float(self.design["tyres"]["pressure_front_psi"])
        pr = float(self.design["tyres"]["pressure_rear_psi"])
        cf = float(self.design["suspension"]["camber_front_deg"])
        cr = float(self.design["suspension"]["camber_rear_deg"])

        t_f = 0.5 * (self.tyre_surf_c[0] + self.tyre_surf_c[1])
        t_r = 0.5 * (self.tyre_surf_c[2] + self.tyre_surf_c[3])
        w_f = max(self.tyre_wear[0], self.tyre_wear[1])
        w_r = max(self.tyre_wear[2], self.tyre_wear[3])

        mu_f = tyre_grip_mu(comp_spec, "front", fz_front_axle * 0.5, t_f, w_f, pf, cf, wet, self.cal)
        mu_r = tyre_grip_mu(comp_spec, "rear", fz_rear_axle * 0.5, t_r, w_r, pr, cr, wet, self.cal)

        # Slip angles with published tyre relaxation length (front 0.35m, rear 0.40m; line 2290)
        a_cg = 3.4 * (1.0 - self.wf)
        b_cg = 3.4 * self.wf
        vx_safe = max(abs(self.vx), 4.0)
        alpha_f_raw = self.steer_rad - math.atan2(self.vy + a_cg * self.yaw_rate, vx_safe)
        alpha_r_raw = -math.atan2(self.vy - b_cg * self.yaw_rate, vx_safe)
        self.alpha_f_eff += (alpha_f_raw - self.alpha_f_eff) * min(1.0, (vx_safe / 0.35) * dt)
        self.alpha_r_eff += (alpha_r_raw - self.alpha_r_eff) * min(1.0, (vx_safe / 0.40) * dt)
        alpha_f = self.alpha_f_eff
        alpha_r = self.alpha_r_eff

        # Magic Formula lateral force using exact published c1/c2/cy/ey (racing-sporting.zh.md line 2290)
        def _mf_lat(alpha: float, fz_axle: float, mu: float, fz0: float, c1: float, c2: float, p_psi: float) -> Tuple[float, float]:
            d_peak = fz_axle * mu
            cy, ey = 1.35, -0.5
            fz_wheel = 0.5 * fz_axle
            stiff_p = 1.0 + 0.012 * (p_psi - 29.0)
            bcd_wheel = c1 * fz0 * math.sin(2.0 * math.atan(fz_wheel / (c2 * fz0))) * stiff_p
            # Effective axle cornering stiffness (N/rad)
            c_axle = 2.0 * bcd_wheel * 0.45
            b_val = c_axle / max(cy * d_peak, 1e-4)
            ba = b_val * alpha
            fy = d_peak * math.sin(cy * math.atan(ba - ey * (ba - math.atan(ba))))
            return fy, c_axle

        fy_f_raw, cf_ax = _mf_lat(alpha_f, fz_front_axle, mu_f, 3800.0, 58.0, 1.4, pf)
        fy_r_raw, cr_ax = _mf_lat(alpha_r, fz_rear_axle, mu_r, 5200.0, 56.0, 1.4, pr)

        # Requested longitudinal wheel forces with FIA-C C11.6 brake-by-wire blending:
        # MGU-K regen replaces rear hydraulic braking rather than stacking on top of it!
        f_drive_req = (ice_w + max(0.0, ers_w)) / max(self.vx, 7.0)
        if self.vx < 30.0:
            f_drive_req = min(f_drive_req, mass * 1.2 * 9.80665)
        f_regen_brake = max(0.0, -ers_w) / max(self.vx, 7.0)
        f_brake_tot = brake * 56000.0 * self.cal.brake_scale
        bias_f = float(self.design["brakes"]["bias_front_pct"]) / 100.0
        fx_f_req = -f_brake_tot * bias_f
        rear_brake_req = max(f_brake_tot * (1.0 - bias_f), f_regen_brake) if brake > 0.02 else f_regen_brake
        fx_r_req = f_drive_req - rear_brake_req

        # Friction ellipse coupling per axle + sliding flags
        fmax_f = fz_front_axle * mu_f * 1.05
        fmax_r = fz_rear_axle * mu_r * 1.05
        util_f = math.hypot(fx_f_req / max(fmax_f, 1.0), fy_f_raw / max(fz_front_axle * mu_f, 1.0))
        util_r = math.hypot(fx_r_req / max(fmax_r, 1.0), fy_r_raw / max(fz_rear_axle * mu_r, 1.0))

        self.front_sliding = util_f > 1.02 or abs(alpha_f) > 0.14
        self.rear_sliding = util_r > 1.02 or abs(alpha_r) > 0.14

        if util_f > 1.0:
            loss_f = 1.0 - 0.12 * min(1.0, max(0.0, util_f - 1.0) / 0.25)
            fx_f = fx_f_req / util_f * loss_f
            fy_f = fy_f_raw / util_f * loss_f
        else:
            fx_f, fy_f = fx_f_req, fy_f_raw

        if util_r > 1.0:
            loss_r = 1.0 - (0.07 if fx_r_req > 0 else 0.12) * min(1.0, max(0.0, util_r - 1.0) / 0.25)
            fx_r = fx_r_req / util_r * loss_r
            fy_r = fy_r_raw / util_r * loss_r
        else:
            fx_r, fy_r = fx_r_req, fy_r_raw

        # Off-track grass/gravel extra drag & grip loss (n outside road width)
        wl = self.geom.interp_scalar(self.geom.left, self.s)
        wr = self.geom.interp_scalar(self.geom.right, self.s)
        off_track = self.n > wl or self.n < -wr
        extra_off_drag = (mass * 4.5 if off_track and self.vx > 8.0 else 0.0)
        if off_track:
            fy_f *= 0.72
            fy_r *= 0.72
            self.current_lap_offtrack = True
            self.offtrack_ticks += 1

        # 8) Integrate rigid body kinematics (bicycle-coupled transient)
        fx_body = (
            fx_r
            + fx_f * math.cos(self.steer_rad)
            - fy_f * math.sin(self.steer_rad)
            - self.drag_n
            - self.rolling_resistance_n
            - extra_off_drag
            - mass * 9.80665 * grade
        )
        fy_body = fy_r + fy_f * math.cos(self.steer_rad) + fx_f * math.sin(self.steer_rad)
        mz_body = a_cg * (fy_f * math.cos(self.steer_rad) + fx_f * math.sin(self.steer_rad)) - b_cg * fy_r

        eff_m = self.effective_mass() if fx_body > 0 else mass
        iz = mass * (1.15 ** 2)

        ax_body = fx_body / eff_m
        ay_body = fy_body / mass
        dvx = ax_body + self.yaw_rate * self.vy
        dvy = ay_body - self.yaw_rate * self.vx
        dr = mz_body / iz

        self.vx = max(0.0, self.vx + dvx * dt)
        if self.vx < 0.2 and throttle < 0.05:
            self.vx = 0.0
            self.vy = 0.0
            self.yaw_rate = 0.0
        else:
            # Semi-implicit stiff tyre integration (unconditionally stable across all speeds!)
            if self.vx < 5.0:
                self.vy = 0.0
                self.yaw_rate = self.vx * math.tan(self.steer_rad) / 3.4
                self.alpha_f_eff = 0.0
                self.alpha_r_eff = 0.0
            else:
                # Linearized implicit diagonal stabilization matching Pacejka cornering stiffness
                k_vy = (cf_ax + cr_ax) / (mass * vx_safe)
                k_r = (a_cg * a_cg * cf_ax + b_cg * b_cg * cr_ax) / (iz * vx_safe)
                self.vy = max(-25.0, min(25.0, self.vy + (dvy * dt) / (1.0 + 0.5 * k_vy * dt)))
                self.yaw_rate = max(-3.0, min(3.0, self.yaw_rate + (dr * dt) / (1.0 + 0.5 * k_r * dt)))

        self.g_long = ax_body / 9.80665
        self.g_lat = ay_body / 9.80665

        # 9) Advance Frenet coordinates (s, n, heading_error)
        kappa = self.geom.interp_scalar(self.geom.curvature, self.s)
        denom = max(0.25, 1.0 - kappa * self.n)
        ds_dt = (self.vx * math.cos(self.heading_error) - self.vy * math.sin(self.heading_error)) / denom
        dn_dt = self.vx * math.sin(self.heading_error) + self.vy * math.cos(self.heading_error)
        dh_dt = self.yaw_rate - kappa * ds_dt

        prev_s = self.s
        prev_dist = self.distance
        self.distance += ds_dt * dt
        self.s = (self.s + ds_dt * dt) % self.geom.length
        self.n += dn_dt * dt
        self.heading_error = (self.heading_error + dh_dt * dt + math.pi) % (2.0 * math.pi) - math.pi

        # 10) Wall collision check (collidable wall -> severe impact retires car)
        wall_l = self.geom.interp_scalar(self.geom.wall_left, self.s)
        wall_r = self.geom.interp_scalar(self.geom.wall_right, self.s)
        if self.n > wall_l - 0.95 or self.n < -(wall_r - 0.95):
            impact_v = abs(dn_dt) + 0.25 * self.vx
            if impact_v > 14.0:
                self.status = "retired"
                self.retire_reason = "wall_crash"
                self.vx = 0.0
                self.vy = 0.0
                return
            # Glancing scrape slows car and bounces back inside wall
            self.n = max(-(wall_r - 1.05), min(wall_l - 1.05, self.n))
            self.vx *= 0.65
            self.vy *= -0.2
            self.heading_error *= 0.5
            self.collision_count += 1

        # 11) Update tyre 2-node thermal model, wear, and PU coolant temp
        slip_work_f = (abs(fy_f * math.sin(alpha_f)) + max(0.0, util_f - 0.65) * abs(fx_f) * 0.15) * max(v, 1.0)
        slip_work_r = (abs(fy_r * math.sin(alpha_r)) + max(0.0, util_r - 0.65) * abs(fx_r) * 0.15) * max(v, 1.0)
        roll_h_f = 0.012 * fz_front_axle * v * 0.5
        roll_h_r = 0.012 * fz_rear_axle * v * 0.5

        for idx in range(4):
            is_front = idx < 2
            area_scale = 1.0 if is_front else 1.3392857
            sw = (0.5 * slip_work_f if is_front else 0.5 * slip_work_r * 0.78) * comp_spec["heat"] * self.cal.heat_scale
            rh = roll_h_f if is_front else roll_h_r
            q_surf = 0.7 * sw
            q_core = 0.3 * sw + rh
            # Cooling (convection + track conduction + core exchange)
            h_surf = (15.0 + 0.5 * v) * area_scale
            h_core = (40.0 + 1.0 * v) * area_scale
            cond_sc = 90.0 * area_scale * (self.tyre_surf_c[idx] - self.tyre_core_c[idx])
            cond_tr = (25.0 + 35.0 * wet) * area_scale * (self.tyre_surf_c[idx] - track_t)
            conv_s = h_surf * (self.tyre_surf_c[idx] - air_t)
            conv_c = h_core * (self.tyre_core_c[idx] - air_t)
            self.tyre_surf_c[idx] += ((q_surf - cond_sc - cond_tr - conv_s) / (5500.0 * area_scale)) * dt
            self.tyre_core_c[idx] += ((q_core + cond_sc - conv_c) / (26000.0 * area_scale)) * dt
            self.tyre_surf_c[idx] = max(air_t, min(165.0, self.tyre_surf_c[idx]))
            self.tyre_core_c[idx] = max(air_t, min(155.0, self.tyre_core_c[idx]))

            # Wear energy (26 MJ per unit for C3) + overheat multiplier
            over_t = max(0.0, self.tyre_surf_c[idx] - (comp_spec["t_opt"] + 12.0))
            over_mult = 1.0 + (over_t / 20.0) ** 2
            p_psi = pf if is_front else pr
            p_mult = 1.0 + 0.06 * max(0.0, 29.0 - p_psi)
            dw = (sw * dt / 26e6) * comp_spec["wear"] * over_mult * p_mult * self.cal.wear_scale
            self.tyre_wear[idx] = min(1.0, self.tyre_wear[idx] + dw)

        # Coolant temperature (lines 2265, 2281, 2287)
        rad = float(self.design["cooling"]["radiator_opening"])
        heat_in = 0.37 * (ice_w / max(0.45, 0.486) * (1.0 - 0.486) + abs(ers_w) * 0.06)
        heat_out = 1500.0 * (rad ** 0.85) * (1.0 + v / 70.0) * max(0.0, self.pu_temp_c - air_t)
        self.pu_temp_c = max(air_t, min(145.0, self.pu_temp_c + ((heat_in - heat_out) / 120000.0) * dt))

        # 12) Pit entry trigger (G11/G12)
        if phase == "race" and self.pit_request and self.distance > 100.0:
            if _crossed_s(prev_s, self.s, self.geom.pit_entry_s, self.geom.length):
                self.pit_state = 1
                self.pit_lane_progress = 0.0
                self.pit_timer = 0.0
                self.pit_locked_compound = self.pit_compound_target or self.compound
                self.aid_available = False
                self.aid_pending = False
                self.aid_zone = None
                self.override_available = False

        # 13) G13 compound use & wet waiver accounting (strict legal forward on-track motion!)
        if phase == "race":
            on_track_now = not off_track and self.pit_state == 0 and self.distance >= 0.0
            if on_track_now and self.prev_on_track and self.distance > self.prev_on_track_dist + 1e-6:
                self.compounds_used.add(self.compound)
                if wet >= 0.5:
                    self.wet_waiver = True
            self.prev_on_track = on_track_now
            self.prev_on_track_dist = self.distance

        # 14) Lap crossing detection
        prev_lap_idx = math.floor(prev_dist / self.geom.length)
        curr_lap_idx = math.floor(self.distance / self.geom.length)
        if curr_lap_idx > prev_lap_idx:
            # Start line crossed!
            if prev_lap_idx < 0:
                self.lap_start_time = phase_time_s
                self.current_lap_offtrack = False
            else:
                lap_dt = phase_time_s - self.lap_start_time
                self.lap_times.append((lap_dt, not self.current_lap_offtrack))
                self.lap += 1
                if self.stints:
                    self.stints[-1]["laps"] += 1
                self.lap_start_time = phase_time_s
                self.current_lap_offtrack = False
                self.ers_lap_harvested_mj = 0.0
                # Extra recharge grant at control line if aid active & enabled
                self.aid_extra_recharge_mj = 0.5 if (self.aid_available and self.aid_active) else 0.0

        self._sync_world_pose()

    def _step_pit_lane(self, dt: float, phase_time_s: float) -> None:
        v_pit = self.geom.pit_speed_mps
        half_len = 0.5 * self.geom.pit_lane_length_m
        self.g_lat = 0.0
        self.g_long = 0.0
        self.vy = 0.0
        self.yaw_rate = 0.0
        self.front_sliding = False
        self.rear_sliding = False
        if self.pit_state == 1:
            self.vx = v_pit
            d = v_pit * dt
            self.pit_lane_progress += d
            prev_dist = self.distance
            self.distance += d
            self.s = (self.geom.pit_entry_s + self.pit_lane_progress) % self.geom.length
            self._check_pit_lap_cross(prev_dist, phase_time_s)
            if self.pit_lane_progress >= half_len:
                self.pit_state = 2
                self.pit_timer = 0.0
                self.vx = 0.0
        elif self.pit_state == 2:
            self.vx = 0.0
            self.pit_timer += dt
            if self.pit_timer >= 2.4:
                # Fresh tyres fitted! (72C surface, 68C core, 0 wear)
                new_c = self.pit_locked_compound or self.compound
                self.compound = new_c
                self.tyre_wear[:] = 0.0
                self.tyre_surf_c[:] = 72.0
                self.tyre_core_c[:] = 68.0
                self.stints.append({"compound": new_c, "laps": 0})
                self.pit_state = 3
        elif self.pit_state == 3:
            self.vx = v_pit
            d = v_pit * dt
            self.pit_lane_progress += d
            prev_dist = self.distance
            self.distance += d
            self.s = (self.geom.pit_entry_s + self.pit_lane_progress) % self.geom.length
            self._check_pit_lap_cross(prev_dist, phase_time_s)
            if self.pit_lane_progress >= self.geom.pit_lane_length_m:
                self.pit_state = 0
                self.pit_stops += 1
                self.s = self.geom.pit_exit_s
                # Keep distance congruent to pit_exit_s mod L (real engine tracks road s at rejoin)
                self.distance = self.lap * self.geom.length + self.s
                self.n = 0.0
                self.heading_error = 0.0
                self.prev_on_track = False
        self._sync_world_pose()

    def _check_pit_lap_cross(self, prev_dist: float, phase_time_s: float) -> None:
        if math.floor(self.distance / self.geom.length) > math.floor(prev_dist / self.geom.length):
            lap_dt = phase_time_s - self.lap_start_time
            self.lap_times.append((lap_dt, True))
            self.lap += 1
            if self.stints:
                self.stints[-1]["laps"] += 1
            self.lap_start_time = phase_time_s
            self.ers_lap_harvested_mj = 0.0
            self.aid_extra_recharge_mj = 0.0


def _crossed_s(prev_s: float, curr_s: float, target_s: float, length: float) -> bool:
    ds = (curr_s - prev_s) % length
    if ds > length * 0.5:
        return False
    dt = (target_s - prev_s) % length
    return 0.0 < dt <= ds


def _build_observation(
    car: CarState,
    all_cars: Sequence[CarState],
    phase: str,
    tick: int,
    phase_tick: int,
    decision: int,
    laps_total: int,
    position: int,
    weather: Dict[str, Any],
    forecast: Dict[str, Any],
) -> Dict[str, Any]:
    car.update_drive_force_vector()
    geom = car.geom
    rivals: List[Dict[str, Any]] = []
    if phase == "race":
        for other in all_cars:
            if other.seat == car.seat:
                continue
            gap_m = other.distance - car.distance
            # Also wrap check within +/- 400m along circuit
            if abs(gap_m) > 400.0:
                continue
            rivals.append(
                {
                    "seat": other.seat,
                    "gap_m": float(gap_m),
                    "dn": float(other.n - car.n),
                    "speed_mps": float(math.hypot(other.vx, other.vy)),
                    "lap": int(other.lap),
                    "status": other.status,
                }
            )

    speed_mps = math.hypot(car.vx, car.vy)
    return {
        "game_id": "racing",
        "seat": car.seat,
        "decision": decision,
        "tick": tick,
        "time": round(tick / 60.0, 6),
        "hold_ticks": 2,
        "laps_total": laps_total,
        "position": position,
        "self": {
            "aid_mode": "electrical_override_v1",
            "aid_extra_recharge_mj": float(car.aid_extra_recharge_mj),
            "ers_deployed_mj": float(car.ers_deployed_mj),
            "ers_harvested_mj": float(car.ers_harvested_mj),
            "aid_extra_deployed_mj": float(car.aid_extra_deployed_mj),
            "ers_power_kw": float(car.ers_power_kw),
            "aid_available": bool(car.aid_available),
            "aid_zone": car.aid_zone,
            "aid_active": bool(car.aid_active),
            "lap": int(car.lap),
            "s": float(car.s),
            "distance": float(car.distance),
            "n": float(car.n),
            "heading_error": float(car.heading_error),
            "speed_mps": float(speed_mps),
            "vx": float(car.vx),
            "vy": float(car.vy),
            "yaw_rate": float(car.yaw_rate),
            "steer": float(car.steer_rad),
            "gear": int(car.gear),
            "rpm": float(car.rpm),
            "throttle": float(car.last_action["throttle"]),
            "brake": float(car.last_action["brake"]),
            "ers_soc_mj": float(car.ers_soc_mj),
            "ers_mode": int(car.last_action["ers_mode"]),
            "aero_mode": int(1 if car.xmode_frac > 0.5 else 0),
            "override_available": bool(car.override_available),
            "override_active": bool(car.override_active),
            "tyre_temp_c": [float(x) for x in car.tyre_surf_c],
            "tyre_wear": [float(x) for x in car.tyre_wear],
            "fuel_kg": float(car.fuel_kg),
            "brake_temp_c": [float(x) for x in car.brake_temp_c],
            "pu_temp_c": float(car.pu_temp_c),
            "ride_height_mm": [float(x) for x in car.ride_height_mm],
            "g_lat": float(car.g_lat),
            "g_long": float(car.g_long),
            "compound": car.compound,
            "pit_state": int(car.pit_state),
            "pit_stops": int(car.pit_stops),
            "status": car.status,
            "x": float(car.x),
            "y": float(car.y),
            "z": float(car.z),
            "yaw": float(car.yaw),
            "drag_n": float(car.drag_n),
            "downforce_n": float(car.downforce_n),
            "effective_mass_kg": float(car.effective_mass()),
            "rolling_resistance_n": float(car.rolling_resistance_n),
            "drive_force_n": list(car.drive_force_n),
            "brake_force_max_n": 56000.0,
            "front_sliding": bool(car.front_sliding),
            "rear_sliding": bool(car.rear_sliding),
        },
        "preview": {
            "columns": ["distance_m", "curvature_1pm", "grade", "half_width_left_m", "half_width_right_m", "aero_zone"],
            "rows": geom.preview_rows(car.s),
        },
        "rivals": rivals,
        "track_length_m": float(geom.length),
        "tyres": {"allocation": dict(geom.compound_map)},
        "limits": {"recharge_mj": 8.5, "soc_window_mj": 4.0},
        "rules_version": "rsi-racing-4.4",
        "kernel_version": "rsi-formula-4.0",
        "phase": phase,
        "phase_tick": phase_tick,
        "weather": weather,
        "forecast": forecast,
    }


def select_sealed_design(design_envelope: Dict[str, Any], forecast: Dict[str, Any]) -> Dict[str, Any]:
    """Official rule: rain_design activates ONLY when forecast.rain_probability > 0.40."""
    if float(forecast.get("rain_probability", 0.0)) > 0.40 and "rain_design" in design_envelope:
        return design_envelope["rain_design"]
    return design_envelope["design"]


def run_weekend(
    track_id: str,
    entries: Sequence[Dict[str, Any]],
    weather_profile: WeatherProfile,
    cal: Optional[SimCalibration] = None,
    race_laps: int = 4,
    run_qualifying: bool = True,
    max_race_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Run a complete 1..10-car weekend (qualifying + 4-lap race) in FormulaSim-v4.4.

    Each entry in `entries` is a dict:
      {
        "name": str,
        "design_envelope": {"design": ..., "rain_design": ... (optional)},
        "act_fn": Callable[[dict], dict],  # already initialized via init(context)
      }
    """
    cal = cal or SimCalibration()
    geom = load_circuit(track_id, resample_ds=4.0)
    forecast = weather_profile.forecast()
    n_cars = len(entries)
    cars = [
        CarState(seat=i, design=select_sealed_design(e["design_envelope"], forecast), geom=geom, cal=cal)
        for i, e in enumerate(entries)
    ]
    act_fns = [e["act_fn"] for e in entries]
    faults = [0] * n_cars
    cpu_times = [0.0] * n_cars

    global_tick = 0
    decision_counts = [0] * n_cars
    qual_results: List[Tuple[bool, float, int]] = []  # (invalid_flag, lap_time, seat)

    import time as _time

    # ---- PHASE 1: QUALIFYING (1 timed lap per seat) ----
    if run_qualifying:
        q_max_ticks = int(math.ceil(60.0 * (geom.length / 55.0 + 120.0)))
        for seat in range(n_cars):
            car = cars[seat]
            car.reset_for_phase("qualifying", grid_slot=seat)
            for pt in range(q_max_ticks):
                w = weather_profile.at("qualifying", pt / 60.0)
                if pt % 2 == 0:
                    obs = _build_observation(car, cars, "qualifying", global_tick, pt, decision_counts[seat], 1, 1, w, forecast)
                    t0 = _time.process_time()
                    try:
                        raw_act = act_fns[seat](obs)
                    except Exception as exc:
                        faults[seat] += 1
                        car.status = "retired"
                        car.fault_reason = f"exception:{type(exc).__name__}"
                        break
                    cpu_times[seat] += _time.process_time() - t0
                    decision_counts[seat] += 1
                    ok, err, parsed = validate_action(raw_act)
                    if not ok:
                        faults[seat] += 1
                        car.status = "retired"
                        car.fault_reason = f"invalid_action:{err}"
                        break
                    car.last_action = parsed
                # 2 physics substeps per 60 Hz tick
                for _ in range(2):
                    car.step_sub(1.0 / 120.0, "qualifying", pt / 60.0, w, 0.0, 0.0)
                global_tick += 1
                if car.status != "racing" or car.lap >= 1:
                    break
            if car.lap >= 1 and car.lap_times:
                lt, valid = car.lap_times[0]
                qual_results.append((not valid, lt, seat))
            else:
                qual_results.append((True, 9999.0, seat))
        qual_results.sort(key=lambda x: (x[0], x[1], x[2]))
        grid_order = [seat for _, _, seat in qual_results]
    else:
        grid_order = list(range(n_cars))
        qual_results = [(False, 90.0 + i * 0.1, i) for i in range(n_cars)]

    qual_pos_by_seat = {seat: idx + 1 for idx, seat in enumerate(grid_order)}
    qual_time_by_seat = {seat: (lt if not inv else None) for inv, lt, seat in qual_results}

    # ---- PHASE 2: RACE (4-lap standing start, all cars simultaneous) ----
    # Official race absolute time limit: T = 7200*4/(floor(305000/L)+1), max T+120s
    t_scale = 7200.0 * race_laps / (math.floor(305000.0 / geom.length) + 1)
    race_limit_s = min(t_scale + 120.0, max_race_seconds) if max_race_seconds else (t_scale + 120.0)
    r_max_ticks = int(math.ceil(60.0 * race_limit_s))

    for slot_idx, seat in enumerate(grid_order):
        cars[seat].reset_for_phase("race", grid_slot=slot_idx)

    # Track detection line crossing times for 1.0s Manual Overtake eligibility
    last_det_cross_time: Dict[int, float] = {}
    det_cross_history: List[Tuple[float, int]] = []  # (time_s, seat)

    for pt in range(r_max_ticks):
        phase_s = pt / 60.0
        w = weather_profile.at("race", phase_s)
        wet = float(w["track_wetness"])

        # Compute current live running positions
        order = sorted(range(n_cars), key=lambda i: (-cars[i].lap, -cars[i].distance, i))
        pos_by_seat = {seat: rank + 1 for rank, seat in enumerate(order)}

        if pt % 2 == 0:
            # Freeze observations for all racing cars simultaneously before applying actions
            obs_list: List[Optional[Dict[str, Any]]] = [None] * n_cars
            for i in range(n_cars):
                if cars[i].status == "racing":
                    obs_list[i] = _build_observation(
                        cars[i], cars, "race", global_tick, pt, decision_counts[i], race_laps, pos_by_seat[i], w, forecast
                    )
            for i in range(n_cars):
                if obs_list[i] is None:
                    continue
                t0 = _time.process_time()
                try:
                    raw_act = act_fns[i](obs_list[i])
                except Exception as exc:
                    faults[i] += 1
                    cars[i].status = "retired"
                    cars[i].fault_reason = f"exception:{type(exc).__name__}"
                    continue
                cpu_times[i] += _time.process_time() - t0
                decision_counts[i] += 1
                ok, err, parsed = validate_action(raw_act)
                if not ok:
                    faults[i] += 1
                    cars[i].status = "retired"
                    cars[i].fault_reason = f"invalid_action:{err}"
                    continue
                cars[i].last_action = parsed
                # G11/G12 pit request & compound target update
                cars[i].pit_request = bool(parsed["pit_request"])
                if parsed["pit_compound"] is not None:
                    cars[i].pit_compound_target = parsed["pit_compound"]

        # Slipstream & dirty-air wake from ahead cars
        slip_cuts = [0.0] * n_cars
        wake_losses = [0.0] * n_cars
        for i in range(n_cars):
            ci = cars[i]
            if ci.status != "racing" or ci.pit_state > 0:
                continue
            for j in range(n_cars):
                if i == j or cars[j].status != "racing" or cars[j].pit_state > 0:
                    continue
                gap = cars[j].distance - ci.distance
                if 2.0 < gap < 45.0:
                    dn = abs(cars[j].n - ci.n)
                    lat_f = math.exp(-0.5 * (dn / 1.3) ** 2)
                    slip_cuts[i] = max(slip_cuts[i], 0.30 * math.exp(-gap / 15.0) * lat_f)
                    wake_losses[i] = max(wake_losses[i], min(0.40, 0.40 * math.exp(-gap / 14.43) * lat_f))

        # 2 physics substeps per 60 Hz tick
        prev_s_list = [c.s for c in cars]
        for _ in range(2):
            for i in range(n_cars):
                cars[i].step_sub(1.0 / 120.0, "race", phase_s, w, slip_cuts[i], wake_losses[i])

        # Manual Overtake Mode detection & activation lines
        for i in range(n_cars):
            ci = cars[i]
            if ci.status != "racing":
                continue
            if wet >= 0.5 or ci.pit_state > 0:
                ci.aid_available = False
                ci.aid_pending = False
                ci.override_available = False
                ci.aid_zone = None
                continue
            if ci.distance > 0.0 and _crossed_s(prev_s_list[i], ci.s, geom.aid_detection_s, geom.length):
                # Check if any car crossed detection within <= 1.0s earlier
                eligible = any(0.0 < (phase_s - t_prev) <= 1.0 and s_prev != i for t_prev, s_prev in det_cross_history[-15:])
                ci.aid_pending = eligible
                last_det_cross_time[i] = phase_s
                det_cross_history.append((phase_s, i))
            if ci.distance > 0.0 and _crossed_s(prev_s_list[i], ci.s, geom.aid_activation_s, geom.length):
                ci.aid_available = ci.aid_pending
                ci.override_available = ci.aid_pending
                ci.aid_zone = 1 if ci.aid_pending else None
                ci.aid_pending = False

        # Car-to-car contact separation (prevent ghost overlapping)
        for i in range(n_cars):
            ci = cars[i]
            if ci.status != "racing" or ci.pit_state > 0:
                continue
            for j in range(i + 1, n_cars):
                cj = cars[j]
                if cj.status != "racing" or cj.pit_state > 0:
                    continue
                ds = abs(ci.distance - cj.distance)
                dn = abs(ci.n - cj.n)
                if ds < 5.0 and dn < 1.9:
                    ci.collision_count += 1
                    cj.collision_count += 1
                    # Speed penalty & gentle lateral push
                    ci.vx *= 0.88
                    cj.vx *= 0.88
                    push = 0.25 if ci.n >= cj.n else -0.25
                    ci.n += push
                    cj.n -= push

        # Check finishes & 30s stall rule
        all_done = True
        for i in range(n_cars):
            ci = cars[i]
            if ci.status == "racing":
                if ci.lap >= race_laps:
                    ci.status = "finished"
                    ci.finish_time_s = phase_s
                else:
                    all_done = False
                    if phase_s > 10.0 and ci.pit_state == 0:
                        ci.stall_window_dist.append((phase_s, ci.distance))
                        while ci.stall_window_dist and phase_s - ci.stall_window_dist[0][0] > 30.0:
                            ci.stall_window_dist.pop(0)
                        if ci.stall_window_dist and phase_s - ci.stall_window_dist[0][0] >= 29.5:
                            if ci.distance - ci.stall_window_dist[0][1] < 10.0:
                                ci.status = "retired"
                                ci.retire_reason = "stalled"
                    else:
                        ci.stall_window_dist.clear()

        global_tick += 1
        if all_done:
            break

    # Final classification (official G13 dry 2-compound check + laps + finish time)
    classification = []
    for i, car in enumerate(cars):
        st = car.status
        dsq = False
        if st == "finished" and not car.wet_waiver and len(car.compounds_used) < 2:
            st = "disqualified"
            dsq = True
        fl = min((lt for lt, ok in car.lap_times if ok), default=None)
        tot = car.finish_time_s if car.finish_time_s is not None else (
            sum(lt for lt, _ in car.lap_times) if car.lap_times else race_limit_s
        )
        classification.append(
            {
                "seat": i,
                "name": entries[i]["name"],
                "status": st,
                "disqualified": dsq,
                "faults": faults[i],
                "fault_reason": car.fault_reason,
                "retire_reason": car.retire_reason,
                "laps": car.lap,
                "distance_m": round(car.distance, 2),
                "total_time_s": round(tot, 3),
                "qualifying_position": qual_pos_by_seat[i],
                "qualifying_time_s": round(qual_time_by_seat[i], 3) if qual_time_by_seat[i] is not None else None,
                "fastest_lap_s": round(fl, 3) if fl is not None else None,
                "pit_stops": car.pit_stops,
                "compounds_used": sorted(car.compounds_used),
                "wet_waiver": car.wet_waiver,
                "stints": car.stints,
                "end_fuel_kg": round(car.fuel_kg, 3),
                "end_ers_soc_mj": round(car.ers_soc_mj, 3),
                "max_tyre_wear": round(float(np.max(car.tyre_wear)), 4),
                "offtrack_ticks": car.offtrack_ticks,
                "collisions": car.collision_count,
                "cpu_seconds": round(cpu_times[i], 4),
            }
        )

    # Sort per sporting rules: non-DSQ before DSQ, more laps first, earlier total_time_s, seat tiebreak
    classification.sort(key=lambda r: (r["disqualified"], -r["laps"], r["total_time_s"], r["seat"]))
    for pos_idx, row in enumerate(classification):
        row["position"] = pos_idx + 1

    return {
        "track_id": track_id,
        "weather": weather_profile.forecast(),
        "race_laps": race_laps,
        "classification": classification,
    }
