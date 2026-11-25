"""Experimental FormulaSim-v5/v3 CPU four-wheel weekend simulator for rsi-racing-4.4.

The indexed public parameter table is not the server source. See the timestamped
validation record before using this model for control decisions.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path
import time as _time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from circuit_geom import CircuitGeometry, load_circuit
from formula_sim_v5_gpu import disc_friction, wheel_dynamics, wheel_thermal
from formula_sim import (
    COMPOUND_SPECS,
    MGUK_SPEED_KPH,
    MGUK_NORMAL_W,
    MGUK_OVERRIDE_W,
    WHEEL_EFFICIENCY,
    WeatherProfile,
    standard_weather_scenarios,
    validate_action,
    select_sealed_design,
    effective_mass_for_gear,
    _crossed_s,
)

DEFAULT_V5_PARAMS_PATH = Path(__file__).resolve().parent / "configs" / "sim_v5_params_v3.json"


def air_rho_from_alt_temp(z_m: float, air_temp_c: float) -> float:
    """ISA barometric + thermal air density referenced to 1.200 kg/m^3 at 130m / 20C."""
    p_rel = math.exp(-(float(z_m) - 130.0) / 8434.5)
    t_rel = 293.15 / max(250.0, float(air_temp_c) + 273.15)
    return 1.200 * p_rel * t_rel


@dataclass
class SimV5ParamsCPU:
    """Exact CPU mirror of `SimV5Params` (work/racing-distill/formula_sim_v5_gpu.py)."""
    # Aero downforce & balance & rake stall
    fw_cza_k: float = 0.582
    rw_cza_k: float = 0.747
    floor_cza_k: float = 0.960
    body_cza: float = 0.100
    fw_x_retain: float = 0.300
    rw_x_retain: float = 0.130
    downforce_scale: float = 1.000
    rake_ref_mm: float = 34.0
    rake_gain_per_mm: float = 0.0040
    rake_stall_mm: float = 90.0
    rake_stall_slope: float = 0.015
    floor_f_share_base: float = 0.420
    floor_f_share_rake_k: float = -0.0015
    rh_f_stall_mm: float = 18.0
    # Aero drag
    cda_base: float = 0.478
    fw_ind_k: float = 0.140
    rw_ind_k: float = 0.390
    floor_ind_k: float = 0.040
    rad_cda_k: float = 0.145
    drag_scale: float = 1.000
    roll_res_c: float = 0.0120
    # Ride height dynamics
    k_rh_f_scale: float = 1.000
    k_rh_r_scale: float = 1.000
    pitch_rh_scale: float = 1.000
    # Tyre Pacejka & grip & post-peak drop-off
    mu_f_base: float = 2.520
    mu_r_base: float = 2.520
    load_sens_f: float = 0.260
    load_sens_r: float = 0.200
    stiff_f_scale: float = 1.000
    stiff_r_scale: float = 1.000
    cy_lat: float = 1.350
    ey_lat: float = -0.500
    post_peak_drop_f: float = 0.020
    post_peak_drop_r: float = 0.060
    wheelspin_drop: float = 0.070
    lockup_drop: float = 0.120
    rear_diff_lock: float = 0.250
    grip_scale: float = 1.000
    # Continuous nonlinear wet grip + hydroplaning
    wet_grip_drop: float = 0.400
    wet_grip_exp: float = 0.850
    wet_hydro_v_k: float = 0.0050
    # Off-track grass/gravel penalty
    offtrack_grip_mult: float = 0.720
    offtrack_drag_g: float = 0.460
    # Powertrain, engine braking, braking & inertia
    ice_power_scale: float = 0.985
    engine_brake_g: float = 0.110
    lift_regen_scale: float = 0.300
    fuel_burn_scale: float = 1.025
    brake_max_n: float = 56000.0
    brake_scale: float = 1.000
    iz_scale: float = 1.000
    # Thermal & wear
    heat_scale: float = 1.120
    wet_cool_k: float = 35.0
    wear_scale: float = 1.000
    # Slipstream & wake
    slipstream_max: float = 0.300
    dirty_air_max: float = 0.400

    def to_dict(self) -> Dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SimV5ParamsCPU":
        valid = {k: float(v) for k, v in d.items() if hasattr(cls, k)}
        return cls(**valid)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "SimV5ParamsCPU":
        p = Path(path) if path is not None else DEFAULT_V5_PARAMS_PATH
        if p.is_file():
            return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))
        return cls()


def compute_aero_forces_v5(
    v: float,
    fw: float,
    rw: float,
    rh_f_mm: float,
    rh_r_mm: float,
    rad: float,
    xmode_frac: float,
    params: SimV5ParamsCPU,
    duct: float = 0.45,
    rho: float = 1.2,
) -> Tuple[float, float, float]:
    """Exact scalar mirror of `SimV5PhysicsModule.aero_forces` -> (downforce_n, drag_n, front_share)."""
    fw_z = params.fw_cza_k * (1.0 - (1.0 - min(1.0, max(0.0, fw / 32.0))) ** 2)
    rw_z = params.rw_cza_k * (1.0 - (1.0 - min(1.0, max(0.0, rw / 34.0))) ** 2)
    fw_cza = fw_z * (1.0 - (1.0 - params.fw_x_retain) * xmode_frac)
    rw_cza = rw_z * (1.0 - (1.0 - params.rw_x_retain) * xmode_frac)

    rh_fc = max(3.0, rh_f_mm)
    rh_rc = max(10.0, rh_r_mm)
    rake_mm = max(0.0, rh_rc - rh_fc)

    floor_gain = 1.0 + 0.01 * (28.0 - rh_fc) + params.rake_gain_per_mm * (rake_mm - params.rake_ref_mm)
    floor_gain = min(1.40, max(0.40, floor_gain))

    stall_ratio = min(1.0, rh_fc / params.rh_f_stall_mm)
    floor_gain *= stall_ratio ** 2.5

    excess_rake = max(0.0, rake_mm - params.rake_stall_mm)
    floor_gain *= min(1.0, max(0.35, 1.0 - params.rake_stall_slope * excess_rake))

    floor_cza = params.floor_cza_k * floor_gain
    body_cza = params.body_cza
    cza_raw = fw_cza + rw_cza + floor_cza + body_cza
    cza_tot = cza_raw * params.downforce_scale

    floor_f_sh = min(
        0.62,
        max(0.25, params.floor_f_share_base + params.floor_f_share_rake_k * (rake_mm - params.rake_ref_mm)),
    )
    front_cza = fw_cza + floor_cza * floor_f_sh + 0.45 * body_cza
    front_share = front_cza / max(cza_raw, 1e-4)

    fw_prof = 0.040 * (1.0 - xmode_frac) + 0.035 * xmode_frac
    rw_prof = 0.050 * (1.0 - xmode_frac) + 0.035 * xmode_frac
    fw_ind = params.fw_ind_k * fw_cza * fw_cza
    rw_ind = params.rw_ind_k * rw_cza * rw_cza
    floor_ind = params.floor_ind_k * ((floor_cza / 0.96) ** 2)
    cool_cda = params.rad_cda_k * (max(0.1, rad) ** 1.25)
    duct_cda = 0.020 * duct
    cda_tot = (
        params.cda_base + fw_prof + rw_prof + fw_ind + rw_ind + floor_ind + cool_cda + duct_cda
    ) * params.drag_scale

    q_dyn = 0.5 * rho * v * v
    return q_dyn * cza_tot, q_dyn * cda_tot, front_share


def compute_mguk_power_v5(v_mps: float, override_active: bool) -> float:
    kph = v_mps * 3.6
    if override_active:
        if kph <= 337.5:
            return 350000.0
        if kph <= 340.0:
            return 350000.0 - (kph - 337.5) * 20000.0
        if kph <= 345.0:
            return 300000.0 - (kph - 340.0) * 20000.0
        return max(0.0, 200000.0 - (kph - 345.0) * 20000.0)
    if kph <= 290.0:
        return 350000.0
    if kph <= 337.5:
        return 350000.0 - (kph - 290.0) * (237500.0 / 47.5)
    if kph <= 340.0:
        return 112500.0 - (kph - 337.5) * 5000.0
    return max(0.0, 100000.0 - (kph - 340.0) * 20000.0)


def tyre_mu_v5(
    axle: str,
    fz_wheel: float,
    temp_c: float,
    wear: float,
    wet: float,
    v_mps: float,
    comp_grip: float,
    comp_t_opt: float,
    params: SimV5ParamsCPU,
    pressure_psi: Optional[float] = None,
    camber_deg: Optional[float] = None,
) -> float:
    """Legacy standalone mu helper. pressure_psi, if passed, is operating (hot) pressure."""
    if axle == "front":
        fz0, mu0, sens, g_opt = 3800.0, params.mu_f_base, params.load_sens_f, -3.3
    else:
        fz0, mu0, sens, g_opt = 5200.0, params.mu_r_base, params.load_sens_r, -2.4

    load_ratio = max(0.15, fz_wheel / fz0)
    mu = mu0 * (load_ratio ** (-sens)) * comp_grip

    w_t = 45.0 if temp_c < comp_t_opt else 40.0
    therm = min(1.0, max(0.45, 1.0 - 0.5 * ((temp_c - comp_t_opt) / w_t) ** 2))
    mu *= therm

    wc = min(1.0, max(0.0, wear))
    cliff = max(0.0, wc - 0.80) / 0.20
    w_fac = min(1.0, max(0.45, 1.0 - 0.05 * wc - 0.40 * cliff * cliff))
    mu *= w_fac

    if pressure_psi is not None:
        mu *= max(0.85, 1.0 - 0.01 * (pressure_psi - 29.0))
    if camber_deg is not None:
        mu *= max(0.85, 1.0 - 0.0035 * ((camber_deg - g_opt) ** 2))

    wetc = min(1.0, max(0.0, wet))
    wet_term = params.wet_grip_drop * ((wetc + 1e-6) ** params.wet_grip_exp) if wetc > 1e-4 else 0.0
    wet_fac = min(1.0, max(0.30, 1.0 - wet_term - params.wet_hydro_v_k * wetc * ((v_mps / 80.0) ** 2)))
    return mu * wet_fac * params.grip_scale


def heave_axle(dt, stat, prev, u, k, m, force, damping_ratio, static_weight):
    """Implicit spring/damper with unilateral plank support. Heights mm; mass kg."""
    x = stat - prev
    m_mm = m * 0.001
    c = 2.0 * damping_ratio * math.sqrt(m_mm * k)
    numerator = m_mm * u + dt * (force - k * x)
    denominator = m_mm + dt * c + dt * dt * k
    u1 = numerator / denominator
    limit = stat - 3.0
    if x + dt * u1 > limit:
        # 4e6 N/m acts in parallel with the spring only after contact.
        u1 = (numerator - dt * 4000.0 * (x - limit)) / (denominator + dt * dt * 4000.0)
    x1 = x + dt * u1
    return stat - x1, u1, max(0.0, static_weight + k * x1 + c * u1), 4000.0 * max(0.0, x1 - limit)


def _disc_capacity(params, bias_f, disc_t):
    half_cap = 0.5 * params.brake_max_n * params.brake_scale
    return tuple(half_cap * (bias_f if i < 2 else 1.0 - bias_f) * disc_friction(t)
                 for i, t in enumerate(disc_t))


def _wheel_requests(disc_cap, brake, drive, regen_brk, regen_other, eng_brk):
    """Rear brake-by-wire substitutes regen for part of the requested torque, not in addition."""
    requested = tuple(brake * cap for cap in disc_cap)
    rear_sum = max(1e-12, requested[2] + requested[3])
    hydraulic = (requested[0], requested[1],
                 max(0.0, requested[2] * (1.0 - regen_brk / rear_sum)),
                 max(0.0, requested[3] * (1.0 - regen_brk / rear_sum)))
    net_rear = 0.5 * (drive - regen_other - eng_brk)
    return (-requested[0], -requested[1], net_rear - requested[2], net_rear - requested[3]), hydraulic


def step_physics_v5_scalar(
    params: SimV5ParamsCPU,
    *,
    vx: float,
    vy: float,
    yaw_rate: float,
    steer_rad: float,
    rh_f_prev: float,
    rh_r_prev: float,
    fuel: float,
    temp_f: float,
    temp_r: float,
    wear: float,
    ers_soc: float,
    throttle: float,
    brake: float,
    ers_mode: float,
    xmode: float,
    override: float,
    grade: float,
    a_vert: float,
    wet: float,
    off_track: float,
    fw: float,
    rw: float,
    rh_f_stat: float,
    rh_r_stat: float,
    ballast: float,
    rad: float,
    wf: float,
    sf: float,
    sr: float,
    bias_f: float,
    comp_grip: float,
    comp_t_opt: float,
    dt: float = 1.0 / 24.0,
    slipstream_cut: float = 0.0,
    wake_loss: float = 0.0,
    prev_glong: float = 0.0,
    rho: float = 1.2,
    duct: float = 0.45,
    harvest_level: float = 5.0,
    rpm: Optional[float] = None,
    total_ratio: Optional[float] = None,
    launch_active: bool = False,
    prev_glat: float = 0.0,
    transfer_f: float = 0.0,
    transfer_r: float = 0.0,
    heave_u_f: float = 0.0,
    heave_u_r: float = 0.0,
    arb_front: float = 5.0,
    arb_rear: float = 5.0,
    damping_ratio: float = 0.5,
    pressure_front_psi: float = 29.0,
    pressure_rear_psi: float = 28.0,
    camber_front_deg: float = -2.5,
    camber_rear_deg: float = -1.5,
    tyre_surf_c: Optional[Sequence[float]] = None,
    tyre_core_c: Optional[Sequence[float]] = None,
    tyre_wear: Optional[Sequence[float]] = None,
    brake_temp_c: Sequence[float] = (500.0,) * 4,
    alpha_eff: Sequence[float] = (0.0,) * 4,
    comp_heat: float = 1.0,
    comp_wear: float = 1.0,
    track_temp_c: float = 30.0,
    air_temp_c: float = 24.0,
) -> Dict[str, Any]:
    """Single force/integration substep, without weekend gearbox, actuator or sporting state.

    Supply total_ratio for exact reflected inertia (rpm/speed inference is invalid
    during clutch slip). Supply the four-wheel next states on subsequent calls.
    """
    mass = 748.0 + ballast + max(0.0, fuel)
    v = math.sqrt(vx * vx + vy * vy + 1e-6)

    df0, _, f_sh0 = compute_aero_forces_v5(v, fw, rw, rh_f_prev, rh_r_prev, rad, xmode, params, duct=duct, rho=rho)
    df0 *= 1.0 - wake_loss
    k_f = 2.0 * (sf * 380.0) / max(sf + 380.0, 1.0) * params.k_rh_f_scale
    k_r = 2.0 * (sr * 450.0) / max(sr + 450.0, 1.0) * params.k_rh_r_scale
    d_fz_pitch = mass * (prev_glong * 9.80665) * 0.272 / 3.4 * params.pitch_rh_scale
    extra_f = df0 * f_sh0 + mass * a_vert * wf - d_fz_pitch
    extra_r = df0 * (1.0 - f_sh0) + mass * a_vert * (1.0 - wf) + d_fz_pitch
    rh_f, uf, fz_f_ax, plank_f = heave_axle(dt, rh_f_stat, rh_f_prev, heave_u_f, k_f,
        mass * wf - 48.0, extra_f, damping_ratio, mass * 9.80665 * wf)
    rh_r, ur, fz_r_ax, plank_r = heave_axle(dt, rh_r_stat, rh_r_prev, heave_u_r, k_r,
        mass * (1.0 - wf) - 52.0, extra_r, damping_ratio, mass * 9.80665 * (1.0 - wf))
    df, drag, f_share = compute_aero_forces_v5(v, fw, rw, rh_f, rh_r, rad, xmode, params, duct=duct, rho=rho)
    df *= 1.0 - wake_loss
    drag *= 1.0 - slipstream_cut
    roll = params.roll_res_c * (fz_f_ax + fz_r_ax)

    if rpm is not None:
        xr = (rpm - 12500.0) / 3500.0
        eta_ice = max(0.25, 0.486 * (1.0 - 0.034 * xr * xr - 0.14 * (max(0.0, -xr) ** 2)))
    else:
        eta_ice = 0.482
    ice_w = (3000e6 / 3600.0) * eta_ice * params.ice_power_scale * throttle * 0.965 * (1.0 if fuel > 0.0 else 0.0)

    max_m = compute_mguk_power_v5(vx, override > 0.5)
    hl = min(5.0, max(1.0, harvest_level))
    regen_cap = 70000.0 * hl
    is_brk = 1.0 if brake > 0.05 else 0.0
    is_m3 = 1.0 if ers_mode >= 2.5 else 0.0
    is_m2 = 1.0 if (1.5 <= ers_mode < 2.5) else 0.0
    is_m0 = 1.0 if ers_mode < 0.5 else 0.0
    is_thr = 1.0 if throttle > 0.05 else 0.0
    has_soc = 1.0 if ers_soc > 0.01 else 0.0
    can_regen = 1.0 if (vx > 10.0 and ers_soc < 3.98) else 0.0

    deploy_frac = min(1.0, max(0.0, is_m3 + 0.5 * is_m2 + override * (1.0 - is_m3) * is_m2))
    ers_deploy_w = max_m * deploy_frac * throttle * WHEEL_EFFICIENCY * has_soc * (1.0 - is_brk) * is_thr

    disc_cap = _disc_capacity(params, bias_f, brake_temp_c)
    regen_brk = min(regen_cap, brake * (disc_cap[2] + disc_cap[3]) * vx) * is_brk
    is_m0_harv = (1.0 - is_brk) * is_m0 * is_thr
    regen_m0_thr = regen_cap * throttle * is_m0_harv
    is_coast = (1.0 - is_brk) * (1.0 - is_thr)
    regen_coast = min(regen_cap, mass * 0.04 * hl * 9.80665 * vx * params.lift_regen_scale) * is_coast
    regen_w = (regen_brk + regen_m0_thr + regen_coast) * can_regen
    ers_w = (ers_deploy_w / WHEEL_EFFICIENCY) - regen_w

    f_raw = (ice_w + ers_deploy_w - regen_m0_thr * can_regen) / max(vx, 7.0)
    drive = max(0.0, f_raw)
    # No gear is available in the legacy scalar API; when omitted infer a ratio
    # only from an engaged shaft or use the neutral-equivalent inertia.
    if total_ratio is None:
        total_ratio = (rpm * math.pi * 0.343 / (30.0 * vx)) if rpm is not None and vx > 7.0 else 0.0
    eff_m = effective_mass_for_gear(748.0 + ballast, max(0.0, fuel), total_ratio)
    if launch_active and vx < 30.0:
        drive = min(drive, eff_m * 1.2 * 9.80665)
    eng_brk = mass * 9.80665 * params.engine_brake_g * min(1.0, max(0.0, 1.0 - throttle / 0.15)) if vx > 8.0 else 0.0
    request, hydraulic = _wheel_requests(disc_cap, brake, drive, regen_brk * can_regen / max(vx, 7.0),
                                         regen_coast * can_regen / max(vx, 7.0) + max(0.0, -f_raw), eng_brk)
    surface = (temp_f, temp_f, temp_r, temp_r) if tyre_surf_c is None else tyre_surf_c
    core = surface if tyre_core_c is None else tyre_core_c
    wear4 = (wear,) * 4 if tyre_wear is None else tyre_wear
    wheels = wheel_dynamics(params, mass=mass, wf=wf, vx=vx, vy=vy, yaw_rate=yaw_rate, steer_rad=steer_rad,
        dt=dt, prev_glat=prev_glat, transfer_f=transfer_f, transfer_r=transfer_r, fz_f_ax=fz_f_ax, fz_r_ax=fz_r_ax,
        sf=sf, sr=sr, arb_front=arb_front, arb_rear=arb_rear, damping_ratio=damping_ratio,
        camber_front_deg=camber_front_deg, camber_rear_deg=camber_rear_deg, pressure_front_psi=pressure_front_psi,
        pressure_rear_psi=pressure_rear_psi, tyre_surf_c=surface, tyre_core_c=core, tyre_wear=wear4,
        alpha_eff=alpha_eff, wet=wet, off_track=off_track, comp_grip=comp_grip, comp_t_opt=comp_t_opt,
        fx_request=request, hydraulic_request=hydraulic,
        harvest_level=hl if (regen_brk + regen_coast) * can_regen > 0.0 else 0.0)
    thermal = wheel_thermal(params, wheels=wheels, tyre_surf_c=surface, tyre_core_c=core, tyre_wear=wear4,
        brake_temp_c=brake_temp_c, dt=dt, v=v, wet=wet, air_temp_c=air_temp_c, track_temp_c=track_temp_c,
        duct=duct, comp_heat=comp_heat, comp_wear=comp_wear, comp_t_opt=comp_t_opt)
    ot_drag = mass * 9.80665 * params.offtrack_drag_g if off_track > 0.5 and vx > 6.0 else 0.0
    ax_b = (wheels["fx_body"] - drag - roll - ot_drag - mass * 9.80665 * grade) / eff_m
    ay_b = wheels["fy_body"] / mass
    vx_next = max(0.0, vx + (ax_b + yaw_rate * vy) * dt)
    vy_dyn = vy + (ay_b - yaw_rate * vx) * dt
    if math.hypot(vx_next, vy_dyn) < 5.0:
        vy_next, r_next = 0.0, vx_next * math.tan(steer_rad) / 3.4
    else:
        vy_next = vy_dyn
        r_next = yaw_rate + wheels["mz_body"] * dt / (mass * 1.15 ** 2 * params.iz_scale)

    burn_kg_s = (3000e6 / 3600.0 / 44.0e6) * throttle * params.fuel_burn_scale
    fuel_next = max(0.0, fuel - burn_kg_s * dt)
    ers_soc_next = min(4.0, max(0.0, ers_soc - (ers_deploy_w / WHEEL_EFFICIENCY - regen_w * 0.97) * dt / 1e6))
    return {
        "downforce_n": df,
        "drag_n": drag,
        "rh_f": rh_f,
        "rh_r": rh_r,
        "g_lat": ay_b / 9.80665,
        "g_long": ax_b / 9.80665,
        "ers_power_kw": ers_w / 1000.0,
        "vx_next": vx_next,
        "vy_next": vy_next,
        "yaw_rate_next": r_next,
        "fuel_next": fuel_next,
        "ers_soc_next": ers_soc_next,
        "front_share": f_share, "heave_u_f_next": uf, "heave_u_r_next": ur,
        "plank_n": (plank_f, plank_r), "brake_force_max_n": sum(disc_cap),
        **wheels, **thermal,
    }


class CarStateV5:
    """Single car physics & sporting state powered by experimental four-wheel v3 physics."""

    def __init__(
        self,
        seat: int,
        design: Dict[str, Any],
        geom: CircuitGeometry,
        params: SimV5ParamsCPU,
    ):
        self.seat = seat
        self.design = design
        self.geom = geom
        self.params = params
        self.z_mean = float(np.mean(geom.z))
        self.dry_mass = 748.0 + float(design["mass"]["ballast_kg"])
        self.wf = float(design["mass"]["weight_front_pct"]) / 100.0
        self.ratios = [float(r) for r in design["gearbox"]["ratios"]]
        self.final_drive = float(design["gearbox"]["final_drive"])
        self.ers_harvest_level = int(design["power_unit"]["ers_harvest_level"])
        sus = design["suspension"]
        self.damping_ratio = {"soft": 0.3, "medium": 0.5, "stiff": 0.75}[sus["damping"]]
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

        self.heave_u_f = self.heave_u_r = 0.0
        self.transfer_f = self.transfer_r = 0.0
        self.alpha_eff = (0.0,) * 4
        self.launch_active = phase == "race"
        # Offline diagnostics only; no extra simulator-only input is given to agents.
        self.plank_n = (0.0, 0.0)
        self.wheel_fz = self.wheel_fx = self.wheel_fy = self.wheel_util = (0.0,) * 4
        self.hydraulic_force_n = (0.0,) * 4
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

        self.compounds_used: set[str] = set()
        self.wet_waiver = False
        self.prev_on_track = False
        self.prev_on_track_dist = 0.0
        self.current_lap_offtrack = False
        self.lap_start_time = 0.0
        self.lap_times: List[Tuple[float, bool]] = []
        self.finish_time_s: Optional[float] = None
        self.stints: List[Dict[str, Any]] = [{"compound": self.compound, "laps": 0}]
        self.stall_window_dist: List[Tuple[float, float]] = []
        self.offtrack_ticks = 0
        self.collision_count = 0
        self.min_rh_f_mm = float(self.ride_height_mm[0])
        self.max_pu_temp_c = self.pu_temp_c
        self.front_slide_ticks = 0
        self.rear_slide_ticks = 0
        self.spinout_ticks = 0

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

    def _ice_power_w(self, rpm: float) -> float:
        x = (rpm - 12500.0) / 3500.0
        eta = max(0.22, 0.486 * (1.0 - 0.034 * x * x - 0.14 * max(0.0, -x) ** 2))
        if rpm > 14800.0:
            eta *= max(0.2, 1.0 - (rpm - 14800.0) / 1500.0)
        p = (3000e6 / 3600.0) * eta * self.params.ice_power_scale
        if self.pu_temp_c > 108.0:
            p *= 1.0 - min(0.35, 0.03 * (self.pu_temp_c - 108.0))
        return max(0.0, p)

    def update_drive_force_vector(self) -> None:
        v = max(self.vx, 1.0)
        tot_ratio = self.ratios[self.gear - 1] * self.final_drive
        rpm = max(7500.0, min(15500.0, (v / 0.343) * tot_ratio * 60.0 / (2.0 * math.pi)))
        ice_w = self._ice_power_w(rpm) * 0.965 if self.fuel_kg > 0.0 else 0.0
        f_ice = ice_w / max(v, 8.0)
        mguk_norm = compute_mguk_power_v5(v, override_active=False) * WHEEL_EFFICIENCY
        f_m_full = (mguk_norm / max(v, 8.0)) if self.ers_soc_mj > 0.01 else 0.0
        regen_w = 70000.0 * self.ers_harvest_level * 0.965
        f_harvest = max(0.0, f_ice - regen_w / max(v, 10.0))
        self.drive_force_n = [
            float(f_harvest),
            float(f_ice),
            float(f_ice + 0.5 * f_m_full),
            float(f_ice + f_m_full),
        ]
        if self.launch_active and self.vx < 30.0:
            cap = self.effective_mass() * 1.2 * 9.80665
            self.drive_force_n = [min(fx, cap) for fx in self.drive_force_n]

    def _disc_capacity(self):
        return _disc_capacity(self.params, self.design["brakes"]["bias_front_pct"] * 0.01, self.brake_temp_c)

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

        if dt <= 0.0 or not math.isfinite(dt):
            raise ValueError("dt must be finite and positive")
        # Public records can be 2 or 3 engine ticks apart; the tyre/rigid-body
        # integration is defined at <=1/120s, not at the replay sample rate.
        if dt > 1.0 / 120.0 + 1e-12:
            steps = math.ceil(dt * 120.0 - 1e-12)
            h = dt / steps
            for i in range(steps):
                self.step_sub(h, phase, phase_time_s + i * h, weather, slipstream_cut, wake_loss)
            return

        p = self.params
        act = self.last_action
        wet = float(weather["track_wetness"])
        air_t = float(weather["air_temp_c"])
        track_t = float(weather["track_temp_c"])
        rho = air_rho_from_alt_temp(self.z_mean, air_t)
        mass = self.dry_mass + max(0.0, self.fuel_kg)
        comp_id = self.geom.compound_map[self.compound]
        comp_spec = COMPOUND_SPECS[comp_id]

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

        # 3) Aero forces + FormulaSim-v5 dynamic ride height compression & rake stall
        fw = float(self.design["aero"]["front_wing"])
        rw = float(self.design["aero"]["rear_wing"])
        rad = float(self.design["cooling"]["radiator_opening"])
        duct = float(self.design["brakes"]["duct"])
        sf = float(self.design["suspension"]["spring_front_n_per_mm"])
        sr = float(self.design["suspension"]["spring_rear_n_per_mm"])
        rh_f_stat = float(self.design["ride_height_mm"]["front"])
        rh_r_stat = float(self.design["ride_height_mm"]["rear"])

        v = math.hypot(self.vx, self.vy)
        v_sq = v * v
        k_f = 2.0 * (sf * 380.0) / max(sf + 380.0, 1.0) * p.k_rh_f_scale
        k_r = 2.0 * (sr * 450.0) / max(sr + 450.0, 1.0) * p.k_rh_r_scale
        df0, _, f_share0 = compute_aero_forces_v5(
            v, fw, rw, self.ride_height_mm[0], self.ride_height_mm[1], rad, self.xmode_frac, p, duct=duct, rho=rho
        )
        df0 *= 1.0 - wake_loss
        g_ahead = self.geom.interp_scalar(self.geom.grade, self.s + 6.0)
        g_behind = self.geom.interp_scalar(self.geom.grade, self.s - 6.0)
        a_vert = max(-3.0, min(8.0, v_sq * (g_ahead - g_behind) / 12.0))
        d_fz_pitch = mass * (self.g_long * 9.80665) * 0.272 / 3.4 * p.pitch_rh_scale
        extra_f = df0 * f_share0 + mass * a_vert * self.wf - d_fz_pitch
        extra_r = df0 * (1.0 - f_share0) + mass * a_vert * (1.0 - self.wf) + d_fz_pitch
        rh_f, self.heave_u_f, fz_front_axle, plank_f = heave_axle(dt, rh_f_stat, self.ride_height_mm[0],
            self.heave_u_f, k_f, mass * self.wf - 48.0, extra_f, self.damping_ratio, mass * 9.80665 * self.wf)
        rh_r, self.heave_u_r, fz_rear_axle, plank_r = heave_axle(dt, rh_r_stat, self.ride_height_mm[1],
            self.heave_u_r, k_r, mass * (1.0 - self.wf) - 52.0, extra_r, self.damping_ratio, mass * 9.80665 * (1.0 - self.wf))
        self.plank_n = plank_f, plank_r
        self.ride_height_mm = [rh_f, rh_r]
        if rh_f < self.min_rh_f_mm:
            self.min_rh_f_mm = rh_f

        df, drag, f_share = compute_aero_forces_v5(
            v, fw, rw, rh_f, rh_r, rad, self.xmode_frac, p, duct=duct, rho=rho
        )
        self.downforce_n = df * (1.0 - wake_loss)
        self.drag_n = drag * (1.0 - slipstream_cut)

        # 4) Vertical axle loads
        grade = self.geom.interp_scalar(self.geom.grade, self.s)
        self.rolling_resistance_n = p.roll_res_c * (fz_front_axle + fz_rear_axle)

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

        self.rpm = max(7500.0, wheel_rpm * self.ratios[self.gear - 1] * self.final_drive)

        # 6) Powertrain: ICE, turbo and MGU-K (brake regen allowed in all modes).
        throttle = act["throttle"]
        brake = act["brake"]
        tau_turbo = 0.10 if throttle > self.turbo_boost else 0.35
        self.turbo_boost += (throttle - self.turbo_boost) * min(1.0, dt / tau_turbo)

        ice_w = 0.0
        if self.fuel_kg > 0.0 and self.shift_timer <= 0.0:
            ice_w = self._ice_power_w(self.rpm) * self.turbo_boost * 0.965
            burn_kg_s = (3000e6 / 3600.0 / 44.0e6) * self.turbo_boost * p.fuel_burn_scale
            self.fuel_kg = max(0.0, self.fuel_kg - burn_kg_s * dt)

        ov_req = bool((act["override"] or act["aid"]) and self.aid_available and brake < 0.05 and phase == "race")
        self.override_active = bool(ov_req and self.ers_soc_mj > 0.01 and throttle > 0.1)
        self.aid_active = self.override_active

        ers_mode = act["ers_mode"]
        disc_cap = self._disc_capacity()
        regen_cap = 70000.0 * self.ers_harvest_level
        recharge_cap = 8.5 + self.aid_extra_recharge_mj

        ers_deploy_w = 0.0
        ers_deploy_dc_w = 0.0
        regen_brk_w = 0.0
        regen_m0_thr_w = 0.0
        regen_coast_w = 0.0

        can_regen = self.vx > 10.0 and self.ers_soc_mj < 4.0 and self.ers_lap_harvested_mj < recharge_cap
        if brake > 0.05:
            # FIA-C C11.6: brake regen ALWAYS active regardless of ers_mode or downshift torque cut
            if can_regen:
                req_r = min(regen_cap, brake * (disc_cap[2] + disc_cap[3]) * self.vx)
                headroom = min(4.0 - self.ers_soc_mj, recharge_cap - self.ers_lap_harvested_mj) * 1e6 / dt
                actual_dc = min(req_r * 0.97, max(0.0, headroom))
                regen_brk_w = actual_dc / 0.97
                d_mj = (actual_dc * dt) / 1e6
                self.ers_soc_mj = min(4.0, self.ers_soc_mj + d_mj)
                self.ers_harvested_mj += d_mj
                self.ers_lap_harvested_mj += d_mj
        elif self.shift_timer <= 0.0:
            if ers_mode == 0 and throttle > 0.05:
                # Mode 0 under throttle: harvests directly from ICE shaft proportional to throttle
                if can_regen:
                    req_r = regen_cap * throttle
                    headroom = min(4.0 - self.ers_soc_mj, recharge_cap - self.ers_lap_harvested_mj) * 1e6 / dt
                    actual_dc = min(req_r * 0.97, max(0.0, headroom))
                    regen_m0_thr_w = actual_dc / 0.97
                    d_mj = (actual_dc * dt) / 1e6
                    self.ers_soc_mj = min(4.0, self.ers_soc_mj + d_mj)
                    self.ers_harvested_mj += d_mj
                    self.ers_lap_harvested_mj += d_mj
            elif ers_mode in (2, 3) and throttle > 0.05 and self.ers_soc_mj > 0.005:
                max_m = compute_mguk_power_v5(self.vx, self.override_active)
                frac = 1.0 if (ers_mode == 3 or self.override_active) else 0.5
                req_dc = max_m * frac * throttle
                max_dc_soc = max(0.0, (self.ers_soc_mj - 0.002) * 1e6 / dt)
                dc_w = min(req_dc, max_dc_soc)
                ers_deploy_dc_w = dc_w
                ers_deploy_w = dc_w * WHEEL_EFFICIENCY
                d_mj = (dc_w * dt) / 1e6
                self.ers_soc_mj = max(0.0, self.ers_soc_mj - d_mj)
                self.ers_deployed_mj += d_mj
                if self.override_active:
                    self.aid_extra_deployed_mj += d_mj
            elif throttle <= 0.05 and can_regen:
                # Coast lift-off regen
                req_r = min(regen_cap, mass * 0.04 * self.ers_harvest_level * 9.80665 * self.vx * p.lift_regen_scale)
                headroom = min(4.0 - self.ers_soc_mj, recharge_cap - self.ers_lap_harvested_mj) * 1e6 / dt
                actual_dc = min(req_r * 0.97, max(0.0, headroom))
                regen_coast_w = actual_dc / 0.97
                d_mj = (actual_dc * dt) / 1e6
                self.ers_soc_mj = min(4.0, self.ers_soc_mj + d_mj)
                self.ers_harvested_mj += d_mj
                self.ers_lap_harvested_mj += d_mj

        ers_w = ers_deploy_dc_w - (regen_brk_w + regen_m0_thr_w + regen_coast_w)
        self.ers_power_kw = ers_w / 1000.0

        # 7) Each corner gets its own load, hot pressure, camber, slip and brake torque.
        eff_m = self.effective_mass()
        if self.vx >= 30.0:
            self.launch_active = False
        f_raw = (ice_w + ers_deploy_w - regen_m0_thr_w) / max(self.vx, 7.0)
        drive = max(0.0, f_raw)
        if self.launch_active:
            drive = min(drive, eff_m * 1.2 * 9.80665)
        eng_brk = mass * 9.80665 * p.engine_brake_g * min(1.0, max(0.0, 1.0 - throttle / 0.15)) if self.vx > 8.0 else 0.0
        request, hydraulic = _wheel_requests(disc_cap, brake, drive, regen_brk_w / max(self.vx, 7.0),
                                             regen_coast_w / max(self.vx, 7.0) + max(0.0, -f_raw), eng_brk)
        wl = self.geom.interp_scalar(self.geom.left, self.s)
        wr = self.geom.interp_scalar(self.geom.right, self.s)
        off_track = self.n > wl or self.n < -wr
        extra_off_drag = mass * 9.80665 * p.offtrack_drag_g if off_track and self.vx > 6.0 else 0.0
        if off_track:
            self.current_lap_offtrack = True
            self.offtrack_ticks += 1
        sus, tyres = self.design["suspension"], self.design["tyres"]
        wheels = wheel_dynamics(p, mass=mass, wf=self.wf, vx=self.vx, vy=self.vy, yaw_rate=self.yaw_rate,
            steer_rad=self.steer_rad, dt=dt, prev_glat=self.g_lat, transfer_f=self.transfer_f, transfer_r=self.transfer_r,
            fz_f_ax=fz_front_axle, fz_r_ax=fz_rear_axle, sf=sf, sr=sr, arb_front=sus["arb_front"],
            arb_rear=sus["arb_rear"], damping_ratio=self.damping_ratio, camber_front_deg=sus["camber_front_deg"],
            camber_rear_deg=sus["camber_rear_deg"], pressure_front_psi=tyres["pressure_front_psi"],
            pressure_rear_psi=tyres["pressure_rear_psi"], tyre_surf_c=self.tyre_surf_c, tyre_core_c=self.tyre_core_c,
            tyre_wear=self.tyre_wear, alpha_eff=self.alpha_eff, wet=wet, off_track=off_track,
            comp_grip=comp_spec["grip"], comp_t_opt=comp_spec["t_opt"], fx_request=request, hydraulic_request=hydraulic,
            harvest_level=self.ers_harvest_level if regen_brk_w + regen_coast_w > 0 else 0)
        self.transfer_f, self.transfer_r = wheels["transfer_f_next"], wheels["transfer_r_next"]
        self.alpha_eff = wheels["alpha_eff_next"]
        self.alpha_f_eff = 0.5 * (self.alpha_eff[0] + self.alpha_eff[1])
        self.alpha_r_eff = 0.5 * (self.alpha_eff[2] + self.alpha_eff[3])
        self.wheel_fz, self.wheel_fx, self.wheel_fy = wheels["fz"], wheels["fx"], wheels["fy"]
        self.wheel_util = wheels["util"]
        self.hydraulic_force_n = wheels["hydraulic_force"]
        self.front_sliding, self.rear_sliding = bool(wheels["front_sliding"]), bool(wheels["rear_sliding"])
        self.front_slide_ticks += int(self.front_sliding)
        self.rear_slide_ticks += int(self.rear_sliding)

        # 8) No lateral/yaw-rate attenuation, irrespective of tyre utilisation.
        ax_body = (wheels["fx_body"] - self.drag_n - self.rolling_resistance_n - extra_off_drag - mass * 9.80665 * grade) / eff_m
        ay_body = wheels["fy_body"] / mass
        dvy = ay_body - self.yaw_rate * self.vx
        self.vx = max(0.0, self.vx + (ax_body + self.yaw_rate * self.vy) * dt)
        # Body-longitudinal speed can approach zero in a fast broadside slide.
        # Only use the near-rest model when the *ground-speed magnitude* is low.
        vy_dyn = self.vy + dvy * dt
        ground_speed = math.hypot(self.vx, vy_dyn)
        if ground_speed < 0.2 and throttle < 0.05:
            self.vx = self.vy = self.yaw_rate = 0.0
            self.alpha_eff = (0.0,) * 4
        elif ground_speed < 5.0:
            self.vy = 0.0
            self.yaw_rate = self.vx * math.tan(self.steer_rad) / 3.4
            self.alpha_eff = (0.0,) * 4
        else:
            self.vy = vy_dyn
            self.yaw_rate += wheels["mz_body"] * dt / (mass * 1.15 ** 2 * p.iz_scale)
        self.g_long = ax_body / 9.80665
        self.g_lat = ay_body / 9.80665

        # 9) Advance Frenet coordinates
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
        if abs(self.heading_error) > 0.65 and self.vx > 10.0:
            self.spinout_ticks += 1

        # 10) Wall collision check
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
            self.n = max(-(wall_r - 1.05), min(wall_l - 1.05, self.n))
            self.vx *= 0.65
            self.vy *= -0.2
            self.heading_error *= 0.5
            self.collision_count += 1

        # 11) Tyre 2-node thermal model, wear, and PU coolant temp
        thermal = wheel_thermal(p, wheels=wheels, tyre_surf_c=self.tyre_surf_c, tyre_core_c=self.tyre_core_c,
            tyre_wear=self.tyre_wear, brake_temp_c=self.brake_temp_c, dt=dt, v=v, wet=wet, air_temp_c=air_t,
            track_temp_c=track_t, duct=duct, comp_heat=comp_spec["heat"], comp_wear=comp_spec["wear"], comp_t_opt=comp_spec["t_opt"])
        self.tyre_surf_c[:] = thermal["tyre_surf_c_next"]
        self.tyre_core_c[:] = thermal["tyre_core_c_next"]
        self.tyre_wear[:] = thermal["tyre_wear_next"]
        self.brake_temp_c[:] = thermal["brake_temp_c_next"]

        heat_in = 0.37 * (ice_w / 0.486 * (1.0 - 0.486) + abs(ers_w) * 0.06)
        heat_out = 1500.0 * (rad ** 0.85) * (1.0 + v / 70.0) * max(0.0, self.pu_temp_c - air_t)
        self.pu_temp_c = max(air_t, min(145.0, self.pu_temp_c + ((heat_in - heat_out) / 120000.0) * dt))
        if self.pu_temp_c > self.max_pu_temp_c:
            self.max_pu_temp_c = self.pu_temp_c

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

        # 13) G13 compound use & wet waiver accounting
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


def _build_observation_v5(
    car: CarStateV5,
    all_cars: Sequence[CarStateV5],
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
            if abs(gap_m) > 400.0:
                continue
            rivals.append({
                "seat": other.seat,
                "gap_m": float(gap_m),
                "dn": float(other.n - car.n),
                "speed_mps": float(math.hypot(other.vx, other.vy)),
                "lap": int(other.lap),
                "status": other.status,
            })

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
            "brake_force_max_n": float(sum(car._disc_capacity())),
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
        "kernel_version": "rsi-formula-5.0-v3",
        "phase": phase,
        "phase_tick": phase_tick,
        "weather": weather,
        "forecast": forecast,
    }


def run_weekend(
    track_id: str,
    entries: Sequence[Dict[str, Any]],
    weather_profile: WeatherProfile,
    params: Optional[SimV5ParamsCPU] = None,
    race_laps: int = 4,
    run_qualifying: bool = True,
    max_race_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Run a complete 1..10-car weekend (qualifying + 4-lap race) in FormulaSim-v5 v3."""
    p = params if params is not None else SimV5ParamsCPU.load()
    geom = load_circuit(track_id, resample_ds=4.0)
    forecast = weather_profile.forecast()
    n_cars = len(entries)
    cars = [
        CarStateV5(seat=i, design=select_sealed_design(e["design_envelope"], forecast), geom=geom, params=p)
        for i, e in enumerate(entries)
    ]
    act_fns = [e["act_fn"] for e in entries]
    faults = [0] * n_cars
    cpu_times = [0.0] * n_cars

    global_tick = 0
    decision_counts = [0] * n_cars
    qual_results: List[Tuple[bool, float, int]] = []

    # ---- PHASE 1: QUALIFYING ----
    if run_qualifying:
        q_max_ticks = int(math.ceil(60.0 * (geom.length / 55.0 + 120.0)))
        for seat in range(n_cars):
            car = cars[seat]
            car.reset_for_phase("qualifying", grid_slot=seat)
            for pt in range(q_max_ticks):
                w = weather_profile.at("qualifying", pt / 60.0)
                if pt % 2 == 0:
                    obs = _build_observation_v5(car, cars, "qualifying", global_tick, pt, decision_counts[seat], 1, 1, w, forecast)
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

    # ---- PHASE 2: RACE ----
    t_scale = 7200.0 * race_laps / (math.floor(305000.0 / geom.length) + 1)
    race_limit_s = min(t_scale + 120.0, max_race_seconds) if max_race_seconds else (t_scale + 120.0)
    r_max_ticks = int(math.ceil(60.0 * race_limit_s))

    for slot_idx, seat in enumerate(grid_order):
        cars[seat].reset_for_phase("race", grid_slot=slot_idx)

    det_cross_history: List[Tuple[float, int]] = []

    for pt in range(r_max_ticks):
        phase_s = pt / 60.0
        w = weather_profile.at("race", phase_s)
        wet = float(w["track_wetness"])

        order = sorted(range(n_cars), key=lambda i: (-cars[i].lap, -cars[i].distance, i))
        pos_by_seat = {seat: rank + 1 for rank, seat in enumerate(order)}

        if pt % 2 == 0:
            obs_list: List[Optional[Dict[str, Any]]] = [None] * n_cars
            for i in range(n_cars):
                if cars[i].status == "racing":
                    obs_list[i] = _build_observation_v5(
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
                cars[i].pit_request = bool(parsed["pit_request"])
                if parsed["pit_compound"] is not None:
                    cars[i].pit_compound_target = parsed["pit_compound"]

        # Slipstream & dirty-air wake
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
                    slip_cuts[i] = max(slip_cuts[i], p.slipstream_max * math.exp(-gap / 15.0) * lat_f)
                    wake_losses[i] = max(wake_losses[i], min(p.dirty_air_max, p.dirty_air_max * math.exp(-gap / 14.43) * lat_f))

        prev_s_list = [c.s for c in cars]
        for _ in range(2):
            for i in range(n_cars):
                cars[i].step_sub(1.0 / 120.0, "race", phase_s, w, slip_cuts[i], wake_losses[i])

        # Manual Overtake Mode detection & activation
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
                eligible = any(0.0 < (phase_s - t_prev) <= 1.0 and s_prev != i for t_prev, s_prev in det_cross_history[-15:])
                ci.aid_pending = eligible
                det_cross_history.append((phase_s, i))
            if ci.distance > 0.0 and _crossed_s(prev_s_list[i], ci.s, geom.aid_activation_s, geom.length):
                ci.aid_available = ci.aid_pending
                ci.override_available = ci.aid_pending
                ci.aid_zone = 1 if ci.aid_pending else None
                ci.aid_pending = False

        # Car-to-car contact separation
        for i in range(n_cars):
            ci = cars[i]
            if ci.status != "racing" or ci.pit_state > 0:
                continue
            for j in range(i + 1, n_cars):
                cj = cars[j]
                if cj.status != "racing" or cj.pit_state > 0:
                    continue
                if abs(ci.distance - cj.distance) < 5.0 and abs(ci.n - cj.n) < 1.9:
                    ci.collision_count += 1
                    cj.collision_count += 1
                    ci.vx *= 0.88
                    cj.vx *= 0.88
                    push = 0.25 if ci.n >= cj.n else -0.25
                    ci.n += push
                    cj.n -= push

        # Finishes & official 30s stall retirement
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
        classification.append({
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
            "min_rh_f_mm": round(car.min_rh_f_mm, 2),
            "max_pu_temp_c": round(car.max_pu_temp_c, 2),
            "offtrack_ticks": car.offtrack_ticks,
            "front_slide_ticks": car.front_slide_ticks,
            "rear_slide_ticks": car.rear_slide_ticks,
            "spinout_ticks": car.spinout_ticks,
            "collisions": car.collision_count,
            "cpu_seconds": round(cpu_times[i], 4),
        })

    classification.sort(key=lambda r: (r["disqualified"], -r["laps"], r["total_time_s"], r["seat"]))
    for pos_idx, row in enumerate(classification):
        row["position"] = pos_idx + 1

    return {
        "track_id": track_id,
        "simulator": "FormulaSim-v5-v3",
        "weather": weather_profile.forecast(),
        "race_laps": race_laps,
        "classification": classification,
    }
