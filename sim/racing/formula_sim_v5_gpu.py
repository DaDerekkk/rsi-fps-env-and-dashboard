"""FormulaSim-v5 v3: shared four-wheel contact and thermal physics plus a torch batch substep.

The scalar weekend engine calls the same contact functions without requiring
PyTorch. The tensor entrypoint is a system-identification kernel, not the
entire weekend/actuator/sporting simulation. Replay-rate input should be
substepped and latent state propagated, rather than reset from future targets.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    torch = nn = F = None  # The scalar weekend requires only the standard library and numpy.

# Official compound specs (racing-sporting.zh.md)
# index 0=C5/soft_park, etc. Mapped per (track_idx, comp_idx)
TRACK_COMP_SPECS = {
    # (grip, t_opt, wear, heat)
    0: {  # parkland: C5, C4, C3
        0: (1.024, 88.0, 1.80, 1.10),
        1: (1.012, 92.0, 1.25, 1.05),
        2: (1.000, 96.0, 0.90, 1.00),
    },
    1: {  # highland: C4, C3, C2
        0: (1.012, 92.0, 1.25, 1.05),
        1: (1.000, 96.0, 0.90, 1.00),
        2: (0.988, 101.0, 0.70, 0.95),
    },
    2: {  # crossover: C3, C2, C1
        0: (1.000, 96.0, 0.90, 1.00),
        1: (0.988, 101.0, 0.70, 0.95),
        2: (0.976, 106.0, 0.55, 0.90),
    },
}

WHEEL_EFF = 0.93605


def air_rho_from_alt_temp(z_m: float, air_temp_c: float) -> float:
    """ISA barometric + thermal air density referenced to 1.200 kg/m^3 at 130m / 20C."""
    p_rel = math.exp(-(float(z_m) - 130.0) / 8434.5)
    t_rel = 293.15 / max(250.0, float(air_temp_c) + 273.15)
    return 1.200 * p_rel * t_rel


@dataclass
class SimV5Params:
    """Parameters and public structural anchors; defaults are not fit results."""
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
    rake_stall_slope: float = 0.015    # fractional floor loss per mm above rake_stall_mm
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
    # Tyre Pacejka & grip & post-peak drop-off (lateral oversteer vs longitudinal wheelspin/lockup)
    mu_f_base: float = 2.520
    mu_r_base: float = 2.520
    load_sens_f: float = 0.260
    load_sens_r: float = 0.200
    stiff_f_scale: float = 1.000
    stiff_r_scale: float = 1.000
    cy_lat: float = 1.350
    ey_lat: float = -0.500
    post_peak_drop_f: float = 0.020
    post_peak_drop_r: float = 0.060    # gentle extra lateral drop past peak (Pacejka already drops past peak)
    wheelspin_drop: float = 0.070      # official tyre.wheelspin_force_loss (applied to fx_r > 0)
    lockup_drop: float = 0.120         # official tyre.lockup_force_loss (applied to braking fx < 0)
    rear_diff_lock: float = 0.250      # 0=open diff equal rear drive fx, 1=locked spool asymmetry
    grip_scale: float = 1.000
    # Continuous nonlinear wet grip + hydroplaning
    wet_grip_drop: float = 0.400
    wet_grip_exp: float = 0.850
    wet_hydro_v_k: float = 0.0050      # extra high-speed wet grip loss proportional to wet*(v/80)^2
    # Off-track grass/gravel penalty
    offtrack_grip_mult: float = 0.720
    offtrack_drag_g: float = 0.460
    # Powertrain, engine braking, braking & inertia
    ice_power_scale: float = 0.985
    engine_brake_g: float = 0.110      # ICE overrun drag on throttle lift-off
    lift_regen_scale: float = 0.300    # coast lift-off MGU-K regen factor
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
    def from_dict(cls, d: Dict[str, Any]) -> "SimV5Params":
        valid = {k: float(v) for k, v in d.items() if hasattr(cls, k)}
        return cls(**valid)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


# Physical bounds (lo, hi) for constrained differentiable SysID fitting
PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "fw_cza_k": (0.582, 0.582),
    "rw_cza_k": (0.747, 0.747),
    "floor_cza_k": (0.85, 1.05),
    "body_cza": (0.05, 0.16),
    "fw_x_retain": (0.20, 0.40),
    "rw_x_retain": (0.08, 0.22),
    "downforce_scale": (0.94, 1.06),
    "rake_ref_mm": (26.0, 42.0),
    "rake_gain_per_mm": (0.001, 0.008),
    "rake_stall_mm": (90.0, 90.0),
    "rake_stall_slope": (0.002, 0.035),
    "floor_f_share_base": (0.34, 0.50),
    "floor_f_share_rake_k": (-0.0015, -0.0015),
    "rh_f_stall_mm": (15.0, 21.0),
    "cda_base": (0.42, 0.54),
    "fw_ind_k": (0.08, 0.22),
    "rw_ind_k": (0.25, 0.52),
    "floor_ind_k": (0.02, 0.08),
    "rad_cda_k": (0.10, 0.20),
    "drag_scale": (0.94, 1.06),
    "roll_res_c": (0.008, 0.018),
    "k_rh_f_scale": (0.80, 1.35),
    "k_rh_r_scale": (0.75, 1.30),
    "pitch_rh_scale": (0.70, 1.40),
    "mu_f_base": (2.25, 2.75),
    "mu_r_base": (2.25, 2.75),
    "load_sens_f": (0.16, 0.34),
    "load_sens_r": (0.12, 0.30),
    "stiff_f_scale": (0.60, 1.50),
    "stiff_r_scale": (0.60, 1.50),
    "cy_lat": (1.15, 1.65),
    "ey_lat": (-1.10, 0.20),
    "post_peak_drop_f": (0.00, 0.20),
    "post_peak_drop_r": (0.00, 0.25),
    "wheelspin_drop": (0.03, 0.14),
    "lockup_drop": (0.04, 0.18),
    "rear_diff_lock": (0.00, 0.70),
    "grip_scale": (0.90, 1.12),
    "wet_grip_drop": (0.25, 0.55),
    "wet_grip_exp": (0.55, 1.25),
    "wet_hydro_v_k": (0.00, 0.06),
    "offtrack_grip_mult": (0.50, 0.85),
    "offtrack_drag_g": (0.20, 0.80),
    "ice_power_scale": (0.94, 1.05),
    "engine_brake_g": (0.02, 0.22),
    "lift_regen_scale": (0.10, 0.85),
    "fuel_burn_scale": (0.92, 1.10),
    "brake_max_n": (56000.0, 56000.0),
    "brake_scale": (0.90, 1.10),
    "iz_scale": (0.75, 1.35),
    "heat_scale": (0.40, 2.20),
    "wet_cool_k": (5.0, 90.0),
    "wear_scale": (0.30, 2.50),
    "slipstream_max": (0.22, 0.35),
    "dirty_air_max": (0.28, 0.45),
}


# Shared CPU/Torch equations. Each wheel value is either a Python scalar or a
# vector of independent cars (no conversion to numpy / detach in the tensor path).
def _clip(x, lo, hi):
    if torch is not None and torch.is_tensor(x):
        return x.clamp(lo, hi)
    return max(lo, min(hi, x))


def _pos(x):
    if torch is not None and torch.is_tensor(x):
        return x.clamp_min(0.)
    return max(0., x)


def disc_friction(temp_c):
    """Piecewise linear through the five registered carbon-disc anchors."""
    return (0.60 + 0.25 * _clip(temp_c / 200., 0., 1.)
            + 0.15 * _clip((temp_c - 200.) / 250., 0., 1.)
            - 0.55 * _clip((temp_c - 950.) / 300., 0., 1.))


def wheel_dynamics(p, *, mass, wf, vx, vy, yaw_rate, steer_rad, dt, prev_glat,
                   transfer_f, transfer_r, fz_f_ax, fz_r_ax, sf, sr, arb_front,
                   arb_rear, damping_ratio, camber_front_deg, camber_rear_deg,
                   pressure_front_psi, pressure_rear_psi, tyre_surf_c, tyre_core_c,
                   tyre_wear, alpha_eff, wet, off_track, comp_grip, comp_t_opt,
                   fx_request, hydraulic_request, harvest_level):
    """Dual-track tyre contact, FL/FR/RL/RR; coordinates x forward, y left.

    Positive lateral acceleration transfers load to the right. Splitting lateral
    transfer into geometric, unsprung and elastic terms conserves each axle's
    total load, even after inner-wheel lift. Spring/ARB roll sharing and
    first-order damping-aware lag are modeling closures, not published engine code.
    """
    batched = torch is not None and torch.is_tensor(vx)
    m = torch if batched else math
    select = torch.where if batched else (lambda cond, yes, no: yes if cond else no)
    f_s, r_s = mass * wf - 48., mass * (1. - wf) - 52.
    kf = sf * 1000. * 1.58 ** 2 / 2. + arb_front * (600. * 180. / math.pi)
    kr = sr * 1000. * 1.52 ** 2 / 2. + arb_rear * (600. * 180. / math.pi)
    ay = prev_glat * 9.80665
    mf_geo, mr_geo = f_s * .055 + 48. * .34, r_s * .115 + 52. * .34
    m_el = ay * (mass * .272 - mf_geo - mr_geo)
    roll_goal = m_el / (kf + kr)
    w = 1. - m.exp(-dt / (.05 * damping_ratio / .5))
    tf = transfer_f + w * ((ay * mf_geo + kf * roll_goal) / 1.58 - transfer_f)
    tr = transfer_r + w * ((ay * mr_geo + kr * roll_goal) / 1.52 - transfer_r)
    # Recover lagged acceleration from total roll moment, so camber cannot
    # jump while the transferred axle loads still represent the previous state.
    ay_lag = (tf * 1.58 + tr * 1.52) / (mass * .272)
    roll_deg = ay_lag * (mass * .272 - mf_geo - mr_geo) / (kf + kr) * (180. / math.pi)
    speed = m.sqrt(vx * vx + vy * vy + 1.e-12)
    vx_safe = 4. + _pos(abs(vx) - 4.)
    wet_c = _clip(wet, 0., 1.)
    wet_term = p.wet_grip_drop * ((wet_c + 1.e-6) ** p.wet_grip_exp) * (wet_c > 1.e-4)
    wet_grip = _clip(1. - wet_term - p.wet_hydro_v_k * wet_c * (speed / 80.) ** 2, .30, 1.)
    grip0 = wet_grip * p.grip_scale * comp_grip
    lateral_surface = 1. - (1. - p.offtrack_grip_mult) * (off_track > .5)
    wheel_grip = 0.
    out = {k: [] for k in ('fz', 'fx', 'fy', 'util', 'pressure_psi', 'camber_deg',
                            'alpha_eff_next', 'hydraulic_force')}
    fx_body = fy_body = mz_body = 0.
    slide = []
    fxs, fys, steers, xs, ys = [], [], [], [], []
    for i in range(4):
        front = i < 2
        side = 1. if i % 2 == 0 else -1.
        track, fz_ax, transfer = (1.58, fz_f_ax, tf) if front else (1.52, fz_r_ax, tr)
        half = .5 * fz_ax
        fz = half * (1. - side * _clip(transfer / (half + 1.e-12), -1., 1.))
        y = .5 * track * side
        x = 3.4 * (1. - wf) if front else -3.4 * wf
        steer = 0. * vx
        if front:
            tangent = m.tan(steer_rad)
            ideal = m.atan(tangent / (1. - y * tangent / 3.4))
            steer = steer_rad + .6 * (ideal - steer_rad)
        vl = 4. + _pos(vx - y * yaw_rate - 4.)
        alpha_target = steer - m.atan2(vy + x * yaw_rate, vl)
        alpha0 = alpha_eff[i]
        alpha = alpha0 + (alpha_target - alpha0) * (1. - m.exp(-vx_safe * dt / (.35 if front else .40)))
        fz0, c1, opt = (3800., 58., -3.3) if front else (5200., 56., -2.4)
        pressure = (pressure_front_psi if front else pressure_rear_psi) + .13 * (tyre_core_c[i] - 70.)
        camber = (camber_front_deg if front else camber_rear_deg) - side * .25 * roll_deg
        mu0, sens = (p.mu_f_base, p.load_sens_f) if front else (p.mu_r_base, p.load_sens_r)
        mu = mu0 * ((.15 + _pos(fz / fz0 - .15)) ** (-sens)) * grip0
        dtemp = tyre_surf_c[i] - comp_t_opt
        width = 40. + 5. * (dtemp < 0.)
        mu = mu * _clip(1. - .5 * (dtemp / width) ** 2, .45, 1.)
        wear = _clip(tyre_wear[i], 0., 1.)
        mu = mu * _clip(1. - .05 * wear - .40 * (_pos(wear - .80) / .20) ** 2, .45, 1.)
        mu = mu * (.85 + _pos(1. - .01 * (pressure - 29.) - .85))
        mu = mu * (.85 + _pos(1. - .0035 * (camber - opt) ** 2 - .85))
        peak = mu * fz
        wheel_grip = wheel_grip + peak
        stiff_scale = p.stiff_f_scale if front else p.stiff_r_scale
        stiff = c1 * fz0 * m.sin(2. * m.atan(fz / (1.4 * fz0))) * stiff_scale * (1. + .012 * (pressure - 29.))
        ba = stiff * alpha / (.0001 + _pos(p.cy_lat * peak - .0001))
        fy_raw = peak * m.sin(p.cy_lat * m.atan(ba - p.ey_lat * (ba - m.atan(ba))))
        fy_raw = fy_raw + .6 * fz * side * camber * (math.pi / 180.)
        # Rear regenerative braking reduces only the available *braking*
        # margin, not lateral peak or drive grip. Zero Fz means zero force.
        fx_req = fx_request[i]
        long_margin = 1. if front else (1. - .012 * harvest_level * (fx_req < 0.))
        peak_x = 1. + _pos(1.05 * peak * long_margin - 1.)
        peak_y = 1. + _pos(peak - 1.)
        ux_raw = fx_req / peak_x
        uy_raw = fy_raw / peak_y
        util = m.sqrt(ux_raw * ux_raw + uy_raw * uy_raw + 1.e-12)
        excess = _clip((util - 1.) / .25, 0., 1.)
        has_fz = (fz > 1.e-8)
        drop_x = p.lockup_drop if front else select(fx_req > 0., p.wheelspin_drop, p.lockup_drop)
        drop_y = p.post_peak_drop_f if front else p.post_peak_drop_r
        # Drive (fx_req > 0): cap drive force to remaining friction-ellipse capacity
        # so large low-gear/wet torque requests do not erase rear lateral cornering stiffness.
        uy_c = _clip(uy_raw, -.99, .99)
        avail_x = peak_x * m.sqrt(_pos(1. - uy_c * uy_c) + 1.e-6)
        fx_drv = select(fx_req > avail_x, avail_x * (1. - drop_x * excess), fx_req) * has_fz
        fy_drv = fy_raw * (1. - drop_y * excess) * lateral_surface * has_fz
        # Braking (fx_req <= 0): clamped-ux friction ellipse preserves authentic trail-braking
        # combined-slip tradeoff and asymmetric inside-wheel lockup yaw moment.
        ux_c = _clip(ux_raw, -1.35, 1.35)
        u_eff = m.sqrt(ux_c * ux_c + uy_raw * uy_raw + 1.e-12)
        denom_brk = 1. + _pos(u_eff - 1.)
        fx_brk = (ux_c * peak_x) * (1. - drop_x * excess) / denom_brk * has_fz
        fy_brk = fy_raw * (1. - drop_y * excess) / denom_brk * lateral_surface * has_fz
        is_drv = (fx_req > 0.)
        fx = select(is_drv, fx_drv, fx_brk)
        fy = select(is_drv, fy_drv, fy_brk)
        scale_x = select(abs(fx_req) > 1.e-9, fx / select(abs(fx_req) > 1.e-9, fx_req, 1. + 0. * vx), 0. * vx)
        fxs.append(fx)
        fys.append(fy)
        steers.append(steer)
        xs.append(x)
        ys.append(y)
        slide.append((util > 1.02) | (abs(alpha) > .14))
        for k, value in zip(('fz', 'util', 'pressure_psi', 'camber_deg', 'alpha_eff_next', 'hydraulic_force'),
                            (fz, util, pressure, camber, alpha, hydraulic_request[i] * scale_x)):
            out[k].append(value)
    # Rear electronic differential yaw-couple factor on positive drive torque:
    # every wheel obeys its own fz/friction-circle cap (fz==0 => fx==0), total rear
    # drive force is conserved, braking asymmetry stays 100% full-strength, and
    # rear drive-torque yaw coupling scales cleanly with `rear_diff_lock` in [0, 1].
    both_drv = (fx_request[2] > 0.) & (fx_request[3] > 0.)
    diff_lock = getattr(p, 'rear_diff_lock', 0.25)
    for i in range(4):
        fx_i, fy_i, steer_i = fxs[i], fys[i], steers[i]
        fx_b = fx_i * m.cos(steer_i) - fy_i * m.sin(steer_i)
        fy_b = fy_i * m.cos(steer_i) + fx_i * m.sin(steer_i)
        fx_body, fy_body = fx_body + fx_b, fy_body + fy_b
        if i >= 2:
            mz_x_term = select(both_drv, diff_lock * ys[i] * fx_b, ys[i] * fx_b)
        else:
            mz_x_term = ys[i] * fx_b
        mz_body = mz_body + xs[i] * fy_b - mz_x_term
        out['fx'].append(fx_i)
        out['fy'].append(fy_i)
    out.update(fx_body=fx_body, fy_body=fy_body, mz_body=mz_body, ay_max=wheel_grip / mass,
               front_sliding=slide[0] | slide[1], rear_sliding=slide[2] | slide[3],
               transfer_f_next=tf, transfer_r_next=tr)
    return out


def wheel_thermal(p, *, wheels, tyre_surf_c, tyre_core_c, tyre_wear, brake_temp_c,
                  dt, v, wet, air_temp_c, track_temp_c, duct, comp_heat, comp_wear, comp_t_opt):
    """Explicit energy balances. Only applied hydraulic force heats the disc."""
    batched = torch is not None and torch.is_tensor(v)
    m = torch if batched else math
    out = {k: [] for k in ('tyre_surf_c_next', 'tyre_core_c_next', 'tyre_wear_next', 'brake_temp_c_next')}
    for i in range(4):
        front = i < 2
        area = 1. if front else 1.339285714
        ts, tc, tb = tyre_surf_c[i], tyre_core_c[i], brake_temp_c[i]
        slip_w = (abs(wheels['fy'][i] * m.sin(wheels['alpha_eff_next'][i]))
                  + .15 * _pos(wheels['util'][i] - .65) * abs(wheels['fx'][i])) * v
        cold_p = _pos(29. - wheels['pressure_psi'][i])
        cam = _pos(abs(wheels['camber_deg'][i]) - 2.5)
        slip_heat = slip_w * (1. if front else .78) * comp_heat * p.heat_scale * (1. + .1 * cam)
        roll_heat = .012 * wheels['fz'][i] * v * (1. + .05 * cold_p)
        contact = (25. + p.wet_cool_k * wet) * area * (ts - track_temp_c)
        exchange = 90. * area * (ts - tc)
        soak = .9 * (tb - tc)
        dts = (.7 * slip_heat - contact - exchange - (15. + .5 * v) * area * (ts - air_temp_c)) / (5500. * area)
        dtc = (.3 * slip_heat + roll_heat + exchange + soak - (40. + v) * area * (tc - air_temp_c)) / (26000. * area)
        # The register specifies a reference conductance at duct=.5, not a
        # transfer function. This bounded linear opening law is a closure.
        cooling = (14. + .35 * v) * (.5 + duct) * (tb - air_temp_c)
        dtb = (wheels['hydraulic_force'][i] * v - cooling - soak) / (2800. if front else 2100.)
        over = _pos(ts - comp_t_opt - 12.) / 20.
        dw = slip_w / 26.e6 * comp_wear * p.wear_scale * (1. + over * over) * (1. + .06 * cold_p) * (1. + .1 * cam)
        for k, value in zip(out, (ts + dt * dts, tc + dt * dtc,
                                  _clip(tyre_wear[i] + dt * dw, 0., 1.), tb + dt * dtb)):
            out[k].append(value)
    return out


class SimV5PhysicsModule(nn.Module if nn is not None else object):
    """Differentiable PyTorch GPU physics core for both System Identification & 2048-Env RL!"""

    def __init__(self, init_params: Optional[SimV5Params] = None):
        if torch is None:
            raise ImportError("PyTorch is required only for SimV5PhysicsModule")
        super().__init__()
        p = init_params or SimV5Params()
        self.param_names = list(PARAM_BOUNDS.keys())
        raw_vals = []
        los, his = [], []
        for k in self.param_names:
            lo, hi = PARAM_BOUNDS[k]
            v = float(getattr(p, k))
            frac = 0.5 if lo == hi else max(1e-4, min(1.0 - 1e-4, (v - lo) / (hi - lo)))
            logit = math.log(frac / (1.0 - frac))
            raw_vals.append(logit)
            los.append(lo)
            his.append(hi)
        self.raw = nn.Parameter(torch.tensor(raw_vals, dtype=torch.float32))
        self.register_buffer("lo", torch.tensor(los, dtype=torch.float32))
        self.register_buffer("hi", torch.tensor(his, dtype=torch.float32))
        self._idx = {k: i for i, k in enumerate(self.param_names)}

    def p(self, name: str) -> torch.Tensor:
        i = self._idx[name]
        return self.lo[i] + (self.hi[i] - self.lo[i]) * torch.sigmoid(self.raw[i])

    def export_params(self) -> SimV5Params:
        vals = (self.lo + (self.hi - self.lo) * torch.sigmoid(self.raw)).detach().cpu().tolist()
        return SimV5Params(**{k: float(v) for k, v in zip(self.param_names, vals)})

    def aero_forces(
        self,
        v: torch.Tensor,
        fw: torch.Tensor,
        rw: torch.Tensor,
        rh_f: torch.Tensor,
        rh_r: torch.Tensor,
        rad: torch.Tensor,
        xmode: torch.Tensor,
        duct: Optional[torch.Tensor] = None,
        rho: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (downforce_n, drag_n, front_share) vectorized over [B]."""
        fw_z = self.p("fw_cza_k") * (1.0 - (1.0 - torch.clamp(fw / 32.0, 0.0, 1.0)) ** 2)
        rw_z = self.p("rw_cza_k") * (1.0 - (1.0 - torch.clamp(rw / 34.0, 0.0, 1.0)) ** 2)
        fw_cza = fw_z * (1.0 - (1.0 - self.p("fw_x_retain")) * xmode)
        rw_cza = rw_z * (1.0 - (1.0 - self.p("rw_x_retain")) * xmode)

        rh_f_c = torch.clamp(rh_f, min=3.0)
        rh_r_c = torch.clamp(rh_r, min=10.0)
        rake_mm = torch.clamp(rh_r_c - rh_f_c, min=0.0)

        floor_gain = 1.0 + 0.01 * (28.0 - rh_f_c) + self.p("rake_gain_per_mm") * (rake_mm - self.p("rake_ref_mm"))
        floor_gain = torch.clamp(floor_gain, 0.40, 1.40)

        # Front floor stall below rh_f_stall_mm
        rh_s = self.p("rh_f_stall_mm")
        stall_ratio = torch.clamp(rh_f_c / rh_s, max=1.0)
        floor_gain = floor_gain * (stall_ratio ** 2.5)

        # Public rake stall threshold is 90 mm; the post-stall slope is a residual.
        excess_rake = F.relu(rake_mm - self.p("rake_stall_mm"))
        floor_gain = floor_gain * torch.clamp(1.0 - self.p("rake_stall_slope") * excess_rake, 0.35, 1.0)

        floor_cza = self.p("floor_cza_k") * floor_gain
        body_cza = self.p("body_cza")
        cza_raw = fw_cza + rw_cza + floor_cza + body_cza
        cza_tot = cza_raw * self.p("downforce_scale")

        # Positive rake moves floor balance rearward.
        floor_f_sh = torch.clamp(
            self.p("floor_f_share_base") + self.p("floor_f_share_rake_k") * (rake_mm - self.p("rake_ref_mm")),
            0.25, 0.62,
        )
        front_cza = fw_cza + floor_cza * floor_f_sh + 0.45 * body_cza
        front_share = front_cza / torch.clamp(cza_raw, min=1e-4)

        fw_prof = 0.040 * (1.0 - xmode) + 0.035 * xmode
        rw_prof = 0.050 * (1.0 - xmode) + 0.035 * xmode
        fw_ind = self.p("fw_ind_k") * fw_cza * fw_cza
        rw_ind = self.p("rw_ind_k") * rw_cza * rw_cza
        floor_ind = self.p("floor_ind_k") * (floor_cza / 0.96) ** 2
        cool_cda = self.p("rad_cda_k") * (torch.clamp(rad, min=0.1) ** 1.25)
        duct_cda = 0.020 * (duct if duct is not None else 0.45)
        cda_tot = (self.p("cda_base") + fw_prof + rw_prof + fw_ind + rw_ind + floor_ind + cool_cda + duct_cda) * self.p("drag_scale")

        rho_v = rho if rho is not None else 1.2
        q_dyn = 0.5 * rho_v * v * v
        return q_dyn * cza_tot, q_dyn * cda_tot, front_share

    def tyre_mu(
        self,
        axle: str,
        fz_wheel: torch.Tensor,
        temp_c: torch.Tensor,
        wear: torch.Tensor,
        wet: torch.Tensor,
        v: torch.Tensor,
        comp_grip: torch.Tensor,
        comp_t_opt: torch.Tensor,
    ) -> torch.Tensor:
        if axle == "front":
            fz0 = 3800.0
            mu0 = self.p("mu_f_base")
            sens = self.p("load_sens_f")
        else:
            fz0 = 5200.0
            mu0 = self.p("mu_r_base")
            sens = self.p("load_sens_r")

        load_ratio = torch.clamp(fz_wheel / fz0, min=0.15)
        mu = mu0 * (load_ratio ** (-sens)) * comp_grip

        w_t = torch.where(temp_c < comp_t_opt, torch.full_like(temp_c, 45.0), torch.full_like(temp_c, 40.0))
        therm = torch.clamp(1.0 - 0.5 * ((temp_c - comp_t_opt) / w_t) ** 2, 0.45, 1.0)
        mu = mu * therm

        wc = torch.clamp(wear, 0.0, 1.0)
        cliff = F.relu(wc - 0.80) / 0.20
        w_fac = torch.clamp(1.0 - 0.05 * wc - 0.40 * cliff * cliff, 0.45, 1.0)
        mu = mu * w_fac

        # Continuous nonlinear wet grip + speed hydroplaning term
        wet_c = torch.clamp(wet, 0.0, 1.0)
        wet_fac = torch.clamp(
            1.0 - self.p("wet_grip_drop") * ((wet_c + 1e-6) ** self.p("wet_grip_exp")) * (wet_c > 1e-4).float()
            - self.p("wet_hydro_v_k") * wet_c * ((v / 80.0) ** 2),
            0.30, 1.0,
        )
        return mu * wet_fac * self.p("grip_scale")

    def mguk_power_w(self, v: torch.Tensor, override: torch.Tensor) -> torch.Tensor:
        kph = v * 3.6
        # Piecewise linear interpolation matching official mechanics.npz
        # Normal: 0..290 -> 350kW, 290..337.5 -> 350..112.5, 337.5..340 -> 112.5..100, 340..345 -> 100..0
        p_norm = torch.where(
            kph <= 290.0,
            torch.full_like(kph, 350000.0),
            torch.where(
                kph <= 337.5,
                350000.0 - (kph - 290.0) * (237500.0 / 47.5),
                torch.where(
                    kph <= 340.0,
                    112500.0 - (kph - 337.5) * (12500.0 / 2.5),
                    torch.clamp(100000.0 - (kph - 340.0) * 20000.0, min=0.0),
                ),
            ),
        )
        p_ov = torch.where(
            kph <= 337.5,
            torch.full_like(kph, 350000.0),
            torch.where(
                kph <= 340.0,
                350000.0 - (kph - 337.5) * (50000.0 / 2.5),
                torch.where(
                    kph <= 345.0,
                    300000.0 - (kph - 340.0) * (100000.0 / 5.0),
                    torch.clamp(200000.0 - (kph - 345.0) * 20000.0, min=0.0),
                ),
            ),
        )
        return torch.where(override > 0.5, p_ov, p_norm)

    def predict_forces_and_next(
        self,
        # State [B]
        vx: torch.Tensor,
        vy: torch.Tensor,
        yaw_rate: torch.Tensor,
        steer_rad: torch.Tensor,
        rh_f_prev: torch.Tensor,
        rh_r_prev: torch.Tensor,
        fuel: torch.Tensor,
        temp_f: torch.Tensor,
        temp_r: torch.Tensor,
        wear: torch.Tensor,
        ers_soc: torch.Tensor,
        # Action [B]
        throttle: torch.Tensor,
        brake: torch.Tensor,
        ers_mode: torch.Tensor,
        xmode: torch.Tensor,
        override: torch.Tensor,
        # Track/Weather/Design [B]
        grade: torch.Tensor,
        a_vert: torch.Tensor,
        wet: torch.Tensor,
        off_track: torch.Tensor,
        fw: torch.Tensor,
        rw: torch.Tensor,
        rh_f_stat: torch.Tensor,
        rh_r_stat: torch.Tensor,
        ballast: torch.Tensor,
        rad: torch.Tensor,
        wf: torch.Tensor,
        sf: torch.Tensor,
        sr: torch.Tensor,
        bias_f: torch.Tensor,
        comp_grip: torch.Tensor,
        comp_t_opt: torch.Tensor,
        dt: float | torch.Tensor = 1.0 / 24.0,
        slipstream_cut: Optional[torch.Tensor] = None,
        wake_loss: Optional[torch.Tensor] = None,
        prev_glong: Optional[torch.Tensor] = None,
        rho: Optional[torch.Tensor] = None,
        duct: Optional[torch.Tensor] = None,
        harvest_level: Optional[torch.Tensor] = None,
        rpm: Optional[torch.Tensor] = None,
        *,
        tyre_surf_c=None, tyre_core_c=None, tyre_wear=None, brake_temp_c=None, alpha_eff=None,
        transfer_f=None, transfer_r=None, heave_v_f=None, heave_v_r=None, prev_glat=None,
        arb_front=None, arb_rear=None, damping_ratio=None,
        camber_front_deg=None, camber_rear_deg=None, pressure_front_psi=None, pressure_rear_psi=None,
        total_ratio=None, launch_active=None, air_temp_c=None, track_temp_c=None, comp_heat=None, comp_wear=None,
    ) -> Dict[str, torch.Tensor]:
        """One differentiable physical step, not a callback or full weekend.

        Per-wheel tensors are [B,4] FL/FR/RL/RR; all other inputs [B] or
        scalar dt. Missing latent state uses explicit neutral initial guesses;
        pass next fields back for multi-step fitting (including predicted g's).
        Supply the design gear*final_drive and launch latch for faithful inertia.
        Controls steer_rad and xmode represent *actuated* states.
        """
        z = torch.zeros_like(vx)
        def opt(a, default):
            return z + default if a is None else z + a

        def corners(a, default):
            return default if a is None else a.unbind(-1)

        dt = z + dt
        p = SimpleNamespace(**dict(zip(self.param_names, (
            self.lo + (self.hi - self.lo) * self.raw.sigmoid()).unbind())))
        surf = corners(tyre_surf_c, (temp_f, temp_f, temp_r, temp_r))
        core = corners(tyre_core_c, surf)
        wear4 = corners(tyre_wear, (wear,) * 4)
        discs = corners(brake_temp_c, (z + 500.,) * 4)
        alphas = corners(alpha_eff, (z,) * 4)
        zeta = opt(damping_ratio, .5)
        duct = opt(duct, .45)
        mass = 748.0 + ballast + torch.clamp(fuel, min=0.0)
        v = torch.sqrt(vx * vx + vy * vy + 1e-6)

        # 1) Dynamic ride height from aero + vertical curvature + braking pitch
        df0, _, f_sh0 = self.aero_forces(v, fw, rw, rh_f_prev, rh_r_prev, rad, xmode, duct=duct, rho=rho)
        if wake_loss is not None:
            df0 = df0 * (1.0 - wake_loss)
        k_f = 2.0 * (sf * 380.0) / torch.clamp(sf + 380.0, min=1.0) * self.p("k_rh_f_scale")
        k_r = 2.0 * (sr * 450.0) / torch.clamp(sr + 450.0, min=1.0) * self.p("k_rh_r_scale")
        gl0 = prev_glong if prev_glong is not None else torch.zeros_like(v)
        d_fz_pitch = mass * (gl0 * 9.80665) * 0.272 / 3.4 * self.p("pitch_rh_scale")
        extra_f = df0 * f_sh0 + mass * a_vert * wf - d_fz_pitch
        extra_r = df0 * (1.0 - f_sh0) + mass * a_vert * (1.0 - wf) + d_fz_pitch
        def heave(stat, prev, u, k, sprung_kg, force, static_weight):
            x = stat - prev
            m = sprung_kg / 1000.
            c = 2. * zeta * torch.sqrt(m * k)
            u0 = (m * u + dt * (force - k * x)) / (m + dt * c + dt * dt * k)
            limit = stat - 3.
            u_plank = ((m * u + dt * (force - k * x - 4000. * (x - limit))) /
                       (m + dt * c + dt * dt * (k + 4000.)))
            un = torch.where(x + dt * u0 > limit, u_plank, u0)
            xn = x + dt * un
            return stat - xn, un, (static_weight + k * xn + c * un).clamp_min(0.), 4000. * (xn - limit).clamp_min(0.)

        rh_f, uf, fz_f_ax, plank_f = heave(rh_f_stat, rh_f_prev, opt(heave_v_f, 0.), k_f,
            mass * wf - 48., extra_f, mass * 9.80665 * wf)
        rh_r, ur, fz_r_ax, plank_r = heave(rh_r_stat, rh_r_prev, opt(heave_v_r, 0.), k_r,
            mass * (1. - wf) - 52., extra_r, mass * 9.80665 * (1. - wf))

        df, drag, f_share = self.aero_forces(v, fw, rw, rh_f, rh_r, rad, xmode, duct=duct, rho=rho)
        if wake_loss is not None:
            df = df * (1.0 - wake_loss)
        if slipstream_cut is not None:
            drag = drag * (1.0 - slipstream_cut)

        roll = p.roll_res_c * (fz_f_ax + fz_r_ax)

        # 2) Powertrain (Official ICE RPM curve + Fixed Real-Server MGU-K Deploy/Regen!)
        if rpm is not None:
            xr = (rpm - 12500.0) / 3500.0
            eta_ice = torch.clamp(0.486 * (1.0 - 0.034 * xr * xr - 0.14 * (F.relu(-xr) ** 2)), min=0.25)
        else:
            eta_ice = 0.482
        ice_w = (3000e6 / 3600.0) * eta_ice * self.p("ice_power_scale") * throttle * 0.965 * (fuel > 0.0).float()

        max_m = self.mguk_power_w(vx, override)
        hl = harvest_level if harvest_level is not None else torch.full_like(v, 5.0)
        regen_cap = 70000.0 * torch.clamp(hl, 1.0, 5.0)

        is_brk = (brake > 0.05).float()
        is_m3 = (ers_mode >= 2.5).float()
        is_m2 = ((ers_mode >= 1.5) & (ers_mode < 2.5)).float()
        is_m0 = (ers_mode < 0.5).float()
        is_thr = (throttle > 0.05).float()
        has_soc = (ers_soc > 0.01).float()
        can_regen = ((vx > 10.0) & (ers_soc < 3.98)).float()

        # Deploy ONLY when not braking and throttle > 0.05 and mode in {2, 3}
        deploy_frac = torch.clamp(is_m3 + 0.5 * is_m2 + override * (1.0 - is_m3) * is_m2, 0.0, 1.0)
        ers_deploy_w = max_m * deploy_frac * throttle * WHEEL_EFF * has_soc * (1.0 - is_brk) * is_thr

        # Regen:
        # (a) Braking (ANY ers_mode!): FIA-C C11.6 rear brake-by-wire blends regen up to regen_cap
        brake_cap = [p.brake_max_n * p.brake_scale * .5 * (bias_f if i < 2 else 1. - bias_f) *
                     disc_friction(discs[i]) for i in range(4)]
        brk_req = [b * brake for b in brake_cap]
        regen_brk = torch.minimum(regen_cap, (brk_req[2] + brk_req[3]) * vx) * is_brk
        # (b) Mode 0 under throttle: harvests directly from ICE shaft proportional to throttle
        is_m0_harv = (1.0 - is_brk) * is_m0 * is_thr
        regen_m0_thr = regen_cap * throttle * is_m0_harv
        # (c) Coasting (brake <= 0.05 and throttle <= 0.05, ANY mode): lift-off MGU-K regen
        is_coast = (1.0 - is_brk) * (1.0 - is_thr)
        regen_coast = torch.minimum(
            regen_cap,
            mass * 0.04 * hl * 9.80665 * vx * self.p("lift_regen_scale"),
        ) * is_coast
        regen_w = (regen_brk + regen_m0_thr + regen_coast) * can_regen
        ers_w = (ers_deploy_w / WHEEL_EFF) - regen_w

        if total_ratio is None:
            # Legacy calls can infer the engaged ratio only in a locked clutch;
            # this fallback is not a substitute for the actual selected gear.
            total_ratio = z if rpm is None else torch.where(vx > 7., rpm * (math.pi * .343 / 30.) / vx.clamp_min(7.), z)
        eff_m = mass + (5. + .08 * total_ratio ** 2) / .343 ** 2
        launch = (opt(launch_active, 0.) > .5) & (vx < 30.)
        launch_next = launch.clone()
        f_drv = (ice_w + ers_deploy_w - regen_m0_thr * can_regen) / vx.clamp_min(7.)
        drive = torch.where(launch, torch.minimum(f_drv.clamp_min(0.), eff_m * 1.2 * 9.80665), f_drv.clamp_min(0.))
        coast = regen_coast * can_regen / vx.clamp_min(7.) + (-f_drv).clamp_min(0.)
        eng_brk = mass * 9.80665 * p.engine_brake_g * (1. - throttle / .15).clamp(0., 1.) * (vx > 8.)
        fx_req = [-brk_req[0], -brk_req[1]] + [
            .5 * (drive - coast - eng_brk) - brk_req[i] for i in (2, 3)]
        regen_brk_force = regen_brk * can_regen / vx.clamp_min(7.)
        hydraulic = list(brk_req[:2]) + [
            (brk_req[i] - regen_brk_force * brk_req[i] / (brk_req[2] + brk_req[3]).clamp_min(1.e-12)).clamp_min(0.)
            for i in (2, 3)]
        wheels = wheel_dynamics(p, mass=mass, wf=wf, vx=vx, vy=vy, yaw_rate=yaw_rate, steer_rad=steer_rad,
            dt=dt, prev_glat=opt(prev_glat, 0.), transfer_f=opt(transfer_f, 0.), transfer_r=opt(transfer_r, 0.),
            fz_f_ax=fz_f_ax, fz_r_ax=fz_r_ax, sf=sf, sr=sr, arb_front=opt(arb_front, 5.), arb_rear=opt(arb_rear, 5.),
            damping_ratio=zeta, camber_front_deg=opt(camber_front_deg, -2.5), camber_rear_deg=opt(camber_rear_deg, -1.5),
            pressure_front_psi=opt(pressure_front_psi, 29.), pressure_rear_psi=opt(pressure_rear_psi, 28.),
            tyre_surf_c=surf, tyre_core_c=core, tyre_wear=wear4, alpha_eff=alphas, wet=wet, off_track=off_track,
            comp_grip=comp_grip, comp_t_opt=comp_t_opt, fx_request=fx_req, hydraulic_request=hydraulic,
            harvest_level=hl * ((regen_brk + regen_coast) * can_regen > 0.))
        therm = wheel_thermal(p, wheels=wheels, tyre_surf_c=surf, tyre_core_c=core, tyre_wear=wear4, brake_temp_c=discs,
            dt=dt, v=v, wet=wet, air_temp_c=opt(air_temp_c, 24.), track_temp_c=opt(track_temp_c, 30.),
            duct=duct, comp_heat=opt(comp_heat, 1.), comp_wear=opt(comp_wear, 1.), comp_t_opt=comp_t_opt)
        offdrag = mass * 9.80665 * p.offtrack_drag_g * (off_track > .5) * (vx > 6.)
        ax_b = (wheels['fx_body'] - drag - roll - offdrag - mass * 9.80665 * grade) / eff_m
        ay_b = wheels['fy_body'] / mass
        iz = mass * (1.15 ** 2) * p.iz_scale
        vx_next = (vx + (ax_b + yaw_rate * vy) * dt).clamp_min(0.)
        vy_dyn = vy + (ay_b - yaw_rate * vx) * dt
        r_dyn = yaw_rate + wheels['mz_body'] / iz * dt
        # A spinning car can have vx near zero yet carry substantial lateral velocity.
        # Select the near-static bicycle branch on provisional *ground* speed.
        low_v = torch.hypot(vx_next, vy_dyn) < 5.
        vy_next = torch.where(low_v, z, vy_dyn)
        r_next = torch.where(low_v, vx_next * torch.tan(steer_rad) / 3.4, r_dyn)
        launch_next = launch_next & (vx_next < 30.)
        burn_kg_s = (3000e6 / 3600.0 / 44.0e6) * throttle * self.p("fuel_burn_scale")
        fuel_next = torch.clamp(fuel - burn_kg_s * dt, min=0.0)
        ers_soc_next = torch.clamp(ers_soc - (ers_deploy_w / WHEEL_EFF - regen_w * 0.97) * dt / 1e6, 0.0, 4.0)

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
            "front_sliding": wheels["front_sliding"],
            "rear_sliding": wheels["rear_sliding"],
            "front_share": f_share,
            "ay_max": wheels["ay_max"],
            "launch_active_next": launch_next,
            "effective_mass_kg": eff_m,
            "brake_force_max_n": sum(brake_cap),
            "heave_v_f_next": uf, "heave_v_r_next": ur,
            "plank_front_n": plank_f, "plank_rear_n": plank_r,
            "transfer_f_next": wheels["transfer_f_next"], "transfer_r_next": wheels["transfer_r_next"],
            **{k: torch.stack(wheels[k], -1) for k in ("fz", "fx", "fy", "util", "pressure_psi", "camber_deg", "alpha_eff_next", "hydraulic_force")},
            **{k: torch.stack(vv, -1) for k, vv in therm.items()},
        }


class TrackGeometryGPU:
    """GPU-resident lookup tables for fast vectorized Frenet queries across 2,048 parallel cars."""

    def __init__(self, geom_cpu: Any, device: torch.device):
        self.track_id = geom_cpu.track_id
        self.length = float(geom_cpu.length)
        self.ds = float(geom_cpu.ds)
        self.n_nodes = int(geom_cpu.n_nodes)
        self.pit_entry_s = float(geom_cpu.pit_entry_s)
        self.pit_exit_s = float(geom_cpu.pit_exit_s)
        self.pit_lane_length_m = float(geom_cpu.pit_lane_length_m)
        self.pit_speed_mps = float(getattr(geom_cpu, "pit_speed_mps", 80.0 / 3.6))
        self.aid_detection_s = float(geom_cpu.aid_detection_s)

        def _t(arr: np.ndarray) -> torch.Tensor:
            return torch.tensor(np.asarray(arr, dtype=np.float32), device=device)

        self.cx = _t(geom_cpu.x)
        self.cy = _t(geom_cpu.y)
        self.nx = _t(geom_cpu.nx)
        self.ny = _t(geom_cpu.ny)
        self.curvature = _t(geom_cpu.curvature)
        self.grade = _t(geom_cpu.grade)
        self.left = _t(geom_cpu.left)
        self.right = _t(geom_cpu.right)
        self.wall_left = _t(geom_cpu.wall_left)
        self.wall_right = _t(geom_cpu.wall_right)
        self.xmode = _t(geom_cpu.xmode)
        self.aero_rem = _t(geom_cpu.aero_remaining)
        # Precompute vertical grade derivative dg/ds (for Eau Rouge compression!)
        g_p = torch.roll(self.grade, -1)
        g_m = torch.roll(self.grade, 1)
        self.dgrade_ds = (g_p - g_m) / (2.0 * self.ds)

    def interp(self, arr: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        u = torch.remainder(s, self.length) / self.ds
        i0 = torch.floor(u).long() % self.n_nodes
        i1 = (i0 + 1) % self.n_nodes
        f = u - torch.floor(u)
        return arr[i0] + (arr[i1] - arr[i0]) * f
