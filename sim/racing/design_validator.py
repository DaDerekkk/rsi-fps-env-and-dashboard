"""Exact AGX Formula (rsi-racing-4.4) design.json validator.

Implements every field range, grid step, enum, cross-field constraint, and
all-3-tracks (parkland, highland, crossover) technical regulation from
work/racing-bootstrap/docs/racing-regs.zh.md and racing-starter.zh.md.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DESIGN_BOOK_MAX_BYTES = 4096
DESIGN_ENVELOPE_MAX_BYTES = 8320

TRACK_LIMITS: Dict[str, Dict[str, float]] = {
    "crossover": {
        "pressure_front_min": 28.0,
        "pressure_rear_min": 26.0,
        "camber_front_min": -2.75,
        "camber_rear_min": -1.5,
    },
    "highland": {
        "pressure_front_min": 28.0,
        "pressure_rear_min": 28.0,
        "camber_front_min": -2.75,
        "camber_rear_min": -1.5,
    },
    "parkland": {
        "pressure_front_min": 26.0,
        "pressure_rear_min": 26.0,
        "camber_front_min": -3.0,
        "camber_rear_min": -2.0,
    },
}

GEAR_MIN = [3.4, 2.45, 1.95, 1.6, 1.35, 1.15, 1.0, 0.86]
GEAR_MAX = [4.8, 3.5, 2.7, 2.2, 1.85, 1.55, 1.3, 1.12]

TOP_LEVEL_KEYS = {
    "schema",
    "aero",
    "ride_height_mm",
    "suspension",
    "mass",
    "gearbox",
    "brakes",
    "cooling",
    "power_unit",
    "tyres",
}


def _is_number(val: Any) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(float(val))


def _is_int(val: Any) -> bool:
    return isinstance(val, int) and not isinstance(val, bool)


def _check_grid(
    violations: List[str],
    path: str,
    val: Any,
    lo: float,
    hi: float,
    step: float,
    integer: bool = False,
) -> Optional[float]:
    if integer:
        if not _is_int(val):
            violations.append(f"{path}: expected integer, got {type(val).__name__}")
            return None
    else:
        if not _is_number(val):
            violations.append(f"{path}: expected finite number, got {type(val).__name__}")
            return None
    f = float(val)
    if f < lo - 1e-9 or f > hi + 1e-9:
        violations.append(f"{path}: value {f} outside [{lo}, {hi}]")
        return f
    steps = (f - lo) / step
    if abs(steps - round(steps)) > 1e-6:
        violations.append(f"{path}: value {f} not on grid min={lo} step={step}")
    return f


def _check_keys(violations: List[str], prefix: str, obj: Any, expected: set[str]) -> bool:
    if not isinstance(obj, dict):
        violations.append(f"{prefix}: expected object, got {type(obj).__name__}")
        return False
    actual = set(obj.keys())
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        violations.append(f"{prefix}: missing keys {missing}")
    if extra:
        violations.append(f"{prefix}: unexpected keys {extra}")
    return not missing and not extra


def validate_design_book(
    book: Dict[str, Any],
    tracks: Tuple[str, ...] = ("parkland", "highland", "crossover"),
    prefix: str = "design",
) -> List[str]:
    violations: List[str] = []
    if not isinstance(book, dict):
        return [f"{prefix}: expected JSON object"]
    raw_bytes = len(json.dumps(book, allow_nan=False, separators=(",", ":")).encode("utf-8"))
    if raw_bytes > DESIGN_BOOK_MAX_BYTES:
        violations.append(f"{prefix}: UTF-8 byte length {raw_bytes} > {DESIGN_BOOK_MAX_BYTES}")

    _check_keys(violations, prefix, book, TOP_LEVEL_KEYS)
    if book.get("schema") != "formula-v4-design-1":
        violations.append(f"{prefix}.schema: expected 'formula-v4-design-1', got {book.get('schema')!r}")

    # Aero
    aero = book.get("aero")
    if _check_keys(violations, f"{prefix}.aero", aero, {"front_wing", "rear_wing"}):
        _check_grid(violations, f"{prefix}.aero.front_wing", aero["front_wing"], 10.0, 30.0, 0.5)
        _check_grid(violations, f"{prefix}.aero.rear_wing", aero["rear_wing"], 10.0, 32.0, 0.5)

    # Ride height
    rh = book.get("ride_height_mm")
    if _check_keys(violations, f"{prefix}.ride_height_mm", rh, {"front", "rear"}):
        rh_f = _check_grid(violations, f"{prefix}.ride_height_mm.front", rh["front"], 25.0, 55.0, 1.0)
        rh_r = _check_grid(violations, f"{prefix}.ride_height_mm.rear", rh["rear"], 50.0, 100.0, 1.0)
        if rh_f is not None and rh_r is not None:
            rake = rh_r - rh_f
            if rake < -1e-9 or rake > 70.0 + 1e-9:
                violations.append(f"{prefix}.ride_height_mm: rake rear-front={rake} outside [0, 70]")

    # Suspension
    susp = book.get("suspension")
    susp_keys = {
        "spring_front_n_per_mm",
        "spring_rear_n_per_mm",
        "arb_front",
        "arb_rear",
        "damping",
        "camber_front_deg",
        "camber_rear_deg",
        "toe_front_deg",
        "toe_rear_deg",
    }
    camber_f = camber_r = None
    if _check_keys(violations, f"{prefix}.suspension", susp, susp_keys):
        _check_grid(violations, f"{prefix}.suspension.spring_front_n_per_mm", susp["spring_front_n_per_mm"], 120.0, 420.0, 5.0)
        _check_grid(violations, f"{prefix}.suspension.spring_rear_n_per_mm", susp["spring_rear_n_per_mm"], 100.0, 380.0, 5.0)
        _check_grid(violations, f"{prefix}.suspension.arb_front", susp["arb_front"], 1, 10, 1, integer=True)
        _check_grid(violations, f"{prefix}.suspension.arb_rear", susp["arb_rear"], 1, 10, 1, integer=True)
        if susp["damping"] not in ("soft", "medium", "stiff"):
            violations.append(f"{prefix}.suspension.damping: invalid enum {susp['damping']!r}")
        camber_f = _check_grid(violations, f"{prefix}.suspension.camber_front_deg", susp["camber_front_deg"], -3.25, -1.0, 0.05)
        camber_r = _check_grid(violations, f"{prefix}.suspension.camber_rear_deg", susp["camber_rear_deg"], -2.0, -0.5, 0.05)
        _check_grid(violations, f"{prefix}.suspension.toe_front_deg", susp["toe_front_deg"], -0.3, 0.1, 0.01)
        _check_grid(violations, f"{prefix}.suspension.toe_rear_deg", susp["toe_rear_deg"], 0.0, 0.6, 0.01)

    # Mass
    mass = book.get("mass")
    if _check_keys(violations, f"{prefix}.mass", mass, {"ballast_kg", "weight_front_pct"}):
        bal = _check_grid(violations, f"{prefix}.mass.ballast_kg", mass["ballast_kg"], 20.0, 60.0, 0.5)
        wf = _check_grid(violations, f"{prefix}.mass.weight_front_pct", mass["weight_front_pct"], 44.0, 46.0, 0.1)
        if bal is not None and wf is not None:
            dry_mass = 748.0 + bal
            if dry_mass < 768.0 - 1e-9:
                violations.append(f"{prefix}.mass: dry_mass={dry_mass} < 768.0")
            if dry_mass * wf / 100.0 < 768.0 * 0.44 - 1e-9:
                violations.append(f"{prefix}.mass: front_axle_kg below minimum")
            if dry_mass * (100.0 - wf) / 100.0 < 768.0 * 0.54 - 1e-9:
                violations.append(f"{prefix}.mass: rear_axle_kg below minimum")
            min_bal = 20.0 + 8.2 * abs(wf - 45.0)
            if bal < min_bal - 1e-9:
                violations.append(f"{prefix}.mass.ballast_kg: {bal} < required {min_bal:.4f} for weight_front_pct={wf}")

    # Gearbox
    gb = book.get("gearbox")
    if _check_keys(violations, f"{prefix}.gearbox", gb, {"ratios", "final_drive"}):
        ratios = gb["ratios"]
        fd = _check_grid(violations, f"{prefix}.gearbox.final_drive", gb["final_drive"], 4.4, 6.6, 0.01)
        if not isinstance(ratios, list) or len(ratios) != 8:
            violations.append(f"{prefix}.gearbox.ratios: expected list of length 8")
        else:
            parsed: List[float] = []
            for idx, r in enumerate(ratios):
                rv = _check_grid(violations, f"{prefix}.gearbox.ratios[{idx}]", r, GEAR_MIN[idx], GEAR_MAX[idx], 0.001)
                if rv is not None:
                    parsed.append(rv)
            if len(parsed) == 8:
                for i in range(7):
                    if parsed[i] <= parsed[i + 1]:
                        violations.append(f"{prefix}.gearbox.ratios: must strictly decrease at index {i}")
                    else:
                        step_ratio = parsed[i] / parsed[i + 1]
                        if step_ratio < 1.08 - 1e-9 or step_ratio > 1.45 + 1e-9:
                            violations.append(f"{prefix}.gearbox.ratios[{i}]/[{i+1}]={step_ratio:.4f} outside [1.08, 1.45]")
                if fd is not None and parsed[0] * fd > 25.1 + 1e-9:
                    violations.append(f"{prefix}.gearbox: ratios[0]*final_drive={parsed[0]*fd:.4f} > 25.1")

    # Brakes
    brakes = book.get("brakes")
    if _check_keys(violations, f"{prefix}.brakes", brakes, {"bias_front_pct", "duct"}):
        _check_grid(violations, f"{prefix}.brakes.bias_front_pct", brakes["bias_front_pct"], 50.0, 64.0, 0.1)
        _check_grid(violations, f"{prefix}.brakes.duct", brakes["duct"], 0.2, 1.0, 0.05)

    # Cooling
    cool = book.get("cooling")
    if _check_keys(violations, f"{prefix}.cooling", cool, {"radiator_opening"}):
        _check_grid(violations, f"{prefix}.cooling.radiator_opening", cool["radiator_opening"], 0.25, 1.0, 0.05)

    # Power unit
    pu = book.get("power_unit")
    if _check_keys(violations, f"{prefix}.power_unit", pu, {"fuel_load_kg", "ers_harvest_level"}):
        _check_grid(violations, f"{prefix}.power_unit.fuel_load_kg", pu["fuel_load_kg"], 5.0, 100.0, 0.5)
        _check_grid(violations, f"{prefix}.power_unit.ers_harvest_level", pu["ers_harvest_level"], 0, 5, 1, integer=True)

    # Tyres
    tyres = book.get("tyres")
    pf = pr = None
    if _check_keys(violations, f"{prefix}.tyres", tyres, {"start_compound", "pressure_front_psi", "pressure_rear_psi"}):
        if tyres["start_compound"] not in ("soft", "medium", "hard"):
            violations.append(f"{prefix}.tyres.start_compound: invalid enum {tyres['start_compound']!r}")
        pf = _check_grid(violations, f"{prefix}.tyres.pressure_front_psi", tyres["pressure_front_psi"], 24.0, 32.0, 0.1)
        pr = _check_grid(violations, f"{prefix}.tyres.pressure_rear_psi", tyres["pressure_rear_psi"], 24.0, 32.0, 0.1)

    # Per-track regulations (sealed design must satisfy ALL tracks in competition rotation!)
    for tid in tracks:
        lim = TRACK_LIMITS[tid]
        if pf is not None and pf < lim["pressure_front_min"] - 1e-9:
            violations.append(f"{prefix}.tyres.pressure_front_psi ({pf}) < {tid} min {lim['pressure_front_min']}")
        if pr is not None and pr < lim["pressure_rear_min"] - 1e-9:
            violations.append(f"{prefix}.tyres.pressure_rear_psi ({pr}) < {tid} min {lim['pressure_rear_min']}")
        if camber_f is not None and camber_f < lim["camber_front_min"] - 1e-9:
            violations.append(f"{prefix}.suspension.camber_front_deg ({camber_f}) < {tid} min {lim['camber_front_min']}")
        if camber_r is not None and camber_r < lim["camber_rear_min"] - 1e-9:
            violations.append(f"{prefix}.suspension.camber_rear_deg ({camber_r}) < {tid} min {lim['camber_rear_min']}")

    return violations


def validate_design_envelope(
    raw_bytes: bytes,
    tracks: Tuple[str, ...] = ("parkland", "highland", "crossover"),
) -> Tuple[bool, List[str], Optional[Dict[str, Any]]]:
    violations: List[str] = []
    if len(raw_bytes) > DESIGN_ENVELOPE_MAX_BYTES:
        violations.append(f"envelope: byte length {len(raw_bytes)} > {DESIGN_ENVELOPE_MAX_BYTES}")

    def _reject_dup_keys(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in pairs:
            if k in out:
                violations.append(f"duplicate JSON key: {k!r}")
            out[k] = v
        return out

    try:
        obj = json.loads(raw_bytes.decode("utf-8"), object_pairs_hook=_reject_dup_keys)
    except Exception as exc:
        return False, [f"invalid JSON: {exc}"], None

    if not isinstance(obj, dict):
        return False, ["envelope must be a JSON object"], None
    allowed = {"design", "rain_design"}
    extra = sorted(set(obj.keys()) - allowed)
    if extra:
        violations.append(f"envelope: unexpected top-level keys {extra}")
    if "design" not in obj:
        violations.append("envelope: missing required 'design' object")
    else:
        violations.extend(validate_design_book(obj["design"], tracks=tracks, prefix="design"))
    if "rain_design" in obj:
        if obj["rain_design"] is None:
            violations.append("envelope: rain_design cannot be null; omit if unused")
        else:
            violations.extend(validate_design_book(obj["rain_design"], tracks=tracks, prefix="rain_design"))

    return len(violations) == 0, violations, obj


def snap_to_grid(val: float, lo: float, hi: float, step: float) -> float:
    clamped = max(lo, min(hi, float(val)))
    n = round((clamped - lo) / step)
    res = lo + n * step
    # Clean IEEE floating point noise
    return round(res, 6)


if __name__ == "__main__":
    import sys

    p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("work/racing-bootstrap/official-api-template/design.json")
    ok, errs, _ = validate_design_envelope(p.read_bytes())
    print(json.dumps({"path": str(p), "ok": ok, "violations": errs}, indent=2, ensure_ascii=False))
    sys.exit(0 if ok else 1)
