"""Strictly read-only (GET-only) AGX FPS live dashboard client & replay parser.

Credentials are read on demand from macOS Keychain via /usr/bin/security and
never logged, saved, returned to HTTP clients, or used for any write/mutation.
"""
from __future__ import annotations

import collections
import concurrent.futures
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import http.client
import io
import json
import math
import os
from pathlib import Path
import re
import ssl
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

DASH_DIR = Path(__file__).resolve().parent
if str(DASH_DIR) not in sys.path:
    sys.path.insert(0, str(DASH_DIR))
from replay_codec import decode_chunk  # noqa: E402

TZ = timezone(timedelta(hours=8))
COMPETITION_ID = os.environ.get("FPS_COMPETITION_ID", "2a97f94f-816a-52b7-b4b8-5925db160245")
RACING_COMPETITION_ID = os.environ.get("RACING_COMPETITION_ID", "1eb91364-950b-5725-aa4b-3501b16860ec")
KEYCHAIN_SERVICE = os.environ.get("FPS_KEYCHAIN_SERVICE", "agx.cool.fps.api")
ROLES_PATH = Path(os.environ.get("FPS_DASHBOARD_ACCOUNTS", str(DASH_DIR / "accounts.json")))
CACHE_ROOT = Path(os.environ.get("FPS_DASHBOARD_CACHE", str(DASH_DIR / ".cache")))
STATS_CACHE_DIR = CACHE_ROOT / "stats"
REPLAY_CACHE_DIR = CACHE_ROOT / "replays"
STATE_SNAPSHOT_PATH = CACHE_ROOT / "latest_state.json"
VERSION_BINDINGS_PATH = CACHE_ROOT / "version_bindings.json"

_REFRESH_LOCK = threading.Lock()
_REPLAY_LOCKS_GUARD = threading.Lock()
_REPLAY_LOCKS: dict[str, threading.Lock] = {}
_SSL_CTX = ssl.create_default_context()
_PROGRESSIVE_GUARD = threading.Lock()
_PROGRESSIVE_JOBS: dict[str, dict] = {}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def now_iso() -> str:
    return datetime.now(TZ).isoformat()


def _get_replay_lock(job_id: str) -> threading.Lock:
    with _REPLAY_LOCKS_GUARD:
        lock = _REPLAY_LOCKS.get(job_id)
        if lock is None:
            lock = threading.Lock()
            _REPLAY_LOCKS[job_id] = lock
        return lock


def _read_keychain(alias: str) -> str:
    """Read API token from env (`AGX_TOKEN_<ALIAS>`) first, falling back to macOS Keychain."""
    env_key = "AGX_TOKEN_" + re.sub(r"[^A-Za-z0-9]+", "_", alias).upper().strip("_")
    env_tok = os.environ.get(env_key, "").strip()
    if env_tok:
        return env_tok
    out = subprocess.run(
        ["/usr/bin/security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-a", alias, "-w"],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    return out.stdout.strip()


def agx_get_json(path: str, alias: str | None = None, timeout: int = 25) -> dict:
    """Strictly GET-only JSON request to https://agx.cool/api/v1/..."""
    if not path.startswith("/api/v1/") or "://" in path:
        raise ValueError(f"Disallowed API path: {path}")
    headers = {"Accept": "application/json", "Accept-Encoding": "gzip"}
    if alias is not None:
        token = _read_keychain(alias)
        headers["Authorization"] = f"Bearer {token}"
        del token
    req = urllib.request.Request("https://agx.cool" + path, method="GET", headers=headers)
    try:
        with _OPENER.open(req, timeout=timeout) as resp:
            raw = resp.read(16 * 1024 * 1024 + 1)
            if len(raw) > 16 * 1024 * 1024:
                raise ValueError("Response exceeded 16 MiB budget")
            if resp.headers.get("Content-Encoding") == "gzip":
                raw = gzip.decompress(raw)
            return {"http_status": resp.status, "body": json.loads(raw.decode("utf-8"))}
    except urllib.error.HTTPError as err:
        try:
            body = json.loads(err.read(1024 * 1024).decode("utf-8", errors="replace"))
        except Exception:
            body = {"error": f"HTTP {err.code}"}
        return {"http_status": err.code, "body": body}
    except Exception as exc:
        return {"http_status": None, "error": type(exc).__name__, "body": None}


class _KeepAliveClient:
    """Thread-local persistent HTTPS/1.1 Keep-Alive client for strictly GET-only public chunks."""

    def __init__(self, timeout: int = 30) -> None:
        self.timeout = timeout
        self._conn: http.client.HTTPSConnection | None = None

    def _connect(self) -> http.client.HTTPSConnection:
        if self._conn is None:
            self._conn = http.client.HTTPSConnection("agx.cool", 443, timeout=self.timeout, context=_SSL_CTX)
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def get_public_bytes(self, path: str, max_bytes: int = 16 * 1024 * 1024) -> tuple[bytes, dict]:
        if not path.startswith("/api/v1/") or "://" in path:
            raise ValueError(f"Disallowed public path: {path}")
        last_err: Exception | None = None
        for attempt in range(1, 4):
            try:
                conn = self._connect()
                conn.request("GET", path, headers={"Accept-Encoding": "gzip", "Connection": "keep-alive"})
                resp = conn.getresponse()
                status = resp.status
                headers = dict(resp.getheaders())
                data = resp.read(max_bytes + 1025)
                if status != 200:
                    if status in (429, 500, 502, 503, 504) and attempt < 3:
                        self.close()
                        time.sleep(0.6 * attempt)
                        continue
                    raise urllib.error.HTTPError("https://agx.cool" + path, status, resp.reason, headers, None)
                if len(data) > max_bytes + 1024:
                    raise ValueError("Public chunk exceeded transport budget")
                if headers.get("Content-Encoding") == "gzip" or headers.get("content-encoding") == "gzip":
                    with gzip.GzipFile(fileobj=io.BytesIO(data)) as zf:
                        data = zf.read(max_bytes + 1)
                codec_headers = {k: v for k, v in headers.items() if k.lower().startswith("x-agx-")}
                if codec_headers:
                    data = decode_chunk(data, codec_headers)
                if len(data) > max_bytes:
                    raise ValueError("Decoded public chunk exceeded raw budget")
                return data, headers
            except urllib.error.HTTPError:
                raise
            except Exception as err:
                last_err = err
                self.close()
                if attempt == 3:
                    raise
                time.sleep(0.5 * attempt)
        raise last_err or RuntimeError("Public chunk fetch failed")


def agx_get_public_bytes(path: str, max_bytes: int = 16 * 1024 * 1024, timeout: int = 45) -> tuple[bytes, dict]:
    """Strictly GET-only public replay stream fetch (no auth header) with bounded transient GET retry."""
    cli = _KeepAliveClient(timeout=timeout)
    try:
        return cli.get_public_bytes(path, max_bytes=max_bytes)
    finally:
        cli.close()


def load_account_roles() -> list[dict]:
    data = json.loads(ROLES_PATH.read_text(encoding="utf-8"))
    accounts = []
    for item in data.get("accounts", []):
        agent_ids = item.get("fps_agent_ids") or []
        if not agent_ids:
            continue
        accounts.append(
            {
                "alias": item["alias"],
                "username": item.get("username") or item.get("display_name") or item["alias"],
                "short_name": item.get("short_username") or item.get("display_name") or item["alias"],
                "role": item.get("role", "secondary"),
                "label": item.get("label") or item.get("display_name") or item["alias"],
                "agent_id": agent_ids[0],
                "purpose": item.get("purpose", ""),
            }
        )
    return accounts


def _safe_atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{hashlib.sha256(now_iso().encode()).hexdigest()[:8]}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _finite(val) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val)


def _coord3(val) -> list[float] | None:
    if isinstance(val, list) and len(val) >= 3 and all(_finite(x) for x in val[:3]):
        return [round(float(val[0]), 2), round(float(val[1]), 2), round(float(val[2]), 2)]
    return None


def _download_chunks_keepalive(
    job_id: str,
    desc: dict,
    stream_name: str,
    indices: list[int],
    clients: list[_KeepAliveClient],
) -> list[bytes]:
    """Download a batch of chunk indices using a pool of persistent Keep-Alive clients."""
    max_chunk = int(desc.get("max_chunk_bytes", 1048576))
    template = desc.get("chunk_url_template") or f"/api/v1/python-submissions/recordings/{job_id}/replay/streams/{{stream}}/chunks/{{index}}"
    advertised = (desc.get("static", {}).get("streams", {}) or {}).get(stream_name)
    results: dict[int, bytes] = {}
    n_workers = min(len(clients), max(1, len(indices)))
    buckets = [indices[w::n_workers] for w in range(n_workers)]

    def _run_bucket(w_idx: int) -> None:
        cli = clients[w_idx]
        for idx in buckets[w_idx]:
            path = template.format(stream=stream_name, index=idx) + "?encoding=agx-v1"
            raw, _ = cli.get_public_bytes(path, max_bytes=max_chunk)
            if advertised and idx < len(advertised):
                if hashlib.sha256(raw).hexdigest() != advertised[idx]:
                    raise ValueError(f"Chunk digest mismatch for {stream_name}/{idx}")
            results[idx] = raw

    if n_workers == 1:
        _run_bucket(0)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
            list(pool.map(_run_bucket, range(n_workers)))
    return [results[i] for i in indices]


def _download_stream_bytes(job_id: str, desc: dict, stream_name: str, max_workers: int = 12) -> bytes:
    meta = desc.get("streams", {}).get(stream_name)
    if not meta:
        raise ValueError(f"Missing stream {stream_name} in descriptor")
    n_chunks = int(meta["chunks"])
    if n_chunks <= 0:
        return b""
    w = min(max_workers, n_chunks)
    clients = [_KeepAliveClient(timeout=30) for _ in range(w)]
    try:
        parts = _download_chunks_keepalive(job_id, desc, stream_name, list(range(n_chunks)), clients)
        return b"".join(parts)
    finally:
        for c in clients:
            c.close()


def parse_match_stats_from_streams(job_id: str, static_obj: dict, events: list[dict]) -> dict:
    """Compute full 10-player stats, dual-phase Attack(T)/Defense(CT) stats, and round ledger."""
    map_raw = static_obj.get("map") or {}
    map_id = map_raw.get("id") or "unknown"
    physics_hz = float(static_obj.get("physics_hz") or 60)
    frame_hz = float(static_obj.get("frame_hz") or 24)
    static_players = static_obj.get("players") or static_obj.get("participants") or []
    seats = sorted({int(p["seat"]) for p in static_players if "seat" in p}) or list(range(10))

    starts: dict[int, dict] = {}
    ends: dict[int, dict] = {}
    for e in events:
        et = e.get("type")
        ri = e.get("round_index")
        if ri is None:
            continue
        ri = int(ri)
        if et == "round_start":
            starts[ri] = e
        elif et == "round_end":
            ends[ri] = e

    all_ris = sorted(set(starts.keys()) | set(ends.keys()))
    init_att = int(starts[all_ris[0]].get("attack_team", 0)) if all_ris and all_ris[0] in starts else 0

    def _empty_bucket() -> dict:
        return {
            "rounds": 0,
            "wins": 0,
            "losses": 0,
            "k": 0,
            "d": 0,
            "a": 0,
            "dmg": 0.0,
            "fires": 0,
            "hits": 0,
            "hs": 0,
            "plants": 0,
            "defuses": 0,
            "faults": 0,
            "untraded_deaths": 0,
            "kast_rounds": 0,
        }

    seat_overall = {s: _empty_bucket() for s in seats}
    seat_attack = {s: _empty_bucket() for s in seats}
    seat_defend = {s: _empty_bucket() for s in seats}

    # Pre-group events by round
    events_by_round: dict[int, list[dict]] = collections.defaultdict(list)
    for e in events:
        ri = e.get("round_index")
        if ri is not None:
            events_by_round[int(ri)].append(e)

    rounds_summary = []
    fault_details = []
    plant_detail_seen = {
        (int(e["round_index"]), int(e["match_tick"]), e["type"].removesuffix("_detail"))
        for e in events
        if e.get("type") in ("plant_complete_detail", "defuse_complete_detail") and e.get("round_index") is not None
    }

    for ri in all_ris:
        st = starts.get(ri, {})
        en = ends.get(ri, {})
        att_team = int(st.get("attack_team", init_att if ri < 12 else 1 - init_att))
        winner = en.get("round_winner")
        reason = en.get("reason", "")
        score = en.get("round_wins")
        live_tick = int(st.get("buy_freeze_end_tick", st.get("match_tick", 0)))
        end_tick = int(en.get("match_tick", live_tick))
        dur_s = round(max(0.0, (end_tick - live_tick) / physics_hz), 1)

        r_events = sorted(events_by_round.get(ri, []), key=lambda x: (x.get("match_tick", 0), x.get("event_id", 0)))
        kills_in_round = []
        r_seat = {
            s: {"k": 0, "d": 0, "a": 0, "dmg": 0.0, "fires": 0, "hits": 0, "hs": 0, "faults": 0, "plant": 0, "defuse": 0}
            for s in seats
        }
        bomb_site = None
        planted_by = None
        defused_by = None

        for e in r_events:
            et = e.get("type")
            tick = int(e.get("match_tick", 0))
            if et == "fire":
                p = e.get("player")
                if p in r_seat:
                    r_seat[p]["fires"] += 1
            elif et == "hit":
                p = e.get("player")
                tgt = e.get("target")
                if p in r_seat and tgt is not None and (p < 5) != (tgt < 5):
                    r_seat[p]["hits"] += 1
                    r_seat[p]["dmg"] += float(e.get("hp_damage", 0.0))
                    if e.get("part") == "head":
                        r_seat[p]["hs"] += 1
            elif et == "kill":
                killer = e.get("killer")
                tgt = e.get("target")
                assists = e.get("assists") or []
                if killer in r_seat and killer != tgt:
                    r_seat[killer]["k"] += 1
                if tgt in r_seat:
                    r_seat[tgt]["d"] += 1
                for a_seat in assists:
                    if a_seat in r_seat and a_seat != killer and a_seat != tgt:
                        r_seat[a_seat]["a"] += 1
                kills_in_round.append(
                    {
                        "tick": tick,
                        "t": round((tick - live_tick) / physics_hz, 1),
                        "killer": killer,
                        "target": tgt,
                        "assists": assists,
                        "weapon": e.get("weapon") or "",
                        "hs": e.get("part") == "head",
                    }
                )
            elif et in ("plant_complete_detail", "plant_complete"):
                if et == "plant_complete" and (ri, tick, "plant_complete") in plant_detail_seen:
                    continue
                p = e.get("player", e.get("seat"))
                if p in r_seat:
                    r_seat[p]["plant"] += 1
                planted_by = p
                bomb_site = e.get("site") or bomb_site
            elif et in ("defuse_complete_detail", "defuse_complete"):
                if et == "defuse_complete" and (ri, tick, "defuse_complete") in plant_detail_seen:
                    continue
                p = e.get("player", e.get("seat"))
                if p in r_seat:
                    r_seat[p]["defuse"] += 1
                defused_by = p
                bomb_site = e.get("site") or bomb_site
            elif et == "fault":
                p = e.get("seat", e.get("player"))
                if p in r_seat:
                    r_seat[p]["faults"] += 1
                fault_details.append(
                    {
                        "round": ri + 1,
                        "tick": tick,
                        "seat": p,
                        "code": e.get("code") or e.get("reason") or "fault",
                    }
                )

        # Compute traded vs untraded deaths (traded if victim's teammate kills the killer within 4s = 240 ticks)
        trade_window = int(4.0 * physics_hz)
        traded_victims = set()
        for idx_k, k_ev in enumerate(kills_in_round):
            victim = k_ev["target"]
            killer = k_ev["killer"]
            if victim is None or killer is None or victim == killer:
                continue
            v_team = 0 if victim < 5 else 1
            for later in kills_in_round[idx_k + 1 :]:
                if later["tick"] - k_ev["tick"] > trade_window:
                    break
                if later["target"] == killer and later["killer"] is not None and (0 if later["killer"] < 5 else 1) == v_team:
                    traded_victims.add(victim)
                    break

        for s in seats:
            team = 0 if s < 5 else 1
            is_att = team == att_team
            rs = r_seat[s]
            won = winner == team
            kast = (rs["k"] > 0) or (rs["a"] > 0) or (rs["d"] == 0) or (s in traded_victims)
            untraded = 1 if (rs["d"] > 0 and s not in traded_victims) else 0
            for b in (seat_overall[s], seat_attack[s] if is_att else seat_defend[s]):
                b["rounds"] += 1
                if won:
                    b["wins"] += 1
                elif winner is not None:
                    b["losses"] += 1
                b["k"] += rs["k"]
                b["d"] += rs["d"]
                b["a"] += rs["a"]
                b["dmg"] += rs["dmg"]
                b["fires"] += rs["fires"]
                b["hits"] += rs["hits"]
                b["hs"] += rs["hs"]
                b["plants"] += rs["plant"]
                b["defuses"] += rs["defuse"]
                b["faults"] += rs["faults"]
                b["untraded_deaths"] += untraded
                if kast:
                    b["kast_rounds"] += 1

        rounds_summary.append(
            {
                "round_index": ri,
                "round_num": ri + 1,
                "attack_team": att_team,
                "winner": winner,
                "reason": reason,
                "score": score,
                "duration_s": dur_s,
                "bomb_site": bomb_site,
                "planted_by": planted_by,
                "defused_by": defused_by,
                "kills": kills_in_round,
                "seats": {
                    str(s): {
                        "k": r_seat[s]["k"],
                        "d": r_seat[s]["d"],
                        "a": r_seat[s]["a"],
                        "dmg": round(r_seat[s]["dmg"], 1),
                        "fires": r_seat[s]["fires"],
                        "hits": r_seat[s]["hits"],
                        "hs": r_seat[s]["hs"],
                        "faults": r_seat[s]["faults"],
                    }
                    for s in seats
                },
            }
        )

    def _finalize(b: dict) -> dict:
        r = max(1, b["rounds"])
        d = max(1, b["d"])
        return {
            **b,
            "dmg": round(b["dmg"], 1),
            "kd": round(b["k"] / d, 2),
            "kda": round((b["k"] + b["a"]) / d, 2),
            "adr": round(b["dmg"] / r, 1) if b["rounds"] > 0 else 0.0,
            "acc_pct": round(100.0 * b["hits"] / b["fires"], 1) if b["fires"] > 0 else 0.0,
            "hs_pct": round(100.0 * b["hs"] / b["hits"], 1) if b["hits"] > 0 else 0.0,
            "kast_pct": round(100.0 * b["kast_rounds"] / r, 1) if b["rounds"] > 0 else 0.0,
        }

    scoreboard = {}
    for s in seats:
        scoreboard[str(s)] = {
            "seat": s,
            "team": 0 if s < 5 else 1,
            "overall": _finalize(seat_overall[s]),
            "attack": _finalize(seat_attack[s]),
            "defend": _finalize(seat_defend[s]),
        }

    return {
        "schema_version": 1,
        "job_id": job_id,
        "map_id": map_id,
        "physics_hz": physics_hz,
        "frame_hz": frame_hz,
        "total_rounds": len(all_ris),
        "scoreboard": scoreboard,
        "rounds": rounds_summary,
        "faults": fault_details,
        "parsed_at": now_iso(),
    }


def get_or_fetch_match_stats(job_id: str) -> dict | None:
    """Fetch & cache lightweight public_static + public_events stats for a completed match."""
    STATS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = STATS_CACHE_DIR / f"{job_id}.json"
    if cache_path.is_file():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    lock = _get_replay_lock(f"stats:{job_id}")
    with lock:
        if cache_path.is_file():
            try:
                return json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        desc_res = agx_get_json(f"/api/v1/python-submissions/recordings/{job_id}/replay/descriptor", alias=None, timeout=20)
        if desc_res.get("http_status") != 200 or not isinstance(desc_res.get("body"), dict):
            return None
        desc = desc_res["body"]
        if desc.get("complete") is not True:
            return None

        static_bytes = _download_stream_bytes(job_id, desc, "public_static", max_workers=2)
        static_lines = [ln for ln in static_bytes.splitlines() if ln.strip()]
        if not static_lines:
            return None
        static_obj = json.loads(static_lines[0].decode("utf-8"))

        events_bytes = _download_stream_bytes(job_id, desc, "public_events", max_workers=4)
        events = [json.loads(ln.decode("utf-8")) for ln in events_bytes.splitlines() if ln.strip()]

        stats = parse_match_stats_from_streams(job_id, static_obj, events)
        _safe_atomic_write(cache_path, stats)
        return stats


def _extract_map_geometry(raw_map: dict) -> dict:
    polygons = []
    roles = raw_map.get("render", {}).get("solids", {})
    wall_pts = []
    for box in raw_map.get("boxes", []):
        role = roles.get(box.get("id", ""), {}).get("role")
        kind = "floor" if str(box.get("id", "")).startswith("floor_") or role == "platform" else "wall" if role == "wall" else "cover"
        x0, x1 = box["x"] - box["w"] / 2.0, box["x"] + box["w"] / 2.0
        y0, y1 = box["y"] - box["d"] / 2.0, box["y"] + box["d"] / 2.0
        z = box["z"] + box["h"] if kind == "floor" else box["z"]
        polygons.append(
            {
                "kind": kind,
                "z": round(float(z), 2),
                "top_z": round(float(box["z"] + box["h"]), 2),
                "pts": [
                    [round(x0, 2), round(y0, 2)],
                    [round(x1, 2), round(y0, 2)],
                    [round(x1, 2), round(y1, 2)],
                    [round(x0, 2), round(y1, 2)],
                ],
            }
        )
        if kind == "wall":
            wall_pts.extend([[x0, y0], [x1, y1]])
    if wall_pts:
        bounds = [
            round(min(p[0] for p in wall_pts), 2),
            round(min(p[1] for p in wall_pts), 2),
            round(max(p[0] for p in wall_pts), 2),
            round(max(p[1] for p in wall_pts), 2),
        ]
    else:
        b = raw_map.get("bounds") or {"min": [-50, -50], "max": [50, 50]}
        bounds = [round(b["min"][0], 2), round(b["min"][1], 2), round(b["max"][0], 2), round(b["max"][1], 2)]
    for ramp in raw_map.get("ramps", []):
        x0, x1 = ramp["x"] - ramp["w"] / 2.0, ramp["x"] + ramp["w"] / 2.0
        y0, y1 = ramp["y"] - ramp["d"] / 2.0, ramp["y"] + ramp["d"] / 2.0
        polygons.append(
            {
                "kind": "ramp",
                "z": round(float((ramp["z0"] + ramp["z1"]) / 2.0), 2),
                "top_z": round(float(max(ramp["z0"], ramp["z1"])), 2),
                "pts": [
                    [round(x0, 2), round(y0, 2)],
                    [round(x1, 2), round(y0, 2)],
                    [round(x1, 2), round(y1, 2)],
                    [round(x0, 2), round(y1, 2)],
                ],
            }
        )
    sites = []
    for s in raw_map.get("sites", []):
        c = s.get("center") or s.get("position")
        if c and len(c) >= 2:
            sites.append({"name": s.get("id") or s.get("name") or "?", "x": round(float(c[0]), 2), "y": round(float(c[1]), 2), "r": round(float(s.get("radius", 3.5)), 2)})
    spawns = {}
    for side, pts in (raw_map.get("spawns") or {}).items():
        spawns[side] = [[round(float(p["position"][0]), 2), round(float(p["position"][1]), 2)] for p in pts if "position" in p]

    # On CS2 triangle maps (dust2/mirage), boxes/ramps are empty -> extract nav walkable lines & callout labels
    nav_lines = []
    labels = []
    if not polygons:
        nav = raw_map.get("nav") or {}
        nodes_by_id = {}
        for n in nav.get("nodes") or []:
            pos = n.get("position")
            if isinstance(pos, list) and len(pos) >= 2:
                nodes_by_id[str(n.get("id"))] = (round(float(pos[0]), 1), round(float(pos[1]), 1))
        seen_edges = set()
        for e in nav.get("edges") or []:
            u = str(e.get("from"))
            v = str(e.get("to"))
            if u in nodes_by_id and v in nodes_by_id and u != v:
                key = (u, v) if u < v else (v, u)
                if key not in seen_edges:
                    seen_edges.add(key)
                    p0, p1 = nodes_by_id[u], nodes_by_id[v]
                    nav_lines.append([p0[0], p0[1], p1[0], p1[1]])
        # Include nav bounds in overall bounds if needed
        if nodes_by_id:
            nxs = [p[0] for p in nodes_by_id.values()]
            nys = [p[1] for p in nodes_by_id.values()]
            bounds = [
                round(min(bounds[0], min(nxs)), 2),
                round(min(bounds[1], min(nys)), 2),
                round(max(bounds[2], max(nxs)), 2),
                round(max(bounds[3], max(nys)), 2),
            ]

    s2p = (raw_map.get("render") or {}).get("source_to_platform")
    if isinstance(s2p, dict) and isinstance(s2p.get("basis"), list) and isinstance(s2p.get("offset"), list):
        basis = s2p["basis"]
        off = s2p["offset"]
        for lb in raw_map.get("labels") or []:
            if "x" in lb and "z" in lb and lb.get("text"):
                vx, vy, vz = float(lb["x"]), float(lb.get("y", 0.0)), float(lb["z"])
                px = basis[0][0] * vx + basis[0][1] * vy + basis[0][2] * vz + off[0]
                py = basis[1][0] * vx + basis[1][1] * vy + basis[1][2] * vz + off[1]
                labels.append({"text": str(lb["text"]), "x": round(px, 1), "y": round(py, 1)})

    return {
        "id": raw_map.get("id", "unknown"),
        "bounds": bounds,
        "polygons": polygons,
        "nav_lines": nav_lines,
        "labels": labels,
        "sites": sites,
        "spawns": spawns,
    }


def _build_replay_skeleton(
    job_id: str,
    static_obj: dict,
    events: list[dict],
    participants_meta: list[dict] | None = None,
) -> tuple[dict, dict[int, dict], list[int], float]:
    """Build replay metadata + per-round events immediately from public_static + public_events (~0.8s)."""
    physics_hz = float(static_obj.get("physics_hz") or 60)
    starts = {int(e["round_index"]): e for e in events if e.get("type") == "round_start" and e.get("round_index") is not None}
    ends = {int(e["round_index"]): e for e in events if e.get("type") == "round_end" and e.get("round_index") is not None}
    all_ris = sorted(set(starts.keys()) | set(ends.keys()))
    init_att = int(starts[all_ris[0]].get("attack_team", 0)) if all_ris and all_ris[0] in starts else 0

    rounds: dict[int, dict] = {}
    for ri in all_ris:
        st = starts.get(ri, {})
        en = ends.get(ri, {})
        live_tick = int(st.get("buy_freeze_end_tick", st.get("match_tick", 0)))
        start_tick = int(st.get("match_tick", live_tick))
        end_tick = int(en.get("match_tick", live_tick))
        rounds[ri] = {
            "index": ri,
            "round_num": ri + 1,
            "start_tick": start_tick,
            "live_tick": live_tick,
            "end_tick": end_tick,
            "duration": round(max(0.0, (end_tick - live_tick) / physics_hz), 2),
            "min_t": round((start_tick - live_tick) / physics_hz, 2),
            "attack_team": int(st.get("attack_team", init_att if ri < 12 else 1 - init_att)),
            "winner": en.get("round_winner"),
            "reason": en.get("reason", ""),
            "score": en.get("round_wins"),
            "tracks": {str(s): [] for s in range(10)},
            "bomb": [],
            "events": [],
        }

    keep_types = {
        "fire",
        "hit",
        "kill",
        "fault",
        "plant_complete_detail",
        "plant_complete",
        "defuse_complete_detail",
        "defuse_complete",
        "bomb_exploded",
        "bomb_defused",
        "throw",
    }
    for e in sorted(events, key=lambda x: (x.get("match_tick", 0), x.get("event_id", 0))):
        et = e.get("type")
        ri = e.get("round_index")
        if et not in keep_types or ri is None or int(ri) not in rounds:
            continue
        r = rounds[int(ri)]
        t = round((int(e.get("match_tick", 0)) - r["live_tick"]) / physics_hz, 2)
        ev = {
            "t": t,
            "tick": int(e.get("match_tick", 0)),
            "type": et,
            "player": e.get("player", e.get("seat")),
            "killer": e.get("killer"),
            "target": e.get("target"),
            "assists": e.get("assists") or [],
            "weapon": e.get("weapon") or "",
            "part": e.get("part"),
            "dmg": round(float(e.get("hp_damage", 0.0)), 1) if e.get("hp_damage") is not None else None,
            "site": e.get("site"),
        }
        frm = _coord3(e.get("from"))
        to = _coord3(e.get("to"))
        pos = _coord3(e.get("position"))
        if frm:
            ev["from"] = frm[:2]
        if to:
            ev["to"] = to[:2]
        if pos:
            ev["pos"] = pos[:2]
        r["events"].append(ev)

    static_players = static_obj.get("players") or static_obj.get("participants") or []
    p_meta_by_seat = {int(p["seat"]): p for p in (participants_meta or []) if "seat" in p}
    participants = []
    for s in range(10):
        sp = next((p for p in static_players if int(p.get("seat", -1)) == s), {})
        mp = p_meta_by_seat.get(s, {})
        participants.append(
            {
                "seat": s,
                "team": mp.get("team", 0 if s < 5 else 1),
                "display_name": mp.get("display_name") or sp.get("display_name") or sp.get("name") or f"Seat {s}",
                "is_mine": bool(mp.get("is_mine", False)),
            }
        )

    skeleton = {
        "schema_version": 1,
        "job_id": job_id,
        "map": _extract_map_geometry(static_obj.get("map") or {}),
        "physics_hz": physics_hz,
        "participants": participants,
        "rounds": [rounds[ri] for ri in all_ris],
        "built_at": now_iso(),
    }
    return skeleton, rounds, all_ris, physics_hz


def _consume_frame_lines(
    lines: list[bytes],
    rounds: dict[int, dict],
    physics_hz: float,
    sample_ticks: float,
    state_ctx: dict,
) -> int | None:
    """Fast stride-aware frame parser: skips ~75% of JSON parses (24Hz->6Hz) for 4x CPU speedup."""
    last_bin = state_ctx["last_bin"]
    last_bomb_bin = state_ctx["last_bomb_bin"]
    line_idx = state_ctx["line_idx"]
    stride = state_ctx["stride"]
    max_ri_seen = state_ctx["max_ri_seen"]

    for raw_line in lines:
        if not raw_line:
            continue
        line_idx += 1
        # Keep every Nth line (24Hz / 4 = 6Hz) so we skip 75% of json.loads()
        if (line_idx % stride) != 1:
            continue
        fr = json.loads(raw_line)
        ri = fr.get("round_index")
        if ri is None or int(ri) not in rounds:
            continue
        ri = int(ri)
        if max_ri_seen is None or ri > max_ri_seen:
            max_ri_seen = ri
        r = rounds[ri]
        tick = int(fr["match_tick"])
        if tick < r["start_tick"] or tick > r["end_tick"]:
            continue
        t = round((tick - r["live_tick"]) / physics_hz, 2)
        cur_bin = int(math.floor((tick - r["live_tick"]) / sample_ticks))

        for p in fr.get("players", []):
            s = int(p["seat"])
            skey = str(s)
            if skey not in r["tracks"]:
                continue
            key = (ri, s)
            if last_bin.get(key) != cur_bin:
                r["tracks"][skey].append(
                    [
                        t,
                        round(float(p["x"]), 2),
                        round(float(p["y"]), 2),
                        round(float(p["z"]), 2),
                        round(float(p.get("yaw", 0.0)), 3),
                        int(p.get("hp", 0)),
                        1 if p.get("alive") else 0,
                        1 if p.get("crouch") else 0,
                        p.get("weapon_id") or "",
                    ]
                )
                last_bin[key] = cur_bin

        bomb = fr.get("bomb")
        if isinstance(bomb, dict) and last_bomb_bin.get(ri) != cur_bin:
            bpos = _coord3(bomb.get("position"))
            if bpos:
                r["bomb"].append([t, bpos[0], bpos[1], bomb.get("state") or "", bomb.get("carrier")])
                last_bomb_bin[ri] = cur_bin

    state_ctx["line_idx"] = line_idx
    state_ctx["max_ri_seen"] = max_ri_seen
    return max_ri_seen


def _run_progressive_worker(job_id: str, desc: dict, job_state: dict) -> None:
    """Background 16-worker Keep-Alive batch streamer: Round 1 playable in ~3s, full match cached in ~35-45s."""
    meta = desc.get("streams", {}).get("public_frames") or {}
    n_chunks = int(meta.get("chunks", 0))
    rounds = job_state["rounds_map"]
    all_ris = job_state["all_ris"]
    physics_hz = job_state["physics_hz"]
    frame_hz = float(job_state["frame_hz"] or 24)
    sample_hz = 6.0
    sample_ticks = physics_hz / sample_hz
    stride = max(1, int(round(frame_hz / sample_hz)))

    parse_ctx = {"last_bin": {}, "last_bomb_bin": {}, "line_idx": 0, "stride": stride, "max_ri_seen": None}
    n_workers = min(16, max(1, n_chunks))
    clients = [_KeepAliveClient(timeout=30) for _ in range(n_workers)]
    pending = b""
    batch_size = 48
    try:
        for start in range(0, n_chunks, batch_size):
            batch_indices = list(range(start, min(n_chunks, start + batch_size)))
            parts = _download_chunks_keepalive(job_id, desc, "public_frames", batch_indices, clients)
            chunk_blob = pending + b"".join(parts)
            lines = chunk_blob.split(b"\n")
            pending = lines.pop()
            max_ri = _consume_frame_lines(lines, rounds, physics_hz, sample_ticks, parse_ctx)

            with job_state["lock"]:
                job_state["chunks_done"] = start + len(batch_indices)
                # Mark all rounds strictly before max_ri (and any round that already has tracks if it's the only one so far) as ready
                ready_set = set(job_state["rounds_ready"])
                if max_ri is not None:
                    for ri in all_ris:
                        if ri < max_ri:
                            ready_set.add(ri)
                    # Also allow watching the currently streaming round once it has >= 10 track points
                    if any(len(tr) >= 10 for tr in rounds[max_ri]["tracks"].values()):
                        ready_set.add(max_ri)
                job_state["rounds_ready"] = sorted(ready_set)

        if pending.strip():
            _consume_frame_lines([pending], rounds, physics_hz, sample_ticks, parse_ctx)

        # Finalize all rounds & save gzip cache to disk
        REPLAY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = REPLAY_CACHE_DIR / f"{job_id}.json.gz"
        bundle = job_state["bundle"]
        bundle["built_at"] = now_iso()
        gz_bytes = gzip.compress(json.dumps(bundle, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), compresslevel=5)
        tmp = cache_path.with_name(f".{cache_path.name}.tmp")
        tmp.write_bytes(gz_bytes)
        tmp.replace(cache_path)

        with job_state["lock"]:
            job_state["chunks_done"] = n_chunks
            job_state["rounds_ready"] = list(all_ris)
            job_state["status"] = "ready"
    except Exception as exc:
        with job_state["lock"]:
            job_state["status"] = "error"
            job_state["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        for c in clients:
            c.close()


def _fps_cache_is_valid(cache_path: Path) -> bool:
    if not cache_path.is_file():
        return False
    try:
        obj = json.loads(gzip.decompress(cache_path.read_bytes()))
        m = obj.get("map") or {}
        if not m.get("polygons") and not m.get("nav_lines"):
            return False
        return True
    except Exception:
        return False


def init_progressive_replay(job_id: str, participants_meta: list[dict] | None = None) -> dict | None:
    """Fast (<1s) init: returns cached=True if on disk, else returns map+events skeleton & starts 16-worker stream."""
    REPLAY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = REPLAY_CACHE_DIR / f"{job_id}.json.gz"
    if _fps_cache_is_valid(cache_path):
        return {"cached": True, "job_id": job_id}

    with _PROGRESSIVE_GUARD:
        existing = _PROGRESSIVE_JOBS.get(job_id)
        if existing is not None and existing["status"] in ("loading", "ready"):
            with existing["lock"]:
                return {
                    "cached": False,
                    "job_id": job_id,
                    "status": existing["status"],
                    "chunks_done": existing["chunks_done"],
                    "chunks_total": existing["chunks_total"],
                    "rounds_ready": list(existing["rounds_ready"]),
                    "skeleton": {
                        "schema_version": 1,
                        "job_id": job_id,
                        "map": existing["bundle"]["map"],
                        "physics_hz": existing["bundle"]["physics_hz"],
                        "participants": existing["bundle"]["participants"],
                        "rounds": [
                            {k: v for k, v in r.items() if k not in ("tracks", "bomb")}
                            for r in existing["bundle"]["rounds"]
                        ],
                    },
                }

    desc_res = agx_get_json(f"/api/v1/python-submissions/recordings/{job_id}/replay/descriptor", alias=None, timeout=20)
    if desc_res.get("http_status") != 200 or not isinstance(desc_res.get("body"), dict):
        return None
    desc = desc_res["body"]
    if desc.get("complete") is not True:
        return None
    if int(desc.get("streams", {}).get("public_frames", {}).get("raw_bytes", 0)) > 1024 * 1024 * 1024:
        raise ValueError("Replay exceeds 1 GiB safety budget")

    static_bytes = _download_stream_bytes(job_id, desc, "public_static", max_workers=2)
    static_obj = json.loads(static_bytes.splitlines()[0].decode("utf-8"))
    events_bytes = _download_stream_bytes(job_id, desc, "public_events", max_workers=4)
    events = [json.loads(ln.decode("utf-8")) for ln in events_bytes.splitlines() if ln.strip()]

    bundle, rounds_map, all_ris, physics_hz = _build_replay_skeleton(job_id, static_obj, events, participants_meta)
    n_chunks = int(desc.get("streams", {}).get("public_frames", {}).get("chunks", 0))

    job_state = {
        "lock": threading.Lock(),
        "status": "loading",
        "error": None,
        "chunks_done": 0,
        "chunks_total": n_chunks,
        "rounds_ready": [],
        "bundle": bundle,
        "rounds_map": rounds_map,
        "all_ris": all_ris,
        "physics_hz": physics_hz,
        "frame_hz": float(static_obj.get("frame_hz") or 24),
    }
    with _PROGRESSIVE_GUARD:
        _PROGRESSIVE_JOBS[job_id] = job_state

    th = threading.Thread(target=_run_progressive_worker, args=(job_id, desc, job_state), daemon=True)
    th.start()

    return {
        "cached": False,
        "job_id": job_id,
        "status": "loading",
        "chunks_done": 0,
        "chunks_total": n_chunks,
        "rounds_ready": [],
        "skeleton": {
            "schema_version": 1,
            "job_id": job_id,
            "map": bundle["map"],
            "physics_hz": bundle["physics_hz"],
            "participants": bundle["participants"],
            "rounds": [{k: v for k, v in r.items() if k not in ("tracks", "bomb")} for r in bundle["rounds"]],
        },
    }


def poll_progressive_replay(job_id: str, have_rounds: set[int]) -> dict:
    """Return current progress + any newly ready rounds' tracks & bomb arrays."""
    with _PROGRESSIVE_GUARD:
        job_state = _PROGRESSIVE_JOBS.get(job_id)
    if job_state is None:
        cache_path = REPLAY_CACHE_DIR / f"{job_id}.json.gz"
        if cache_path.is_file():
            return {"status": "cached", "chunks_done": 1, "chunks_total": 1, "rounds_ready": [], "new_rounds": {}}
        return {"status": "missing", "error": "Replay job not initialized"}

    with job_state["lock"]:
        status = job_state["status"]
        chunks_done = job_state["chunks_done"]
        chunks_total = job_state["chunks_total"]
        ready = list(job_state["rounds_ready"])
        error = job_state["error"]
        new_rounds = {}
        for ri in ready:
            # Always refresh the latest streaming round until next round starts or job completes
            is_frontier = (status == "loading" and ready and ri == ready[-1])
            if ri not in have_rounds or is_frontier:
                r = job_state["rounds_map"][ri]
                new_rounds[str(ri)] = {"tracks": r["tracks"], "bomb": r["bomb"]}
    return {
        "status": status,
        "error": error,
        "chunks_done": chunks_done,
        "chunks_total": chunks_total,
        "rounds_ready": ready,
        "new_rounds": new_rounds,
    }


def get_or_fetch_replay_bundle(job_id: str, participants_meta: list[dict] | None = None) -> bytes | None:
    """Return cached .json.gz replay bundle, or build synchronously using fast 16-worker Keep-Alive."""
    REPLAY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = REPLAY_CACHE_DIR / f"{job_id}.json.gz"
    if _fps_cache_is_valid(cache_path):
        return cache_path.read_bytes()

    lock = _get_replay_lock(f"replay:{job_id}")
    with lock:
        if _fps_cache_is_valid(cache_path):
            return cache_path.read_bytes()

        desc_res = agx_get_json(f"/api/v1/python-submissions/recordings/{job_id}/replay/descriptor", alias=None, timeout=25)
        if desc_res.get("http_status") != 200 or not isinstance(desc_res.get("body"), dict):
            return None
        desc = desc_res["body"]
        if desc.get("complete") is not True:
            return None
        if int(desc.get("streams", {}).get("public_frames", {}).get("raw_bytes", 0)) > 1024 * 1024 * 1024:
            raise ValueError("Replay exceeds 1 GiB safety budget")

        static_bytes = _download_stream_bytes(job_id, desc, "public_static", max_workers=2)
        static_obj = json.loads(static_bytes.splitlines()[0].decode("utf-8"))
        events_bytes = _download_stream_bytes(job_id, desc, "public_events", max_workers=4)
        events = [json.loads(ln.decode("utf-8")) for ln in events_bytes.splitlines() if ln.strip()]
        frames_bytes = _download_stream_bytes(job_id, desc, "public_frames", max_workers=16)

        bundle, rounds_map, _, physics_hz = _build_replay_skeleton(job_id, static_obj, events, participants_meta)
        frame_hz = float(static_obj.get("frame_hz") or 24)
        sample_hz = 6.0
        parse_ctx = {
            "last_bin": {},
            "last_bomb_bin": {},
            "line_idx": 0,
            "stride": max(1, int(round(frame_hz / sample_hz))),
            "max_ri_seen": None,
        }
        _consume_frame_lines(frames_bytes.splitlines(), rounds_map, physics_hz, physics_hz / sample_hz, parse_ctx)

        gz_bytes = gzip.compress(json.dumps(bundle, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), compresslevel=5)
        tmp = cache_path.with_name(f".{cache_path.name}.tmp")
        tmp.write_bytes(gz_bytes)
        tmp.replace(cache_path)
        return gz_bytes


def _load_version_bindings() -> dict[str, dict]:
    """Load persisted exact job_id -> python_version bindings (from round_match, collector, experiments)."""
    bindings: dict[str, dict] = {}
    if VERSION_BINDINGS_PATH.is_file():
        try:
            raw = json.loads(VERSION_BINDINGS_PATH.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                bindings.update(raw)
        except Exception:
            pass
    # Seed from existing local receipts if not already present
    try:
        for p in (ROOT / "outputs/fps-experiments").glob("*/FINAL_TRIAL_RESULT.json"):
            d = json.loads(p.read_text(encoding="utf-8"))
            for m in d.get("actual_metrics", []):
                jid = m.get("job_id")
                if jid and m.get("version_number") is not None and jid not in bindings:
                    bindings[jid] = {
                        "id": m.get("version_id"),
                        "number": m.get("version_number"),
                        "bundle_sha256": m.get("zip_sha256"),
                        "source": "experiment_receipt",
                    }
    except Exception:
        pass
    try:
        lg = ROOT / "outputs/fps-operations/collector/last-good.json"
        if lg.is_file():
            d = json.loads(lg.read_text(encoding="utf-8"))
            for acc in (d.get("accounts") or {}).values():
                rm = acc.get("round_match") or {}
                if rm.get("job_id") and rm.get("python_version") and rm["job_id"] not in bindings:
                    pv = dict(rm["python_version"])
                    pv["source"] = "collector_last_good"
                    bindings[rm["job_id"]] = pv
    except Exception:
        pass
    return bindings


def _save_version_bindings(bindings: dict[str, dict]) -> None:
    try:
        _safe_atomic_write(VERSION_BINDINGS_PATH, bindings)
    except Exception:
        pass


def _infer_version_from_bundles(match_created_at: str | None, bundles: list[dict]) -> dict | None:
    """Infer the latest passed bundle created at or before match_created_at (explicitly marked inferred)."""
    if not match_created_at or not bundles:
        return None
    passed = [
        b for b in bundles
        if b.get("created_at") and b.get("number") is not None
        and b.get("admission") in (None, "passed")
        and b["created_at"] <= match_created_at
    ]
    if not passed:
        return None
    best = max(passed, key=lambda b: (b["created_at"], b.get("number") or 0))
    return {
        "id": best.get("version_id") or best.get("id"),
        "number": best.get("number"),
        "bundle_sha256": best.get("bundle_sha256"),
        "source": "inferred_from_bundle_time",
    }


def _fetch_one_account(acc: dict, matches_limit: int = 50, known_bindings: dict[str, dict] | None = None) -> dict:
    alias = acc["alias"]
    agent_id = acc["agent_id"]
    result = {
        "alias": alias,
        "username": acc["username"],
        "short_name": acc["short_name"],
        "role": acc["role"],
        "label": acc["label"],
        "agent_id": agent_id,
        "purpose": acc["purpose"],
        "status": "ok",
        "errors": [],
        "fetched_at": now_iso(),
        "standing": None,
        "latest_published": None,
        "current_version": None,
        "round_match": None,
        "queue": None,
        "next_round_start_at": None,
        "recent_matches": [],
    }

    # 1. Identity verification
    ident = agx_get_json(f"/api/v1/agents/{agent_id}", alias=alias)
    if ident.get("http_status") != 200 or not isinstance(ident.get("body"), dict):
        result["status"] = "error"
        result["errors"].append(f"identity HTTP {ident.get('http_status') or ident.get('error')}")
        return result
    if ident["body"].get("is_mine") is not True:
        result["status"] = "error"
        result["errors"].append("identity_mismatch: is_mine != true")
        return result
    if ident["body"].get("display_name"):
        result["username"] = ident["body"]["display_name"]

    # 2. Competitions + Bundles + Recent Matches + Mine Agents (for racing discovery) in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        f_comp = pool.submit(agx_get_json, "/api/v1/me/competitions", alias)
        f_bund = pool.submit(agx_get_json, f"/api/v1/python-submissions/agents/{agent_id}/bundles", alias)
        f_mat = pool.submit(agx_get_json, f"/api/v1/matches?agent_id={agent_id}&limit={matches_limit}", alias)
        f_mine = pool.submit(agx_get_json, "/api/v1/agents?mine=true&limit=20", alias)
        comp_res = f_comp.result()
        bund_res = f_bund.result()
        mat_res = f_mat.result()
        mine_res = f_mine.result()

    # Check if this account also has a racing agent/competition
    racing_agent = None
    if mine_res.get("http_status") == 200 and isinstance(mine_res.get("body"), dict):
        racing_agent = next((x for x in (mine_res["body"].get("items") or []) if x.get("game_id") == "racing"), None)
    racing_comp = None
    if comp_res.get("http_status") == 200 and isinstance(comp_res.get("body"), dict):
        racing_comp = next((x for x in (comp_res["body"].get("items") or []) if (x.get("competition") or {}).get("game_id") == "racing"), None)
    if racing_agent or racing_comp:
        r_aid = (racing_agent or {}).get("id")
        r_sh_raw = (racing_comp or {}).get("score_history") or []
        r_sh = [
            {
                "t": x.get("t"),
                "rating": round(float(x["rating"]), 2) if _finite(x.get("rating")) else None,
                "job_id": x.get("job_id"),
                "version_number": x.get("version_number"),
            }
            for x in r_sh_raw
            if isinstance(x, dict) and _finite(x.get("rating"))
        ]
        r_info = {
            "alias": alias,
            "username": (racing_agent or {}).get("display_name") or result["username"],
            "short_name": acc["short_name"],
            "role": acc["role"],
            "agent_id": r_aid,
            "standing": (racing_comp or {}).get("standing"),
            "round_match": (racing_comp or {}).get("round_match"),
            "queue": (racing_comp or {}).get("queue"),
            "next_round_start_at": (racing_comp or {}).get("next_round_start_at"),
            "score_history": r_sh,
            "latest_published": None,
            "recent_matches": [],
        }
        if r_aid:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as rpool:
                fr_b = rpool.submit(agx_get_json, f"/api/v1/python-submissions/agents/{r_aid}/bundles", alias)
                fr_m = rpool.submit(agx_get_json, f"/api/v1/matches?agent_id={r_aid}&limit={matches_limit}", alias)
                rb_res = fr_b.result()
                rm_res = fr_m.result()
            r_bundles = []
            if rb_res.get("http_status") == 200 and isinstance(rb_res.get("body"), dict):
                r_bundles = [x for x in rb_res["body"].get("items", []) if isinstance(x, dict) and x.get("version_id")]
                if r_bundles:
                    rlat = max(r_bundles, key=lambda x: x.get("number") or 0)
                    r_info["latest_published"] = {
                        "version_id": rlat.get("version_id"),
                        "number": rlat.get("number"),
                        "bundle_sha256": rlat.get("bundle_sha256"),
                        "admission": rlat.get("admission"),
                        "created_at": rlat.get("created_at"),
                    }
            r_rm = r_info.get("round_match") or {}
            r_bound = {}
            sh_rc_by_job = {}
            prev_r = 1000.0
            for sh_item in r_sh:
                jid = sh_item.get("job_id")
                if jid and sh_item.get("version_number") is not None:
                    r_bound[jid] = {"number": sh_item["version_number"], "source": "score_history"}
                if jid and sh_item.get("rating") is not None:
                    after_r = float(sh_item["rating"])
                    sh_rc_by_job[jid] = {
                        "before": round(prev_r, 2),
                        "after": round(after_r, 2),
                        "delta": round(after_r - prev_r, 2),
                    }
                    prev_r = after_r
            if r_rm.get("job_id") and r_rm.get("python_version"):
                r_bound[r_rm["job_id"]] = {**r_rm["python_version"], "source": "exact_round_match"}
            if rm_res.get("http_status") == 200 and isinstance(rm_res.get("body"), dict):
                for m in rm_res["body"].get("items", []):
                    ps = m.get("participants") or []
                    own = next((p for p in ps if p.get("is_mine") is True), None)
                    if not own:
                        continue
                    own_seat = own.get("seat")
                    ranks = (m.get("result") or {}).get("ranks")
                    place = (ranks[own_seat] + 1) if (isinstance(ranks, list) and own_seat is not None and 0 <= own_seat < len(ranks) and isinstance(ranks[own_seat], int)) else None
                    r_info["recent_matches"].append({
                        "job_id": m["id"],
                        "account_alias": alias,
                        "account_username": r_info["username"],
                        "account_short_name": r_info["short_name"],
                        "agent_id": r_aid,
                        "status": m.get("status"),
                        "created_at": m.get("created_at"),
                        "finished_at": m.get("finished_at"),
                        "own_seat": own_seat,
                        "place": place,
                        "total_racers": len(ps),
                        "rating_change": own.get("rating_change") or sh_rc_by_job.get(m["id"]),
                        "bound_version": r_bound.get(m["id"]) or _infer_version_from_bundles(m.get("created_at"), r_bundles),
                        "participants": [
                            {
                                "seat": p.get("seat"),
                                "display_name": p.get("display_name") or f"Seat {p.get('seat')}",
                                "is_mine": bool(p.get("is_mine")),
                                "place": (ranks[p.get("seat")] + 1) if (isinstance(ranks, list) and p.get("seat") is not None and 0 <= p.get("seat") < len(ranks) and isinstance(ranks[p.get("seat")], int)) else None,
                                "rating_change": p.get("rating_change"),
                            }
                            for p in ps
                        ],
                        "agx_url": f"https://agx.cool/#/matches/{m['id']}",
                    })

        # Compute global & recent rating delta summary for this racing account
        cur_r = (r_info.get("standing") or {}).get("rating")
        ordered_ms = sorted(
            [m for m in r_info["recent_matches"] if isinstance((m.get("rating_change") or {}).get("delta"), (int, float))],
            key=lambda x: x.get("created_at") or "",
        )
        base_r = ordered_ms[0]["rating_change"].get("before") if ordered_ms else 1000.0
        if not _finite(base_r):
            base_r = 1000.0
        latest_delta = ordered_ms[-1]["rating_change"]["delta"] if ordered_ms else None
        recent_net = sum(m["rating_change"]["delta"] for m in ordered_ms) if ordered_ms else None
        r_info["rating_stats"] = {
            "current": round(float(cur_r), 2) if _finite(cur_r) else None,
            "net_from_1000": round(float(cur_r) - 1000.0, 2) if _finite(cur_r) else None,
            "recent_net": round(float(recent_net), 2) if recent_net is not None else None,
            "latest_delta": round(float(latest_delta), 2) if latest_delta is not None else None,
            "base_rating": round(float(base_r), 2),
        }
        result["racing_info"] = r_info
    else:
        result["racing_info"] = None

    if comp_res.get("http_status") == 200 and isinstance(comp_res.get("body"), dict):
        items = comp_res["body"].get("items") or []
        fps_comp = next((x for x in items if (x.get("competition") or {}).get("game_id") == "fps"), None)
        if fps_comp:
            result["standing"] = fps_comp.get("standing")
            result["round_match"] = fps_comp.get("round_match")
            result["queue"] = fps_comp.get("queue")
            result["next_round_start_at"] = fps_comp.get("next_round_start_at")
            result["current_version"] = fps_comp.get("current_version")
            result["score_history"] = fps_comp.get("score_history") or []
    else:
        result["errors"].append(f"competitions HTTP {comp_res.get('http_status')}")

    bundles_list: list[dict] = []
    if bund_res.get("http_status") == 200 and isinstance(bund_res.get("body"), dict):
        versions = [x for x in bund_res["body"].get("items", []) if isinstance(x, dict) and x.get("version_id")]
        bundles_list = versions
        if versions:
            latest = max(versions, key=lambda x: x.get("number") or 0)
            result["latest_published"] = {
                "version_id": latest.get("version_id"),
                "number": latest.get("number"),
                "bundle_sha256": latest.get("bundle_sha256"),
                "admission": latest.get("admission"),
                "created_at": latest.get("created_at"),
            }
    else:
        result["errors"].append(f"bundles HTTP {bund_res.get('http_status')}")

    bound_job_versions: dict[str, dict] = dict(known_bindings or {})
    for sh_item in result.get("score_history") or []:
        if isinstance(sh_item, dict) and sh_item.get("job_id") and sh_item.get("version_number") is not None:
            if sh_item["job_id"] not in bound_job_versions:
                bound_job_versions[sh_item["job_id"]] = {"number": sh_item["version_number"], "source": "score_history"}
    rm = result.get("round_match") or {}
    if rm.get("job_id") and rm.get("python_version"):
        pv = dict(rm["python_version"])
        pv["source"] = "exact_round_match"
        bound_job_versions[rm["job_id"]] = pv
        result["new_exact_binding"] = (rm["job_id"], pv)

    if mat_res.get("http_status") == 200 and isinstance(mat_res.get("body"), dict):
        matches_clean = []
        for m in mat_res["body"].get("items", []):
            participants = m.get("participants") or []
            own = next((p for p in participants if p.get("is_mine") is True), None)
            if not own:
                continue
            own_team = own.get("team", 0 if own.get("seat", 0) < 5 else 1)
            scores = m.get("team_scores") or m.get("scores")
            own_score = None
            opp_score = None
            if isinstance(scores, list) and len(scores) >= 2:
                own_score = scores[own_team]
                opp_score = scores[1 - own_team]
            elif isinstance(m.get("result"), dict) and isinstance(m["result"].get("round_wins"), list):
                rw = m["result"]["round_wins"]
                if len(rw) >= 2:
                    own_score = rw[own_team]
                    opp_score = rw[1 - own_team]

            outcome = "pending"
            if m.get("status") == "succeeded":
                if own_score is not None and opp_score is not None:
                    outcome = "W" if own_score > opp_score else "L" if own_score < opp_score else "D"
                elif own.get("score") is not None:
                    outcome = "W" if own["score"] > 0.5 else "L" if own["score"] < 0.5 else "D"
            elif m.get("status") in ("running", "queued", "pending"):
                outcome = "live"
            else:
                outcome = m.get("status") or "unknown"

            matches_clean.append(
                {
                    "job_id": m["id"],
                    "account_alias": alias,
                    "account_username": result["username"],
                    "account_short_name": result["short_name"],
                    "agent_id": agent_id,
                    "status": m.get("status"),
                    "created_at": m.get("created_at"),
                    "started_at": m.get("started_at"),
                    "finished_at": m.get("finished_at"),
                    "own_seat": own.get("seat"),
                    "own_team": own_team,
                    "own_score": own_score,
                    "opp_score": opp_score,
                    "outcome": outcome,
                    "rating_change": own.get("rating_change"),
                    "bound_version": bound_job_versions.get(m["id"]) or _infer_version_from_bundles(m.get("created_at"), bundles_list),
                    "participants": [
                        {
                            "seat": p.get("seat"),
                            "team": p.get("team", 0 if (p.get("seat") or 0) < 5 else 1),
                            "display_name": p.get("display_name") or f"Seat {p.get('seat')}",
                            "is_mine": bool(p.get("is_mine")),
                            "rating_change": p.get("rating_change"),
                        }
                        for p in participants
                    ],
                    "agx_url": f"https://agx.cool/#/matches/{m['id']}",
                    "stats": None,
                }
            )
        result["recent_matches"] = matches_clean
    else:
        result["errors"].append(f"matches HTTP {mat_res.get('http_status')}")

    return result


def fetch_full_leaderboard(alias: str | None = None, competition_id: str = COMPETITION_ID) -> dict[str, list[dict]]:
    """Fetch complete competition leaderboard (GET-only, paginated 100/page) indexed by display_name."""
    by_name: dict[str, list[dict]] = collections.defaultdict(list)
    offset = 0
    for _ in range(10):
        res = agx_get_json(
            f"/api/v1/competitions/{competition_id}/leaderboard?limit=100&offset={offset}",
            alias=alias,
            timeout=20,
        )
        if res.get("http_status") != 200 or not isinstance(res.get("body"), dict):
            break
        items = res["body"].get("items") or []
        if not items:
            break
        for it in items:
            dn = it.get("display_name")
            if dn:
                by_name[dn].append(
                    {
                        "entry_id": it.get("entry_id"),
                        "display_name": dn,
                        "rank": it.get("rank"),
                        "rating": round(float(it["rating"]), 1) if _finite(it.get("rating")) else None,
                        "win_rate_percent": it.get("win_rate_percent"),
                    }
                )
        offset += len(items)
        total = res["body"].get("total")
        if len(items) < 100 or (isinstance(total, int) and offset >= total):
            break
    return dict(by_name)


def _match_leaderboard_entry(p: dict, lb_by_name: dict[str, list[dict]]) -> dict | None:
    dn = p.get("display_name")
    if not dn or dn not in lb_by_name:
        return None
    cands = lb_by_name[dn]
    if len(cands) == 1:
        return cands[0]
    rc = p.get("rating_change") or {}
    ref = rc.get("after") if _finite(rc.get("after")) else rc.get("before")
    if _finite(ref):
        ranked_cands = [c for c in cands if _finite(c.get("rating"))]
        if ranked_cands:
            return min(ranked_cands, key=lambda c: abs(float(c["rating"]) - float(ref)))
    return None


def _enrich_match_lobby(m: dict, lb_by_name: dict[str, list[dict]]) -> None:
    own_team = m.get("own_team")
    ally_ratings: list[float] = []
    enemy_ratings: list[float] = []
    ally_ranks: list[int] = []
    enemy_ranks: list[int] = []
    ally_best_at_match: dict | None = None
    enemy_best_at_match: dict | None = None

    for p in m.get("participants") or []:
        lb = _match_leaderboard_entry(p, lb_by_name)
        if lb:
            p["rank"] = lb.get("rank")
            p["rating"] = lb.get("rating")
            p["win_rate_percent"] = lb.get("win_rate_percent")
        else:
            p.setdefault("rank", None)
            p.setdefault("rating", None)
            p.setdefault("win_rate_percent", None)

        rc = p.get("rating_change") or {}
        match_rating = (
            round(float(rc["before"]), 1)
            if _finite(rc.get("before"))
            else round(float(p["rating"]), 1)
            if _finite(p.get("rating"))
            else None
        )
        p["match_rating"] = match_rating
        is_ally = (p.get("team") == own_team)
        if match_rating is not None:
            if is_ally:
                ally_ratings.append(match_rating)
                if ally_best_at_match is None or match_rating > ally_best_at_match["rating"]:
                    ally_best_at_match = {"display_name": p.get("display_name"), "rating": match_rating, "current_rank": p.get("rank")}
            else:
                enemy_ratings.append(match_rating)
                if enemy_best_at_match is None or match_rating > enemy_best_at_match["rating"]:
                    enemy_best_at_match = {"display_name": p.get("display_name"), "rating": match_rating, "current_rank": p.get("rank")}
        if isinstance(p.get("rank"), int):
            (ally_ranks if is_ally else enemy_ranks).append(p["rank"])

    m["lobby_summary"] = {
        "ally_avg_rating": round(sum(ally_ratings) / len(ally_ratings), 1) if ally_ratings else None,
        "enemy_avg_rating": round(sum(enemy_ratings) / len(enemy_ratings), 1) if enemy_ratings else None,
        "ally_best_rating": ally_best_at_match["rating"] if ally_best_at_match else None,
        "enemy_best_rating": enemy_best_at_match["rating"] if enemy_best_at_match else None,
        "ally_best_player": ally_best_at_match,
        "enemy_best_player": enemy_best_at_match,
        "ally_avg_rank": round(sum(ally_ranks) / len(ally_ranks), 1) if ally_ranks else None,
        "enemy_avg_rank": round(sum(enemy_ranks) / len(enemy_ranks), 1) if enemy_ranks else None,
        "ally_best_rank": min(ally_ranks) if ally_ranks else None,
        "enemy_best_rank": min(enemy_ranks) if enemy_ranks else None,
    }


def refresh_live_state(enrich_limit_per_account: int = 50) -> dict:
    """Refresh all accounts from AGX (GET-only), full leaderboard, and recent match dual-phase stats."""
    with _REFRESH_LOCK:
        started = now_iso()
        roles = load_account_roles()
        known_bindings = _load_version_bindings()
        lb_alias = roles[0]["alias"] if roles else None
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(2, len(roles) + 2)) as pool:
            f_lb = pool.submit(fetch_full_leaderboard, lb_alias, COMPETITION_ID)
            f_rlb = pool.submit(fetch_full_leaderboard, lb_alias, RACING_COMPETITION_ID)
            accounts = list(pool.map(lambda r: _fetch_one_account(r, matches_limit=50, known_bindings=known_bindings), roles))
            try:
                lb_by_name = f_lb.result()
            except Exception:
                lb_by_name = {}
            try:
                racing_lb_by_name = f_rlb.result()
            except Exception:
                racing_lb_by_name = {}

        # Extract isolated racing accounts & matches (never mixed with FPS)
        racing_accounts = []
        racing_matches = []
        for acc in accounts:
            ri = acc.pop("racing_info", None)
            if ri:
                for rm in ri.get("recent_matches", []):
                    match_ratings: list[float] = []
                    for p in rm.get("participants", []):
                        lb = _match_leaderboard_entry(p, racing_lb_by_name)
                        p["rank"] = lb.get("rank") if lb else None
                        p["rating"] = lb.get("rating") if lb else None
                        p["win_rate_percent"] = lb.get("win_rate_percent") if lb else None
                        rc = p.get("rating_change") or {}
                        mr = (
                            round(float(rc["before"]), 1)
                            if _finite(rc.get("before"))
                            else round(float(p["rating"]), 1)
                            if _finite(p.get("rating"))
                            else None
                        )
                        p["match_rating"] = mr
                        if mr is not None:
                            match_ratings.append(mr)
                    rm["lobby_summary"] = {
                        "avg_rating": round(sum(match_ratings) / len(match_ratings), 1) if match_ratings else None,
                        "best_rating": round(max(match_ratings), 1) if match_ratings else None,
                    }
                    racing_matches.append(rm)
                racing_accounts.append(ri)
        racing_matches.sort(key=lambda x: x.get("created_at") or "", reverse=True)

        # Persist any newly observed exact round_match bindings
        bindings_updated = False
        for acc in accounts:
            nb = acc.pop("new_exact_binding", None)
            if nb:
                jid, pv = nb
                if known_bindings.get(jid) != pv:
                    known_bindings[jid] = pv
                    bindings_updated = True
        if bindings_updated:
            _save_version_bindings(known_bindings)

        # Collect unique completed job_ids to enrich (public_static + public_events)
        jobs_to_enrich: list[str] = []
        seen_jobs = set()
        for acc in accounts:
            count = 0
            for m in acc["recent_matches"]:
                if m["status"] == "succeeded" and count < enrich_limit_per_account:
                    count += 1
                    if m["job_id"] not in seen_jobs:
                        seen_jobs.add(m["job_id"])
                        jobs_to_enrich.append(m["job_id"])

        stats_by_job: dict[str, dict | None] = {}
        if jobs_to_enrich:
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
                fut_map = {pool.submit(get_or_fetch_match_stats, jid): jid for jid in jobs_to_enrich}
                for fut in concurrent.futures.as_completed(fut_map):
                    jid = fut_map[fut]
                    try:
                        stats_by_job[jid] = fut.result()
                    except Exception:
                        stats_by_job[jid] = None

        # Attach enriched own-seat stats & compute per-account dual-phase aggregates
        all_matches = []
        for acc in accounts:
            agg_overall = {"matches": 0, "wins": 0, "losses": 0, "rounds": 0, "k": 0, "d": 0, "a": 0, "dmg": 0.0, "fires": 0, "hits": 0, "hs": 0, "faults": 0, "untraded": 0}
            agg_attack = {"rounds": 0, "wins": 0, "k": 0, "d": 0, "a": 0, "dmg": 0.0, "fires": 0, "hits": 0, "hs": 0, "untraded": 0, "plants": 0}
            agg_defend = {"rounds": 0, "wins": 0, "k": 0, "d": 0, "a": 0, "dmg": 0.0, "fires": 0, "hits": 0, "hs": 0, "untraded": 0, "defuses": 0}

            for m in acc["recent_matches"]:
                _enrich_match_lobby(m, lb_by_name)
                st = stats_by_job.get(m["job_id"])
                if st and str(m["own_seat"]) in st.get("scoreboard", {}):
                    sb_own = st["scoreboard"][str(m["own_seat"])]
                    # Fix score from final round_end if match list didn't include team_scores
                    if (m["own_score"] is None or m["opp_score"] is None) and st.get("rounds"):
                        last_score = st["rounds"][-1].get("score")
                        if isinstance(last_score, list) and len(last_score) >= 2:
                            m["own_score"] = last_score[m["own_team"]]
                            m["opp_score"] = last_score[1 - m["own_team"]]
                            if m["status"] == "succeeded":
                                m["outcome"] = "W" if m["own_score"] > m["opp_score"] else "L" if m["own_score"] < m["opp_score"] else "D"
                    m["map_id"] = st["map_id"]
                    m["total_rounds"] = st["total_rounds"]
                    m["own_stats"] = {
                        "overall": sb_own["overall"],
                        "attack": sb_own["attack"],
                        "defend": sb_own["defend"],
                    }
                    # Accumulate account summary across enriched matches
                    ov = sb_own["overall"]
                    at = sb_own["attack"]
                    df = sb_own["defend"]
                    agg_overall["matches"] += 1
                    if m["outcome"] == "W":
                        agg_overall["wins"] += 1
                    elif m["outcome"] == "L":
                        agg_overall["losses"] += 1
                    for k in ("rounds", "k", "d", "a", "dmg", "fires", "hits", "hs", "faults"):
                        agg_overall[k] += ov.get(k, 0)
                    agg_overall["untraded"] += ov.get("untraded_deaths", 0)
                    for k in ("rounds", "wins", "k", "d", "a", "dmg", "fires", "hits", "hs", "plants"):
                        agg_attack[k] += at.get(k, 0)
                    agg_attack["untraded"] += at.get("untraded_deaths", 0)
                    for k in ("rounds", "wins", "k", "d", "a", "dmg", "fires", "hits", "hs", "defuses"):
                        agg_defend[k] += df.get(k, 0)
                    agg_defend["untraded"] += df.get("untraded_deaths", 0)
                else:
                    m["map_id"] = None
                    m["total_rounds"] = None
                    m["own_stats"] = None
                all_matches.append(m)

            def _rate(b: dict) -> dict:
                r = max(1, b["rounds"])
                d = max(1, b["d"])
                return {
                    **b,
                    "dmg": round(b["dmg"], 1),
                    "kd": round(b["k"] / d, 2),
                    "kda": round((b["k"] + b["a"]) / d, 2),
                    "adr": round(b["dmg"] / r, 1) if b["rounds"] > 0 else 0.0,
                    "acc_pct": round(100.0 * b["hits"] / b["fires"], 1) if b["fires"] > 0 else 0.0,
                    "hs_pct": round(100.0 * b["hs"] / b["hits"], 1) if b["hits"] > 0 else 0.0,
                }

            acc["recent_summary"] = {
                "overall": _rate(agg_overall),
                "attack": _rate(agg_attack),
                "defend": _rate(agg_defend),
            }

        all_matches.sort(key=lambda x: x.get("created_at") or "", reverse=True)
        # Global summary across racing accounts
        active_r = [r for r in racing_accounts if (r.get("standing") or {}).get("games", 0) > 0]
        r_ranks = [r["standing"]["rank"] for r in racing_accounts if (r.get("standing") or {}).get("rank") is not None]
        r_ratings = [r["standing"]["rating"] for r in active_r if (r.get("standing") or {}).get("rating") is not None]
        racing_summary = {
            "total_accounts": len(racing_accounts),
            "active_accounts": len(active_r),
            "total_games": sum((r.get("standing") or {}).get("games", 0) for r in racing_accounts),
            "best_rank": min(r_ranks) if r_ranks else None,
            "best_rating": round(max(r_ratings), 2) if r_ratings else None,
            "avg_rating": round(sum(r_ratings) / len(r_ratings), 2) if r_ratings else None,
            "total_net_gain": round(sum(r - 1000.0 for r in r_ratings), 2) if r_ratings else 0.0,
        }

        payload = {
            "schema_version": 1,
            "refreshed_at": now_iso(),
            "started_at": started,
            "competition_id": COMPETITION_ID,
            "leaderboard_url": f"https://agx.cool/#/competitions/{COMPETITION_ID}/leaderboard",
            "racing_competition_id": RACING_COMPETITION_ID,
            "racing_leaderboard_url": f"https://agx.cool/#/competitions/{RACING_COMPETITION_ID}/leaderboard",
            "accounts": accounts,
            "matches": all_matches,
            "racing_accounts": racing_accounts,
            "racing_matches": racing_matches,
            "racing_summary": racing_summary,
        }
        _safe_atomic_write(STATE_SNAPSHOT_PATH, payload)
        return payload


def load_cached_state() -> dict | None:
    if STATE_SNAPSHOT_PATH.is_file():
        try:
            return json.loads(STATE_SNAPSHOT_PATH.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


# ==============================================================================
# Racing 2D Full-Track Multi-Lane Replay Decoder & Progressive Streamer
# ==============================================================================

_RACING_PROG_GUARD = threading.Lock()
_RACING_PROG_JOBS: dict[str, dict] = {}
_COMPOUND_CODE = {"soft": "S", "medium": "M", "hard": "H", "inter": "I", "wet": "W"}


def _build_racing_skeleton(
    job_id: str,
    meta: dict,
    participants_meta: list[dict] | None = None,
) -> dict:
    trk = meta.get("track") or {}
    pit = trk.get("pit") or {}
    pit_cl = [[round(float(pt["x"]), 2), round(float(pt["y"]), 2)] for pt in (pit.get("centerline") or []) if "x" in pt and "y" in pt]

    meta_cars = meta.get("cars") or []
    p_by_seat = {int(p["seat"]): p for p in (participants_meta or []) if "seat" in p}
    participants = []
    for s in range(10):
        mc = next((c for c in meta_cars if int(c.get("seat", -1)) == s), {})
        mp = p_by_seat.get(s, {})
        liv = (mc.get("livery") or {}).get("primary_rgb")
        participants.append({
            "seat": s,
            "display_name": mp.get("display_name") or mc.get("display_name") or f"Car {s+1}",
            "is_mine": bool(mp.get("is_mine", False)),
            "place": mp.get("place"),
            "rank": mp.get("rank"),
            "rating": mp.get("match_rating") or mp.get("rating"),
            "rgb": liv if isinstance(liv, list) and len(liv) == 3 else None,
        })

    track_name = trk.get("display_name")
    if isinstance(track_name, dict):
        track_name = track_name.get("zh") or track_name.get("en") or trk.get("id")
    return {
        "schema_version": 1,
        "game_id": "racing",
        "job_id": job_id,
        "track": {
            "id": trk.get("id") or "circuit",
            "name": track_name or "Circuit",
            "length_m": float(trk.get("length_m") or 5000.0),
            "race_laps": int(meta.get("race_laps") or 4),
            "sectors": trk.get("sectors") or [],
            "pit_centerline": pit_cl,
            "loop": [],
            "bounds": [-500, -500, 500, 500],
        },
        "participants": participants,
        "phases": {},
        "built_at": now_iso(),
    }


def _consume_racing_lines(
    lines: list[bytes],
    bundle: dict,
    state_ctx: dict,
) -> str | None:
    stride = state_ctx["stride"]
    line_idx = state_ctx["line_idx"]
    track_bins = state_ctx["track_bins"]
    n_bins = state_ctx["n_bins"]
    length_m = bundle["track"]["length_m"]
    last_phase = state_ctx["last_phase"]

    for raw_line in lines:
        if not raw_line:
            continue
        line_idx += 1
        if (line_idx % stride) != 1:
            continue
        fr = json.loads(raw_line)
        phase = fr.get("phase") or "race"
        last_phase = phase
        t = round(float(fr.get("time") or 0.0), 2)
        ph = bundle["phases"].get(phase)
        if ph is None:
            ph = {
                "name": phase,
                "min_t": t,
                "max_t": t,
                "tracks": {str(s): [] for s in range(10)},
            }
            bundle["phases"][phase] = ph
        ph["max_t"] = max(ph["max_t"], t)

        for c in fr.get("cars", []):
            s = int(c["seat"])
            skey = str(s)
            if skey not in ph["tracks"]:
                continue
            x = float(c["x"])
            y = float(c["y"])
            # Update track loop geometry from non-pit cars
            if not c.get("in_pit") and len(track_bins) < n_bins:
                s_val = float(c.get("s") or 0.0) % length_m
                b = int((s_val / length_m) * n_bins) % n_bins
                if b not in track_bins:
                    track_bins[b] = [round(x, 2), round(y, 2)]

            spd_kph = round(float(c.get("speed_mps") or 0.0) * 3.6, 1)
            comp = _COMPOUND_CODE.get(str(c.get("compound") or "").lower(), "S")
            ers_pct = int(round(min(100.0, max(0.0, float(c.get("ers_soc_mj") or 0.0) / 4.0 * 100.0))))
            # Compact tuple: [t, x, y, yaw, speed_kph, lap, pos, in_pit(0/1), comp, ers_pct]
            ph["tracks"][skey].append([
                t,
                round(x, 2),
                round(y, 2),
                round(float(c.get("yaw") or 0.0), 3),
                spd_kph,
                int(c.get("lap") or 0),
                int(c.get("position") or (s + 1)),
                1 if c.get("in_pit") else 0,
                comp,
                ers_pct,
            ])

    state_ctx["line_idx"] = line_idx
    state_ctx["last_phase"] = last_phase

    # Rebuild sorted track loop + bounds if we have points
    if len(track_bins) >= 24:
        pts = [track_bins[k] for k in sorted(track_bins.keys())]
        bundle["track"]["loop"] = pts
        all_pts = pts + (bundle["track"].get("pit_centerline") or [])
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        bundle["track"]["bounds"] = [round(min(xs), 1), round(min(ys), 1), round(max(xs), 1), round(max(ys), 1)]
    return last_phase


def _run_racing_progressive_worker(job_id: str, desc: dict, job_state: dict) -> None:
    meta = desc.get("streams", {}).get("public_frames") or {}
    n_chunks = int(meta.get("chunks", 0))
    bundle = job_state["bundle"]
    frame_hz = float(job_state["frame_hz"] or 24.0)
    stride = max(1, int(round(frame_hz / 4.0)))  # 4 Hz smooth playback

    parse_ctx = {"stride": stride, "line_idx": 0, "track_bins": {}, "n_bins": 240, "last_phase": None}
    n_workers = min(16, max(1, n_chunks))
    clients = [_KeepAliveClient(timeout=30) for _ in range(n_workers)]
    pending = b""
    batch_size = 32
    try:
        for start in range(0, n_chunks, batch_size):
            batch_indices = list(range(start, min(n_chunks, start + batch_size)))
            parts = _download_chunks_keepalive(job_id, desc, "public_frames", batch_indices, clients)
            chunk_blob = pending + b"".join(parts)
            lines = chunk_blob.split(b"\n")
            pending = lines.pop()
            with job_state["lock"]:
                _consume_racing_lines(lines, bundle, parse_ctx)
                job_state["chunks_done"] = start + len(batch_indices)

        if pending.strip():
            with job_state["lock"]:
                _consume_racing_lines([pending], bundle, parse_ctx)

        REPLAY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = REPLAY_CACHE_DIR / f"racing_{job_id}.json.gz"
        bundle["built_at"] = now_iso()
        gz_bytes = gzip.compress(json.dumps(bundle, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), compresslevel=5)
        tmp = cache_path.with_name(f".{cache_path.name}.tmp")
        tmp.write_bytes(gz_bytes)
        tmp.replace(cache_path)

        with job_state["lock"]:
            job_state["chunks_done"] = n_chunks
            job_state["status"] = "ready"
    except Exception as exc:
        with job_state["lock"]:
            job_state["status"] = "error"
            job_state["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        for c in clients:
            c.close()


def init_progressive_racing_replay(job_id: str, participants_meta: list[dict] | None = None) -> dict | None:
    REPLAY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = REPLAY_CACHE_DIR / f"racing_{job_id}.json.gz"
    if cache_path.is_file():
        return {"cached": True, "job_id": job_id}

    with _RACING_PROG_GUARD:
        existing = _RACING_PROG_JOBS.get(job_id)
        if existing is not None and existing["status"] in ("loading", "ready"):
            with existing["lock"]:
                return {
                    "cached": False,
                    "job_id": job_id,
                    "status": existing["status"],
                    "chunks_done": existing["chunks_done"],
                    "chunks_total": existing["chunks_total"],
                    "skeleton": {
                        "schema_version": 1,
                        "game_id": "racing",
                        "job_id": job_id,
                        "track": existing["bundle"]["track"],
                        "participants": existing["bundle"]["participants"],
                        "phases": existing["bundle"]["phases"],
                    },
                }

    desc_res = agx_get_json(f"/api/v1/python-submissions/recordings/{job_id}/replay/descriptor", alias=None, timeout=20)
    if desc_res.get("http_status") != 200 or not isinstance(desc_res.get("body"), dict):
        return None
    desc = desc_res["body"]
    if desc.get("complete") is not True:
        return None

    static_bytes = _download_stream_bytes(job_id, desc, "public_static", max_workers=2)
    static_rows = [json.loads(ln.decode("utf-8")) for ln in static_bytes.splitlines() if ln.strip()]
    meta = static_rows[-1].get("metadata") or static_rows[0].get("metadata") or {}
    bundle = _build_racing_skeleton(job_id, meta, participants_meta)
    n_chunks = int(desc.get("streams", {}).get("public_frames", {}).get("chunks", 0))

    job_state = {
        "lock": threading.Lock(),
        "status": "loading",
        "error": None,
        "chunks_done": 0,
        "chunks_total": n_chunks,
        "bundle": bundle,
        "frame_hz": float(meta.get("frame_hz") or 24.0),
    }
    with _RACING_PROG_GUARD:
        _RACING_PROG_JOBS[job_id] = job_state

    th = threading.Thread(target=_run_racing_progressive_worker, args=(job_id, desc, job_state), daemon=True)
    th.start()

    return {
        "cached": False,
        "job_id": job_id,
        "status": "loading",
        "chunks_done": 0,
        "chunks_total": n_chunks,
        "skeleton": {
            "schema_version": 1,
            "game_id": "racing",
            "job_id": job_id,
            "track": bundle["track"],
            "participants": bundle["participants"],
            "phases": {},
        },
    }


def poll_progressive_racing_replay(job_id: str) -> dict:
    with _RACING_PROG_GUARD:
        job_state = _RACING_PROG_JOBS.get(job_id)
    if job_state is None:
        cache_path = REPLAY_CACHE_DIR / f"racing_{job_id}.json.gz"
        if cache_path.is_file():
            return {"status": "cached", "chunks_done": 1, "chunks_total": 1}
        return {"status": "missing", "error": "Racing replay job not initialized"}

    with job_state["lock"]:
        return {
            "status": job_state["status"],
            "error": job_state["error"],
            "chunks_done": job_state["chunks_done"],
            "chunks_total": job_state["chunks_total"],
            "loop_pts": len((job_state["bundle"].get("track") or {}).get("loop") or []),
        }


def get_cached_racing_replay_bytes(job_id: str) -> bytes | None:
    cache_path = REPLAY_CACHE_DIR / f"racing_{job_id}.json.gz"
    if cache_path.is_file():
        return cache_path.read_bytes()
    return None
