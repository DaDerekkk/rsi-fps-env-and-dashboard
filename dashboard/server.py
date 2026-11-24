"""Loopback-only local HTTP server for the FPS Multi-Account Live Dashboard.

Strictly read-only toward AGX (GET-only via api_client.py). Never exposes
credentials, release gates, or arbitrary filesystem paths.
"""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
import uuid

from api_client import (
    get_cached_racing_replay_bytes,
    get_or_fetch_match_stats,
    get_or_fetch_replay_bundle,
    init_progressive_racing_replay,
    init_progressive_replay,
    load_cached_state,
    poll_progressive_racing_replay,
    poll_progressive_replay,
    refresh_live_state,
)

WEB_DIR = Path(__file__).resolve().parent
STATIC_ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/replay.js": ("replay.js", "text/javascript; charset=utf-8"),
}


def _valid_uuid(raw: str) -> str | None:
    try:
        return str(uuid.UUID(raw))
    except (ValueError, TypeError):
        return None


def _find_match_meta(job_id: str) -> dict | None:
    state = load_cached_state()
    if not state:
        return None
    for m in state.get("matches", []) + state.get("racing_matches", []):
        if m.get("job_id") == job_id:
            return m
    return None


class DashboardHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        return

    def _check_host(self) -> bool:
        host = (self.headers.get("Host") or "").split(":")[0]
        if host not in ("127.0.0.1", "localhost"):
            self.send_error(403, "Loopback host only")
            return False
        return True

    def _send_bytes(
        self,
        body: bytes,
        mime: str,
        status: int = 200,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        headers = {
            "Content-Type": mime,
            "Content-Length": str(len(body)),
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": (
                "default-src 'self'; "
                "style-src 'self' 'unsafe-inline'; "
                "script-src 'self'; "
                "img-src 'self' data:; "
                "object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
            ),
        }
        if extra_headers:
            headers.update(extra_headers)
        for k, v in headers.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj: dict, status: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self._send_bytes(body, "application/json; charset=utf-8", status=status)

    def do_GET(self) -> None:
        if not self._check_host():
            return
        path = urlsplit(self.path).path

        if path in STATIC_ASSETS:
            fname, mime = STATIC_ASSETS[path]
            self._send_bytes((WEB_DIR / fname).read_bytes(), mime)
            return

        if path == "/api/state":
            state = load_cached_state()
            if state is None:
                try:
                    state = refresh_live_state()
                except Exception as exc:
                    self._send_json({"error": f"Initial refresh failed: {type(exc).__name__}"}, status=503)
                    return
            self._send_json(state)
            return

        if path.startswith("/api/match/"):
            job_id = _valid_uuid(path.removeprefix("/api/match/"))
            if not job_id:
                self._send_json({"error": "Invalid job_id"}, status=400)
                return
            try:
                stats = get_or_fetch_match_stats(job_id)
            except Exception as exc:
                self._send_json({"error": f"Match stats unavailable: {type(exc).__name__}"}, status=502)
                return
            if not stats:
                self._send_json({"error": "Match replay stats not ready yet"}, status=404)
                return
            meta = _find_match_meta(job_id)
            self._send_json(
                {
                    "job_id": job_id,
                    "agx_url": f"https://agx.cool/#/matches/{job_id}",
                    "meta": meta,
                    "stats": stats,
                }
            )
            return

        if path.startswith("/api/racing-replay-init/"):
            job_id = _valid_uuid(path.removeprefix("/api/racing-replay-init/"))
            if not job_id:
                self._send_json({"error": "Invalid job_id"}, status=400)
                return
            meta = _find_match_meta(job_id)
            participants = (meta or {}).get("participants")
            try:
                res = init_progressive_racing_replay(job_id, participants_meta=participants)
            except Exception as exc:
                self._send_json({"error": f"Racing replay init failed: {type(exc).__name__}"}, status=502)
                return
            if not res:
                self._send_json({"error": "Public racing replay not available"}, status=404)
                return
            self._send_json(res)
            return

        if path.startswith("/api/racing-replay-poll/"):
            job_id = _valid_uuid(path.removeprefix("/api/racing-replay-poll/"))
            if not job_id:
                self._send_json({"error": "Invalid job_id"}, status=400)
                return
            self._send_json(poll_progressive_racing_replay(job_id))
            return

        if path.startswith("/api/racing-replay/"):
            job_id = _valid_uuid(path.removeprefix("/api/racing-replay/"))
            if not job_id:
                self._send_json({"error": "Invalid job_id"}, status=400)
                return
            gz_bytes = get_cached_racing_replay_bytes(job_id)
            if not gz_bytes:
                self._send_json({"error": "Cached racing replay not found"}, status=404)
                return
            self._send_bytes(
                gz_bytes,
                "application/json; charset=utf-8",
                status=200,
                extra_headers={"Content-Encoding": "gzip"},
            )
            return

        if path.startswith("/api/replay-init/"):
            job_id = _valid_uuid(path.removeprefix("/api/replay-init/"))
            if not job_id:
                self._send_json({"error": "Invalid job_id"}, status=400)
                return
            meta = _find_match_meta(job_id)
            participants = (meta or {}).get("participants")
            try:
                res = init_progressive_replay(job_id, participants_meta=participants)
            except Exception as exc:
                self._send_json({"error": f"Replay init failed: {type(exc).__name__}"}, status=502)
                return
            if not res:
                self._send_json({"error": "Public replay not available for this match"}, status=404)
                return
            self._send_json(res)
            return

        if path.startswith("/api/replay-poll/"):
            job_id = _valid_uuid(path.removeprefix("/api/replay-poll/"))
            if not job_id:
                self._send_json({"error": "Invalid job_id"}, status=400)
                return
            qs = parse_qs(urlsplit(self.path).query)
            have_raw = (qs.get("have") or [""])[0]
            have_set = {int(x) for x in have_raw.split(",") if x.strip().isdigit()}
            res = poll_progressive_replay(job_id, have_set)
            self._send_json(res)
            return

        if path.startswith("/api/replay/"):
            job_id = _valid_uuid(path.removeprefix("/api/replay/"))
            if not job_id:
                self._send_json({"error": "Invalid job_id"}, status=400)
                return
            meta = _find_match_meta(job_id)
            participants = (meta or {}).get("participants")
            try:
                gz_bytes = get_or_fetch_replay_bundle(job_id, participants_meta=participants)
            except Exception as exc:
                self._send_json({"error": f"Replay load failed: {type(exc).__name__}"}, status=502)
                return
            if not gz_bytes:
                self._send_json({"error": "Public replay not available for this match"}, status=404)
                return
            self._send_bytes(
                gz_bytes,
                "application/json; charset=utf-8",
                status=200,
                extra_headers={"Content-Encoding": "gzip"},
            )
            return

        self.send_error(404, "Not found")

    def do_POST(self) -> None:
        if not self._check_host():
            return
        path = urlsplit(self.path).path
        if path == "/api/refresh":
            # Drain any small request body cleanly
            length = int(self.headers.get("Content-Length") or 0)
            if length > 0:
                self.rfile.read(min(length, 4096))
            try:
                state = refresh_live_state()
                self._send_json(state)
            except Exception as exc:
                self._send_json({"error": f"Refresh failed: {type(exc).__name__}"}, status=502)
            return
        self.send_error(404, "Not found")


def main() -> None:
    parser = argparse.ArgumentParser(description="FPS Multi-Account Live Dashboard Server")
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument("--no-initial-refresh", action="store_true")
    args = parser.parse_args()

    if not args.no_initial_refresh and load_cached_state() is None:
        print("Warming initial live snapshot (GET-only)...", flush=True)
        try:
            refresh_live_state()
        except Exception as exc:
            print(f"Initial warm failed (will retry on request): {exc}", flush=True)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), DashboardHandler)
    print(f"FPS Live Dashboard ready at: http://127.0.0.1:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
