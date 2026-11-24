"""Offline unit & security smoke tests for the Multi-Account Live Dashboard."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

DASH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DASH_DIR))


class DashboardSmokeTests(unittest.TestCase):
    def test_load_accounts_and_env_token(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            acc_p = Path(td) / "accounts.json"
            acc_p.write_text(
                (DASH_DIR / "accounts.example.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            os.environ["FPS_DASHBOARD_ACCOUNTS"] = str(acc_p)
            os.environ["AGX_TOKEN_MAIN_ACCOUNT"] = "synthetic_test_token_only"
            try:
                import importlib
                import api_client
                importlib.reload(api_client)
                roles = api_client.load_account_roles()
                self.assertEqual([r["alias"] for r in roles], ["main_account", "alt_account_01"])
                self.assertEqual(api_client._read_keychain("main_account"), "synthetic_test_token_only")
            finally:
                os.environ.pop("AGX_TOKEN_MAIN_ACCOUNT", None)
                os.environ.pop("FPS_DASHBOARD_ACCOUNTS", None)

    def test_disallowed_api_paths_rejected(self) -> None:
        import api_client
        with self.assertRaises(ValueError):
            api_client.agx_get_json("https://evil.example/api/v1/test")
        with self.assertRaises(ValueError):
            api_client.agx_get_json("/etc/passwd")

    def test_static_assets_present(self) -> None:
        import server
        for route, (fname, _) in server.STATIC_ASSETS.items():
            self.assertTrue((DASH_DIR / fname).is_file(), f"Missing static asset for {route}: {fname}")


if __name__ == "__main__":
    unittest.main()
