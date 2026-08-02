"""User API-key store and resolution-order tests (offline, deterministic)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from harness import config


class TestUserKeyStore(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._old_xdg = os.environ.get("XDG_CONFIG_HOME")
        os.environ["XDG_CONFIG_HOME"] = str(self.root / "config")

    def tearDown(self) -> None:
        if self._old_xdg is None:
            os.environ.pop("XDG_CONFIG_HOME", None)
        else:
            os.environ["XDG_CONFIG_HOME"] = self._old_xdg
        self._tmp.cleanup()

    def test_save_load_round_trip(self):
        path = config.user_key_path()
        self.assertIsNone(config.load_user_api_key())
        config.save_user_api_key("sk-roundtrip")
        self.assertEqual(path.read_text(encoding="utf-8"), "sk-roundtrip")  # no newline
        self.assertEqual(config.load_user_api_key(), "sk-roundtrip")
        if os.name != "nt":
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_env_wins_over_user_store(self):
        config.save_user_api_key("sk-user")
        with mock.patch.dict(os.environ, {"OPENCODE_API_KEY": "sk-env"}):
            self.assertEqual(config.get_api_key(), "sk-env")

    def test_user_store_wins_over_omp_db(self):
        config.save_user_api_key("sk-user")
        with mock.patch.dict(os.environ, {"OPENCODE_API_KEY": ""}):
            with mock.patch.object(Path, "home", return_value=self.root / "nohome"):
                # ~/nohome/.omp/agent/agent.db does not exist -> omp store misses
                self.assertEqual(config.get_api_key(), "sk-user")


if __name__ == "__main__":
    unittest.main()
