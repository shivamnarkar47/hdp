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

    # -- cost estimation ----------------------------------------------------

    def test_estimate_cost_zero_tokens(self):
        self.assertEqual(config.estimate_cost(0, 0), 0.0)

    def test_estimate_cost_1m_each(self):
        # input $0.14/M + output $0.28/M = $0.42 per 1M tokens each.
        self.assertEqual(config.estimate_cost(1_000_000, 1_000_000), 0.42)

    def test_estimate_cost_cache_read_terms(self):
        self.assertEqual(config.estimate_cost(0, 0, 1_000_000), 0.0028)

    # -- model catalog / default model -------------------------------------

    def test_model_store_round_trip(self):
        self.assertIsNone(config.load_user_model())
        config.save_user_model("deepseek-v4-pro")
        self.assertEqual(config.load_user_model(), "deepseek-v4-pro")
        # Resolution order: flag > saved > MODEL_ID.
        self.assertEqual(config.resolve_model_id(None), "deepseek-v4-pro")
        self.assertEqual(config.resolve_model_id("kimi-k2.5"), "kimi-k2.5")

    def test_catalog_sorted_free_first_ascending(self):
        self.assertEqual(config.MODELS[0]["id"], "deepseek-v4-flash-free")
        rates = [m["input_per_m"] for m in config.MODELS]
        self.assertEqual(rates, sorted(rates))  # free (0) first, then ascending
        free = [m for m in config.MODELS if m.get("base_url")]
        self.assertTrue(free)
        self.assertTrue(all(m["input_per_m"] == 0.0 for m in free))
        self.assertTrue(all(m["base_url"] == config.FREE_BASE_URL for m in free))

    def test_model_rates_and_base_urls(self):
        self.assertEqual(config.model_rates("deepseek-v4-flash"), (0.14, 0.28))
        self.assertEqual(config.model_rates("deepseek-v4-flash-free"), (0.0, 0.0))
        self.assertEqual(config.model_rates("unknown-model"), (0.14, 0.28))
        self.assertEqual(config.model_base_url("deepseek-v4-flash"), config.BASE_URL)
        self.assertEqual(
            config.model_base_url("deepseek-v4-flash-free"), config.FREE_BASE_URL
        )

    def test_estimate_cost_uses_model_rates(self):
        # Free tier: zero. Pro: 0.435 + 0.87 = 1.305 per 1M each.
        self.assertEqual(
            config.estimate_cost(1_000_000, 1_000_000, model_id="deepseek-v4-flash-free"),
            0.0,
        )
        self.assertEqual(
            config.estimate_cost(1_000_000, 1_000_000, model_id="deepseek-v4-pro"),
            1.305,
        )


if __name__ == "__main__":
    unittest.main()
