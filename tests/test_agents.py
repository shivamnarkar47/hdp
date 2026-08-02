"""Agent persona definitions + persistence tests (.kala/agents.json)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from harness.agents import (
    DEFAULT_AGENTS,
    active_agent,
    agent_names,
    load,
    save,
)


class TestAgents(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_default_agents_are_the_five_pandavas(self):
        """Exactly five defaults, the Pandavas, each with a description."""
        self.assertEqual(len(DEFAULT_AGENTS), 5)
        names = [a["name"] for a in DEFAULT_AGENTS]
        self.assertEqual(names, ["Yudhishthira", "Bhima", "Arjuna", "Nakula", "Sahadeva"])
        for agent in DEFAULT_AGENTS:
            self.assertIn("description", agent)
            self.assertTrue(agent["description"])

    def test_load_missing_file_seeds_defaults(self):
        """Missing file -> seeded 5, active None; load never writes."""
        data = load(self.root)
        self.assertEqual([a["name"] for a in data["agents"]], [a["name"] for a in DEFAULT_AGENTS])
        self.assertIsNone(data["active"])
        self.assertFalse((self.root / ".kala" / "agents.json").exists())

    def test_save_load_round_trip(self):
        """Custom agents + active survive a save/load round-trip."""
        data = {
            "agents": [
                {"name": "Karna", "description": "the relentless executor"},
                {"name": "Arjuna", "description": "precise"},
            ],
            "active": "Karna",
        }
        save(self.root, data)
        self.assertTrue((self.root / ".kala" / "agents.json").is_file())
        self.assertEqual(load(self.root), data)

    def test_load_corrupt_file_seeds_defaults(self):
        path = self.root / ".kala" / "agents.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{ definitely not json !!!", encoding="utf-8")
        data = load(self.root)
        self.assertEqual(len(data["agents"]), 5)
        self.assertIsNone(data["active"])

    def test_load_malformed_shape_seeds_defaults(self):
        """A valid-JSON-but-wrong-shape file (agents not a list of dicts)."""
        path = self.root / ".kala" / "agents.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"agents": "nope", "active": 3}), encoding="utf-8")
        data = load(self.root)
        self.assertEqual(len(data["agents"]), 5)
        self.assertIsNone(data["active"])

    def test_active_agent_resolves_by_name(self):
        data = {"agents": DEFAULT_AGENTS, "active": "Arjuna"}
        self.assertEqual(active_agent(data)["name"], "Arjuna")
        # Bad name -> None; no active -> None.
        self.assertIsNone(active_agent({"agents": DEFAULT_AGENTS, "active": "Bogus"}))
        self.assertIsNone(active_agent({"agents": DEFAULT_AGENTS, "active": None}))
        self.assertIsNone(active_agent({"agents": [], "active": "Arjuna"}))

    def test_agent_names(self):
        data = {"agents": DEFAULT_AGENTS, "active": None}
        self.assertEqual(agent_names(data), [a["name"] for a in DEFAULT_AGENTS])
        self.assertEqual(agent_names({"agents": [], "active": None}), [])


if __name__ == "__main__":
    unittest.main()
