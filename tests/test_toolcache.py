"""Tool cache tests: keying, persistence, atomic writes, corruption handling."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from harness.toolcache import ToolCache


class TestToolCache(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / ".kaal" / "tool-cache.json"

    # -- put/get ------------------------------------------------------------

    def test_put_get_round_trip(self):
        cache = ToolCache(self.path)
        cache.put("read", '{"path": "a.txt"}', "sig1", "contents of a")
        self.assertEqual(cache.get("read", '{"path": "a.txt"}', "sig1"), "contents of a")
        self.assertTrue(self.path.is_file())

    def test_different_signature_misses(self):
        """A changed structure signature (tree edited) must miss the old key."""
        cache = ToolCache(self.path)
        cache.put("read", '{"path": "a.txt"}', "sig1", "stale")
        self.assertIsNone(cache.get("read", '{"path": "a.txt"}', "sig2"))

    def test_different_args_misses(self):
        cache = ToolCache(self.path)
        cache.put("read", '{"path": "a.txt"}', "sig1", "stale")
        self.assertIsNone(cache.get("read", '{"path": "b.txt"}', "sig1"))

    def test_different_tool_misses(self):
        cache = ToolCache(self.path)
        cache.put("read", '{"path": "a.txt"}', "sig1", "stale")
        self.assertIsNone(cache.get("grep", '{"path": "a.txt"}', "sig1"))

    def test_missing_file_returns_none(self):
        cache = ToolCache(self.path)
        self.assertIsNone(cache.get("read", '{"path": "a.txt"}', "sig1"))

    # -- drop ---------------------------------------------------------------

    def test_drop_clears(self):
        cache = ToolCache(self.path)
        cache.put("read", '{"path": "a.txt"}', "sig1", "x")
        self.assertTrue(self.path.exists())
        cache.drop()
        self.assertFalse(self.path.exists())
        # The in-memory copy is forgotten too: a later get misses.
        self.assertIsNone(cache.get("read", '{"path": "a.txt"}', "sig1"))

    # -- size cap -----------------------------------------------------------

    def test_oversized_put_does_not_grow_file(self):
        cache = ToolCache(self.path, max_bytes=100)
        cache.put("read", '{"path": "a.txt"}', "sig1", "small")
        self.assertTrue(self.path.is_file())
        size_before = self.path.stat().st_size
        cache.put("read", '{"path": "b.txt"}', "sig1", "x" * 500)
        self.assertEqual(self.path.stat().st_size, size_before)

    # -- corruption / atomicity ----------------------------------------------

    def test_corrupt_file_tolerated(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text("{not valid json", encoding="utf-8")
        cache = ToolCache(self.path)
        self.assertIsNone(cache.get("read", '{"path": "a.txt"}', "sig1"))
        # A put after corruption rewrites a clean, loadable file.
        cache.put("read", '{"path": "a.txt"}', "sig1", "fresh")
        self.assertEqual(cache.get("read", '{"path": "a.txt"}', "sig1"), "fresh")

    def test_non_dict_payload_tolerated(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text("[1, 2, 3]", encoding="utf-8")
        cache = ToolCache(self.path)
        self.assertIsNone(cache.get("read", '{"path": "a.txt"}', "sig1"))

    def test_atomic_write_leaves_no_tmp(self):
        cache = ToolCache(self.path)
        cache.put("read", '{"path": "a.txt"}', "sig1", "x")
        cache.put("read", '{"path": "b.txt"}', "sig1", "y")
        self.assertEqual(list(self.path.parent.glob("tool-cache.json.tmp*")), [])
        self.assertTrue(self.path.is_file())

    # -- persistence ----------------------------------------------------------

    def test_persists_across_instances(self):
        ToolCache(self.path).put("read", '{"path": "a.txt"}', "sig1", "persisted")
        again = ToolCache(self.path)
        self.assertEqual(again.get("read", '{"path": "a.txt"}', "sig1"), "persisted")

    def test_stored_json_shape(self):
        cache = ToolCache(self.path)
        cache.put("read", '{"path": "a.txt"}', "sig1", "x")
        parsed = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertIsInstance(parsed, dict)
        key = next(iter(parsed))
        # tool|sha256(args_json)|signature — the args digest keeps the key
        # short while the signature in the key makes changed trees miss.
        tool, digest, sig = key.split("|")
        self.assertEqual(tool, "read")
        self.assertEqual(len(digest), 64)
        self.assertEqual(sig, "sig1")


if __name__ == "__main__":
    unittest.main()
