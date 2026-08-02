"""StructureManager tests: cache creation, signature refresh, caps (temp dirs)."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from harness.structure import MAX_ENTRIES, StructureManager


def _populate(root: Path) -> None:
    (root / "src").mkdir()
    (root / "src" / "main.py").write_text("print('hi')\n" * 30, encoding="utf-8")
    (root / "src" / "util.py").write_text("x = 1\n", encoding="utf-8")
    (root / "README.md").write_text("# readme\n", encoding="utf-8")
    for noise in ("node_modules", ".venv", "__pycache__", ".git"):
        (root / noise).mkdir()
        (root / noise / "junk.txt").write_text("junk\n", encoding="utf-8")


class TestStructureManager(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _manager(self) -> StructureManager:
        return StructureManager(self.root)

    def test_first_scan_creates_cache(self):
        _populate(self.root)
        mgr = self._manager()
        doc = mgr.ensure()
        path = mgr.cache_path
        self.assertTrue(path.is_file())
        self.assertEqual(path.read_text(encoding="utf-8"), doc)
        self.assertIn("# Project Structure", doc)
        self.assertIn("Root: ", doc)
        self.assertIn("Files: 3 · Dirs: 1", doc)
        self.assertIn("<!-- sig: ", doc)
        for noise in ("node_modules", ".venv", "__pycache__", ".git", "junk"):
            self.assertNotIn(noise, doc)
        self.assertIn("main.py", doc)
        self.assertIn("README.md", doc)

    def test_ensure_no_rescan_until_change(self):
        _populate(self.root)
        mgr = self._manager()
        doc1 = mgr.ensure()
        # Mutate a file's content AND mtime: shape is unchanged.
        p = self.root / "src" / "main.py"
        p.write_text("changed content that is longer\n", encoding="utf-8")
        st = p.stat()
        os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
        # ensure() reads the cache — no rescan.
        self.assertEqual(mgr.ensure(), doc1)
        # refresh() sees the new signature and regenerates (new size shown).
        doc2 = mgr.refresh()
        self.assertNotEqual(doc2, doc1)
        self.assertIn("main.py (31 B)", doc2)

    def test_tree_change_detection(self):
        _populate(self.root)
        mgr = self._manager()
        mgr.ensure()
        (self.root / "newfile.txt").write_text("n\n", encoding="utf-8")
        doc = mgr.refresh()
        self.assertIn("newfile.txt", doc)
        shutil.rmtree(self.root / "src")
        doc = mgr.refresh()
        self.assertNotIn("src", doc)
        self.assertIn("Files: 2 · Dirs: 0", doc)  # README.md + newfile.txt

    def test_digest_capped_and_contains_root(self):
        _populate(self.root)
        mgr = self._manager()
        mgr.ensure()
        dig = mgr.digest()
        self.assertLessEqual(len(dig), 4000 + len("\n… (truncated)"))
        self.assertIn(str(self.root), dig)
        small = mgr.digest(max_chars=60)
        self.assertLessEqual(len(small), 60 + len("\n… (truncated)"))

    def test_depth_cap_notes_ellipsis(self):
        d = self.root
        for i in range(8):
            d = d / f"level{i}"
            d.mkdir()
        (d / "deep.txt").write_text("x\n", encoding="utf-8")
        doc = self._manager().ensure()
        self.assertIn("level5/", doc)  # depth cap: deepest dir shown
        self.assertNotIn("deep.txt", doc)  # its children cut
        self.assertIn("…", doc)

    def test_entry_cap_notes_truncation(self):
        for i in range(20):
            (self.root / f"f{i}.txt").write_text("x\n", encoding="utf-8")
        with mock.patch("harness.structure.MAX_ENTRIES", 10):
            doc = self._manager().ensure()
        self.assertIn("structure truncated", doc)
        self.assertLess(doc.count("f"), 20)


if __name__ == "__main__":
    unittest.main()
