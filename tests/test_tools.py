"""Tool registry tests: schemas, execution, path safety, DENY list, caps."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from harness.tools import (
    DENY_PATTERNS,
    MAX_RESULT_CHARS,
    TRUNCATED_SUFFIX,
    ToolError,
    ToolRegistry,
)


class TestToolRegistry(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.project = Path(self._tmp.name)
        self.reg = ToolRegistry(project_dir=self.project)

    def make(self, rel: str, content: str = "") -> Path:
        path = self.project / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def test_edit_old_text_not_found(self):
        self.make("a.txt", "hello world")
        result = self.reg.execute("edit", {"path": "a.txt", "old_text": "missing", "new_text": "x"})
        self.assertEqual(result, "old_text not found")

    def test_write_outside_cwd_rejected(self):
        result = self.reg.execute("write", {"path": "../evil.txt", "content": "x"})
        self.assertTrue(result.startswith("blocked: "))
        self.assertFalse((self.project.parent / "evil.txt").exists())

    def test_bash_deny_list(self):
        result = self.reg.execute("bash", {"command": "rm -rf /tmp/x"})
        self.assertEqual(result, "blocked by harness policy (destructive command)")

    def test_bash_allow_dangerous_disables_deny_check(self):
        reg = ToolRegistry(project_dir=self.project, allow_dangerous=True)
        result = reg.execute("bash", {"command": 'echo "git push"'})
        self.assertIn("git push", result)

    def test_deny_patterns_match(self):
        self.assertTrue(any(p.search("rm -rf /") for p in DENY_PATTERNS))
        self.assertTrue(any(p.search("git reset --hard HEAD") for p in DENY_PATTERNS))
        self.assertFalse(any(p.search("echo hello") for p in DENY_PATTERNS))

    def test_grep_skips_deny_dirs(self):
        self.make("node_modules/x.txt", "needle\n")
        self.make("src/y.txt", "needle\n")
        result = self.reg.execute("grep", {"pattern": "needle"})
        self.assertIn("src/y.txt:1: needle", result)
        self.assertNotIn("node_modules", result)

    def test_bash_output_capped(self):
        result = self.reg.execute("bash", {"command": "python3 -c \"print('x' * 20000)\""})
        self.assertLessEqual(len(result), MAX_RESULT_CHARS + len(TRUNCATED_SUFFIX))
        self.assertTrue(result.endswith(TRUNCATED_SUFFIX))

    def test_read_output_capped(self):
        self.make("big.txt", "x" * 20_000)
        result = self.reg.execute("read", {"path": "big.txt"})
        self.assertLessEqual(len(result), MAX_RESULT_CHARS + len(TRUNCATED_SUFFIX))
        self.assertTrue(result.endswith(TRUNCATED_SUFFIX))

    def test_unknown_tool_raises(self):
        with self.assertRaises(ToolError):
            self.reg.execute("no_such_tool", {})

    def test_non_dict_args_raise(self):
        with self.assertRaises(ToolError):
            self.reg.execute("bash", ["echo", "hi"])

    def test_write_inside_cwd_round_trip(self):
        result = self.reg.execute("write", {"path": "notes.txt", "content": "hello\nworld"})
        self.assertTrue(result.startswith("wrote notes.txt"))
        self.assertIn("bytes", result)
        self.assertEqual(self.reg.execute("read", {"path": "notes.txt"}), "hello\nworld")
        self.assertEqual((self.project / "notes.txt").read_text(encoding="utf-8"), "hello\nworld")

    def test_read_missing_file(self):
        result = self.reg.execute("read", {"path": "nope.txt"})
        self.assertTrue(result.startswith("read: "))

    def test_read_directory_listing(self):
        self.make("src/a.py", "x")
        self.make("src/nested/b.py", "x")
        self.make("top.txt", "x")
        result = self.reg.execute("read", {"path": "."})
        self.assertIn("src/", result)
        self.assertIn("src/nested/", result)
        self.assertIn("top.txt", result)

    def test_read_offset_limit(self):
        self.make("lines.txt", "one\ntwo\nthree\nfour\n")
        result = self.reg.execute("read", {"path": "lines.txt", "offset": 2, "limit": 2})
        self.assertEqual(result, "two\nthree\n")

    def test_edit_multiple_matches_requires_all(self):
        self.make("b.txt", "a b a")
        result = self.reg.execute("edit", {"path": "b.txt", "old_text": "a", "new_text": "x"})
        self.assertEqual(result, "old_text matches 2 times; pass all=true to replace all")
        self.assertEqual((self.project / "b.txt").read_text(encoding="utf-8"), "a b a")

    def test_edit_single_replacement(self):
        self.make("d.txt", "a b")
        result = self.reg.execute("edit", {"path": "d.txt", "old_text": "a", "new_text": "x"})
        self.assertTrue(result.startswith("edited d.txt"))
        self.assertEqual((self.project / "d.txt").read_text(encoding="utf-8"), "x b")

    def test_edit_all_replaces_everywhere(self):
        self.make("c.txt", "a b a a")
        result = self.reg.execute("edit", {"path": "c.txt", "old_text": "a", "new_text": "x", "all": True})
        self.assertTrue(result.startswith("edited c.txt"))
        self.assertEqual((self.project / "c.txt").read_text(encoding="utf-8"), "x b x x")

    def test_memory_append_without_store(self):
        result = self.reg.execute("memory_append", {"section": "decisions", "text": "note"})
        self.assertEqual(result, "memory_append: no memory store configured")

    def test_memory_append_invalid_section(self):
        result = self.reg.execute("memory_append", {"section": "bogus", "text": "note"})
        self.assertTrue(result.startswith("memory_append: invalid section"))

    def test_memory_append_calls_injected_store(self):
        class FakeMemory:
            def __init__(self):
                self.calls = []

            def append(self, section, text):
                self.calls.append((section, text))
                return "already recorded"

        memory = FakeMemory()
        reg = ToolRegistry(memory=memory, project_dir=self.project)
        result = reg.execute("memory_append", {"section": "patterns", "text": "use boring code"})
        self.assertEqual(result, "already recorded")
        self.assertEqual(memory.calls, [("patterns", "use boring code")])

    def test_schemas_cover_all_tools(self):
        names = {schema["function"]["name"] for schema in self.reg.schemas()}
        self.assertEqual(
            names,
            {"read", "grep", "glob", "write", "edit", "bash", "memory_append"},
        )
        for schema in self.reg.schemas():
            self.assertEqual(schema["type"], "function")
            self.assertIn("description", schema["function"])
            self.assertEqual(schema["function"]["parameters"]["type"], "object")


if __name__ == "__main__":
    unittest.main()
