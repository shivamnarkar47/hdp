"""Tool registry tests: schemas, execution, path safety, DENY list, caps."""

import json
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from harness.tools import (
    DENY_PATTERNS,
    MAX_RESULT_CHARS,
    TRUNCATED_SUFFIX,
    ToolError,
    ToolRegistry,
)
from harness.toolcache import ToolCache


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
            {"read", "grep", "glob", "write", "edit", "bash", "memory_append", "spawn_agent"},
        )
        for schema in self.reg.schemas():
            self.assertEqual(schema["type"], "function")
            self.assertIn("description", schema["function"])
            self.assertEqual(schema["function"]["parameters"]["type"], "object")

    def test_spawn_agent_schema(self):
        schema = next(
            s for s in self.reg.schemas() if s["function"]["name"] == "spawn_agent"
        )
        params = schema["function"]["parameters"]
        self.assertEqual(params["required"], ["task"])
        self.assertEqual(params["properties"]["max_steps"]["maximum"], 5)
        self.assertEqual(params["properties"]["timeout"]["maximum"], 300)
        self.assertIn("dir", params["properties"])
        self.assertIn("session_id", schema["function"]["description"])

    def test_spawn_agent_without_handler(self):
        result = self.reg.execute("spawn_agent", {"task": "do the thing"})
        self.assertEqual(result, "spawn_agent: not available in this context")

    def test_spawn_agent_with_stub_handler(self):
        calls = []

        def handler(task, dir_, max_steps, timeout):
            calls.append((task, dir_, max_steps, timeout))
            return json.dumps({"answer": "done", "steps": 1, "usage": {}, "session_id": "n-1"})

        self.reg.set_spawn_handler(handler)
        result = self.reg.execute(
            "spawn_agent",
            {"task": "do it", "dir": "sub", "max_steps": 3, "timeout": 42},
        )
        self.assertEqual(json.loads(result)["answer"], "done")
        self.assertEqual(calls, [("do it", "sub", 3, 42)])

    def test_spawn_agent_handler_defaults(self):
        calls = []
        self.reg.set_spawn_handler(
            lambda task, dir_, max_steps, timeout: calls.append((task, dir_, max_steps, timeout))
            or "ok"
        )
        self.reg.execute("spawn_agent", {"task": "t"})
        self.assertEqual(calls, [("t", None, 5, 120)])  # dir None, max_steps 5, timeout 120

    def test_spawn_agent_clamps_out_of_range(self):
        calls = []
        self.reg.set_spawn_handler(
            lambda task, dir_, max_steps, timeout: calls.append((task, dir_, max_steps, timeout))
            or "ok"
        )
        self.reg.execute(
            "spawn_agent", {"task": "t", "max_steps": 99, "timeout": 9999}
        )
        self.assertEqual(calls, [("t", None, 5, 300)])
        self.reg.execute("spawn_agent", {"task": "t", "max_steps": 0, "timeout": 0})
        self.assertEqual(calls[-1], ("t", None, 1, 1))


    def test_read_range_on_huge_file(self):
        # 100k lines: a range read must stream only the requested window.
        content = "".join(f"line {i}\n" for i in range(1, 100_001))
        self.make("huge.txt", content)
        result = self.reg.execute("read", {"path": "huge.txt", "offset": 5, "limit": 3})
        self.assertEqual(result, "line 5\nline 6\nline 7\n")
        result = self.reg.execute("read", {"path": "huge.txt", "limit": 2})
        self.assertEqual(result, "line 1\nline 2\n")

    def test_edit_range_scoped_replacement(self):
        # "needle" appears at line 2 AND line 9; a range-scoped edit must
        # replace only the occurrence inside [9, 11), leaving line 2 intact.
        self.make(
            "r.txt",
            "line1\nneedle\nline3\nline4\nline5\nline6\nline7\nline8\nneedle\nline10\n",
        )
        result = self.reg.execute(
            "edit", {"path": "r.txt", "old_text": "needle", "new_text": "NINE", "offset": 9, "limit": 2}
        )
        self.assertTrue(result.startswith("edited r.txt"))
        self.assertEqual(
            (self.project / "r.txt").read_text(encoding="utf-8"),
            "line1\nneedle\nline3\nline4\nline5\nline6\nline7\nline8\nNINE\nline10\n",
        )

    def test_grep_stops_scanning_at_cap(self):
        # File A (visited first — root files precede subdirectories in walk
        # order) matches thousands of lines, so the cap is hit inside it.
        self.make("many.txt", "".join(f"needle {i}\n" for i in range(1, 3001)))
        # File B sits after the cap would be reached; its matching line must
        # never be scanned, so it must not appear in the result.
        self.make("sub/sentinel.txt", "needle sentinel\n")
        result = self.reg.execute("grep", {"pattern": "needle"})
        self.assertLessEqual(len(result), MAX_RESULT_CHARS + len(TRUNCATED_SUFFIX))
        self.assertTrue(result.endswith(TRUNCATED_SUFFIX))
        self.assertTrue(result.startswith("many.txt:1: needle 1"))
        self.assertNotIn("sentinel", result)

    def test_read_no_limit_caps_at_10k(self):
        # ~2M lines / ~20MB: a no-limit read must stop once it has enough
        # characters to decide the cap applies, not read the whole file.
        content = "".join(f"line {i}\n" for i in range(1, 2_000_001))
        self.make("huge2.txt", content)
        result = self.reg.execute("read", {"path": "huge2.txt"})
        self.assertEqual(len(result), MAX_RESULT_CHARS + len(TRUNCATED_SUFFIX))
        self.assertTrue(result.endswith(TRUNCATED_SUFFIX))
        self.assertTrue(result.startswith("line 1\n"))
        # offset without limit still honors the start line on the capped path.
        tail = self.reg.execute("read", {"path": "huge2.txt", "offset": 1_999_999})
        self.assertEqual(tail, "line 1999999\nline 2000000\n")

    def test_edit_range_beyond_eof(self):
        self.make("s.txt", "\n".join(f"line{i}" for i in range(1, 11)))
        result = self.reg.execute(
            "edit", {"path": "s.txt", "old_text": "line9", "new_text": "NINE", "offset": 90, "limit": 5}
        )
        self.assertEqual(result, "old_text not found")

    def test_bash_sanitized_path(self):
        result = self.reg.execute("bash", {"command": "echo $PATH"})
        self.assertTrue(result.startswith("/"))
        self.assertIn("/usr/bin", result)
        self.assertEqual(result.strip().split(":"), ["/usr/local/bin", "/usr/bin", "/bin"])

    def test_bash_sanitized_path_prepends_project_venv(self):
        (self.project / ".venv" / "bin").mkdir(parents=True)
        result = self.reg.execute("bash", {"command": "echo $PATH"})
        expected = [str(self.project / ".venv" / "bin"), "/usr/local/bin", "/usr/bin", "/bin"]
        self.assertEqual(result.strip().split(":"), expected)


    def test_grep_stops_scanning_at_cap(self):
        # File A (root level, visited first by os.walk): 3000 matching lines,
        # ~40KB of matches — the result cap is hit inside this file.
        self.make("many.txt", "".join(f"needle {i}\n" for i in range(1, 3001)))
        # File B (subdirectory, walked after root files): its matching line must
        # never be scanned once the cap is reached, so it cannot appear.
        self.make("sub/sentinel.txt", "needle sentinel\n")
        result = self.reg.execute("grep", {"pattern": "needle"})
        self.assertLessEqual(len(result), MAX_RESULT_CHARS + len(TRUNCATED_SUFFIX))
        self.assertTrue(result.endswith(TRUNCATED_SUFFIX))
        self.assertTrue(result.startswith("many.txt:1: needle 1"))
        self.assertNotIn("sentinel", result)

    def test_read_no_limit_caps_at_10k(self):
        # ~20MB / 2M lines: an unlimited read must fetch only the cap+slack
        # prefix, still returning the exact same truncated result as before.
        content = "".join(f"line {i}\n" for i in range(1, 2_000_001))
        self.make("huge2.txt", content)
        result = self.reg.execute("read", {"path": "huge2.txt"})
        self.assertEqual(len(result), MAX_RESULT_CHARS + len(TRUNCATED_SUFFIX))
        self.assertTrue(result.endswith(TRUNCATED_SUFFIX))
        self.assertTrue(result.startswith("line 1\n"))
        # offset without limit still honors the start line, then caps.
        tail = self.reg.execute("read", {"path": "huge2.txt", "offset": 1_999_999})
        self.assertEqual(tail, "line 1999999\nline 2000000\n")

    def test_schemas_memoized(self):
        # First call deep-copies the spec list once; later calls return a
        # fresh top-level list (so caller mutation can't corrupt the cache)
        # but reuse the same inner dicts.
        first = self.reg.schemas()
        second = self.reg.schemas()
        self.assertEqual(first, second)
        self.assertIsNot(first, second)  # shallow copy: distinct list object
        self.assertEqual(len(first), 8)
        for inner_first, inner_second in zip(first, second):
            self.assertIs(inner_first, inner_second)  # memoized inner dicts

    def test_grep_skips_kaal_dir(self):
        # The tool cache lives under .kaal and must never be scanned by grep.
        self.make(".kaal/tool-cache.json", '{"needle": "cached secret"}')
        self.make("visible.txt", "needle visible\n")
        result = self.reg.execute("grep", {"pattern": "needle"})
        self.assertIn("visible.txt:1: needle visible", result)
        self.assertNotIn("tool-cache", result)

    def test_grep_rg_python_equivalence(self):
        """rg (when available) and the Python fallback return the same matches."""
        if shutil.which("rg") is None:
            self.skipTest("rg not on PATH")
        self.make("alpha.txt", "needle alpha\nplain line\n")
        self.make("beta.txt", "NEEDLE upper\n")
        self.make("sub/gamma.txt", "nothing here\nneedle gamma\n")
        rg_result = self.reg.execute("grep", {"pattern": "needle"})
        rg_case = self.reg.execute("grep", {"pattern": "NEEDLE", "case": True})
        with mock.patch("harness.tools.shutil.which", return_value=None):
            py_result = self.reg.execute("grep", {"pattern": "needle"})
            py_case = self.reg.execute("grep", {"pattern": "NEEDLE", "case": True})
        self.assertEqual(sorted(rg_result.splitlines()), sorted(py_result.splitlines()))
        self.assertEqual(sorted(rg_case.splitlines()), sorted(py_case.splitlines()))
        self.assertIn("alpha.txt:1: needle alpha", rg_result.splitlines())
        self.assertIn("sub/gamma.txt:2: needle gamma", rg_result.splitlines())
        self.assertIn("beta.txt:1: NEEDLE upper", rg_result.splitlines())  # case-insensitive default
        self.assertIn("beta.txt:1: NEEDLE upper", rg_case.splitlines())  # case-sensitive
        self.assertNotIn("needle alpha", rg_case)

    def test_grep_cap_sentinel_on_both_engines(self):
        """The cap sentinel guarantee holds for rg AND the Python fallback."""
        self.make("many.txt", "".join(f"needle {i}\n" for i in range(1, 3001)))
        self.make("sub/sentinel.txt", "needle sentinel\n")
        results = [self.reg.execute("grep", {"pattern": "needle"})]
        with mock.patch("harness.tools.shutil.which", return_value=None):
            results.append(self.reg.execute("grep", {"pattern": "needle"}))
        for result in results:
            self.assertLessEqual(len(result), MAX_RESULT_CHARS + len(TRUNCATED_SUFFIX))
            self.assertTrue(result.endswith(TRUNCATED_SUFFIX))
            self.assertTrue(result.startswith("many.txt:1: needle 1"))
            self.assertNotIn("sentinel", result)

    def test_grep_backreference_falls_back_to_python(self):
        # rg rejects backreferences (exit 2) -> the Python fallback runs, where
        # the pattern is valid and matches. Prove the fallback by comparing
        # with the forced-Python result.
        self.make("x.txt", "aa\n")
        rg_path = self.reg.execute("grep", {"pattern": r"(a)\1"})
        with mock.patch("harness.tools.shutil.which", return_value=None):
            py_path = self.reg.execute("grep", {"pattern": r"(a)\1"})
        self.assertEqual(rg_path, py_path)
        self.assertEqual(rg_path, "x.txt:1: aa")

    def test_grep_invalid_pattern_reports_error_via_fallback(self):
        # A pattern invalid for both engines surfaces as an error string from
        # the Python fallback — not a crash and not a raw rg error.
        self.make("x.txt", "aa\n")
        result = self.reg.execute("grep", {"pattern": "[z-a]"})
        self.assertIn("invalid regex", result)


class TestCacheCounters(unittest.TestCase):
    """Cache-hit/miss visibility on a registry with a real ToolCache."""

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.project = Path(self._tmp.name)
        self.make("a.txt", "hello\n")

    def make(self, rel: str, content: str = "") -> Path:
        path = self.project / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def _cached_registry(self) -> ToolRegistry:
        return ToolRegistry(
            project_dir=self.project,
            cache=ToolCache(self.project / ".kaal" / "tool-cache.json"),
        )

    def test_first_read_misses_second_hits(self):
        """Same-signature repeat read: first miss, second hit, rate 0.5."""
        reg = self._cached_registry()
        reg.begin_batch(["read"], "sig-1")
        reg.execute("read", {"path": "a.txt"})
        reg.execute("read", {"path": "a.txt"})
        self.assertEqual(reg.cache_misses, 1)
        self.assertEqual(reg.cache_hits, 1)
        self.assertEqual(reg.cache_hit_rate(), 0.5)

    def test_no_lookups_rate_is_none(self):
        reg = self._cached_registry()
        self.assertEqual(reg.cache_hits, 0)
        self.assertEqual(reg.cache_misses, 0)
        self.assertIsNone(reg.cache_hit_rate())

    def test_mutator_batch_bypass_counts_neither(self):
        """A batch containing a mutator skips lookups; counters stay at zero."""
        reg = self._cached_registry()
        reg.begin_batch(["write", "read"], "sig-1")
        reg.execute("read", {"path": "a.txt"})
        self.assertEqual(reg.cache_hits, 0)
        self.assertEqual(reg.cache_misses, 0)
        self.assertIsNone(reg.cache_hit_rate())

    def test_uncached_registry_never_looks_up(self):
        """cache=None means every execute bypasses; rate stays None."""
        reg = ToolRegistry(project_dir=self.project)
        reg.begin_batch(["read"], "sig-1")
        reg.execute("read", {"path": "a.txt"})
        self.assertEqual(reg.cache_hits, 0)
        self.assertEqual(reg.cache_misses, 0)
        self.assertIsNone(reg.cache_hit_rate())


if __name__ == "__main__":
    unittest.main()
