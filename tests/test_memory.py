"""Memory, prompt, and session round-trip tests."""

from __future__ import annotations

import os
import tempfile
import threading
import unittest
from datetime import date
from pathlib import Path

from harness.context import estimate_tokens
from harness.memory import Memory
from harness.prompts import FIXED_PREFIX, build_project_context, build_system_prompt
from harness.sessions import (
    append_event,
    get_store_dir,
    list_sessions,
    load_messages,
    new_session_id,
)

SECTIONS = ("project-state", "decisions", "patterns", "lessons-learned")


class TestMemory(unittest.TestCase):
    def setUp(self) -> None:
        self._root_tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._root_tmp.cleanup)
        self.root = Path(self._root_tmp.name)
        self._sessions_tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._sessions_tmp.cleanup)
        self._old_sessions_dir = os.environ.get("HARNESSDP_SESSIONS_DIR")
        os.environ["HARNESSDP_SESSIONS_DIR"] = self._sessions_tmp.name

    def tearDown(self) -> None:
        if self._old_sessions_dir is None:
            os.environ.pop("HARNESSDP_SESSIONS_DIR", None)
        else:
            os.environ["HARNESSDP_SESSIONS_DIR"] = self._old_sessions_dir

    def _memory(self) -> Memory:
        return Memory(self.root)

    def test_digest_caps_content_lines(self):
        memory = self._memory()
        memory.append("project-state", "\n".join(f"detail line {i}" for i in range(80)))
        digest = memory.load_digest()
        # The single seeded file: path present, at most 60 content lines.
        self.assertIn(str((self.root / "project-state.md").absolute()), digest)
        block = digest.split("### Project State", 1)[1].split("### Decisions", 1)[0]
        block_lines = block.splitlines()
        self.assertGreaterEqual(len(block_lines), 2)  # path line + content
        content_lines = [ln for ln in block_lines[1:] if ln.strip()]
        self.assertLessEqual(len(content_lines), 60)
        # The other three sections appear with their paths too.
        for section in SECTIONS[1:]:
            self.assertIn(str((self.root / f"{section}.md").absolute()), digest)

    def test_digest_token_cap_all_sections(self):
        memory = self._memory()
        for section in SECTIONS:
            memory.append(
                section,
                "\n".join(
                    f"{section} data line {i} " + "x" * 40 for i in range(70)
                ),
            )
        digest = memory.load_digest()
        self.assertLessEqual(estimate_tokens(digest), 4000)
        for section in SECTIONS:
            self.assertIn(str((self.root / f"{section}.md").absolute()), digest)

    def test_append_dedupes(self):
        memory = self._memory()
        path = memory.append("decisions", "Use JSONL for sessions.")
        self.assertEqual(path, str((self.root / "decisions.md").absolute()))
        self.assertEqual(memory.append("decisions", "Use JSONL for sessions."), "already recorded")
        content = (self.root / "decisions.md").read_text(encoding="utf-8")
        self.assertEqual(content.count("## "), 1)
        self.assertIn("Use JSONL for sessions.", content)

    def test_append_prunes_oldest_section(self):
        memory = self._memory()
        for i in range(5):
            text = f"run {i}\n" + "\n".join(f"body line {j}" for j in range(149))
            memory.append("patterns", text)
        content = (self.root / "patterns.md").read_text(encoding="utf-8")
        self.assertLessEqual(len(content.splitlines()), 200)
        # Oldest sections dropped, newest kept.
        self.assertEqual(content.count("## "), 1)
        self.assertNotIn("run 0", content)
        self.assertIn("run 4", content)

    def test_concurrent_append_is_lossless(self):
        """Distinct texts appended from many threads all survive (flock serializes)."""
        memory = self._memory()
        section = "lessons-learned"
        texts = [f"note-{i}" for i in range(8)]
        errors: list[BaseException] = []

        def worker(text: str) -> None:
            try:
                memory.append(section, text)
            except BaseException as exc:  # noqa: BLE001 - record and surface below
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(text,)) for text in texts]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        content = (self.root / f"{section}.md").read_text(encoding="utf-8")
        for text in texts:
            self.assertIn(text, content)

    def test_invalid_section_raises(self):
        memory = self._memory()
        with self.assertRaises(ValueError):
            memory.append("bogus", "x")
        with self.assertRaises(ValueError):
            memory.file_path("bogus")
        # Valid sections keep working after the failure.
        self.assertIsInstance(memory.file_path("decisions"), Path)

    def test_session_round_trip(self):
        sid = new_session_id()
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "memory_append", "arguments": "{}"},
            }
        ]
        append_event(sid, {"type": "user", "data": {"content": "hello"}})
        append_event(
            sid,
            {
                "type": "assistant",
                "data": {
                    "content": "hi",
                    "reasoning_content": "thinking step",
                    "tool_calls": tool_calls,
                },
            },
        )
        append_event(
            sid,
            {"type": "tool_result", "data": {"tool_call_id": "call_1", "content": "ok"}},
        )
        messages = load_messages(sid)
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[0], {"role": "user", "content": "hello"})
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(messages[1]["content"], "hi")
        self.assertEqual(messages[1]["reasoning_content"], "thinking step")
        self.assertEqual(messages[1]["tool_calls"], tool_calls)
        self.assertEqual(
            messages[2], {"role": "tool", "tool_call_id": "call_1", "content": "ok"}
        )
        sessions = list_sessions()
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["id"], sid)
        self.assertEqual(sessions[0]["prompt"], "hello")

    def test_load_messages_missing_and_corrupt(self):
        self.assertEqual(load_messages("does-not-exist"), [])
        sid = new_session_id()
        path = get_store_dir() / f"{sid}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            '{"ts": "t", "type": "user", "data": {"content": "ok"}}\n'
            "THIS IS NOT JSON\n",
            encoding="utf-8",
        )
        self.assertEqual(load_messages(sid), [{"role": "user", "content": "ok"}])

    def test_append_event_invalid_type(self):
        with self.assertRaises(ValueError):
            append_event(new_session_id(), {"type": "bogus", "data": {}})

    def test_store_dir_override(self):
        self.assertEqual(get_store_dir(), Path(os.environ["HARNESSDP_SESSIONS_DIR"]))
        self.assertEqual(list_sessions(), [])


class TestPrompts(unittest.TestCase):
    def test_build_project_context_with_agents(self):
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp) / "proj"
            cwd.mkdir()
            (cwd / "AGENTS.md").write_text(
                "# Agents\n\nBuild with stdlib only.\n", encoding="utf-8"
            )
            ctx = build_project_context(cwd)
            self.assertIn(date.today().isoformat(), ctx)
            self.assertIn(str(cwd.resolve()), ctx)
            self.assertIn("## AGENTS.md (first 200 lines)", ctx)
            self.assertIn("# Agents", ctx)
            self.assertIn("Build with stdlib only.", ctx)

    def test_build_project_context_without_agents(self):
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp) / "proj2"
            cwd.mkdir()
            ctx = build_project_context(cwd)
            self.assertIn(date.today().isoformat(), ctx)
            self.assertIn(str(cwd.resolve()), ctx)
            self.assertIn("No AGENTS.md", ctx)
            self.assertNotIn("## AGENTS.md (first 200 lines)", ctx)

    def test_project_context_includes_structure_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp) / "proj3"
            cwd.mkdir()
            (cwd / ".hdp").mkdir()
            (cwd / ".hdp" / "STRUCTURE.md").write_text(
                "# Project Structure\nRoot: x\n## Tree\n└── README.md (5 B)\n",
                encoding="utf-8",
            )
            ctx = build_project_context(cwd)
            self.assertIn("## Project structure", ctx)
            self.assertIn("└── README.md (5 B)", ctx)
            self.assertIn("re-read it if the files change", ctx)

    def test_project_context_missing_structure_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp) / "proj4"
            cwd.mkdir()
            ctx = build_project_context(cwd)
            self.assertIn("No structure cache yet", ctx)

    def test_build_system_prompt(self):
        digest = (
            "### Project State\n"
            "path: /tmp/x/project-state.md\n"
            "# Project State\n"
            "- session: work → done"
        )
        project = "Date: 2026-08-02\nCWD: /tmp/x"
        prompt = build_system_prompt(digest, project)
        self.assertIn("hdp — DeepSeek V4 Flash harness agent", prompt)
        self.assertIn(
            "When you need a fact or a file operation, call a tool. You may batch "
            "independent tool calls. The harness parses your DSML tool calls "
            "automatically.",
            prompt,
        )
        self.assertIn(
            "Final answers are plain text. Never emit tool markup, "
            "`reasoning_content`, or `<think>` blocks in your visible answer.",
            prompt,
        )
        self.assertIn(".agent-memory/", prompt)
        self.assertIn("## Memory Guidance", prompt)
        self.assertIn(digest, prompt)
        self.assertIn("## Project", prompt)
        self.assertIn(project, prompt)
        self.assertIn(FIXED_PREFIX, prompt)
        self.assertLessEqual(estimate_tokens(FIXED_PREFIX), 8000)


if __name__ == "__main__":
    unittest.main()
