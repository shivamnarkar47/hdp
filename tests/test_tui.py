"""Textual TUI pilot tests (FakeGateway, no network).

Drives the real TUI through ``run_test``: a typed prompt is submitted, the
agent loop runs on its worker thread against a scripted fake gateway, and the
tool it calls (`write`) actually executes against a temp project dir. The
prompt is the multi-line TextArea; Enter submits it.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from textual.widgets import Input, Label, ListView, Static, TextArea

from harness import config, sessions
from harness.art import SEA_LION
from harness.tui import ConnectScreen, HarnessTui, SessionsScreen

FW = "\uff5c"  # fullwidth vertical bar: DSML envelope delimiter


def _write_envelope() -> str:
    """A DSML tool_calls envelope for `write(path="hello.txt", content="hi")`."""
    return (
        f"<{FW}DSML{FW}tool_calls>"
        f'<{FW}DSML{FW}invoke name="write">'
        f'<{FW}DSML{FW}parameter name="path" string="true">hello.txt</{FW}DSML{FW}parameter>'
        f'<{FW}DSML{FW}parameter name="content" string="true">hi</{FW}DSML{FW}parameter>'
        f"</{FW}DSML{FW}invoke>"
        f"</{FW}DSML{FW}tool_calls>"
    )


TURN_TOOL = [
    ("reasoning", "Let me check"),
    # Real envelopes are generation-leading: the envelope must be the FIRST
    # content after the think span, or DialectFeed reads it as a prose quote
    # of the envelope (see dialect.py Part A guard).
    ("content", _write_envelope()),
    ("content", "Let me check the directory. "),
    ("done", "tool_calls"),
]
TURN_STOP = [("content", "Wrote hello.txt."), ("done", "stop")]


class FakeGateway:
    """Scripted gateway: each call to stream() yields the next script."""

    def __init__(self, *scripts):
        self.scripts = list(scripts)
        self.model_id = "fake-model"

    def stream(self, messages, tools):
        yield from self.scripts.pop(0)


THINK_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def _thinking_script():
    """Reasoning, then a pause, then the final answer (worker-thread sleep)."""
    yield ("reasoning", "thinking hard")
    time.sleep(0.4)
    yield ("content", "answer")
    yield ("done", "stop")


class TestTui(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._old_sessions_dir = os.environ.get("HARNESSDP_SESSIONS_DIR")
        os.environ["HARNESSDP_SESSIONS_DIR"] = str(self.root / "sessions")
        self._old_xdg = os.environ.get("XDG_CONFIG_HOME")
        os.environ["XDG_CONFIG_HOME"] = str(self.root / "config")

    def tearDown(self) -> None:
        if self._old_sessions_dir is None:
            os.environ.pop("HARNESSDP_SESSIONS_DIR", None)
        else:
            os.environ["HARNESSDP_SESSIONS_DIR"] = self._old_sessions_dir
        if self._old_xdg is None:
            os.environ.pop("XDG_CONFIG_HOME", None)
        else:
            os.environ["XDG_CONFIG_HOME"] = self._old_xdg
        self._tmp.cleanup()

    def _app(self) -> HarnessTui:
        return HarnessTui(
            gateway=FakeGateway(list(TURN_TOOL), list(TURN_STOP)),
            memory_root=self.root / ".agent-memory",
            project_dir=self.root,
        )

    @staticmethod
    async def _submit_and_wait(app: HarnessTui, prompt: str, pilot) -> None:
        prompt_widget = app.query_one("#prompt", TextArea)
        prompt_widget.text = prompt
        prompt_widget.focus()
        await pilot.pause()
        await pilot.press("enter")
        for _ in range(200):  # up to ~10s
            if not app.turn_active:
                break
            await pilot.pause(0.05)
        await pilot.pause()  # let any trailing main-thread renders land

    def test_agent_turn_executes_tool_end_to_end(self):
        async def flow() -> None:
            app = self._app()
            async with app.run_test() as pilot:
                await self._submit_and_wait(app, "write hello.txt", pilot)
                self.assertFalse(app.turn_active)
                transcript = "".join(app.transcript)
                self.assertIn("Wrote hello.txt.", transcript)
                self.assertTrue(any(line.startswith("⚙") for line in app.transcript))
                self.assertEqual((self.root / "hello.txt").read_text(encoding="utf-8"), "hi")

        asyncio.run(flow())

    def test_verify_hook_renders_dim_line(self):
        """A hooks file + a mutating turn renders the dim `🧪 verify` pane line
        (fast hook: python -c print)."""
        async def flow() -> None:
            hooks = self.root / ".hdp" / "hooks.json"
            hooks.parent.mkdir(parents=True, exist_ok=True)
            hooks.write_text(
                json.dumps({"verify": ["python", "-c", "print('verify-ok')"]}),
                encoding="utf-8",
            )
            app = self._app()
            async with app.run_test() as pilot:
                await self._submit_and_wait(app, "write hello.txt", pilot)
                self.assertFalse(app.turn_active)
                transcript = "".join(app.transcript)
                self.assertIn("🧪 verify: verify-ok", transcript)

        asyncio.run(flow())

    def test_verify_off_without_hooks_file(self):
        """No hooks file: the verify hook never runs, no pane line."""
        async def flow() -> None:
            app = self._app()
            async with app.run_test() as pilot:
                await self._submit_and_wait(app, "write hello.txt", pilot)
                self.assertFalse(app.turn_active)
                transcript = "".join(app.transcript)
                self.assertNotIn("🧪 verify", transcript)

        asyncio.run(flow())

    def test_sessions_popup_resumes(self):
        async def flow() -> None:
            app = self._app()
            async with app.run_test() as pilot:
                sessions.append_event(
                    "20260802-120000", {"type": "user", "data": {"content": "first"}}
                )
                sessions.append_event(
                    "20260802-130000", {"type": "user", "data": {"content": "second"}}
                )
                prompt = app.query_one("#prompt", TextArea)
                prompt.text = "/sessions"
                prompt.focus()
                await pilot.pause()
                await pilot.press("enter")
                await pilot.pause()
                # The modal switcher is up with both sessions, newest first.
                self.assertIsInstance(app.screen, SessionsScreen)
                list_view = app.screen.query_one("#session-list", ListView)
                self.assertEqual(len(list_view.children), 2)
                first_row = str(list_view.children[0].query_one(Label).render())
                self.assertIn("20260802-130000", first_row)
                self.assertIn("second", first_row)
                # Enter resumes the highlighted (newest) session and the
                # history is rendered into the pane.
                await pilot.press("enter")
                await pilot.pause()
                self.assertNotIsInstance(app.screen, SessionsScreen)
                self.assertEqual(app.session_id, "20260802-130000")
                self.assertTrue(app.resume_next)
                transcript = "".join(app.transcript)
                self.assertIn("── resumed session 20260802-130000 ──", transcript)
                self.assertIn("second", transcript)  # the session's user prompt

        asyncio.run(flow())

    def test_resume_shows_history(self):
        async def flow() -> None:
            app = self._app()
            async with app.run_test() as pilot:
                sid = "20260802-140000"
                sessions.append_event(
                    sid, {"type": "user", "data": {"content": "first question"}}
                )
                sessions.append_event(
                    sid,
                    {
                        "type": "assistant",
                        "data": {
                            "content": "first answer",
                            "reasoning_content": "thoughts",
                            "tool_calls": [
                                {"id": "call_1", "name": "read", "arguments": '{"path":"x"}'}
                            ],
                        },
                    },
                )
                sessions.append_event(
                    sid,
                    {
                        "type": "tool_result",
                        "data": {"tool_call_id": "call_1", "content": "file contents"},
                    },
                )
                prompt = app.query_one("#prompt", TextArea)
                prompt.text = f"/resume {sid}"
                prompt.focus()
                await pilot.pause()
                await pilot.press("enter")
                await pilot.pause()
                self.assertEqual(app.session_id, sid)
                self.assertTrue(app.resume_next)
                transcript = "".join(app.transcript)
                self.assertIn("── resumed session 20260802-140000 ──", transcript)
                self.assertIn("first question", transcript)  # user block
                self.assertIn("first answer", transcript)  # assistant markdown
                self.assertIn("⚙ read(", transcript)  # tool call line
                self.assertIn("  → file contents", transcript)  # tool-result preview
                self.assertNotIn("thoughts", transcript)  # verbose off

        asyncio.run(flow())

    def test_connect_popup(self):
        async def flow() -> None:
            app = self._app()
            async with app.run_test() as pilot:
                prompt = app.query_one("#prompt", TextArea)
                prompt.text = "/connect"
                prompt.focus()
                await pilot.pause()
                await pilot.press("enter")
                await pilot.pause()
                self.assertIsInstance(app.screen, ConnectScreen)
                key_input = app.screen.query_one("#key-input", Input)
                key_input.value = "sk-test123"
                key_input.focus()
                await pilot.pause()
                await pilot.press("enter")  # Input.Submitted -> Save
                await pilot.pause()
                self.assertNotIsInstance(app.screen, ConnectScreen)
                self.assertEqual(app.gateway.api_key, "sk-test123")
                self.assertEqual(config.user_key_path().read_text(), "sk-test123")
                self.assertIn("connected: API key saved", "".join(app.transcript))

        asyncio.run(flow())

    def test_slash_suggestions(self):
        async def flow() -> None:
            app = self._app()
            async with app.run_test() as pilot:
                prompt = app.query_one("#prompt", TextArea)
                # A bare "/" lists every command.
                prompt.text = "/"
                await pilot.pause()
                self.assertTrue(app._suggestions_visible)
                # 9 commands now, but the popup caps at 8 rows + a "…" row.
                self.assertEqual(len(app._suggestion_rows), 8)
                self.assertTrue(app._suggest_more)
                for cmd in (
                    "/help",
                    "/new",
                    "/resume",
                    "/sessions",
                    "/memory",
                    "/model",
                    "/verbose",
                    "/connect",
                ):
                    self.assertIn(cmd, app._suggestion_rows)
                # The capped command still resolves by prefix.
                prompt.text = "/q"
                await pilot.pause()
                self.assertEqual(app._suggestion_rows, ["/quit"])
                # Prefix filter narrows the list.
                prompt.text = "/res"
                await pilot.pause()
                self.assertEqual(app._suggestion_rows, ["/resume"])
                # Tab completes and closes the popup.
                await pilot.press("tab")
                await pilot.pause()
                self.assertEqual(app.query_one("#prompt", TextArea).text, "/resume")
                self.assertFalse(app._suggestions_visible)
                # Escape keeps the popup closed.
                await pilot.press("escape")
                await pilot.pause()
                self.assertFalse(app._suggestions_visible)
                # "/resume <arg>" suggests session ids, newest first.
                sessions.append_event("20260802-120000", {"type": "user", "data": {"content": "x"}})
                sessions.append_event("20260802-130000", {"type": "user", "data": {"content": "y"}})
                prompt.text = "/resume 20260802-1"
                await pilot.pause()
                self.assertEqual(
                    app._suggestion_rows, ["20260802-130000", "20260802-120000"]
                )

        asyncio.run(flow())

    def test_sea_lion_art_shape(self):
        lines = SEA_LION.splitlines()
        self.assertTrue(lines)
        self.assertLessEqual(len(lines), 26)
        self.assertGreaterEqual(len(lines), 20)
        for line in lines:
            self.assertLessEqual(len(line), 80)
            self.assertNotIn("\t", line)
        widths = [len(line) for line in lines]
        self.assertLessEqual(max(widths) - min(widths), 2)  # padded rectangle

    def test_home_art(self):
        async def flow() -> None:
            app = self._app()
            async with app.run_test() as pilot:
                await pilot.pause()
                transcript = "\n".join(app.transcript)
                self.assertIn(SEA_LION.splitlines()[0], transcript)
                self.assertIn("Ask a task, or /help", transcript)
                # /new re-renders the home hero with a fresh session.
                app.resume_next = True
                prompt = app.query_one("#prompt", TextArea)
                prompt.text = "/new"
                prompt.focus()
                with mock.patch(
                    "harness.sessions.new_session_id", return_value="20260802-999999"
                ):
                    await pilot.pause()
                    await pilot.press("enter")
                    await pilot.pause()
                self.assertEqual(app.session_id, "20260802-999999")
                self.assertFalse(app.resume_next)
                transcript = "\n".join(app.transcript)
                self.assertIn(SEA_LION.splitlines()[0], transcript)
                self.assertIn("Ask a task, or /help", transcript)

        asyncio.run(flow())

    def test_structure_command(self):
        async def flow() -> None:
            app = self._app()
            async with app.run_test() as pilot:
                await pilot.pause()
                # Mount created the cache and wrote the one-line notice.
                transcript = "\n".join(app.transcript)
                self.assertIn("structure:", transcript)
                self.assertIn("files", transcript)
                # /structure refreshes and dumps the doc + cache path.
                prompt = app.query_one("#prompt", TextArea)
                prompt.text = "/structure"
                prompt.focus()
                await pilot.pause()
                await pilot.press("enter")
                await pilot.pause()
                transcript = "\n".join(app.transcript)
                self.assertIn(".hdp/STRUCTURE.md", transcript)
                self.assertIn("# Project Structure", transcript)

        asyncio.run(flow())

    def test_large_answer_uses_raw_fallback(self):
        async def flow() -> None:
            big = "x" * 12_000
            app = HarnessTui(
                gateway=FakeGateway([("content", big), ("done", "stop")]),
                memory_root=self.root / ".agent-memory",
                project_dir=self.root,
            )
            async with app.run_test() as pilot:
                prompt = app.query_one("#prompt", TextArea)
                prompt.text = "big answer"
                prompt.focus()
                await pilot.pause()
                await pilot.press("enter")
                for _ in range(200):  # up to ~10s
                    if not app.turn_active:
                        break
                    await pilot.pause(0.05)
                await pilot.pause()
                # Past MD_CHAR_CAP the overflow appends via the raw block;
                # the full answer (incl. the tail) is mirrored to the transcript.
                self.assertIn(big[-20:], "".join(app.transcript))

        asyncio.run(flow())

    def test_tokens_per_sec_helper(self):
        self.assertEqual(HarnessTui._tokens_per_sec(300, 10), 10.0)
        self.assertEqual(HarnessTui._tokens_per_sec(0, 0), 0.0)

    def test_status_shows_tokens_per_sec(self):
        async def flow() -> None:
            app = HarnessTui(
                gateway=FakeGateway([("content", "hello world"), ("done", "stop")]),
                memory_root=self.root / ".agent-memory",
                project_dir=self.root,
            )
            async with app.run_test() as pilot:
                prompt = app.query_one("#prompt", TextArea)
                prompt.text = "say hi"
                prompt.focus()
                await pilot.pause()
                await pilot.press("enter")
                for _ in range(200):  # up to ~10s
                    if not app.turn_active:
                        break
                    await pilot.pause(0.05)
                await pilot.pause()
                # The turn's average throughput is frozen on the bar.
                status = str(app.query_one("#status", Static).render())
                self.assertIn("tok/s", status)

        asyncio.run(flow())

    def test_thinking_indicator(self):
        async def flow() -> None:
            app = HarnessTui(
                gateway=FakeGateway(_thinking_script()),
                memory_root=self.root / ".agent-memory",
                project_dir=self.root,
            )
            async with app.run_test() as pilot:
                prompt = app.query_one("#prompt", TextArea)
                prompt.text = "think"
                prompt.focus()
                await pilot.pause()
                await pilot.press("enter")
                # The worker is mid-reasoning (sleeping); the spinner must be up.
                await pilot.pause(0.15)
                self.assertTrue(app._thinking_visible)
                for _ in range(200):  # up to ~10s
                    if not app.turn_active:
                        break
                    await pilot.pause(0.05)
                self.assertFalse(app._thinking_visible)
                transcript = "".join(app.transcript)
                self.assertIn("answer", transcript)
                # Transient spinner never leaks into the transcript mirror.
                for frame in THINK_FRAMES:
                    self.assertNotIn(frame, transcript)
                self.assertNotIn("thinking hard", transcript)  # verbose off

        asyncio.run(flow())


if __name__ == "__main__":
    unittest.main()
