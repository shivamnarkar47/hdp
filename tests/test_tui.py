"""Textual TUI pilot tests (FakeGateway, no network).

Drives the real TUI through ``run_test``: a typed prompt is submitted, the
agent loop runs on its worker thread against a scripted fake gateway, and the
tool it calls (`write`) actually executes against a temp project dir. The
prompt is the multi-line TextArea; Enter submits it.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
import unittest
from pathlib import Path

from textual.widgets import TextArea

from harness import sessions
from harness.tui import HarnessTui

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
    ("content", "Let me check the directory. "),
    ("content", _write_envelope()),
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

    def tearDown(self) -> None:
        if self._old_sessions_dir is None:
            os.environ.pop("HARNESSDP_SESSIONS_DIR", None)
        else:
            os.environ["HARNESSDP_SESSIONS_DIR"] = self._old_sessions_dir
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

    def test_slash_sessions_lists_current_session(self):
        async def flow() -> None:
            app = self._app()
            async with app.run_test() as pilot:
                await self._submit_and_wait(app, "write hello.txt", pilot)
                sid = app.session_id
                app.query_one("#prompt", TextArea).text = "/sessions"
                await pilot.pause()
                await pilot.press("enter")
                for _ in range(200):  # up to ~10s
                    if sid in "".join(app.transcript):
                        break
                    await pilot.pause(0.05)
                self.assertIn(sid, "".join(app.transcript))
                self.assertIn("write hello.txt", "".join(app.transcript))

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
                self.assertEqual(len(app._suggestion_rows), 8)
                for cmd in (
                    "/help",
                    "/new",
                    "/resume",
                    "/sessions",
                    "/memory",
                    "/model",
                    "/verbose",
                    "/quit",
                ):
                    self.assertIn(cmd, app._suggestion_rows)
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
