"""Textual TUI pilot tests (FakeGateway, no network).

Drives the real TUI through ``run_test``: a typed prompt is submitted, the
agent loop runs on its worker thread against a scripted fake gateway, and the
tool it calls (`write`) actually executes against a temp project dir.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path

from textual.widgets import Input

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
        app.query_one("#prompt", Input).value = prompt
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
                output = "".join(app.output_lines)
                self.assertIn("Wrote hello.txt.", output)
                self.assertTrue(any(line.startswith("⚙") for line in app.output_lines))
                self.assertEqual((self.root / "hello.txt").read_text(encoding="utf-8"), "hi")

        asyncio.run(flow())

    def test_slash_sessions_lists_current_session(self):
        async def flow() -> None:
            app = self._app()
            async with app.run_test() as pilot:
                await self._submit_and_wait(app, "write hello.txt", pilot)
                sid = app.session_id
                app.query_one("#prompt", Input).value = "/sessions"
                await pilot.pause()
                await pilot.press("enter")
                for _ in range(200):  # up to ~10s
                    if sid in "".join(app.output_lines):
                        break
                    await pilot.pause(0.05)
                self.assertIn(sid, "".join(app.output_lines))
                self.assertIn("write hello.txt", "".join(app.output_lines))

        asyncio.run(flow())


if __name__ == "__main__":
    unittest.main()
