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

from textual.containers import Vertical
from textual.widgets import Button, Input, Label, ListView, Static, TextArea

from harness import agents, config, sessions
from harness.art import (
    BANNER_TAGLINE,
    BANNER_TITLE,
    KAAL_ART,
    SEA_LION,
)
from harness.messages import ToolCall
from harness.sessions import read_events
from harness.tui import (
    AGENT_GENERATOR_SYSTEM_PROMPT,
    AgentFormScreen,
    AgentIntentScreen,
    AgentsScreen,
    AskScreen,
    AskTextArea,
    ConnectScreen,
    HarnessTui,
    SessionsScreen,
)

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

    def stream(self, messages, tools=None, max_tokens=None):
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
        self._old_sessions_dir = os.environ.get("KAAL_SESSIONS_DIR")
        os.environ["KAAL_SESSIONS_DIR"] = str(self.root / "sessions")
        self._old_xdg = os.environ.get("XDG_CONFIG_HOME")
        os.environ["XDG_CONFIG_HOME"] = str(self.root / "config")

    def tearDown(self) -> None:
        if self._old_sessions_dir is None:
            os.environ.pop("KAAL_SESSIONS_DIR", None)
        else:
            os.environ["KAAL_SESSIONS_DIR"] = self._old_sessions_dir
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

    def test_prompt_input_renders_single_line(self):
        """The prompt is a compact one-line input: resolved height is 1 cell,
        and Shift+Enter newlines stay in the document (internal scroll) instead
        of clipping the typed text."""
        async def flow() -> None:
            app = self._app()
            async with app.run_test() as pilot:
                prompt = app.query_one("#prompt", TextArea)
                self.assertEqual(prompt.outer_size.height, 1)
                self.assertEqual(prompt.styles.height.value, 1)
                prompt.focus()
                await pilot.press("a", "shift+enter", "b")
                await pilot.pause()
                # Still one cell tall, but the full multi-line text is intact.
                self.assertEqual(prompt.outer_size.height, 1)
                self.assertEqual(prompt.text, "a\nb")

        asyncio.run(flow())

    def test_verify_hook_renders_dim_line(self):
        """A hooks file + a mutating turn renders the dim `🧪 verify` pane line
        (fast hook: python -c print)."""
        async def flow() -> None:
            hooks = self.root / ".kaal" / "hooks.json"
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

    def test_ask_user_modal_free_text(self):
        """ask_user mid-turn pops the AskScreen; the typed answer becomes the
        tool result and the turn completes with the modal dismissed."""
        async def flow() -> None:
            ask_turn = [
                ("tool_call", ToolCall("a1", "ask_user", '{"question": "Proceed?"}')),
                ("done", "tool_calls"),
            ]
            final = [("content", "Proceeding with your answer."), ("done", "stop")]
            app = HarnessTui(
                gateway=FakeGateway(list(ask_turn), list(final)),
                memory_root=self.root / ".agent-memory",
                project_dir=self.root,
            )
            async with app.run_test() as pilot:
                prompt = app.query_one("#prompt", TextArea)
                prompt.text = "ask me"
                prompt.focus()
                await pilot.pause()
                await pilot.press("enter")
                # The modal appears once the worker thread executes ask_user.
                for _ in range(200):
                    if isinstance(app.screen, AskScreen):
                        break
                    await pilot.pause(0.05)
                self.assertIsInstance(app.screen, AskScreen)
                answer_input = app.screen.query_one("#ask-text", AskTextArea)
                answer_input.text = "go ahead"
                answer_input.focus()
                await pilot.pause()
                await pilot.press("enter")  # AskTextArea.Submitted -> dismiss
                for _ in range(200):  # up to ~10s for the turn to finish
                    if not app.turn_active:
                        break
                    await pilot.pause(0.05)
                await pilot.pause()
                self.assertFalse(app.turn_active)
                self.assertNotIsInstance(app.screen, AskScreen)  # modal dismissed
                transcript = "".join(app.transcript)
                self.assertIn("Proceeding with your answer.", transcript)
                # The typed answer IS the persisted tool result (the model
                # used it to keep going).
                persisted = [
                    r for r in read_events(app.session_id) if r["type"] == "tool_result"
                ]
                self.assertEqual(persisted[-1]["data"]["content"], "go ahead")

        asyncio.run(flow())

    def test_ask_user_modal_option_buttons(self):
        """ask_user with options: the modal shows one Button per option and
        Enter on the focused (first) option picks it."""
        async def flow() -> None:
            ask_turn = [
                (
                    "tool_call",
                    ToolCall(
                        "a1",
                        "ask_user",
                        '{"question": "Which?", "options": ["alpha", "beta"]}',
                    ),
                ),
                ("done", "tool_calls"),
            ]
            final = [("content", "picked"), ("done", "stop")]
            app = HarnessTui(
                gateway=FakeGateway(list(ask_turn), list(final)),
                memory_root=self.root / ".agent-memory",
                project_dir=self.root,
            )
            async with app.run_test() as pilot:
                prompt = app.query_one("#prompt", TextArea)
                prompt.text = "pick"
                prompt.focus()
                await pilot.pause()
                await pilot.press("enter")
                for _ in range(200):
                    if isinstance(app.screen, AskScreen):
                        break
                    await pilot.pause(0.05)
                self.assertIsInstance(app.screen, AskScreen)
                buttons = app.screen.query(Button)
                self.assertEqual(len(buttons), 2)
                await pilot.press("enter")  # first option focused -> picks alpha
                for _ in range(200):
                    if not app.turn_active:
                        break
                    await pilot.pause(0.05)
                await pilot.pause()
                self.assertFalse(app.turn_active)
                self.assertNotIsInstance(app.screen, AskScreen)
                persisted = [
                    r for r in read_events(app.session_id) if r["type"] == "tool_result"
                ]
                self.assertEqual(persisted[-1]["data"]["content"], "alpha")

        asyncio.run(flow())

    def test_agents_popup_lists_and_activates(self):
        """/agents opens the switcher listing the five Pandavas; Enter on the
        highlighted row (Yudhishthira) activates it, persists, and the status
        bar leads with the active agent's name."""
        async def flow() -> None:
            app = self._app()
            async with app.run_test() as pilot:
                # Startup auto-activation: the first Pandava is active already.
                self.assertEqual(app._active_agent["name"], "Yudhishthira")
                prompt = app.query_one("#prompt", TextArea)
                prompt.text = "/agents"
                prompt.focus()
                await pilot.pause()
                await pilot.press("enter")
                await pilot.pause()
                self.assertIsInstance(app.screen, AgentsScreen)
                list_view = app.screen.query_one("#agent-list", ListView)
                self.assertGreaterEqual(len(list_view.children), 5)
                first_row = str(list_view.children[0].query_one(Label).render())
                self.assertIn("Yudhishthira", first_row)
                # Enter activates the highlighted (first: Yudhishthira).
                await pilot.press("enter")
                await pilot.pause()
                self.assertNotIsInstance(app.screen, AgentsScreen)
                self.assertEqual(app._active_agent["name"], "Yudhishthira")
                self.assertIn(
                    "agent: Yudhishthira active", "\n".join(app.transcript)
                )
                # Persisted: a fresh load sees Yudhishthira active.
                self.assertEqual(agents.load(self.root)["active"], "Yudhishthira")
                # The status bar starts with the active agent's name.
                status = str(app.query_one("#status", Static).render())
                self.assertTrue(status.lstrip().startswith("Yudhishthira"))
                # Selecting Arjuna updates both the persona and the bar.
                prompt.text = "/agents"
                prompt.focus()
                await pilot.pause()
                await pilot.press("enter")
                await pilot.pause()
                await pilot.press("down")
                await pilot.pause()
                await pilot.press("down")  # row 2: Bhima(1) -> Arjuna(2)
                await pilot.pause()
                await pilot.press("enter")
                await pilot.pause()
                self.assertEqual(app._active_agent["name"], "Arjuna")
                status = str(app.query_one("#status", Static).render())
                self.assertTrue(status.lstrip().startswith("Arjuna"))

        asyncio.run(flow())

    def test_agents_form_generates_agent(self):
        """/agents -> n: the form has a SINGLE description input (no manual
        name field); saving runs the AI generator and the name comes back from
        the scripted gateway JSON, added + activated."""
        async def flow() -> None:
            script = [
                (
                    "content",
                    '{"name": "Karna", "description": "the relentless executor"}',
                ),
                ("done", "stop"),
            ]
            app = HarnessTui(
                gateway=FakeGateway(script),
                memory_root=self.root / ".agent-memory",
                project_dir=self.root,
            )
            async with app.run_test() as pilot:
                prompt = app.query_one("#prompt", TextArea)
                prompt.text = "/agents"
                prompt.focus()
                await pilot.pause()
                await pilot.press("enter")
                await pilot.pause()
                await pilot.press("n")  # new
                await pilot.pause()
                self.assertIsInstance(app.screen, AgentFormScreen)
                # One input only — the description; no name input exists.
                self.assertEqual(len(app.screen.query(Input)), 1)
                self.assertEqual(
                    app.screen.query_one("#agent-desc-input", Input).id, "agent-desc-input"
                )
                desc_input = app.screen.query_one("#agent-desc-input", Input)
                desc_input.value = "a relentless executor who never stops"
                desc_input.focus()
                await pilot.pause()
                await pilot.press("enter")  # Enter saves -> AI generation
                # Poll until the generator worker marshals the result back.
                for _ in range(200):  # up to ~10s
                    if any("created and active" in line for line in app.transcript):
                        break
                    await pilot.pause(0.05)
                await pilot.pause()
                self.assertIn(
                    "agent: Karna created and active", "\n".join(app.transcript)
                )
                self.assertEqual(app._active_agent["name"], "Karna")
                self.assertIn("Karna", [a["name"] for a in app._agents["agents"]])
                self.assertFalse(app.turn_active)

        asyncio.run(flow())

    def test_ctrl_g_generates_agent(self):
        """Ctrl+G opens the intent screen; a scripted gateway completion
        streams a JSON agent dict; the new agent lands in state and activates."""
        async def flow() -> None:
            script = [
                (
                    "content",
                    '{"name": "Karna", "description": "the relentless executor"}',
                ),
                ("done", "stop"),
            ]
            app = HarnessTui(
                gateway=FakeGateway(script),
                memory_root=self.root / ".agent-memory",
                project_dir=self.root,
            )
            async with app.run_test() as pilot:
                await pilot.pause()
                await pilot.press("ctrl+g")
                await pilot.pause()
                self.assertIsInstance(app.screen, AgentIntentScreen)
                intent_input = app.screen.query_one("#agent-intent-input", Input)
                intent_input.value = "a relentless executor who never stops"
                intent_input.focus()
                await pilot.pause()
                await pilot.press("enter")
                # Poll until the generator worker marshals the result back.
                for _ in range(200):  # up to ~10s
                    if any("generated and active" in line for line in app.transcript):
                        break
                    await pilot.pause(0.05)
                await pilot.pause()
                self.assertIn(
                    "agent: Karna generated and active", "\n".join(app.transcript)
                )
                self.assertEqual(app._active_agent["name"], "Karna")
                self.assertIn("Karna", [a["name"] for a in app._agents["agents"]])
                self.assertFalse(app.turn_active)

        asyncio.run(flow())

    def test_ctrl_g_refuses_while_turn_active(self):
        """Ctrl+G during a live turn is refused with a notice, no popup."""
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
                await pilot.pause(0.15)  # worker is mid-reasoning
                self.assertTrue(app.turn_active)
                await pilot.press("ctrl+g")
                await pilot.pause()
                self.assertNotIsInstance(app.screen, AgentIntentScreen)
                self.assertIn(
                    "(busy — wait for the current turn)", "\n".join(app.transcript)
                )
                for _ in range(200):  # let the turn finish cleanly
                    if not app.turn_active:
                        break
                    await pilot.pause(0.05)

        asyncio.run(flow())

    def test_form_flow_adds_exactly_one_agent(self):
        """Flow A (form): /agents -> n -> description -> Enter must add EXACTLY
        ONE agent — Karna appears exactly once in state."""
        async def flow() -> None:
            script = [
                (
                    "content",
                    '{"name": "Karna", "description": "the relentless executor"}',
                ),
                ("done", "stop"),
            ]
            app = HarnessTui(
                gateway=FakeGateway(script),
                memory_root=self.root / ".agent-memory",
                project_dir=self.root,
            )
            async with app.run_test() as pilot:
                before = len(app._agents["agents"])
                prompt = app.query_one("#prompt", TextArea)
                prompt.text = "/agents"
                prompt.focus()
                await pilot.pause()
                await pilot.press("enter")
                await pilot.pause()
                await pilot.press("n")  # new -> the form appears
                await pilot.pause()
                self.assertIsInstance(app.screen, AgentFormScreen)
                desc_input = app.screen.query_one("#agent-desc-input", Input)
                desc_input.value = "a relentless executor who never stops"
                desc_input.focus()
                await pilot.pause()
                await pilot.press("enter")  # save -> AI generation
                for _ in range(200):  # up to ~10s
                    if any("created and active" in line for line in app.transcript):
                        break
                    await pilot.pause(0.05)
                await pilot.pause()
                names = [a["name"] for a in app._agents["agents"]]
                self.assertEqual(len(app._agents["agents"]), before + 1)
                self.assertEqual(names.count("Karna"), 1)

        asyncio.run(flow())

    def test_ctrl_g_flow_adds_exactly_one_agent(self):
        """Flow B (Ctrl+G): intent -> Enter must add EXACTLY ONE agent."""
        async def flow() -> None:
            script = [
                (
                    "content",
                    '{"name": "Karna", "description": "the relentless executor"}',
                ),
                ("done", "stop"),
            ]
            app = HarnessTui(
                gateway=FakeGateway(script),
                memory_root=self.root / ".agent-memory",
                project_dir=self.root,
            )
            async with app.run_test() as pilot:
                before = len(app._agents["agents"])
                await pilot.pause()
                await pilot.press("ctrl+g")
                await pilot.pause()
                self.assertIsInstance(app.screen, AgentIntentScreen)
                intent_input = app.screen.query_one("#agent-intent-input", Input)
                intent_input.value = "a relentless executor who never stops"
                intent_input.focus()
                await pilot.pause()
                await pilot.press("enter")
                for _ in range(200):  # up to ~10s
                    if any("generated and active" in line for line in app.transcript):
                        break
                    await pilot.pause(0.05)
                await pilot.pause()
                names = [a["name"] for a in app._agents["agents"]]
                self.assertEqual(len(app._agents["agents"]), before + 1)
                self.assertEqual(names.count("Karna"), 1)

        asyncio.run(flow())

    def test_generator_reentry_guarded(self):
        """Two generator runs started in quick succession (before either
        worker completes) run EXACTLY ONE generator: the second start is
        refused with a notice, and the list gains one agent."""
        class CountingGateway:
            """Servable by concurrent stream() calls; counts invocations."""
            model_id = "fake-model"

            def __init__(self) -> None:
                self.calls = 0

            def stream(self, messages, tools=None, max_tokens=None):
                self.calls += 1
                yield (
                    "content",
                    '{"name": "Karna", "description": "the relentless executor"}',
                )
                time.sleep(0.4)  # both workers in flight before done
                yield ("done", "stop")

        async def flow() -> None:
            gateway = CountingGateway()
            app = HarnessTui(
                gateway=gateway,
                memory_root=self.root / ".agent-memory",
                project_dir=self.root,
            )
            async with app.run_test() as pilot:
                await pilot.pause()
                before = len(app._agents["agents"])
                # Two rapid starts before either worker completes.
                app._generate_agent(
                    "an executor",
                    AGENT_GENERATOR_SYSTEM_PROMPT,
                    "generated and active",
                )
                app._generate_agent(
                    "an executor",
                    AGENT_GENERATOR_SYSTEM_PROMPT,
                    "generated and active",
                )
                for _ in range(200):  # up to ~10s
                    if not app.turn_active:
                        break
                    await pilot.pause(0.05)
                await pilot.pause(0.6)  # let any second completion land
                # Exactly one generator stream ran; the second was refused.
                self.assertEqual(gateway.calls, 1)
                self.assertIn(
                    "agent generator: already running", "\n".join(app.transcript)
                )
                names = [a["name"] for a in app._agents["agents"]]
                self.assertEqual(names.count("Karna"), 1)
                self.assertEqual(len(app._agents["agents"]), before + 1)

        asyncio.run(flow())

    def test_generator_empty_reply_notice(self):
        """An empty generator reply (no content chunks) is handled: the
        existing 'could not parse' notice, no crash, no agent added."""
        async def flow() -> None:
            app = HarnessTui(
                gateway=FakeGateway([("done", "stop")]),
                memory_root=self.root / ".agent-memory",
                project_dir=self.root,
            )
            async with app.run_test() as pilot:
                await pilot.pause()
                before = len(app._agents["agents"])
                app._generate_agent(
                    "an executor",
                    AGENT_GENERATOR_SYSTEM_PROMPT,
                    "generated and active",
                )
                for _ in range(200):  # up to ~10s
                    if not app.turn_active:
                        break
                    await pilot.pause(0.05)
                await pilot.pause()
                self.assertIn(
                    "agent generator: could not parse a name/description",
                    "\n".join(app.transcript),
                )
                self.assertEqual(len(app._agents["agents"]), before)
                self.assertFalse(app.turn_active)

        asyncio.run(flow())

    def test_duplicate_name_replaced(self):
        """A generator returning the SAME name as an existing agent replaces
        that entry in place (list length unchanged), updates the description,
        and activates it — no duplicate accumulates."""
        async def flow() -> None:
            script = [
                (
                    "content",
                    '{"name": "Yudhishthira", "description": "the brand-new dharma"}',
                ),
                ("done", "stop"),
            ]
            app = HarnessTui(
                gateway=FakeGateway(script),
                memory_root=self.root / ".agent-memory",
                project_dir=self.root,
            )
            async with app.run_test() as pilot:
                before = len(app._agents["agents"])
                prompt = app.query_one("#prompt", TextArea)
                prompt.text = "/agents"
                prompt.focus()
                await pilot.pause()
                await pilot.press("enter")
                await pilot.pause()
                await pilot.press("n")  # new -> the form appears
                await pilot.pause()
                desc_input = app.screen.query_one("#agent-desc-input", Input)
                desc_input.value = "a new take on dharma"
                desc_input.focus()
                await pilot.pause()
                await pilot.press("enter")  # save -> AI generation
                for _ in range(200):  # up to ~10s
                    if any("replaced existing" in line for line in app.transcript):
                        break
                    await pilot.pause(0.05)
                await pilot.pause()
                # Length unchanged; the existing entry was replaced in place.
                self.assertEqual(len(app._agents["agents"]), before)
                yudi = [
                    a
                    for a in app._agents["agents"]
                    if a["name"] == "Yudhishthira"
                ]
                self.assertEqual(len(yudi), 1)  # exactly one Yudhishthira
                self.assertEqual(yudi[0]["description"], "the brand-new dharma")
                self.assertEqual(app._active_agent["name"], "Yudhishthira")
                self.assertIn(
                    "agent: Yudhishthira created (replaced existing)",
                    "\n".join(app.transcript),
                )
                self.assertFalse(app.turn_active)

        asyncio.run(flow())

    def test_agents_popup_design(self):
        """The /agents popup: count title, the active agent's row starts with
        a ✓ marker, and rows are two-line (full description present — a
        substring beyond 40 chars renders)."""
        async def flow() -> None:
            app = self._app()
            async with app.run_test() as pilot:
                # Startup auto-activation: Yudhishthira is active (first row).
                self.assertEqual(app._active_agent["name"], "Yudhishthira")
                prompt = app.query_one("#prompt", TextArea)
                prompt.text = "/agents"
                prompt.focus()
                await pilot.pause()
                await pilot.press("enter")
                await pilot.pause()
                self.assertIsInstance(app.screen, AgentsScreen)
                # Count title.
                title = str(app.screen.query_one(".connect-title", Static).render())
                self.assertRegex(title, r"^Agents \(\d+\)$")
                # The agents box uses its dedicated (wider) styling.
                self.assertIsNotNone(app.screen.query_one("#agents-box"))
                list_view = app.screen.query_one("#agent-list", ListView)
                first_row = list_view.children[0]
                # Two-line row: a bold name label and a full-description label.
                labels = list(first_row.query(Label))
                self.assertGreaterEqual(len(labels), 2)
                # The active (first: Yudhishthira) row's name starts with ✓.
                self.assertTrue(str(labels[0].render()).startswith("✓ Yudhishthira"))
                # The description line carries the FULL description — a
                # substring beyond the old 40-char truncation appears.
                self.assertIn("never cuts corners", str(labels[1].render()))
                # The hint line documents every binding.
                hint = str(app.screen.query_one(".connect-hint", Static).render())
                self.assertIn("↑/↓ select", hint)
                self.assertIn("n new", hint)
                self.assertIn("d delete", hint)

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
                # 11 commands now, but the popup caps at 8 rows + a "…" row.
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
                    "/sidebar",
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

    def test_sidebar_toggle(self):
        """Ctrl+S and /sidebar hide/show the right sidebar; the conversation
        pane keeps running a full turn while the sidebar is hidden."""

        async def flow() -> None:
            app = self._app()
            async with app.run_test() as pilot:
                await pilot.pause()
                # Sidebar starts visible.
                self.assertIs(app._sidebar_visible, True)
                self.assertTrue(app.query_one("#sidebar", Vertical).display)

                # App-level Ctrl+S hides it (TextArea never binds Ctrl+S).
                await pilot.press("ctrl+s")
                await pilot.pause()
                self.assertIs(app._sidebar_visible, False)
                self.assertFalse(app.query_one("#sidebar", Vertical).display)
                self.assertIn("sidebar hidden", "\n".join(app.transcript))

                # The conversation pane still runs a full turn while hidden.
                await self._submit_and_wait(app, "write hello.txt", pilot)
                self.assertFalse(app.turn_active)
                self.assertIn("Wrote hello.txt.", "\n".join(app.transcript))

                # The /sidebar command toggles it back on.
                prompt = app.query_one("#prompt", TextArea)
                prompt.text = "/sidebar"
                prompt.focus()
                await pilot.pause()
                await pilot.press("enter")
                await pilot.pause()
                self.assertIs(app._sidebar_visible, True)
                self.assertTrue(app.query_one("#sidebar", Vertical).display)
                self.assertIn("sidebar shown", "\n".join(app.transcript))

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

    def test_kaal_art_shape(self):
        lines = KAAL_ART.splitlines()
        self.assertTrue(lines)
        self.assertEqual(len(lines), 5)  # the block wordmark is exactly 5 lines
        for line in lines:
            self.assertLessEqual(len(line), 80)
            self.assertNotIn("\t", line)
        widths = [len(line) for line in lines]
        self.assertEqual(len(set(widths)), 1)  # padded rectangle

    def test_home_banner(self):
        async def flow() -> None:
            app = self._app()
            async with app.run_test() as pilot:
                await pilot.pause()
                transcript = "\n".join(app.transcript)
                self.assertIn(BANNER_TITLE, transcript)
                self.assertIn(BANNER_TAGLINE, transcript)
                self.assertIn("Ask a task, or /help", transcript)
                # The KAAL block wordmark is the home hero.
                self.assertIn(KAAL_ART.splitlines()[0], transcript)
                self.assertIn(KAAL_ART.splitlines()[-1], transcript)
                # The sea lion is no longer the home hero (still exported).
                self.assertNotIn(SEA_LION.splitlines()[0], transcript)
                # /new re-renders the home banner with a fresh session.
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
                self.assertIn(BANNER_TITLE, transcript)
                self.assertIn(BANNER_TAGLINE, transcript)
                self.assertIn("Ask a task, or /help", transcript)

        asyncio.run(flow())

    def test_workspace_chrome_and_starter_action(self):
        """The empty state exposes context, a clear composer, and a working starter."""
        async def flow() -> None:
            app = HarnessTui(
                gateway=FakeGateway([("content", "Starter answer."), ("done", "stop")]),
                memory_root=self.root / ".agent-memory",
                project_dir=self.root,
            )
            async with app.run_test() as pilot:
                await pilot.pause()
                self.assertEqual(str(app.query_one("#brand-mark").render()), "KAAL")
                self.assertIn(
                    "fake-model", str(app.query_one("#topbar-session").render())
                )
                self.assertIn(
                    "Yudhishthira",
                    str(app.query_one("#conversation-context").render()),
                )
                self.assertIn(
                    f"{app._tool_count} tools",
                    str(app.query_one("#sidebar-summary").render()),
                )
                # The composer is clean: no placeholder text, no key-hint line
                # (the /help command documents the keys).
                self.assertEqual(app.query_one("#prompt").placeholder, "")
                self.assertEqual(len(app.query("#composer-hint")), 0)
                self.assertEqual(len(app.query(".starter-explore")), 1)
                clicked = await pilot.click(app.query_one(".starter-explore"))
                self.assertTrue(clicked)
                for _ in range(200):
                    if not app.turn_active:
                        break
                    await pilot.pause(0.05)
                await pilot.pause()
                self.assertFalse(app.turn_active)
                transcript = "".join(app.transcript)
                self.assertIn(
                    "Explore this repository and summarize its architecture.", transcript
                )
                self.assertIn("Starter answer.", transcript)

        asyncio.run(flow())

    def test_mermaid_auto_renders_at_turn_end(self):
        """A ```mermaid fence in the answer is auto-converted: termaid runs
        on a worker and the Unicode art mounts below the assistant block."""
        async def flow() -> None:
            answer = (
                "Here is the flow.\n\n"
                "```mermaid\n"
                "flowchart LR\n  A --> B\n"
                "```\n"
            )
            app = HarnessTui(
                gateway=FakeGateway([("content", answer), ("done", "stop")]),
                memory_root=self.root / ".agent-memory",
                project_dir=self.root,
            )
            async with app.run_test() as pilot:
                fake = mock.Mock(returncode=0, stdout="A --> B\n", stderr="")
                with mock.patch(
                    "harness.tui.shutil.which", return_value="/usr/bin/termaid"
                ), mock.patch(
                    "harness.tui.subprocess.run", return_value=fake
                ) as run:
                    prompt = app.query_one("#prompt", TextArea)
                    prompt.text = "draw the flow"
                    prompt.focus()
                    await pilot.pause()
                    await pilot.press("enter")
                    for _ in range(200):
                        if not app.turn_active:
                            break
                        await pilot.pause(0.05)
                    # The worker thread marshals the art back; wait for mount.
                    for _ in range(100):
                        if app.query(".diagram-box"):
                            break
                        await pilot.pause(0.05)
                    await pilot.pause()
                run.assert_called_once()
                box = app.query_one(".diagram-box", Static)
                self.assertIn("A --> B", str(box.render()))
                # Transcript keeps the verbatim mermaid source, not the art.
                self.assertIn("```mermaid", "".join(app.transcript))

        asyncio.run(flow())

    def test_mermaid_missing_termaid_notice(self):
        """Fence detected but no termaid: one dim notice, no crash."""
        async def flow() -> None:
            answer = "```mermaid\nflowchart LR\n  A --> B\n```\n"
            app = HarnessTui(
                gateway=FakeGateway([("content", answer), ("done", "stop")]),
                memory_root=self.root / ".agent-memory",
                project_dir=self.root,
            )
            async with app.run_test() as pilot:
                with mock.patch("harness.tui.shutil.which", return_value=None):
                    prompt = app.query_one("#prompt", TextArea)
                    prompt.text = "draw"
                    prompt.focus()
                    await pilot.pause()
                    await pilot.press("enter")
                    for _ in range(200):
                        if not app.turn_active:
                            break
                        await pilot.pause(0.05)
                    await pilot.pause()
                self.assertIn(
                    "termaid not installed", "\n".join(app.transcript)
                )
                self.assertEqual(len(app.query(".diagram-box")), 0)

        asyncio.run(flow())

    def test_diagram_command_renders_via_termaid(self):
        """/diagram <file> renders the mermaid file through termaid and
        prints its Unicode art into the conversation."""
        async def flow() -> None:
            app = self._app()
            async with app.run_test() as pilot:
                fake = mock.Mock(returncode=0, stdout="A --> B\n", stderr="")
                with mock.patch(
                    "harness.tui.shutil.which", return_value="/usr/bin/termaid"
                ), mock.patch(
                    "harness.tui.subprocess.run", return_value=fake
                ) as run:
                    prompt = app.query_one("#prompt", TextArea)
                    prompt.text = "/diagram plan.mmd"
                    prompt.focus()
                    await pilot.pause()
                    await pilot.press("enter")
                    await pilot.pause()
                run.assert_called_once()
                self.assertIn("A --> B", "\n".join(app.transcript))

        asyncio.run(flow())

    def test_send_button_submits_composer_text(self):
        """The visible Send button follows the same path as Enter."""
        async def flow() -> None:
            app = HarnessTui(
                gateway=FakeGateway([("content", "Button answer."), ("done", "stop")]),
                memory_root=self.root / ".agent-memory",
                project_dir=self.root,
            )
            async with app.run_test() as pilot:
                prompt = app.query_one("#prompt", TextArea)
                prompt.text = "sent from the button"
                await pilot.click("#send-button")
                for _ in range(200):
                    if not app.turn_active:
                        break
                    await pilot.pause(0.05)
                await pilot.pause()
                transcript = "".join(app.transcript)
                self.assertIn("sent from the button", transcript)
                self.assertIn("Button answer.", transcript)
                self.assertEqual(app.query_one("#send-button").label, "Send")

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
                self.assertIn(".kaal/STRUCTURE.md", transcript)
                self.assertIn("# Project Structure", transcript)

        asyncio.run(flow())

    def test_large_answer_windowed_markdown(self):
        """A >20k answer streams into bounded markdown windows instead of a
        raw-fallback block: at least 4 windows are created, the window texts
        concatenate back to the exact answer in order, and the full tail is
        mirrored to the transcript."""

        async def flow() -> None:
            para = "A paragraph of streaming prose for the windowed test. " * 8 + "\n\n"
            chunk = para * 10
            tail = para * 9 + "THE FINAL UNIQUE TAIL MARKER."
            big = chunk * 5 + tail
            self.assertGreater(len(big), 20_000)
            app = HarnessTui(
                gateway=FakeGateway(
                    [("content", chunk)] * 5 + [("content", tail)] + [("done", "stop")]
                ),
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
                # Windowed: >= 4 bounded markdown windows were created.
                self.assertGreaterEqual(len(app._md_windows), 4)
                # No raw fallback: the windows concatenate to the full answer.
                self.assertEqual("".join(app._md_window_text), big)
                transcript = "".join(app.transcript)
                self.assertIn("THE FINAL UNIQUE TAIL MARKER", transcript)
                # Content order preserved: the answer start precedes the tail.
                self.assertLess(
                    transcript.index(big[:20]),
                    transcript.index("THE FINAL UNIQUE TAIL MARKER"),
                )

        asyncio.run(flow())

    def test_small_chunk_instant_flush(self):
        """Small deltas flush SYNCHRONOUSLY — no 100 ms timer wait. After a
        short pause (well under the throttle) the answer is already rendered
        and no flush timer is armed."""

        async def flow() -> None:
            app = HarnessTui(
                gateway=FakeGateway(
                    [("content", "hello "), ("content", "world"), ("done", "stop")]
                ),
                memory_root=self.root / ".agent-memory",
                project_dir=self.root,
            )
            async with app.run_test() as pilot:
                prompt = app.query_one("#prompt", TextArea)
                prompt.text = "say hi"
                prompt.focus()
                await pilot.pause()
                await pilot.press("enter")
                # ~50 ms is under the 100 ms throttle: the instant-flush path
                # must have rendered already (a timer-based flush would still
                # be pending, with _md_timer armed).
                await pilot.pause(0.05)
                self.assertIn("hello world", "".join(app.transcript))
                self.assertIsNone(app._md_timer)
                self.assertEqual(app._md_pending, [])
                for _ in range(200):  # let the turn finish cleanly
                    if not app.turn_active:
                        break
                    await pilot.pause(0.05)
                await pilot.pause()

        asyncio.run(flow())

    def test_fence_spans_markdown_windows(self):
        """A code fence that crosses a window boundary renders contiguously:
        every window stays balanced, all code lines land inside fence blocks,
        and the transcript keeps the verbatim answer."""
        from markdown_it import MarkdownIt

        para = "prose paragraph " * 60 + "\n\n"
        pre = para * 3 + "x" * 1000 + "\n"
        body = "```\n" + "code one\n" * 20 + "\n\n" + "code two\n" * 20 + "```\n"
        tail = para + "\n## The tail\n\nReal markdown prose at the end.\n"
        answer = pre + body + tail

        async def flow() -> None:
            app = HarnessTui(
                gateway=FakeGateway([("content", answer), ("done", "stop")]),
                memory_root=self.root / ".agent-memory",
                project_dir=self.root,
            )
            async with app.run_test() as pilot:
                prompt = app.query_one("#prompt", TextArea)
                prompt.text = "fence"
                prompt.focus()
                await pilot.pause()
                await pilot.press("enter")
                for _ in range(200):  # up to ~10s
                    if not app.turn_active:
                        break
                    await pilot.pause(0.05)
                await pilot.pause()
                texts = app._md_window_text
                self.assertGreaterEqual(len(texts), 2)
                code: list[str] = []
                for text in texts:
                    fences = [
                        tok
                        for tok in MarkdownIt("commonmark").parse(text)
                        if tok.type == "fence"
                    ]
                    # Each window is independently balanced (0 or 1 fence).
                    self.assertLessEqual(len(fences), 1)
                    code += [tok.content for tok in fences]
                joined_code = "\n".join(code)
                self.assertEqual(joined_code.count("code one"), 20)
                self.assertEqual(joined_code.count("code two"), 20)
                # The tail after the fence renders as real markdown.
                self.assertIn("## The tail", "".join(texts))
                # Verbatim transcript mirror keeps the whole answer.
                self.assertIn(answer[-30:], "".join(app.transcript))

        asyncio.run(flow())

    def test_tool_call_renders_compact_line(self):
        """Live tool calls render as ONE compact dim line (no bordered box):
        '⚙ name(args) → ✓ preview' in the conversation, with the detailed
        preview kept on the sidebar Trace tab."""

        async def flow() -> None:
            app = self._app()
            async with app.run_test() as pilot:
                await self._submit_and_wait(app, "write hello.txt", pilot)
                # No bordered tool boxes remain.
                self.assertEqual(len(app.query(".tool-box")), 0)
                tool_line = app.query_one(".tool-line", Static)
                text = str(tool_line.render())
                self.assertTrue(text.startswith("⚙ write("))
                self.assertIn("→ ✓", text)
                self.assertIn("wrote hello.txt", text)
                # The sidebar Trace tab keeps the detailed preview line.
                trace_lines = app.query_one("#trace").query(".trace-line")
                self.assertGreaterEqual(len(trace_lines), 1)
                trace_text = str(trace_lines[0].render())
                self.assertIn("⚙ write(", trace_text)
                self.assertIn("wrote hello.txt", trace_text)

        asyncio.run(flow())

    def test_glued_fence_auto_repairs(self):
        """A closing ``` glued to a content line (the DSML-sample slip) no
        longer dumps the whole tail into one code block: the glued close is
        split onto its own line at turn end, the tail renders as real
        markdown, and the transcript keeps the model's verbatim text."""
        from markdown_it import MarkdownIt

        answer = (
            "Intro with **bold** prose.\n\n"
            "```\n"
            '<｜DSML｜invoke name="read"><｜DSML｜parameter name="path">foo.py```\n'
            "\n"
            "## The tail\n\n"
            "Prose that must render as markdown, not code.\n"
        )

        async def flow() -> None:
            app = HarnessTui(
                gateway=FakeGateway([("content", answer), ("done", "stop")]),
                memory_root=self.root / ".agent-memory",
                project_dir=self.root,
            )
            async with app.run_test() as pilot:
                prompt = app.query_one("#prompt", TextArea)
                prompt.text = "repro"
                prompt.focus()
                await pilot.pause()
                await pilot.press("enter")
                for _ in range(200):  # up to ~10s
                    if not app.turn_active:
                        break
                    await pilot.pause(0.05)
                await pilot.pause()
                md_text = app._turn_md_text
                # The glued close was split: the envelope is a one-line code
                # block and the tail is back to real markdown.
                self.assertIn('path">foo.py\n```', md_text)
                toks = MarkdownIt("commonmark").parse(md_text)
                fences = [t for t in toks if t.type == "fence"]
                self.assertEqual(len(fences), 1)
                self.assertEqual(
                    fences[0].content.rstrip("\n"),
                    '<｜DSML｜invoke name="read"><｜DSML｜parameter name="path">foo.py',
                )
                self.assertTrue(any(t.type == "heading_open" for t in toks))
                transcript = "".join(app.transcript)
                # Verbatim mirror: tail present, and the repair's extra fence
                # never leaks into the transcript (model wrote exactly 2 ```).
                self.assertIn(answer[-30:], transcript)
                self.assertEqual(transcript.count("```"), 2)

        asyncio.run(flow())

    def test_dangling_fence_auto_closes(self):
        """An answer that simply ends with an unclosed ``` gets the closing
        fence appended at turn end so the tail isn't swallowed as code."""
        async def flow() -> None:
            answer = "Intro **bold**.\n\n```"
            app = HarnessTui(
                gateway=FakeGateway([("content", answer), ("done", "stop")]),
                memory_root=self.root / ".agent-memory",
                project_dir=self.root,
            )
            async with app.run_test() as pilot:
                prompt = app.query_one("#prompt", TextArea)
                prompt.text = "repro"
                prompt.focus()
                await pilot.pause()
                await pilot.press("enter")
                for _ in range(200):  # up to ~10s
                    if not app.turn_active:
                        break
                    await pilot.pause(0.05)
                await pilot.pause()
                # Balanced: the markdown widget now ends with a closing fence.
                self.assertTrue(app._turn_md_text.endswith("\n```"))
                # Transcript still mirrors the verbatim (unbalanced) answer.
                self.assertTrue("".join(app.transcript).endswith("```"))

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

    def test_tmux_bar_contents(self):
        """After two scripted turns the #status bar is tmux-style: agent
        block, step/tok/s/cache/cost segments, and a formatted clock — the raw
        full session id never appears. Model/session identity lives in the
        workbench topbar, not the bar."""

        async def flow() -> None:
            app = HarnessTui(
                gateway=FakeGateway(
                    list(TURN_TOOL), list(TURN_STOP), list(TURN_TOOL), list(TURN_STOP)
                ),
                memory_root=self.root / ".agent-memory",
                project_dir=self.root,
            )
            async with app.run_test() as pilot:
                await self._submit_and_wait(app, "turn one", pilot)
                await self._submit_and_wait(app, "turn two", pilot)
                status = str(app.query_one("#status", Static).render())
                # The bar leads with the active agent's name (auto-activated
                # Yudhishthira at startup), then the live metric segments.
                self.assertTrue(status.lstrip().startswith("Yudhishthira"))
                self.assertIn("step", status)
                self.assertIn("tok/s", status)
                self.assertIn("cache", status)
                self.assertIn("$", status)
                self.assertRegex(status, r"\d{2}:\d{2}")  # HH:MM clock
                self.assertNotRegex(status, r"\d{8}-")  # raw session id stays out
                self.assertGreater(app._total_cost, 0.0)

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
                # Live elapsed seconds: the wait is measured, not silent.
                self.assertRegex(
                    str(app._thinking.render()), r"💭 thinking \d+\.\ds"
                )
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
