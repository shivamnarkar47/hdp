"""Agent loop tests: streaming, DSML healing, tool execution, persistence."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from harness.loop import AgentLoop, LoopError
from harness.memory import Memory
from harness.sessions import load_messages
from harness.tools import ToolRegistry

# DSML envelope escape helpers (see harness/dialect.py).
FW = "\uff5c"  # fullwidth pipe, U+FF5C
B = "\u2581"  # block glyph, U+2581

DSML_WRITE = (
    f"<{FW}DSML{FW}tool_calls>"
    f"<{FW}DSML{FW}invoke name=\"write\">"
    f"<{FW}DSML{FW}parameter name=\"path\" string=\"true\">hello.txt</{FW}DSML{FW}parameter>"
    f"<{FW}DSML{FW}parameter name=\"content\" string=\"true\">hi</{FW}DSML{FW}parameter>"
    f"</{FW}DSML{FW}invoke>"
    f"</{FW}DSML{FW}tool_calls>"
)


class FakeGateway:
    """Yields pre-scripted StreamEvents; records every stream() invocation."""

    def __init__(self, *scripts):
        self.scripts = list(scripts)
        self.calls = []  # (messages, tools) per stream() call

    def stream(self, messages, tools):
        self.calls.append((messages, tools))
        yield from self.scripts.pop(0)


class TestAgentLoop(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tempdir = Path(self._tmp.name)
        self.memory = Memory(self.tempdir / ".agent-memory")
        self.tools = ToolRegistry(memory=self.memory, project_dir=self.tempdir)
        self._old_sessions_dir = os.environ.get("HARNESSDP_SESSIONS_DIR")
        os.environ["HARNESSDP_SESSIONS_DIR"] = str(self.tempdir / "sessions")
        self.session_id = "test-loop"

    def tearDown(self):
        if self._old_sessions_dir is None:
            os.environ.pop("HARNESSDP_SESSIONS_DIR", None)
        else:
            os.environ["HARNESSDP_SESSIONS_DIR"] = self._old_sessions_dir
        self._tmp.cleanup()

    # -- tests ---------------------------------------------------------------

    def test_two_turn_tool_call_flow(self):
        turn1 = [
            ("reasoning", "Let me check the directory"),
            ("content", "I will write the file. "),
            ("content", DSML_WRITE),
            ("done", "tool_calls"),
        ]
        turn2 = [("content", "Wrote hello.txt."), ("done", "stop")]
        gateway = FakeGateway(turn1, turn2)
        events: list = []
        loop = AgentLoop(gateway, self.tools, self.memory, self.session_id)
        answer = loop.run("Write the file", emit=events.append)

        self.assertEqual(answer, "Wrote hello.txt.")
        self.assertEqual((self.tempdir / "hello.txt").read_text(encoding="utf-8"), "hi")

        # Turn-2 wire history replays the assistant turn — reasoning verbatim,
        # tool call embedded — followed by the tool result.
        history = gateway.calls[1][0]
        assistant = next(m for m in history if m["role"] == "assistant")
        self.assertEqual(assistant["content"], "I will write the file. ")
        self.assertEqual(assistant["reasoning_content"], "Let me check the directory")
        self.assertEqual(len(assistant["tool_calls"]), 1)
        call_wire = assistant["tool_calls"][0]
        self.assertEqual(call_wire["type"], "function")
        self.assertEqual(call_wire["function"]["name"], "write")
        self.assertEqual(
            json.loads(call_wire["function"]["arguments"]),
            {"path": "hello.txt", "content": "hi"},
        )
        tool_msgs = [m for m in history if m["role"] == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertEqual(tool_msgs[0]["tool_call_id"], call_wire["id"])
        self.assertIn("wrote", tool_msgs[0]["content"])

        # No DSML markers leak into visible content or the final answer.
        content = "".join(e[1] for e in events if e[0] == "content")
        self.assertNotIn("DSML", content)
        self.assertNotIn(FW, content)
        self.assertNotIn("DSML", answer)
        self.assertNotIn(FW, answer)

        # Ordered emit sequence (reasoning and step events filtered out).
        filtered = [e for e in events if e[0] not in ("reasoning", "step")]
        self.assertEqual(
            [e[0] for e in filtered],
            ["content", "tool_start", "tool_result", "content", "done"],
        )

    def test_think_span_is_reasoning(self):
        gateway = FakeGateway([("content", "<think>hmm</think>answer"), ("done", "stop")])
        events: list = []
        loop = AgentLoop(gateway, self.tools, self.memory, self.session_id)
        answer = loop.run("Think", emit=events.append)
        self.assertEqual(answer, "answer")
        reasoning = "".join(e[1] for e in events if e[0] == "reasoning")
        self.assertEqual(reasoning, "hmm")
        content = "".join(e[1] for e in events if e[0] == "content")
        self.assertNotIn("think", content)

    def test_tool_loop_abort(self):
        script = [("content", DSML_WRITE), ("done", "tool_calls")]
        gateway = FakeGateway(script, script, script)
        loop = AgentLoop(gateway, self.tools, self.memory, self.session_id)
        with self.assertRaises(LoopError) as ctx:
            loop.run("Loop")
        self.assertIn("tool loop detected", str(ctx.exception))

    def test_max_steps(self):
        script = [("content", DSML_WRITE), ("done", "tool_calls")]
        gateway = FakeGateway(script, script, script)
        loop = AgentLoop(gateway, self.tools, self.memory, self.session_id, max_steps=2)
        with self.assertRaises(LoopError) as ctx:
            loop.run("Loop")
        self.assertIn("max steps reached", str(ctx.exception))

    def test_overflow_retry_truncates_and_retries(self):
        gateway = FakeGateway(
            [("done", "length")],
            [("content", "ok"), ("done", "stop")],
        )
        loop = AgentLoop(gateway, self.tools, self.memory, self.session_id)
        answer = loop.run("Overflow")
        self.assertEqual(answer, "ok")
        self.assertEqual(len(gateway.calls), 2)

    def test_sessions_round_trip(self):
        turn1 = [
            ("reasoning", "Let me check the directory"),
            ("content", "I will write the file. "),
            ("content", DSML_WRITE),
            ("done", "tool_calls"),
        ]
        turn2 = [("content", "Wrote hello.txt."), ("done", "stop")]
        gateway = FakeGateway(turn1, turn2)
        AgentLoop(gateway, self.tools, self.memory, self.session_id).run("Write the file")

        replay = load_messages(self.session_id)
        self.assertEqual(replay[0]["role"], "user")
        assistant = next(m for m in replay if m["role"] == "assistant")
        self.assertEqual(assistant["reasoning_content"], "Let me check the directory")
        self.assertEqual(assistant["tool_calls"][0]["name"], "write")
        self.assertEqual(
            json.loads(assistant["tool_calls"][0]["arguments"]),
            {"path": "hello.txt", "content": "hi"},
        )
        tool_msgs = [m for m in replay if m["role"] == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertEqual(tool_msgs[0]["tool_call_id"], assistant["tool_calls"][0]["id"])
        self.assertIn("wrote", tool_msgs[0]["content"])

    def test_emit_none(self):
        gateway = FakeGateway([("content", "hi"), ("done", "stop")])
        loop = AgentLoop(gateway, self.tools, self.memory, self.session_id)
        self.assertEqual(loop.run("Say hi"), "hi")


if __name__ == "__main__":
    unittest.main()
