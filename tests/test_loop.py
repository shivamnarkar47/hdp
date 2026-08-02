"""Agent loop tests: streaming, DSML healing, tool execution, persistence."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path

from harness.config import CONTEXT_WINDOW, MAX_OUTPUT_TOKENS
from harness.context import wire_token_count
from harness.loop import AgentLoop, LoopError
from harness.memory import Memory
from harness.messages import ToolCall, to_wire_messages
from harness.sessions import append_event, load_messages, read_events
from harness.toolcache import ToolCache
from harness.tools import ToolError, ToolRegistry

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


class StubRegistry:
    """Minimal tool registry: sleepable read handler recording call order/threads.

    `record` gets (name, path_or_None, thread_name) per execution, in execution
    order. Reads with a path in `fail_reads` raise ToolError (a tool failure
    that counts toward the consecutive-failure budget).
    """

    def __init__(self, project_dir, read_delay=0.0, fail_reads=()):
        self.project_dir = Path(project_dir)
        self._read_delay = read_delay
        self._fail_reads = set(fail_reads)
        self.record = []

    def schemas(self):
        return []

    def begin_batch(self, names, structure_sig):
        """No cache configured -> no-op (the registry's cache contract)."""
        pass

    def end_batch(self, mutated):
        pass

    def execute(self, name, args):
        thread = threading.current_thread().name
        path = args.get("path") if isinstance(args, dict) else None
        self.record.append((name, path, thread))
        if name == "read":
            if self._read_delay:
                time.sleep(self._read_delay)
            if path in self._fail_reads:
                raise ToolError(f"blocked: no such file: {path}")
            return f"read {path}"
        if name == "write":
            return "wrote"
        if name == "grep":
            return "no matches"
        if name == "glob":
            return "[]"
        raise ToolError(f"unknown tool: {name}")


class CountingToolRegistry(ToolRegistry):
    """Real ToolRegistry wired to a real ToolCache, counting read handler calls.

    `read_calls` counts ACTUAL handler invocations, so a cache hit (which
    returns without the handler) is observable as a missing increment.
    """

    def __init__(self, project_dir):
        self._cache_path = Path(project_dir) / ".kala" / "tool-cache.json"
        self.read_calls = 0
        super().__init__(project_dir=project_dir, cache=ToolCache(self._cache_path))

    def _tool_read(self, args):
        self.read_calls += 1
        return super()._tool_read(args)


class TestAgentLoop(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tempdir = Path(self._tmp.name)
        self.memory = Memory(self.tempdir / ".agent-memory")
        self.tools = ToolRegistry(memory=self.memory, project_dir=self.tempdir)
        self._old_sessions_dir = os.environ.get("KALA_SESSIONS_DIR")
        os.environ["KALA_SESSIONS_DIR"] = str(self.tempdir / "sessions")
        self.session_id = "test-loop"

    def tearDown(self):
        if self._old_sessions_dir is None:
            os.environ.pop("KALA_SESSIONS_DIR", None)
        else:
            os.environ["KALA_SESSIONS_DIR"] = self._old_sessions_dir
        self._tmp.cleanup()

    # -- tests ---------------------------------------------------------------

    def test_two_turn_tool_call_flow(self):
        turn1 = [
            ("reasoning", "Let me check the directory"),
            # Real envelopes are generation-leading: the envelope must be the
            # FIRST content after the think span, or DialectFeed reads it as a
            # prose quote of the envelope (see dialect.py Part A guard).
            ("content", DSML_WRITE),
            ("content", "I will write the file. "),
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

    def test_agent_persona_injected_into_system_prompt(self):
        """With an agent persona, the turn-1 wire system message names it."""
        gateway = FakeGateway([("content", "ok"), ("done", "stop")])
        loop = AgentLoop(
            gateway,
            self.tools,
            self.memory,
            self.session_id,
            agent={"name": "Arjuna", "description": "precise"},
        )
        loop.run("hi")
        wire = gateway.calls[0][0]
        system = next(m for m in wire if m["role"] == "system")
        self.assertIn("Arjuna", system["content"])
        self.assertIn("precise", system["content"])

    def test_no_agent_no_persona(self):
        """Without an agent, the system prompt carries no persona block."""
        gateway = FakeGateway([("content", "ok"), ("done", "stop")])
        loop = AgentLoop(gateway, self.tools, self.memory, self.session_id)
        loop.run("hi")
        wire = gateway.calls[0][0]
        system = next(m for m in wire if m["role"] == "system")
        self.assertNotIn("Arjuna", system["content"])
        self.assertNotIn("## Agent", system["content"])

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
            # Real envelopes are generation-leading: the envelope must be the
            # FIRST content after the think span, or DialectFeed reads it as a
            # prose quote of the envelope (see dialect.py Part A guard).
            ("content", DSML_WRITE),
            ("content", "I will write the file. "),
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

    def test_structure_cache_refreshed_after_tools(self):
        turn1 = [
            ("reasoning", "Let me check the directory"),
            # Real envelopes are generation-leading: the envelope must be the
            # FIRST content after the think span, or DialectFeed reads it as a
            # prose quote of the envelope (see dialect.py Part A guard).
            ("content", DSML_WRITE),
            ("content", "I will write the file. "),
            ("done", "tool_calls"),
        ]
        turn2 = [("content", "Wrote hello.txt."), ("done", "stop")]
        gateway = FakeGateway(turn1, turn2)
        loop = AgentLoop(gateway, self.tools, self.memory, self.session_id)
        loop.run("Write the file")
        # The loop created the structure cache and refreshed it after the
        # write tool, so the new file appears in the doc.
        doc = (self.tempdir / ".kala" / "STRUCTURE.md").read_text(encoding="utf-8")
        self.assertIn("hello.txt", doc)
        self.assertIn("<!-- sig: ", doc)

    def test_emit_none(self):
        gateway = FakeGateway([("content", "hi"), ("done", "stop")])
        loop = AgentLoop(gateway, self.tools, self.memory, self.session_id)
        self.assertEqual(loop.run("Say hi"), "hi")

    def test_preemptive_truncation_before_stream(self):
        """A wire history over the prompt budget is truncated BEFORE streaming.

        The huge user message is seeded as an old turn (user + assistant +
        tool result) so the loop's own small prompt is the protected last user
        message; truncate_history can then drop the oversized turn.
        """
        budget = CONTEXT_WINDOW - MAX_OUTPUT_TOKENS
        big = "x" * 700_000  # ~233k tokens per message; 3 messages blow past 616k
        append_event(self.session_id, {"type": "user", "data": {"content": big}})
        append_event(self.session_id, {"type": "assistant", "data": {"content": big}})
        append_event(
            self.session_id,
            {"type": "tool_result", "data": {"tool_call_id": "t1", "content": big}},
        )

        gateway = FakeGateway([("content", "Wrote the file."), ("done", "stop")])
        loop = AgentLoop(gateway, self.tools, self.memory, self.session_id, resume=True)
        answer = loop.run("finish")
        self.assertEqual(answer, "Wrote the file.")

        # The wire recorded on the very first stream() call was already
        # truncated: it fits the budget and no longer carries the huge content.
        recorded = gateway.calls[0][0]
        self.assertLessEqual(wire_token_count(recorded), budget)
        self.assertNotIn(big, json.dumps(recorded, ensure_ascii=False))
        # Only one turn ran (single stream call), so usage reflects it.
        self.assertEqual(len(gateway.calls), 1)
        self.assertGreater(loop.usage["input_tokens"], 0)
        self.assertGreater(loop.usage["output_tokens"], 0)

    def test_usage_accounting_two_turns(self):
        """usage accumulates input+output tokens over a normal two-turn run."""
        turn1 = [
            ("reasoning", "Let me check the directory"),
            # Real envelopes are generation-leading: the envelope must be the
            # FIRST content after the think span, or DialectFeed reads it as a
            # prose quote of the envelope (see dialect.py Part A guard).
            ("content", DSML_WRITE),
            ("content", "I will write the file. "),
            ("done", "tool_calls"),
        ]
        turn2 = [("content", "Wrote hello.txt."), ("done", "stop")]
        gateway = FakeGateway(turn1, turn2)
        loop = AgentLoop(gateway, self.tools, self.memory, self.session_id)
        loop.run("Write the file")
        self.assertGreater(loop.usage["input_tokens"], 0)
        self.assertGreater(loop.usage["output_tokens"], 0)
        self.assertEqual(loop.usage["input_tokens"], sum(
            wire_token_count(calls[0]) for calls in gateway.calls
        ))

    def test_prose_envelope_quote_not_truncated(self):
        """A mid-answer quote of the DSML envelope must not swallow the rest.

        Regression: the model explained the wire format in prose (backticked
        ``<|DSML|tool_calls>``, no close tag).  DialectFeed used to treat the
        quote as a real section open, swallow the remainder of the stream, and
        discard it at flush() — the persisted answer was cut off mid-sentence
        while finish_reason was ``stop``, so the partial answer was returned
        as ok.  The complete-envelope variant used to heal a phantom
        ToolCall that the loop would have EXECUTED (a ``write``!).
        """
        quoted = (
            "Let me explain the wire format. The gateway streams the model's "
            "output as SSE deltas, and when the model wants to call a tool it "
            "emits an XML-style envelope `<|DSML|tool_calls>` followed by "
            "invoke and parameter tags, which leak into the visible content "
            "stream. Now, the important part: when the model quotes that "
            "envelope in prose, the healer must not mistake the quote for a "
            "real tool call, because doing so swallows everything that "
            "follows and the answer sticks on half — cut off right where the "
            "model explained the format. The rest of this answer must be "
            "preserved completely, with every single word intact."
        )
        gateway = FakeGateway([("content", quoted), ("done", "stop")])
        loop = AgentLoop(gateway, self.tools, self.memory, self.session_id)
        answer = loop.run("Explain the wire format")
        self.assertTrue(answer.endswith("with every single word intact."))
        self.assertNotIn("DSML", answer)
        self.assertNotIn(FW, answer)
        self.assertEqual(len(gateway.calls), 1)  # no tool turn was executed
        self.assertFalse((self.tempdir / "hello.txt").exists())

        # Variant 2: a COMPLETE envelope quoted inside prose — the old bug
        # healed a phantom write call and executed it; now nothing runs and
        # the full answer is returned.
        complete = (
            "The write tool writes files; its envelope looks like this: "
            "<|DSML|tool_calls>"
            '<|DSML|invoke name="write">'
            '<|DSML|parameter name="path" string="true">hello.txt</|DSML|parameter>'
            '<|DSML|parameter name="content" string="true">boom</|DSML|parameter>'
            "</|DSML|invoke>"
            "</|DSML|tool_calls>"
            " — but that is just an example, I will not actually call it."
        )
        gateway2 = FakeGateway([("content", complete), ("done", "stop")])
        loop2 = AgentLoop(gateway2, self.tools, self.memory, self.session_id + "-2")
        answer2 = loop2.run("Show the envelope")
        self.assertFalse((self.tempdir / "hello.txt").exists())  # nothing executed
        self.assertIn("but that is just an example", answer2)
        self.assertTrue(answer2.endswith("I will not actually call it."))
        self.assertEqual(len(gateway2.calls), 1)

    def test_incremental_wire_matches_full_rebuild(self):
        """Finding 7: after a multi-turn run the incremental wire cache is
        byte-identical to a full to_wire_messages rebuild."""
        turn1 = [
            ("reasoning", "Let me check the directory"),
            ("content", DSML_WRITE),
            ("content", "I will write the file. "),
            ("done", "tool_calls"),
        ]
        turn2 = [("content", "Wrote hello.txt."), ("done", "stop")]
        gateway = FakeGateway(turn1, turn2)
        loop = AgentLoop(gateway, self.tools, self.memory, self.session_id)
        loop.run("Write the file")
        self.assertEqual(loop._wire, to_wire_messages(loop._messages))
        self.assertEqual(loop._wire_tokens, wire_token_count(loop._wire))

    def test_parallel_reads_wall_time_and_order(self):
        """An all-read batch runs concurrently: wall time ~ max, not sum, with
        results and events in original call order."""
        delay = 0.4

        single_turn1 = [
            ("tool_call", ToolCall("s1", "read", '{"path": "a.txt"}')),
            ("done", "tool_calls"),
        ]
        single_turn2 = [("content", "ok"), ("done", "stop")]
        single_gateway = FakeGateway(single_turn1, single_turn2)
        single_stub = StubRegistry(self.tempdir, read_delay=delay)
        t0 = time.monotonic()
        AgentLoop(single_gateway, single_stub, self.memory, self.session_id + "-single").run("one")
        single_time = time.monotonic() - t0

        batch_turn1 = [
            ("tool_call", ToolCall("c1", "read", '{"path": "a.txt"}')),
            ("tool_call", ToolCall("c2", "read", '{"path": "b.txt"}')),
            ("tool_call", ToolCall("c3", "read", '{"path": "c.txt"}')),
            ("done", "tool_calls"),
        ]
        batch_turn2 = [("content", "all read"), ("done", "stop")]
        gateway = FakeGateway(batch_turn1, batch_turn2)
        stub = StubRegistry(self.tempdir, read_delay=delay)
        events: list = []
        loop = AgentLoop(gateway, stub, self.memory, self.session_id + "-batch")
        t0 = time.monotonic()
        answer = loop.run("read three", emit=events.append)
        batch_time = time.monotonic() - t0

        self.assertEqual(answer, "all read")
        # Parallel: a serial 3-read batch would take ~3x the delay; require
        # the batch to be well under 2x the single-call wall time.
        self.assertLess(batch_time, 2 * single_time)

        # All three results present, in call order, on pool threads.
        reads = [r for r in stub.record if r[0] == "read"]
        self.assertEqual([r[1] for r in reads], ["a.txt", "b.txt", "c.txt"])
        self.assertGreaterEqual(len({r[2] for r in reads}), 2)
        self.assertTrue(all(r[2].startswith("ThreadPoolExecutor") for r in reads))

        # Events: all tool_starts first (in order), then tool_results in order.
        starts = [(e[1].id) for e in events if e[0] == "tool_start"]
        results = [(e[1], e[2]) for e in events if e[0] == "tool_result"]
        self.assertEqual(starts, ["c1", "c2", "c3"])
        self.assertEqual([r[0] for r in results], ["c1", "c2", "c3"])
        self.assertEqual([r[1] for r in results], ["read a.txt", "read b.txt", "read c.txt"])

        # Persisted order == call order (replayable through the session store).
        replay = load_messages(self.session_id + "-batch")
        tool_msgs = [m for m in replay if m["role"] == "tool"]
        self.assertEqual(
            [(m["tool_call_id"], m["content"]) for m in tool_msgs],
            [("c1", "read a.txt"), ("c2", "read b.txt"), ("c3", "read c.txt")],
        )

    def test_mixed_batch_runs_serially(self):
        """A batch containing a mutator runs serially in call order."""
        turn1 = [
            ("tool_call", ToolCall("c1", "read", '{"path": "a.txt"}')),
            ("tool_call", ToolCall("c2", "write", '{"path": "out.txt", "content": "x"}')),
            ("tool_call", ToolCall("c3", "read", '{"path": "c.txt"}')),
            ("done", "tool_calls"),
        ]
        turn2 = [("content", "done"), ("done", "stop")]
        gateway = FakeGateway(turn1, turn2)
        stub = StubRegistry(self.tempdir)
        events: list = []
        loop = AgentLoop(gateway, stub, self.memory, self.session_id + "-mixed")
        loop.run("mix", emit=events.append)

        # Serial: execution order == call order, on one (main) thread.
        self.assertEqual(
            [(r[0], r[1]) for r in stub.record],
            [("read", "a.txt"), ("write", "out.txt"), ("read", "c.txt")],
        )
        self.assertEqual(len({r[2] for r in stub.record}), 1)
        # Emit interleaving preserved: start/result alternate per call.
        filtered = []
        for e in events:
            if e[0] == "tool_start":
                filtered.append((e[0], e[1].id))
            elif e[0] == "tool_result":
                filtered.append((e[0], e[1]))
        self.assertEqual(
            filtered,
            [
                ("tool_start", "c1"), ("tool_result", "c1"),
                ("tool_start", "c2"), ("tool_result", "c2"),
                ("tool_start", "c3"), ("tool_result", "c3"),
            ],
        )

    def test_consecutive_failure_abort_across_parallel_batch(self):
        """5 consecutive tool failures still abort inside a parallel all-read
        batch, order-deterministically on the main thread. Distinct paths keep
        the (name, args) keys distinct so the tool-loop guard does not fire
        first."""
        turn1 = [
            ("tool_call", ToolCall(f"f{i}", "read", f'{{"path": "missing{i}.txt"}}'))
            for i in range(5)
        ] + [("done", "tool_calls")]
        gateway = FakeGateway(turn1)
        stub = StubRegistry(self.tempdir, fail_reads={f"missing{i}.txt" for i in range(5)})
        events: list = []
        loop = AgentLoop(gateway, stub, self.memory, self.session_id + "-fail")
        with self.assertRaises(LoopError) as ctx:
            loop.run("fail", emit=events.append)
        self.assertIn("5 consecutive tool failures", str(ctx.exception))
        # All five started; the 5th failure aborts BEFORE its tool_result.
        self.assertEqual(
            [e[1].id for e in events if e[0] == "tool_start"], ["f0", "f1", "f2", "f3", "f4"]
        )
        self.assertEqual(
            [e[1] for e in events if e[0] == "tool_result"], ["f0", "f1", "f2", "f3"]
        )

    # -- tool-result cache ---------------------------------------------------

    def test_tool_cache_serves_repeat_reads_and_drops_after_write(self):
        """Two consecutive same-file reads with no mutation between them: the
        second is served from the cache (handler ran once). A later write step
        invalidates the cache, so the following read hits the handler again."""
        turn1 = [
            ("tool_call", ToolCall("r1", "read", '{"path": "a.txt"}')),
            ("done", "tool_calls"),
        ]
        turn2 = [
            ("tool_call", ToolCall("r2", "read", '{"path": "a.txt"}')),
            ("done", "tool_calls"),
        ]
        turn3 = [
            ("tool_call", ToolCall("w1", "write", '{"path": "out.txt", "content": "x"}')),
            ("done", "tool_calls"),
        ]
        turn4 = [
            ("tool_call", ToolCall("r3", "read", '{"path": "a.txt"}')),
            ("done", "tool_calls"),
        ]
        turn5 = [("content", "done"), ("done", "stop")]
        gateway = FakeGateway(turn1, turn2, turn3, turn4, turn5)
        stub = CountingToolRegistry(self.tempdir)
        events: list = []
        loop = AgentLoop(gateway, stub, self.memory, self.session_id + "-cache")
        loop.run("cache test", emit=events.append)

        # Steps 1 and 2 read the same file with no mutation between: the
        # handler ran ONCE (step 2 was a cache hit), results identical.
        self.assertEqual(stub.read_calls, 2)
        results = [e[2] for e in events if e[0] == "tool_result"]
        self.assertEqual(results[0], results[1])  # step-2 read served the cache
        # Step 3 wrote -> the write step's next read (step 4) missed the cache
        # and ran the handler again (read_calls 2, not 3 from steps 1/2/4).
        self.assertEqual(results[3], results[0])
        self.assertEqual((self.tempdir / "out.txt").read_text(encoding="utf-8"), "x")

    def test_same_step_write_bypasses_cache_lookups(self):
        """A batch [read, write, read] disables cache lookups for the WHOLE
        step: both reads run the handler (the same-step write-then-read hole
        would otherwise serve stale data from before the write)."""
        # Step 1 populates the cache with a same-key entry; a broken bypass
        # would then serve THAT stale result for step 2's reads.
        turn1 = [
            ("tool_call", ToolCall("r1", "read", '{"path": "x.txt"}')),
            ("done", "tool_calls"),
        ]
        turn2 = [
            ("tool_call", ToolCall("r2", "read", '{"path": "x.txt"}')),
            ("tool_call", ToolCall("w1", "write", '{"path": "y.txt", "content": "y"}')),
            ("tool_call", ToolCall("r3", "read", '{"path": "x.txt"}')),
            ("done", "tool_calls"),
        ]
        turn3 = [("content", "done"), ("done", "stop")]
        gateway = FakeGateway(turn1, turn2, turn3)
        stub = CountingToolRegistry(self.tempdir)
        events: list = []
        loop = AgentLoop(gateway, stub, self.memory, self.session_id + "-bypass")
        loop.run("mix", emit=events.append)

        # Step 1 read + the two same-batch reads = 3 handler runs; a cache
        # lookup for r2/r3 would have left read_calls at 1.
        self.assertEqual(stub.read_calls, 3)
        self.assertEqual((self.tempdir / "y.txt").read_text(encoding="utf-8"), "y")
        results = [e[2] for e in events if e[0] == "tool_result"]
        # tool_result order: r1, r2, w1, r3 — both same-batch reads (r2, r3)
        # ran the handler, returning identical fresh results.
        self.assertEqual(results[1], results[3])

    def test_tool_cache_disabled_without_signature(self):
        """No structure signature -> execute() never consults the cache (the
        graceful path for FakeGateway-style tests without a StructureManager)."""
        stub = CountingToolRegistry(self.tempdir)
        args_json = json.dumps({"path": "a.txt"}, sort_keys=True)
        stub.execute("read", {"path": "a.txt"})
        stub.execute("read", {"path": "a.txt"})
        self.assertEqual(stub.read_calls, 2)  # never cached
        # The cache file was never even created.
        self.assertFalse((self.tempdir / ".kala" / "tool-cache.json").exists())

    # -- verify hooks --------------------------------------------------------

    def _write_hooks(self, verify_cmd):
        hooks = self.tempdir / ".kala" / "hooks.json"
        hooks.parent.mkdir(parents=True, exist_ok=True)
        hooks.write_text(json.dumps({"verify": verify_cmd}), encoding="utf-8")

    def _write_turn_scripts(self):
        return [
            [("reasoning", "Let me check"), ("content", DSML_WRITE), ("done", "tool_calls")],
            [("content", "Wrote hello.txt."), ("done", "stop")],
        ]

    def test_verify_hook_runs_after_mutation(self):
        """A mutating batch with a configured verify hook runs the command and
        feeds its output back: a ("verify", ...) event, a persisted user event
        with the [verify] note, and the note as the final wire user message."""
        self._write_hooks(["python", "-c", "print('verify-ok')"])
        gateway = FakeGateway(*self._write_turn_scripts())
        events: list = []
        sid = self.session_id + "-verify"
        loop = AgentLoop(gateway, self.tools, self.memory, sid)
        answer = loop.run("Write the file", emit=events.append)
        self.assertEqual(answer, "Wrote hello.txt.")

        # New AgentEvent kind with the command output.
        verify_events = [e for e in events if e[0] == "verify"]
        self.assertEqual(len(verify_events), 1)
        self.assertIn("verify-ok", verify_events[0][1])

        # Persisted as a user event carrying the [verify] note.
        persisted = read_events(sid)
        verify_users = [
            r for r in persisted
            if r["type"] == "user" and "[verify]" in r["data"].get("content", "")
        ]
        self.assertEqual(len(verify_users), 1)
        self.assertIn("verify-ok", verify_users[0]["data"]["content"])

        # The next stream's wire carries the verify note as the LAST user
        # message (the truncation-protected position, Finding 8).
        wire = gateway.calls[-1][0]
        users = [m for m in wire if m["role"] == "user"]
        self.assertIn("[verify]", users[-1]["content"])
        self.assertIn("verify-ok", users[-1]["content"])

    def test_no_hooks_file_no_verify(self):
        gateway = FakeGateway(*self._write_turn_scripts())
        events: list = []
        loop = AgentLoop(gateway, self.tools, self.memory, self.session_id + "-noverify")
        loop.run("Write the file", emit=events.append)
        self.assertNotIn("verify", [e[0] for e in events])

    def test_invalid_hooks_json_disables_verify(self):
        hooks = self.tempdir / ".kala" / "hooks.json"
        hooks.parent.mkdir(parents=True, exist_ok=True)
        hooks.write_text("{not json", encoding="utf-8")
        gateway = FakeGateway(*self._write_turn_scripts())
        events: list = []
        loop = AgentLoop(gateway, self.tools, self.memory, self.session_id + "-badhooks")
        loop.run("Write the file", emit=events.append)
        self.assertNotIn("verify", [e[0] for e in events])

    def test_empty_verify_array_disables_verify(self):
        self._write_hooks([])
        gateway = FakeGateway(*self._write_turn_scripts())
        events: list = []
        loop = AgentLoop(gateway, self.tools, self.memory, self.session_id + "-emptyhooks")
        loop.run("Write the file", emit=events.append)
        self.assertNotIn("verify", [e[0] for e in events])

    def test_enable_verify_false_forces_off(self):
        self._write_hooks(["python", "-c", "print('verify-ok')"])
        gateway = FakeGateway(*self._write_turn_scripts())
        events: list = []
        loop = AgentLoop(
            gateway,
            self.tools,
            self.memory,
            self.session_id + "-noverifyflag",
            enable_verify=False,
        )
        loop.run("Write the file", emit=events.append)
        self.assertNotIn("verify", [e[0] for e in events])

    def test_verify_does_not_abort_on_failure(self):
        """A failing verify command is CONTENT, never a loop abort."""
        self._write_hooks(["python", "-c", "import sys; sys.exit(1)"])
        gateway = FakeGateway(*self._write_turn_scripts())
        events: list = []
        sid = self.session_id + "-verifyfail"
        loop = AgentLoop(gateway, self.tools, self.memory, sid)
        answer = loop.run("Write the file", emit=events.append)
        self.assertEqual(answer, "Wrote hello.txt.")
        verify_events = [e for e in events if e[0] == "verify"]
        self.assertEqual(len(verify_events), 1)
        # Exit code 1 with empty output still produced the note.
        persisted = read_events(sid)
        verify_users = [
            r for r in persisted
            if r["type"] == "user" and "[verify]" in r["data"].get("content", "")
        ]
        self.assertEqual(len(verify_users), 1)

    # -- spawn_agent --------------------------------------------------------

    def test_spawn_agent_nested_loop(self):
        """A spawn tool call runs a nested AgentLoop on the SAME gateway; the
        parent's tool result is a JSON summary carrying the nested answer and
        session id, and the nested session is persisted in the store."""
        parent_spawn = [
            ("tool_call", ToolCall("sp1", "spawn_agent", '{"task": "nested task"}')),
            ("done", "tool_calls"),
        ]
        nested_answer = [("content", "nested answer"), ("done", "stop")]
        parent_final = [("content", "parent final"), ("done", "stop")]
        gateway = FakeGateway(parent_spawn, nested_answer, parent_final)
        loop = AgentLoop(gateway, self.tools, self.memory, self.session_id)
        answer = loop.run("run spawn")
        self.assertEqual(answer, "parent final")
        # Three stream() calls total: parent turn 1, nested turn, parent turn 2.
        self.assertEqual(len(gateway.calls), 3)

        # The spawn tool result persisted in the parent session is the JSON
        # summary {answer, steps, usage, session_id}.
        parent_events = read_events(self.session_id)
        spawn_results = [
            r for r in parent_events
            if r["type"] == "tool_result" and "session_id" in r["data"].get("content", "")
        ]
        self.assertEqual(len(spawn_results), 1)
        summary = json.loads(spawn_results[0]["data"]["content"])
        self.assertEqual(summary["answer"], "nested answer")
        self.assertIn("session_id", summary)
        self.assertGreater(summary["steps"], 0)
        self.assertIn("usage", summary)
        # The nested session was persisted and replays the nested conversation.
        nested_events = read_events(summary["session_id"])
        self.assertNotEqual(nested_events, [])
        self.assertEqual(nested_events[0]["type"], "meta")
        self.assertIn("nested task", load_messages(summary["session_id"])[0]["content"])

    def test_spawn_agent_recursion_limit(self):
        """A spawn INSIDE the nested loop (depth 2) returns the limit string:
        no third loop is created, so the shared gateway serves exactly four
        turns (parent spawn, nested spawn attempt, nested answer, parent
        answer)."""
        parent_spawn = [
            ("tool_call", ToolCall("sp1", "spawn_agent", '{"task": "outer"}')),
            ("done", "tool_calls"),
        ]
        nested_spawn = [
            ("tool_call", ToolCall("sp2", "spawn_agent", '{"task": "inner"}')),
            ("done", "tool_calls"),
        ]
        nested_final = [("content", "nested done"), ("done", "stop")]
        parent_final = [("content", "parent done"), ("done", "stop")]
        gateway = FakeGateway(parent_spawn, nested_spawn, nested_final, parent_final)
        loop = AgentLoop(gateway, self.tools, self.memory, self.session_id)
        answer = loop.run("spawn twice")
        self.assertEqual(answer, "parent done")
        # No third loop ran: exactly these four stream() calls.
        self.assertEqual(len(gateway.calls), 4)
        self.assertEqual(gateway.scripts, [])

        parent_events = read_events(self.session_id)
        spawn_results = [
            r for r in parent_events
            if r["type"] == "tool_result" and "session_id" in r["data"].get("content", "")
        ]
        self.assertEqual(len(spawn_results), 1)
        nested_sid = json.loads(spawn_results[0]["data"]["content"])["session_id"]
        # The nested loop's own spawn_agent tool result is the limit string.
        nested_events = read_events(nested_sid)
        inner_results = [r for r in nested_events if r["type"] == "tool_result"]
        self.assertEqual(len(inner_results), 1)
        self.assertEqual(
            inner_results[0]["data"]["content"], "spawn_agent: recursion limit reached"
        )

    def test_spawn_agent_dir_escape_blocked(self):
        """A spawn with a dir escaping the project directory is an error
        string; no nested loop is created (two stream calls only)."""
        spawn_call = [
            ("tool_call", ToolCall("sp1", "spawn_agent", '{"task": "t", "dir": "../evil"}')),
            ("done", "tool_calls"),
        ]
        final = [("content", "done"), ("done", "stop")]
        gateway = FakeGateway(spawn_call, final)
        loop = AgentLoop(gateway, self.tools, self.memory, self.session_id)
        answer = loop.run("spawn")
        self.assertEqual(answer, "done")
        self.assertEqual(len(gateway.calls), 2)  # no nested loop ran
        results = [r for r in read_events(self.session_id) if r["type"] == "tool_result"]
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]["data"]["content"].startswith("spawn_agent: blocked:"))

    def test_spawn_agent_dir_not_a_directory(self):
        (self.tempdir / "plain.txt").write_text("x", encoding="utf-8")
        spawn_call = [
            ("tool_call", ToolCall("sp1", "spawn_agent", '{"task": "t", "dir": "plain.txt"}')),
            ("done", "tool_calls"),
        ]
        final = [("content", "done"), ("done", "stop")]
        gateway = FakeGateway(spawn_call, final)
        loop = AgentLoop(gateway, self.tools, self.memory, self.session_id)
        loop.run("spawn")
        results = [r for r in read_events(self.session_id) if r["type"] == "tool_result"]
        self.assertTrue(results[0]["data"]["content"].startswith("spawn_agent: not a directory:"))


if __name__ == "__main__":
    unittest.main()
