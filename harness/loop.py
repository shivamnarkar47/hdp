"""The agent loop: stream, heal DSML, execute tools, persist.

Runs one agent task end to end: builds the system prompt from memory and
project context, streams the conversation through the gateway one turn at a
time, heals leaked DSML envelopes back into ToolCall objects via DialectFeed,
executes the resolved calls against the ToolRegistry, and persists every turn
to the JSONL session store.

Loop-level failures (context overflow, max steps, tool loops, consecutive tool
failures, gateway errors) raise LoopError (exit code 2) after emitting a final
("error", ...) event and recording a session summary.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from harness import prompts, sessions
from harness.config import CONTEXT_WINDOW, MAX_OUTPUT_TOKENS
from harness.context import truncate_history
from harness.dialect import DialectFeed
from harness.messages import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
    to_wire_messages,
)
from harness.tools import MAX_RESULT_CHARS, TRUNCATED_SUFFIX, ToolError


class LoopError(Exception):
    """Loop-level failure: overflow, max steps, tool loop, abort. Exit code 2."""


# Events emitted to the front end synchronously as they happen:
#   ("content", str) | ("reasoning", str) | ("tool_start", ToolCall)
#   | ("tool_result", tool_call_id: str, content: str)
#   | ("step", int) | ("done", answer: str) | ("error", str)
AgentEvent = tuple[str, str | ToolCall | int] | tuple[str, str, str]

# Events the gateway yields on its stream:
#   ("content", str) | ("reasoning", str) | ("tool_call", ToolCall)
#   | ("done", finish_reason: str | None) | ("error", str)
StreamEvent = tuple[str, str | ToolCall | None]


def _from_wire(wire: dict[str, Any]) -> Message | None:
    """Convert one session wire dict back into a message object (or None).

    Assistant tool_calls may arrive either in OpenAI wire form
    ({"id", "type", "function": {"name", "arguments"}}) or in the compact
    persisted form ({"id", "name", "arguments"}); both are accepted. Unknown
    roles are skipped.
    """
    role = wire.get("role")
    if role == "user":
        return UserMessage(wire.get("content", ""))
    if role == "assistant":
        calls = None
        raw_calls = wire.get("tool_calls")
        if raw_calls:
            calls = []
            for tc in raw_calls:
                function = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                calls.append(
                    ToolCall(
                        tc.get("id", ""),
                        function.get("name", tc.get("name", "")),
                        function.get("arguments", tc.get("arguments", "")),
                    )
                )
        return AssistantMessage(wire.get("content", ""), wire.get("reasoning_content"), calls)
    if role == "tool":
        return ToolResultMessage(wire.get("tool_call_id", ""), wire.get("content", ""))
    return None


class AgentLoop:
    """One runnable task loop over a streaming gateway and a tool registry."""

    def __init__(
        self,
        gateway: Any,
        tools: Any,
        memory: Any,
        session_id: str,
        max_steps: int = 20,
        allow_dangerous: bool = False,
        resume: bool = False,
    ) -> None:
        self.gateway = gateway
        self.tools = tools
        self.memory = memory
        self.session_id = session_id
        self.max_steps = max_steps
        self.allow_dangerous = allow_dangerous
        self.resume = resume
        self._messages: list[Message] = []
        self._system: SystemMessage | None = None
        self._consecutive_failures = 0
        self._last_call_key: tuple[Any, Any] | None = None
        self._same_call_count = 0
        self._ran = False
        self.steps = 0

    # -- public --------------------------------------------------------------

    def run(self, task: str, emit: Callable[[AgentEvent], None] | None = None) -> str:
        """Run one task; returns the final answer, emitting events as it goes."""
        assert not self._ran, "run() may only be called once per AgentLoop instance"
        self._ran = True
        self.steps = 0

        system = SystemMessage(
            prompts.build_system_prompt(
                self.memory.load_digest(), prompts.build_project_context(self.tools.project_dir)
            )
        )
        self._system = system
        self._messages = [system]

        if self.resume:
            for wire in sessions.load_messages(self.session_id):
                message = _from_wire(wire)
                if message is not None:
                    self._messages.append(message)
        self._messages.append(UserMessage(task))

        sessions.append_event(
            self.session_id,
            {"type": "meta", "data": {"kind": "start", "model": getattr(self.gateway, "model_id", None)}},
        )
        sessions.append_event(self.session_id, {"type": "user", "data": {"content": task}})

        try:
            answer = self._step_loop(emit)
        except LoopError as exc:
            if emit is not None:
                emit(("error", str(exc)))
            self.memory.record_session_summary(task, f"error: {exc}")
            raise
        if emit is not None:
            emit(("done", answer))
        self.memory.record_session_summary(task, "ok")
        return answer

    # -- turn machinery ------------------------------------------------------

    def _step_loop(self, emit: Callable[[AgentEvent], None] | None) -> str:
        """Drive one full turn per step until an answer is produced."""
        for _ in range(self.max_steps):
            answer = self._one_step(emit)
            if answer is not None:
                return answer
        raise LoopError("max steps reached")

    def _one_step(self, emit: Callable[[AgentEvent], None] | None) -> str | None:
        """One full turn: stream, heal, resolve, persist, execute.

        Returns the final answer string, or None when tool calls were executed
        (the loop should continue with another step). An overflow retry re-runs
        the turn inside this step and must not count as a new step.
        """
        self.steps += 1
        if emit is not None:
            emit(("step", self.steps))
        retried = False
        while True:  # overflow retry re-runs the turn without consuming a step
            content_parts: list[str] = []
            reasoning_parts: list[str] = []
            healed_calls: list[ToolCall] = []
            structured_calls: list[ToolCall] = []
            finish_reason: str | None = None

            feed = DialectFeed()
            for kind, payload in self.gateway.stream(
                to_wire_messages(self._messages), self.tools.schemas()
            ):
                if kind == "content":
                    for event in feed.feed(payload):
                        self._route(event, content_parts, reasoning_parts, healed_calls, emit)
                elif kind == "reasoning":
                    reasoning_parts.append(payload)
                    if emit is not None:
                        emit(("reasoning", payload))
                elif kind == "tool_call":
                    structured_calls.append(payload)
                elif kind == "error":
                    raise LoopError(f"gateway error: {payload}")
                elif kind == "done":
                    finish_reason = payload
                    break  # the generator ends anyway after "done"

            for event in feed.flush():
                self._route(event, content_parts, reasoning_parts, healed_calls, emit)

            calls = structured_calls if structured_calls else healed_calls
            if finish_reason == "length" and not content_parts and not calls:
                if retried:
                    raise LoopError("context overflow: model hit max output with empty turn")
                retried = True
                self._messages = truncate_history(
                    self._messages,
                    self._system,
                    (CONTEXT_WINDOW - MAX_OUTPUT_TOKENS) // 2,
                )
                continue
            break

        # R7: persist the assistant turn before executing anything (the wire
        # order is assistant-with-tool_calls, then tool results).
        content = "".join(content_parts)
        reasoning = "".join(reasoning_parts)
        self._messages.append(AssistantMessage(content, reasoning or None, calls or None))
        sessions.append_event(
            self.session_id,
            {
                "type": "assistant",
                "data": {
                    "content": content,
                    "reasoning_content": reasoning or None,
                    "tool_calls": [
                        {"id": call.id, "name": call.name, "arguments": call.arguments}
                        for call in calls
                    ]
                    or None,
                },
            },
        )

        if not calls:
            return content

        for call in calls:
            self._execute_one(call, emit)
        return None

    @staticmethod
    def _route(
        event: tuple[str, str | ToolCall],
        content_parts: list[str],
        reasoning_parts: list[str],
        healed_calls: list[ToolCall],
        emit: Callable[[AgentEvent], None] | None,
    ) -> None:
        """Route one DialectFeed event into the turn accumulators."""
        kind = event[0]
        if kind == "text":
            content_parts.append(event[1])
            if emit is not None:
                emit(("content", event[1]))
        elif kind == "reasoning":
            reasoning_parts.append(event[1])
            if emit is not None:
                emit(("reasoning", event[1]))
        elif kind == "tool_call":
            healed_calls.append(event[1])

    def _execute_one(self, call: ToolCall, emit: Callable[[AgentEvent], None] | None) -> None:
        """Execute one tool call: emit events, persist, and update bookkeeping.

        Raises LoopError on 5 consecutive tool failures or on the 3rd
        consecutive identical (name, args) tuple.
        """
        if emit is not None:
            emit(("tool_start", call))

        result, args = self._run_tool(call)
        if len(result) > MAX_RESULT_CHARS:
            result = result[:MAX_RESULT_CHARS] + TRUNCATED_SUFFIX

        # Tool-loop detection: the same (name, args) tuple 3x in a row aborts.
        key = (call.name, args)
        if key == self._last_call_key:
            self._same_call_count += 1
        else:
            self._last_call_key = key
            self._same_call_count = 1
        if self._same_call_count >= 3:
            raise LoopError("tool loop detected")

        if emit is not None:
            emit(("tool_result", call.id, result))
        sessions.append_event(
            self.session_id,
            {"type": "tool_call", "data": {"id": call.id, "name": call.name, "arguments": call.arguments}},
        )
        sessions.append_event(
            self.session_id,
            {"type": "tool_result", "data": {"tool_call_id": call.id, "content": result}},
        )
        self._messages.append(ToolResultMessage(call.id, result))

    def _run_tool(self, call: ToolCall) -> tuple[str, Any]:
        """Execute the call; returns (result, parsed args).

        ToolError and unparseable arguments are surfaced as result strings and
        count toward the consecutive-failure budget; 5 in a row raises
        LoopError. `args` is None when the arguments JSON failed to parse.
        """
        try:
            args = json.loads(call.arguments) if call.arguments else {}
            result = self.tools.execute(call.name, args)
        except ToolError as exc:
            result = str(exc)
            self._consecutive_failures += 1
        except (ValueError, TypeError) as exc:
            result = f"invalid tool arguments: {exc}"
            args = None
            self._consecutive_failures += 1
        else:
            self._consecutive_failures = 0
        if self._consecutive_failures >= 5:
            raise LoopError("5 consecutive tool failures")
        return result, args
