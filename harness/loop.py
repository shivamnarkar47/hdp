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

import concurrent.futures
import json
import subprocess
from typing import Any, Callable

from harness import prompts, sessions
from harness.config import CONTEXT_WINDOW, MAX_OUTPUT_TOKENS
from harness.context import estimate_tokens, truncate_history, wire_token_count
from harness.dialect import DialectFeed
from harness.gateway import GatewayError
from harness.memory import Memory
from harness.messages import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
    to_wire_messages,
)
from harness.tools import (
    MAX_RESULT_CHARS,
    TRUNCATED_SUFFIX,
    ToolError,
    ToolRegistry,
    resolve_relative,
)

# Prompt budget: the wire history must leave MAX_OUTPUT_TOKENS of headroom
# for the model's reply (CONTEXT_WINDOW - MAX_OUTPUT_TOKENS == 616_000).
PROMPT_BUDGET = CONTEXT_WINDOW - MAX_OUTPUT_TOKENS


class LoopError(Exception):
    """Loop-level failure: overflow, max steps, tool loop, abort. Exit code 2."""


# Events emitted to the front end synchronously as they happen:
#   ("content", str) | ("reasoning", str) | ("tool_start", ToolCall)
#   | ("tool_result", tool_call_id: str, content: str) | ("verify", str)
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
        structure: Any = None,
        enable_verify: bool = True,
        spawn_depth: int = 1,
        agent: dict | None = None,
    ) -> None:
        self.gateway = gateway
        self.tools = tools
        self.memory = memory
        self.session_id = session_id
        self.max_steps = max_steps
        self.allow_dangerous = allow_dangerous
        self.resume = resume
        self._structure = structure
        self.enable_verify = enable_verify
        # Active persona (a harness.agents dict {name, description}) injected
        # into the system prompt; None = no persona. Deliberately NOT inherited
        # by nested spawn_agent loops (see _spawn) — a spawned sub-task runs as
        # a plain agent, not as a second copy of the persona.
        self.agent = agent
        # Nested-agent nesting level: the top-level loop is depth 1, a spawned
        # loop is depth 2, and spawning is disabled at depth >= 2 (recursion
        # capped at two nested loops). See _spawn.
        self._spawn_depth = spawn_depth
        # Verify hook command from .kaal/hooks.json, read ONCE at run() start
        # (None = feature off: missing file, invalid JSON, empty array, or
        # enable_verify=False). Runs after any mutating batch; its output is
        # CONTENT for the model, never a loop abort.
        self._verify_cmd: list[str] | None = None
        self._messages: list[Message] = []
        # Incremental wire cache: self._wire is the byte-identical conversion
        # of self._messages (system coalesced once), self._wire_tokens its
        # token cost. New messages are converted and costed individually; the
        # cache is rebuilt only after truncation (rare).
        self._wire: list[dict[str, Any]] = []
        self._wire_tokens = 0
        self._system: SystemMessage | None = None
        self._consecutive_failures = 0
        self._last_call_key: tuple[Any, Any] | None = None
        self._same_call_count = 0
        self._ran = False
        self.steps = 0
        self.usage = {"input_tokens": 0, "output_tokens": 0}
        # Finding 6 wiring: cli.py builds the ToolRegistry BEFORE the loop, so
        # the registry cannot know this loop's nested-run implementation at
        # construction. The loop injects it here, after the registry exists.
        # Registries without the setter (stub registries in tests) simply get
        # no spawn_agent support.
        setter = getattr(self.tools, "set_spawn_handler", None)
        if setter is not None:
            setter(self._spawn)

    # -- public --------------------------------------------------------------

    def run(self, task: str, emit: Callable[[AgentEvent], None] | None = None) -> str:
        """Run one task; returns the final answer, emitting events as it goes."""
        assert not self._ran, "run() may only be called once per AgentLoop instance"
        self._ran = True
        self.steps = 0
        self.usage = {"input_tokens": 0, "output_tokens": 0}

        self._load_verify_cmd()

        if self._structure is None:
            from harness.structure import StructureManager

            self._structure = StructureManager(self.tools.project_dir)
        try:
            self._structure.ensure()  # cache is best-effort; never break the turn
        except OSError:
            pass

        system = SystemMessage(
            prompts.build_system_prompt(
                self.memory.load_digest(),
                prompts.build_project_context(self.tools.project_dir),
                self.agent,
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
        # Wire built once per run: system (coalesced once) + resumed history +
        # task. Kept incrementally up to date from here on; rebuilt only when
        # truncate_history drops turns.
        self._rebuild_wire()

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

            # Preemptive truncation: if the full wire history would exceed the
            # prompt budget, drop old turns BEFORE streaming — otherwise a full
            # max-token turn is burned only to overflow on finish_reason ==
            # "length" and truncate retroactively. The budget check is O(1)
            # against the incremental wire cache; the wire is rebuilt from
            # scratch only when truncation actually drops turns (rare).
            if self._wire_tokens > PROMPT_BUDGET:
                self._messages = truncate_history(self._messages, self._system, PROMPT_BUDGET)
                self._rebuild_wire()
            self.usage["input_tokens"] += self._wire_tokens

            feed = DialectFeed()
            # A shallow list copy keeps the streamed snapshot immutable per
            # call (the wire dicts themselves are the cached ones — zero
            # re-serialization) so recorded call history stays accurate.
            for kind, payload in self.gateway.stream(list(self._wire), self.tools.schemas()):
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
                # The wire cache must reflect the truncated history BEFORE the
                # retry stream call.
                self._rebuild_wire()
                continue
            break

        # Per-turn session events are collected and flushed in ONE batch; the
        # batch order (assistant-with-tool_calls first, then tool_call +
        # tool_result per call) is the persisted order, matching the wire
        # order. The meta/user start events stay as individual appends in
        # run().
        content = "".join(content_parts)
        reasoning = "".join(reasoning_parts)
        self.usage["output_tokens"] += estimate_tokens(content + reasoning)
        self._append_wire(AssistantMessage(content, reasoning or None, calls or None))
        session_events: list[dict[str, Any]] = [
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
            }
        ]

        if not calls:
            sessions.append_events(self.session_id, session_events)
            return content

        self.tools.begin_batch(
            [call.name for call in calls],
            getattr(self._structure, "last_signature", None),
        )
        self._execute_many(calls, emit, session_events)
        sessions.append_events(self.session_id, session_events)
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

    # -- wire cache ----------------------------------------------------------

    def _rebuild_wire(self) -> None:
        """Rebuild the incremental wire cache from scratch (after truncation)."""
        self._wire = to_wire_messages(self._messages)
        self._wire_tokens = wire_token_count(self._wire)

    def _append_wire(self, message: Message) -> None:
        """Append one message to the history AND the incremental wire cache.

        Only the new message is converted to wire and token-costed; the system
        coalescing happened once in the first build.
        """
        self._messages.append(message)
        wire = message.to_wire()
        self._wire.append(wire)
        self._wire_tokens += estimate_tokens(json.dumps(wire, ensure_ascii=False))

    # -- tool execution ------------------------------------------------------

    def _run_one(self, call: ToolCall) -> tuple[str, Any, bool]:
        """Execute one tool call in isolation; worker-safe, no side effects.

        Returns (result, parsed_args, failed). ToolError and unparseable
        arguments are surfaced as result strings with failed=True; `args` is
        None when the arguments JSON failed to parse. No emit/persist/
        bookkeeping happens here — that is _record_result's job, on the main
        thread, in original call order.
        """
        try:
            args = json.loads(call.arguments) if call.arguments else {}
            result = self.tools.execute(call.name, args)
        except ToolError as exc:
            return str(exc), args, True
        except (ValueError, TypeError) as exc:
            return f"invalid tool arguments: {exc}", None, True
        return result, args, False

    def _record_result(
        self,
        call: ToolCall,
        result: str,
        args: Any,
        failed: bool,
        emit: Callable[[AgentEvent], None] | None,
        session_events: list[dict[str, Any]],
    ) -> None:
        """Record one tool outcome on the main thread, in original call order.

        Applies the defensive cap, emits ("tool_result", ...), persists the
        tool_call + tool_result events into the per-turn batch, appends the
        ToolResultMessage to the wire cache, and maintains ALL consecutive-
        failure counting and tool-loop detection. Raises LoopError at 5
        consecutive tool failures or on the 3rd consecutive identical
        (name, args) tuple.
        """
        if len(result) > MAX_RESULT_CHARS:
            result = result[:MAX_RESULT_CHARS] + TRUNCATED_SUFFIX

        # Consecutive-failure budget: ToolError / unparseable args count; the
        # 5th in a row aborts BEFORE any emit/persist for this call (as before
        # the refactor).
        if failed:
            self._consecutive_failures += 1
            if self._consecutive_failures >= 5:
                raise LoopError("5 consecutive tool failures")
        else:
            self._consecutive_failures = 0

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
        session_events.append(
            {"type": "tool_call", "data": {"id": call.id, "name": call.name, "arguments": call.arguments}}
        )
        session_events.append(
            {"type": "tool_result", "data": {"tool_call_id": call.id, "content": result}}
        )
        self._append_wire(ToolResultMessage(call.id, result))

    def _execute_many(
        self,
        calls: list[ToolCall],
        emit: Callable[[AgentEvent], None] | None,
        session_events: list[dict[str, Any]],
    ) -> None:
        """Execute a tool-call batch; results recorded in original call order.

        All-read batches (read/grep/glob) of more than one call run
        concurrently in a small thread pool; any batch containing a mutator
        (write/edit/bash/memory_append) runs fully serially in call order —
        zero file-race risk and deterministic bash side effects. tool_start
        events are emitted first (in order) for the parallel path; on both
        paths _record_result runs on the main thread in call order. The
        structure refresh runs only when the batch mutated the tree.
        """
        parallel = len(calls) > 1 and all(call.name in ("read", "grep", "glob") for call in calls)
        if parallel:
            if emit is not None:
                for call in calls:
                    emit(("tool_start", call))
            outcomes = [None] * len(calls)  # each slot filled by index -> call order
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(calls))) as pool:
                futures = {pool.submit(self._run_one, call): index for index, call in enumerate(calls)}
                for future in concurrent.futures.as_completed(futures):
                    outcomes[futures[future]] = future.result()
            for call, outcome in zip(calls, outcomes):
                self._record_result(call, *outcome, emit, session_events)
        else:
            for call in calls:
                if emit is not None:
                    emit(("tool_start", call))
                self._record_result(call, *self._run_one(call), emit, session_events)

        # Only write/edit/bash mutate the tree; read/grep/glob never do, so
        # skip the full-tree signature scan for them (measured ~30us/entry,
        # up to ~0.5s at the 20k-entry cap).
        mutated = any(call.name in ("write", "edit", "bash") for call in calls)
        if mutated:
            try:
                self._structure.refresh()
            except OSError:
                pass
        self.tools.end_batch(mutated=mutated)
        if mutated:
            self._run_verify(emit, session_events)

    # -- verify hooks -------------------------------------------------------

    def _load_verify_cmd(self) -> None:
        """Read .kaal/hooks.json once into self._verify_cmd (None = off).

        Explicit config only — no auto-derivation. Missing file, invalid
        JSON, a non-list or empty ``verify`` value, or enable_verify=False
        all turn the feature off.
        """
        self._verify_cmd = None
        if not self.enable_verify:
            return
        hooks_path = self.tools.project_dir / ".kaal" / "hooks.json"
        try:
            raw = json.loads(hooks_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return  # missing file / invalid JSON -> feature off
        cmd = raw.get("verify")
        if (
            isinstance(cmd, list)
            and cmd
            and all(isinstance(part, str) for part in cmd)
        ):
            self._verify_cmd = list(cmd)

    def _run_verify(
        self,
        emit: Callable[[AgentEvent], None] | None,
        session_events: list[dict[str, Any]],
    ) -> None:
        """Run the configured verify command after a mutating batch.

        The output is CONTENT for the model, never a loop abort: it becomes a
        user message on the wire (the last user message, which truncation
        protects — Finding 8), persists as a user event, and emits a new
        ("verify", str) AgentEvent kind. Synchronous (v1); 30s worst case.
        """
        if not self._verify_cmd:
            return
        try:
            proc = subprocess.run(
                self._verify_cmd,
                cwd=str(self.tools.project_dir),
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            content = "[verify] timed out after 30s"
        except OSError as exc:
            content = f"[verify] failed to run: {exc}"
        else:
            content = (proc.stdout or "") + (proc.stderr or "")
            if len(content) > MAX_RESULT_CHARS:
                content = content[:MAX_RESULT_CHARS] + TRUNCATED_SUFFIX
        if content.startswith("[verify]"):
            message = content
        else:
            message = f"[verify]\n{content}"
        if emit is not None:
            emit(("verify", content))
        session_events.append({"type": "user", "data": {"content": message}})
        self._append_wire(UserMessage(message))

    # -- nested agents (spawn_agent) ----------------------------------------

    def _spawn(self, task: str, dir: str | None, max_steps: int, timeout: int) -> str:
        """Run a nested AgentLoop on a sub-task; return its JSON summary.

        This is the registry's injected spawn handler (wired in __init__).
        Guardrails FIRST: recursion is capped — a loop at spawn_depth >= 2
        returns the limit string without creating anything. Otherwise the
        nested run gets a FRESH ToolRegistry (no tool cache — Finding 6),
        fresh Memory under the resolved dir, its own session id (visible in
        ``kaal sessions list``), allow_dangerous=False, enable_verify=False,
        max_steps <= 5, and a wall-clock timeout.

        The nested loop ALWAYS runs on its own worker thread (a fresh
        ThreadPoolExecutor with one worker): the parent may itself be running
        on a --batch pool thread, and for serial batches a synchronous nested
        run would block the main loop thread. The worker thread lets
        future.result(timeout=...) enforce the wall timeout and keeps the
        parent responsive. emit=None: no events bubble up — the JSON summary
        is the result.
        """
        if self._spawn_depth >= 2:
            return "spawn_agent: recursion limit reached"
        if dir is not None:
            try:
                target = resolve_relative(dir, self.tools.project_dir)
            except ToolError as exc:
                return f"spawn_agent: {exc}"
            if not target.is_dir():
                return f"spawn_agent: not a directory: {dir}"
        else:
            target = self.tools.project_dir
        session_id = sessions.new_session_id()
        memory = Memory(target / ".agent-memory")
        nested_tools = ToolRegistry(
            memory=memory,
            project_dir=target,
            allow_dangerous=False,
            cache=None,  # no tool cache in nested runs (Finding 6)
        )
        nested = AgentLoop(
            self.gateway,  # shared Gateway: stateless; keep-alive sockets are per-thread
            nested_tools,
            memory,
            session_id,
            max_steps=min(max_steps, 5),
            allow_dangerous=False,
            enable_verify=False,
            spawn_depth=self._spawn_depth + 1,
            # no agent=: the persona is NOT inherited by nested loops — a
            # spawned sub-task runs as a plain agent, never as a second copy
            # of the active persona (see __init__).
        )
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(nested.run, task)  # emit=None: nothing bubbles up
        try:
            answer = future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            # The orphaned thread keeps running (bounded by its own max_steps
            # and gateway timeouts); never block the parent on it.
            executor.shutdown(wait=False)
            return f"spawn_agent: timed out after {timeout}s"
        except LoopError as exc:
            executor.shutdown(wait=False)
            return f"spawn_agent: {exc}"
        except GatewayError as exc:
            executor.shutdown(wait=False)
            return f"spawn_agent: {exc}"
        executor.shutdown(wait=False)
        return json.dumps(
            {
                "answer": answer[:50000],
                "steps": nested.steps,
                "usage": nested.usage,
                "session_id": session_id,
            },
            ensure_ascii=False,
        )
