"""Textual split-pane TUI — the default hdp surface.

A polished agent client over :class:`harness.loop.AgentLoop`: a conversation
pane (user blocks, streamed-markdown assistant turns, reasoning, collapsed
tool boxes) on the left, a fixed sidebar (Trace / Memory / Sessions tabs) on
the right, a multi-line prompt, and a one-line live status bar.

The heavy loop runs in a worker *thread* so streaming never blocks the UI.
Thread widgets are untouchable, so the thread's emit callback marshals every
event back to the main thread via ``App.call_from_thread``; the main-thread
handler is the only place that writes to widgets.

Streaming markdown is throttled: Textual's ``Markdown`` re-renders its whole
document on every update, so content chunks accumulate in a pending buffer and
are flushed at most once per ~100 ms (and always on turn end). A single turn's
markdown is capped; past the cap the overflow appends as a plain text block.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from textual import events, on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    Markdown,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)

from harness import config, sessions
from harness.art import SEA_LION
from harness.gateway import Gateway
from harness.loop import AgentEvent, AgentLoop, ToolCall
from harness.memory import SECTIONS, Memory
from harness.structure import StructureManager
from harness.tools import ToolRegistry

# Streaming markdown is re-rendered whole-document per update; flush at most
# every ~100 ms and cap one turn's markdown so a pathological turn can't jank
# the UI or blow up the widget.
MD_FLUSH_SECONDS = 0.1
MD_CHAR_CAP = 200_000

# Result previews shown in collapsed tool boxes / the trace tab.
PREVIEW_CHARS = 200

# Content starting with any of these counts as a failed tool result.
_ERROR_STARTS = (
    "blocked",
    "error",
    "failed",
    "invalid",
    "missing",
    "no such",
    "not found",
    "unknown tool",
    "denied",
)

# Slash commands offered by the suggestion popup (order is the popup order).
COMMANDS = [
    "/help",
    "/new",
    "/resume",
    "/sessions",
    "/memory",
    "/model",
    "/verbose",
    "/connect",
    "/quit",
    "/structure",
]

# Braille spinner frames for the "thinking" indicator.
THINK_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class TurnCancelled(Exception):
    """Raised by the emit callback when the user cancels the active turn."""


def _call_name_args(call: dict) -> tuple[str, str]:
    """Extract (name, arguments) from a persisted or OpenAI-wire tool_call dict."""
    function = call.get("function")
    if isinstance(function, dict):
        return function.get("name", ""), function.get("arguments", "")
    return call.get("name", ""), call.get("arguments", "")


class PromptSubmitted(Message):
    """The prompt was submitted (Enter pressed with non-empty content)."""

    def __init__(self, text: str) -> None:
        self.text = text
        super().__init__()


class PromptInput(TextArea):
    """Multi-line prompt: Enter submits, Shift+Enter inserts a newline.

    TextArea's default up/down move the caret, so history recall lives on
    Ctrl+P (previous) / Ctrl+N (next), matching readline. Ctrl+C is rebound
    from TextArea's copy to cancel the active turn (the terminal's
    Ctrl+Shift+C still copies).
    """

    BINDINGS = [
        # priority=True so these are checked in the App's priority pass before
        # TextArea's own key handling can consume them.
        Binding("enter", "submit", "Send", priority=True, show=False),
        Binding("up", "suggest_up", "Suggestion up", priority=True, show=False),
        Binding("down", "suggest_down", "Suggestion down", priority=True, show=False),
        Binding("tab", "suggest_tab", "Complete suggestion", priority=True, show=False),
        Binding("escape", "suggest_escape", "Close suggestions", priority=True, show=False),
        Binding("shift+enter", "newline", "New line", show=False),
        Binding("ctrl+p", "history_prev", "History previous", show=False),
        Binding("ctrl+n", "history_next", "History next", show=False),
        Binding("ctrl+c", "cancel", "Cancel turn", show=False),
    ]

    def action_submit(self) -> None:
        """Enter: submit normally, or complete a suggestion when the popup is open.

        Bash-style: with the popup open, Enter completes to the highlighted row
        WITHOUT closing the popup — unless the typed text already is a full
        command, or the completion changes nothing, in which case it submits.
        """
        app = self.app
        if app._suggestions_visible:
            current = self.text
            if current.strip() in app._suggestion_rows:
                self._submit_text(current)
                return
            completed = app._complete_suggestion_text()
            if completed is not None:
                if completed == current:
                    self._submit_text(current)
                else:
                    self.text = completed
                    self.move_cursor(self.document.end)
            return
        self._submit_text(self.text)

    def _submit_text(self, text: str) -> None:
        text = text.rstrip("\n")
        self.text = ""
        if text.strip():
            self.post_message(PromptSubmitted(text))

    def action_newline(self) -> None:
        self.insert("\n")

    def action_suggest_up(self) -> None:
        app = self.app
        if app._suggestions_visible and app._suggestion_rows:
            n = len(app._suggestion_rows)
            app._suggest_index = (app._suggest_index - 1) % n
            app._render_suggestions()
        else:
            super().action_cursor_up()

    def action_suggest_down(self) -> None:
        app = self.app
        if app._suggestions_visible and app._suggestion_rows:
            n = len(app._suggestion_rows)
            app._suggest_index = (app._suggest_index + 1) % n
            app._render_suggestions()
        else:
            super().action_cursor_down()

    def action_suggest_tab(self) -> None:
        """Tab: complete to the highlighted row and close the popup."""
        app = self.app
        if not app._suggestions_visible:
            # Popup closed: keep Textual's default tab behavior (focus next).
            self.screen.focus_next()
            return
        completed = app._complete_suggestion_text()
        if completed is not None and completed != self.text:
            # Swallow the completion's own Changed so it can't re-open the popup.
            app._suppress_reopen = True
            self.text = completed
            self.move_cursor(self.document.end)
        app._close_suggestions()

    def action_suggest_escape(self) -> None:
        app = self.app
        if app._suggestions_visible:
            app._close_suggestions()

    def action_history_prev(self) -> None:
        app = self.app
        history: list[str] = app.prompt_history
        if not history:
            return
        if app._history_index is None:
            app._history_index = len(history) - 1
        elif app._history_index > 0:
            app._history_index -= 1
        else:
            return  # already at the oldest entry
        self.text = history[app._history_index]
        self.move_cursor(self.document.end)

    def action_history_next(self) -> None:
        app = self.app
        if app._history_index is None:
            return
        app._history_index += 1
        if app._history_index >= len(app.prompt_history):
            app._history_index = None
            self.text = ""
        else:
            self.text = app.prompt_history[app._history_index]
        self.move_cursor(self.document.end)

    def action_cancel(self) -> None:
        self.app.action_cancel_turn()


class ConversationScroll(VerticalScroll):
    """Conversation pane; reports user scrolling so auto-follow can pause.

    The app only auto-scrolls on new content while the user is following the
    bottom; scrolling up (wheel or keys) stops the follow until the user
    scrolls back to the bottom or presses Ctrl+L.
    """

    def _on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        self.app.set_follow_scroll(False)
        super()._on_mouse_scroll_up(event)

    def _on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        self.app.maybe_resume_follow()
        super()._on_mouse_scroll_down(event)

    def action_scroll_up(self) -> None:
        self.app.set_follow_scroll(False)
        super().action_scroll_up()

    def action_scroll_home(self) -> None:
        self.app.set_follow_scroll(False)
        super().action_scroll_home()

    def action_scroll_down(self) -> None:
        self.app.maybe_resume_follow()
        super().action_scroll_down()

    def action_scroll_end(self) -> None:
        self.app.maybe_resume_follow()
        super().action_scroll_end()


class ConnectScreen(ModalScreen[str | None]):
    """Modal popup for entering the OpenCode Zen/Go API key.

    Dismisses with the entered key on Save, or None on Cancel/Esc.
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("enter", "save", "Save"),
    ]

    def __init__(self, existing: str | None = None) -> None:
        super().__init__()
        self._existing = existing

    def compose(self) -> ComposeResult:
        with Vertical(id="connect-box"):
            yield Static("OpenCode Zen/Go API key", classes="connect-title")
            if self._existing:
                masked = "sk-••••" + self._existing[-4:]
                yield Static(f"(key already set: {masked})", classes="connect-hint")
            yield Input(id="key-input", placeholder="sk-…", password=True)
            with Horizontal(id="connect-buttons"):
                yield Button("Save", variant="primary", id="save-btn")
                yield Button("Cancel", id="cancel-btn")

    def on_mount(self) -> None:
        self.query_one("#key-input", Input).focus()

    @on(Input.Submitted)
    def _on_key_submitted(self, event: Input.Submitted) -> None:
        self._save()

    @on(Button.Pressed)
    def _on_button(self, event: Button.Pressed) -> None:
        if event.button.id == "save-btn":
            self._save()
        elif event.button.id == "cancel-btn":
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_save(self) -> None:
        self._save()

    def _save(self) -> None:
        key = self.query_one("#key-input", Input).value.strip()
        if not key:
            self.query_one("#key-input", Input).focus()  # guard empty input
            return
        self.dismiss(key)


class SessionsScreen(ModalScreen[str | None]):
    """Modal session switcher: Enter resumes the highlighted session.

    Dismisses with the chosen session id, or None on Cancel/Esc.
    """

    BINDINGS = [
        Binding("escape", "cancel", "Close"),
    ]

    def __init__(self, entries: list[dict]) -> None:
        super().__init__()
        self._entries = entries

    def compose(self) -> ComposeResult:
        with Vertical(id="sessions-box"):
            yield Static("Sessions", classes="connect-title")
            if self._entries:
                yield ListView(
                    *[
                        ListItem(Label(f"{e['id']}  {(e['prompt'] or '')[:50]}"))
                        for e in self._entries
                    ],
                    id="session-list",
                )
            else:
                yield Static("(no sessions yet)", classes="connect-hint")
            yield Static("Enter resume · Esc close", classes="connect-hint")

    def on_mount(self) -> None:
        if self._entries:
            self.query_one("#session-list", ListView).focus()

    @on(ListView.Selected)
    def _on_selected(self, event: ListView.Selected) -> None:
        index = self.query_one("#session-list", ListView).index
        entry = self._entries[index]
        self.dismiss(entry["id"])

    def action_cancel(self) -> None:
        self.dismiss(None)


class HarnessTui(App):
    """Split-pane chat TUI for hdp."""

    TITLE = "hdp"

    CSS = """
    Screen {
        background: $surface;
    }

    #conversation {
        width: 2fr;
        height: 1fr;
        background: $background;
        padding: 0 1 0 1;
    }

    #sidebar {
        width: 34;
        height: 1fr;
        background: $panel;
        border-left: solid $panel-lighten-1;
    }

    #sidebar TabbedContent {
        height: 1fr;
    }

    #sidebar TabPane {
        height: 1fr;
    }

    #bottom {
        dock: bottom;
        height: auto;
    }

    #prompt {
        height: 3;
        margin: 0 1 0 1;
    }

    #status {
        height: 1;
        color: $text-muted;
    }

    .welcome {
        color: $text-muted;
        margin: 1 0 1 0;
    }

    .user-block {
        border: round $accent;
        padding: 0 1;
        margin: 1 0 1 0;
    }

    .assistant-label {
        color: $accent;
        text-style: bold;
        margin-top: 1;
    }

    .assistant-md {
        margin: 0 0 1 0;
    }

    .reasoning {
        color: $text-muted;
        text-style: dim;
        margin: 0 0 1 0;
    }

    .tool-box {
        border: round $surface-lighten-1;
        padding: 0 1;
        margin: 0 0 1 0;
        color: $text-muted;
    }

    .turn-raw {
        margin: 0 0 1 0;
    }

    .error-box {
        border: round $error;
        padding: 0 1;
        margin: 1 0 1 0;
    }

    .notice {
        color: $text-muted;
        margin: 0 0 1 0;
    }

    .trace-line {
        color: $text-muted;
        padding: 0 0 0 1;
    }

    .sidebar-empty {
        color: $text-muted;
        padding: 1;
    }

    .session-btn {
        width: 100%;
        max-width: 100%;
        margin: 0 0 0 0;
    }

    #suggestions {
        display: none;
        height: auto;
        max-height: 10;
        border: round $accent;
        background: $panel;
        margin: 0 1 0 1;
        padding: 0 1;
        overflow: hidden auto;
    }

    .suggest-row {
        color: $text-muted;
    }

    .suggest-row.highlight {
        color: $accent;
        text-style: bold;
    }

    .suggest-more {
        color: $text-muted;
        text-style: dim;
    }

    .thinking {
        color: $text-muted;
        text-style: dim;
        margin: 0 0 1 0;
    }

    ConnectScreen,
    SessionsScreen {
        align: center middle;
    }

    #connect-box,
    #sessions-box {
        width: 52;
        height: auto;
        max-height: 80%;
        border: round $accent;
        background: $panel;
        padding: 1 2;
    }

    .connect-title {
        text-style: bold;
        margin-bottom: 1;
    }

    .connect-hint {
        color: $text-muted;
        margin-bottom: 1;
    }

    #connect-box Input {
        margin-bottom: 1;
    }

    #connect-buttons {
        height: auto;
    }

    #connect-buttons Button {
        width: 1fr;
    }

    #session-list {
        height: auto;
        max-height: 50%;
        margin-bottom: 1;
    }

    .sea-lion {
        color: $accent;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "cancel_turn", "Cancel"),
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+l", "jump_to_bottom", "Jump to bottom", show=False),
    ]

    def __init__(
        self,
        gateway: Any = None,
        memory_root: Path | None = None,
        project_dir: Path | None = None,
        model_id: str | None = None,
        max_steps: int = 20,
        allow_dangerous: bool = False,
    ) -> None:
        super().__init__()
        if gateway is None:
            # config.get_api_key() may SystemExit(1) when no key is configured.
            gateway = Gateway(config.BASE_URL, config.get_api_key(), model_id or config.MODEL_ID)
        self.gateway = gateway
        self.model_id: str = getattr(gateway, "model_id", None) or model_id or config.MODEL_ID
        self.project_dir = Path(project_dir or Path.cwd())
        self.max_steps = max_steps
        self.allow_dangerous = allow_dangerous

        self.memory = Memory(memory_root or self.project_dir / ".agent-memory")
        self.tools = ToolRegistry(
            memory=self.memory,
            project_dir=self.project_dir,
            allow_dangerous=allow_dangerous,
        )
        self.structure = StructureManager(self.project_dir)

        self.session_id = sessions.new_session_id()
        self.sub_title = f"{self.model_id} · {self.session_id}"
        self.resume_next = False
        self._cancel_turn = False
        self.turn_active = False
        self.verbose = False
        self.prompt_history: list[str] = []
        self._history_index: int | None = None
        # Plain-text mirror of everything rendered, in order (used by tests).
        self.transcript: list[str] = []
        # NB: do NOT use the name `_loop` here — App._loop is Textual's
        # internal asyncio event loop that call_from_thread() relies on.
        self._agent_loop: AgentLoop | None = None
        self._current_task: str | None = None

        self._prompt_input: PromptInput | None = None
        self._conversation: ConversationScroll | None = None
        self._trace: VerticalScroll | None = None
        self._tool_count = len(self.tools.schemas())
        self._steps = 0

        # Slash-command suggestion popup state.
        self._suggestions_visible = False
        self._suggestion_rows: list[str] = []
        self._suggest_index = 0
        self._suggest_mode: str | None = None
        self._suggest_more = False
        self._suppress_reopen = False
        self._suggestions: Vertical | None = None
        # Thinking-indicator state (transient, never mirrored to transcript).
        self._thinking_visible = False
        self._thinking: Static | None = None
        self._thinking_frame = 0
        self._thinking_timer: Any = None

        # Auto-follow state for the conversation pane.
        self._follow = True
        # Per-turn streaming state.
        self._turn_md: Markdown | None = None
        self._turn_raw: Static | None = None
        self._turn_raw_text = ""
        self._turn_md_text = ""
        self._md_over_cap = False
        self._md_pending: list[str] = []
        self._md_timer: Any = None
        self._reasoning: Static | None = None
        self._reasoning_text = ""
        # call_id -> (conversation box, trace line, name, arguments)
        self._tool_boxes: dict[str, tuple[Static, Static, str, str]] = {}

    # -- widgets ------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal(id="main"):
            with ConversationScroll(id="conversation"):
                pass
            with Vertical(id="sidebar"):
                with TabbedContent(initial="pane-trace"):
                    with TabPane("Trace", id="pane-trace"):
                        yield VerticalScroll(id="trace")
                    with TabPane("Memory", id="pane-memory"):
                        with VerticalScroll(id="memory-pane"):
                            yield Static(id="memory-view")
                    with TabPane("Sessions", id="pane-sessions"):
                        with VerticalScroll(id="sessions-view"):
                            pass
        with Vertical(id="bottom"):
            yield Vertical(id="suggestions")
            yield PromptInput(
                id="prompt",
                placeholder="Ask hdp… (/help)",
                soft_wrap=True,
            )
            yield Static(id="status")

    def on_mount(self) -> None:
        self._conversation = self.query_one("#conversation", ConversationScroll)
        self._trace = self.query_one("#trace", VerticalScroll)
        self._prompt_input = self.query_one("#prompt", PromptInput)
        self._suggestions = self.query_one("#suggestions", Vertical)
        self._render_home()
        self._structure_notice()
        self._refresh_memory()
        self._refresh_sessions()
        self._render_status()
        self._prompt_input.focus()

    # -- input --------------------------------------------------------------

    @on(PromptSubmitted)
    def _on_prompt_submitted(self, event: PromptSubmitted) -> None:
        if self.turn_active:
            self._write_line("(busy — Ctrl+C cancels the current turn)", classes="notice")
            return
        text = event.text.strip()
        if not text:
            return
        self.prompt_history.append(text)
        self._history_index = None
        if text.startswith("/"):
            self._run_command(text)
        else:
            self._start_turn(text)

    # -- slash-command suggestions -------------------------------------------

    @on(TextArea.Changed)
    def _on_prompt_changed(self, event: TextArea.Changed) -> None:
        if self._suppress_reopen:
            # The change came from a tab-completion; keep the popup closed.
            self._suppress_reopen = False
            return
        self._update_suggestions()

    def _update_suggestions(self) -> None:
        """Recompute the popup rows from the current prompt text.

        Typed text starting with ``/`` (no newline) filters the command list by
        prefix; once the text is ``/resume <arg>`` it filters session ids
        (newest first) by the arg prefix instead.
        """
        if self._prompt_input is None:
            return
        text = self._prompt_input.text
        rows: list[str] = []
        mode: str | None = None
        self._suggest_more = False
        if text.startswith("/") and "\n" not in text:
            if text.startswith("/resume "):
                mode = "session"
                arg = text[len("/resume "):]
                entries = sorted(
                    sessions.list_sessions(), key=lambda e: e["id"], reverse=True
                )
                rows = [e["id"] for e in entries if e["id"].startswith(arg)]
            else:
                mode = "command"
                lowered = text.lower()
                rows = [c for c in COMMANDS if c.lower().startswith(lowered)]
        if len(rows) > 8:
            self._suggest_more = True
            rows = rows[:8]
        self._suggest_mode = mode
        self._suggestion_rows = rows
        self._suggest_index = 0
        self._render_suggestions()

    def _render_suggestions(self) -> None:
        """(Re)draw the popup rows with the current highlight; hide when empty."""
        visible = bool(self._suggestion_rows)
        self._suggestions_visible = visible
        if self._suggestions is None:
            return
        self._suggestions.display = visible
        if not visible:
            return
        if self._suggest_index >= len(self._suggestion_rows):
            self._suggest_index = 0
        self._suggestions.remove_children()
        for i, row in enumerate(self._suggestion_rows):
            cls = "suggest-row highlight" if i == self._suggest_index else "suggest-row"
            self._suggestions.mount(Static(row, classes=cls, markup=False))
        if self._suggest_more:
            self._suggestions.mount(Static("…", classes="suggest-more", markup=False))

    def _close_suggestions(self) -> None:
        self._suggestions_visible = False
        self._suggestion_rows = []
        if self._suggestions is not None:
            self._suggestions.display = False

    def _complete_suggestion_text(self) -> str | None:
        """The full prompt text produced by completing the highlighted row."""
        rows = self._suggestion_rows
        if not rows:
            return None
        idx = self._suggest_index % len(rows)
        row = rows[idx]
        if self._suggest_mode == "session":
            return "/resume " + row
        return row

    def _start_turn(self, task: str) -> None:
        """Render the user block, build a fresh one-shot loop, run it."""
        self._render_user_block(task)
        loop = AgentLoop(
            self.gateway,
            self.tools,
            self.memory,
            self.session_id,
            max_steps=self.max_steps,
            allow_dangerous=self.allow_dangerous,
            resume=self.resume_next,
            structure=self.structure,
        )
        self._agent_loop = loop
        self._current_task = task
        self._cancel_turn = False
        self.turn_active = True
        self._reset_turn_stream()
        self._show_thinking()
        self._prompt_input.disabled = True
        self._render_status()
        self.run_worker(self._thread_run, thread=True, group="agent")

    def _thread_run(self, task: str | None = None) -> None:
        """Worker thread body. Never touches widgets directly."""
        loop = self._agent_loop
        try:
            loop.run(
                task if task is not None else self._current_task,
                emit=lambda event, loop=loop: self._emit_cb(loop, event),
            )
        except TurnCancelled:
            pass  # canceled partial turn; the loop never persisted it
        except Exception as exc:  # LoopError, GatewayError, anything else
            self.call_from_thread(self._on_loop_error, str(exc))

    def _emit_cb(self, loop: AgentLoop, event: AgentEvent) -> None:
        """Called from the worker thread; marshals to the main thread.

        Also the cooperative-cancel point: raise TurnCancelled so a canceled
        (or superseded) turn's stream aborts instead of rendering.
        """
        if self._cancel_turn or self._agent_loop is not loop:
            raise TurnCancelled()
        self.call_from_thread(self._on_loop_event, event)

    # -- main-thread rendering ----------------------------------------------

    def _on_loop_event(self, event: AgentEvent) -> None:
        kind = event[0]
        if kind == "step":
            self._steps = int(event[1])
            if self._steps > 1:
                # A new model generation started: flush the previous one's
                # streaming markdown and open a fresh block, so tool boxes
                # interleave between generations instead of stacking after.
                self._flush_md()
                self._reset_turn_stream()
            self._show_thinking()
            self._render_status()
        elif kind == "content":
            self._ensure_assistant()
            self._append_content(event[1])  # type: ignore[arg-type]
        elif kind == "reasoning":
            self._show_thinking()
            if self.verbose:
                self._append_reasoning(event[1])  # type: ignore[arg-type]
        elif kind == "tool_start":
            self._on_tool_start(event[1])  # type: ignore[arg-type]
        elif kind == "tool_result":
            _, call_id, content = event
            self._on_tool_result(call_id, content)
        elif kind == "error":
            self._write_line(f"error: {event[1]}", classes="error-box")
            self._flush_md()
            self.turn_finished()
        elif kind == "done":
            self._flush_md()
            self.turn_finished()

    def _on_loop_error(self, message: str) -> None:
        # The ("error", ...) event already rendered this and finished the turn.
        if not self.turn_active:
            return
        self._write_line(f"error: {message}", classes="error-box")
        self.turn_finished()

    def _reset_turn_stream(self) -> None:
        self._turn_md = None
        self._turn_raw = None
        self._turn_raw_text = ""
        self._turn_md_text = ""
        self._md_over_cap = False
        self._md_pending = []
        self._md_timer = None
        self._reasoning = None
        self._reasoning_text = ""
        self._tool_boxes = {}

    def _ensure_assistant(self) -> None:
        """Mount the '▌ hdp' label + the turn's streaming Markdown widget."""
        if self._turn_md is not None:
            return
        self._hide_thinking()  # first content: thinking phase is over
        self.transcript.append("▌ hdp")
        self._conversation.mount(Static("▌ hdp", classes="assistant-label", markup=False))
        self._turn_md = Markdown("", classes="assistant-md")
        self._conversation.mount(self._turn_md)
        self._scroll_follow()

    def _append_content(self, chunk: str) -> None:
        self.transcript.append(chunk)
        self._md_pending.append(chunk)
        if self._md_timer is None:
            self._md_timer = self.set_timer(MD_FLUSH_SECONDS, self._flush_md)

    def _flush_md(self) -> None:
        """Throttled markdown re-render; also the turn-end flush."""
        if self._md_timer is not None:
            self._md_timer.stop()
        self._md_timer = None
        if not self._md_pending:
            return
        chunk = "".join(self._md_pending)
        self._md_pending.clear()
        if self._turn_md is None:
            return
        if not self._md_over_cap:
            room = MD_CHAR_CAP - len(self._turn_md_text)
            if len(chunk) <= room:
                self._turn_md_text += chunk
                self._turn_md.update(self._turn_md_text)
            else:
                self._turn_md_text += chunk[:room]
                self._turn_md.update(self._turn_md_text)
                self._md_over_cap = True
                self._turn_raw = Static(chunk[room:], classes="turn-raw", markup=False)
                self._turn_raw_text = chunk[room:]
                self._conversation.mount(self._turn_raw, after=self._turn_md)
        else:
            self._turn_raw_text += chunk
            self._turn_raw.update(self._turn_raw_text)
        self._scroll_follow()

    def _append_reasoning(self, chunk: str) -> None:
        self.transcript.append(f"💭 {chunk}")
        self._reasoning_text += chunk
        if self._reasoning is None:
            self._reasoning = Static("", classes="reasoning", markup=False)
            self._conversation.mount(self._reasoning)
        self._reasoning.update(f"💭 {self._reasoning_text}")
        self._scroll_follow()

    # -- thinking indicator ---------------------------------------------------

    def _show_thinking(self) -> None:
        """Mount the animated '💭 thinking' spinner (idempotent).

        Shown whenever the model is generating but no content has arrived yet;
        hidden again at the first content of a generation or at turn end. The
        spinner is transient — it is never mirrored to the transcript.
        """
        if self._thinking_visible:
            return
        self._thinking_visible = True
        self._thinking_frame = 0
        self._thinking = Static("", classes="thinking", markup=False)
        self._conversation.mount(self._thinking)
        self._thinking_timer = self.set_interval(0.08, self._tick_thinking)
        self._tick_thinking()
        self._scroll_follow()

    def _tick_thinking(self) -> None:
        if not self._thinking_visible or self._thinking is None:
            return
        frame = THINK_FRAMES[self._thinking_frame % len(THINK_FRAMES)]
        self._thinking_frame += 1
        self._thinking.update(f"💭 thinking {frame}")

    def _hide_thinking(self) -> None:
        if not self._thinking_visible:
            return
        self._thinking_visible = False
        if self._thinking_timer is not None:
            self._thinking_timer.stop()
            self._thinking_timer = None
        if self._thinking is not None:
            self._thinking.remove()
            self._thinking = None

    def _on_tool_start(self, call: ToolCall) -> None:
        self.transcript.append(f"⚙ {call.name}({call.arguments})")
        box = Static(f"⚙ {call.name}({call.arguments})", classes="tool-box", markup=False)
        self._conversation.mount(box)
        trace = Static(f"⚙ {call.name}({call.arguments})", classes="trace-line", markup=False)
        self._trace.mount(trace)
        self._trace.scroll_end(animate=False)
        self._tool_boxes[call.id] = (box, trace, call.name, call.arguments)
        self._scroll_follow()

    def _on_tool_result(self, call_id: str, content: str) -> None:
        preview = content[:PREVIEW_CHARS] + ("…" if len(content) > PREVIEW_CHARS else "")
        glyph = "✓" if self._looks_ok(content) else "⚠"
        self.transcript.append(f"  {glyph} {preview}")
        entry = self._tool_boxes.pop(call_id, None)
        if entry is not None:
            box, trace, name, args = entry
            box.update(f"⚙ {name}({args})\n{glyph} {preview}")
            trace.update(f"{glyph} ⚙ {name}({args}) → {preview}")

    @staticmethod
    def _looks_ok(content: str) -> bool:
        low = content.strip().lower()
        return not any(low.startswith(hint) for hint in _ERROR_STARTS)

    # -- conversation helpers -----------------------------------------------

    def _render_home(self) -> None:
        """Render the sea lion hero + welcome line (startup and `/new`).

        Clears the pane and live-stream state, then mounts the art as one
        Static and mirrors every art line plus the welcome line into the
        transcript, in order.
        """
        self._conversation.remove_children()
        self.transcript.clear()
        self._follow = True
        self._reset_turn_stream()
        self._hide_thinking()
        for line in SEA_LION.splitlines():
            self.transcript.append(line)
        self._conversation.mount(Static(SEA_LION, classes="sea-lion", markup=False))
        welcome = f"hdp — {self.model_id} agent. Ask a task, or /help for commands."
        self.transcript.append(welcome)
        self._conversation.mount(Static(welcome, classes="welcome", markup=False))
        # Hero art: start at the top so the sea lion's head is in view.
        self._conversation.scroll_to(y=0, animate=False)

    def _structure_notice(self) -> None:
        """Ensure the structure cache and write a one-line dim summary."""
        try:
            self.structure.ensure()
            text = self.structure.cache_path.read_text(encoding="utf-8")
        except OSError:
            self._write_line("structure: cache unavailable", classes="notice")
            return
        summary = ""
        for line in text.splitlines():
            if line.startswith("Files: "):
                files, _, dirs = line.partition("·")
                files = files.replace("Files:", "").strip()
                dirs = dirs.replace("Dirs:", "").strip()
                plural = "dir" if dirs == "1" else "dirs"
                summary = f"{files} files · {dirs} {plural}"
                break
        if summary:
            self._write_line(f"structure: {summary}", classes="notice")
        else:
            self._write_line(f"structure: {self.structure.cache_path}", classes="notice")

    def _render_user_block(self, text: str) -> None:
        self.transcript.append(f"▌ you\n{text}")
        self._conversation.mount(Static(f"▌ you\n{text}", classes="user-block", markup=False))
        self._scroll_follow()

    def _write_line(self, text: str, classes: str = "") -> None:
        self.transcript.append(text)
        self._conversation.mount(Static(text, classes=classes, markup=False))
        self._scroll_follow()

    def set_follow_scroll(self, follow: bool) -> None:
        self._follow = follow

    def maybe_resume_follow(self) -> None:
        conv = self._conversation
        if conv.max_scroll_y is None or conv.scroll_y >= conv.max_scroll_y - 1:
            self._follow = True

    def _scroll_follow(self) -> None:
        if self._follow and self._conversation is not None:
            self._conversation.scroll_end(animate=False)
            # Widget heights (markdown, boxes) land on the next layout pass,
            # so the computed bottom is stale; re-assert it once layout settles.
            self.set_timer(0.05, self._scroll_follow_settled)

    def _scroll_follow_settled(self) -> None:
        if self._follow and self._conversation is not None:
            self._conversation.scroll_end(animate=False)

    # -- sidebar ------------------------------------------------------------

    def _refresh_memory(self) -> None:
        lines = [str(self.memory.file_path(section)) for section in SECTIONS]
        digest = self.memory.load_digest()
        if digest:
            lines.append("")
            lines.extend(digest.splitlines())
        if len(lines) > 24:
            lines = lines[:24]
            lines.append("…")
        self.query_one("#memory-view", Static).update("\n".join(lines))

    def _refresh_sessions(self) -> None:
        view = self.query_one("#sessions-view", VerticalScroll)
        view.remove_children()
        entries = sessions.list_sessions()
        if not entries:
            view.mount(Static("(no sessions yet)", classes="sidebar-empty", markup=False))
            return
        for entry in entries:
            sid = entry["id"]
            label = f"{sid}  {(entry['prompt'] or '')[:50]}"
            view.mount(
                Button(label, classes="session-btn", action=f"resume_session({sid!r})")
            )

    def _resume_session(self, sid: str) -> None:
        """Switch the active session and render its history into the pane.

        Clears the conversation, draws a dim header notice, then replays the
        session's wire history (at most the last 40 messages) with the same
        visual language as live turns, so the resumed conversation is visible
        immediately. The next submitted prompt continues from this history.
        """
        if self.turn_active:
            self._write_line("(busy — finish or cancel the current turn first)", classes="notice")
            return
        self.session_id = sid
        self.resume_next = True
        self.sub_title = f"{self.model_id} · {self.session_id}"
        self._conversation.remove_children()
        self.transcript.clear()
        self._follow = True
        self._reset_turn_stream()
        self._hide_thinking()
        self._write_line(f"── resumed session {sid} ──", classes="notice")
        self._render_history(sessions.load_messages(sid))
        self._refresh_sessions()
        self._render_status()

    def _render_history(self, wire_messages: list[dict]) -> None:
        """Render a session's wire history (OpenAI wire dicts) into the pane.

        Same visual language as live turns: user blocks, assistant label +
        markdown (+ optional reasoning when verbose, + one dim line per tool
        call), dim tool-result preview lines. Every line mirrors to
        ``transcript``. At most the last 40 messages are rendered.
        """
        if len(wire_messages) > 40:
            self._write_line("… (earlier messages omitted)", classes="notice")
            wire_messages = wire_messages[-40:]
        for wire in wire_messages:
            role = wire.get("role")
            if role == "user":
                self._render_user_block(wire.get("content", ""))
            elif role == "assistant":
                self._render_history_assistant(wire)
            elif role == "tool":
                content = wire.get("content", "")
                preview = content[:200] + ("…" if len(content) > 200 else "")
                self.transcript.append(f"  → {preview}")
                self._conversation.mount(
                    Static(f"  → {preview}", classes="trace-line", markup=False)
                )
                self._scroll_follow()

    def _render_history_assistant(self, wire: dict) -> None:
        reasoning = wire.get("reasoning_content")
        if reasoning and self.verbose:
            self.transcript.append(f"💭 {reasoning}")
            self._conversation.mount(
                Static(f"💭 {reasoning}", classes="reasoning", markup=False)
            )
        self.transcript.append("▌ hdp")
        content = wire.get("content", "")
        self.transcript.append(content)
        self._conversation.mount(Static("▌ hdp", classes="assistant-label", markup=False))
        self._conversation.mount(Markdown(content, classes="assistant-md"))
        for call in wire.get("tool_calls") or []:
            name, args = _call_name_args(call)
            self.transcript.append(f"⚙ {name}({args})")
            self._conversation.mount(
                Static(f"⚙ {name}({args})", classes="trace-line", markup=False)
            )
        self._scroll_follow()

    def action_resume_session(self, sid: str) -> None:
        """Resume a session from a sidebar Sessions row click."""
        self._resume_session(sid)

    def _on_session_selected(self, sid: str | None) -> None:
        """Result callback from the /sessions popup (None = dismissed)."""
        if sid:
            self._resume_session(sid)

    def _set_api_key(self, key: str) -> None:
        """Persist a key, rebuild the gateway, confirm (never echo the key)."""
        config.save_user_api_key(key)
        self.gateway = Gateway(config.BASE_URL, key, self.model_id)
        self._write_line("connected: API key saved")

    def _on_connect_result(self, key: str | None) -> None:
        """Result callback from the /connect popup (None = cancelled)."""
        if key:
            self._set_api_key(key)

    # -- status bar ---------------------------------------------------------

    def _render_status(self) -> None:
        verbose = "verbose on" if self.verbose else "verbose off"
        self.query_one("#status", Static).update(
            f"{self.model_id} · {self.session_id} · step {self._steps}/{self.max_steps} · "
            f"{self._tool_count} tools · {verbose} · Ctrl+C cancel"
        )

    # -- turn lifecycle -----------------------------------------------------

    def turn_finished(self) -> None:
        self._hide_thinking()
        self._flush_md()
        self.turn_active = False
        self.resume_next = True
        self._prompt_input.disabled = False
        self._prompt_input.focus()
        try:
            self.structure.refresh()  # tool-driven changes between turns
        except OSError:
            pass
        self._refresh_memory()
        self._refresh_sessions()
        self._render_status()

    # -- actions ------------------------------------------------------------

    def action_cancel_turn(self) -> None:
        if self.turn_active:
            self._cancel_turn = True
            self._write_line("canceled", classes="notice")
            self.turn_finished()  # the thread aborts on its next emit
        else:
            self._write_line("nothing to cancel", classes="notice")

    def action_jump_to_bottom(self) -> None:
        self._follow = True
        self._conversation.scroll_end(animate=False)

    # -- slash commands -----------------------------------------------------

    def _run_command(self, text: str) -> None:
        parts = text.split(maxsplit=1)
        cmd = parts[0]
        arg = parts[1].strip() if len(parts) > 1 else ""
        if cmd == "/help":
            self._help()
        elif cmd == "/new":
            self.session_id = sessions.new_session_id()
            self.resume_next = False
            self.sub_title = f"{self.model_id} · {self.session_id}"
            self._render_home()
            self._refresh_sessions()
            self._render_status()
        elif cmd == "/resume":
            if not arg:
                self._write_line("usage: /resume <session-id>", classes="notice")
                return
            self._resume_session(arg)
        elif cmd == "/sessions":
            entries = sorted(
                sessions.list_sessions(), key=lambda e: e["id"], reverse=True
            )
            self.push_screen(SessionsScreen(entries), self._on_session_selected)
            self._refresh_sessions()
        elif cmd == "/connect":
            if arg:
                self._set_api_key(arg)  # inline key, no popup
            else:
                self.push_screen(
                    ConnectScreen(config.load_user_api_key()), self._on_connect_result
                )
        elif cmd == "/structure":
            try:
                doc = self.structure.refresh()
            except OSError as exc:
                self._write_line(f"structure: error: {exc}", classes="error-box")
                return
            self._write_line(f"structure: {self.structure.cache_path}", classes="notice")
            for line in doc.splitlines()[:100]:
                self._write_line(line)
        elif cmd == "/memory":
            digest = self.memory.load_digest()
            if digest:
                self._write_line(digest)
            else:
                self._write_line("(memory empty)")
            for section in SECTIONS:
                self._write_line(str(self.memory.file_path(section)))
            self._refresh_memory()
        elif cmd == "/model":
            self._write_line(self.model_id)
        elif cmd == "/verbose":
            self.verbose = not self.verbose
            self._write_line(f"verbose {'on' if self.verbose else 'off'}")
            self._render_status()
        elif cmd == "/quit":
            self.exit()
        else:
            self._write_line(f"unknown command: {cmd} (try /help)")

    def _help(self) -> None:
        self._write_line(
            "commands: /help /new /resume <id> /sessions /memory /model /verbose "
            "/connect /structure /quit"
        )
        self._write_line(
            "keys: enter send · shift+enter newline · ctrl+p/n history · "
            "tab complete · ctrl+l bottom · ctrl+c cancel · ctrl+q quit"
        )


def main() -> None:
    """Launch the TUI (the default `hdp` surface)."""
    app = HarnessTui()
    app.run()
    # Textual has restored the terminal by now; print the resume hint using
    # the app's last session id (so /new or /resume mid-session is reflected).
    print(
        f"Session {app.session_id} — resume with: hdp run --resume {app.session_id}  "
        f"(or /resume {app.session_id} in the TUI)"
    )


if __name__ == "__main__":
    main()
