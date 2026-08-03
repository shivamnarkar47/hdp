"""Textual split-pane TUI — the default kaal surface.

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
At turn end, a code fence the model left dangling (a closing ``` glued to a
content line) is auto-closed so the tail of the answer isn't swallowed into
one literal code block.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.text import Text
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

from harness import agents, config, sessions
from harness.art import BANNER_TAGLINE, BANNER_TITLE
from harness.gateway import Gateway, GatewayError
from harness.loop import AgentEvent, AgentLoop, ToolCall
from harness.memory import SECTIONS, Memory
from harness.structure import StructureManager
from harness.tools import ToolRegistry

# Streaming markdown is re-rendered whole-document per update; flush at most
# every ~100 ms (slower for big docs) and cap one turn's markdown so a
# pathological turn can't jank the UI or blow up the widget. Past the cap the
# overflow appends as a plain text block (~16-20 ms worst flush at 10k).
MD_FLUSH_SECONDS = 0.1
MD_CHAR_CAP = 10_000


def _repair_dangling_fence(text: str) -> str:
    """Fix a code fence the model left dangling (render-side repair).

    CommonMark only honors a closing fence at the start of a line (≤3-space
    indent, backtick run ≥ the opener, then whitespace only). Two model slips
    leave a fence open and dump the rest of the answer into one literal code
    block:

    * a closing `` ``` `` glued to the end of a content line (``foo.py``````) —
      split it onto its own line so the block closes where the model intended;
    * a fence left open at EOF (odd line-start fence count) — append the
      missing closing fence.

    Returns the repaired text; identical to `text` when already balanced.
    """
    out: list[str] = []
    in_fence = False
    fence_len = 0
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip(" ")
        if stripped.startswith("```"):
            run = len(stripped) - len(stripped.lstrip("`"))
            rest = stripped[run:].strip()
            if not in_fence:
                in_fence = True
                fence_len = run
            elif run >= fence_len and not rest:
                in_fence = False
            out.append(line)
            continue
        if in_fence and line.rstrip("\n").endswith("```"):
            # Glued close: `<…>content``` → `<…>content` on its own line + close.
            body = line.rstrip("\n")[:-3]
            out.append(body + "\n```" + ("\n" if line.endswith("\n") else ""))
            in_fence = False
            continue
        out.append(line)
    result = "".join(out)
    if in_fence:
        result += "\n```"
    return result

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
    "/sidebar",
    "/connect",
    "/quit",
    "/structure",
    "/agents",
]

# Braille spinner frames for the "thinking" indicator.
THINK_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

# System prompt for the Ctrl+G AI agent generator: the model must reply with
# ONLY a JSON object (name + description). Kept tight; no tools are offered.
AGENT_GENERATOR_SYSTEM_PROMPT = """\
You are an agent designer for a coding harness. The user describes an agent
persona they want. Respond with ONLY a JSON object: {"name": "<a strong,
fitting name — prefer Sanskrit/epic-flavored names in the spirit of the
Pandavas>", "description": "<one or two sentences describing the persona's
strengths and style>"}."""

# System prompt for the /agents -> n create flow: the user typed only a
# description; the model picks a fitting Mahabharata name and polishes the
# description (same JSON contract as Ctrl+G).
AGENT_FROM_DESCRIPTION_SYSTEM_PROMPT = """\
You are an agent designer for a coding harness. The user describes an agent
persona they want. Generate a fitting NAME for it in the spirit of the
Mahabharata — a character, title, or concept from the epic (e.g. Karna,
Dronacharya, Shakuni, Abhimanyu), or a Sanskrit-flavored title. Respond with
ONLY a JSON object: {"name": "<name>", "description": "<a polished one or
two sentence version of the user's description>"}."""

# Cap for the generator completion: a tiny JSON reply, nothing more.
AGENT_GENERATOR_MAX_TOKENS = 300


def _parse_agent_json(reply: str) -> dict | None:
    """Extract a {name, description} agent dict from a generator reply.

    ``json.loads`` first; on failure, tolerantly grabs the first ``{`` to the
    last ``}`` (models love surrounding prose). Returns None when no usable
    name survives.
    """
    text = reply.strip()
    for candidate in (text, text[text.find("{"): text.rfind("}") + 1]):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(parsed, dict) and parsed.get("name"):
            return {
                "name": str(parsed["name"]).strip(),
                "description": str(parsed.get("description", "")).strip(),
            }
    return None


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


class AgentsScreen(ModalScreen[tuple[str, str] | str | None]):
    """Modal agent switcher: Enter activates, `n` creates, `d` deletes.

    Dismisses with a payload the app's result callback understands:
    ``("activate", name)`` | ``("delete", name)`` | ``"new"`` | None (Esc).
    The currently-active agent's row is marked with a ``✓`` before its name
    and styled accent/bold; each row is two lines (bold name + dim, fully
    wrapped description).
    """

    BINDINGS = [
        Binding("escape", "cancel", "Close"),
        Binding("n", "new", "New"),
        Binding("d", "delete", "Delete"),
    ]

    def __init__(self, agent_list: list[dict], active_name: str | None = None) -> None:
        super().__init__()
        self._agents = agent_list
        self._active_name = active_name

    def compose(self) -> ComposeResult:
        with Vertical(id="agents-box"):
            yield Static(f"Agents ({len(self._agents)})", classes="connect-title")
            if self._agents:
                yield ListView(*[self._row(a) for a in self._agents], id="agent-list")
            else:
                yield Static("(no agents — press n to create)", classes="connect-hint")
            yield Static(
                "↑/↓ select · Enter activate · n new · d delete · Esc close",
                classes="connect-hint",
            )

    def _row(self, agent: dict) -> ListItem:
        name = agent.get("name", "")
        description = agent.get("description", "")
        is_active = agent.get("name") == self._active_name
        name_label = Label(
            (f"✓ {name}" if is_active else name),
            classes="agent-name active" if is_active else "agent-name",
        )
        return ListItem(Vertical(name_label, Label(description, classes="agent-desc")))

    def on_mount(self) -> None:
        if self._agents:
            self.query_one("#agent-list", ListView).focus()

    @on(ListView.Selected)
    def _on_selected(self, event: ListView.Selected) -> None:
        index = self.query_one("#agent-list", ListView).index
        self.dismiss(("activate", self._agents[index]["name"]))

    def action_new(self) -> None:
        self.dismiss("new")

    def action_delete(self) -> None:
        if not self._agents:
            return
        index = self.query_one("#agent-list", ListView).index
        self.dismiss(("delete", self._agents[index]["name"]))

    def action_cancel(self) -> None:
        self.dismiss(None)


class AgentFormScreen(ModalScreen[str | None]):
    """Agent creation form: ONE description input; the name is AI-generated.

    The user describes the persona; Save runs the same generator path as
    Ctrl+G — the model picks a Mahabharata-spirited name and polishes the
    description. Enter saves, Esc cancels.
    """

    BINDINGS = [
        Binding("escape", "cancel", "Close"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="connect-box"):
            yield Static("New agent", classes="connect-title")
            yield Label("Description", classes="connect-hint")
            yield Input(
                placeholder="describe the persona — the name will be AI-generated",
                id="agent-desc-input",
            )
            with Horizontal(id="connect-buttons"):
                yield Button("Save", id="save-btn", variant="primary")
                yield Button("Cancel", id="cancel-btn")

    def on_mount(self) -> None:
        self.query_one("#agent-desc-input", Input).focus()

    @on(Input.Submitted)
    def _on_submitted(self, event: Input.Submitted) -> None:
        self._save()

    @on(Button.Pressed)
    def _on_button(self, event: Button.Pressed) -> None:
        if event.button.id == "save-btn":
            self._save()
        elif event.button.id == "cancel-btn":
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def _save(self) -> None:
        description = self.query_one("#agent-desc-input", Input).value.strip()
        if not description:
            self.query_one("#agent-desc-input", Input).focus()  # guard empty
            return
        self.dismiss(description)


class AgentIntentScreen(ModalScreen[str | None]):
    """AI agent generator prompt: one intent line; Enter generates, Esc cancels."""

    BINDINGS = [
        Binding("escape", "cancel", "Close"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="connect-box"):
            yield Static("Generate agent (AI)", classes="connect-title")
            yield Input(placeholder="Describe the agent you want…", id="agent-intent-input")
            yield Static("Enter generates · Esc cancels", classes="connect-hint")

    def on_mount(self) -> None:
        self.query_one("#agent-intent-input", Input).focus()

    @on(Input.Submitted)
    def _on_submitted(self, event: Input.Submitted) -> None:
        intent = event.input.value.strip()
        if intent:
            self.dismiss(intent)

    def action_cancel(self) -> None:
        self.dismiss(None)


class HarnessTui(App):
    """Split-pane chat TUI for kaal."""

    TITLE = "kaal"

    CSS = """
    Screen {
        background: $surface;
        padding: 1;
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
    SessionsScreen,
    AgentsScreen,
    AgentFormScreen,
    AgentIntentScreen {
        align: center middle;
    }

    #connect-box,
    #sessions-box,
    #agents-box {
        width: 52;
        height: auto;
        max-height: 80%;
        border: round $accent;
        background: $panel;
        padding: 1 2;
    }

    #agents-box {
        width: 60;
        max-height: 60%;
    }

    #session-list {
        height: auto;
        max-height: 50%;
        margin-bottom: 1;
    }

    #agent-list {
        height: auto;
        max-height: 12;
        margin-bottom: 1;
    }

    #agent-list .agent-name {
        text-style: bold;
    }

    #agent-list .agent-name.active {
        color: $accent;
        text-style: bold;
    }

    #agent-list .agent-desc {
        color: $text-muted;
        text-style: dim;
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

    .banner-title {
        color: $accent;
        text-style: bold;
        text-align: center;
        margin-top: 2;
    }

    .banner-tagline {
        color: $text-muted;
        text-align: center;
        margin-bottom: 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "cancel_turn", "Cancel"),
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+l", "jump_to_bottom", "Jump to bottom", show=False),
        # Plain app-level binding is enough: TextArea never binds Ctrl+S, so
        # the unhandled key bubbles up from the focused prompt to the App.
        Binding("ctrl+s", "toggle_sidebar", "Sidebar", show=False),
        # Ctrl+G ("generate") opens the AI agent generator. priority=True so it
        # wins even while the prompt has focus: TextArea binds Ctrl+A to
        # cursor_line_start (select-all territory), so Ctrl+A would be
        # swallowed whenever the prompt is focused — Ctrl+G is unbound there.
        Binding("ctrl+g", "agent_intent", "New agent (AI)", show=False, priority=True),
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
        # Agent personas (harness.agents): seeded with the five Pandavas when
        # .kaal/agents.json is missing; `active` picks the persona injected
        # into every fresh AgentLoop (None = no persona). The persona is on by
        # default: when nothing is active, activate the first agent
        # (Yudhishthira) and persist, so the status bar always leads with a
        # real name and the persona shapes turns from the first prompt.
        self._agents = agents.load(self.project_dir)
        if agents.active_agent(self._agents) is None and self._agents["agents"]:
            self._agents["active"] = self._agents["agents"][0]["name"]
            agents.save(self.project_dir, self._agents)
        self._active_agent = agents.active_agent(self._agents)
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
        # AI agent generator in flight (Ctrl+G or the /agents -> n form):
        # guards against two overlapping generator workers.
        self._generating_agent = False
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
        # Session-only preference (NOT persisted): the sidebar always starts
        # visible; Ctrl+S / /sidebar hide or show it for this session.
        self._sidebar_visible = True
        self._tool_count = len(self.tools.schemas())
        self._steps = 0

        # Live throughput tracking for the status bar (tokens/sec).
        self._stream_chars = 0
        self._last_rate_chars = 0
        self._turn_start: float | None = None
        self._last_tick_time = 0.0
        self._rate = 0.0
        self._rate_timer: Any = None

        # Running token/cost totals for the status bar (session-wide).
        self._total_usage = {"input_tokens": 0, "output_tokens": 0}
        self._total_cost = 0.0
        self._clock_timer: Any = None

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
        self._scroll_settled_pending = False
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
                placeholder="Ask kaal… (/help)",
                soft_wrap=True,
            )
            yield Static(id="status")

    def on_mount(self) -> None:
        self._conversation = self.query_one("#conversation", ConversationScroll)
        self._trace = self.query_one("#trace", VerticalScroll)
        self._prompt_input = self.query_one("#prompt", PromptInput)
        self._suggestions = self.query_one("#suggestions", Vertical)
        # Apply the session state (covers a pre-mount action_toggle_sidebar).
        self.query_one("#sidebar", Vertical).display = self._sidebar_visible
        self._render_home()
        self._structure_notice()
        self._refresh_memory()
        self._refresh_sessions()
        self._render_status()
        # 30s clock ticker keeps the status-bar date fresh; lives for the app
        # lifetime (cheap single-widget update).
        self._clock_timer = self.set_interval(30.0, self._render_status)
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
            agent=self._active_agent,
        )
        self._agent_loop = loop
        self._current_task = task
        self._cancel_turn = False
        self.turn_active = True
        self._reset_turn_stream()
        self._show_thinking()
        # Reset the throughput baseline and start the 1-second ticker.
        self._turn_start = time.monotonic()
        self._last_tick_time = self._turn_start
        self._stream_chars = 0
        self._last_rate_chars = 0
        self._rate = 0.0
        if self._rate_timer is None:
            self._rate_timer = self.set_interval(1.0, self._tick_rate)
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
            self._stream_chars += len(event[1])  # type: ignore[arg-type]
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
        elif kind == "verify":
            # Post-mutation verify output: a dim pane line, NOT streaming md.
            text = event[1]  # type: ignore[arg-type]
            preview = text if len(text) <= 160 else text[:160] + "…"
            self._write_line(f"🧪 verify: {preview}", classes="notice")
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
        """Mount the '▌ kaal' label + the turn's streaming Markdown widget."""
        if self._turn_md is not None:
            return
        self._hide_thinking()  # first content: thinking phase is over
        self.transcript.append("▌ kaal")
        self._conversation.mount(Static("▌ kaal", classes="assistant-label", markup=False))
        self._turn_md = Markdown("", classes="assistant-md")
        self._conversation.mount(self._turn_md)
        self._scroll_follow()

    def _append_content(self, chunk: str) -> None:
        self.transcript.append(chunk)
        self._md_pending.append(chunk)
        if self._md_timer is None:
            self._md_timer = self.set_timer(self._md_flush_interval(), self._flush_md)

    def _md_flush_interval(self) -> float:
        """Adaptive flush cadence: big markdown docs re-render at 4 Hz, not 10 Hz."""
        return MD_FLUSH_SECONDS if len(self._turn_md_text) < 20_000 else 0.25

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

    def _close_unclosed_fence(self) -> None:
        """Repair a code fence the model left dangling (render-side only).

        Called at turn end: split glued closing fences onto their own lines
        and append a missing final close, so the tail of the answer isn't
        swallowed into one literal code block. The transcript mirror keeps
        the model's verbatim text untouched. No-op when already balanced.
        """
        if self._turn_md is None:
            return
        repaired = _repair_dangling_fence(self._turn_md_text)
        if repaired != self._turn_md_text:
            self._turn_md_text = repaired
            self._turn_md.update(self._turn_md_text)

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
        """Render the KESHAVLOK banner + welcome line (startup and `/new`).

        Clears the pane and live-stream state, then mounts the banner title
        (accent/bold) and tagline (dim) as styled Statics, and mirrors every
        line plus the welcome line into the transcript, in order.
        """
        self._conversation.remove_children()
        self.transcript.clear()
        self._follow = True
        self._reset_turn_stream()
        self._hide_thinking()
        self.transcript.append(BANNER_TITLE)
        self._conversation.mount(Static(BANNER_TITLE, classes="banner-title", markup=False))
        self.transcript.append(BANNER_TAGLINE)
        self._conversation.mount(
            Static(BANNER_TAGLINE, classes="banner-tagline", markup=False)
        )
        welcome = f"kaal — {self.model_id} agent. Ask a task, or /help for commands."
        self.transcript.append(welcome)
        self._conversation.mount(Static(welcome, classes="welcome", markup=False))
        # Banner first: start at the top so the title is in view.
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
            # so the computed bottom is stale; re-assert it once layout
            # settles. Coalesce the settle timer: at high chunk rates only one
            # pending re-assert is needed.
            if not self._scroll_settled_pending:
                self._scroll_settled_pending = True
                self.set_timer(0.05, self._scroll_follow_settled)

    def _scroll_follow_settled(self) -> None:
        self._scroll_settled_pending = False
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
        self.transcript.append("▌ kaal")
        content = wire.get("content", "")
        self.transcript.append(content)
        self._conversation.mount(Static("▌ kaal", classes="assistant-label", markup=False))
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

    # -- agents -------------------------------------------------------------

    def _on_agents_result(self, result: tuple[str, str] | str | None) -> None:
        """Result callback from the /agents popup.

        ("activate", name) / ("delete", name) / "new" / None (cancelled).
        Agent activations are session state, not session events — nothing is
        # written to the session JSONL; .kaal/agents.json covers persistence.
        """
        if result is None:
            return
        if isinstance(result, tuple):
            action, name = result
            if action == "activate":
                self._activate_agent(name)
            elif action == "delete":
                self._delete_agent(name)
        elif result == "new":
            self.push_screen(AgentFormScreen(), self._on_agent_form_result)

    def _on_agent_form_result(self, description: str | None) -> None:
        """Result callback from the /agents -> n form (None = cancelled).

        The user only writes a description; the name is AI-generated by the
        same generator path as Ctrl+G (Mahabharata-spirited, polished).
        """
        if description:
            self._generate_agent(
                description, AGENT_FROM_DESCRIPTION_SYSTEM_PROMPT, "created and active"
            )

    def _activate_agent(self, name: str) -> None:
        """Set the active persona, persist, and confirm (next turn uses it)."""
        self._agents["active"] = name
        agents.save(self.project_dir, self._agents)
        self._active_agent = agents.active_agent(self._agents)
        self._write_line(f"agent: {name} active", classes="notice")
        self._render_status()  # the bar leads with the active agent's name

    def _add_agent(self, agent: dict, notice: str | None = None) -> None:
        """Add a new agent, activate it, persist, and confirm.

        Dedupes case-insensitively by name: an existing name is REPLACED in
        place (its position kept) instead of appended, so overlapping
        generators or a same-name re-run can never accumulate duplicates.
        """
        name = agent["name"]
        for i, existing in enumerate(self._agents["agents"]):
            if existing.get("name", "").lower() == name.lower():
                self._agents["agents"][i] = agent
                self._agents["active"] = name
                agents.save(self.project_dir, self._agents)
                self._active_agent = agents.active_agent(self._agents)
                self._write_line(
                    f"agent: {name} created (replaced existing)", classes="notice"
                )
                self._render_status()
                return
        self._agents["agents"].append(agent)
        self._agents["active"] = name
        agents.save(self.project_dir, self._agents)
        self._active_agent = agents.active_agent(self._agents)
        self._write_line(notice or f"agent: {name} added and active", classes="notice")
        self._render_status()

    def _delete_agent(self, name: str) -> None:
        """Delete an agent; if it was active, active becomes None."""
        self._agents["agents"] = [
            a for a in self._agents["agents"] if a.get("name") != name
        ]
        if self._agents.get("active") == name:
            self._agents["active"] = None
        agents.save(self.project_dir, self._agents)
        self._active_agent = agents.active_agent(self._agents)
        self._write_line(f"agent: {name} deleted", classes="notice")
        self._render_status()

    # -- Ctrl+G: AI agent generator -----------------------------------------

    def action_agent_intent(self) -> None:
        """Ctrl+G: open the AI agent generator (gateway completion, no tools)."""
        if self._generating_agent:
            self._write_line("agent generator: already running", classes="notice")
            return
        if self.turn_active:
            self._write_line("(busy — wait for the current turn)", classes="notice")
            return
        if len(self.screen_stack) > 1:
            # A modal is already up (e.g. /sessions); don't stack another.
            return
        self.push_screen(AgentIntentScreen(), self._on_agent_intent)

    def _on_agent_intent(self, intent: str | None) -> None:
        """Result callback from the intent screen (None = cancelled)."""
        if intent:
            self._generate_agent(
                intent, AGENT_GENERATOR_SYSTEM_PROMPT, "generated and active"
            )

    def _generate_agent(self, prompt: str, system_prompt: str, phrase: str) -> None:
        """Run one agent-generation completion on a worker thread.

        Shared by Ctrl+G (a free-form intent; the model invents the whole
        persona) and the /agents -> n form (a description; the model picks the
        name and polishes the text) — the only difference is the system-prompt
        wording. On success the new agent is added + activated + persisted;
        on failure a notice is written, no crash. The thinking indicator
        stays up until the worker marshals the result back (call_from_thread).

        Re-entry guard: while a generator is already in flight (or a turn is
        active) a second start is refused with a notice, so two overlapping
        workers can never both land an agent.
        """
        if self._generating_agent or self.turn_active:
            self._write_line("agent generator: already running", classes="notice")
            return
        self._generating_agent = True
        self.turn_active = True  # guards re-entry (action_agent_intent)
        self._show_thinking()
        # run_worker takes no worker args (Textual 8.x), so the generator
        # arguments ride in a closure, like _start_turn reads _current_task.
        self.run_worker(
            lambda: self._generate_agent_thread(prompt, system_prompt, phrase),
            thread=True,
            group="agent-generator",
        )

    def _generate_agent_thread(self, prompt: str, system_prompt: str, phrase: str) -> None:
        """Worker thread body for the generator. Never touches widgets."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            reply = ""
            for event in self.gateway.stream(
                messages, tools=None, max_tokens=AGENT_GENERATOR_MAX_TOKENS
            ):
                kind = event[0]
                if kind == "content":
                    reply += event[1]  # type: ignore[operator]
                elif kind == "error":
                    raise GatewayError(event[1])  # type: ignore[arg-type]
                elif kind == "done":
                    break
            self.call_from_thread(self._on_generator_done, reply, phrase)
        except Exception as exc:
            # Broad catch (incl. GatewayError): a transport/model slip must
            # never crash the TUI.
            self.call_from_thread(self._on_generator_error, str(exc))

    def _on_generator_done(self, reply: str, phrase: str) -> None:
        """Main thread: parse the reply; add + activate on success."""
        self.turn_active = False
        self._generating_agent = False
        self._hide_thinking()
        agent = _parse_agent_json(reply)
        if agent is None:
            self._write_line(
                "agent generator: could not parse a name/description", classes="error-box"
            )
            return
        self._add_agent(agent, notice=f"agent: {agent['name']} {phrase}")

    def _on_generator_error(self, message: str) -> None:
        """Main thread: generator failure — dim notice, no crash."""
        self.turn_active = False
        self._generating_agent = False
        self._hide_thinking()
        self._write_line(f"agent generator: {message}", classes="error-box")

    # -- status bar ---------------------------------------------------------

    # tmux-style segmented blocks. Hardcoded ANSI-safe colors that read well
    # on Textual's default dark theme (Rich Text styles cannot resolve
    # Textual $design tokens, so the block backgrounds are literals); the
    # center segment is unstyled and inherits #status's muted CSS color. The
    # agent block is inverted (dark text on light) so it pops from the rest.
    _STATUS_AGENT_STYLE = "bold #1f2430 on #81a1c1"
    _STATUS_LEFT_STYLE = "bold #d7dae0 on #1f2430"
    _STATUS_RIGHT_STYLE = "bold #eceff4 on #3b4252"

    def _session_short(self) -> str:
        """The HHMMSS part of a %Y%m%d-%H%M%S-%f session id (else last 6)."""
        parts = self.session_id.split("-")
        if len(parts) >= 2 and parts[1]:
            return parts[1]
        return self.session_id[-6:]

    def _status_bar(self, short_model: bool = False) -> Text:
        """Build the tmux-style bar: agent block, left block, muted center, clock.

        The first segment is the active agent's name (its own inverted block);
        "—" when no agent is active. ``short_model`` is the >90-char fallback:
        it shortens the model, drops the session suffix, and caps an over-long
        agent name (agent names are short by design) — every metric segment
        and the clock stay.
        """
        model = self.model_id
        if short_model:
            # deepseek-v4-flash -> v4-flash (last two dash chunks).
            model = "-".join(model.split("-")[-2:])
        agent_name = self._active_agent["name"] if self._active_agent else "—"
        clock = datetime.now().strftime("%a %d %b %H:%M")
        left = f" kaal · {model} · {self._session_short()} "
        if short_model:
            left = f" kaal · {model} "
            # Agent names are short by design (<= the 12-char Yudhishthira);
            # the >90-char fallback also drops the clock's weekday.
            if len(agent_name) > 12:
                agent_name = agent_name[:12]
            clock = datetime.now().strftime("%d %b %H:%M")
        rate = self.tools.cache_hit_rate()
        cache = "cache –" if rate is None else f"cache {rate * 100:.0f}%"
        bar = Text()
        bar.append(f" {agent_name} ", style=self._STATUS_AGENT_STYLE)
        bar.append("│")
        bar.append(left, style=self._STATUS_LEFT_STYLE)
        bar.append("│")
        bar.append(f" step {self._steps}/{self.max_steps} ")
        bar.append("│")
        bar.append(f" {self._rate:.0f} tok/s ")
        bar.append("│")
        bar.append(f" {cache} ")
        bar.append("│")
        bar.append(f" ${self._total_cost:.4f} ")
        bar.append("│")
        bar.append(f" {clock} ", style=self._STATUS_RIGHT_STYLE)
        return bar

    def _render_status(self) -> None:
        bar = self._status_bar()
        if len(bar.plain) > 90:
            bar = self._status_bar(short_model=True)
        self.query_one("#status", Static).update(bar)

    @staticmethod
    def _tokens_per_sec(chars: int, seconds: float) -> float:
        """Chars→tokens at chars // 3 (matches the harness's estimate_tokens)."""
        return chars / 3 / seconds if seconds > 0 else 0.0

    def _tick_rate(self) -> None:
        """1-second ticker: instantaneous tokens/sec since the last tick."""
        if self._turn_start is None:
            return
        now = time.monotonic()
        elapsed = now - self._last_tick_time
        chars = self._stream_chars - self._last_rate_chars
        self._rate = self._tokens_per_sec(chars, elapsed)
        self._last_rate_chars = self._stream_chars
        self._last_tick_time = now
        self._render_status()

    # -- turn lifecycle -----------------------------------------------------

    def turn_finished(self) -> None:
        self._hide_thinking()
        self._flush_md()
        self._close_unclosed_fence()
        self.turn_active = False
        self.resume_next = True
        # Freeze the turn's average throughput on the bar until the next turn.
        if self._turn_start is not None:
            self._rate = self._tokens_per_sec(
                self._stream_chars, time.monotonic() - self._turn_start
            )
        if self._rate_timer is not None:
            self._rate_timer.stop()
            self._rate_timer = None
        # Accumulate the session's running cost on the bar (loop is still set
        # here; each turn builds a fresh AgentLoop with its own usage dict).
        if self._agent_loop is not None:
            usage = self._agent_loop.usage
            self._total_usage["input_tokens"] += usage.get("input_tokens", 0)
            self._total_usage["output_tokens"] += usage.get("output_tokens", 0)
            self._total_cost = config.estimate_cost(
                self._total_usage["input_tokens"], self._total_usage["output_tokens"]
            )
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

    def action_toggle_sidebar(self) -> None:
        """Ctrl+S / /sidebar: hide or show the right sidebar (session-only)."""
        self._sidebar_visible = not self._sidebar_visible
        if self._conversation is None:
            return  # pre-mount: on_mount applies the flipped state
        self.query_one("#sidebar", Vertical).display = self._sidebar_visible
        self._write_line(
            "sidebar hidden" if not self._sidebar_visible else "sidebar shown",
            classes="notice",
        )

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
        elif cmd == "/agents":
            self.push_screen(
                AgentsScreen(
                    self._agents.get("agents", []),
                    self._agents.get("active"),
                ),
                self._on_agents_result,
            )
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
        elif cmd == "/sidebar":
            self.action_toggle_sidebar()
        elif cmd == "/quit":
            self.exit()
        else:
            self._write_line(f"unknown command: {cmd} (try /help)")

    def _help(self) -> None:
        self._write_line(
            "commands: /help /new /resume <id> /sessions /agents /memory /model "
            "/verbose /sidebar /connect /structure /quit"
        )
        self._write_line(
            "keys: enter send · shift+enter newline · ctrl+p/n history · "
            "tab complete · ctrl+l bottom · ctrl+s sidebar · ctrl+g new agent (AI) · "
            "ctrl+c cancel · ctrl+q quit"
        )


def main() -> None:
    """Launch the TUI (the default `kaal` surface)."""
    app = HarnessTui()
    app.run()
    # Textual has restored the terminal by now; print the resume hint using
    # the app's last session id (so /new or /resume mid-session is reflected).
    print(
        f"Session {app.session_id} — resume with: kaal run --resume {app.session_id}  "
        f"(or /resume {app.session_id} in the TUI)"
    )


if __name__ == "__main__":
    main()
