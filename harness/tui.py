"""Textual full-pane TUI — the default hdp surface.

A chat interface over :class:`harness.loop.AgentLoop`: prompts go in at the
bottom, the agent's streamed answer renders live in the main pane, and tool
calls / results appear as dimmed trace lines. Slash commands manage sessions
and memory; Ctrl+C cancels the running turn cooperatively.

The heavy loop runs in a worker *thread* so streaming never blocks the UI.
Thread widgets are untouchable, so the thread's emit callback marshals every
event back to the main thread via ``App.call_from_thread``; the main-thread
handler is the only place that writes to widgets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.text import Text
from textual import events, on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, Input, RichLog

from harness import config, sessions
from harness.gateway import Gateway
from harness.loop import AgentEvent, AgentLoop
from harness.memory import SECTIONS, Memory
from harness.tools import ToolRegistry


class TurnCancelled(Exception):
    """Raised by the emit callback when the user cancels the active turn."""


class HistoryInput(Input):
    """Prompt input that walks submitted-prompt history with up/down arrows.

    Textual routes key events through :meth:`handle_key`; pressing up shows
    the previous prompt (starting at the newest), down walks back toward the
    newest and eventually back to an empty buffer.
    """

    def handle_key(self, event: events.Key) -> bool:
        app = self.app
        history: list[str] = app.prompt_history
        if event.key == "up":
            if not history:
                return super().handle_key(event)
            if app._history_index is None:
                app._history_index = len(history) - 1
            elif app._history_index > 0:
                app._history_index -= 1
            else:
                return True  # already at the oldest entry; consume the key
            self.value = history[app._history_index]
            self.cursor_position = len(self.value)
            return True
        if event.key == "down":
            if app._history_index is None:
                return super().handle_key(event)
            app._history_index += 1
            if app._history_index >= len(history):
                app._history_index = None
                self.value = ""
            else:
                self.value = history[app._history_index]
            self.cursor_position = len(self.value)
            return True
        return super().handle_key(event)


class HarnessTui(App):
    """Full-pane chat TUI for hdp."""

    TITLE = "hdp"

    CSS = """
    #output {
        height: 1fr;
        width: 1fr;
    }
    #prompt {
        dock: bottom;
        height: 3;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "cancel_turn", "Cancel"),
        Binding("ctrl+q", "quit", "Quit"),
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

        self.session_id = sessions.new_session_id()
        self.sub_title = f"{self.model_id} · {self.session_id}"
        self.resume_next = False
        self._cancel_turn = False
        self.turn_active = False
        self.verbose = False
        self.prompt_history: list[str] = []
        self._history_index: int | None = None
        self.output_lines: list[str] = []
        # NB: do NOT use the name `_loop` here — App._loop is Textual's
        # internal asyncio event loop that call_from_thread() relies on.
        self._agent_loop: AgentLoop | None = None
        self._current_task: str | None = None
        self._prompt_input: HistoryInput | None = None
        self._output: RichLog | None = None

    # -- widgets ------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield RichLog(id="output", markup=False, wrap=True)
        yield HistoryInput(id="prompt", placeholder="Ask hdp… (/help)")
        yield Footer()

    def on_mount(self) -> None:
        self._output = self.query_one("#output", RichLog)
        self._prompt_input = self.query_one("#prompt", HistoryInput)
        self._prompt_input.focus()

    # -- input --------------------------------------------------------------

    @on(Input.Submitted)
    def _on_submitted(self, event: Input.Submitted) -> None:
        if self.turn_active:
            self._write_line("(busy — Ctrl+C cancels the current turn)", style="dim")
            return
        text = event.value.strip()
        self._prompt_input.value = ""
        if not text:
            return
        self.prompt_history.append(text)
        self._history_index = None
        if text.startswith("/"):
            self._run_command(text)
        else:
            self._start_turn(text)

    def _start_turn(self, task: str) -> None:
        """Build a fresh one-shot loop and run it on a worker thread."""
        loop = AgentLoop(
            self.gateway,
            self.tools,
            self.memory,
            self.session_id,
            max_steps=self.max_steps,
            allow_dangerous=self.allow_dangerous,
            resume=self.resume_next,
        )
        self._agent_loop = loop
        self._current_task = task
        self._cancel_turn = False
        self.turn_active = True
        self._prompt_input.disabled = True
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
        if kind == "content":
            self._write(event[1])  # type: ignore[arg-type]
        elif kind == "reasoning":
            if self.verbose:
                self._write_line(f"[think] {event[1]}", style="dim")
        elif kind == "tool_start":
            call = event[1]
            self._write_line(f"⚙ {call.name}({call.arguments})", style="dim")
        elif kind == "tool_result":
            _, _call_id, content = event
            collapsed = content[:200] + ("…" if len(content) > 200 else "")
            self._write_line(f"  → {collapsed}", style="dim")
        elif kind == "error":
            self._write_line(f"error: {event[1]}", style="red")
            self.turn_finished()
        elif kind == "done":
            self.turn_finished()

    def _on_loop_error(self, message: str) -> None:
        # The ("error", ...) event already rendered this and finished the turn.
        if not self.turn_active:
            return
        self._write_line(f"error: {message}", style="red")
        self.turn_finished()

    def _write(self, text: str) -> None:
        self.output_lines.append(text)
        self._output.write(text)

    def _write_line(self, text: str, style: str | None = None) -> None:
        self.output_lines.append(text)
        if style:
            self._output.write(Text(text, style=style))
        else:
            self._output.write(text)

    def turn_finished(self) -> None:
        self._prompt_input.disabled = False
        self._prompt_input.focus()
        self.turn_active = False
        self.resume_next = True

    # -- actions ------------------------------------------------------------

    def action_cancel_turn(self) -> None:
        if self.turn_active:
            self._cancel_turn = True
            self._write_line("canceled", style="dim")
            self.turn_finished()  # the thread aborts on its next emit
        else:
            self._write_line("nothing to cancel", style="dim")

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
            self._output.clear()
            self.output_lines.clear()
        elif cmd == "/resume":
            if not arg:
                self._write_line("usage: /resume <session-id>", style="dim")
                return
            self.session_id = arg
            self.resume_next = True
            self.sub_title = f"{self.model_id} · {self.session_id}"
            self._write_line(f"resuming {arg}")
        elif cmd == "/sessions":
            for entry in sessions.list_sessions():
                self._write_line(
                    f"{entry['id']}  {entry['ts'] or '-'}  {(entry['prompt'] or '')[:60]}"
                )
        elif cmd == "/memory":
            digest = self.memory.load_digest()
            if digest:
                self._write_line(digest)
            else:
                self._write_line("(memory empty)")
            for section in SECTIONS:
                self._write_line(str(self.memory.file_path(section)))
        elif cmd == "/model":
            self._write_line(self.model_id)
        elif cmd == "/verbose":
            self.verbose = not self.verbose
            self._write_line(f"verbose {'on' if self.verbose else 'off'}")
        elif cmd == "/quit":
            self.exit()
        else:
            self._write_line(f"unknown command: {cmd} (try /help)")

    def _help(self) -> None:
        self._write_line(
            "commands: /help /new /resume <id> /sessions /memory /model /verbose /quit"
        )


def main() -> None:
    """Launch the TUI (the default `hdp` surface)."""
    HarnessTui().run()


if __name__ == "__main__":
    main()
