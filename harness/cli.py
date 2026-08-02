"""Command-line entry point: the hdp script.

`hdp` with no subcommand launches the Textual TUI (textual is imported
lazily so every other command works even when it is not installed).
`hdp run` is a one-shot, non-interactive agent run; `hdp sessions list`
shows the persisted session store.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from harness import config, sessions
from harness.gateway import Gateway, GatewayError
from harness.loop import AgentLoop, LoopError
from harness.memory import Memory
from harness.tools import ToolRegistry


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hdp",
        description="hdp — DeepSeek V4 Flash agent harness",
    )
    sub = parser.add_subparsers(dest="subcommand")

    run_p = sub.add_parser("run", help="run a one-shot prompt")
    run_p.add_argument("prompt", help="the task to run")
    run_p.add_argument("--dir", default=None, help="project directory (default: cwd)")
    run_p.add_argument("--model", default=None, help=f"model id (default: {config.MODEL_ID})")
    run_p.add_argument("--max-steps", type=int, default=20, help="max agent turns (default: 20)")
    run_p.add_argument(
        "--memory-root", type=Path, default=None, help="memory root (default: <dir>/.agent-memory)"
    )
    run_p.add_argument("--allow-dangerous", action="store_true", help="permit destructive commands")
    run_p.add_argument("--resume", metavar="SESSION_ID", default=None, help="continue a session")
    run_p.add_argument("--verbose", action="store_true", help="print reasoning to stderr")
    run_p.add_argument("--json", action="store_true", help="final JSON line with session/answer")

    sessions_p = sub.add_parser("sessions", help="session store commands")
    sessions_sub = sessions_p.add_subparsers(dest="sessions_subcommand")
    sessions_sub.add_parser("list", help="list sessions")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.subcommand is None:
        # Lazy import: `hdp run` must work even without textual installed.
        from harness.tui import main as tui_main

        tui_main()
        sys.exit(0)
    if args.subcommand == "run":
        code = _run(args)
    else:
        code = _sessions(args)
    sys.exit(code)


def _run(args: argparse.Namespace) -> int:
    # SystemExit(1) from a missing/invalid key is the correct exit for config errors.
    key = config.get_api_key()
    project_dir = Path(args.dir or Path.cwd())
    memory_root = Path(args.memory_root or project_dir / ".agent-memory")
    memory = Memory(memory_root)
    tools = ToolRegistry(
        memory=memory,
        project_dir=project_dir,
        allow_dangerous=args.allow_dangerous,
    )
    gateway = Gateway(config.BASE_URL, key, args.model or config.MODEL_ID)
    session_id = args.resume or sessions.new_session_id()
    loop = AgentLoop(
        gateway,
        tools,
        memory,
        session_id,
        max_steps=args.max_steps,
        allow_dangerous=args.allow_dangerous,
        resume=bool(args.resume),
    )
    tool_calls = 0

    def emit_cb(event) -> None:
        nonlocal tool_calls
        kind = event[0]
        if kind == "content":
            print(event[1], end="", flush=True)
        elif kind == "reasoning":
            if args.verbose:
                print(f"[think] {event[1]}", file=sys.stderr)
        elif kind == "tool_start":
            tool_calls += 1

    try:
        answer = loop.run(args.prompt, emit=emit_cb)
    except LoopError as exc:
        print(f"hdp: {exc}", file=sys.stderr)
        if args.json:
            print(json.dumps({"session_id": session_id, "error": str(exc)}))
        return 2
    except GatewayError as exc:
        print(f"hdp: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(
            json.dumps(
                {
                    "session_id": session_id,
                    "answer": answer,
                    "steps": loop.steps,
                    "tool_calls": tool_calls,
                }
            )
        )
    return 0


def _sessions(args: argparse.Namespace) -> int:
    if args.sessions_subcommand != "list":
        print("hdp: unknown sessions subcommand", file=sys.stderr)
        return 2
    for entry in sessions.list_sessions():
        print(f"{entry['id']}  {entry['ts'] or '-'}  {(entry['prompt'] or '')[:80]}")
    return 0
