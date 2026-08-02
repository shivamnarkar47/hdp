"""Command-line entry point: the hdp script.

`hdp` with no subcommand launches the Textual TUI (textual is imported
lazily so every other command works even when it is not installed).
`hdp run` is a one-shot, non-interactive agent run; `hdp sessions list`
shows the persisted session store; `hdp sessions show|delete|prune` manage
individual sessions; `hdp doctor` self-checks the environment without
spending tokens.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import urllib.error
import urllib.request
from pathlib import Path

from harness import __version__, config, sessions
from harness.gateway import Gateway, GatewayError
from harness.loop import AgentLoop, LoopError
from harness.memory import Memory
from harness.tools import ToolRegistry

# The same UA gateway.py sends (proven to pass Cloudflare WAF error 1010).
_DOCTOR_UA = "python-requests/2.31.0"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hdp",
        description="hdp — DeepSeek V4 Flash agent harness",
    )
    parser.add_argument("--version", action="version", version=f"hdp {__version__}")
    sub = parser.add_subparsers(dest="subcommand")

    run_p = sub.add_parser("run", help="run a one-shot prompt")
    run_p.add_argument("prompt", help="the task to run, or `-` to read it from stdin")
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
    show_p = sessions_sub.add_parser("show", help="print a session's events")
    show_p.add_argument("id", help="session id")
    delete_p = sessions_sub.add_parser("delete", help="delete a session")
    delete_p.add_argument("id", help="session id")
    prune_p = sessions_sub.add_parser("prune", help="delete old sessions")
    prune_p.add_argument(
        "--keep", type=int, default=20, help="keep the newest N sessions (default: 20)"
    )

    sub.add_parser("doctor", help="self-check the environment")

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
    elif args.subcommand == "sessions":
        code = _sessions(args)
    elif args.subcommand == "doctor":
        code = _doctor(args)
    else:
        code = 2  # unreachable: argparse rejects unknown subcommands
    sys.exit(code)


def _run(args: argparse.Namespace) -> int:
    prompt = args.prompt
    if prompt == "-":
        # Reading from a TTY blocks; that is the user's explicit choice.
        prompt = sys.stdin.read().strip()
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
        answer = loop.run(prompt, emit=emit_cb)
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
                    "usage": loop.usage,
                }
            )
        )
    return 0


def _sessions(args: argparse.Namespace) -> int:
    sub = args.sessions_subcommand
    if sub == "list":
        for entry in sessions.list_sessions():
            print(f"{entry['id']}  {entry['ts'] or '-'}  {(entry['prompt'] or '')[:80]}")
        return 0
    if sub == "show":
        return _sessions_show(args.id)
    if sub == "delete":
        return _sessions_delete(args.id)
    if sub == "prune":
        return _sessions_prune(args.keep)
    print("hdp: unknown sessions subcommand", file=sys.stderr)
    return 2


def _sessions_show(session_id: str) -> int:
    if not (sessions.get_store_dir() / f"{session_id}.jsonl").is_file():
        print(f"hdp: no such session: {session_id}", file=sys.stderr)
        return 1
    for record in sessions.read_events(session_id):
        data = json.dumps(record.get("data", {}), ensure_ascii=False, separators=(",", ":"))
        print(f"{record.get('ts')} | {record.get('type')} | {data}")
    return 0


def _sessions_delete(session_id: str) -> int:
    if sessions.delete_session(session_id):
        print(f"deleted {session_id}")
        return 0
    print(f"no such session: {session_id}")
    return 1


def _sessions_prune(keep: int) -> int:
    deleted = sessions.prune_sessions(keep)
    if not deleted:
        print("nothing to prune")
        return 0
    for session_id in deleted:
        print(f"deleted {session_id}")
    return 0


def _doctor(args: argparse.Namespace) -> int:
    """Self-check: print a terse checklist; no tokens are spent."""
    print(f"python: {sys.version.split()[0]}")

    try:
        import textual

        textual_version = getattr(textual, "__version__", "unknown")
    except ImportError:
        textual_version = "MISSING"
    print(f"textual: {textual_version}")

    key_source = _api_key_source()
    print(f"api key: {key_source}")

    gateway_ok = _gateway_reachable()
    print(f"gateway: {'reachable' if gateway_ok else 'unreachable'}")

    structure_path = Path.cwd() / ".hdp" / "STRUCTURE.md"
    if structure_path.is_file():
        print(f"structure cache: exists · {_structure_entry_count(structure_path)} entries")
    else:
        print("structure cache: missing")

    store = sessions.get_store_dir()
    file_count = len(list(store.glob("*.jsonl"))) if store.is_dir() else 0
    print(f"sessions dir: {store} · {file_count} files")

    ok = textual_version != "MISSING" and key_source != "MISSING" and gateway_ok
    if not ok:
        print("doctor: FAILED")
        return 1
    return 0


def _api_key_source() -> str:
    """Which API-key source resolves, without ever printing the key itself.

    Mirrors config.get_api_key()'s resolution order but reports each source
    independently, so doctor works even when the key is missing.
    """
    if os.environ.get("OPENCODE_API_KEY"):
        return "env"
    if config.load_user_api_key():
        return "user store"
    db_path = Path.home() / ".omp" / "agent" / "agent.db"
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
        try:
            row = con.execute(
                "SELECT data FROM auth_credentials "
                "WHERE provider = 'opencode-go' AND credential_type = 'api_key'"
            ).fetchone()
        finally:
            con.close()
    except Exception:
        row = None
    if row:
        try:
            key = json.loads(row[0]).get("key")
        except (json.JSONDecodeError, IndexError, TypeError):
            key = None
        if key:
            return "omp store"
    return "MISSING"


def _gateway_reachable() -> bool:
    """GET the gateway base URL; any HTTP status counts as reachable.

    Never sends the API key. Timeouts and transport errors report unreachable.
    """
    request = urllib.request.Request(
        config.BASE_URL,
        headers={"User-Agent": _DOCTOR_UA},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            response.read()
        return True
    except urllib.error.HTTPError:
        return True  # even 4xx/5xx proves the gateway is up
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _structure_entry_count(path: Path) -> int:
    """Entries from the STRUCTURE.md header line `Files: N · Dirs: M`."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return 0
    for line in text.splitlines():
        if not line.startswith("Files:"):
            continue
        files = dirs = 0
        for part in line.split("·"):
            part = part.strip()
            if part.startswith("Files:"):
                try:
                    files = int(part[len("Files:"):].strip())
                except ValueError:
                    files = 0
            elif part.startswith("Dirs:"):
                try:
                    dirs = int(part[len("Dirs:"):].strip())
                except ValueError:
                    dirs = 0
        return files + dirs
    return 0
