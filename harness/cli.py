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
import concurrent.futures
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
    prompt_group = run_p.add_mutually_exclusive_group()
    prompt_group.add_argument(
        "prompt",
        nargs="?",
        help="the task to run, or `-` to read it from stdin",
    )
    prompt_group.add_argument(
        "--batch",
        metavar="FILE",
        default=None,
        help="run prompts from FILE (one per line, or a JSON array), one session each",
    )
    run_p.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="max concurrent --batch tasks (default: min(4, cpu count))",
    )
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
    run_p.add_argument(
        "--no-tool-cache",
        action="store_true",
        help="disable the read-only tool-result cache (.hdp/tool-cache.json)",
    )
    run_p.add_argument(
        "--no-verify",
        action="store_true",
        help="disable verify hooks after mutation (.hdp/hooks.json)",
    )

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
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.subcommand is None:
        # Lazy import: `hdp run` must work even without textual installed.
        from harness.tui import main as tui_main

        tui_main()
        sys.exit(0)
    if args.subcommand == "run":
        code = _run(args, parser)
    elif args.subcommand == "sessions":
        code = _sessions(args)
    elif args.subcommand == "doctor":
        code = _doctor(args)
    else:
        code = 2  # unreachable: argparse rejects unknown subcommands
    sys.exit(code)


def _run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if args.prompt is None:
        if args.batch is None:
            parser.error("the following arguments are required: prompt")
        return _run_batch(args, parser)
    prompt = args.prompt
    if prompt == "-":
        # Reading from a TTY blocks; that is the user's explicit choice.
        prompt = sys.stdin.read().strip()
    session_id = args.resume or sessions.new_session_id()
    record = _run_one(prompt, args, session_id)
    if "error" in record:
        if record.get("error_kind") == "config":
            # config.get_api_key already printed its message; nothing to add.
            return 1
        print(f"hdp: {record['error']}", file=sys.stderr)
        if args.json:
            print(json.dumps({"session_id": record["session_id"], "error": record["error"]}))
        return 2 if record.get("error_kind") == "loop" else 1
    if args.json:
        print(json.dumps(_public_record(record)))
    return 0


def _run_one(prompt: str, args: argparse.Namespace, session_id: str) -> dict:
    """Run one prompt through the full per-run machinery; return its record.

    Shared by the single-run path and every ``--batch`` worker. ``session_id``
    is chosen by the caller (single run: ``--resume`` or a fresh id; batch:
    one fresh id per task — microsecond-precision, collision-free). Returns
    the success record ``{session_id, answer, steps, tool_calls, usage}`` or
    an error record ``{session_id, error, error_kind}`` where ``error_kind``
    is one of "config" | "gateway" | "loop" (config/gateway -> exit 1,
    loop -> exit 2, matching the pre-refactor single-run behavior).
    """
    try:
        key = config.get_api_key()
    except SystemExit:
        # Missing/invalid key: config already printed the instructions.
        return {"session_id": session_id, "error": "no API key", "error_kind": "config"}
    project_dir = Path(args.dir or Path.cwd())
    memory_root = Path(args.memory_root or project_dir / ".agent-memory")
    memory = Memory(memory_root)
    cache = None
    if not args.no_tool_cache:
        from harness.toolcache import ToolCache

        cache = ToolCache(project_dir / ".hdp" / "tool-cache.json")
    tools = ToolRegistry(
        memory=memory,
        project_dir=project_dir,
        allow_dangerous=args.allow_dangerous,
        cache=cache,
    )
    gateway = Gateway(config.BASE_URL, key, args.model or config.MODEL_ID)
    loop = AgentLoop(
        gateway,
        tools,
        memory,
        session_id,
        max_steps=args.max_steps,
        allow_dangerous=args.allow_dangerous,
        resume=bool(args.resume),
        enable_verify=not args.no_verify,
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
        elif kind == "verify":
            print(f"[verify] {event[1]}", file=sys.stderr)
        elif kind == "tool_start":
            tool_calls += 1

    try:
        answer = loop.run(prompt, emit=emit_cb)
    except LoopError as exc:
        return {"session_id": session_id, "error": str(exc), "error_kind": "loop"}
    except GatewayError as exc:
        return {"session_id": session_id, "error": str(exc), "error_kind": "gateway"}
    return {
        "session_id": session_id,
        "answer": answer,
        # getattr defaults keep minimal loop stubs (flag-plumbing tests)
        # working; a real AgentLoop always exposes these.
        "steps": getattr(loop, "steps", 0),
        "tool_calls": tool_calls,
        "usage": getattr(loop, "usage", {}),
    }


def _public_record(record: dict) -> dict:
    """Strip the internal ``error_kind`` field before JSON output.

    Success records gain ``cost`` (estimated dollars from the run's usage);
    error records have no usage and therefore no cost field.
    """
    public = {key: value for key, value in record.items() if key != "error_kind"}
    usage = record.get("usage")
    if usage:
        public["cost"] = round(
            config.estimate_cost(
                usage.get("input_tokens", 0), usage.get("output_tokens", 0)
            ),
            6,
        )
    return public


def _read_batch_prompts(path: str) -> list[str]:
    """Read a --batch file: a JSON array of strings, else one prompt per line.

    Blank lines and empty strings are dropped in both modes. A file that is
    valid JSON but not a string array (e.g. ``[1, 2]`` or a bare string)
    falls back to the line-per-prompt interpretation.
    """
    text = Path(path).read_text(encoding="utf-8")
    prompts: list[str] = []
    stripped = text.strip()
    if stripped:
        try:
            parsed = json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            parsed = None
        if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
            prompts = [item for item in parsed if item.strip()]
    if not prompts:
        prompts = [line.strip() for line in text.splitlines()]
    return [prompt for prompt in prompts if prompt]


def _batch_task(prompt: str, args: argparse.Namespace, session_id: str) -> dict:
    """One --batch worker: announce the task, then run it via _run_one."""
    if not args.json:
        print(f"--- {session_id} ---", flush=True)
    return _run_one(prompt, args, session_id)


def _run_batch(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Run every prompt in the --batch file concurrently; one session each.

    Each task gets its own AgentLoop, Gateway, Memory, ToolRegistry and a
    fresh session id (a fresh registry per task avoids cross-task
    begin_batch/end_batch cache-state leakage; the project dir is shared).
    Records land in file order whatever the worker count. Exit: 0 if all
    tasks produced answers, 1 if any config/key/gateway error, 2 if any
    loop error; per-kind failure counts go to stderr.
    """
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    try:
        prompts = _read_batch_prompts(args.batch)
    except OSError as exc:
        print(f"hdp: cannot read batch file: {exc}", file=sys.stderr)
        return 1
    if not prompts:
        print("hdp: batch file contains no prompts", file=sys.stderr)
        return 1
    session_ids = [sessions.new_session_id() for _ in prompts]
    records: list[dict | None] = [None] * len(prompts)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_batch_task, prompt, args, session_id): index
            for index, (prompt, session_id) in enumerate(zip(prompts, session_ids))
        }
        for future in concurrent.futures.as_completed(futures):
            records[futures[future]] = future.result()
    assert all(record is not None for record in records)
    counts = {"config": 0, "gateway": 0, "loop": 0}
    for record in records:
        kind = record.get("error_kind")
        if kind in counts:
            counts[kind] += 1
    failed = sum(counts.values())
    if failed:
        detail = ", ".join(
            f"{count} {kind}" for kind, count in counts.items() if count
        )
        print(
            f"hdp: batch: {failed} of {len(prompts)} task(s) failed ({detail})",
            file=sys.stderr,
        )
    if args.json:
        print(json.dumps([_public_record(record) for record in records]))
    if counts["config"] or counts["gateway"]:
        return 1
    if counts["loop"]:
        return 2
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
