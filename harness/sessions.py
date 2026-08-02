"""JSONL session store for interactive and resumed conversations."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

VALID_TYPES = {"user", "assistant", "tool_call", "tool_result", "error", "meta"}


def get_store_dir() -> Path:
    """Session store directory: $HARNESSDP_SESSIONS_DIR or the default XDG-ish path."""
    override = os.environ.get("HARNESSDP_SESSIONS_DIR")
    if override:
        return Path(override)
    return Path.home() / ".local" / "share" / "harnessdp" / "sessions"


def new_session_id() -> str:
    """A session id unique to the microsecond: `%Y%m%d-%H%M%S-%f`.

    Second-granularity ids let two runs in the same second silently share a
    JSONL file; the microsecond suffix keeps ids collision-free in practice.
    """
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")


def _session_path(session_id: str) -> Path:
    return get_store_dir() / f"{session_id}.jsonl"


def _record(event: dict) -> dict:
    """Validate one event and build its `{"ts", "type", "data"}` JSONL record.

    `type` must be one of user|assistant|tool_call|tool_result|error|meta
    (ValueError otherwise).
    """
    etype = event.get("type")
    if etype not in VALID_TYPES:
        raise ValueError(f"invalid event type: {etype!r}; allowed: {sorted(VALID_TYPES)}")
    data = event.get("data")
    if not isinstance(data, dict):
        data = {}
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "type": etype,
        "data": data,
    }


def append_events(session_id: str, events: list[dict]) -> None:
    """Append N events as JSON lines in one open/write/close cycle.

    Every event is validated (and its record built) before the file is
    touched, so an invalid type raises ValueError with nothing written. An
    empty list is a no-op. The store directory is created as needed.
    """
    if not events:
        return
    records = [_record(event) for event in events]
    store = get_store_dir()
    store.mkdir(parents=True, exist_ok=True)
    with _session_path(session_id).open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_event(session_id: str, event: dict) -> None:
    """Append one event as a JSON line: `{"ts", "type", "data"}`.

    `type` must be one of user|assistant|tool_call|tool_result|error|meta
    (ValueError otherwise). Thin wrapper over append_events.
    """
    append_events(session_id, [event])


def _event_to_wire(record: dict) -> dict | None:
    """Convert one persisted event to an OpenAI wire dict, or None to skip it."""
    etype = record.get("type")
    data = record.get("data")
    if not isinstance(data, dict):
        data = {}
    if etype == "user":
        return {"role": "user", "content": data.get("content", "")}
    if etype == "assistant":
        wire: dict[str, Any] = {"role": "assistant", "content": data.get("content", "")}
        reasoning = data.get("reasoning_content")
        if reasoning:
            wire["reasoning_content"] = reasoning
        tool_calls = data.get("tool_calls")
        if tool_calls:
            wire["tool_calls"] = tool_calls
        return wire
    if etype == "tool_result":
        return {
            "role": "tool",
            "tool_call_id": data.get("tool_call_id", ""),
            "content": data.get("content", ""),
        }
    return None  # tool_call, error, meta carry no replayable message


def _iter_records(path: Path) -> Iterator[dict]:
    """Yield the JSON records of a session file, one per line.

    The single tolerant JSONL reader: blank lines and corrupt (non-JSON) lines
    are skipped, and a file that cannot be opened (gone, permissions) simply
    yields nothing. Callers must not rely on the file existing first.
    """
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError:
        return
    with handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def read_events(session_id: str) -> list[dict]:
    """Raw JSONL records `{"ts", "type", "data"}` for a session, in file order.

    A missing session file yields []; blank and corrupt lines are skipped.
    """
    path = _session_path(session_id)
    if not path.exists():
        return []
    return list(_iter_records(path))


def load_messages(session_id: str) -> list[dict]:
    """Replay a session as OpenAI wire dicts in file order.

    tool_call/error/meta events are skipped (tool calls ride inside the
    assistant event). A missing session file yields []; corrupt JSON lines are
    skipped.
    """
    messages: list[dict] = []
    for record in read_events(session_id):
        wire = _event_to_wire(record)
        if wire is not None:
            messages.append(wire)
    return messages


def list_sessions() -> list[dict]:
    """Every `<id>.jsonl` sorted by id: `{"id", "ts", "prompt"}`.

    `ts` is the first event's timestamp (or None), `prompt` the first user
    event's content (or None). Missing or empty files are tolerated.
    """
    store = get_store_dir()
    if not store.is_dir():
        return []
    sessions: list[dict] = []
    for path in sorted(store.glob("*.jsonl"), key=lambda p: p.stem):
        first_ts: str | None = None
        first_prompt: str | None = None
        for record in _iter_records(path):
            if first_ts is None and record.get("ts"):
                first_ts = record["ts"]
            if first_prompt is None and record.get("type") == "user":
                data = record.get("data")
                if isinstance(data, dict) and data.get("content"):
                    first_prompt = data["content"]
            if first_ts is not None and first_prompt is not None:
                break
        sessions.append({"id": path.stem, "ts": first_ts, "prompt": first_prompt})
    return sessions


def delete_session(session_id: str) -> bool:
    """Delete `<store>/<id>.jsonl`; True if the file existed.

    Only ever removes the session's own file — never other files.
    """
    path = _session_path(session_id)
    if not path.is_file():
        return False
    path.unlink()
    return True


def prune_sessions(keep: int = 20) -> list[str]:
    """Delete all session files except the newest `keep`, sorted by id.

    Returns the deleted session ids in deletion order. `keep <= 0` deletes
    every session file.
    """
    store = get_store_dir()
    if not store.is_dir():
        return []
    paths = sorted(store.glob("*.jsonl"), key=lambda p: p.stem)
    deleted: list[str] = []
    for path in paths[: max(0, len(paths) - keep)]:
        try:
            path.unlink()
        except FileNotFoundError:
            continue  # vanished between listing and unlink; nothing to report
        deleted.append(path.stem)
    return deleted
