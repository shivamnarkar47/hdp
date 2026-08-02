"""Read-only tool-result cache: a small JSON file keyed by (tool, args, sig).

Iterative loops re-run the same greps/reads on unchanged trees. This cache
stores read/grep/glob results in ``.hdp/tool-cache.json`` (git-ignored)
keyed by ``tool|sha256(args_json)|structure_signature`` — the sha256 keeps
the file small and the structure signature in the key means a changed tree
auto-misses. Storage is stdlib-only: lazy JSON load on first access, atomic
write (temp file + os.replace) like STRUCTURE.md, a hard size cap, and a
``threading.Lock`` so parallel read batches never interleave on the file.
Missing or corrupt files degrade to an empty cache, never a crash.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path


class ToolCache:
    """Persistent tool-result cache under ``.hdp/tool-cache.json``."""

    def __init__(self, path: Path, max_bytes: int = 4_000_000) -> None:
        self._path = Path(path)
        self._max_bytes = max_bytes
        self._lock = threading.Lock()
        self._data: dict[str, str] | None = None  # lazy: loaded on first access

    # -- keying -------------------------------------------------------------

    @staticmethod
    def _key(tool: str, args_json: str, signature: str) -> str:
        digest = hashlib.sha256(args_json.encode("utf-8")).hexdigest()
        return f"{tool}|{digest}|{signature}"

    # -- public API ---------------------------------------------------------

    def get(self, tool: str, args_json: str, signature: str) -> str | None:
        """Return the cached result string, or None on a miss."""
        with self._lock:
            return self._load().get(self._key(tool, args_json, signature))

    def put(self, tool: str, args_json: str, signature: str, result: str) -> None:
        """Store a result string; skips the disk write when the file would
        exceed ``max_bytes`` (the in-memory copy still serves this process).

        Disk writes are best-effort: the in-memory dict is updated first and
        any OSError (including concurrent-writer races on the shared cache
        file) is swallowed so a cache failure never breaks a tool call.
        """
        with self._lock:
            data = self._load()
            data[self._key(tool, args_json, signature)] = result
            try:
                self._write(data)
            except OSError:
                pass  # cache is best-effort; the in-memory copy still serves

    def drop(self) -> None:
        """Delete the cache file (and forget the in-memory copy)."""
        with self._lock:
            self._data = None
            try:
                self._path.unlink()
            except FileNotFoundError:
                pass

    # -- internals ----------------------------------------------------------

    def _load(self) -> dict[str, str]:
        """Lazily load the JSON file; missing/corrupt -> empty cache."""
        if self._data is not None:
            return self._data
        data: dict[str, str] = {}
        try:
            parsed = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                data = {
                    str(k): str(v) for k, v in parsed.items() if isinstance(v, str)
                }
        except (OSError, ValueError):
            pass  # missing file, bad JSON, or a non-dict payload: empty cache
        self._data = data
        return data

    def _write(self, data: dict[str, str]) -> None:
        payload = json.dumps(data, ensure_ascii=False)
        if len(payload.encode("utf-8")) > self._max_bytes:
            return  # cap hit: never grow the file unbounded
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Unique temp name: concurrent --batch workers share this cache file
        # and must never collide on the same .tmp path (a race would make
        # os.replace fail with FileNotFoundError).
        tmp = self._path.with_name(
            self._path.name + f".tmp{os.getpid()}{threading.get_ident()}"
        )
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, self._path)
