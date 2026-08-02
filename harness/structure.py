"""Project-structure awareness: a regenerable `.hdp/STRUCTURE.md` cache.

Scans the project tree once, writes a markdown tree into `.hdp/STRUCTURE.md`
(git-ignored), and refreshes it only when the tree changes (a cheap signature
scan over relpath/size/mtime). Reopen reads the cache directly for instant
context. This is regenerable cache — NOT durable memory (that stays in
``.agent-memory/``).
"""

from __future__ import annotations

import datetime
import hashlib
import os
from pathlib import Path
from typing import Iterator

# Directories skipped at any depth (also keeps the cache from invalidating
# itself: `.hdp` holds STRUCTURE.md, whose own mtime must never enter the sig).
NOISE_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "node_modules",
    ".hdp",
    "dist",
    "build",
    ".omp",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}

MAX_DEPTH = 6       # don't descend deeper than this (note `…` in the tree)
MAX_ENTRIES = 20_000  # stop walking past this many entries
DOC_MAX_LINES = 500  # STRUCTURE.md is capped; truncate with a notice line

_SIG_PREFIX = "<!-- sig: "


def _human_size(n: int) -> str:
    """'74 B', '1.2 KB', '3.4 MB'."""
    if n < 1024:
        return f"{n} B"
    for unit in ("KB", "MB", "GB"):
        n /= 1024
        if n < 1024:
            return f"{n:.1f} {unit}"
    return f"{n:.1f} TB"


class StructureManager:
    """Scan + cache a project's tree under ``.hdp/STRUCTURE.md``."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        # Most recent structure signature (set by ensure/refresh). The loop
        # hands it to the tool cache each batch so read results are keyed to
        # the tree state they were computed against.
        self.last_signature: str | None = None

    # -- cache --------------------------------------------------------------

    @property
    def cache_path(self) -> Path:
        return self.root / ".hdp" / "STRUCTURE.md"

    def ensure(self) -> str:
        """Return the cached doc; scan + write on first use (no rescan)."""
        path = self.cache_path
        if path.is_file():
            doc = path.read_text(encoding="utf-8")
            # Warm start: recover the signature from the cached doc so the
            # tool cache can serve hits from the very first batch.
            self.last_signature = self._stored_signature()
            return doc
        return self.refresh()

    def refresh(self) -> str:
        """Cheap signature scan; regenerate only when the tree changed."""
        sig = self.signature()
        self.last_signature = sig
        if sig == self._stored_signature():
            path = self.cache_path
            if path.is_file():
                return path.read_text(encoding="utf-8")
        doc = self.scan()
        self._write(doc)
        return doc

    def digest(self, max_chars: int = 4000) -> str:
        """Head-capped excerpt for the system prompt (includes the path)."""
        text = self.ensure()
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n… (truncated)"

    def scan(self) -> str:
        """Build the markdown document (tree + counts + signature comment)."""
        entries = self._ordered(list(self._walk()))
        files = sum(1 for _, is_dir, _, _ in entries if not is_dir)
        dirs = len(entries) - files
        lines = [
            "# Project Structure",
            f"Generated: {datetime.datetime.now().isoformat(timespec='seconds')}",
            f"Root: {self.root}",
            f"Files: {files} · Dirs: {dirs}",
            "## Tree",
        ]
        lines.extend(self._tree_lines(entries))
        if len(entries) >= MAX_ENTRIES:
            lines.append("… (structure truncated: entry cap reached)")
        if len(lines) > DOC_MAX_LINES:
            lines = lines[:DOC_MAX_LINES]
            lines.append("… (structure truncated)")
        lines.append(f"{_SIG_PREFIX}{self._signature_from(entries)}-->")
        return "\n".join(lines)

    # -- walking / hashing --------------------------------------------------

    def _walk(self) -> Iterator[tuple[str, bool, int, int]]:
        """Yield (relpath, is_dir, size, mtime_ns) for non-noise entries, honoring caps.

        Directories first (sorted, case-insensitive) then files, in DFS order;
        a global entry cap stops the walk. Dirs at the depth cap are yielded
        but not descended into. size and mtime_ns come from one stat call so
        _signature_from never has to re-stat.
        """
        count = 0
        for dirpath, dirnames, filenames in os.walk(self.root):
            rel = Path(dirpath).relative_to(self.root)
            depth = len(rel.parts)
            if depth >= MAX_DEPTH:
                dirnames[:] = []  # too deep: don't descend
            dirnames[:] = sorted(
                (d for d in dirnames if d not in NOISE_DIRS), key=str.lower
            )
            for name in dirnames:
                if count >= MAX_ENTRIES:
                    return
                path = Path(dirpath) / name
                try:
                    st = path.stat()
                except OSError:
                    continue
                count += 1
                yield (str(rel / name) if str(rel) != "." else name, True, st.st_size, st.st_mtime_ns)
            for name in sorted(filenames, key=str.lower):
                if count >= MAX_ENTRIES:
                    return
                path = Path(dirpath) / name
                try:
                    st = path.stat()
                except OSError:
                    continue
                count += 1
                yield (str(rel / name) if str(rel) != "." else name, False, st.st_size, st.st_mtime_ns)

    def signature(self) -> str:
        """Stable hash over (relpath, size, mtime_ns) of non-noise entries."""
        return self._signature_from(self._ordered(list(self._walk())))

    @staticmethod
    def _ordered(
        entries: list[tuple[str, bool, int, int]]
    ) -> list[tuple[str, bool, int, int]]:
        """Preorder sort: dirs before files at each level, DFS-expanded."""

        def key(entry: tuple[str, bool, int, int]) -> tuple[tuple[str, str], ...]:
            rel, is_dir, _, _ = entry
            parts = Path(rel).parts
            return tuple(("d", p) for p in parts[:-1]) + (
                ("d" if is_dir else "f", parts[-1]),
            )

        return sorted(entries, key=key)

    @staticmethod
    def _signature_from(entries: Iterator[tuple[str, bool, int, int]]) -> str:
        """Stable hash over (relpath, size, mtime_ns); mtime comes from the walk."""
        hasher = hashlib.sha256()
        for rel, is_dir, size, mtime_ns in sorted(entries, key=lambda e: e[0].lower()):
            hasher.update(f"{rel}\0{size}\0{mtime_ns}\n".encode("utf-8"))
        return hasher.hexdigest()

    def _stored_signature(self) -> str | None:
        path = self.cache_path
        if not path.is_file():
            return None
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return None
        idx = text.rfind(_SIG_PREFIX)
        if idx < 0:
            return None
        end = text.find("-->", idx)
        if end < 0:
            return None
        return text[idx + len(_SIG_PREFIX): end].strip()

    # -- rendering ----------------------------------------------------------

    def _tree_lines(self, entries: list[tuple[str, bool, int, int]]) -> list[str]:
        """Box-drawing tree lines: ``├──``/``└──``/``│``, dirs first."""
        last_flags: dict[tuple[str, ...], bool] = {}
        for i, (rel, _, _, _) in enumerate(entries):
            parts = Path(rel).parts
            nxt = entries[i + 1] if i + 1 < len(entries) else None
            parent = parts[:-1]
            last_flags[parts] = nxt is None or Path(nxt[0]).parts[: len(parent)] != parent
        lines: list[str] = []
        for rel, is_dir, size, _ in entries:
            parts = Path(rel).parts
            prefix = ""
            for k in range(1, len(parts)):
                prefix += "    " if last_flags[parts[:k]] else "│   "
            branch = "└── " if last_flags[parts] else "├── "
            name = parts[-1] + ("/" if is_dir else "")
            if is_dir:
                lines.append(prefix + branch + name)
                if len(parts) >= MAX_DEPTH:
                    below = "    " if last_flags[parts] else "│   "
                    lines.append(prefix + below + "…")
            else:
                lines.append(prefix + branch + f"{name} ({_human_size(size)})")
        return lines

    # -- persistence --------------------------------------------------------

    def _write(self, doc: str) -> None:
        path = self.cache_path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(doc, encoding="utf-8")
        os.replace(tmp, path)
