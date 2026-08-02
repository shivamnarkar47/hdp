"""Persistent project memory under .agent-memory/."""

from __future__ import annotations

import contextlib
from datetime import datetime
from pathlib import Path
from typing import Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows has no fcntl module
    fcntl = None  # type: ignore[assignment]

from harness.context import estimate_tokens

# Section key -> display name. Order here is the digest order.
SECTIONS: dict[str, str] = {
    "project-state": "Project State",
    "decisions": "Decisions",
    "patterns": "Patterns",
    "lessons-learned": "Lessons Learned",
}

DIGEST_TOKEN_CAP = 4000
DIGEST_MAX_LINES = 60
MAX_LINES_PER_FILE = 200


@contextlib.contextmanager
def _locked(section_file: Path) -> Iterator[None]:
    """Serialize the whole append sequence with an advisory exclusive flock.

    The lock is taken on the section file itself (always present — created in
    ``__init__``), so no separate lock file is needed and flock needs nothing
    persistent on disk. On platforms without ``fcntl`` (Windows) this is a
    documented no-op: appends still work, but concurrent processes are not
    serialized.
    """
    handle = section_file.open("a", encoding="utf-8")
    try:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


class Memory:
    """Persistent project memory under a root directory (`.agent-memory/`).

    Four markdown files, each created on demand with a `# <Name>` heading:
    project-state.md, decisions.md, patterns.md, lessons-learned.md.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        for section, display in SECTIONS.items():
            path = self.root / f"{section}.md"
            if not path.exists():
                path.write_text(f"# {display}\n\n", encoding="utf-8")

    def file_path(self, section: str) -> Path:
        """Absolute path of a section file (raises ValueError for bad sections)."""
        self._check_section(section)
        return self.root / f"{section}.md"

    def load_digest(self) -> str:
        """Bulleted digest of all four files, capped at 4000 estimated tokens.

        Each section contributes a `### <Name>` heading, the file's absolute
        path, and the first 60 lines of the file's content. If the assembled
        digest exceeds the token cap, content lines are dropped from the end
        until it fits; a final `…[truncated]` marker notes the cut. File paths
        are always kept so the model can read the rest itself.
        """
        lines: list[str] = []
        content_indexes: list[int] = []
        for section, display in SECTIONS.items():
            path = self.file_path(section)
            lines.append(f"### {display}")
            lines.append(f"path: {path.absolute()}")
            content = path.read_text(encoding="utf-8").splitlines()[: DIGEST_MAX_LINES]
            while content and not content[-1].strip():
                content.pop()  # trim trailing whitespace-only lines
            for line in content:
                content_indexes.append(len(lines))
                lines.append(line)

        cut = False
        while content_indexes and estimate_tokens("\n".join(lines)) > DIGEST_TOKEN_CAP:
            lines.pop(content_indexes.pop())  # drop the last content line
            cut = True
        # Make room for the marker so the cap holds after appending it.
        while cut and content_indexes and estimate_tokens(
            "\n".join(lines) + "\n…[truncated]"
        ) > DIGEST_TOKEN_CAP:
            lines.pop(content_indexes.pop())
        if cut:
            lines.append("…[truncated]")
        return "\n".join(lines)

    def append(self, section: str, text: str) -> str:
        """Record `text` under a timestamped `##` heading; dedupes verbatim.

        Returns `"already recorded"` when `text` already appears in the file,
        otherwise the absolute file path after enforcing the 200-line cap (the
        oldest `##` section is dropped until the file fits). The whole
        read → dedupe → append → prune sequence runs under an advisory flock
        on the section file so concurrent kala runs cannot interleave and lose
        entries.
        """
        path = self.file_path(section)
        with _locked(path):
            content = path.read_text(encoding="utf-8")
            if text in content:
                return "already recorded"
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            with path.open("a", encoding="utf-8") as f:
                f.write(f"\n## {timestamp}\n{text}\n")
            self._prune(path)
            return str(path.absolute())

    def record_session_summary(self, task: str, outcome: str) -> None:
        """Record a one-line session summary in project-state.md."""
        self.append("project-state", f"session: {task} → {outcome}")

    @staticmethod
    def _prune(path: Path) -> None:
        """Drop the oldest `##` section repeatedly until the file is ≤ 200 lines."""
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        while len(lines) > MAX_LINES_PER_FILE:
            try:
                start = next(i for i, ln in enumerate(lines) if ln.startswith("## "))
            except StopIteration:
                break  # nothing to prune (e.g. only the header, somehow huge)
            end = next(
                (i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")),
                len(lines),
            )
            del lines[start:end]
        path.write_text("".join(lines), encoding="utf-8")

    @staticmethod
    def _check_section(section: str) -> None:
        if section not in SECTIONS:
            raise ValueError(
                f"invalid section: {section!r}; allowed: {', '.join(sorted(SECTIONS))}"
            )
