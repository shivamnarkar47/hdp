"""Tool registry: OpenAI function schemas plus safe local execution.

Exposes the agent's tools as OpenAI function-call schemas (the API ``tools``
param) and executes calls with guardrails: file paths are constrained to the
project directory (no ``..`` escapes) and destructive shell commands hit a
DENY list. Shell commands run in a sanitized environment: ``PATH`` is limited
to the project's ``.venv/bin`` when present, plus ``/usr/local/bin``,
``/usr/bin``, and ``/bin``. ``execute`` always returns a string or raises
``ToolError`` (unknown tool, non-dict args, internal errors).
"""

from __future__ import annotations

import copy
import itertools
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Callable

MAX_RESULT_CHARS = 10_000
TRUNCATED_SUFFIX = "…[truncated]"

# Destructive command patterns, matched case-insensitively against the shell
# command before execution (skipped when the registry is allow_dangerous).
DENY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"rm\s+-rf", re.IGNORECASE),
    re.compile(r"git\s+push", re.IGNORECASE),
    re.compile(r"git\s+reset\s+--hard", re.IGNORECASE),
    re.compile(r"git\s+clean\s+-f", re.IGNORECASE),
    re.compile(r"mkfs", re.IGNORECASE),
    re.compile(r"dd\s+if=", re.IGNORECASE),
    re.compile(r":\{\(\)", re.IGNORECASE),
    re.compile(r">\s*/dev/sd", re.IGNORECASE),
]

_SKIP_DIRS = {".git", "__pycache__", "node_modules"}
_MEMORY_SECTIONS = ("project-state", "decisions", "patterns", "lessons-learned")
_DENY_MESSAGE = "blocked by harness policy (destructive command)"


class ToolError(Exception):
    """Hard failure: unknown tool, non-dict args, or an internal error."""


def _cap(text: str) -> str:
    """Truncate result strings longer than MAX_RESULT_CHARS."""
    if len(text) > MAX_RESULT_CHARS:
        return text[:MAX_RESULT_CHARS] + TRUNCATED_SUFFIX
    return text


def resolve_relative(path: str, cwd: Path) -> Path:
    """Resolve a user-supplied path against ``cwd``, rejecting any escape.

    Absolute paths inside ``cwd`` are allowed (``resolve`` handles them);
    absolute paths outside it and ``..`` escapes raise ``ToolError`` with a
    ``blocked: ...`` message that ``execute`` surfaces as a result string.
    """
    cwd = Path(cwd)
    resolved = (cwd / str(path)).resolve()
    if not resolved.is_relative_to(cwd.resolve()):
        raise ToolError(f"blocked: path escapes project directory: {path}")
    return resolved


class ToolRegistry:
    """Registry of the agent's tools: schemas for the API, safe execution."""

    def __init__(self, memory=None, project_dir=None, allow_dangerous=False) -> None:
        self._memory = memory
        self._project_dir = (
            Path(project_dir).resolve() if project_dir is not None else Path.cwd().resolve()
        )
        self.allow_dangerous = allow_dangerous
        self._handlers: dict[str, Callable[[dict[str, Any]], str]] = {
            "read": self._tool_read,
            "grep": self._tool_grep,
            "glob": self._tool_glob,
            "write": self._tool_write,
            "edit": self._tool_edit,
            "bash": self._tool_bash,
            "memory_append": self._tool_memory_append,
        }

    @property
    def project_dir(self) -> Path:
        return self._project_dir

    def schemas(self) -> list[dict]:
        """OpenAI function-call schemas for every tool."""
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": copy.deepcopy(parameters),
                },
            }
            for name, description, parameters in _TOOL_SPECS
        ]

    def execute(self, name: str, args: dict) -> str:
        """Run tool ``name`` with ``args``; return a result string or raise ToolError."""
        if not isinstance(args, dict):
            raise ToolError(f"tool {name}: arguments must be a dict")
        handler = self._handlers.get(name)
        if handler is None:
            raise ToolError(f"unknown tool: {name}")
        try:
            return handler(args)
        except ToolError as exc:
            # Tool-level failures (blocked paths, missing/invalid args) are
            # surfaced as result strings; unknown tools and non-dict args are
            # raised above. Unexpected exceptions are internal errors.
            return str(exc)
        except Exception as exc:  # noqa: BLE001 - wrap as an internal error
            raise ToolError(f"tool {name} failed: {exc}") from exc

    # -- argument plumbing --------------------------------------------------

    def _require(self, args: dict[str, Any], name: str, tool: str) -> Any:
        if name not in args or args[name] is None:
            raise ToolError(f"{tool}: missing required argument: {name}")
        return args[name]

    def _int_arg(self, args: dict[str, Any], name: str, tool: str, default: int | None = None) -> int | None:
        if name not in args or args[name] is None:
            return default
        try:
            return int(args[name])
        except (TypeError, ValueError):
            raise ToolError(f"{tool}: invalid {name}: {args[name]!r}") from None

    # -- tools ---------------------------------------------------------------

    def _tool_read(self, args: dict[str, Any]) -> str:
        tool = "read"
        path = self._require(args, "path", tool)
        offset = self._int_arg(args, "offset", tool)
        limit = self._int_arg(args, "limit", tool)
        if limit is not None and limit < 0:
            raise ToolError(f"{tool}: limit must be >= 0")
        target = resolve_relative(path, self.project_dir)
        if target.is_dir():
            return _cap(self._directory_listing(target))
        if not target.is_file():
            return f"read: no such file: {path}"
        try:
            text = self._read_lines(target, offset, limit)
        except OSError as exc:
            return f"read: {exc}"
        return _cap(text)

    def _read_lines(self, path: Path, offset: int | None, limit: int | None) -> str:
        start = max(0, (offset or 1) - 1)
        if limit is None:
            # Whole-file read: documented behavior; the 10k result cap applies.
            # Read at most cap + suffix slack — the caller truncates to the first
            # 10k chars anyway, so reading further is pure waste. Skip `start`
            # lines first (lazily) to honor offset.
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                if start:
                    for _ in range(start):
                        next(f, None)
                return f.read(MAX_RESULT_CHARS + len(TRUNCATED_SUFFIX))
        # Stream only the requested window — never materialize the whole file.
        # The TextIOWrapper decodes lazily, line by line, with errors="replace".
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return "".join(itertools.islice(f, start, start + limit))

    def _directory_listing(self, path: Path) -> str:
        """Depth-limited listing: immediate children, then one more level."""
        lines: list[str] = []
        for dirpath, dirnames, filenames in os.walk(path):
            depth = len(Path(dirpath).relative_to(path).parts)
            if depth == 0:
                for name in sorted(dirnames):
                    lines.append(f"{name}/")
                for name in sorted(filenames):
                    lines.append(name)
            elif depth == 1:
                parent = Path(dirpath).name
                for name in sorted(dirnames):
                    lines.append(f"{parent}/{name}/")
                for name in sorted(filenames):
                    lines.append(f"{parent}/{name}")
            if depth >= 1:
                dirnames[:] = []  # never descend past level 2
        return "\n".join(lines)

    def _tool_grep(self, args: dict[str, Any]) -> str:
        tool = "grep"
        pattern = self._require(args, "pattern", tool)
        root = self.project_dir
        if args.get("path") is not None:
            root = resolve_relative(args["path"], self.project_dir)
        if root.name in _SKIP_DIRS:
            return f"no matches for {pattern}"
        if not root.is_dir():
            return f"grep: no such directory: {args['path']}"
        flags = 0 if bool(args.get("case")) else re.IGNORECASE
        try:
            rx = re.compile(pattern, flags)
        except re.error as exc:
            return f"grep: invalid regex {pattern!r}: {exc}"
        matches: list[str] = []
        total_chars = 0
        try:
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
                for filename in filenames:
                    full = Path(dirpath) / filename
                    try:
                        with open(full, "r", encoding="utf-8", errors="replace") as f:
                            for lineno, line in enumerate(f, 1):
                                if rx.search(line):
                                    rel = full.relative_to(self.project_dir)
                                    entry = f"{rel}:{lineno}: {line.rstrip(chr(10) + chr(13))}"
                                    if matches:
                                        total_chars += 1  # the "\n" separator
                                    total_chars += len(entry)
                                    matches.append(entry)
                                    # The result is capped at MAX_RESULT_CHARS anyway;
                                    # stop scanning as soon as the join would exceed it.
                                    if total_chars > MAX_RESULT_CHARS:
                                        return _cap("\n".join(matches))
                    except OSError:
                        continue
        except OSError as exc:
            return f"grep: {exc}"
        if not matches:
            return f"no matches for {pattern}"
        return _cap("\n".join(matches))

    def _tool_glob(self, args: dict[str, Any]) -> str:
        tool = "glob"
        pattern = self._require(args, "pattern", tool)
        root = self.project_dir.resolve()
        if ".." in Path(pattern).parts:
            return f"blocked: path escapes project directory: {pattern}"
        try:
            matches = sorted(root.glob(pattern))
        except (ValueError, OSError) as exc:
            return f"glob: invalid pattern {pattern!r}: {exc}"
        lines: list[str] = []
        for match in matches:
            resolved = match.resolve()
            if resolved.is_relative_to(root):
                lines.append(str(resolved.relative_to(root)))
        return _cap("\n".join(lines))

    def _tool_write(self, args: dict[str, Any]) -> str:
        tool = "write"
        path = self._require(args, "path", tool)
        content = self._require(args, "content", tool)
        if not isinstance(content, str):
            content = str(content)
        target = resolve_relative(path, self.project_dir)
        data = content.encode("utf-8")
        try:
            target.write_bytes(data)
        except OSError as exc:
            return f"write: {exc}"
        return f"wrote {path} ({len(data)} bytes)"

    def _tool_edit(self, args: dict[str, Any]) -> str:
        tool = "edit"
        path = self._require(args, "path", tool)
        old_text = self._require(args, "old_text", tool)
        new_text = self._require(args, "new_text", tool)
        all_ = bool(args.get("all"))
        offset = self._int_arg(args, "offset", tool)
        limit = self._int_arg(args, "limit", tool)
        if limit is not None and limit < 0:
            raise ToolError(f"{tool}: limit must be >= 0")
        target = resolve_relative(path, self.project_dir)
        if not target.is_file():
            return f"edit: no such file: {path}"
        try:
            text = target.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            return f"edit: not valid UTF-8 text: {exc}"
        except OSError as exc:
            return f"edit: {exc}"
        if offset is not None or limit is not None:
            # Scope the exact-substring replace to the 1-based line range
            # [offset, offset+limit): split, replace within the range only,
            # splice the result back into the line list.
            lines = text.splitlines(keepends=True)
            lo = max(0, (offset or 1) - 1)
            hi = lo + limit if limit is not None else len(lines)
            range_text = "".join(lines[lo:hi])
            count = range_text.count(old_text)
            if count == 0:
                return "old_text not found"
            if count > 1 and not all_:
                return f"old_text matches {count} times; pass all=true to replace all"
            if all_:
                updated_range = range_text.replace(old_text, new_text)
            else:
                updated_range = range_text.replace(old_text, new_text, 1)
            lines[lo:hi] = [updated_range]
            updated = "".join(lines)
        else:
            count = text.count(old_text)
            if count == 0:
                return "old_text not found"
            if count > 1 and not all_:
                return f"old_text matches {count} times; pass all=true to replace all"
            if all_:
                updated = text.replace(old_text, new_text)
            else:
                updated = text.replace(old_text, new_text, 1)
        try:
            target.write_text(updated, encoding="utf-8")
        except OSError as exc:
            return f"edit: {exc}"
        if all_:
            return f"edited {path} ({count} replacements)"
        return f"edited {path}"

    def _tool_bash(self, args: dict[str, Any]) -> str:
        tool = "bash"
        command = self._require(args, "command", tool)
        if not isinstance(command, str):
            command = str(command)
        timeout = self._int_arg(args, "timeout", tool, default=30) or 30
        if timeout > 300:
            return f"bash: timeout must be at most 300 seconds (got {timeout})"
        if not self.allow_dangerous:
            for pattern in DENY_PATTERNS:
                if pattern.search(command):
                    return _DENY_MESSAGE
        # Sanitized environment: keep the ambient env but restrict PATH to the
        # project venv (when present) plus a minimal POSIX search path.
        env = dict(os.environ)
        path_dirs: list[str] = []
        venv_bin = self._project_dir / ".venv" / "bin"
        if venv_bin.is_dir():
            path_dirs.append(str(venv_bin))
        path_dirs.extend(["/usr/local/bin", "/usr/bin", "/bin"])
        env["PATH"] = os.pathsep.join(path_dirs)
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(self.project_dir),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return f"bash: timed out after {timeout}s"
        except OSError as exc:
            return f"bash: {exc}"
        output = (proc.stdout or "") + (proc.stderr or "")
        return _cap(output)

    def _tool_memory_append(self, args: dict[str, Any]) -> str:
        tool = "memory_append"
        section = self._require(args, "section", tool)
        text = self._require(args, "text", tool)
        if section not in _MEMORY_SECTIONS:
            return f"memory_append: invalid section: {section} (expected one of {', '.join(_MEMORY_SECTIONS)})"
        if self._memory is None:
            return "memory_append: no memory store configured"
        try:
            result = self._memory.append(section, text)
        except Exception as exc:  # noqa: BLE001 - the injected store is external
            raise ToolError(f"memory_append: memory store failed: {exc}") from exc
        return str(result)


# -- schemas ----------------------------------------------------------------

_TOOL_SPECS: list[tuple[str, str, dict[str, Any]]] = [
    (
        "read",
        "Read text from a file, or list a directory. For files, prefer `:N-M`-style line "
        "selectors: `offset` is the 1-based start line and `limit` is the number of lines to "
        "read (omit both to read the whole file). A directory path returns a depth-limited "
        "listing (about 2 levels; subdirectories end with '/'). Paths are resolved against the "
        "project directory; escaping paths are blocked.",
        {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File or directory path, relative to the project directory "
                    "(absolute paths inside it are allowed).",
                },
                "offset": {"type": "integer", "description": "1-based start line; omit to read from the top."},
                "limit": {"type": "integer", "description": "Number of lines to read; omit to read to the end."},
            },
            "required": ["path"],
        },
    ),
    (
        "grep",
        "Regex-search text files under `path` (default: the project directory), skipping "
        "`.git`, `__pycache__`, and `node_modules` directories. Case-insensitive by default; "
        "set `case: true` for case-sensitive matching. Returns matching lines as "
        "`path:line_number: text`. An invalid pattern returns an error string; no matches "
        "returns a short notice.",
        {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regular expression to search for."},
                "path": {
                    "type": "string",
                    "description": "Root to search under (default: the project directory).",
                },
                "case": {
                    "type": "boolean",
                    "description": "Set to true for a case-sensitive match (default: case-insensitive).",
                },
            },
            "required": ["pattern"],
        },
    ),
    (
        "glob",
        "Glob-match paths relative to the project directory (e.g. `src/**/*.py`), one path per "
        "line. Patterns that would escape the project directory (e.g. `../`) are blocked.",
        {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern, relative to the project directory.",
                }
            },
            "required": ["pattern"],
        },
    ),
    (
        "write",
        "Write UTF-8 text to a file inside the project directory, overwriting existing "
        "content. Returns `wrote <path> (<n> bytes)`. Escaping paths are blocked.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path, relative to the project directory."},
                "content": {"type": "string", "description": "UTF-8 text to write."},
            },
            "required": ["path", "content"],
        },
    ),
    (
        "edit",
        "Exact-substring edit of a file: replace `old_text` with `new_text`. A single match is "
        "replaced; zero matches return 'old_text not found'; multiple matches with `all` unset "
        "return an error listing the count. With `all: true`, every occurrence is replaced. "
        "Optionally scope the replacement to a line range: `offset` is the 1-based start line "
        "and `limit` is the number of lines covered; when given, only matches inside that "
        "range [offset, offset+limit) are considered. Escaping paths are blocked.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path, relative to the project directory."},
                "old_text": {"type": "string", "description": "Exact substring to replace."},
                "new_text": {"type": "string", "description": "Replacement text."},
                "offset": {
                    "type": "integer",
                    "description": "1-based start line of the replacement scope; omit to scope from the top of the file.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of lines in the replacement scope; omit to scope to the end of the file.",
                },
                "all": {
                    "type": "boolean",
                    "description": "Replace every occurrence (default: replace only the single match).",
                },
            },
            "required": ["path", "old_text", "new_text"],
        },
    ),
    (
        "bash",
        "Run a shell command in the project directory and return combined stdout and stderr "
        "(capped at 10000 chars). Runs in a sanitized environment with a minimal PATH (the "
        "project `.venv/bin` when present, plus `/usr/local/bin`, `/usr/bin`, `/bin`). "
        "`timeout` defaults to 30 seconds and may not exceed 300. Destructive commands (rm -rf, "
        "git push, git reset --hard, git clean -f, mkfs, dd, fork bombs, `> /dev/sd*`) are "
        "blocked by policy unless the registry allows dangerous commands.",
        {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run."},
                "timeout": {
                    "type": "integer",
                    "description": "Seconds before the command is killed; defaults to 30, max 300.",
                },
            },
            "required": ["command"],
        },
    ),
    (
        "memory_append",
        "Append a note to the project memory store under a fixed section, returning the memory "
        "file path or 'already recorded' when the note is a duplicate.",
        {
            "type": "object",
            "properties": {
                "section": {
                    "type": "string",
                    "enum": list(_MEMORY_SECTIONS),
                    "description": "Memory section to append to.",
                },
                "text": {"type": "string", "description": "Note text to record."},
            },
            "required": ["section", "text"],
        },
    ),
]
