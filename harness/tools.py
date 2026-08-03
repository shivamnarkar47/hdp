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
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

MAX_RESULT_CHARS = 10_000
TRUNCATED_SUFFIX = "…[truncated]"

# Tools whose results are pure reads of the project tree: safe to cache.
_CACHEABLE_TOOLS = frozenset({"read", "grep", "glob"})
# Tools that mutate state (files or memory): a batch containing any of these
# bypasses cache lookups entirely (Finding 3 — same-step write-then-read
# staleness) and invalidates the cache after the batch.
_MUTATOR_TOOLS = frozenset({"write", "edit", "bash", "memory_append"})

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

_SKIP_DIRS = {".git", "__pycache__", "node_modules", ".kaal"}
_MEMORY_SECTIONS = ("project-state", "decisions", "patterns", "lessons-learned")
_DENY_MESSAGE = "blocked by harness policy (destructive command)"


class ToolError(Exception):
    """Hard failure: unknown tool, non-dict args, or an internal error."""


def _cap(text: str) -> str:
    """Truncate result strings longer than MAX_RESULT_CHARS."""
    if len(text) > MAX_RESULT_CHARS:
        return text[:MAX_RESULT_CHARS] + TRUNCATED_SUFFIX
    return text


def _format_rg_line(line: str) -> str:
    """Normalize an rg output line to the harness 'path:lineno: text' format.

    Real ripgrep emits 'path:lineno:text' (no space after the line number)
    and prefixes paths with './' when the search root is '.'. Strip the
    prefix and insert the separator space so the result matches the
    pure-Python scan byte for byte.
    """
    if line.startswith("./"):
        line = line[2:]
    path, _, rest = line.partition(":")
    lineno, _, text = rest.partition(":")
    return f"{path}:{lineno}: {text}"


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

    def __init__(
        self, memory=None, project_dir=None, allow_dangerous=False, cache=None
    ) -> None:
        self._memory = memory
        self._project_dir = (
            Path(project_dir).resolve() if project_dir is not None else Path.cwd().resolve()
        )
        self.allow_dangerous = allow_dangerous
        self._cache = cache
        # Per-batch cache state set by the loop via begin_batch/end_batch:
        # the structure signature for the current batch (None = caching off,
        # e.g. when structure is unavailable) and whether this batch contains
        # a mutator (which disables cache lookups for the WHOLE batch).
        self._cache_signature: str | None = None
        self._batch_has_mutator = False
        # Cache lookup counters (hit/miss; bypassed lookups count as neither).
        self._cache_hits = 0
        self._cache_misses = 0
        self._cached_schemas: list[dict] | None = None
        self._handlers: dict[str, Callable[[dict[str, Any]], str]] = {
            "read": self._tool_read,
            "grep": self._tool_grep,
            "glob": self._tool_glob,
            "write": self._tool_write,
            "edit": self._tool_edit,
            "bash": self._tool_bash,
            "memory_append": self._tool_memory_append,
            "spawn_agent": self._tool_spawn_agent,
            "ask_user": self._tool_ask_user,
            "spawn_parallel_task": self._tool_spawn_parallel_task,
        }
        # Nested-agent runners injected by AgentLoop after construction
        # (Finding 6: cli.py builds the registry before the loop). None when
        # the registry is used standalone — spawn_agent / spawn_parallel_task
        # then return an error string instead of running anything.
        self._spawn_handler: Callable[[str, str | None, int, int], str] | None = None
        self._spawn_many_handler: Callable[[list[dict[str, Any]], int], str] | None = None
        # ask_user answer provider injected by AgentLoop (default: read stdin).
        # None when the registry is used standalone — ask_user then returns an
        # error string instead of blocking.
        self._ask_handler: Callable[[str, list[str] | None], str] | None = None

    @property
    def project_dir(self) -> Path:
        return self._project_dir

    def set_spawn_handler(
        self, handler: Callable[[str, str | None, int, int], str] | None
    ) -> None:
        """Inject the nested-agent runner used by the spawn_agent tool.

        AgentLoop calls this with its own ``_spawn`` after construction
        (Finding 6: cli.py builds the registry before the loop). Without a
        handler — a standalone registry — spawn_agent returns
        ``spawn_agent: not available in this context``.
        """
        self._spawn_handler = handler

    def set_spawn_many_handler(
        self, handler: Callable[[list[dict[str, Any]], int], str] | None
    ) -> None:
        """Inject the parallel nested-agent runner used by spawn_parallel_task.

        Mirrors :meth:`set_spawn_handler`: AgentLoop wires its own
        ``_spawn_many`` after construction. Without a handler — a standalone
        registry — spawn_parallel_task returns
        ``spawn_parallel_task: not available in this context``.
        """
        self._spawn_many_handler = handler

    def set_ask_handler(
        self, handler: Callable[[str, list[str] | None], str] | None
    ) -> None:
        """Inject the ask_user answer provider (the user-in-the-loop tool).

        AgentLoop wires this in ``run()`` with its default headless handler
        (print the question, read a line from stdin) or the caller's handler
        (CLI batch: refuse; TUI: a modal). Without a handler — a standalone
        registry — ask_user returns ``ask_user: not available in this context``.
        """
        self._ask_handler = handler

    def schemas(self) -> list[dict]:
        """OpenAI function-call schemas for every tool.

        The handler set (and therefore ``_TOOL_SPECS``) is fixed after
        ``__init__``, so the deep-copied schema list is built once and
        memoized. Each call returns a fresh top-level list — a caller that
        mutates the list cannot corrupt the cache; the inner dicts are never
        mutated by callers and are shared across calls.
        """
        if self._cached_schemas is None:
            self._cached_schemas = [
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
        return list(self._cached_schemas)

    def execute(self, name: str, args: dict) -> str:
        """Run tool ``name`` with ``args``; return a result string or raise ToolError."""
        if not isinstance(args, dict):
            raise ToolError(f"tool {name}: arguments must be a dict")
        handler = self._handlers.get(name)
        if handler is None:
            raise ToolError(f"unknown tool: {name}")
        # Read-only tools consult the tool-result cache when a structure
        # signature is available AND the batch contains no mutator (the
        # same-step write-then-read staleness hole). Cached values were stored
        # as result strings, so a hit returns directly without the handler.
        if (
            self._cache is not None
            and self._cache_signature is not None
            and not self._batch_has_mutator
            and name in _CACHEABLE_TOOLS
        ):
            args_json = json.dumps(args, sort_keys=True)
            cached = self._cache.get(name, args_json, self._cache_signature)
            if cached is not None:
                self._cache_hits += 1
                return cached
            self._cache_misses += 1
            result = self._run_handler(handler, name, args)
            self._cache.put(name, args_json, self._cache_signature, result)
            return result
        return self._run_handler(handler, name, args)

    # -- cache visibility ---------------------------------------------------

    @property
    def cache_hits(self) -> int:
        """Number of cache lookups that hit (served a stored result)."""
        return self._cache_hits

    @property
    def cache_misses(self) -> int:
        """Number of cache lookups that missed (ran the handler instead)."""
        return self._cache_misses

    def cache_hit_rate(self) -> float | None:
        """Fraction of lookups that hit, or None when no lookups occurred."""
        lookups = self._cache_hits + self._cache_misses
        if lookups == 0:
            return None
        return self._cache_hits / lookups

    @staticmethod
    def _run_handler(
        handler: Callable[[dict[str, Any]], str], name: str, args: dict[str, Any]
    ) -> str:
        try:
            return handler(args)
        except ToolError as exc:
            # Tool-level failures (blocked paths, missing/invalid args) are
            # surfaced as result strings; unknown tools and non-dict args are
            # raised above. Unexpected exceptions are internal errors.
            return str(exc)
        except Exception as exc:  # noqa: BLE001 - wrap as an internal error
            raise ToolError(f"tool {name} failed: {exc}") from exc

    # -- per-batch cache state ----------------------------------------------

    def begin_batch(self, names: list[str], structure_sig: str | None) -> None:
        """Open a tool batch: pin the structure signature and detect mutators.

        Called by the loop before each batch. A batch containing any mutator
        disables cache lookups for the entire step, closing the same-step
        write-then-read staleness hole. No-op when no cache is configured.
        """
        if self._cache is None:
            return
        self._cache_signature = structure_sig
        self._batch_has_mutator = any(name in _MUTATOR_TOOLS for name in names)

    def end_batch(self, mutated: bool) -> None:
        """Close a tool batch: drop the cache after a mutating batch.

        ``mutated`` is the loop's own determination (write/edit/bash). A
        mutating batch refreshes the structure afterwards, which changes the
        signature anyway — the drop just retires stale entries immediately.
        A non-mutating batch only clears the batch flag. No-op with no cache.
        """
        if self._cache is None:
            return
        if mutated:
            self._cache.drop()
        self._batch_has_mutator = False
        self._cache_signature = None

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
        case = bool(args.get("case"))
        if pattern:
            # rg first when available; None means "fall back" (missing binary,
            # rg error such as an unsupported regex feature, or OSError).
            result = self._grep_rg(pattern, root, case)
            if result is not None:
                return result
        return self._grep_python(pattern, root, case)

    def _grep_rg(self, pattern: str, root: Path, case: bool) -> str | None:
        """rg-backed grep; returns a result string, or None to fall back.

        Streams rg's stdout so scanning stops as soon as the result reaches
        MAX_RESULT_CHARS — files after the cap are never read, matching the
        pure-Python path. ``--sort-files`` makes the traversal order
        deterministic (paths sorted), which is what guarantees the post-cap
        files never appear regardless of filesystem readdir order.
        """
        if shutil.which("rg") is None:
            return None
        cmd = [
            "rg",
            "--line-number",
            "--no-heading",
            "--color",
            "never",
            "--sort-files",
        ]
        if not case:
            cmd.append("-i")
        cmd.extend(
            [
                "--glob",
                "!.git/**",
                "--glob",
                "!node_modules/**",
                "--glob",
                "!__pycache__/**",
                "--glob",
                "!.kaal/**",
                "--",
                pattern,
                str(root.relative_to(self.project_dir)),
            ]
        )
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(self.project_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except OSError:
            return None
        assert proc.stdout is not None
        matches: list[str] = []
        total_chars = 0
        try:
            for line in proc.stdout:
                entry = _format_rg_line(line.rstrip("\n"))
                if matches:
                    total_chars += 1  # the "\n" separator
                total_chars += len(entry)
                matches.append(entry)
                # Same cap semantics as the Python scan: stop the moment the
                # join would exceed MAX_RESULT_CHARS; the process is reaped in
                # the finally (stdout close triggers EPIPE, then SIGTERM).
                if total_chars > MAX_RESULT_CHARS:
                    return _cap("\n".join(matches))
        finally:
            proc.stdout.close()
            if proc.poll() is None:
                proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        if matches:
            # Best effort: rg may have exited non-zero mid-stream (e.g. an
            # error on a later file); keep whatever matched before the failure.
            return _cap("\n".join(matches))
        if proc.returncode in (0, 1):
            return f"no matches for {pattern}"
        return None  # rg error (exit 2) or other non-zero -> Python fallback

    def _grep_python(self, pattern: str, root: Path, case: bool) -> str:
        """Reference pure-Python grep scan; also the rg fallback."""
        flags = 0 if case else re.IGNORECASE
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

    def _tool_spawn_agent(self, args: dict[str, Any]) -> str:
        tool = "spawn_agent"
        task = self._require(args, "task", tool)
        if not isinstance(task, str):
            task = str(task)
        if self._spawn_handler is None:
            return "spawn_agent: not available in this context"
        max_steps = self._int_arg(args, "max_steps", tool, default=5)
        timeout = self._int_arg(args, "timeout", tool, default=120)
        # Defensive clamping: the schema already limits these ranges, but a
        # stray value must not crash the nested loop (at most 5 steps, 300s).
        max_steps = max(1, min(int(max_steps), 5))
        timeout = max(1, min(int(timeout), 300))
        return self._spawn_handler(task, args.get("dir"), max_steps, timeout)

    def _tool_ask_user(self, args: dict[str, Any]) -> str:
        tool = "ask_user"
        question = self._require(args, "question", tool)
        if not isinstance(question, str):
            question = str(question)
        if self._ask_handler is None:
            return "ask_user: not available in this context"
        options = args.get("options")
        if options is not None and (
            not isinstance(options, list)
            or not all(isinstance(option, str) for option in options)
        ):
            raise ToolError(f"{tool}: options must be an array of strings")
        return str(self._ask_handler(question, options))

    def _tool_spawn_parallel_task(self, args: dict[str, Any]) -> str:
        tool = "spawn_parallel_task"
        tasks = self._require(args, "tasks", tool)
        if not isinstance(tasks, list) or not tasks:
            raise ToolError(f"{tool}: tasks must be a non-empty array")
        if self._spawn_many_handler is None:
            return "spawn_parallel_task: not available in this context"
        timeout = self._int_arg(args, "timeout", tool, default=120) or 120
        timeout = max(1, min(int(timeout), 300))
        clean: list[dict[str, Any]] = []
        for index, task in enumerate(tasks):
            if not isinstance(task, dict):
                raise ToolError(f"{tool}: tasks[{index}] must be an object")
            task_text = task.get("task")
            if not isinstance(task_text, str) or not task_text:
                raise ToolError(f"{tool}: tasks[{index}]: missing required argument: task")
            max_steps = task.get("max_steps")
            if max_steps is not None:
                try:
                    max_steps = int(max_steps)
                except (TypeError, ValueError):
                    raise ToolError(
                        f"{tool}: tasks[{index}]: invalid max_steps: {max_steps!r}"
                    ) from None
                max_steps = max(1, min(max_steps, 5))
            else:
                max_steps = 5
            task_timeout = task.get("timeout")
            if task_timeout is not None:
                try:
                    task_timeout = int(task_timeout)
                except (TypeError, ValueError):
                    raise ToolError(
                        f"{tool}: tasks[{index}]: invalid timeout: {task_timeout!r}"
                    ) from None
                task_timeout = max(1, min(task_timeout, 300))
            else:
                task_timeout = timeout
            clean.append(
                {
                    "task": task_text,
                    "dir": task.get("dir"),
                    "max_steps": max_steps,
                    "timeout": task_timeout,
                }
            )
        return self._spawn_many_handler(clean, timeout)


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
    (
        "spawn_agent",
        "Run a nested kaal agent on a sub-task and return its JSON summary "
        "{answer, steps, usage, session_id}; answer capped at 50000 chars.",
        {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The sub-task for the nested agent to complete.",
                },
                "dir": {
                    "type": "string",
                    "description": "Sub-project directory for the nested agent (default: the "
                    "current project directory; escaping paths are blocked).",
                },
                "max_steps": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "description": "Maximum turns for the nested agent (default: 5).",
                },
                "timeout": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 300,
                    "description": "Wall-clock seconds before the nested run is abandoned "
                    "(default: 120).",
                },
            },
            "required": ["task"],
        },
    ),
    (
        "ask_user",
        "Ask the user a question and wait for their answer. Use when you need a "
        "decision, a confirmation, or information only the user has. The answer "
        "is returned as the tool result.",
        {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask the user.",
                },
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional choices; when given, the user picks one.",
                },
            },
            "required": ["question"],
        },
    ),
    (
        "spawn_parallel_task",
        "Run several independent sub-tasks in parallel as nested kaal agents. "
        "Each task: {task: string, dir?: string, max_steps?: int (1-5), "
        "timeout?: int (1-300)}. Returns a JSON array of "
        "{index, answer, steps, usage, session_id, error?}.",
        {
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "description": "Sub-tasks to run concurrently; results come back in this order.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string",
                                "description": "The sub-task for the nested agent.",
                            },
                            "dir": {
                                "type": "string",
                                "description": "Sub-project directory for the nested agent (default: the "
                                "current project directory; escaping paths are blocked).",
                            },
                            "max_steps": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 5,
                                "description": "Maximum turns for the nested agent (default: 5).",
                            },
                            "timeout": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 300,
                                "description": "Wall-clock seconds before the nested run is abandoned "
                                "(default: the tool-level timeout).",
                            },
                        },
                        "required": ["task"],
                    },
                },
                "timeout": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 300,
                    "description": "Default per-task wall-clock seconds (default: 120).",
                },
            },
            "required": ["tasks"],
        },
    ),
]
