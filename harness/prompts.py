"""System-prompt assembly (memory digest + project context)."""

from __future__ import annotations

import datetime
from pathlib import Path

FIXED_PREFIX: str = (
    "You are kaal — DeepSeek V4 Flash harness agent.\n"
    "\n"
    "Working rules:\n"
    "- Cite file paths with backticks (`src/foo.py`), never bare prose names.\n"
    "- Verify claims against actual file contents; do not trust remembered code.\n"
    "- Prefer targeted reads — directory listing, then grep, then a line-selected "
    "read — over whole-file reads.\n"
    "\n"
    "Tool use:\n"
    "When you need a fact or a file operation, call a tool. You may batch "
    "independent tool calls. The harness parses your DSML tool calls automatically.\n"
    "If you need a decision or information only the user has, call `ask_user`.\n"
    "For independent sub-tasks, delegate with `spawn_agent` or `spawn_parallel_task` "
    "and synthesize their JSON summaries into your answer.\n"
    "\n"
    "Output contract:\n"
    "Final answers are plain text. Never emit tool markup, `reasoning_content`, "
    "or `<think>` blocks in your visible answer.\n"
    "\n"
    "Boundaries:\n"
    "Destructive bash commands are blocked; ask the user instead.\n"
    "\n"
    "Memory:\n"
    "Project memory lives in `.agent-memory/`; after recording a decision or a "
    "lesson, call `memory_append` to persist it.\n"
    "\n"
    "Tool schemas are provided only in the API `tools` parameter, never in prose."
)


def build_system_prompt(
    memory_digest: str, project_context: str, agent: dict | None = None
) -> str:
    """Assemble the full system prompt: fixed prefix + memory guidance + project.

    When ``agent`` is given (a ``{name, description}`` persona dict from
    harness.agents), a third `## Agent` block is appended telling the model
    to adopt that persona fully. Tier 3 is empty by design: per-turn content
    is the conversation itself, and tool schemas travel only in the API
    `tools` parameter, never in prose.
    """
    dynamic = (
        f"## Memory Guidance\n\n{memory_digest}\n\n## Project\n\n{project_context}"
    )
    if agent:
        name = agent.get("name", "")
        description = agent.get("description", "")
        dynamic += (
            f"\n\n## Agent\n"
            f"You are operating as **{name}** — {description}.\n"
            f"Adopt this persona fully: let {name}'s strengths shape how you "
            f"approach the task."
        )
    return "\n\n".join([FIXED_PREFIX, dynamic])


def build_project_context(cwd: Path | str) -> str:
    """Short project context: today's date, the absolute cwd, and AGENTS.md.

    When `<cwd>/AGENTS.md` exists its first 200 lines are included under a
    `## AGENTS.md (first 200 lines)` heading; otherwise a line notes its
    absence. The regenerable structure cache (`.kaal/STRUCTURE.md`, if present)
    is appended under `## Project structure` — read only, never scanned here
    (fast reopen is the point).
    """
    cwd = Path(cwd).resolve()
    lines = [f"Date: {datetime.date.today().isoformat()}", f"CWD: {cwd}"]
    agents = cwd / "AGENTS.md"
    if agents.is_file():
        lines.append("## AGENTS.md (first 200 lines)")
        lines.extend(agents.read_text(encoding="utf-8").splitlines()[:200])
    else:
        lines.append("No AGENTS.md found in this project.")
    structure = cwd / ".kaal" / "STRUCTURE.md"
    if structure.is_file():
        lines.append("## Project structure")
        try:
            text = structure.read_text(encoding="utf-8")
        except OSError:
            text = ""
        lines.extend(text.splitlines()[:120])
        lines.append("(full: .kaal/STRUCTURE.md — re-read it if the files change)")
    else:
        lines.append("No structure cache yet (.kaal/STRUCTURE.md missing)")
    return "\n".join(lines)
