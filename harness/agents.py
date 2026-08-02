"""Agent personas: definitions + persistence (`.hdp/agents.json`).

The harness ships with five default personas — the Pandavas of the
Mahabharata — each a distinctive way of approaching a task. Users can
activate one for a session (the persona is injected into the system prompt),
create new ones in the TUI (`/agents`), and have the model invent one via
Ctrl+A. State lives in ``<project_dir>/.hdp/agents.json`` (git-ignored via
``.hdp/``) as ``{"agents": [ {name, description}, ... ], "active": name|None}``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# The five Pandava personas — the default cast. Exactly five, names and
# descriptions distinctive; each offers a genuinely different working style.
DEFAULT_AGENTS: list[dict] = [
    {
        "name": "Yudhishthira",
        "description": (
            "the Dharma Architect: principled, correctness-first architecture; "
            "deliberate planning, ethics of the codebase; never cuts corners"
        ),
    },
    {
        "name": "Bhima",
        "description": (
            "the Mighty Performer: brute-force execution; heavy refactors, big "
            "sweeps, gets the job done with overwhelming force"
        ),
    },
    {
        "name": "Arjuna",
        "description": (
            "the Precise Marksman: surgical precision; focused bug-hunting, "
            "minimal diffs, one clean shot at the target"
        ),
    },
    {
        "name": "Nakula",
        "description": (
            "the Graceful Stylist: beauty and polish; UI/UX, naming, "
            "formatting, code that reads like poetry"
        ),
    },
    {
        "name": "Sahadeva",
        "description": (
            "the Wise Strategist: the whole board; architecture strategy, "
            "tradeoffs, docs, sees moves ahead"
        ),
    },
]


def _path(root: Path) -> Path:
    return Path(root) / ".hdp" / "agents.json"


def _seeded() -> dict:
    """A fresh state dict: the five defaults, no active agent."""
    return {"agents": [dict(a) for a in DEFAULT_AGENTS], "active": None}


def _valid(data: object) -> bool:
    """True when `data` is a dict with a sane agents list + active field.

    Tolerant on shape (load never crashes); the entries only need a truthy
    `name` — description is optional for user-created agents.
    """
    if not isinstance(data, dict):
        return False
    agents = data.get("agents")
    if not isinstance(agents, list) or not all(
        isinstance(a, dict) and a.get("name") for a in agents
    ):
        return False
    active = data.get("active")
    return active is None or isinstance(active, str)


def load(root: Path) -> dict:
    """Load agent state; missing/corrupt file seeds the defaults, active None.

    Read-only: a missing file is NOT written on load — the file only appears
    once something is saved (an activation or a new agent).
    """
    path = _path(root)
    if not path.is_file():
        return _seeded()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return _seeded()
    if not _valid(raw):
        return _seeded()
    return {"agents": [dict(a) for a in raw["agents"]], "active": raw.get("active")}


def save(root: Path, data: dict) -> None:
    """Persist agent state atomically (temp file + os.replace, like structure.py)."""
    path = _path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def active_agent(data: dict) -> dict | None:
    """Resolve the active agent dict by name; None for no/bad active name."""
    active = data.get("active")
    if not active:
        return None
    for agent in data.get("agents", []):
        if agent.get("name") == active:
            return agent
    return None


def agent_names(data: dict) -> list[str]:
    """The names of every agent in state, in list order."""
    return [a.get("name", "") for a in data.get("agents", [])]
