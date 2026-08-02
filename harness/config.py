"""Constants and API-key resolution for the opencode-go gateway.

The wire contract below was verified against the omp bundled model catalog
(`@oh-my-pi/pi-catalog/src/models.json`, the `opencode-go/deepseek-v4-flash`
compat block). Do not change any constant without re-verifying there.

API-key resolution order: env `OPENCODE_API_KEY` → user key store
(`user_key_path()`, written by the TUI's `/connect`) → omp auth store
(`~/.omp/agent/agent.db`) → hard error with instructions.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path

# Gateway (OpenAI-compatible chat completions).
BASE_URL = "https://opencode.ai/zen/go/v1"
CHAT_COMPLETIONS_URL = f"{BASE_URL}/chat/completions"
MODEL_ID = "deepseek-v4-flash"

# Model limits (catalog: contextWindow / maxTokens).
CONTEXT_WINDOW = 1_000_000
MAX_OUTPUT_TOKENS = 384_000

# Reasoning is streamed in this delta field (catalog: reasoningContentField).
REASONING_FIELD = "reasoning_content"

# Providers whose DSML envelopes leak into `delta.content` and must be healed
# client-side (catalog: DSML healing gate in compat/openai.ts).
DSML_HEALING_PROVIDERS = {"opencode-go"}

# Request timeout per HTTP request, seconds.
REQUEST_TIMEOUT = 120


def user_key_path() -> Path:
    """Path of the user API-key store (`$XDG_CONFIG_HOME/harnessdp/api_key`).

    Windows uses `%APPDATA%\harnessdp\api_key`; POSIX falls back to
    `~/.config` when `XDG_CONFIG_HOME` is unset.
    """
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "harnessdp" / "api_key"


def save_user_api_key(key: str) -> None:
    """Persist the user's API key: raw text, no trailing newline, 0600 on POSIX."""
    path = user_key_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(key, encoding="utf-8")
    if os.name != "nt":
        os.chmod(path, 0o600)


def load_user_api_key() -> str | None:
    """The stored user key, or None. Never printed or cached."""
    path = user_key_path()
    if not path.exists():
        return None
    try:
        key = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return key or None


def get_api_key() -> str:
    """Resolve the gateway API key: env, then the user key store, then omp's
    auth store, then fail. Never writes or caches the key.
    """
    env_key = os.environ.get("OPENCODE_API_KEY")
    if env_key:
        return env_key

    user_key = load_user_api_key()
    if user_key:
        return user_key

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
        if row:
            key = json.loads(row[0]).get("key")
            if key:
                return key
    except Exception:
        pass  # fall through to the hard error below

    print(
        "hdp: no API key found. Set OPENCODE_API_KEY, run `hdp` and use "
        "/connect, or re-add the opencode-go credential (`opencode` / "
        "`omp /connect`). Checked: env OPENCODE_API_KEY, "
        f"{user_key_path()}, and ~/.omp/agent/agent.db.",
        file=sys.stderr,
    )
    raise SystemExit(1)
