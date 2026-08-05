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
# The opencode free tier lives on the zen/v1 endpoint (same OPENCODE_API_KEY);
# catalog models that list this base_url route there.
FREE_BASE_URL = "https://opencode.ai/zen/v1"
MODEL_ID = "deepseek-v4-flash"

# Model limits (catalog: contextWindow / maxTokens).
# MAX_OUTPUT_TOKENS is deliberately capped well below the catalog's 384k:
# an unbounded output budget lets the model's reasoning run away (the
# dominant cost of a slow response). 32k ≈ 96k chars — still room for big
# tool-call payloads; raise only if large file writes truncate.
CONTEXT_WINDOW = 1_000_000
MAX_OUTPUT_TOKENS = 32_000

# Reasoning is streamed in this delta field (catalog: reasoningContentField).
REASONING_FIELD = "reasoning_content"

# Providers whose DSML envelopes leak into `delta.content` and must be healed
# client-side (catalog: DSML healing gate in compat/openai.ts).
DSML_HEALING_PROVIDERS = {"opencode-go"}

# Request timeout per HTTP request, seconds.
REQUEST_TIMEOUT = 120

# Catalog pricing (verified against @oh-my-pi/pi-catalog models.json, the
# opencode-go/deepseek-v4-flash compat block): input $0.14/M, output $0.28/M,
# cache-read $0.0028/M. Cache-read tokens are NOT received client-side
# (stream_options rejected), so callers always pass cache_read_tokens=0.
INPUT_COST_PER_M = 0.14
OUTPUT_COST_PER_M = 0.28
CACHE_READ_COST_PER_M = 0.0028

# Model catalog — verified against the local opencode cache
# (~/.cache/opencode/models.json; prices in USD per 1M tokens).
# Sorted free first, then ascending by price. deepseek-v4-flash is the
# shipped default (MODEL_ID). Paid models route to BASE_URL (opencode-go);
# free-tier entries route to FREE_BASE_URL (the opencode provider, same
# OPENCODE_API_KEY, cost 0).
MODELS: list[dict] = [
    # -- opencode free tier (zen/v1, $0) --
    {"id": "deepseek-v4-flash-free", "name": "DeepSeek V4 Flash (Free)", "input_per_m": 0.0, "output_per_m": 0.0, "base_url": FREE_BASE_URL},
    {"id": "qwen3.6-plus-free", "name": "Qwen3.6 Plus (Free)", "input_per_m": 0.0, "output_per_m": 0.0, "base_url": FREE_BASE_URL},
    {"id": "kimi-k2.5-free", "name": "Kimi K2.5 (Free)", "input_per_m": 0.0, "output_per_m": 0.0, "base_url": FREE_BASE_URL},
    {"id": "glm-5-free", "name": "GLM-5 (Free)", "input_per_m": 0.0, "output_per_m": 0.0, "base_url": FREE_BASE_URL},
    {"id": "hy3-free", "name": "Hy3 (Free)", "input_per_m": 0.0, "output_per_m": 0.0, "base_url": FREE_BASE_URL},
    {"id": "minimax-m2.5-free", "name": "MiniMax-M2.5 (Free)", "input_per_m": 0.0, "output_per_m": 0.0, "base_url": FREE_BASE_URL},
    {"id": "mimo-v2.5-free", "name": "MiMo V2.5 (Free)", "input_per_m": 0.0, "output_per_m": 0.0, "base_url": FREE_BASE_URL},
    {"id": "mimo-v2-pro-free", "name": "MiMo V2 Pro (Free)", "input_per_m": 0.0, "output_per_m": 0.0, "base_url": FREE_BASE_URL},
    {"id": "minimax-m3-free", "name": "MiniMax-M3 (Free)", "input_per_m": 0.0, "output_per_m": 0.0, "base_url": FREE_BASE_URL},
    {"id": "glm-4.7-free", "name": "GLM-4.7 (Free)", "input_per_m": 0.0, "output_per_m": 0.0, "base_url": FREE_BASE_URL},
    {"id": "nemotron-3-super-free", "name": "Nemotron 3 Super (Free)", "input_per_m": 0.0, "output_per_m": 0.0, "base_url": FREE_BASE_URL},
    {"id": "north-mini-code-free", "name": "North Mini Code (Free)", "input_per_m": 0.0, "output_per_m": 0.0, "base_url": FREE_BASE_URL},
    # -- paid, ascending --
    {"id": "gpt-5.6-luna", "name": "GPT-5.6 Luna", "input_per_m": 0.1, "output_per_m": 0.6},
    {"id": "deepseek-v4-flash", "name": "DeepSeek V4 Flash", "input_per_m": 0.14, "output_per_m": 0.28},
    {"id": "mimo-v2.5", "name": "MiMo V2.5", "input_per_m": 0.14, "output_per_m": 0.28},
    {"id": "hy3", "name": "Hy3", "input_per_m": 0.14, "output_per_m": 0.58},
    {"id": "qwen3.5-plus", "name": "Qwen3.5 Plus", "input_per_m": 0.2, "output_per_m": 1.2},
    {"id": "minimax-m2.5", "name": "MiniMax-M2.5", "input_per_m": 0.3, "output_per_m": 1.2},
    {"id": "minimax-m2.7", "name": "MiniMax-M2.7", "input_per_m": 0.3, "output_per_m": 1.2},
    {"id": "minimax-m3", "name": "MiniMax-M3", "input_per_m": 0.3, "output_per_m": 1.2},
    {"id": "qwen3.7-plus", "name": "Qwen3.7 Plus", "input_per_m": 0.4, "output_per_m": 1.6},
    {"id": "mimo-v2-omni", "name": "MiMo V2 Omni", "input_per_m": 0.4, "output_per_m": 2.0},
    {"id": "deepseek-v4-pro", "name": "DeepSeek V4 Pro", "input_per_m": 0.435, "output_per_m": 0.87},
    {"id": "qwen3.6-plus", "name": "Qwen3.6 Plus", "input_per_m": 0.5, "output_per_m": 3.0},
    {"id": "kimi-k2.5", "name": "Kimi K2.5", "input_per_m": 0.6, "output_per_m": 3.0},
    {"id": "kimi-k2.6", "name": "Kimi K2.6", "input_per_m": 0.95, "output_per_m": 4.0},
    {"id": "kimi-k2.7-code", "name": "Kimi K2.7 Code", "input_per_m": 0.95, "output_per_m": 4.0},
    {"id": "mimo-v2-pro", "name": "MiMo V2 Pro", "input_per_m": 1.0, "output_per_m": 3.0},
    {"id": "glm-5", "name": "GLM-5", "input_per_m": 1.0, "output_per_m": 3.2},
    {"id": "glm-5.1", "name": "GLM-5.1", "input_per_m": 1.4, "output_per_m": 4.4},
    {"id": "glm-5.2", "name": "GLM-5.2", "input_per_m": 1.4, "output_per_m": 4.4},
    {"id": "qwen3.8-max", "name": "Qwen3.8 Max", "input_per_m": 2.0, "output_per_m": 6.0},
    {"id": "grok-4.5", "name": "Grok 4.5", "input_per_m": 2.0, "output_per_m": 6.0},
    {"id": "qwen3.7-max", "name": "Qwen3.7 Max", "input_per_m": 2.5, "output_per_m": 7.5},
    {"id": "kimi-k3", "name": "Kimi K3", "input_per_m": 3.0, "output_per_m": 15.0},
]


def model_rates(model_id: str) -> tuple[float, float]:
    """(input_per_m, output_per_m) for a catalog model; falls back to the
    deepseek-v4-flash rates for unknown ids (the historical default)."""
    for model in MODELS:
        if model["id"] == model_id:
            return model["input_per_m"], model["output_per_m"]
    return 0.14, 0.28


def model_base_url(model_id: str) -> str:
    """The gateway endpoint for a model: free-tier entries route to
    FREE_BASE_URL (opencode zen/v1), everything else to BASE_URL."""
    for model in MODELS:
        if model["id"] == model_id and model.get("base_url"):
            return model["base_url"]
    return BASE_URL


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    model_id: str | None = None,
) -> float:
    """Estimated dollar cost for a token mix (per-1M catalog rates).

    ``model_id`` uses that model's own rates from MODELS; None keeps the
    deepseek-v4-flash defaults (backward-compatible). Free-tier models
    estimate to zero. Rounded to 6 decimals so the sum of the three per-1M
    terms is exact for whole-M token counts (0.14 + 0.28 would otherwise
    drift to 0.42000000000000004).
    """
    if model_id:
        input_per_m, output_per_m = model_rates(model_id)
    else:
        input_per_m, output_per_m = INPUT_COST_PER_M, OUTPUT_COST_PER_M
    return round(
        input_tokens / 1_000_000 * input_per_m
        + output_tokens / 1_000_000 * output_per_m
        + cache_read_tokens / 1_000_000 * CACHE_READ_COST_PER_M,
        6,
    )


def user_model_path() -> Path:
    r"""Path of the user model preference (`$XDG_CONFIG_HOME/kaal/model`)."""
    base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "kaal" / "model"


def save_user_model(model_id: str) -> None:
    """Persist the default model id: raw text, no trailing newline."""
    path = user_model_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(model_id, encoding="utf-8")
    os.replace(tmp, path)


def load_user_model() -> str | None:
    """The stored default model id, or None. Never cached."""
    try:
        model = user_model_path().read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return model or None


def resolve_model_id(flag: str | None) -> str:
    """Model selection order: --model flag > saved default > MODEL_ID."""
    return flag or load_user_model() or MODEL_ID


def user_key_path() -> Path:
    r"""Path of the user API-key store (`$XDG_CONFIG_HOME/kaal/api_key`).

    Windows uses `%APPDATA%\kaal\api_key`; POSIX falls back to
    `~/.config` when `XDG_CONFIG_HOME` is unset.
    """
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "kaal" / "api_key"


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
        "kaal: no API key found. Set OPENCODE_API_KEY, run `kaal` and use "
        "/connect, or re-add the opencode-go credential (`opencode` / "
        "`omp /connect`). Checked: env OPENCODE_API_KEY, "
        f"{user_key_path()}, and ~/.omp/agent/agent.db.",
        file=sys.stderr,
    )
    raise SystemExit(1)
