"""Token estimation and context-budget bookkeeping."""

from __future__ import annotations

import json
from typing import Any

from harness.config import CONTEXT_WINDOW, MAX_OUTPUT_TOKENS
from harness.messages import Message, SystemMessage, UserMessage


def estimate_tokens(text: str) -> int:
    """Rough token estimate: one token per three characters."""
    return len(text) // 3


def context_budget(prompt_tokens: int, max_output_tokens: int = MAX_OUTPUT_TOKENS) -> bool:
    """True if the prompt fits alongside the requested max output."""
    return prompt_tokens <= CONTEXT_WINDOW - max_output_tokens


def _wire_token_count(messages: list[Message]) -> int:
    total = 0
    for message in messages:
        total += estimate_tokens(json.dumps(message.to_wire(), ensure_ascii=False))
    return total


def truncate_history(
    messages: list[Message], system_message: SystemMessage, max_prompt_tokens: int
) -> list[Message]:
    """Drop oldest user+assistant+tool-result triples until the history fits.

    Never drops the system message, the last user message, or the turn that
    follows it (its assistant reply and tool results).
    """
    # Keep exactly one system message, at the front.
    result = [m for m in messages if not isinstance(m, SystemMessage)]

    while _wire_token_count(result) > max_prompt_tokens:
        # Oldest droppable turn: the first UserMessage that is not the last one.
        # Recompute the last-user index each round (deletions shift it).
        last_user = max(i for i, m in enumerate(result) if isinstance(m, UserMessage))
        candidates = [
            i for i, m in enumerate(result) if isinstance(m, UserMessage) and i != last_user
        ]
        if not candidates:
            break
        start = candidates[0]
        end = next(
            (i for i in range(start + 1, len(result)) if isinstance(result[i], UserMessage)),
            len(result),
        )
        del result[start:end]

    return [system_message, *result]


def wire_token_count(messages: list[dict[str, Any]]) -> int:
    """Token count for already-converted wire dicts."""
    total = 0
    for message in messages:
        total += estimate_tokens(json.dumps(message, ensure_ascii=False))
    return total
