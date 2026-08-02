"""Token estimation and context-budget bookkeeping."""

from __future__ import annotations

import json
from typing import Any

from harness.config import CONTEXT_WINDOW, MAX_OUTPUT_TOKENS
from harness.messages import Message, SystemMessage, UserMessage, wire_token_cost


def estimate_tokens(text: str) -> int:
    """Rough token estimate: one token per three characters."""
    return len(text) // 3


def context_budget(prompt_tokens: int, max_output_tokens: int = MAX_OUTPUT_TOKENS) -> bool:
    """True if the prompt fits alongside the requested max output."""
    return prompt_tokens <= CONTEXT_WINDOW - max_output_tokens


def message_token_costs(messages: list[Message]) -> list[int]:
    """Per-message wire token cost, computed once (the truncation ledger).

    Same formula as wire_token_count: estimate_tokens of each message's compact
    JSON wire dict. O(n) once; callers subtract ledger slices instead of
    re-serializing the remaining history per dropped turn.
    """
    return [
        estimate_tokens(json.dumps(message.to_wire(), ensure_ascii=False))
        for message in messages
    ]


def truncate_history(
    messages: list[Message], system_message: SystemMessage, max_prompt_tokens: int
) -> list[Message]:
    """Drop oldest user+assistant+tool-result triples until the history fits.

    Never drops the system message, the last user message, or the turn that
    follows it (its assistant reply and tool results). O(n) total: per-message
    token costs are computed once into a ledger, and each dropped turn
    subtracts its ledger slice instead of re-serializing the remainder.
    """
    # Keep exactly one system message, at the front. The system stays EXCLUDED
    # from the token count (matching the previous behavior).
    result = [m for m in messages if not isinstance(m, SystemMessage)]
    ledger = message_token_costs(result)
    total = sum(ledger)

    while total > max_prompt_tokens:
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
        total -= sum(ledger[start:end])
        del ledger[start:end]

    return [system_message, *result]


def wire_token_count(messages: list[dict[str, Any]]) -> int:
    """Token count for already-converted wire dicts.

    Delegates to messages.wire_token_cost, which applies the same per-dict
    formula (estimate_tokens of the compact JSON) — so callers can switch
    between the two freely.
    """
    return wire_token_cost(messages)
