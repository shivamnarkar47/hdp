"""Wire message model for the OpenAI chat-completions API.

This is a protocol-critical module: the `reasoning_content`-replay rule lives
here. Assistant turns that made tool calls MUST re-send their `reasoning_content`
verbatim; dropping it causes a 400 on the next turn with this gateway.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Union


@dataclass(frozen=True)
class ToolCall:
    """A function call as the OpenAI spec defines it: `arguments` is a JSON string."""

    id: str
    name: str
    arguments: str

    def to_wire(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass(frozen=True)
class SystemMessage:
    text: str

    def to_wire(self) -> dict[str, Any]:
        return {"role": "system", "content": self.text}


@dataclass(frozen=True)
class UserMessage:
    text: str

    def to_wire(self) -> dict[str, Any]:
        return {"role": "user", "content": self.text}


@dataclass(frozen=True)
class AssistantMessage:
    content: str
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] | None = None

    def to_wire(self) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": "assistant", "content": self.content or ""}
        # Reasoning is always replayed when present; never synthesize a
        # placeholder (V4 requires exact replay).
        if self.reasoning_content:
            msg["reasoning_content"] = self.reasoning_content
        if self.tool_calls:
            msg["tool_calls"] = [call.to_wire() for call in self.tool_calls]
        return msg


@dataclass(frozen=True)
class ToolResultMessage:
    tool_call_id: str
    content: str

    def to_wire(self) -> dict[str, Any]:
        return {"role": "tool", "tool_call_id": self.tool_call_id, "content": self.content}


Message = Union[SystemMessage, UserMessage, AssistantMessage, ToolResultMessage]


def wire_token_cost(wire: list[dict[str, Any]]) -> int:
    """Total token cost of already-converted wire dicts.

    Same formula as context.estimate_tokens (one token per three characters of
    the compact JSON serialization): ``sum(estimate_tokens(json.dumps(m,
    ensure_ascii=False)) for m in wire)``. Computed inline here to keep this
    module free of a context import (context imports messages).
    """
    return sum(len(json.dumps(message, ensure_ascii=False)) // 3 for message in wire)


def to_wire_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert message objects to wire dicts.

    Multiple system blocks are coalesced into one message ("\n\n".join) before
    sending — safe for strict chat templates.
    """
    out: list[dict[str, Any]] = []
    system_parts: list[str] = []
    first_system_index: int | None = None
    for message in messages:
        if isinstance(message, SystemMessage):
            if first_system_index is None:
                first_system_index = len(out)
            system_parts.append(message.text)
        else:
            out.append(message.to_wire())
    if first_system_index is not None:
        out.insert(first_system_index, {"role": "system", "content": "\n\n".join(system_parts)})
    return out
