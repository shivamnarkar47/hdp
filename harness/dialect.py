"""Incremental DSML envelope parser/healer plus leaked-token stripper.

DeepSeek V4 emits tool calls inside a DSML (XML-style) envelope that, on this
gateway, leaks into the model's visible ``delta.content`` stream instead of
arriving as structured ``tool_calls``.  ``DialectFeed`` is an incremental state
machine that heals those envelopes back into
:class:`~harness.messages.ToolCall` objects as chunks arrive, and strips leaked
chat-template tokens out of the visible text.

All envelope markers use their exact Unicode forms (fullwidth pipe U+FF5C and
the block glyph U+2581); the model never trained on ASCII substitutes, so the
markers are load-bearing and must never be transliterated.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from harness.messages import ToolCall

logger = logging.getLogger("harness.dialect")

# -- Unicode markers ---------------------------------------------------------
# Fullwidth pipe ｜ is U+FF5C; the block glyph ▁ (used inside template tokens)
# is U+2581.  Built from escapes rather than pasted glyphs to avoid encoding
# accidents in transit.
FW = "\uff5c"
B = "\u2581"

# DSML envelope tags.  The open tags come in (fullwidth, ascii) variants; the
# invoke/parameter *open* entries are prefixes — the full tag carries a
# ``name="..."`` / ``string="..."`` attribute before its closing ``>``.
_SECTION_OPEN = (f"<{FW}DSML{FW}tool_calls>", "<|DSML|tool_calls>")
_SECTION_CLOSE = (f"</{FW}DSML{FW}tool_calls>", "</|DSML|tool_calls>")
_INVOKE_OPEN = (f"<{FW}DSML{FW}invoke", "<|DSML|invoke")
_INVOKE_CLOSE = (f"</{FW}DSML{FW}invoke>", "</|DSML|invoke>")
_PARAM_OPEN = (f"<{FW}DSML{FW}parameter", "<|DSML|parameter")
_PARAM_CLOSE = (f"</{FW}DSML{FW}parameter>", "</|DSML|parameter>")

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"

# Chat-template control tokens to strip from visible text (all fullwidth-pipe
# forms except the ASCII ``<|EOT|>``).
_CONTROL_TOKENS = (
    f"<{FW}begin{B}of{B}sentence{FW}>",
    f"<{FW}end{B}of{B}sentence{FW}>",
    f"<{FW}{B}pad{FW}>",
    f"<{FW}User{FW}>",
    f"<{FW}Assistant{FW}>",
    "<|EOT|>",
    f"<{FW}search{B}begin{FW}>",
    f"<{FW}search{B}end{FW}>",
    f"<{FW}fim{B}hole{FW}>",
    f"<{FW}end{B}of{B}turn{FW}>",
)

MAX_PARAMETER_CHARS = 1_000_000
_TRUNCATED_SUFFIX = "\u2026[parameter truncated]"  # fullwidth ellipsis …
# Accumulation slack so a runaway parameter stays memory-bounded even if the
# closing tag never arrives; values longer than this are truncated at close
# time exactly as if the full raw value had been read.
_PARAM_OVERFLOW_SLACK = 1024

# Token sets per parser state, deduplicated (order-preserving).
_OUTSIDE_TOKENS = tuple(
    dict.fromkeys(
        _SECTION_OPEN
        + _SECTION_CLOSE
        + _INVOKE_CLOSE
        + _PARAM_CLOSE
        + _CONTROL_TOKENS
        + (_THINK_OPEN, _THINK_CLOSE)
    )
)
_THINK_TOKENS = tuple(dict.fromkeys(_CONTROL_TOKENS + (_THINK_CLOSE,)))
_SECTION_TAGS = tuple(
    dict.fromkeys(_SECTION_CLOSE + _INVOKE_OPEN + _INVOKE_CLOSE + _PARAM_OPEN + _PARAM_CLOSE)
)
_INVOKE_TAGS = tuple(dict.fromkeys(_INVOKE_CLOSE + _PARAM_OPEN + _PARAM_CLOSE))

# Every token in every parser state starts with "<", so any partial-token
# suffix of a buffer must contain "<" within its last _MAX_TOKEN_PREFIX chars
# (the longest proper token prefix). Chunks whose tail has no "<" cannot be
# mid-token, so skip the O(tokens x prefix) suffix scan — ~35us to ~0.2us for
# plain text. Sound: an overlap of length k equals token[:k], starts with
# token[0] == "<" at len(text) - k, and k <= len(token) - 1 <= _MAX_TOKEN_PREFIX.
_MAX_TOKEN_PREFIX = max(len(t) - 1 for t in (*_OUTSIDE_TOKENS, *_THINK_TOKENS, *_PARAM_CLOSE))

_NAME_RE = re.compile(r'\sname="([^"]*)"')
_STRING_RE = re.compile(r'\sstring="([^"]*)"')

# Parser states.
_OUTSIDE, _THINK, _DSML_SECTION, _DSML_INVOKE, _DSML_PARAM = range(5)

# One parsed event: ("text", str) | ("reasoning", str) | ("tool_call", ToolCall).
Event = tuple[str, str | ToolCall]


# -- Pure helpers ------------------------------------------------------------
def find_earliest_token(text: str, tokens: tuple[str, ...]) -> tuple[int, str] | None:
    """Earliest occurrence of any token in ``text``; longest token wins ties.

    The longest-wins tie-break is prefix safety: ``<｜tool▁outputs▁begin｜>``
    must not be shadowed by the shorter ``<｜tool▁output▁begin｜>`` at the same
    index.  Returns ``(index, token)`` or ``None``.
    """
    best_index: int | None = None
    best_token: str | None = None
    for token in tokens:
        index = text.find(token)
        if index == -1:
            continue
        if best_index is None or index < best_index or (
            index == best_index and len(token) > len(best_token)
        ):
            best_index = index
            best_token = token
    if best_token is None:
        return None
    return best_index, best_token


def partial_suffix_overlap(text: str, tokens: tuple[str, ...]) -> int:
    """Longest suffix of ``text`` that is a proper prefix of any token."""
    if not text or "<" not in text[-_MAX_TOKEN_PREFIX:]:
        return 0
    best = 0
    for token in tokens:
        for k in range(min(len(text), len(token) - 1), 0, -1):
            if text.endswith(token[:k]):
                best = max(best, k)
                break
    return best


def matching_token(text: str, tokens: tuple[str, ...]) -> str | None:
    """Return the token exactly equal to ``text``, else ``None``."""
    for token in tokens:
        if text == token:
            return token
    return None


def starts_with_any(text: str, tokens: tuple[str, ...]) -> str | None:
    """Return the longest token ``text`` starts with, else ``None``."""
    best: str | None = None
    for token in tokens:
        if text.startswith(token) and (best is None or len(token) > len(best)):
            best = token
    return best


def coerce_dsml_value(raw: str, is_string: bool) -> Any:
    """Coerce a parameter value per its ``string`` attribute.

    ``is_string`` (the default, per the DSML spec) keeps the raw text;
    ``False`` attempts ``json.loads`` on the trimmed value and falls back to
    the raw string when parsing fails.
    """
    if is_string:
        return raw
    trimmed = raw.strip()
    try:
        return json.loads(trimmed)
    except (ValueError, TypeError):
        return raw


def _strip_controls(text: str) -> str:
    """Belt-and-suspenders: remove any leaked control tokens (R9)."""
    for token in _CONTROL_TOKENS:
        text = text.replace(token, "")
    return text


def _is_proper_prefix_of_any(text: str, tokens: tuple[str, ...]) -> bool:
    return bool(text) and any(
        token.startswith(text) and len(text) < len(token) for token in tokens
    )


# -- Incremental state machine ------------------------------------------------
class DialectFeed:
    """Incremental healer for leaked DSML envelopes and chat-template tokens.

    Feed content chunks to :meth:`feed` and collect the events it returns; call
    :meth:`flush` at end of stream to drain remaining buffered text (or to
    discard an unclosed DSML section, per R8).
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._state = _OUTSIDE
        self._swallow_ws = False  # drop leading whitespace after a stripped token
        self._call_counter = 0  # single counter; never resets across sections
        self._calls: list[ToolCall] = []  # completed calls of the current section
        # Current invoke context.
        self._invoke_name = ""
        self._args: dict[str, Any] = {}
        # Current parameter context.
        self._param_name = ""
        self._param_is_string = True
        self._param_value: list[str] = []
        self._param_len = 0
        self._param_overflow = False

    # -- public API -----------------------------------------------------------
    def feed(self, text: str) -> list[Event]:
        """Process one chunk; return events newly parsed from it (R10: "" -> [])."""
        if not text:
            return []
        self._buffer += text
        return self._process()

    def flush(self) -> list[Event]:
        """End of stream: drain remaining text, or discard an unclosed section."""
        events: list[Event] = []
        if self._swallow_ws:
            self._buffer = self._buffer.lstrip()
            self._swallow_ws = False
        if self._state in (_DSML_SECTION, _DSML_INVOKE, _DSML_PARAM):
            logger.warning(
                "Discarding unclosed DSML section at end of stream "
                "(%d buffered chars, %d calls parsed)",
                len(self._buffer),
                len(self._calls),
            )
            self._reset_section()
            return []
        if self._buffer:
            if self._state == _THINK:
                events.append(("reasoning", _strip_controls(self._buffer)))
            else:
                events.append(("text", _strip_controls(self._buffer)))
            self._buffer = ""
        self._state = _OUTSIDE
        return events

    # -- internals -------------------------------------------------------------
    def _reset_section(self) -> None:
        self._calls = []
        self._invoke_name = ""
        self._args = {}
        self._param_value = []
        self._param_len = 0
        self._param_overflow = False
        self._state = _OUTSIDE

    def _process(self) -> list[Event]:
        events: list[Event] = []
        while self._buffer:
            if self._state == _OUTSIDE:
                step_events, progressed = self._step_outside()
            elif self._state == _THINK:
                step_events, progressed = self._step_think()
            elif self._state == _DSML_SECTION:
                step_events, progressed = self._step_section()
            elif self._state == _DSML_INVOKE:
                step_events, progressed = self._step_invoke()
            else:
                step_events, progressed = self._step_param()
            events.extend(step_events)
            if not progressed:
                break  # holding a partial token prefix; wait for more input
        return events

    def _step_outside(self) -> tuple[list[Event], bool]:
        events: list[Event] = []
        if self._swallow_ws:
            self._buffer = self._buffer.lstrip()
            self._swallow_ws = False
            if not self._buffer:
                return events, True
        hit = find_earliest_token(self._buffer, _OUTSIDE_TOKENS)
        if hit is None:
            overlap = partial_suffix_overlap(self._buffer, _OUTSIDE_TOKENS)
            if overlap:
                if overlap < len(self._buffer):
                    events.append(("text", _strip_controls(self._buffer[:-overlap])))
                    self._buffer = self._buffer[-overlap:]
                    return events, True
                return events, False  # whole buffer is a partial token prefix
            events.append(("text", _strip_controls(self._buffer)))
            self._buffer = ""
            return events, True
        index, token = hit
        if index:
            events.append(("text", _strip_controls(self._buffer[:index])))
        self._buffer = self._buffer[index:]
        if matching_token(token, _SECTION_OPEN):
            self._buffer = self._buffer[len(token) :]
            self._calls = []
            self._state = _DSML_SECTION
            return events, True
        if token == _THINK_OPEN:
            self._buffer = self._buffer[len(token) :]
            self._state = _THINK
            self._swallow_ws = True
            return events, True
        # Control token or stray close token with no open: drop it together
        # with any immediately following template padding (R1/R2).
        self._buffer = self._buffer[len(token) :]
        self._swallow_ws = True
        return events, True

    def _step_think(self) -> tuple[list[Event], bool]:
        events: list[Event] = []
        if self._swallow_ws:
            self._buffer = self._buffer.lstrip()
            self._swallow_ws = False
            if not self._buffer:
                return events, True
        hit = find_earliest_token(self._buffer, _THINK_TOKENS)
        if hit is None:
            overlap = partial_suffix_overlap(self._buffer, _THINK_TOKENS)
            if overlap:
                if overlap < len(self._buffer):
                    events.append(("reasoning", _strip_controls(self._buffer[:-overlap])))
                    self._buffer = self._buffer[-overlap:]
                    return events, True
                return events, False
            events.append(("reasoning", _strip_controls(self._buffer)))
            self._buffer = ""
            return events, True
        index, token = hit
        if index:
            events.append(("reasoning", _strip_controls(self._buffer[:index])))
        self._buffer = self._buffer[index:]
        if token == _THINK_CLOSE:
            self._buffer = self._buffer[len(token) :]
            self._state = _OUTSIDE
            self._swallow_ws = True
            return events, True
        # Control token inside the think span: strip it and its padding.
        self._buffer = self._buffer[len(token) :]
        self._swallow_ws = True
        return events, True

    def _step_section(self) -> tuple[list[Event], bool]:
        events: list[Event] = []
        self._buffer = self._buffer.lstrip()  # whitespace between tags is insignificant
        if not self._buffer:
            return events, True
        close = starts_with_any(self._buffer, _SECTION_CLOSE)
        if close is not None:
            self._buffer = self._buffer[len(close) :]
            calls, self._calls = self._calls, []
            for call in calls:
                events.append(("tool_call", call))
            self._state = _OUTSIDE
            return events, True
        if starts_with_any(self._buffer, _INVOKE_OPEN):
            end = self._buffer.find(">")
            if end == -1:
                return events, False  # incomplete open tag: wait for the ">"
            tag = self._buffer[: end + 1]
            self._buffer = self._buffer[end + 1 :]
            match = _NAME_RE.search(tag)
            self._invoke_name = match.group(1) if match else ""
            self._args = {}
            self._state = _DSML_INVOKE
            return events, True
        if _is_proper_prefix_of_any(self._buffer, _SECTION_TAGS):
            return events, False  # partial tag: wait for more input
        # Malformed content inside the envelope: consume up to the next "<".
        next_lt = self._buffer.find("<")
        if next_lt == -1:
            self._buffer = ""
        elif next_lt == 0:
            self._buffer = self._buffer[1:]
        else:
            self._buffer = self._buffer[next_lt:]
        return events, True

    def _step_invoke(self) -> tuple[list[Event], bool]:
        events: list[Event] = []
        self._buffer = self._buffer.lstrip()
        if not self._buffer:
            return events, True
        close = starts_with_any(self._buffer, _INVOKE_CLOSE)
        if close is not None:
            self._buffer = self._buffer[len(close) :]
            self._call_counter += 1
            self._calls.append(
                ToolCall(
                    f"call_{self._call_counter}",
                    self._invoke_name,
                    json.dumps(self._args, ensure_ascii=False),
                )
            )
            self._state = _DSML_SECTION
            return events, True
        if starts_with_any(self._buffer, _PARAM_OPEN):
            end = self._buffer.find(">")
            if end == -1:
                return events, False  # incomplete open tag: wait for the ">"
            tag = self._buffer[: end + 1]
            self._buffer = self._buffer[end + 1 :]
            name_match = _NAME_RE.search(tag)
            self._param_name = name_match.group(1) if name_match else ""
            string_match = _STRING_RE.search(tag)
            self._param_is_string = (string_match.group(1) if string_match else "true") != "false"
            self._param_value = []
            self._param_len = 0
            self._param_overflow = False
            self._state = _DSML_PARAM
            return events, True
        if _is_proper_prefix_of_any(self._buffer, _INVOKE_TAGS):
            return events, False
        next_lt = self._buffer.find("<")
        if next_lt == -1:
            self._buffer = ""
        elif next_lt == 0:
            self._buffer = self._buffer[1:]
        else:
            self._buffer = self._buffer[next_lt:]
        return events, True

    def _step_param(self) -> tuple[list[Event], bool]:
        events: list[Event] = []
        hit = find_earliest_token(self._buffer, _PARAM_CLOSE)
        if hit is None:
            overlap = partial_suffix_overlap(self._buffer, _PARAM_CLOSE)
            if overlap:
                keep = self._buffer[:-overlap]
                self._buffer = self._buffer[-overlap:]
                if keep:
                    self._append_param_value(keep)
                return events, bool(keep)
            self._append_param_value(self._buffer)
            self._buffer = ""
            return events, True
        index, close = hit
        self._append_param_value(self._buffer[:index])
        self._buffer = self._buffer[index + len(close) :]
        raw = "".join(self._param_value)
        if len(raw) > MAX_PARAMETER_CHARS:
            raw = raw[:MAX_PARAMETER_CHARS] + _TRUNCATED_SUFFIX
        self._args[self._param_name] = coerce_dsml_value(raw, self._param_is_string)
        self._state = _DSML_INVOKE
        return events, True

    def _append_param_value(self, part: str) -> None:
        if self._param_overflow:
            return
        self._param_value.append(part)
        self._param_len += len(part)
        if self._param_len > MAX_PARAMETER_CHARS + _PARAM_OVERFLOW_SLACK:
            self._param_overflow = True
