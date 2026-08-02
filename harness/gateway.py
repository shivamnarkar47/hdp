"""SSE streaming client for the OpenAI-compatible chat-completions gateway.

Pure and self-contained: gateway.py and dialect.py are the only modules that
know the wire protocol, and this file maps 1:1 to the future Rust/Go port.
"""

from __future__ import annotations

import email.utils
import http.client
import io
import json
import socket
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Generator, Iterator

from harness.config import MAX_OUTPUT_TOKENS, REASONING_FIELD, REQUEST_TIMEOUT
from harness.messages import ToolCall

StreamEvent = tuple[str, str | ToolCall] | tuple[str, str]
# ("content", str) | ("reasoning", str) | ("tool_call", ToolCall)
# | ("done", finish_reason: str | None) | ("error", str)


class GatewayError(Exception):
    """Raised for gateway failures that are not retryable or exhausted retries."""


def _build_headers(api_key: str) -> dict:
    """Return the HTTP headers for a gateway request."""
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        # Cloudflare WAF (error 1010) blocks urllib's default UA; this value
        # is proven to pass. Do not invent a custom UA.
        "User-Agent": "python-requests/2.31.0",
    }


def _build_body(model_id: str, messages: list, tools: list | None) -> dict:
    """Build the chat-completions request body.

    The "tools" key is omitted entirely when no tools are given, and unsupported
    fields (tool_choice, temperature, stream_options, store) are never sent —
    this model rejects them.
    """
    body: dict = {
        "model": model_id,
        "messages": messages,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "stream": True,
    }
    if tools:
        body["tools"] = tools
    return body


def _parse_sse_line(line: str | bytes) -> str | None:
    """Extract the payload of a `data:` SSE line; return None for any other line."""
    if isinstance(line, bytes):
        line = line.decode("utf-8")
    stripped = line.strip()
    if not stripped.startswith("data:"):
        return None
    return stripped[len("data:"):].strip()


def _merge_tool_calls(acc: dict[int, dict], items: list) -> None:
    """Merge streaming tool-call deltas into per-index accumulators."""
    for item in items:
        index = item.get("index", 0)
        entry = acc.setdefault(index, {"id": "", "name": "", "arguments": ""})
        if item.get("id"):
            entry["id"] = item["id"]
        fn = item.get("function", {})
        if fn.get("name"):
            entry["name"] = fn["name"]
        for source in (item, fn):
            piece = source.get("arguments")
            if not isinstance(piece, str):
                piece = source.get("arguments_delta")
            if isinstance(piece, str):
                entry["arguments"] += piece


def _join_content_parts(content) -> str:
    """Join string parts of a list/dict content delta; ignore non-str parts."""
    parts = content if isinstance(content, list) else list(content.values())
    return "".join(part for part in parts if isinstance(part, str))


def _emit_tool_calls(acc: dict[int, dict]) -> Iterator[StreamEvent]:
    """Yield accumulated tool calls in ascending index order; skip nameless ones."""
    for index in sorted(acc):
        entry = acc[index]
        if not entry["name"]:
            continue
        yield ("tool_call", ToolCall(entry["id"] or f"call_{index}", entry["name"], entry["arguments"]))


def _parse_retry_after(headers) -> int | None:
    """Parse a Retry-After header into whole seconds, or None when absent/unparseable.

    Accepts delta-seconds ("5") or an HTTP-date; the date form is measured from
    now, floored to whole seconds and clamped to >= 0.
    """
    if headers is None:
        return None
    value = headers.get("Retry-After")
    if value is None:
        return None
    try:
        return max(0, int(value))
    except ValueError:
        pass
    try:
        retry_at = email.utils.parsedate_to_datetime(value)
        # "-0000" or a missing zone parses to a naive datetime; treat as UTC.
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        delay = (retry_at - datetime.now(timezone.utc)).total_seconds()
        return max(0, int(delay))
    except (TypeError, ValueError, OverflowError):
        return None


class Gateway:
    """SSE streaming client for OpenAI-compatible chat completions."""

    def __init__(self, base_url: str, api_key: str, model_id: str) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model_id = model_id

    def stream(
        self, messages: list[dict], tools: list[dict] | None = None
    ) -> Generator[StreamEvent, None, None]:
        """Stream chat completions; yields content/reasoning/tool_call/done events.

        Retries (5xx, network errors, and HTTP 429 rate limits — all only before
        any event was yielded) up to 3 attempts total with 1s/2s/4s backoff; a
        429's Retry-After header extends the sleep (capped at 60s). Other 4xx
        errors raise immediately.
        """
        url = self.base_url.rstrip("/") + "/chat/completions"
        request = urllib.request.Request(
            url,
            data=json.dumps(_build_body(self.model_id, messages, tools)).encode("utf-8"),
            headers=_build_headers(self.api_key),
            method="POST",
        )
        backoff = (1, 2, 4)
        last_error = "unknown error"
        emitted_any = False
        for attempt in range(3):
            retry_after: int | None = None
            try:
                for event in self._run_attempt(request):
                    emitted_any = True
                    yield event
                return
            except urllib.error.HTTPError as err:
                body_preview = err.read().decode("utf-8", errors="replace")[:300]
                if 400 <= err.code < 500 and err.code != 429:
                    # Code/key problem — never retry.
                    raise GatewayError(f"gateway HTTP {err.code}: {body_preview}") from err
                last_error = f"HTTP {err.code}: {body_preview}"
                if err.code == 429:
                    retry_after = _parse_retry_after(err.headers)
            except (
                urllib.error.URLError,
                TimeoutError,
                ConnectionError,
                socket.timeout,
                http.client.HTTPException,
            ) as err:
                last_error = err
            if emitted_any:
                # Never retry after visible content.
                raise GatewayError(f"gateway stream interrupted after content: {last_error}")
            if attempt < 2:
                sleep_seconds = backoff[attempt]
                if retry_after is not None:
                    sleep_seconds = min(60, max(sleep_seconds, retry_after))
                time.sleep(sleep_seconds)
        raise GatewayError(f"gateway request failed after 3 attempts: {last_error}")

    def _run_attempt(self, request: urllib.request.Request) -> Iterator[StreamEvent]:
        """One full open+read cycle; raises on any transport failure."""
        response = urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT)
        try:
            wrapper = io.TextIOWrapper(response, encoding="utf-8")
            tool_acc: dict[int, dict] = {}
            last_finish_reason: str | None = None
            for line in wrapper:
                payload = _parse_sse_line(line)
                if payload is None:
                    continue
                if payload == "[DONE]":
                    yield from _emit_tool_calls(tool_acc)
                    yield ("done", last_finish_reason)
                    return
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue  # tolerate malformed heartbeat/noise lines
                choices = chunk.get("choices")
                if not choices:
                    continue  # some servers send empty choices during reasoning
                choice = choices[0]
                finish_reason = choice.get("finish_reason")
                if finish_reason is not None:
                    last_finish_reason = finish_reason
                delta = choice.get("delta", {})
                reasoning = delta.get(REASONING_FIELD)
                if isinstance(reasoning, str):
                    yield ("reasoning", reasoning)
                content = delta.get("content")
                if content is not None:
                    if isinstance(content, str):
                        yield ("content", content)
                    else:
                        joined = _join_content_parts(content)
                        if joined:
                            yield ("content", joined)
                tool_calls = delta.get("tool_calls")
                if tool_calls:
                    _merge_tool_calls(tool_acc, tool_calls)
            # Server closed the stream without [DONE].
            yield from _emit_tool_calls(tool_acc)
            yield ("done", last_finish_reason)
        finally:
            response.close()
