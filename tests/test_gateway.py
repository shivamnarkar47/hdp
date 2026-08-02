"""Gateway SSE client tests — pure unit tests, no network."""

from __future__ import annotations

import email.utils
import http.client
import io
import time
import unittest
import urllib.error
from unittest import mock

from harness.config import MAX_OUTPUT_TOKENS
from harness.gateway import (
    Gateway,
    GatewayError,
    _build_body,
    _build_headers,
    _merge_tool_calls,
    _parse_sse_line,
)
from harness.messages import ToolCall


class TestParseSseLine(unittest.TestCase):
    def test_data_payload(self):
        self.assertEqual(_parse_sse_line('data: {"a":1}'), '{"a":1}')

    def test_done_marker(self):
        self.assertEqual(_parse_sse_line("data: [DONE]"), "[DONE]")

    def test_non_data_line(self):
        self.assertIsNone(_parse_sse_line("event: message"))

    def test_empty_line(self):
        self.assertIsNone(_parse_sse_line(""))

    def test_bare_data_colon(self):
        self.assertEqual(_parse_sse_line("data:"), "")

    def test_bytes_input(self):
        self.assertEqual(_parse_sse_line(b"data: x"), "x")


class TestBuildHeaders(unittest.TestCase):
    def test_headers(self):
        headers = _build_headers("sk-test")
        self.assertEqual(headers["Authorization"], "Bearer sk-test")
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(headers["Accept"], "text/event-stream")
        self.assertEqual(headers["User-Agent"], "python-requests/2.31.0")


class TestMergeToolCalls(unittest.TestCase):
    def test_merge_same_index(self):
        acc = {}
        _merge_tool_calls(
            acc,
            [{"index": 0, "id": "call_1", "function": {"name": "get_weather", "arguments": '{"ci'}}],
        )
        _merge_tool_calls(acc, [{"index": 0, "function": {"arguments": 'ty": "SF"}'}}])
        self.assertEqual(acc[0]["id"], "call_1")
        self.assertEqual(acc[0]["name"], "get_weather")
        self.assertEqual(acc[0]["arguments"], '{"city": "SF"}')

    def test_arguments_delta_concatenates(self):
        acc = {}
        _merge_tool_calls(acc, [{"index": 0, "function": {"name": "f", "arguments_delta": "a"}}])
        _merge_tool_calls(acc, [{"index": 0, "function": {"arguments_delta": "b"}}])
        self.assertEqual(acc[0]["arguments"], "ab")

    def test_different_indices_accumulate_separately(self):
        acc = {}
        _merge_tool_calls(acc, [{"index": 0, "function": {"name": "a"}}])
        _merge_tool_calls(acc, [{"index": 1, "function": {"name": "b"}}])
        self.assertEqual(set(acc), {0, 1})
        self.assertEqual(acc[0]["name"], "a")
        self.assertEqual(acc[1]["name"], "b")

    def test_missing_fields_do_not_clobber(self):
        acc = {}
        _merge_tool_calls(acc, [{"index": 0, "id": "call_x", "function": {"name": "n", "arguments": "{}"}}])
        _merge_tool_calls(acc, [{"index": 0}])
        self.assertEqual(acc[0]["id"], "call_x")
        self.assertEqual(acc[0]["name"], "n")
        self.assertEqual(acc[0]["arguments"], "{}")

    def test_default_index_zero(self):
        acc = {}
        _merge_tool_calls(acc, [{"function": {"name": "f"}}])
        self.assertEqual(list(acc), [0])


class TestBuildBody(unittest.TestCase):
    def test_with_tools(self):
        tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
        body = _build_body("deepseek-v4-flash", [{"role": "user", "content": "hi"}], tools)
        self.assertEqual(body["model"], "deepseek-v4-flash")
        self.assertEqual(body["messages"], [{"role": "user", "content": "hi"}])
        self.assertEqual(body["max_tokens"], MAX_OUTPUT_TOKENS)
        self.assertTrue(body["stream"])
        self.assertEqual(body["tools"], tools)
        for banned in ("tool_choice", "temperature", "stream_options", "store"):
            self.assertNotIn(banned, body)

    def test_without_tools_omits_key(self):
        body = _build_body("deepseek-v4-flash", [{"role": "user", "content": "hi"}], None)
        self.assertNotIn("tools", body)
        body = _build_body("deepseek-v4-flash", [{"role": "user", "content": "hi"}], [])
        self.assertNotIn("tools", body)


class TestStreamEvents(unittest.TestCase):
    def test_stream_reasoning_content_tool_call_done(self):
        payload = (
            'data: {"choices": [{"delta": {"reasoning_content": "think", "content": "hi"}, '
            '"finish_reason": null}]}\n'
            "\n"
            'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_abc", '
            '"function": {"name": "lookup", "arguments": "{}"}}]}, "finish_reason": "tool_calls"}]}\n'
            "\n"
            "data: [DONE]\n"
            "\n"
        )
        fake_response = io.BytesIO(payload.encode("utf-8"))
        gateway = Gateway("https://example.test/v1", "sk-test", "deepseek-v4-flash")
        with mock.patch("urllib.request.urlopen", return_value=fake_response) as urlopen_mock:
            events = list(gateway.stream([{"role": "user", "content": "hi"}]))
        urlopen_mock.assert_called_once()
        self.assertEqual(
            events,
            [
                ("reasoning", "think"),
                ("content", "hi"),
                ("tool_call", ToolCall("call_abc", "lookup", "{}")),
                ("done", "tool_calls"),
            ],
        )


class TestStreamRetryOn429(unittest.TestCase):
    """HTTP 429 rate-limit handling: retry before content, never after."""

    @staticmethod
    def _http_429(retry_after: str | None = None) -> urllib.error.HTTPError:
        hdrs = http.client.HTTPMessage()
        if retry_after is not None:
            hdrs["Retry-After"] = retry_after
        return urllib.error.HTTPError(
            "https://example.test/v1/chat/completions",
            429,
            "Too Many Requests",
            hdrs,
            io.BytesIO(b"rate limited"),
        )

    @staticmethod
    def _success_stream() -> io.BytesIO:
        payload = (
            'data: {"choices": [{"delta": {"content": "hi"}, "finish_reason": null}]}\n'
            "\n"
            "data: [DONE]\n"
            "\n"
        )
        return io.BytesIO(payload.encode("utf-8"))

    def test_429_retries_then_succeeds(self):
        gateway = Gateway("https://example.test/v1", "sk-test", "deepseek-v4-flash")
        sleeps: list[float] = []
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=[self._http_429(retry_after="2"), self._http_429(retry_after="2"), self._success_stream()],
        ) as urlopen_mock, mock.patch(
            "harness.gateway.time.sleep", side_effect=lambda s: sleeps.append(s)
        ):
            events = list(gateway.stream([{"role": "user", "content": "hi"}]))
        self.assertEqual(events, [("content", "hi"), ("done", None)])
        self.assertEqual(urlopen_mock.call_count, 3)
        self.assertEqual(len(sleeps), 2)
        for slept in sleeps:
            self.assertGreaterEqual(slept, 2)  # Retry-After: 2 beats 1s/2s backoff

    def test_429_exhausts_retries(self):
        gateway = Gateway("https://example.test/v1", "sk-test", "deepseek-v4-flash")
        sleeps: list[float] = []
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=[self._http_429(retry_after="1")] * 3,
        ) as urlopen_mock, mock.patch(
            "harness.gateway.time.sleep", side_effect=lambda s: sleeps.append(s)
        ):
            with self.assertRaises(GatewayError) as cm:
                list(gateway.stream([{"role": "user", "content": "hi"}]))
        self.assertIn("429", str(cm.exception))
        self.assertEqual(urlopen_mock.call_count, 3)
        self.assertEqual(len(sleeps), 2)
        for slept in sleeps:
            self.assertGreaterEqual(slept, 1)

    def test_429_after_content_raises(self):
        class _FailingStream:
            """File-like that serves one SSE line, then raises 429 mid-stream."""

            def __init__(self, first: bytes, error: Exception):
                self._first = first
                self._error = error
                self._served = False
                self.closed = False

            def read1(self, size: int = -1) -> bytes:
                if not self._served:
                    self._served = True
                    return self._first
                raise self._error

            def read(self, size: int = -1) -> bytes:
                return self.read1(size)

            def readable(self) -> bool:
                return True

            def writable(self) -> bool:
                return False

            def seekable(self) -> bool:
                return False

            def flush(self) -> None:
                pass

            def close(self) -> None:
                pass

        first_line = b'data: {"choices": [{"delta": {"content": "hi"}, "finish_reason": null}]}\n'
        response = _FailingStream(first_line, self._http_429())
        gateway = Gateway("https://example.test/v1", "sk-test", "deepseek-v4-flash")
        with mock.patch("urllib.request.urlopen", return_value=response) as urlopen_mock, mock.patch(
            "harness.gateway.time.sleep"
        ) as sleep_mock:
            with self.assertRaises(GatewayError) as cm:
                list(gateway.stream([{"role": "user", "content": "hi"}]))
        self.assertIn("gateway stream interrupted after content", str(cm.exception))
        urlopen_mock.assert_called_once()  # retry guard: no second attempt after content
        sleep_mock.assert_not_called()

    def test_429_retry_after_http_date(self):
        gateway = Gateway("https://example.test/v1", "sk-test", "deepseek-v4-flash")
        retry_at = email.utils.formatdate(time.time() + 5)
        sleeps: list[float] = []
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=[self._http_429(retry_after=retry_at), self._success_stream()],
        ) as urlopen_mock, mock.patch(
            "harness.gateway.time.sleep", side_effect=lambda s: sleeps.append(s)
        ):
            events = list(gateway.stream([{"role": "user", "content": "hi"}]))
        self.assertEqual(events, [("content", "hi"), ("done", None)])
        self.assertEqual(urlopen_mock.call_count, 2)
        self.assertEqual(len(sleeps), 1)
        self.assertGreaterEqual(sleeps[0], 2)  # honors date-form Retry-After

    def test_other_4xx_raises_immediately(self):
        """Non-429 4xx stays fatal: no retries, no sleeps."""
        hdrs = http.client.HTTPMessage()
        error = urllib.error.HTTPError(
            "https://example.test/v1/chat/completions",
            400,
            "Bad Request",
            hdrs,
            io.BytesIO(b"bad key"),
        )
        gateway = Gateway("https://example.test/v1", "sk-test", "deepseek-v4-flash")
        with mock.patch("urllib.request.urlopen", side_effect=error) as urlopen_mock, mock.patch(
            "harness.gateway.time.sleep"
        ) as sleep_mock:
            with self.assertRaises(GatewayError) as cm:
                list(gateway.stream([{"role": "user", "content": "hi"}]))
        self.assertIn("gateway HTTP 400", str(cm.exception))
        urlopen_mock.assert_called_once()
        sleep_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
