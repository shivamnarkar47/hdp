"""Gateway SSE client tests — pure unit tests, no network."""

from __future__ import annotations

import io
import unittest
from unittest import mock

from harness.config import MAX_OUTPUT_TOKENS
from harness.gateway import (
    Gateway,
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


if __name__ == "__main__":
    unittest.main()
