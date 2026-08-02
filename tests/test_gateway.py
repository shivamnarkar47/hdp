"""Gateway SSE client tests — pure unit tests, no network."""

from __future__ import annotations

import contextlib
import email.utils
import http.client
import io
import os
import socket
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from harness import gateway as gateway_mod
from harness.config import MAX_OUTPUT_TOKENS
from harness.gateway import (
    Gateway,
    GatewayError,
    KeepAliveOpener,
    _build_body,
    _build_headers,
    _keepalive_urlopen,
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
        with mock.patch("harness.gateway._urlopen", return_value=fake_response) as urlopen_mock:
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
            "harness.gateway._urlopen",
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
            "harness.gateway._urlopen",
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
        with mock.patch("harness.gateway._urlopen", return_value=response) as urlopen_mock, mock.patch(
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
            "harness.gateway._urlopen",
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
        with mock.patch("harness.gateway._urlopen", side_effect=error) as urlopen_mock, mock.patch(
            "harness.gateway.time.sleep"
        ) as sleep_mock:
            with self.assertRaises(GatewayError) as cm:
                list(gateway.stream([{"role": "user", "content": "hi"}]))
        self.assertIn("gateway HTTP 400", str(cm.exception))
        urlopen_mock.assert_called_once()
        sleep_mock.assert_not_called()


# -- keep-alive transport ----------------------------------------------------


class _CountingHTTPServer(ThreadingHTTPServer):
    """Threaded HTTP server that counts connections/requests and can force-close
    established connections (simulating the server going away mid-keep-alive)."""

    daemon_threads = True

    def __init__(self, addr, handler):
        self.connections = 0
        self.requests = 0
        self._sockets = []
        self._sockets_lock = threading.Lock()
        super().__init__(addr, handler)

    def get_request(self):
        sock, addr = super().get_request()
        self.connections += 1  # one accepted socket == one connection
        with self._sockets_lock:
            self._sockets.append(sock)
        return sock, addr

    def close_connections(self):
        with self._sockets_lock:
            sockets, self._sockets = self._sockets, []
        for sock in sockets:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass

    @property
    def base_url(self):
        host, port = self.server_address[:2]
        return f"http://{host}:{port}/v1"


_SSE_BODY = (
    b'data: {"choices": [{"delta": {"content": "hi"}, "finish_reason": null}]}\n'
    b"\n"
    b"data: [DONE]\n"
    b"\n"
)


class _TestHandler(BaseHTTPRequestHandler):
    """Serves one SSE-ish POST response; a fresh instance per TCP connection."""

    protocol_version = "HTTP/1.1"
    status = 200
    body = _SSE_BODY
    extra_headers: dict = {}

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        self.server.requests += 1
        self.send_response(self.status)
        for key, value in self.extra_headers.items():
            self.send_header(key, value)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(self.body)))
        self.end_headers()
        self.wfile.write(self.body)

    def log_message(self, *args, **kwargs):
        pass


@contextlib.contextmanager
def _serve(port: int = 0):
    """A live threaded HTTP server on 127.0.0.1 (yields the server object)."""
    server = _CountingHTTPServer(("127.0.0.1", port), _TestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.close_connections()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class TestKeepAliveTransport(unittest.TestCase):
    """Keep-alive opener: connection reuse, reconnect, HTTPError mapping, and
    the env/proxy switch back to the plain urllib path."""

    def setUp(self):
        _TestHandler.status = 200
        _TestHandler.body = _SSE_BODY
        _TestHandler.extra_headers = {}

    def test_keepalive_two_streams_one_connection_then_reconnect(self):
        # Two sequential streams must ride ONE connection; once the server is
        # replaced on the same port, the stale cached socket is detected and a
        # fresh connection is opened — no crash.
        with mock.patch.dict(os.environ, {}, clear=True):
            gateway_mod._set_transport()
            try:
                self.assertIs(gateway_mod._urlopen, _keepalive_urlopen)
                with _serve() as server:
                    self._stream_twice(server)
                    self.assertEqual(server.connections, 1)
                    self.assertEqual(server.requests, 2)
                    port = server.server_address[1]
                with _serve(port=port) as server2:
                    self._stream_twice(server2)
                    self.assertEqual(server2.connections, 1)
                    self.assertEqual(server2.requests, 2)
            finally:
                gateway_mod._set_transport()

    def _stream_twice(self, server):
        gateway = Gateway(server.base_url, "sk-test", "deepseek-v4-flash")
        msgs = [{"role": "user", "content": "hi"}]
        for _ in range(2):
            self.assertEqual(list(gateway.stream(msgs)), [("content", "hi"), ("done", None)])

    def test_keepalive_opener_raises_http_error_on_non2xx(self):
        _TestHandler.status = 429
        _TestHandler.body = b"rate limited"
        with _serve() as server:
            opener = KeepAliveOpener(timeout=5)
            request = urllib.request.Request(
                server.base_url + "/chat/completions",
                data=b'{"model":"m","messages":[],"max_tokens":1,"stream":true}',
                headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
                method="POST",
            )
            with self.assertRaises(urllib.error.HTTPError) as cm:
                opener.open(request, timeout=5)
            self.assertEqual(cm.exception.code, 429)
            self.assertIn("rate limited", cm.exception.read().decode())

    def test_keepalive_429_maps_to_http_error_and_retries(self):
        # A 429 through the keep-alive opener must raise HTTPError so
        # stream()'s existing retry/backoff logic runs untouched.
        _TestHandler.status = 429
        _TestHandler.body = b"rate limited"
        _TestHandler.extra_headers = {"Retry-After": "1"}
        with mock.patch.dict(os.environ, {}, clear=True):
            gateway_mod._set_transport()
            try:
                with _serve() as server:
                    gateway = Gateway(server.base_url, "sk-test", "deepseek-v4-flash")
                    with mock.patch("harness.gateway.time.sleep"):
                        with self.assertRaises(GatewayError) as cm:
                            list(gateway.stream([{"role": "user", "content": "hi"}]))
                    self.assertIn("429", str(cm.exception))
                    self.assertEqual(server.requests, 3)  # 3 attempts
                    self.assertEqual(server.connections, 3)  # never reuse a 429 conn
            finally:
                gateway_mod._set_transport()

    def test_no_keepalive_env_uses_plain_urlopen(self):
        gateway = Gateway("https://example.test/v1", "sk-test", "deepseek-v4-flash")
        try:
            with mock.patch.dict(os.environ, {"KALA_NO_KEEPALIVE": "1"}), mock.patch(
                "urllib.request.urlopen", return_value=io.BytesIO(_SSE_BODY)
            ) as urlopen_mock:
                gateway_mod._set_transport()
                self.assertIs(gateway_mod._urlopen, urllib.request.urlopen)
                events = list(gateway.stream([{"role": "user", "content": "hi"}]))
            urlopen_mock.assert_called_once()
            self.assertEqual(events, [("content", "hi"), ("done", None)])
        finally:
            gateway_mod._set_transport()

    def test_keepalive_active_does_not_call_urllib_urlopen(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            gateway_mod._set_transport()
            try:
                with _serve() as server:
                    gateway = Gateway(server.base_url, "sk-test", "deepseek-v4-flash")
                    with mock.patch("urllib.request.urlopen") as urlopen_mock:
                        events = list(gateway.stream([{"role": "user", "content": "hi"}]))
                    urlopen_mock.assert_not_called()
                    self.assertEqual(events, [("content", "hi"), ("done", None)])
            finally:
                gateway_mod._set_transport()

    def test_proxy_env_disables_keepalive(self):
        with mock.patch.dict(os.environ, {"http_proxy": "http://proxy:3128"}):
            gateway_mod._set_transport()
            try:
                self.assertIs(gateway_mod._urlopen, urllib.request.urlopen)
            finally:
                gateway_mod._set_transport()


if __name__ == "__main__":
    unittest.main()
