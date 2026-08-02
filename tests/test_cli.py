"""CLI tests: --version, sessions show/delete/prune, run -, doctor.

All network is stubbed (FakeGateway for run, patched urlopen for doctor);
the session store is isolated via HARNESSDP_SESSIONS_DIR.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from harness import sessions
from harness.cli import main


class FakeGateway:
    """Network-free gateway stub; records every stream() invocation."""

    def __init__(self, *scripts):
        self.scripts = list(scripts)
        self.calls = []

    def stream(self, messages, tools):
        self.calls.append((messages, tools))
        yield from self.scripts.pop(0)


class TestCli(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tempdir = Path(self._tmp.name)
        self._old_sessions_dir = os.environ.get("HARNESSDP_SESSIONS_DIR")
        os.environ["HARNESSDP_SESSIONS_DIR"] = str(self.tempdir / "sessions")
        self._old_key = os.environ.get("OPENCODE_API_KEY")
        os.environ["OPENCODE_API_KEY"] = "test-key"

    def tearDown(self):
        if self._old_sessions_dir is None:
            os.environ.pop("HARNESSDP_SESSIONS_DIR", None)
        else:
            os.environ["HARNESSDP_SESSIONS_DIR"] = self._old_sessions_dir
        if self._old_key is None:
            os.environ.pop("OPENCODE_API_KEY", None)
        else:
            os.environ["OPENCODE_API_KEY"] = self._old_key
        self._tmp.cleanup()

    def _run_cli(self, argv):
        """Run main(); returns (exit_code, stdout, stderr)."""
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                main(argv)
            except SystemExit as exc:
                return exc.code, out.getvalue(), err.getvalue()
        return 0, out.getvalue(), err.getvalue()

    # -- version ------------------------------------------------------------

    def test_version(self):
        code, out, _ = self._run_cli(["--version"])
        self.assertEqual(code, 0)
        self.assertIn("hdp 0.1.0", out)

    # -- sessions show ------------------------------------------------------

    def test_sessions_show(self):
        session_id = sessions.new_session_id()
        sessions.append_event(session_id, {"type": "user", "data": {"content": "hello world"}})
        code, out, _ = self._run_cli(["sessions", "show", session_id])
        self.assertEqual(code, 0)
        self.assertIn("user", out)
        self.assertIn("hello world", out)

    def test_sessions_show_missing(self):
        code, _, err = self._run_cli(["sessions", "show", "does-not-exist"])
        self.assertEqual(code, 1)
        self.assertIn("no such session", err)

    # -- sessions delete ----------------------------------------------------

    def test_sessions_delete(self):
        keep_id = sessions.new_session_id()
        sessions.append_event(keep_id, {"type": "user", "data": {"content": "keep me"}})
        delete_id = sessions.new_session_id()
        sessions.append_event(delete_id, {"type": "user", "data": {"content": "delete me"}})
        code, out, _ = self._run_cli(["sessions", "delete", delete_id])
        self.assertEqual(code, 0)
        self.assertIn(f"deleted {delete_id}", out)
        # Only the deleted session's file is removed.
        self.assertFalse((sessions.get_store_dir() / f"{delete_id}.jsonl").exists())
        self.assertTrue((sessions.get_store_dir() / f"{keep_id}.jsonl").is_file())

    def test_sessions_delete_missing(self):
        code, out, _ = self._run_cli(["sessions", "delete", "nope"])
        self.assertEqual(code, 1)
        self.assertIn("no such session", out)

    # -- sessions prune -----------------------------------------------------

    def test_sessions_prune_keep_newest(self):
        ids = []
        for i in range(3):
            session_id = sessions.new_session_id()
            sessions.append_event(session_id, {"type": "user", "data": {"content": f"s{i}"}})
            ids.append(session_id)
        newest = max(ids)  # ids sort chronologically; newest == max
        code, out, _ = self._run_cli(["sessions", "prune", "--keep", "1"])
        self.assertEqual(code, 0)
        remaining = [p.stem for p in sessions.get_store_dir().glob("*.jsonl")]
        self.assertEqual(remaining, [newest])
        for old in ids:
            if old != newest:
                self.assertIn(f"deleted {old}", out)

    def test_sessions_prune_keep_zero_deletes_all(self):
        ids = []
        for i in range(2):
            session_id = sessions.new_session_id()
            sessions.append_event(session_id, {"type": "user", "data": {"content": f"s{i}"}})
            ids.append(session_id)
        code, out, _ = self._run_cli(["sessions", "prune", "--keep", "0"])
        self.assertEqual(code, 0)
        self.assertEqual(list(sessions.get_store_dir().glob("*.jsonl")), [])
        self.assertIn(f"deleted {ids[0]}", out)

    def test_sessions_prune_nothing(self):
        code, out, _ = self._run_cli(["sessions", "prune"])
        self.assertEqual(code, 0)
        self.assertIn("nothing to prune", out)

    # -- run - --------------------------------------------------------------

    def test_run_dash_reads_prompt_from_stdin(self):
        gateway = FakeGateway([("content", "ok\n"), ("done", "stop")])
        with mock.patch.object(sys, "stdin", io.StringIO("hi")), mock.patch(
            "harness.cli.Gateway", return_value=gateway
        ):
            code, out, _ = self._run_cli(["run", "-", "--json", "--dir", str(self.tempdir)])
        self.assertEqual(code, 0)
        payload = json.loads(out.splitlines()[-1])
        self.assertIn("usage", payload)
        self.assertIn("answer", payload)
        self.assertGreater(payload["usage"]["input_tokens"], 0)

    # -- doctor -------------------------------------------------------------

    def test_doctor(self):
        class FakeResponse:
            """A 200-style urlopen result: context manager with read()."""

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return b""

        with mock.patch("urllib.request.urlopen", return_value=FakeResponse()):
            code, out, _ = self._run_cli(["doctor"])
        self.assertIn(code, (0, 1))
        self.assertIn("python:", out)
        self.assertIn("gateway:", out)


if __name__ == "__main__":
    unittest.main()
