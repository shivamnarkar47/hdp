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
from harness.toolcache import ToolCache


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
        self.assertIn("cost", payload)  # estimated dollars from usage
        self.assertGreater(payload["usage"]["input_tokens"], 0)

    # -- run flags (tool cache / verify) -------------------------------------

    def test_run_no_tool_cache_and_no_verify_reach_constructed_objects(self):
        """--no-tool-cache -> registry gets cache=None; --no-verify -> the
        loop gets enable_verify=False (asserted on the constructed objects)."""
        captured = {}

        class FakeAgentLoop:
            def __init__(self, *args, **kwargs):
                captured["tools"] = args[1]
                captured["enable_verify"] = kwargs.get("enable_verify", True)

            def run(self, task, emit=None):
                return "ok"

        gateway = FakeGateway([("content", "hi"), ("done", "stop")])
        with mock.patch("harness.cli.Gateway", return_value=gateway), mock.patch(
            "harness.cli.AgentLoop", FakeAgentLoop
        ):
            code, out, _ = self._run_cli(
                ["run", "hi", "--no-tool-cache", "--no-verify", "--dir", str(self.tempdir)]
            )
        self.assertEqual(code, 0)
        self.assertIsNone(captured["tools"]._cache)
        self.assertFalse(captured["enable_verify"])

    def test_run_defaults_enable_tool_cache_and_verify(self):
        """Without the flags the registry gets a real ToolCache and the loop
        gets enable_verify=True."""
        captured = {}

        class FakeAgentLoop:
            def __init__(self, *args, **kwargs):
                captured["tools"] = args[1]
                captured["enable_verify"] = kwargs.get("enable_verify", True)

            def run(self, task, emit=None):
                return "ok"

        gateway = FakeGateway([("content", "hi"), ("done", "stop")])
        with mock.patch("harness.cli.Gateway", return_value=gateway), mock.patch(
            "harness.cli.AgentLoop", FakeAgentLoop
        ):
            code, _, _ = self._run_cli(["run", "hi", "--dir", str(self.tempdir)])
        self.assertEqual(code, 0)
        self.assertIsInstance(captured["tools"]._cache, ToolCache)
        self.assertTrue(captured["enable_verify"])

    def test_run_no_verify_alone_keeps_cache(self):
        """The flags are independent: --no-verify alone leaves the cache on."""
        captured = {}

        class FakeAgentLoop:
            def __init__(self, *args, **kwargs):
                captured["tools"] = args[1]
                captured["enable_verify"] = kwargs.get("enable_verify", True)

            def run(self, task, emit=None):
                return "ok"

        gateway = FakeGateway([("content", "hi"), ("done", "stop")])
        with mock.patch("harness.cli.Gateway", return_value=gateway), mock.patch(
            "harness.cli.AgentLoop", FakeAgentLoop
        ):
            code, _, _ = self._run_cli(
                ["run", "hi", "--no-verify", "--dir", str(self.tempdir)]
            )
        self.assertEqual(code, 0)
        self.assertIsInstance(captured["tools"]._cache, ToolCache)
        self.assertFalse(captured["enable_verify"])

    # -- run batch -----------------------------------------------------------

    def test_run_batch_two_prompts_json_array(self):
        """Two prompts -> two sessions, both answers present, --json emits a
        single array of two per-run records in file order."""
        batch_file = self.tempdir / "batch.txt"
        batch_file.write_text("first prompt\nsecond prompt\n\n", encoding="utf-8")
        created: list = []

        class EchoGateway:
            """Answers with the last user message so each record is distinct."""

            def __init__(self):
                self.calls = []
                created.append(self)

            def stream(self, messages, tools):
                self.calls.append((messages, tools))
                prompt = next(
                    (m["content"] for m in reversed(messages) if m["role"] == "user"),
                    "",
                )
                # Content ends with a newline, like the suite's other
                # FakeGateway scripts, so the final --json array starts on its
                # own line (the established last-line parse convention).
                yield ("content", f"answer: {prompt}\n")
                yield ("done", "stop")

        with mock.patch("harness.cli.Gateway", side_effect=lambda *a, **k: EchoGateway()):
            code, out, _ = self._run_cli(
                ["run", "--batch", str(batch_file), "--json", "--dir", str(self.tempdir)]
            )
        self.assertEqual(code, 0)
        self.assertEqual(len(created), 2)  # one Gateway per task
        payload = json.loads(out.splitlines()[-1])
        self.assertEqual(len(payload), 2)
        self.assertEqual(
            [r["answer"] for r in payload],
            ["answer: first prompt\n", "answer: second prompt\n"],
        )
        session_ids = [r["session_id"] for r in payload]
        self.assertEqual(len(set(session_ids)), 2)  # distinct sessions
        for record in payload:
            self.assertIn("usage", record)
            self.assertIn("steps", record)
            self.assertIn("cost", record)  # estimated dollars from usage
            self.assertNotIn("error", record)
        files = list(sessions.get_store_dir().glob("*.jsonl"))
        self.assertEqual(len(files), 2)  # both sessions persisted

    def test_run_batch_json_array_file(self):
        """A --batch file that is a JSON array of strings runs each element."""
        batch_file = self.tempdir / "batch.json"
        batch_file.write_text('["alpha", "beta"]', encoding="utf-8")

        class EchoGateway:
            def stream(self, messages, tools):
                prompt = next(
                    (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
                )
                yield ("content", f"answer: {prompt}\n")
                yield ("done", "stop")

        with mock.patch("harness.cli.Gateway", side_effect=lambda *a, **k: EchoGateway()):
            code, out, _ = self._run_cli(
                ["run", "--batch", str(batch_file), "--json", "--dir", str(self.tempdir)]
            )
        self.assertEqual(code, 0)
        payload = json.loads(out.splitlines()[-1])
        self.assertEqual(
            [r["answer"] for r in payload], ["answer: alpha\n", "answer: beta\n"]
        )

    def test_run_batch_and_positional_mutually_exclusive(self):
        batch_file = self.tempdir / "batch.txt"
        batch_file.write_text("p1\np2\n", encoding="utf-8")
        code, out, err = self._run_cli(["run", "a prompt", "--batch", str(batch_file)])
        self.assertEqual(code, 2)
        self.assertIn("not allowed with argument", err)

    def test_run_batch_requires_prompt_without_batch(self):
        code, out, err = self._run_cli(["run"])
        self.assertEqual(code, 2)
        self.assertIn("required: prompt", err)

    def test_run_batch_workers_one_serial_order(self):
        """--workers 1 runs serially: gateway construction (and therefore
        answers) follow the file order exactly."""
        batch_file = self.tempdir / "batch.txt"
        batch_file.write_text("first\nsecond\nthird\n", encoding="utf-8")
        counter = {"n": 0}

        def factory(*a, **k):
            gateway = FakeGateway([("content", f"answer-{counter['n']}\n"), ("done", "stop")])
            counter["n"] += 1
            return gateway

        with mock.patch("harness.cli.Gateway", side_effect=factory):
            code, out, _ = self._run_cli(
                [
                    "run",
                    "--batch",
                    str(batch_file),
                    "--workers",
                    "1",
                    "--json",
                    "--dir",
                    str(self.tempdir),
                ]
            )
        self.assertEqual(code, 0)
        payload = json.loads(out.splitlines()[-1])
        self.assertEqual(
            [r["answer"] for r in payload], ["answer-0\n", "answer-1\n", "answer-2\n"]
        )

    def test_run_batch_missing_file(self):
        code, out, err = self._run_cli(
            ["run", "--batch", str(self.tempdir / "nope.txt"), "--json"]
        )
        self.assertEqual(code, 1)
        self.assertIn("cannot read batch file", err)

    def test_run_batch_empty_file(self):
        batch_file = self.tempdir / "empty.txt"
        batch_file.write_text("\n\n", encoding="utf-8")
        code, out, err = self._run_cli(["run", "--batch", str(batch_file)])
        self.assertEqual(code, 1)
        self.assertIn("contains no prompts", err)

    def test_run_batch_loop_error_exit_2(self):
        """Any loop error -> exit 2; the failed records carry `error` in the
        --json array and a per-kind count goes to stderr."""
        from harness.loop import LoopError

        batch_file = self.tempdir / "batch.txt"
        batch_file.write_text("one\ntwo\n", encoding="utf-8")

        # One task fails with LoopError, one succeeds (shared counter: each
        # task constructs its own gateway instance).
        flaky_state = {"n": 0}

        class FlakyGateway:
            def stream(self, messages, tools):
                flaky_state["n"] += 1
                if flaky_state["n"] == 1:
                    raise LoopError("max steps reached")
                yield ("content", "ok\n")
                yield ("done", "stop")

        with mock.patch("harness.cli.Gateway", side_effect=lambda *a, **k: FlakyGateway()):
            code, out, err = self._run_cli(
                ["run", "--batch", str(batch_file), "--json", "--dir", str(self.tempdir)]
            )
        self.assertEqual(code, 2)
        self.assertIn("1 loop", err)
        payload = json.loads(out.splitlines()[-1])
        self.assertEqual(len(payload), 2)
        failed = next(r for r in payload if "error" in r)
        ok_record = next(r for r in payload if "answer" in r)
        self.assertEqual(failed["error"], "max steps reached")
        self.assertNotIn("answer", failed)  # failed record: no answer
        self.assertEqual(ok_record["answer"], "ok\n")

    def test_run_batch_non_json_separators(self):
        """Without --json each task is announced with a --- session_id ---
        separator before its streamed answer."""
        batch_file = self.tempdir / "batch.txt"
        batch_file.write_text("hi\n", encoding="utf-8")

        class EchoGateway:
            def stream(self, messages, tools):
                yield ("content", "answer\n")
                yield ("done", "stop")

        with mock.patch("harness.cli.Gateway", side_effect=lambda *a, **k: EchoGateway()):
            code, out, err = self._run_cli(
                ["run", "--batch", str(batch_file), "--dir", str(self.tempdir)]
            )
        self.assertEqual(code, 0)
        self.assertIn("--- ", out)
        self.assertIn("answer", out)
        separator = next(line for line in out.splitlines() if line.startswith("--- "))
        session_id = separator.strip("- ").strip()
        self.assertTrue((sessions.get_store_dir() / f"{session_id}.jsonl").is_file())

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
