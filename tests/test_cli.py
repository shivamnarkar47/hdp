"""CLI tests: --version, sessions show/delete/prune, run -, doctor.

All network is stubbed (FakeGateway for run, patched urlopen for doctor);
the session store is isolated via KAAL_SESSIONS_DIR.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time
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
        self._old_sessions_dir = os.environ.get("KAAL_SESSIONS_DIR")
        os.environ["KAAL_SESSIONS_DIR"] = str(self.tempdir / "sessions")
        self._old_key = os.environ.get("OPENCODE_API_KEY")
        os.environ["OPENCODE_API_KEY"] = "test-key"

    def tearDown(self):
        if self._old_sessions_dir is None:
            os.environ.pop("KAAL_SESSIONS_DIR", None)
        else:
            os.environ["KAAL_SESSIONS_DIR"] = self._old_sessions_dir
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
        self.assertIn("kaal 0.1.0", out)

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

    def test_run_single_uses_default_ask_handler(self):
        """A single `run` passes no ask_handler to the loop: run() falls back
        to the stdin-reading default."""
        captured = {}

        class FakeAgentLoop:
            def __init__(self, *args, **kwargs):
                captured["ask_handler"] = kwargs.get("ask_handler")

            def run(self, task, emit=None):
                return "ok"

        gateway = FakeGateway([("content", "hi"), ("done", "stop")])
        with mock.patch("harness.cli.Gateway", return_value=gateway), mock.patch(
            "harness.cli.AgentLoop", FakeAgentLoop
        ):
            code, _, _ = self._run_cli(["run", "hi", "--dir", str(self.tempdir)])
        self.assertEqual(code, 0)
        self.assertIsNone(captured["ask_handler"])

    def test_run_batch_workers_get_refusing_ask_handler(self):
        """--batch workers get a handler that REFUSES ask_user — a batch
        worker must never block on stdin waiting for a user who is not there."""
        captured = []
        batch_file = self.tempdir / "batch.txt"
        batch_file.write_text("one\n", encoding="utf-8")

        class FakeAgentLoop:
            def __init__(self, *args, **kwargs):
                captured.append(kwargs.get("ask_handler"))

            def run(self, task, emit=None):
                return "ok"

        gateway = FakeGateway([("content", "hi"), ("done", "stop")])
        with mock.patch("harness.cli.Gateway", return_value=gateway), mock.patch(
            "harness.cli.AgentLoop", FakeAgentLoop
        ):
            code, _, _ = self._run_cli(
                ["run", "--batch", str(batch_file), "--dir", str(self.tempdir)]
            )
        self.assertEqual(code, 0)
        self.assertEqual(len(captured), 1)
        handler = captured[0]
        self.assertIsNotNone(handler)
        self.assertEqual(
            handler("a question", ["yes", "no"]),
            "ask_user: not available in batch mode",
        )

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

    # -- run --agent ---------------------------------------------------------

    def test_start_progress_tty_writes_and_stops(self):
        """TTY stderr gets a live elapsed progress line; non-TTY gets none
        (pipes stay clean)."""
        from harness import cli

        class FakeStderr:
            def __init__(self) -> None:
                self.buf = ""
                self.tty = True

            def isatty(self) -> bool:
                return self.tty

            def write(self, text: str) -> None:
                self.buf += text

            def flush(self) -> None:
                pass

        fake = FakeStderr()
        args = mock.Mock(verbose=False, batch=None)
        with mock.patch.object(cli.sys, "stderr", fake):
            stop = cli._start_progress(args)
            self.assertIsNotNone(stop)
            time.sleep(0.45)
            stop.set()
            time.sleep(0.1)  # let the ticker thread exit
        self.assertIn("working", fake.buf)
        fake.tty = False
        with mock.patch.object(cli.sys, "stderr", fake):
            self.assertIsNone(cli._start_progress(args))

    def _git(self, cwd: Path, *args: str) -> None:
        subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)

    def test_update_pulls_and_rebuilds(self):
        """`kaal update` pulls origin, then REBUILDS the program into the
        checkout venv (a real pip/uv install); a second run is up to date
        and skips the rebuild."""
        from harness import cli

        origin = self.tempdir / "origin"
        origin.mkdir()
        self._git(origin, "init", "-q")
        self._git(origin, "config", "user.email", "t@example.com")
        self._git(origin, "config", "user.name", "t")
        (origin / "v.txt").write_text("one")
        self._git(origin, "add", ".")
        self._git(origin, "commit", "-qm", "v1")
        checkout = self.tempdir / "checkout"
        self._git(self.tempdir, "clone", "-q", str(origin), str(checkout))
        # A newer commit lands upstream.
        (origin / "v.txt").write_text("two")
        self._git(origin, "add", ".")
        self._git(origin, "commit", "-qm", "v2")
        # The checkout looks installed: a venv python exists.
        venv_python = checkout / ".venv" / "bin" / "python"
        venv_python.parent.mkdir(parents=True)
        venv_python.write_text("#!/bin/sh\nexit 0\n")
        venv_python.chmod(0o755)

        real_run = cli._run_cmd
        real_which = cli.shutil.which
        rebuild_calls: list[list[str]] = []

        def fake_run(cmd, cwd):
            if cmd[-2:] == ["install", "."]:
                rebuild_calls.append(cmd)
                return ""
            return real_run(cmd, cwd)

        def fake_which(name):
            if name == "uv":
                return None  # force the pip rebuild path
            return real_which(name)

        args = mock.Mock()
        out = io.StringIO()
        with mock.patch.dict(os.environ, {"KAAL_INSTALL_DIR": str(checkout)}), mock.patch.object(
            cli, "_run_cmd", side_effect=fake_run
        ), mock.patch.object(cli.shutil, "which", side_effect=fake_which):
            with contextlib.redirect_stdout(out):
                code = cli._update(args)
        self.assertEqual(code, 0)
        self.assertEqual((checkout / "v.txt").read_text(), "two")
        self.assertIn("kaal updated:", out.getvalue())
        self.assertIn("v2", out.getvalue())
        self.assertIn("rebuilt into .venv", out.getvalue())
        # The rebuild ran exactly once: pip install . into the checkout venv.
        self.assertEqual(len(rebuild_calls), 1)
        self.assertEqual(rebuild_calls[0][-2:], ["install", "."])

        # Second run: nothing new upstream -> no rebuild.
        out = io.StringIO()
        with mock.patch.dict(os.environ, {"KAAL_INSTALL_DIR": str(checkout)}), mock.patch.object(
            cli, "_run_cmd", side_effect=fake_run
        ), mock.patch.object(cli.shutil, "which", side_effect=fake_which):
            with contextlib.redirect_stdout(out):
                code = cli._update(args)
        self.assertEqual(code, 0)
        self.assertIn("kaal is up to date", out.getvalue())
        self.assertEqual(len(rebuild_calls), 1)  # still exactly one rebuild

    def test_update_pull_without_venv_tells_user(self):
        """A pull with no .venv in the checkout: clear error, exit 1."""
        from harness import cli

        origin = self.tempdir / "origin"
        origin.mkdir()
        self._git(origin, "init", "-q")
        self._git(origin, "config", "user.email", "t@example.com")
        self._git(origin, "config", "user.name", "t")
        (origin / "v.txt").write_text("one")
        self._git(origin, "add", ".")
        self._git(origin, "commit", "-qm", "v1")
        checkout = self.tempdir / "checkout"
        self._git(self.tempdir, "clone", "-q", str(origin), str(checkout))
        (origin / "v.txt").write_text("two")
        self._git(origin, "add", ".")
        self._git(origin, "commit", "-qm", "v2")

        err = io.StringIO()
        with mock.patch.dict(os.environ, {"KAAL_INSTALL_DIR": str(checkout)}):
            with mock.patch("sys.stderr", new=err):
                code = cli._update(mock.Mock())
        self.assertEqual(code, 1)
        self.assertIn("no .venv in the checkout", err.getvalue())

    def test_diagrams_renders_via_termaid(self):
        """`kaal diagrams` pipes the .mmd through a termaid on PATH."""
        bin_dir = self.tempdir / "bin"
        bin_dir.mkdir()
        termaid = bin_dir / "termaid"
        termaid.write_text("#!/bin/sh\ncat \"$1\"\n")
        termaid.chmod(0o755)
        mmd = self.tempdir / "plan.mmd"
        mmd.write_text("A --> B\n")
        path = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
        with mock.patch.dict(os.environ, {"PATH": path}):
            code, out, _ = self._run_cli(["diagrams", str(mmd)])
        self.assertEqual(code, 0)
        self.assertEqual(out, "A --> B\n")

    def test_diagrams_missing_termaid_hints(self):
        """No termaid on PATH: install hint on stderr, exit 1."""
        with mock.patch.dict(os.environ, {"PATH": str(self.tempdir)}):
            code, _, err = self._run_cli(["diagrams", "x.mmd"])
        self.assertEqual(code, 1)
        self.assertIn("termaid not found", err)

    def test_update_no_checkout_reports_error(self):
        """No checkout found: clear stderr message, exit 1."""
        from harness import cli

        err = io.StringIO()
        with mock.patch.object(cli, "_resolve_checkout", return_value=None):
            with mock.patch("sys.stderr", new=err):
                code = cli._update(mock.Mock())
        self.assertEqual(code, 1)
        self.assertIn("no kaal checkout", err.getvalue())

    def test_run_agent_flag_reaches_loop(self):
        """--agent Arjuna resolves from the seeded defaults and reaches the
        constructed AgentLoop as the persona dict."""
        captured = {}

        class FakeAgentLoop:
            def __init__(self, *args, **kwargs):
                captured["agent"] = kwargs.get("agent")

            def run(self, task, emit=None):
                return "ok"

        gateway = FakeGateway([("content", "hi"), ("done", "stop")])
        with mock.patch("harness.cli.Gateway", return_value=gateway), mock.patch(
            "harness.cli.AgentLoop", FakeAgentLoop
        ):
            code, out, _ = self._run_cli(
                ["run", "hi", "--agent", "Arjuna", "--dir", str(self.tempdir)]
            )
        self.assertEqual(code, 0)
        self.assertEqual(captured["agent"]["name"], "Arjuna")

    def test_run_agent_unknown_errors_exit_1(self):
        """--agent with a name not in the agents file errors and exits 1."""
        gateway = FakeGateway([("content", "hi"), ("done", "stop")])
        with mock.patch("harness.cli.Gateway", return_value=gateway):
            code, _, err = self._run_cli(
                ["run", "hi", "--agent", "Bogus", "--dir", str(self.tempdir)]
            )
        self.assertEqual(code, 1)
        self.assertIn("kaal: no such agent: Bogus", err)
        # The error short-circuits before any gateway stream.
        self.assertEqual(gateway.calls, [])

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
