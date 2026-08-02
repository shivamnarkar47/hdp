"""Session store tests: ids, read_events, delete_session, prune_sessions."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from harness.sessions import (
    append_event,
    delete_session,
    get_store_dir,
    list_sessions,
    load_messages,
    new_session_id,
    prune_sessions,
    read_events,
)


class SessionStoreTestCase(unittest.TestCase):
    """Base with a private, disposable HARNESSDP_SESSIONS_DIR per test."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._old_sessions_dir = os.environ.get("HARNESSDP_SESSIONS_DIR")
        os.environ["HARNESSDP_SESSIONS_DIR"] = self._tmp.name

    def tearDown(self) -> None:
        if self._old_sessions_dir is None:
            os.environ.pop("HARNESSDP_SESSIONS_DIR", None)
        else:
            os.environ["HARNESSDP_SESSIONS_DIR"] = self._old_sessions_dir


class TestNewSessionId(unittest.TestCase):
    def test_ids_are_unique_to_the_microsecond(self):
        first, second = new_session_id(), new_session_id()
        self.assertNotEqual(first, second)
        # A burst must stay collision-free: the id carries microsecond precision.
        ids = {new_session_id() for _ in range(100)}
        self.assertEqual(len(ids), 100)

    def test_id_is_a_plain_string(self):
        self.assertIsInstance(new_session_id(), str)


class TestReadEvents(SessionStoreTestCase):
    def test_round_trip(self):
        sid = new_session_id()
        append_event(sid, {"type": "user", "data": {"content": "hello"}})
        append_event(sid, {"type": "assistant", "data": {"content": "hi"}})
        append_event(
            sid,
            {"type": "tool_result", "data": {"tool_call_id": "call_1", "content": "ok"}},
        )
        events = read_events(sid)
        self.assertEqual(len(events), 3)
        self.assertEqual([e["type"] for e in events], ["user", "assistant", "tool_result"])
        self.assertTrue(events[0]["ts"])
        self.assertEqual(events[0]["data"], {"content": "hello"})
        self.assertEqual(events[1]["data"], {"content": "hi"})
        self.assertEqual(events[2]["data"], {"tool_call_id": "call_1", "content": "ok"})

    def test_missing_session_yields_empty(self):
        self.assertEqual(read_events("20260802-000000-000000"), [])

    def test_skips_blank_and_corrupt_lines(self):
        sid = "20260802-000000-000000"
        path = get_store_dir() / f"{sid}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            '{"ts": "t1", "type": "user", "data": {"content": "ok"}}\n'
            "\n"
            "THIS IS NOT JSON\n"
            '{"ts": "t2", "type": "assistant", "data": {"content": "hi"}}\n',
            encoding="utf-8",
        )
        events = read_events(sid)
        self.assertEqual([e["data"]["content"] for e in events], ["ok", "hi"])


class TestDeleteSession(SessionStoreTestCase):
    def test_delete_returns_true_and_removes_file(self):
        sid = "20260802-000000-000000"
        append_event(sid, {"type": "user", "data": {"content": "x"}})
        path = get_store_dir() / f"{sid}.jsonl"
        self.assertTrue(path.is_file())
        self.assertTrue(delete_session(sid))
        self.assertFalse(path.exists())

    def test_delete_missing_returns_false(self):
        self.assertFalse(delete_session("20260802-000000-000000"))

    def test_delete_leaves_other_files_untouched(self):
        drop = "20260802-000000-000000"
        keep = "20260802-000000-000001"
        append_event(drop, {"type": "user", "data": {"content": "drop"}})
        append_event(keep, {"type": "user", "data": {"content": "keep"}})
        other = get_store_dir() / "notes.txt"
        other.write_text("not a session file\n", encoding="utf-8")
        self.assertTrue(delete_session(drop))
        self.assertTrue((get_store_dir() / f"{keep}.jsonl").is_file())
        self.assertFalse((get_store_dir() / f"{drop}.jsonl").exists())
        self.assertTrue(other.is_file())


class TestPruneSessions(SessionStoreTestCase):
    def test_prune_keeps_newest_and_returns_deleted_ids(self):
        ids = [f"20260802-{i:06d}" for i in range(4)]  # increasing by id
        for sid in ids:
            append_event(sid, {"type": "user", "data": {"content": sid}})
        deleted = prune_sessions(keep=2)
        self.assertEqual(deleted, ids[:2])
        remaining = sorted(p.stem for p in get_store_dir().glob("*.jsonl"))
        self.assertEqual(remaining, ids[2:])

    def test_prune_zero_deletes_everything(self):
        for i in range(3):
            append_event(f"20260802-{i:06d}", {"type": "user", "data": {"content": "x"}})
        deleted = prune_sessions(keep=0)
        self.assertEqual(len(deleted), 3)
        self.assertEqual(list_sessions(), [])

    def test_prune_negative_keep_deletes_everything(self):
        append_event("20260802-000000", {"type": "user", "data": {"content": "x"}})
        self.assertEqual(prune_sessions(keep=-1), ["20260802-000000"])

    def test_prune_keep_beyond_count_deletes_nothing(self):
        for i in range(2):
            append_event(f"20260802-{i:06d}", {"type": "user", "data": {"content": "x"}})
        self.assertEqual(prune_sessions(keep=10), [])
        self.assertEqual(len(list_sessions()), 2)

    def test_prune_empty_store(self):
        self.assertEqual(prune_sessions(keep=20), [])


class TestRefactorCompatibility(SessionStoreTestCase):
    """load_messages / list_sessions still behave after the reader refactor."""

    def test_load_messages_and_list_sessions_round_trip(self):
        sid = "20260802-000000-000042"
        append_event(sid, {"type": "user", "data": {"content": "hello"}})
        append_event(sid, {"type": "assistant", "data": {"content": "hi"}})
        append_event(
            sid,
            {"type": "tool_result", "data": {"tool_call_id": "c1", "content": "ok"}},
        )
        self.assertEqual(
            load_messages(sid),
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "tool", "tool_call_id": "c1", "content": "ok"},
            ],
        )
        sessions = list_sessions()
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["id"], sid)
        self.assertTrue(sessions[0]["ts"])
        self.assertEqual(sessions[0]["prompt"], "hello")

    def test_load_messages_skips_corrupt_via_shared_reader(self):
        sid = "20260802-000000-000043"
        path = get_store_dir() / f"{sid}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            '{"ts": "t", "type": "user", "data": {"content": "ok"}}\n'
            "NOT JSON\n",
            encoding="utf-8",
        )
        self.assertEqual(load_messages(sid), [{"role": "user", "content": "ok"}])


if __name__ == "__main__":
    unittest.main()
