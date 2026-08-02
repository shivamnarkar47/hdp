"""Context budget and history truncation tests."""

import unittest

from harness.config import CONTEXT_WINDOW, MAX_OUTPUT_TOKENS
from harness.context import context_budget, estimate_tokens, truncate_history
from harness.messages import (
    AssistantMessage,
    SystemMessage,
    ToolResultMessage,
    UserMessage,
)


def sample_history() -> list:
    return [
        SystemMessage("you are hdp"),
        UserMessage("old question"),
        AssistantMessage("old answer"),
        ToolResultMessage("call_1", "old result"),
        UserMessage("new question"),
        AssistantMessage("new answer"),
    ]


class TestContext(unittest.TestCase):
    def test_estimate_tokens(self):
        self.assertEqual(estimate_tokens("x" * 300), 100)
        self.assertEqual(estimate_tokens(""), 0)

    def test_budget_boundary(self):
        headroom = CONTEXT_WINDOW - MAX_OUTPUT_TOKENS
        self.assertEqual(headroom, 616_000)
        self.assertTrue(context_budget(headroom))
        self.assertFalse(context_budget(headroom + 1))

    def test_truncate_keeps_system_and_last_user(self):
        history = sample_history()
        truncated = truncate_history(history, SystemMessage("you are hdp"), max_prompt_tokens=10)
        self.assertEqual(truncated[0].text, "you are hdp")
        self.assertNotIn(history[1], truncated)  # oldest user dropped
        self.assertNotIn(history[2], truncated)  # its assistant dropped
        self.assertNotIn(history[3], truncated)  # its tool result dropped
        self.assertEqual(truncated[-2], history[4])  # last user kept
        self.assertEqual(truncated[-1], history[5])  # its assistant kept

    def test_truncate_noop_when_fits(self):
        history = sample_history()
        truncated = truncate_history(history, SystemMessage("you are hdp"), max_prompt_tokens=10**9)
        self.assertEqual(len(truncated), len(history))
        self.assertIsInstance(truncated[0], SystemMessage)

    def test_truncate_never_loses_last_user(self):
        # Even an empty budget must keep system + last user turn.
        history = sample_history()
        truncated = truncate_history(history, SystemMessage("s"), max_prompt_tokens=0)
        self.assertIn(history[4], truncated)
        self.assertIn(history[5], truncated)
