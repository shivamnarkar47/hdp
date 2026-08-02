"""Context budget and history truncation tests."""

import json
import random
import unittest

from harness.config import CONTEXT_WINDOW, MAX_OUTPUT_TOKENS
from harness.context import (
    context_budget,
    estimate_tokens,
    message_token_costs,
    truncate_history,
    wire_token_count,
)
from harness.messages import (
    AssistantMessage,
    SystemMessage,
    ToolResultMessage,
    UserMessage,
)


def reference_truncate_history(messages, system_message, max_prompt_tokens):
    """Reference implementation of the pre-ledger semantics (re-serializes per
    dropped turn). Used only to pin the ledger rewrite to identical output."""
    result = [m for m in messages if not isinstance(m, SystemMessage)]

    def count(msgs):
        return sum(
            estimate_tokens(json.dumps(m.to_wire(), ensure_ascii=False)) for m in msgs
        )

    while count(result) > max_prompt_tokens:
        last_user = max(i for i, m in enumerate(result) if isinstance(m, UserMessage))
        candidates = [
            i for i, m in enumerate(result) if isinstance(m, UserMessage) and i != last_user
        ]
        if not candidates:
            break
        start = candidates[0]
        end = next(
            (i for i in range(start + 1, len(result)) if isinstance(result[i], UserMessage)),
            len(result),
        )
        del result[start:end]

    return [system_message, *result]


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


class TestTruncationLedger(unittest.TestCase):
    def test_equivalent_to_reference_on_random_histories(self):
        """The ledger rewrite drops EXACTLY the same turns as the old
        re-serializing implementation, across varied random histories."""
        rng = random.Random(42)
        for _ in range(80):
            messages = [SystemMessage("you are hdp")]
            for _t in range(rng.randint(3, 12)):
                messages.append(UserMessage("u" * rng.randint(1, 4000)))
                if rng.random() < 0.8:
                    reasoning = "r" * rng.randint(0, 500)
                    messages.append(
                        AssistantMessage(
                            "a" * rng.randint(1, 3000), reasoning or None
                        )
                    )
                    if rng.random() < 0.8:
                        messages.append(ToolResultMessage("c", "t" * rng.randint(1, 3000)))
            system = messages[0]
            budget = rng.choice([0, 1, 500, 5_000, 50_000, 616_000, 10**9])
            new = truncate_history(messages, system, budget)
            old = reference_truncate_history(messages, system, budget)
            self.assertEqual(new, old)

    def test_message_token_costs_formula(self):
        messages = [
            SystemMessage("sys"),
            UserMessage("user text"),
            AssistantMessage("assistant", reasoning_content="r"),
            ToolResultMessage("c", "tool"),
        ]
        costs = message_token_costs(messages)
        for msg, cost in zip(messages, costs):
            self.assertEqual(
                cost, estimate_tokens(json.dumps(msg.to_wire(), ensure_ascii=False))
            )
        # The ledger total equals the wire count of the same messages.
        self.assertEqual(sum(costs), wire_token_count([m.to_wire() for m in messages]))
