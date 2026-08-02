"""Wire message model tests."""

import json
import unittest

from harness.context import estimate_tokens, wire_token_count
from harness.messages import (
    AssistantMessage,
    SystemMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
    to_wire_messages,
    wire_token_cost,
)


class TestMessages(unittest.TestCase):
    def test_assistant_wire_with_reasoning_and_tool_calls(self):
        msg = AssistantMessage(
            content="",
            reasoning_content="let me think",
            tool_calls=[ToolCall("call_1", "get_weather", '{"location": "San Francisco, CA"}')],
        )
        wire = msg.to_wire()
        self.assertEqual(wire["role"], "assistant")
        self.assertEqual(wire["content"], "")
        self.assertEqual(wire["reasoning_content"], "let me think")
        self.assertEqual(
            wire["tool_calls"],
            [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"location": "San Francisco, CA"}'},
                }
            ],
        )
        self.assertNotIn("tool_choice", wire)

    def test_assistant_wire_omits_optional_fields(self):
        wire = AssistantMessage(content="hi").to_wire()
        self.assertNotIn("reasoning_content", wire)
        self.assertNotIn("tool_calls", wire)

    def test_tool_result_wire(self):
        wire = ToolResultMessage("call_1", "ok").to_wire()
        self.assertEqual(wire, {"role": "tool", "tool_call_id": "call_1", "content": "ok"})

    def test_system_messages_coalesced(self):
        msgs = [SystemMessage("first"), UserMessage("q"), SystemMessage("second")]
        wire = to_wire_messages(msgs)
        self.assertEqual(len(wire), 2)
        self.assertEqual(wire[0], {"role": "system", "content": "first\n\nsecond"})
        self.assertEqual(wire[1], {"role": "user", "content": "q"})

    def test_user_wire(self):
        self.assertEqual(UserMessage("hello").to_wire(), {"role": "user", "content": "hello"})


class TestWireTokenCost(unittest.TestCase):
    def _sample_wire(self):
        messages = [
            SystemMessage("you are hdp — the harness agent"),
            UserMessage("read the file, then summarize it carefully"),
            AssistantMessage(
                content="Let me look.",
                reasoning_content="first read, then summarize",
                tool_calls=[ToolCall("c1", "read", '{"path": "a.txt"}')],
            ),
            ToolResultMessage("c1", "file contents here"),
        ]
        return to_wire_messages(messages)

    def test_matches_sum_of_per_message_costs(self):
        wire = self._sample_wire()
        expected = sum(estimate_tokens(json.dumps(m, ensure_ascii=False)) for m in wire)
        self.assertEqual(wire_token_cost(wire), expected)

    def test_empty_wire_costs_zero(self):
        self.assertEqual(wire_token_cost([]), 0)

    def test_context_wire_token_count_delegates(self):
        wire = self._sample_wire()
        self.assertEqual(wire_token_count(wire), wire_token_cost(wire))
