"""Wire message model tests."""

import unittest

from harness.messages import (
    AssistantMessage,
    SystemMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
    to_wire_messages,
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
