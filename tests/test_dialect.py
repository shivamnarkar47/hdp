"""DialectFeed DSML healing tests."""

from __future__ import annotations

import json
import unittest

from harness.dialect import DialectFeed, MAX_PARAMETER_CHARS
from harness.messages import ToolCall

# Unicode markers, built from escapes to avoid encoding accidents.
FW = "\uff5c"  # fullwidth pipe ｜
B = "\u2581"   # block glyph ▁

SECTION_OPEN = f"<{FW}DSML{FW}tool_calls>"
SECTION_CLOSE = f"</{FW}DSML{FW}tool_calls>"
INVOKE_OPEN = f"<{FW}DSML{FW}invoke"
INVOKE_CLOSE = f"</{FW}DSML{FW}invoke>"
PARAM_OPEN = f"<{FW}DSML{FW}parameter"
PARAM_CLOSE = f"</{FW}DSML{FW}parameter>"

FULLWIDTH_SINGLE = (
    f"{SECTION_OPEN}"
    f'{INVOKE_OPEN} name="get_weather">'
    f'{PARAM_OPEN} name="location" string="true">San Francisco, CA{PARAM_CLOSE}'
    f"{INVOKE_CLOSE}"
    f"{SECTION_CLOSE}"
)

ASCII_SINGLE = (
    "<|DSML|tool_calls>"
    '<|DSML|invoke name="add">'
    '<|DSML|parameter name="a" string="false">15</|DSML|parameter>'
    "</|DSML|invoke>"
    "</|DSML|tool_calls>"
)

OBJ_PARAM = (
    f"{SECTION_OPEN}"
    f'{INVOKE_OPEN} name="log">'
    f'{PARAM_OPEN} name="payload" string="false">{{"a":1}}{PARAM_CLOSE}'
    f"{INVOKE_CLOSE}"
    f"{SECTION_CLOSE}"
)

CHAINED = (
    f"{SECTION_OPEN}\n"
    f'\n  {INVOKE_OPEN} name="first">\n'
    f'{PARAM_OPEN} name="x" string="true">1{PARAM_CLOSE}\n'
    f"  {INVOKE_CLOSE}\n"
    f'{INVOKE_OPEN} name="second">'
    f'{PARAM_OPEN} name="y" string="false">2{PARAM_CLOSE}'
    f"{INVOKE_CLOSE}\n"
    f"{SECTION_CLOSE}"
)

LEAK_IN_PROSE = (
    f"The answer is 42{SECTION_OPEN}"
    f'{INVOKE_OPEN} name="get_weather">'
    f'{PARAM_OPEN} name="location" string="true">San Francisco, CA{PARAM_CLOSE}'
    f"{INVOKE_CLOSE}"
    f"{SECTION_CLOSE} done"
)

LEAKED_TOKENS = f"<{FW}begin{B}of{B}sentence{FW}>hello<{FW}Assistant{FW}>"

THINK_SPAN = "<think>let me check</think>The weather is 18°C"

UNCLOSED_SECTION = f'{SECTION_OPEN}{INVOKE_OPEN} name="x">'

TRUNCATED_SUFFIX = "\u2026[parameter truncated]"  # fullwidth ellipsis …


def feed_charwise(text: str) -> list:
    """Feed a fixture one character at a time, then flush."""
    feed = DialectFeed()
    events: list = []
    for ch in text:
        events.extend(feed.feed(ch))
    events.extend(feed.flush())
    return events


def canonicalize(events: list) -> list:
    """Collapse consecutive text/reasoning runs; keep tool calls intact."""
    out: list = []
    for kind, payload in events:
        if kind == "tool_call":
            out.append((kind, payload))
        elif out and out[-1][0] == kind:
            out[-1] = (kind, out[-1][1] + payload)
        else:
            out.append((kind, payload))
    return out


class TestDialectFeed(unittest.TestCase):
    def test_fullwidth_single_call(self):
        events = DialectFeed().feed(FULLWIDTH_SINGLE)
        self.assertEqual(len(events), 1)
        self.assertEqual(
            events[0], ("tool_call", ToolCall("call_1", "get_weather", '{"location": "San Francisco, CA"}'))
        )
        self.assertFalse(any(kind == "text" for kind, _ in events))

    def test_ascii_variant(self):
        events = DialectFeed().feed(ASCII_SINGLE)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0][1], ToolCall("call_1", "add", '{"a": 15}'))

    def test_false_string_object(self):
        events = DialectFeed().feed(OBJ_PARAM)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0][1], ToolCall("call_1", "log", '{"payload": {"a": 1}}'))

    def test_chained_invokes_order(self):
        events = DialectFeed().feed(CHAINED)
        calls = [payload for kind, payload in events if kind == "tool_call"]
        self.assertEqual(
            calls,
            [
                ToolCall("call_1", "first", '{"x": "1"}'),
                ToolCall("call_2", "second", '{"y": 2}'),
            ],
        )

    def test_call_counter_persists_across_sections(self):
        feed = DialectFeed()
        first = feed.feed(FULLWIDTH_SINGLE)
        second = feed.feed(FULLWIDTH_SINGLE)
        self.assertEqual(first[0][1].id, "call_1")
        self.assertEqual(second[0][1].id, "call_2")

    def test_leak_in_prose(self):
        events = DialectFeed().feed(LEAK_IN_PROSE)
        self.assertEqual([kind for kind, _ in events], ["text", "tool_call", "text"])
        self.assertEqual(events[0], ("text", "The answer is 42"))
        self.assertEqual(events[2], ("text", " done"))
        joined_text = "".join(payload for kind, payload in events if kind == "text")
        self.assertNotIn("\uff5c", joined_text)
        self.assertNotIn("DSML", joined_text)

    def test_leaked_chat_tokens(self):
        events = DialectFeed().feed(LEAKED_TOKENS)
        self.assertEqual(events, [("text", "hello")])

    def test_think_span(self):
        events = DialectFeed().feed(THINK_SPAN)
        reasoning = "".join(payload for kind, payload in events if kind == "reasoning")
        text = "".join(payload for kind, payload in events if kind == "text")
        self.assertEqual(reasoning, "let me check")
        self.assertEqual(text, "The weather is 18°C")

    def test_unclosed_section_flush(self):
        feed = DialectFeed()
        self.assertEqual(feed.feed(UNCLOSED_SECTION), [])
        self.assertEqual(feed.flush(), [])

    def test_boundary_safety(self):
        for fixture in (FULLWIDTH_SINGLE, ASCII_SINGLE, CHAINED, LEAK_IN_PROSE, THINK_SPAN):
            with self.subTest(fixture=repr(fixture[:40])):
                whole = DialectFeed()
                whole_events = whole.feed(fixture) + whole.flush()
                self.assertEqual(canonicalize(feed_charwise(fixture)), canonicalize(whole_events))

    def test_parameter_cap(self):
        value = "a" * 1_000_001
        feed = DialectFeed()
        events: list = []
        prefix = (
            f"{SECTION_OPEN}"
            f'{INVOKE_OPEN} name="big">'
            f'{PARAM_OPEN} name="data" string="true">'
        )
        events.extend(feed.feed(prefix))
        chunk = 100_000
        for start in range(0, len(value), chunk):
            events.extend(feed.feed(value[start : start + chunk]))
        events.extend(feed.feed(f"{PARAM_CLOSE}{INVOKE_CLOSE}{SECTION_CLOSE}"))
        calls = [payload for kind, payload in events if kind == "tool_call"]
        self.assertEqual(len(calls), 1)
        kept = json.loads(calls[0].arguments)["data"]
        self.assertTrue(kept.endswith(TRUNCATED_SUFFIX))
        self.assertEqual(len(kept), MAX_PARAMETER_CHARS + len(TRUNCATED_SUFFIX))
        self.assertTrue(kept.startswith("a" * MAX_PARAMETER_CHARS))

    def test_stray_lt_passthrough(self):
        events = DialectFeed().feed("a < b")
        self.assertEqual(events, [("text", "a < b")])


if __name__ == "__main__":
    unittest.main()
