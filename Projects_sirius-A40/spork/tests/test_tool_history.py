import json
import unittest

from spork_core.qwen import parse_baseline_tool_call
from spork_core.turn_runner import _append_tool_messages


class ToolHistoryTests(unittest.TestCase):
    def test_equal_calls_ignore_generated_closers_whitespace_and_argument_order(self):
        first = {"name": "lookup", "arguments": {"z": [1, 2], "a": {"y": 1, "x": "中文"}}}
        second = {"name": "lookup", "arguments": {"a": {"x": "中文", "y": 1}, "z": [1, 2]}}
        histories = []
        for call, suffix in [(first, ""), (second, "\n</tool_response>")]:
            raw = "<think>private thought</think>\n<tool_call>" + json.dumps(call) + suffix
            messages = []
            _append_tool_messages(messages, raw, '{"source":"real tool"}', canonical_call=call)
            histories.append(messages)
        self.assertEqual(*histories)
        text = histories[0][0]["content"]
        self.assertTrue(text.endswith("</tool_call>"))
        self.assertNotIn("</tool_response>", text)
        self.assertNotIn("private thought", text)
        self.assertEqual(parse_baseline_tool_call(text), first)
        self.assertEqual(histories[0][1], {"role": "tool", "content": '{"source":"real tool"}'})

    def test_actual_argument_changes_remain_visible(self):
        histories = []
        for values in ([1, 2], [2, 1]):
            messages = []
            _append_tool_messages(messages, "raw", "real", canonical_call={
                "name": "lookup", "arguments": {"ids": values}})
            histories.append(messages)
        self.assertNotEqual(*histories)

    def test_legacy_path_stays_available_for_reproducing_old_protocol(self):
        messages = []
        _append_tool_messages(messages, "<think>x</think>\n<tool_call>{}", "real")
        self.assertEqual(messages[0]["content"], "<tool_call>{}")


if __name__ == "__main__":
    unittest.main()
