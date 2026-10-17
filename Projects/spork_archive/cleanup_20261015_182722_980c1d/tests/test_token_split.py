"""Offline regression checks with the actual Qwen3 tokenizer."""
import os
import unittest
from transformers import AutoTokenizer
from spork_core.qwen import token_split


class TokenSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = AutoTokenizer.from_pretrained(
            os.environ.get("SPORK_TEST_TOKENIZER", "/mnt/data/zhangshuai/LLMs/Models/Qwen3-32B"),
            local_files_only=True,
        )

    def check_parts(self, text):
        m = token_split(text, self.tok)
        self.assertEqual(m["total_decode_tokens"], len(self.tok.encode(text, add_special_tokens=False)))
        self.assertEqual(m["cot_tokens"] + m["tool_call_tokens"] + m["post_tool_call_tokens"], m["total_decode_tokens"])
        return m

    def test_complete_body_without_stop_string(self):
        prefix = "Let me check."
        body = '<tool_call>\n{"name":"get_order_details","arguments":{"order_id":"#123"}}'
        m = self.check_parts(prefix + body)
        self.assertEqual(m["cot_tokens"], len(self.tok.encode(prefix, add_special_tokens=False)))
        self.assertGreater(m["tool_call_tokens"], 0)
        self.assertFalse(m["tool_call_closed"])
        self.assertTrue(m["tool_call_parse_ok"])

    def test_closed_call_and_suffix(self):
        m = self.check_parts('<tool_call>{"name":"lookup","arguments":{}}</tool_call>Done.')
        self.assertTrue(m["tool_call_closed"])
        self.assertTrue(m["tool_call_parse_ok"])
        self.assertGreater(m["post_tool_call_tokens"], 0)

    def test_truncated_body_is_not_complete(self):
        m = self.check_parts('<tool_call>{"name":"lookup","arguments":')
        self.assertGreater(m["tool_call_tokens"], 0)
        self.assertFalse(m["tool_call_parse_ok"])
        self.assertFalse(m["tool_call_closed"])

    def test_plain_answer_and_empty(self):
        for text in ("Here is the answer.", ""):
            m = self.check_parts(text)
            self.assertEqual(m["tool_call_tokens"], 0)
            self.assertEqual(m["cot_tokens"], m["total_decode_tokens"])

    def test_stray_close_before_open(self):
        m = self.check_parts('</tool_call>text<tool_call>{"name":"lookup"}</tool_call>suffix')
        self.assertTrue(m["tool_call_closed"])
        self.assertGreater(m["tool_call_tokens"], 0)
        self.assertGreater(m["post_tool_call_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
