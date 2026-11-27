"""CPU-only native token/template/type checks against the actual local tokenizer."""
import math
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from transformers import AutoTokenizer

from spork_core.qwen35_native import NativeQwen35Adapter
from spork_core.vllm029_d3 import D3Bridge


class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(
            "/mnt/data/zhangshuai/LLMs/Models/Qwen3.5-9B", local_files_only=True)
        cls.tools = [{"type": "function", "function": {"name": "update",
            "parameters": {"type": "object", "properties": {
                "id": {"type": "string"}, "items": {"type": "array", "items": {"type": "string"}},
                "count": {"type": "integer"}, "enabled": {"type": "boolean"}},
                "required": ["id", "items", "count", "enabled"]}}}]
        cls.adapter = NativeQwen35Adapter(cls.tokenizer, cls.tools)
        cls.call = {"name": "update", "arguments": {
            "id": "000123", "items": ["001", "002"], "count": 2, "enabled": True}}

    def test_native_xml_roundtrip_preserves_argument_types_and_ids(self):
        body = self.adapter.speculative_tool_message(self.call)
        self.assertEqual(self.adapter.parse_main_tool_call(body), self.call)
        self.assertEqual(self.adapter.parse_main_tool_call(body + "</tool_call>"), self.call)
        probe = body.split("<function", 1)[1]
        self.assertEqual(self.adapter.parse_probe_tool_call(probe), self.call)

    def test_upstream_eos_flush_and_duplicate_resolution_with_strict_types(self):
        body = self.adapter.speculative_tool_message(self.call)
        self.assertEqual(self.adapter.parse_main_tool_call(body.replace("</function>", "")), self.call)
        self.assertIsNone(self.adapter.parse_main_tool_call(body.replace("\n2\n", "\ntrue\n")))
        duplicate = "<parameter=id>\nother\n</parameter>\n"
        expected = {"name": "update", "arguments": {**self.call["arguments"], "id": "other"}}
        self.assertEqual(self.adapter.parse_main_tool_call(body.replace("</function>", duplicate + "</function>")), expected)

    def test_implicit_tool_boundary_discards_reasoning_and_matches_canonical_history(self):
        body = self.adapter.speculative_tool_message(self.call)
        self.adapter.validate_actor_completion("private reasoning\n" + body)
        self.assertEqual(self.adapter.parse_main_tool_call("private reasoning\n" + body), self.call)
        def render(text):
            return self.adapter.render_main_prompt([
                {"role": "user", "content": "Update the item."},
                {"role": "assistant", "content": text}, {"role": "tool", "content": "ok"}],
                enable_thinking=True)
        self.assertEqual(render("private reasoning\n" + body), render(body))

    def test_natural_eos_zero_argument_header_is_supported_but_missing_required_args_are_not(self):
        adapter = NativeQwen35Adapter(self.tokenizer, [{"type": "function", "function": {
            "name": "list_all", "parameters": {"type": "object", "properties": {}}}}])
        text = "reasoning</think>\n<tool_call>\n<function=list_all"
        adapter.validate_actor_completion(text)
        self.assertEqual(adapter.parse_main_tool_call(text), {"name": "list_all", "arguments": {}})
        self.assertIsNone(self.adapter.parse_main_tool_call("<tool_call>\n<function=update"))

    def test_probe_keeps_exact_parent_tokens_and_uses_xml_prefix(self):
        prefix = self.tokenizer.encode("history<think>\n", add_special_tokens=False)
        visible = self.tokenizer.encode("visible reasoning", add_special_tokens=False)
        ids = self.adapter.prediction_prompt_ids(prefix, visible, "visible reasoning")
        self.assertEqual(ids[:len(prefix) + len(visible)], prefix + visible)
        self.assertTrue(self.tokenizer.decode(ids).endswith("</think>\n\n<tool_call>\n<function"))

    def test_forced_header_preserves_natural_merged_equal_name_token(self):
        prefix = "<tool_call>\n<function"
        whole = self.tokenizer.encode(prefix + "=get_order_details>", add_special_tokens=False)
        safe = self.tokenizer.encode(prefix, add_special_tokens=False)
        unsafe = self.tokenizer.encode(prefix + "=", add_special_tokens=False)
        self.assertEqual(whole[:len(safe)], safe)
        self.assertNotEqual(whole[:len(unsafe)], unsafe)

    def test_history_drops_completed_thinking_but_keeps_visible_text(self):
        before = [{"role": "user", "content": "Update the item."}]
        candidate = self.adapter.speculative_tool_message(self.call)
        def render(text):
            return self.adapter.render_main_prompt(before + [
                {"role": "assistant", "content": text},
                {"role": "tool", "content": '{"ok": true}'}], enable_thinking=True)
        first = render("private reasoning</think>\n" + candidate + "</tool_call>")
        self.assertEqual(first, render(candidate))
        self.assertNotIn("private reasoning", first)
        self.assertNotEqual(first, render("private reasoning</think>\nPlease confirm.\n" + candidate))

    def test_name_confidence_does_not_score_argument_tokens(self):
        logprobs = {"tokens": ["=up", "date", ">", "argument"],
                    "token_logprobs": [-.01, -.02, -.03, -20],
                    "top_logprobs": [{}, {}, {}, {}]}
        score = self.adapter.span_confidence(self.adapter.extract_probe_logprobs(logprobs))
        self.assertAlmostEqual(score, math.exp(-.03))
        logprobs["tokens"] = ["unfinished"]
        self.assertIsNone(self.adapter.span_confidence(self.adapter.extract_probe_logprobs(logprobs)))

    def test_unfinished_thinking_is_not_a_valid_actor_completion(self):
        with self.assertRaises(RuntimeError):
            self.adapter.validate_actor_completion("thinking without termination")


class Drafter:
    k = 20
    max_model_len = 64
    def propose(self, num_speculative_tokens, sampled_token_ids, num_tokens_no_spec,
                token_ids_cpu, slot_mappings=None):
        return [[]]


class D3Tests(unittest.TestCase):
    def test_qwen35_marker_reaches_real_injection_and_accounting(self):
        rid = "cmpl-native-0"
        runner = SimpleNamespace(use_async_scheduling=False, drafter=Drafter(),
            requests={rid: object()}, input_batch=SimpleNamespace(req_ids=[rid]),
            _bookkeeping_sync=lambda *args, **kwargs: [None, None, None, [[7, 8, 99]], None, [rid]])
        with patch.dict(os.environ, {"SPORK_TOOL_CALL_OPEN_TOKEN_ID": "248058"}):
            bridge = D3Bridge(runner)
        self.assertEqual(bridge.register("native", [7, 8], 1)["status"], "ok")
        proposed = bridge.propose(2, [[248058]], np.array([2]),
                                   np.array([[123, 248058, 0, 0, 0]]))
        self.assertEqual(proposed, [[7, 8]])
        bridge.bookkeeping(SimpleNamespace(scheduled_spec_decode_tokens={rid: [7, 8]}))
        status = bridge.snapshot()
        self.assertEqual(status["verified_injections"], 1)
        self.assertEqual(status["accepted_draft_tokens"], 2)
        self.assertEqual(status["accounting_errors"], 0)


if __name__ == "__main__":
    unittest.main()
