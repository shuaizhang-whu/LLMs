"""CPU-only Gemma 4 checks: call syntax, history, exact next input, probes, D3 and core hooks."""
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from spork_core import gemma4
from spork_core.evaluation_artifacts import visible_final_answers
from spork_core.gemma4 import Gemma4Adapter, format_value, parse_call, split_output
from spork_core.prefix_replay import PrefixHttpClient, action_started, reasoning_closed
from spork_core.qwen import QwenToolCallAdapter
from spork_core.successor_prefetch import generation_key
from spork_core.vllm029_d3 import D3Bridge

MODEL = Path(os.environ.get("SPORK_GEMMA_MODEL", "/mnt/data/zhangshuai/LLMs/Models/gemma-4-31B-it"))
TOOLS = [{"type": "function", "function": {"name": "find_user", "description": "Find a user.",
          "parameters": {"type": "object", "properties": {
              "first_name": {"type": "string", "description": "First name."},
              "zip": {"type": "string", "description": "Zip code."},
              "items": {"type": "array", "items": {"type": "string"}, "description": "Item ids."},
              "count": {"type": "integer", "description": "Count."},
              "express": {"type": "boolean", "description": "Express."}},
              "required": ["first_name", "zip"]}}}]
CALL = {"name": "find_user", "arguments": {"zip": "02134", "first_name": "Yusuf",
                                           "items": ["001", "a b"], "count": 2, "express": True}}
THOUGHT = "<|channel>thought\nThe user gave a name and zip code.\n<channel|>"


class SyntaxTests(unittest.TestCase):
    def test_values_roundtrip_with_types_ids_and_nesting(self):
        args = {"s": "0012", "i": 3, "f": 12.5, "t": True, "n": None, "l": [1, "x", {"k": False}],
                "d": {"inner": [], "e": {}}}
        call = parse_call("call:tool" + format_value(args))
        self.assertEqual(call, {"name": "tool", "arguments": args})
        self.assertEqual(list(call["arguments"]), list(args))

    def test_rejects_malformed_calls_and_trailing_text(self):
        for body in ("call:tool{a:1", "call:tool{a:<|\"|>x}", "call:tool{a:1}junk", "tool{a:1}",
                     "call:tool{a:1,a:2}", "call:tool{:1}"):
            with self.subTest(body=body):
                self.assertIsNone(parse_call(body))
        self.assertEqual(parse_call("call:t{a:1}<tool_call|>")["arguments"], {"a": 1})

    def test_reasoning_and_visible_parts(self):
        self.assertEqual(split_output(THOUGHT + "Done."), (THOUGHT[:-len("<channel|>")], "Done."))
        self.assertEqual(split_output("still thinking\n<|tool_call>call:t{}")[1], "<|tool_call>call:t{}")
        self.assertEqual(split_output("<|channel>thought\nunfinished"), ("<|channel>thought\nunfinished", ""))


@unittest.skipUnless((MODEL / "tokenizer.json").is_file(), "local Gemma 4 tokenizer is not available")
class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        cls.adapter = Gemma4Adapter(cls.tokenizer, TOOLS)
        cls.messages = [{"role": "system", "content": "Follow the policy."},
                        {"role": "user", "content": "Find Yusuf, zip 02134."}]

    def encode_next(self, assistant_text, observation='{"user_id": "yusuf_1"}'):
        history = self.messages + [{"role": "assistant", "content": assistant_text},
                                   {"role": "tool", "content": observation}]
        return self.tokenizer.encode(self.adapter.render_main_prompt(history, enable_thinking=True),
                                     add_special_tokens=False)

    def test_parses_turn_one_and_later_turn_outputs(self):
        body = "<|tool_call>call:find_user" + format_value(CALL["arguments"])
        self.assertEqual(self.adapter.parse_main_tool_call(THOUGHT + body), CALL)
        self.assertEqual(self.adapter.parse_main_tool_call("Next step.\n<channel|>" + body), CALL)
        self.assertIsNone(self.adapter.parse_main_tool_call(THOUGHT + body + body))
        self.assertIsNone(self.adapter.parse_main_tool_call(THOUGHT + "All done."))
        for end in ("", "<eos>", "<turn|>"):
            output = THOUGHT + "All done." + end
            answer = self.adapter.visible_answer(output)
            self.assertEqual(answer, "All done.")
            self.assertEqual(visible_final_answers({"turns": [
                {"baseline_has_tool_call": False, "baseline_text": output, "final_answer": answer}
            ]}), ["All done."])

    def test_completion_requires_a_closed_thought_or_a_parsed_call(self):
        self.adapter.validate_actor_completion(THOUGHT + "All done.")
        self.adapter.validate_actor_completion("thinking\n<|tool_call>call:find_user{zip:<|\"|>1<|\"|>}")
        for text in ("<|channel>thought\nnever closed", THOUGHT + "<|tool_call>call:find_user{zip:"):
            with self.subTest(text=text), self.assertRaises(RuntimeError):
                self.adapter.validate_actor_completion(text)

    def test_next_input_from_model_output_equals_input_from_parsed_call(self):
        # The model's own argument order differs from the template's sorted order.
        model_output = THOUGHT + "<|tool_call>call:find_user" + format_value(CALL["arguments"])
        anchored = self.adapter.speculative_tool_message(
            {"name": "find_user", "arguments": dict(sorted(CALL["arguments"].items()))})
        real_ids = self.encode_next(model_output)
        self.assertEqual(real_ids, self.encode_next(anchored))
        text = self.tokenizer.decode(real_ids, skip_special_tokens=False)
        self.assertNotIn("name and zip code", text)  # completed reasoning is dropped
        self.assertTrue(text.endswith("<tool_response|><|channel>thought\n"))

    def test_visible_text_before_the_call_changes_the_next_input(self):
        call_text = "<|tool_call>call:find_user" + format_value(CALL["arguments"])
        self.assertNotEqual(self.encode_next(THOUGHT + "Let me look this up." + call_text),
                            self.encode_next(self.adapter.speculative_tool_message(CALL)))

    def test_probe_closes_the_thought_where_it_stands_and_forces_call_prefix(self):
        prompt = self.tokenizer.encode(self.adapter.render_main_prompt(self.messages, enable_thinking=True),
                                       add_special_tokens=False)
        enc = lambda s: self.tokenizer.encode(s, add_special_tokens=False)
        first = enc("<|channel>")
        self.assertEqual(self.adapter.prediction_prompt_ids(prompt, first, "<|channel>"),
                         prompt + first + enc("thought\n<channel|><|tool_call>call:"))
        partial = enc("<|channel>thought\nThe user")
        self.assertEqual(self.adapter.prediction_prompt_ids(prompt, partial, "<|channel>thought\nThe user"),
                         prompt + partial + enc("\n<channel|><|tool_call>call:"))
        later = self.encode_next("<|tool_call>call:find_user{zip:<|\"|>1<|\"|>}")
        self.assertEqual(self.adapter.prediction_prompt_ids(later, enc("Next"), "Next"),
                         later + enc("Next") + enc("\n<channel|><|tool_call>call:"))
        closed = enc(THOUGHT)
        self.assertEqual(self.adapter.prediction_prompt_ids(prompt, closed, THOUGHT),
                         prompt + closed + enc("<|tool_call>call:"))

    def test_probe_parse_and_draft_tokens_match_the_generated_call(self):
        generated = self.tokenizer.encode("<|tool_call>call:find_user" + format_value(CALL["arguments"]),
                                          add_special_tokens=False)
        draft = self.tokenizer.encode(self.adapter.draft_body(CALL), add_special_tokens=False)
        self.assertEqual(generated[0], gemma4.CALL_OPEN_TOKEN_ID)
        self.assertEqual(generated[1:], draft)
        self.assertEqual(self.adapter.parse_probe_tool_call(self.adapter.draft_body(CALL)[len("call:"):]), CALL)

    def test_client_decode_restores_marker_text_exactly(self):
        text = THOUGHT + "<|tool_call>call:find_user" + format_value(CALL["arguments"]) + "<tool_call|>"
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        self.assertEqual(self.tokenizer.decode(ids, skip_special_tokens=False), text)

    def test_confidence_gate_is_the_qwen3_rule(self):
        logprobs = {"tokens": [f"t{i}" for i in range(20)],
                    "token_logprobs": [-0.5] + [-0.01] * 19, "top_logprobs": [{}] * 20}
        qwen = QwenToolCallAdapter(self.tokenizer, TOOLS)
        ours = self.adapter.span_confidence(self.adapter.extract_probe_logprobs(logprobs), skip_first=True)
        self.assertEqual(ours, qwen.span_confidence(qwen.extract_probe_logprobs(logprobs), skip_first=True))

    def test_token_metrics_split_at_the_call_marker(self):
        text = THOUGHT + "<|tool_call>call:find_user{zip:<|\"|>1<|\"|>}"
        metrics = self.adapter.token_metrics(text)
        self.assertEqual(metrics["cot_tokens"], len(self.tokenizer.encode(THOUGHT, add_special_tokens=False)))
        self.assertTrue(metrics["tool_call_parse_ok"])


class CoreHookTests(unittest.TestCase):
    def test_hooks_keep_qwen_markers_by_default(self):
        plain = SimpleNamespace()
        self.assertTrue(action_started(plain, "x<tool_call>"))
        self.assertFalse(action_started(plain, "<|tool_call>"))
        self.assertTrue(reasoning_closed(plain, "a</think>"))
        gemma = SimpleNamespace(formal_action_started=lambda t: "<|tool_call>" in t,
                                reasoning_closed=lambda t: "<channel|>" in t)
        self.assertTrue(action_started(gemma, "<|tool_call>"))
        self.assertFalse(reasoning_closed(gemma, "a</think>"))

    def test_request_fields_are_sent_only_when_configured(self):
        plain = PrefixHttpClient("http://h/v1", "m").body([1], max_tokens=2, seed=3, stop=("x",))
        self.assertEqual(plain, {"model": "m", "prompt": [1], "max_tokens": 2, "temperature": 0.0,
                                 "seed": 3, "stop": ["x"], "stream": False, "return_token_ids": True})
        extra = PrefixHttpClient("http://h/v1", "m", extra_body=gemma4.REQUEST_FIELDS).body(
            [1], max_tokens=2, seed=3)
        self.assertIs(extra["skip_special_tokens"], False)
        self.assertIs(extra["spaces_between_special_tokens"], False)

    def test_generation_key_adds_request_fields_only_when_present(self):
        cfg = SimpleNamespace(seed=1, baseline_max_tokens=2, context_window_tokens=3, enable_thinking=True)
        adapter = SimpleNamespace(main_stop=("</tool_call>",))
        client = SimpleNamespace(model="m", base_url="u", presence_penalty=0.0)
        self.assertNotIn("request_fields", generation_key(cfg, client, adapter))
        client.extra_body = dict(gemma4.REQUEST_FIELDS)
        self.assertEqual(generation_key(cfg, client, adapter)["request_fields"], gemma4.REQUEST_FIELDS)


class Drafter:
    k, max_model_len = 4, 100

    def propose(self, num_speculative_tokens, sampled_token_ids, num_tokens_no_spec,
                token_ids_cpu, slot_mappings=None):
        return [[]]


class D3Tests(unittest.TestCase):
    def test_gemma_call_marker_triggers_injection(self):
        rid = "cmpl-gemma-0"
        runner = SimpleNamespace(use_async_scheduling=False, drafter=Drafter(),
            requests={rid: object()}, input_batch=SimpleNamespace(req_ids=[rid]),
            _bookkeeping_sync=lambda *args, **kwargs: [None, None, None, [[7, 8, 99]], None, [rid]])
        with patch.dict(os.environ, {"SPORK_TOOL_CALL_OPEN_TOKEN_ID": str(gemma4.CALL_OPEN_TOKEN_ID)}):
            bridge = D3Bridge(runner)
        self.assertEqual(bridge.register("gemma", [7, 8], 1)["status"], "ok")
        proposed = bridge.propose(2, [[gemma4.CALL_OPEN_TOKEN_ID]], np.array([2]),
                                   np.array([[123, gemma4.CALL_OPEN_TOKEN_ID, 0, 0, 0]]))
        self.assertEqual(proposed, [[7, 8]])


@unittest.skipUnless((MODEL / "tokenizer.json").is_file(), "local Gemma 4 tokenizer is not available")
class RunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_gemma_plan_records_request_fields_and_passes_them_to_every_client(self):
        from scripts import run_successor_full_eval as runner
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        args = SimpleNamespace(output=Path(tmp.name) / "run", scope="train", task_ids="retail:51",
            configs="nospec,spork,raw", max_turns=2, repetitions=1, seed=42, max_tokens=8192, tool_floor=2.,
            model=str(MODEL), model_family="gemma4", serving_mode="invariant",
            base_url="http://127.0.0.1:8017/v1",
            task_split=runner.ROOT / "prefix_data/datasets/controller_stage3_20261008_213013_wi87e7du/task_split.json")
        manifest, tasks = runner.prepare(args)
        self.assertEqual(manifest["request_fields"], gemma4.REQUEST_FIELDS)
        self.assertEqual(manifest["history_policy"], Gemma4Adapter.history_policy)
        self.assertIn("scripts/serve_gemma4_eval.sh", manifest["code_sha256"])
        self.assertIn("spork_core/gemma4.py", manifest["code_sha256"])
        seen = []

        class HTTP:
            def __init__(self, *args, **kwargs):
                self.root = "http://127.0.0.1:8017"

            async def status(self, session):
                from spork_core.prefix_replay import COUNTERS
                return {**{k: 0 for k in COUNTERS}, "active_requests": 0, "pending_verifications": 0,
                        "server_config": {"batch_invariant_requested": True, "prefix_cache_reset_available": True}}

            async def complete(self, *args, **kwargs):
                return {"token_ids": [1], "finish_reason": "length"}

        async def task(session, client, adapter, messages, executor, make_cfg, *, mode, on_turn, trace, **kwargs):
            seen.append((type(adapter).__name__, dict(client.extra_body), mode))
            trace.update(adopted=0, errors=[], turn_limit_reached=False, pairs=[])
            on_turn({"turn": 1, "completion_valid": True, "baseline_has_tool_call": False,
                     "main_token_ids": [1], "baseline_text": "x", "num_probes": 0,
                     "d3_status": None if make_cfg(messages, 1).launch_mode == "off" else {"clear_status": "ok"}})

        async def reset(*args):
            return {"success": True, "attempts": 1, "wall_s": 0.}

        with patch.object(runner, "PrefixHttpClient", HTTP), patch.object(runner, "run_successor_task", task), \
                patch.object(runner, "reset_trial_cache", reset), \
                patch.object(runner, "tau2_action_reward", return_value={"reward": 0.}), \
                patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer):
            await runner.execute(args, manifest, tasks)
        self.assertEqual(sorted(s[2] for s in seen), ["baseline", "baseline", "raw"])
        self.assertTrue(all(s[0] == "Gemma4Adapter" and s[1] == gemma4.REQUEST_FIELDS for s in seen))


if __name__ == "__main__":
    unittest.main()
