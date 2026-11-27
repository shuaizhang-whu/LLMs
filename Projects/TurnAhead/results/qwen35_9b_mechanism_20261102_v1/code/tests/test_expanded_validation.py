import copy
import json
from pathlib import Path
import unittest
import asyncio
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from spork_core.expanded_validation import (OLD_TASKS, CONFIGS, actor_task_failure,
    analyze, d3_integrity_errors, normalized_messages, select_tasks, signature)
from spork_core.evaluation_artifacts import complete_record, digest, record_path, write_json
from spork_core.local_tau2_judge import LocalNLJudge, validate_judgment

ROOT = Path(__file__).resolve().parents[1]


class Tests(unittest.TestCase):
    def failure(self):
        return {"failure_kind": "actor_generation", "termination_reason": "actor_generation_failure",
            "errors": ["interaction: ActorGenerationError: missing reasoning boundary"],
            "failed_generation": {"text": "thinking only", "token_ids": [1],
                "prompt_token_ids": [2], "generation_settings": {"seed": 42}, "finish_reason": "stop"}}

    def test_only_traced_actor_failures_are_nonfatal(self):
        self.assertTrue(actor_task_failure(self.failure()))
        for changes in ({"cleanup_errors": ["failed"]}, {"d3_counters": {"dropped_injections": 1}},
                        {"errors": ["interaction: RuntimeError: bad"]}, {"failed_generation": {}},
                        {"errors": self.failure()["errors"] + ["server_integrity: failure"]}):
            self.assertFalse(actor_task_failure({**self.failure(), **changes}))

    def test_failed_task_is_counted_and_excluded_without_stopping_remaining_schedule(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            rows = []
            for mode in CONFIGS:
                row = {"group_id": "airline:9", "domain": "airline", "task_id": "9", "split": "train",
                    "repeat": 0, "config": mode, "session_id": "s", "turns": [], "user_calls": [],
                    "errors": [], "agent_pipeline_s": 1., "wall_s": 1., "user_wall_s": 0.,
                    "reward_info": {"reward": 1.}, "termination_reason": "user_stop"}
                if mode == "raw":
                    row.update(self.failure(), reward_info=None)
                else:
                    row["turns"] = [{"prompt_token_ids": [2], "generation_settings": {"seed": 42},
                        "baseline_has_tool_call": False, "baseline_text": "</think>Done"}]
                messages = [{"role": "assistant", "content": "Welcome"}]
                if not row["errors"]:
                    messages.append({"role": "assistant", "content": "Done"})
                row["trajectory_path"] = f"trajectories/{mode}.json"
                write_json(output / row["trajectory_path"], {"messages": messages})
                row["trajectory_sha256"] = digest(output / row["trajectory_path"])
                row["normalized_messages"] = normalized_messages(messages)
                complete_record(record_path(output, row), row)
                rows.append(row)
            manifest = {"stage": "expanded", "scope": "test", "schedule": rows}
            report = analyze(output, manifest)
            self.assertTrue(report["complete"])
            self.assertEqual(report["gate_errors"], [])
            self.assertEqual(len(report["task_failures"]), 1)
            self.assertFalse(report["all_outcomes_identical"])
            self.assertEqual(report["common_pairs"], 0)
            self.assertEqual(report["methods"]["raw"]["mean_reward_counting_unscored_as_zero"], 0.)
            self.assertIsNone(report["methods"]["raw"]["mean_native_reward_scored"])
            self.assertTrue(analyze(output, {**manifest, "stage": "preflight"})["gate_errors"])

    def test_d3_prefix_miss_rejects_before_injection_and_is_not_an_integrity_error(self):
        import numpy as np
        from spork_core.vllm029_d3 import D3Bridge
        class Drafter:
            k = 2
            max_model_len = 100
            def propose(self, num_speculative_tokens, sampled_token_ids, *args, **kwargs):
                return [[88, 89]]
        runner = SimpleNamespace(drafter=Drafter(), requests={"cmpl-r-0":None},
            input_batch=SimpleNamespace(req_ids=["cmpl-r-0"]), _bookkeeping_sync=lambda *a, **k:None)
        bridge = D3Bridge(runner)
        bridge.register("r", [10,11], 1)
        ids = np.array([[7, bridge.tool_open_token_id, 99]])
        drafts = bridge.propose(2, [[99]], [3], ids)
        self.assertEqual(drafts, [[88,89]])
        self.assertEqual(bridge.pending, {})
        self.assertEqual(bridge.counts["inject_count"], 0)
        self.assertEqual(bridge.counts["prefix_mismatches"], 1)
        self.assertEqual(d3_integrity_errors(bridge.counts), [])
        for key in ("accounting_errors", "dropped_injections"):
            self.assertEqual(d3_integrity_errors({key:1}), [key])

    def test_selection_is_fixed_training_only_and_preserves_previous_eight(self):
        split = json.loads((ROOT / "prefix_data/datasets/controller_stage3_20261008_213013_wi87e7du/task_split.json").read_text())
        rows = select_tasks(split)
        self.assertEqual(rows, select_tasks(split))
        self.assertEqual(len(rows), 32)
        self.assertEqual(len({r["group_id"] for r in rows}), 32)
        self.assertTrue(all(r["split"] == "train" for r in rows))
        self.assertEqual({r["group_id"] for r in rows if r["cohort"] == "previous8"}, set(OLD_TASKS))
        self.assertEqual(sum(r["domain"] == "airline" for r in rows), 16)

    def test_judge_cannot_pass_empty_partial_duplicate_or_truncated_assertions(self):
        def row(text, result=True):
            return {"expectedOutcome":text,"metExpectation":result,"reasoning":"test"}
        expected = ["one", "two"]
        for rows in [[], [row("one")], [row("one"),row("one")], [row("one"),row("two","true")]]:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                validate_judgment(json.dumps({"results":rows}),expected)
        self.assertTrue(validate_judgment(json.dumps({"results":[row("two",False),row("one")]}),expected))

    def test_user_request_state_and_reward_changes_are_part_of_equality(self):
        record = {"turns":[],"errors":[],"user_calls":[{"prompt_token_ids":[1],"token_ids":[2],
            "text":"hello","finish_reason":"stop","generation_settings":{"seed":42}}],
            "normalized_messages":[],"final_environment":["a",None],"termination_reason":"user_stop",
            "reward_info":{"reward":1.,"reward_breakdown":{"DB":1.}}}
        for mutate in [lambda r:r['user_calls'][0].update(prompt_token_ids=[3]),
                       lambda r:r.update(final_environment=['b',None]),
                       lambda r:r['reward_info'].update(reward=0.),
                       lambda r:r.update(termination_reason='max_steps')]:
            other = copy.deepcopy(record); mutate(other)
            self.assertNotEqual(signature(record), signature(other))

    def test_local_judge_uses_native_prompt_and_sealed_exact_input_cache(self):
        from spork_core.prefix_replay import PrefixHttpClient
        from tau2.evaluator.evaluator_nl_assertions import NLAssertionsEvaluator
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return json.dumps(messages, sort_keys=True)
            def encode(self, text, **kwargs):
                return list(text.encode())
        http = PrefixHttpClient("http://unused/v1", "cpu-model", 30)
        calls = []
        async def complete(*args, **kwargs):
            calls.append(kwargs)
            return {"text":json.dumps({"results":[{"expectedOutcome":"one", "metExpectation":False,"reasoning":"test"}]}),
                    "finish_reason":"stop", "token_ids":[1], "wall_s":0.}
        http.complete = complete
        runtime = SimpleNamespace(judge_http=http, run=asyncio.run, session=None)
        task = SimpleNamespace(evaluation_criteria=SimpleNamespace(nl_assertions=["one"]))
        def native_nl_only(simulation, task, *args, **kwargs):
            return NLAssertionsEvaluator.calculate_reward(task, [])
        with tempfile.TemporaryDirectory() as folder, patch("spork_core.local_tau2_judge.evaluate_simulation", native_nl_only):
            judge = LocalNLJudge(runtime, Tokenizer(), folder)
            self.assertEqual(judge.evaluate(None, task, "retail").reward, 0.)
            self.assertEqual(judge.evaluate(None, task, "retail").reward, 0.)
            self.assertEqual(len(calls), 1)
            self.assertTrue(judge.calls[-1]["cached"])
            cached = next(Path(folder).glob("*.json"))
            cached.write_text("{}")
            with self.assertRaisesRegex(ValueError, "modified"):
                judge.evaluate(None, task, "retail")


if __name__ == "__main__":
    unittest.main()
