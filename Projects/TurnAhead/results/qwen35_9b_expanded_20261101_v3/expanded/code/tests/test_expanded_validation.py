import copy
import json
from pathlib import Path
import unittest
import asyncio
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from spork_core.expanded_validation import OLD_TASKS, select_tasks, signature
from spork_core.local_tau2_judge import LocalNLJudge, validate_judgment

ROOT = Path(__file__).resolve().parents[1]


class Tests(unittest.TestCase):
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
