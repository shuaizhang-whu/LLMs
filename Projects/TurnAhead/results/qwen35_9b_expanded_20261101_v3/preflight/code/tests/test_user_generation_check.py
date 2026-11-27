"""Simulator request policy and gates; local tokenizer, no model inference."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from transformers import AutoTokenizer
from spork_core.evaluation_artifacts import digest, write_json
from spork_core.interactive_successor import InteractiveRuntime, validate_user_completion
from spork_core.user_generation_check import initial_user, check_usercheck, run_usercheck
from tau2.data_model.tasks import Task

ROOT = Path(__file__).resolve().parents[1]
MODEL = "/mnt/data/zhangshuai/LLMs/Models/Qwen3.5-9B"


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        raw = json.loads((ROOT / "tau2-bench/data/tau2/domains/airline/tasks.json").read_text())
        cls.task = Task.model_validate(next(r for r in raw if str(r["id"]) == "2"))

    def runtime(self):
        runtime = InteractiveRuntime("http://127.0.0.1:1/v1", MODEL)
        self.addCleanup(runtime.close)
        return runtime

    def test_role_policies_are_separate(self):
        runtime = self.runtime()
        kw = dict(max_tokens=4096, seed=42, stop=["<|im_end|>"])
        self.assertEqual(runtime.http.body([1], **kw)["presence_penalty"], 1.5)
        self.assertEqual(runtime.user_http.body([1], **kw)["presence_penalty"], 1.5)
        self.assertEqual(runtime.judge_http.body([1], **kw).get("presence_penalty", 0.), 0.)

    def test_actual_simulator_request_matches_failed_input_except_penalty(self):
        runtime = self.runtime()
        seen = []
        async def complete(session, ids, **kw):
            seen.append(runtime.user_http.body(ids, **kw))
            return {"text":"reasoning</think>Hi, I'd like to book a flight.",
                    "token_ids":[1,2], "finish_reason":"stop", "wall_s":0.}
        runtime.user_http.complete = complete
        user, state, message = initial_user(runtime, self.tokenizer, self.task, 42)
        reply, _ = user.generate_next_message(message, state)
        self.assertEqual(reply.content, "Hi, I'd like to book a flight.")
        self.assertEqual(seen[0]["presence_penalty"], 1.5)
        self.assertEqual(seen[0]["temperature"], 0.)
        self.assertEqual(seen[0]["max_tokens"], 4096)
        source = ROOT / "results/qwen35_9b_expanded_20261101_v2/preflight/records/airline_2_r0_spork.json"
        if source.exists():
            old = json.loads(source.read_text())["user_calls"][0]
            self.assertEqual(user.calls[0]["prompt_token_ids"], old["prompt_token_ids"])
            self.assertEqual({k:v for k,v in user.calls[0]["generation_settings"].items()
                              if k != "presence_penalty"}, old["generation_settings"])

    def test_transport_failure_keeps_input_and_user_time(self):
        runtime = self.runtime()
        async def fail(*args, **kwargs):
            raise RuntimeError("synthetic transport failure")
        runtime.user_http.complete = fail
        user, state, message = initial_user(runtime, self.tokenizer, self.task, 42)
        with self.assertRaisesRegex(RuntimeError, "transport"):
            user.generate_next_message(message, state)
        call = user.calls[0]
        self.assertTrue(call["prompt_token_ids"])
        self.assertEqual(call["generation_settings"]["presence_penalty"], 1.5)
        self.assertGreater(call["participant_wall_s"], 0.)
        self.assertIn("synthetic", call["error"])

    def test_truncation_is_not_rescued_by_a_quoted_reply(self):
        for choice in [
            {"text":'I could say "Hi".\n' * 100, "finish_reason":"length"},
            {"text":'I could say "Hi".', "finish_reason":"stop"},
            {"text":"reasoning</think>", "finish_reason":"stop"},
        ]:
            with self.subTest(choice=choice), self.assertRaises(RuntimeError):
                validate_user_completion(choice, thinking=True)

    def records(self, directory, *, different=False, failed=False):
        p = Path(directory)
        spec = {"prompt_token_ids":[1],"generation_settings":{"seed":42,"presence_penalty":1.5},
                "scope":"CPU fixture"}
        write_json(p / "usercheck/input.json", spec)
        for i in range(2):
            row = {"repeat":i,"input_sha256":digest(p / "usercheck/input.json"),
                "errors":["User generation truncated"] if failed else [],
                "visible_text":"Hi", "server_after":{"active_requests":0,"pending_verifications":0},
                "call":{**spec,"text":"reasoning</think>Hi","finish_reason":"stop",
                        "token_ids":[3 if different and i else 2]}}
            path = p / f"usercheck/records/repeat_{i}.json"
            write_json(path,row);write_json(path.with_suffix('.json.seal'),{"sha256":digest(path)})
        return SimpleNamespace(output_root=p), spec

    def test_completed_usercheck_is_reused_without_generating_again(self):
        with tempfile.TemporaryDirectory() as folder:
            args,spec = self.records(folder)
            def forbidden(*args,**kw):
                raise AssertionError("must not contact model service")
            report = run_usercheck(args,spec,{},idle=forbidden,reset_cache=forbidden,check_server=forbidden)
            self.assertTrue(report["passed"])

    def test_divergent_or_failed_usercheck_cannot_unlock_actor_runs(self):
        for different, failed in [(True,False),(False,True)]:
            with tempfile.TemporaryDirectory() as folder:
                args,spec = self.records(folder,different=different,failed=failed)
                with self.assertRaises(ValueError):
                    check_usercheck(args,spec)


if __name__ == "__main__":
    unittest.main()
