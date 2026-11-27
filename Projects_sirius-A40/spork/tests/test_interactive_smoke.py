"""Offline protocol checks; model generation is mocked, not launched."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from scripts.run_interactive_smoke import (
    CheckedUserSimulator, ROOT, QWEN3_32B_MODEL, evaluate_local, local_llm_args,
    local_assertion_judge, nl_module, prepare, summarize,
)
from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
from tau2.data_model.simulation import SimulationRun
from tau2.data_model.tasks import RewardType, Task
from tau2.user.user_simulator_base import UserError


ASSERTION = "Agent asks the user for confirmation."


def judge_reply(results, finish="stop"):
    return AssistantMessage(role="assistant", content=json.dumps({"results": results}),
                            raw_data={"choices": [{"finish_reason": finish}]})


def scenario():
    task = Task(id="quality-test", user_scenario={"instructions": "test only"},
                evaluation_criteria={"actions": [], "nl_assertions": [ASSERTION],
                                     "reward_basis": ["NL_ASSERTION"]})
    simulation = SimulationRun(id="quality-test", task_id=task.id, seed=42, duration=.1,
        start_time="2026-10-26T12:00:00", end_time="2026-10-26T12:00:01",
        termination_reason="user_stop", messages=[UserMessage(role="user", content="Please help."),
        AssistantMessage(role="assistant", content="May I proceed?"), UserMessage(role="user", content="###STOP###")])
    return task, simulation


class InteractiveQualityTests(unittest.TestCase):
    def evaluate(self, response):
        task, simulation = scenario()
        with patch.object(nl_module, "generate", return_value=response) as generation:
            report = evaluate_local(simulation, task, "mock", "/local/Qwen3", "http://127.0.0.1:8017/v1")
            self.assertIs(nl_module.generate, generation)
        return report, generation

    def test_local_judge_receives_local_endpoint_and_complete_checks_are_kept(self):
        report, generation = self.evaluate(judge_reply([
            {"expectedOutcome": ASSERTION, "reasoning": "The agent explicitly asks.", "metExpectation": True}]))
        self.assertIsNone(report["error"])
        self.assertEqual(report["reward_info"]["reward"], 1)
        self.assertEqual(generation.call_args.kwargs["model"], "openai//local/Qwen3")
        self.assertEqual(generation.call_args.kwargs["api_base"], "http://127.0.0.1:8017/v1")
        self.assertEqual(generation.call_args.kwargs["num_retries"], 0)
        self.assertEqual(len(report["judge_responses"]), 1)

    def test_empty_judge_results_are_unavailable_not_success(self):
        report, _ = self.evaluate(judge_reply([]))
        self.assertIsNone(report["reward_info"])
        self.assertIn("omitted", report["error"])

    def test_changed_or_duplicate_assertions_cannot_silently_pass(self):
        for outcomes in (["different assertion"], [ASSERTION, ASSERTION]):
            with self.subTest(outcomes=outcomes):
                report, _ = self.evaluate(judge_reply([
                    {"expectedOutcome": s, "reasoning": "yes", "metExpectation": True} for s in outcomes]))
                self.assertIsNone(report["reward_info"])

    def test_truncated_judge_is_retained_but_not_scored(self):
        report, _ = self.evaluate(judge_reply([], finish="length"))
        self.assertIsNone(report["reward_info"])
        self.assertIn("finish normally", report["error"])
        self.assertEqual(len(report["judge_responses"]), 1)

    def test_assertions_outside_reward_basis_are_checked_without_redefining_score(self):
        task, simulation = scenario()
        task.evaluation_criteria.reward_basis = [RewardType.DB, RewardType.COMMUNICATE]
        for results in ([{"expectedOutcome": ASSERTION, "reasoning": "Confirmation was not obtained.",
                          "metExpectation": False}], []):
            with self.subTest(results=results), patch.object(nl_module, "generate", return_value=judge_reply(results)) as generation:
                report = evaluate_local(simulation, task, "mock", "/local/Qwen3", "http://127.0.0.1:8017/v1")
            self.assertEqual(generation.call_count, 1)
            self.assertFalse(report["assertion_diagnostic"]["included_in_reward"])
            if results:
                self.assertIsNone(report["error"])
                self.assertEqual(report["reward_info"]["reward"], 1)
                self.assertFalse(report["assertion_diagnostic"]["all_met"])
                summary = summarize([{"method": "spork", "group_id": "mock:test", "errors": [],
                                      "termination_reason": "user_stop", "quality": report}])
                self.assertEqual(summary["methods"]["spork"]["local_reward_passes"], 1)
                self.assertEqual(summary["methods"]["spork"]["assertions_evaluated"], 1)
                self.assertEqual(summary["methods"]["spork"]["assertions_passes"], 0)
            else:
                self.assertIsNone(report["reward_info"])
                self.assertIsNone(report["assertion_diagnostic"]["all_met"])
                self.assertIn("omitted", report["error"])

    def test_judge_sees_committed_tool_arguments_but_not_raw_model_data(self):
        task, _ = scenario()
        trajectory = [UserMessage(role="user", content="Please wait for my confirmation."),
            AssistantMessage(role="assistant", tool_calls=[ToolCall(id="c1", name="cancel_reservation",
                arguments={"reservation_id": "3RK2T9"})], raw_data={"private": "SECRET_MODEL_DATA"}),
            ToolMessage(id="c1", role="tool", requestor="assistant", content='{"status":"cancelled"}')]
        with patch.object(nl_module, "generate", return_value=judge_reply([
            {"expectedOutcome": ASSERTION, "reasoning": "No confirmation before cancellation.",
             "metExpectation": False}])) as generation:
            with local_assertion_judge("/local/Qwen3", {}, [], trajectory=trajectory, assertions=[ASSERTION]):
                score = nl_module.NLAssertionsEvaluator.calculate_reward(task, trajectory)
        prompt = generation.call_args.kwargs["messages"][1].content
        self.assertIn("cancel_reservation", prompt)
        self.assertIn("3RK2T9", prompt)
        self.assertLess(prompt.index("Please wait"), prompt.index("cancel_reservation"))
        self.assertNotIn("SECRET_MODEL_DATA", prompt)
        self.assertEqual(score.reward, 0)

    def test_truncated_user_reply_is_not_used_as_a_real_confirmation(self):
        user = CheckedUserSimulator(llm="cpu-only", instructions="private", llm_args={})
        reply = AssistantMessage(role="assistant", content="Yes", raw_data={"choices": [{"finish_reason": "length"}]})
        with patch("tau2.user.user_simulator.generate", return_value=reply):
            with self.assertRaises(UserError):
                user.generate_next_message(AssistantMessage(role="assistant", content="Confirm?"), user.get_init_state())
        self.assertEqual(user.last_response["content"], "Yes")
        self.assertEqual(len(user.wall_s), 1)

    def test_summary_distinguishes_runtime_failure_unknown_quality_and_zero_reward(self):
        records = [{"method": "spork", "group_id": f"mock:{i}", "errors": errors,
                    "termination_reason": end, "quality": quality}
                   for i, (errors, end, quality) in enumerate([
                       (["length"], None, {}),
                       ([], "user_stop", {"error": "judge error", "reward_info": None}),
                       ([], "max_steps", {"reward_info": {"reward": 0}})])]
        result = summarize(records)["methods"]["spork"]
        self.assertEqual(result, {"runs": 3, "runtime_failures": 1, "normal_dialogue_ends": 1,
                                  "quality_evaluated": 1, "local_reward_passes": 0,
                                  "assertions_evaluated": 0, "assertions_passes": 0})


class InteractiveManifestTests(unittest.TestCase):
    def test_prepare_and_resume_preserve_one_deterministic_code_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(task_split=ROOT / "prefix_data/datasets/controller_stage3_20261008_213013_wi87e7du/task_split.json",
                tasks=["airline:2", "airline:48", "retail:39", "retail:76"], modes=["spork", "eta", "eta_retarget"],
                model=str(QWEN3_32B_MODEL), base_url="http://127.0.0.1:8017/v1", max_steps=60,
                max_tokens=8192, tool_floor=2., output=Path(temporary) / "pilot", resume=False,
                eta_reference="paper", repeats=1)
            args.output.mkdir()
            (args.output / "run.log").write_text("launcher started\n")
            (args.output / "exit_code.txt").write_text("1\n")
            original, _ = prepare(args)
            self.assertEqual((args.output / "run.log").read_text(), "launcher started\n")
            self.assertEqual(len(original["schedule"]), 12)
            self.assertEqual(original["eta_reference"], "paper")
            for config in original["exit_configs"].values():
                self.assertEqual(config["guard_policy"], "paper")
                self.assertEqual(config["max_checks"], 15)
            with zipfile.ZipFile(args.output / "code_snapshot.zip") as archive:
                self.assertTrue(all(member.date_time == (1980, 1, 1, 0, 0, 0) for member in archive.infolist()))
                self.assertIn("scripts/interactive_tau2_bridge.py", archive.namelist())
            args.resume = True
            resumed, _ = prepare(args)
            self.assertEqual(original, resumed)
            args.eta_reference = "conservative"
            with self.assertRaisesRegex(ValueError, "Resume would change"):
                prepare(args)
            args.eta_reference = "paper"
            args.max_steps = 61
            with self.assertRaisesRegex(ValueError, "Resume would change"):
                prepare(args)


class LocalUserTransportTests(unittest.TestCase):
    def test_real_sdk_keeps_model_path_and_no_thinking_option_on_loopback(self):
        received = []
        fail_request = [False]

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                received.append((self.path, body))
                response = {"id": "cpu-transport", "object": "chat.completion", "created": 0,
                    "model": "/local/Qwen3", "choices": [{"index": 0, "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "Yes, please proceed."}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
                if fail_request[0]:
                    response = {"error": {"message": "CPU transport test failure", "type": "server_error"}}
                payload = json.dumps(response).encode()
                self.send_response(500 if fail_request[0] else 200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
            thread.start()
            try:
                base = f"http://127.0.0.1:{server.server_port}/v1"
                user = CheckedUserSimulator(llm="openai//local/Qwen3", instructions="PRIVATE_SCENARIO",
                                            llm_args=local_llm_args(base, 2048))
                reply, _ = user.generate_next_message(
                    AssistantMessage(role="assistant", content="May I proceed?"), user.get_init_state())
                fail_request[0] = True
                with self.assertRaises(Exception) as error:
                    user.generate_next_message(AssistantMessage(role="assistant", content="Error test"), user.get_init_state())
                self.assertIn("CPU transport test failure", str(error.exception))
            finally:
                server.shutdown()
                thread.join(timeout=2)
        self.assertEqual(reply.content, "Yes, please proceed.")
        self.assertEqual(len(received), 2)  # One success and one failure, no hidden retries.
        path, request = received[0]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(request["model"], "/local/Qwen3")
        self.assertEqual(request["chat_template_kwargs"], {"enable_thinking": False})
        self.assertIn("PRIVATE_SCENARIO", request["messages"][0]["content"])


if __name__ == "__main__":
    unittest.main()
