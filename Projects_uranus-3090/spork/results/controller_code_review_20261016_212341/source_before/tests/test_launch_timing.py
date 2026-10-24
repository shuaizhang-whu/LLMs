"""Short CPU-only tests of causal launch timing and the real SPORK retry path."""
import asyncio
import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from spork_core.executors import SimulatedToolExecutor
from spork_core.launch_timing import choose_first_launch
from spork_core.prefix_replay import StreamState
from spork_core.turn_runner import RetryTurnConfig, run_spork_retry_turn


class CharTokenizer:
    def encode(self, text, **kwargs):
        return list(text.encode("ascii"))

    def decode(self, ids, **kwargs):
        return bytes(ids).decode("ascii")


class Controller:
    def __init__(self, policy="now", delay=0., error=False):
        self.policy, self.delay, self.error = policy, delay, error
        self.inputs = []

    def predict(self, inp):
        self.inputs.append(copy.deepcopy(inp))
        time.sleep(self.delay)
        if self.error:
            raise RuntimeError("controller test failure")
        now, wait = {"off": (-1., -2.), "now": (2., 1.), "wait": (1., 2.)}[self.policy]
        return {"q_now_s": now, "q_wait_s": wait, "policy": self.policy}


class LaunchTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.state = StreamState()
        self.blocker = asyncio.Event()
        self.main = asyncio.create_task(self.blocker.wait())
        self.tokenizer = CharTokenizer()
        self.messages = [{"role": "user", "content": "task before decision"}]

    async def asyncTearDown(self):
        self.main.cancel()
        await asyncio.gather(self.main, return_exceptions=True)

    def advance(self, count, text=None):
        self.state.text = text or ("<think>\n" + "x" * (count - 8))
        self.state.token_ids = self.tokenizer.encode(self.state.text)
        self.state.changed.set()

    async def choose(self, mode="bert", controller=None):
        return await choose_first_launch(self.state, self.main, self.tokenizer, self.messages, [],
                                         mode=mode, controller=controller)

    async def test_now_uses_exact_prefix_even_when_stream_has_overshot(self):
        self.advance(50)
        controller = Controller("now", delay=.02)
        task = asyncio.create_task(self.choose(controller=controller))
        await asyncio.sleep(.005)
        self.advance(100)
        text, count, trace = await task
        self.assertEqual((len(text), count), (32, 32))
        self.assertEqual(len(controller.inputs), 1)
        self.assertEqual(controller.inputs[0]["visible_generated_tokens"], 32)
        self.assertEqual(len(controller.inputs[0]["prefix_text"]), 32)
        self.assertGreaterEqual(trace["available_tokens_after_decision"], 100)
        self.assertGreater(trace["controller_inference_ms"], 0)

    async def test_wait_is_from_decision_point_and_runs_once(self):
        self.advance(32)
        controller = Controller("wait")
        task = asyncio.create_task(self.choose(controller=controller))
        await asyncio.sleep(.01)
        self.advance(95)
        await asyncio.sleep(.001)
        self.assertFalse(task.done())
        self.advance(96)
        text, count, trace = await task
        self.assertEqual((len(text), count), (96, 96))
        self.assertEqual(trace["wait_target_tokens"], 96)
        self.assertEqual(len(controller.inputs), 1)

    async def test_formal_action_while_waiting_cancels_prediction(self):
        self.advance(32)
        task = asyncio.create_task(self.choose(controller=Controller("wait")))
        await asyncio.sleep(.01)
        self.advance(96, "<think> done </think><tool_call>")
        text, _, trace = await task
        self.assertIsNone(text)
        self.assertIn("while_waiting", trace["status"])

    async def test_formal_action_during_inference_cancels_prediction(self):
        self.advance(32)
        task = asyncio.create_task(self.choose(controller=Controller("now", delay=.02)))
        await asyncio.sleep(.005)
        self.advance(96, "<think> done </think><tool_call>")
        text, _, trace = await task
        self.assertIsNone(text)
        self.assertIn("during_decision", trace["status"])

    async def test_off_skips_every_future_prediction(self):
        self.advance(32)
        text, _, trace = await self.choose(controller=Controller("off"))
        self.assertIsNone(text)
        self.assertEqual((trace["policy"], trace["status"]), ("off", "disabled"))
        controller = Controller("now")
        text, _, trace = await self.choose(mode="off", controller=controller)
        self.assertEqual(controller.inputs, [])

    async def test_controller_error_is_visible_and_finishes_without_speculation(self):
        self.advance(32)
        text, _, trace = await self.choose(controller=Controller(error=True))
        self.assertIsNone(text)
        self.assertEqual(trace["status"], "controller_error")
        self.assertIn("controller test failure", trace["error"])

    async def test_short_generation_and_stream_failure_cannot_hang_waiter(self):
        self.advance(20)
        task = asyncio.create_task(self.choose(mode="wait"))
        await asyncio.sleep(0)
        self.state.finished_at = time.perf_counter()
        self.state.changed.set()
        text, _, _ = await asyncio.wait_for(task, .1)
        self.assertIsNone(text)


CALL = {"name": "lookup", "arguments": {"id": "a"}}


class Adapter:
    uses_chat_api = False
    name = "qwen3_tool_call"
    main_stop = ("</tool_call>",)
    probe_stop = ("</tool_call>",)
    tools = [{"name": "lookup"}]
    tokenizer = CharTokenizer()

    def render_main_prompt(self, messages, **kwargs):
        return "PROMPT"

    def parse_main_tool_call(self, text):
        return copy.deepcopy(CALL) if "<tool_call>" in text else None

    def parse_probe_tool_call(self, text):
        return json.loads(text)

    def extract_probe_logprobs(self, logprobs):
        return logprobs

    def span_confidence(self, logprobs, **kwargs):
        return logprobs["confidence"]

    def token_metrics(self, text):
        return {}

    def preview(self, text):
        return text


class Client:
    base_url = "http://unused.invalid/v1"
    model = "fake-qwen3"
    timeout_s = 1

    def __init__(self, low_first=False, mismatch=False):
        self.prompts = []
        self.low_first, self.mismatch = low_first, mismatch

    async def complete(self, session, prompt, **kwargs):
        self.prompts.append(prompt)
        call = {"name": "wrong", "arguments": {}} if self.mismatch else CALL
        confidence = .1 if self.low_first and len(self.prompts) == 1 else 1.
        return {"text": json.dumps(call), "logprobs": {"confidence": confidence}}, 0.


async def fake_stream(client, session, prompt_ids, state, **kwargs):
    text = "<think>\n" + "x" * 144 + "\n</think>\n<tool_call>\n" + json.dumps(CALL) + "\n</tool_call>"
    try:
        # Each simulated SSE chunk contains 16 true token IDs. A chunk counter
        # would never reach 32 before the Actor finishes this test trajectory.
        for end in range(16, len(text) + 16, 16):
            state.text = text[:end]
            state.token_ids = list(state.text.encode())
            if state.first_output_at is None:
                state.first_output_at = time.perf_counter()
            state.changed.set()
            await asyncio.sleep(.002)
        state.finish_reason = "stop"
    finally:
        state.finished_at = time.perf_counter()
        state.changed.set()


class TurnIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def run_turn(self, mode, controller=None, client=None):
        client = client or Client()
        hooks = []

        async def set_draft(call, prompt):
            hooks.append(call)
            return {"status": "ok"}

        async def clear():
            return {"clear_status": "cleared"}

        cfg = RetryTurnConfig(launch_mode=mode, launch_controller=controller, main_request_id="test-main-id",
                              baseline_max_tokens=8192, context_window_tokens=40960,
                              snap_to_sentence=False, d3_set=set_draft, d3_clear=clear)
        with patch("spork_core.prefix_replay.PrefixHttpClient.stream", new=fake_stream):
            result = await run_spork_retry_turn(None, client, Adapter(), [{"role": "user", "content": "look up a"}],
                                                SimulatedToolExecutor(latency_s=0), 1, cfg)
        return result, client, hooks

    async def test_real_runner_uses_now_wait_off_and_keeps_d3_and_strict_gate(self):
        for mode, expected in (("now", 32), ("wait", 96), ("off", None), ("original", 16)):
            with self.subTest(mode=mode):
                result, client, hooks = await self.run_turn(mode)
                self.assertTrue(result["completion_valid"])
                self.assertEqual(result["baseline_tool_name"], "lookup")
                if expected is None:
                    self.assertEqual(result["num_probes"], 0)
                    self.assertFalse(result["spec_dispatched"])
                    self.assertEqual(hooks, [])
                else:
                    self.assertEqual(result["probes"][0]["tokens_at_dispatch"], expected)
                    self.assertTrue(result["spec_used"])
                    self.assertEqual(result["d3_set_count"], 1)
                    self.assertEqual(len(hooks), 1)
                    self.assertIsInstance(client.prompts[0], list)

    async def test_controller_all_three_policies_reach_existing_turn_runner(self):
        for policy in ("now", "wait", "off"):
            with self.subTest(policy=policy):
                controller = Controller(policy)
                result, client, hooks = await self.run_turn("bert", controller)
                self.assertEqual(result["launch_timing"]["policy"], policy)
                self.assertEqual(len(controller.inputs), 1)
                self.assertEqual(result["num_probes"], 0 if policy == "off" else 1)
                self.assertEqual(result["spec_used"], policy != "off")

    async def test_low_confidence_retries_on_more_actual_tokens(self):
        result, client, hooks = await self.run_turn("now", client=Client(low_first=True))
        self.assertEqual(result["num_probes"], 2)
        self.assertEqual(result["committed_retry_idx"], 1)
        self.assertGreaterEqual(result["probes"][1]["tokens_at_dispatch"], 82)
        self.assertEqual(result["d3_set_count"], 2)
        self.assertTrue(result["gate_match"])

    async def test_wrong_prediction_executes_formal_call_and_drains_tool(self):
        result, client, hooks = await self.run_turn("now", client=Client(mismatch=True))
        self.assertFalse(result["spec_used"])
        self.assertTrue(result["spec_wasted"])
        self.assertEqual(result["baseline_tool_name"], "lookup")
        self.assertIsNotNone(result["canonical_tool_dispatch_ms"])
        self.assertIn("spec_cleanup_ms", result)
        self.assertNotIn("spec_cleanup_error", result)

    async def test_unified_evaluator_wires_loaded_controller_and_d3_hooks(self):
        from spork_unified_eval import run_task
        from spork_core.turn_runner import TurnConfig

        class Sidecar:
            def __init__(self, *args, **kwargs):
                pass

            async def set_from_probe_call(self, call, prompt):
                return {"status": "ok"}

            async def clear(self):
                return {"clear_status": "cleared"}

        controller = Controller("now")
        args = SimpleNamespace(d2_max_retries=5, d2_confidence_threshold=.9, d2_retry_token_step=50,
                               d2_snap_to_sentence=False, d2_probe_max_tokens=160, d2_probe_timeout_s=0.,
                               d2_theta_decay_per_retry=0., d2_min_tokens_first_probe=0,
                               d2_continue_after_dispatch=False, no_forced_conclusion=True,
                               _launch_controllers={"timing_bert": controller})
        with patch("spork_core.prefix_replay.PrefixHttpClient.stream", new=fake_stream), \
                patch("spork_unified_eval._SporkSidecar", new=Sidecar):
            turns = await run_task(args, None, Client(), Adapter(), SimulatedToolExecutor(latency_s=0),
                                   [{"role": "user", "content": "lookup a"}], "timing_bert",
                                   TurnConfig(enable_thinking=True, baseline_max_tokens=8192), max_turns=1)
        self.assertEqual(len(turns), 1)
        self.assertEqual(len(controller.inputs), 1)
        self.assertEqual(turns[0]["launch_timing"]["policy"], "now")
        self.assertTrue(turns[0]["spec_used"])
        self.assertEqual(turns[0]["d3_set_count"], 1)


class ReportTests(unittest.TestCase):
    def test_online_check_accepts_matching_runs_and_rejects_trajectory_drift(self):
        from scripts.check_controller_online import CONFIGS, analyze
        from spork_core.prefix_replay import COUNTERS
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            parent = output / "eval/fixture"
            parent.mkdir(parents=True)
            (parent / "summary.json").write_text("{}")
            for config in CONFIGS:
                (parent / config).mkdir()
                turn = {"completion_valid": True, "main_token_ids": [1, 2], "baseline_has_tool_call": True,
                        "baseline_tool_name": "lookup", "baseline_tool_args": {"id": "a"},
                        "tool_result_sha256": "abc", "environment_after": {"agent_db": "same"},
                        "launch_timing": {"policy": "wait", "status": "ready", "controller_inference_ms": 1.,
                                          "controller_calls": int(config in ("timing_bert", "timing_masked"))}}
                (parent / config / "retail_2.json").write_text(json.dumps(
                    {"domain": "retail", "task_id": "2", "turns": [turn]}))
            before = dict.fromkeys(COUNTERS, 0)
            after = {**before, "verified_injections": 1, "active_requests": 0, "pending_verifications": 0}
            manifest = {"tasks": [{"group_id": "retail:2"}]}
            with contextlib.redirect_stdout(io.StringIO()):
                report = analyze(output, manifest, before, after)
            self.assertTrue(report["passed"])
            path = parent / "timing_bert/retail_2.json"
            changed = json.loads(path.read_text())
            changed["turns"][0]["main_token_ids"] = [1, 3]
            path.write_text(json.dumps(changed))
            with contextlib.redirect_stdout(io.StringIO()):
                report = analyze(output, manifest, before, after)
            self.assertFalse(report["passed"])
            self.assertTrue(any("trajectory/outcome differs" in e for e in report["errors"]))


if __name__ == "__main__":
    unittest.main()
