"""CPU-only tests of prefix isolation, scheduling, labels and HTTP token counts."""
import asyncio
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from spork_core.executors import SpeculativeToolExecution
from spork_core.prefix_replay import (
    PrefixHttpClient, ReplayConfig, StreamState, paired_labels, run_prefix_branch,
)


CALL = {"name": "lookup", "arguments": {"id": "good"}}


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(text.encode())

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        text = json.dumps(messages)
        return self.encode(text) if tokenize else text


class Adapter:
    tokenizer = Tokenizer()
    main_stop = ("</tool_call>",)
    probe_stop = main_stop

    def __init__(self, tokenizer=None, tools=None):
        self.tokenizer = tokenizer or self.tokenizer

    def render_main_prompt(self, messages, **kwargs):
        return json.dumps(messages)

    def parse_main_tool_call(self, text):
        if "<tool_call>" not in text:
            return None
        return json.loads(text.split("<tool_call>")[-1].split("</tool_call>")[0])

    def parse_probe_tool_call(self, text):
        return json.loads(text)

    def extract_probe_logprobs(self, lp):
        return lp

    def span_confidence(self, lp, **kwargs):
        return lp["confidence"]


class FakeClient:
    def __init__(self, chunks=None, prediction=CALL, confidence=.99, finish_reason="stop", **kwargs):
        self.chunks = chunks or [(1, "Reason."), (70, " More reasoning.\n"),
                                 (10, "</think><tool_call>" + json.dumps(CALL) + "</tool_call>")]
        self.prediction = prediction
        self.confidence = confidence
        self.model = "test-model"
        self.finish_reason = finish_reason
        self.calls = []
        self.sets = []
        self.counters = {"verified_injections": 0, "accepted_draft_tokens": 0}

    async def clear(self, session):
        return {"status": "cleared", "active_requests": 0, "pending_verifications": 0}

    async def status(self, session):
        return {"has_drafter": True, **self.counters}

    async def set_draft(self, session, request_id, prompt_len, call, tokenizer):
        self.sets.append(call)
        self.counters["verified_injections"] += 1
        self.counters["accepted_draft_tokens"] += 2
        return {"status": "ok"}

    async def complete(self, session, prompt_ids, **kwargs):
        self.calls.append((list(prompt_ids), kwargs))
        await asyncio.sleep(0)
        if kwargs.get("max_tokens") == 32:
            return {"text": "<think> I will look up the order.", "token_ids": list(range(32)),
                    "finish_reason": "length", "wall_s": .01}
        return {"text": json.dumps(self.prediction), "token_ids": [7, 8],
                "logprobs": {"confidence": self.confidence}, "finish_reason": "stop", "wall_s": .01}

    async def stream(self, session, prompt_ids, state, **kwargs):
        try:
            for count, text in self.chunks:
                await asyncio.sleep(.004)
                if state.first_output_at is None:
                    state.first_output_at = time.perf_counter()
                state.token_ids.extend([200 + len(state.token_ids)] * count)
                if state.record_token_timing:
                    state.token_events.append({"end_token": len(state.token_ids), "received_at": time.perf_counter()})
                state.text += text
                state.changed.set()
            state.finish_reason = self.finish_reason
        finally:
            state.finished_at = time.perf_counter()
            state.changed.set()


class Executor:
    def __init__(self, delay=0):
        self.env = []
        self.delay = delay
        self.finished_spec = False
        self.last_real_wall_s = 0
        self.last_floor_sleep_s = 0

    def execute(self, call):
        self.env.append(copy.deepcopy(call))
        return json.dumps(call), 0

    def execute_speculative(self, call):
        state = copy.deepcopy(self.env)
        state.append(copy.deepcopy(call))
        time.sleep(self.delay)
        self.finished_spec = True
        return SpeculativeToolExecution(json.dumps(call), self.delay, self.delay, 0, state)

    def commit_speculative(self, execution):
        self.env = execution.state


SNAPSHOT = {"prompt_token_ids": [9, 8], "prefix_token_ids": [1, 2, 3],
            "prefix_text": "<think> Before the decision."}


class ReplayTests(unittest.IsolatedAsyncioTestCase):
    async def run_branch(self, policy, client=None, executor=None, cfg=None):
        return await run_prefix_branch(None, client or FakeClient(), Adapter(), SNAPSHOT,
                                       executor or Executor(), policy, cfg or ReplayConfig())

    async def test_now_observes_only_initial_prefix_and_preserves_ids(self):
        client = FakeClient()
        result = await self.run_branch("now", client)
        p = result["probes"][0]
        self.assertEqual(p["new_tokens_at_dispatch"], 0)
        self.assertEqual(p["visible_prefix_text"], SNAPSHOT["prefix_text"])
        self.assertEqual(client.calls[0][0][:5], [9, 8, 1, 2, 3])
        self.assertTrue(result["spec_used"])

    async def test_wait_counts_tokens_not_chunks(self):
        client = FakeClient(chunks=[(3, "a"), (6, "b"),
                                    (1, "<tool_call>" + json.dumps(CALL))])
        result = await self.run_branch("wait", client, cfg=ReplayConfig(delay_tokens=8))
        self.assertEqual(result["probes"][0]["new_tokens_at_dispatch"], 9)
        self.assertTrue(result["spec_used"])

    async def test_wait_cancels_if_formal_call_arrives_first(self):
        client = FakeClient(chunks=[(3, "<tool_call>" + json.dumps(CALL))])
        result = await self.run_branch("wait", client)
        self.assertTrue(result["first_launch_cancelled"])
        self.assertEqual(result["probes"], [])
        self.assertEqual(client.calls, [])

    async def test_off_never_predicts(self):
        client = FakeClient()
        result = await self.run_branch("off", client)
        self.assertFalse(result["spec_used"])
        self.assertEqual(client.calls, [])
        self.assertEqual(client.sets, [])

    async def test_wrong_arguments_do_not_commit_speculative_state(self):
        client = FakeClient(prediction={"name": "lookup", "arguments": {"id": "wrong"}})
        executor = Executor()
        result = await self.run_branch("now", client, executor)
        self.assertTrue(result["spec_wasted"])
        self.assertFalse(result["spec_used"])
        self.assertEqual(executor.env, [CALL])

    async def test_no_tool_still_drains_the_speculative_thread(self):
        client = FakeClient(chunks=[(1, "</think>Here is an answer.")])
        executor = Executor(delay=.04)
        result = await self.run_branch("now", client, executor)
        self.assertTrue(result["spec_wasted"])
        self.assertTrue(executor.finished_spec)
        self.assertEqual(executor.env, [])
        self.assertGreater(result["cleanup_s"], result["wall_s"])

    async def test_low_confidence_still_registers_d3(self):
        client = FakeClient(confidence=.1)
        result = await self.run_branch("now", client, cfg=ReplayConfig(max_retries=1))
        self.assertTrue(client.sets)
        self.assertFalse(result["spec_used"])

    async def test_retries_wait_for_additional_real_tokens(self):
        client = FakeClient(confidence=.1)
        result = await self.run_branch("now", client, cfg=ReplayConfig(max_retries=2))
        self.assertEqual(len(result["probes"]), 2)
        self.assertGreaterEqual(result["probes"][1]["new_tokens_at_dispatch"], 50)

    async def test_token_limited_completion_is_not_a_valid_label(self):
        result = await self.run_branch("off", FakeClient(finish_reason="length"))
        self.assertFalse(result["completion_valid"])
        self.assertFalse(paired_labels({p: result for p in ("off", "now", "wait")})["valid"])

    async def test_failed_stream_is_saved_and_cleaned_up(self):
        class BrokenClient(FakeClient):
            async def stream(self, session, prompt_ids, state, **kwargs):
                state.finished_at = time.perf_counter()
                state.changed.set()
                raise RuntimeError("connection lost")
        result = await self.run_branch("wait", BrokenClient())
        self.assertFalse(result["completion_valid"])
        self.assertIn("connection lost", result["error"])
        self.assertEqual(result["clear_after"]["active_requests"], 0)

    async def test_prediction_only_keeps_gate_but_has_no_speculative_effects(self):
        client, executor = FakeClient(), Executor()
        result = await run_prefix_branch(None, client, Adapter(), SNAPSHOT, executor, "now", ReplayConfig(),
                                         enable_d3=False, enable_speculative_tools=False)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.sets, [])
        self.assertTrue(result["probes"][0]["confidence_gate_passed"])
        self.assertFalse(result["probes"][0]["dispatched"])
        self.assertFalse(executor.finished_spec)
        self.assertFalse(result["spec_used"])
        self.assertFalse(result["spec_wasted"])
        self.assertEqual(executor.env, [CALL], "The formal tool must still execute after generation")

    async def test_prediction_only_retries_low_confidence_without_d3(self):
        client = FakeClient(confidence=.1)
        result = await run_prefix_branch(None, client, Adapter(), SNAPSHOT, Executor(), "now",
                                         ReplayConfig(max_retries=2), enable_d3=False, enable_speculative_tools=False)
        self.assertEqual(len(result["probes"]), 2)
        self.assertEqual(client.sets, [])
        self.assertGreaterEqual(result["probes"][1]["new_tokens_at_dispatch"], 50)

    async def test_diagnostic_modes_send_the_same_actor_request(self):
        from spork_core.fork_diagnostic import MODES
        hashes = []
        for mode, setting in MODES.items():
            out = await run_prefix_branch(None, FakeClient(), Adapter(), SNAPSHOT, Executor(),
                                          setting["policy"], ReplayConfig(), enable_d3=setting["enable_d3"],
                                          enable_speculative_tools=setting["enable_speculative_tools"],
                                          record_token_timing=True)
            hashes.append(out["main_request_sha256"])
            self.assertEqual(out["token_events"][-1]["end_token"], len(out["continuation_token_ids"]))
            self.assertTrue(all(e["received_s"] >= 0 for e in out["token_events"]))
        self.assertEqual(len(set(hashes)), 1)


class FakeResponse:
    status = 200

    def __init__(self, chunks=None, json_value=None):
        self.chunks = chunks or []
        self.json_value = json_value
        self.content = self.lines()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def lines(self):
        for chunk in self.chunks:
            yield b'data: ' + (chunk if isinstance(chunk, bytes) else json.dumps(chunk).encode()) + b'\n'

    def raise_for_status(self):
        pass

    async def json(self):
        return self.json_value


class Session:
    def __init__(self, response, **kwargs):
        self.response = response
        self.body = None

    def post(self, url, json, **kwargs):
        self.body = json
        return self.response


class HttpTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_keeps_all_token_ids_in_a_batched_chunk(self):
        session = Session(FakeResponse([
            {"choices": [{"text": "abc", "token_ids": [1, 2, 3]}]},
            {"choices": [{"text": "def", "token_ids": [4, 5], "finish_reason": "stop"}]},
            b"[DONE]",
        ]))
        state = StreamState(record_token_timing=True)
        await PrefixHttpClient("http://test/v1", "model").stream(
            session, [8, 9], state, max_tokens=50, seed=42)
        self.assertEqual(state.token_ids, [1, 2, 3, 4, 5])
        self.assertEqual(state.text, "abcdef")
        self.assertEqual(session.body["prompt"], [8, 9])
        self.assertTrue(session.body["return_token_ids"])
        self.assertEqual([e["end_token"] for e in state.token_events], [3, 5])

    async def test_missing_ids_or_truncated_stream_are_errors(self):
        for choices in ([{"text": "abc"}], [{"text": "abc", "token_ids": [1]}]):
            state = StreamState()
            with self.assertRaises(RuntimeError):
                await PrefixHttpClient("http://test/v1", "model").stream(
                    Session(FakeResponse([{"choices": choices}])), [8], state, max_tokens=50, seed=42)
            self.assertIsNotNone(state.finished_at)


class LabelTests(unittest.TestCase):
    def test_negative_gains_and_disagreements_are_preserved(self):
        outcomes = {
            p: dict(completion_valid=True, wall_s=t, formal_call=CALL, tool_result="ok")
            for p, t in (("off", 8), ("now", 8.7), ("wait", 5.8))
        }
        outcomes["now"]["formal_call"] = {"name": "other", "arguments": {}}
        labels = paired_labels(outcomes)
        self.assertTrue(labels["valid"])
        self.assertAlmostEqual(labels["gain_now_s"], -.7)
        self.assertAlmostEqual(labels["gain_wait_s"], 2.2)
        self.assertFalse(labels["now_same_action_as_off"])
        self.assertEqual(labels["best_observed_policy"], "wait")


class TinyTools:
    def __init__(self):
        self.actions = []

    def lookup(self, id):
        self.actions.append(id)
        return {"id": id}


class TinyEnv:
    def __init__(self):
        self.tools = TinyTools()

    def get_db_hash(self):
        return json.dumps(self.tools.actions)

    def get_user_db_hash(self):
        return None


class CollectionIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_ten_prefixes_thirty_branches_and_past_only_inputs(self):
        # Exercise the complete writer/restore/history/label path using fake
        # inference and a real isolated Tau2ToolExecutor. No GPU or network.
        import argparse
        import scripts.collect_prefix_pairs as collector

        class CollectionClient(FakeClient):
            def __init__(self, base_url, model, **kwargs):
                super().__init__()
                self.base_url = base_url

        class CollectionSession:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def get(self, url, **kwargs):
                return FakeResponse(json_value={"data": [{"id": "test-model"}]})

        tasks = [("retail" if i % 2 else "airline", {"id": str(i)}) for i in range(5)]
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(
                output=Path(tmp), model_url="http://test/v1", model="test-model",
                timeout_s=10, delay_tokens=64, max_tokens=8192, context_window=40960,
                seed=42, prefixes=10, prefix_tokens=32, max_turns_per_task=2, tool_floor_s=0,
            )
            with patch.object(collector, "PrefixHttpClient", CollectionClient), \
                 patch.object(collector.aiohttp, "ClientSession", CollectionSession), \
                 patch.object(collector, "QwenToolCallAdapter", Adapter), \
                 patch.object(collector, "load_tools", lambda _: []), \
                 patch.object(collector, "load_system_prompt", lambda _: "System"), \
                 patch.object(collector, "build_user_message", lambda _: "User"), \
                 patch.object(collector, "get_environment", lambda _: TinyEnv()), \
                 patch.object(collector, "log", lambda _: None):
                report = await collector.collect(args, Tokenizer(), tasks)
            self.assertTrue(report["pipeline_passed"])
            self.assertEqual(report["branch_runs"], 30)
            rows = [json.loads(line) for line in (Path(tmp) / "pairs.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 10)
            first = json.loads((Path(tmp) / "prefix_000/input.json").read_text())
            second = json.loads((Path(tmp) / "prefix_001/input.json").read_text())
            self.assertEqual(len(first["messages"]), 2)
            self.assertEqual(len(second["messages"]), 4)
            self.assertNotIn("labels", first)
            self.assertNotIn("formal_call", first)
            self.assertEqual(first["prefix_token_ids"], list(range(32)))
            for sample in Path(tmp).glob("prefix_*/"):
                outcomes = [json.loads((sample / f"{p}.json").read_text()) for p in ("off", "now", "wait")]
                self.assertTrue(all(r["environment_before"] == outcomes[0]["environment_before"] for r in outcomes))
                self.assertTrue((sample / "environment.pkl").exists())
            from scripts.validate_prefix_dataset import validate_run
            training_rows, audit = validate_run(Path(tmp))
            self.assertTrue(audit["passed"], audit["errors"])
            self.assertEqual(len(training_rows), 10)
            self.assertNotIn("formal_call", training_rows[0]["input"])
            self.assertNotIn("wall_s", training_rows[0]["input"])
            # A regressed wait trigger must fail the independent saved-data audit.
            wait_file = Path(tmp) / "prefix_000/wait.json"
            bad = json.loads(wait_file.read_text())
            bad["probes"][0]["new_tokens_at_dispatch"] = 1
            wait_file.write_text(json.dumps(bad))
            _, audit = validate_run(Path(tmp))
            self.assertFalse(audit["passed"])
            self.assertTrue(any("wait launched before" in e for e in audit["errors"]))


if __name__ == "__main__":
    unittest.main()
