"""CPU checks for early-exit commit, rollback, and truncated-output boundaries."""
import asyncio
import json
import math
import time
import unittest

from test_launch_timing import Adapter, CALL, Client
from spork_core.early_exit import ExitConfig, run_early_exit_turn, _wait_next_check
from spork_core.executors import SimulatedToolExecutor
from spork_core.prefix_replay import StreamState
from spork_core.qwen import parse_baseline_tool_call
from spork_core.turn_runner import RetryTurnConfig


class CheckedAdapter(Adapter):
    def parse_main_tool_call(self, text):
        return parse_baseline_tool_call(text)


class Env:
    def get_db_hash(self):
        return "unchanged"

    def get_user_db_hash(self):
        return None


class Executor(SimulatedToolExecutor):
    def __init__(self):
        super().__init__(latency_s=.001)
        self.env = Env()
        self.commits = 0
        self.canonical_calls = []

    def commit_speculative(self, execution):
        self.commits += 1

    def execute(self, call):
        self.canonical_calls.append(call)
        return super().execute(call)


class SlowObsoleteExecutor(Executor):
    def execute_speculative(self, call):
        if call["arguments"]["id"] == "wrong":
            time.sleep(.12)
        return super().execute_speculative(call)


class HTTP:
    def __init__(self, continuation="same", free_probability=.99, forced_truncated=False,
                 actor_truncated=False):
        self.continuation = continuation
        self.free_probability = free_probability
        self.forced_truncated = forced_truncated
        self.actor_truncated = actor_truncated
        self.prompts = []
        self.request_options = []
        self.stream_options = None
        self.stopped = False

    async def stream(self, session, prompt, state, **kwargs):
        self.stream_options = kwargs
        body = "<think>\n" + "thinking.\n" * 80
        full = body + "</think>\n<tool_call>" + json.dumps(CALL)
        try:
            for end in range(10, len(full) + 10, 10):
                state.text = full[:end]
                state.token_ids = list(state.text.encode())
                if state.first_output_at is None:
                    state.first_output_at = time.perf_counter()
                state.changed.set()
                await asyncio.sleep(.001)
            state.finish_reason = "length" if self.actor_truncated else "stop"
        finally:
            state.finished_at = time.perf_counter()
            state.changed.set()
            self.stopped = True

    async def complete(self, session, ids, **kwargs):
        text = bytes(ids).decode()
        self.prompts.append(text)
        self.request_options.append((text, kwargs))
        finish = "stop"
        logs = {"confidence": 1.0}
        if kwargs["max_tokens"] == 1:
            output = "<tool_call>"
            logs = {"top_logprobs": [{"<tool_call>": math.log(self.free_probability)}]}
        elif text.endswith('{"name": "'):
            output = json.dumps(CALL)
            finish = "length" if self.forced_truncated else "stop"
        elif text.endswith("</think>\n\n"):
            if self.continuation == "no_tool":
                output = "NO_TOOL_TRIAL_MUST_NOT_COMMIT"
            elif self.continuation == "truncated":
                output, finish = "<tool_call>{", "length"
            else:
                call = CALL if self.continuation == "same" else {"name": "lookup", "arguments": {"id": "b"}}
                output = "<tool_call>" + json.dumps(call)
        else:  # Rollback resumes the unclosed original reasoning.
            output = "resumed original thought</think>\n<tool_call>" + json.dumps(CALL)
        return {"text": output, "token_ids": list(output.encode()), "finish_reason": finish,
                "logprobs": logs, "wall_s": 0.0, "usage": {}}


class ChangingHTTP(HTTP):
    def __init__(self, *, alternate=False, low_free=False):
        super().__init__(free_probability=.1 if low_free else .99)
        self.forced_count = 0
        self.alternate = alternate

    async def complete(self, session, ids, **kwargs):
        result = await super().complete(session, ids, **kwargs)
        if bytes(ids).decode().endswith('{"name": "'):
            wrong = self.forced_count == 0 or (self.alternate and self.forced_count % 2 == 0)
            self.forced_count += 1
            call = {"name": "lookup", "arguments": {"id": "wrong"}} if wrong else CALL
            result.update(text=json.dumps(call), token_ids=list(json.dumps(call).encode()))
        return result


class SequenceHTTP(HTTP):
    def __init__(self, probabilities, confidences=(1.,), calls=(CALL,)):
        super().__init__()
        self.probabilities, self.confidences, self.calls = probabilities, confidences, calls
        self.forced_count = self.free_count = 0

    async def complete(self, session, ids, **kwargs):
        result = await super().complete(session, ids, **kwargs)
        select = lambda values, index: values[min(index, len(values) - 1)]
        if bytes(ids).decode().endswith('{"name": "'):
            index = self.forced_count
            self.forced_count += 1
            call = select(self.calls, index)
            result.update(text=json.dumps(call), token_ids=list(json.dumps(call).encode()),
                          logprobs={"confidence": select(self.confidences, index)})
        elif kwargs["max_tokens"] == 1:
            self.free_count += 1
            result["logprobs"] = {"top_logprobs": [{"<tool_call>": math.log(
                select(self.probabilities, self.forced_count - 1))}]}
        return result


class DeferredHTTP(HTTP):
    def __init__(self, probabilities, mismatch_at=None):
        super().__init__()
        self.probabilities = probabilities
        self.mismatch_at = mismatch_at
        self.prefix_indices = {}
        self.forced_count = self.free_count = 0
        self.forced_counts_at_free = []

    async def complete(self, session, ids, **kwargs):
        result = await super().complete(session, ids, **kwargs)
        text = bytes(ids).decode()
        prefix = text.split("</think>")[0]
        if kwargs["max_tokens"] == 1:
            self.prefix_indices[prefix] = self.free_count
            probability = self.probabilities[min(self.free_count, len(self.probabilities) - 1)]
            self.free_count += 1
            self.forced_counts_at_free.append(self.forced_count)
            result["logprobs"] = {"top_logprobs": [{"<tool_call>": math.log(probability)}]}
        elif text.endswith('{"name": "'):
            self.forced_count += 1
            if self.mismatch_at is not None and self.prefix_indices.get(prefix) == self.mismatch_at:
                result["text"] = json.dumps({"name": "lookup", "arguments": {"id": "wrong"}})
                result["token_ids"] = list(result["text"].encode())
        return result


class ExitTests(unittest.IsolatedAsyncioTestCase):
    async def test_ready_cadence_shortens_an_existing_wait_after_tool_arrival(self):
        loop = asyncio.get_running_loop()
        main, tool = loop.create_future(), loop.create_future()
        state = StreamState()
        state.text, state.token_ids = "thought\n", [1] * 18
        config = ExitConfig(method="eta", guard_policy="paper", check_schedule="ready_fast")
        waiting = asyncio.create_task(_wait_next_check(state, main, tool, 1, config, candidate_clear=True))
        await asyncio.sleep(0)
        self.assertFalse(waiting.done())
        tool.set_result(None)
        state.changed.set()
        self.assertEqual(await asyncio.wait_for(waiting, .2), 16)

    async def test_ready_cadence_keeps_long_interval_without_clear_candidate(self):
        loop = asyncio.get_running_loop()
        main, tool = loop.create_future(), loop.create_future()
        tool.set_result(None)
        state = StreamState()
        state.text, state.token_ids = "thought\n", [1] * 18
        config = ExitConfig(method="eta", guard_policy="paper", check_schedule="ready_fast")
        waiting = asyncio.create_task(_wait_next_check(state, main, tool, 1, config, candidate_clear=False))
        await asyncio.sleep(0)
        self.assertFalse(waiting.done())
        state.token_ids = [1] * 51
        state.changed.set()
        self.assertEqual(await asyncio.wait_for(waiting, .2), 50)

    async def test_fixed_fast_control_uses_short_interval_even_without_ready_tool(self):
        loop = asyncio.get_running_loop()
        main, tool = loop.create_future(), loop.create_future()
        state = StreamState()
        state.text, state.token_ids = "thought\n", [1] * 18
        config = ExitConfig(method="eta", guard_policy="paper", check_schedule="fixed_fast")
        self.assertEqual(await _wait_next_check(state, main, tool, 1, config, candidate_clear=False), 16)

    async def measure(self, http, *, enable=True, executor=None, actor_sampling=None,
                      canonical_tool_history=False, **options):
        executor = executor or Executor()
        messages = [{"role": "user", "content": "lookup a"}]
        clears = []

        async def clear():
            clears.append(True)
            return {"clear_status": "cleared"}

        cfg = RetryTurnConfig(max_retries=2, d3_clear=clear, enable_thinking=True,
                              baseline_max_tokens=8192, context_window_tokens=40960,
                              actor_sampling=actor_sampling, canonical_tool_history=canonical_tool_history)
        params = dict(method="eta", stable_checks=2, check_interval=10, max_checks=4,
                      enable_exit=enable)
        params.update(options)
        result = await run_early_exit_turn(None, Client(), CheckedAdapter(), messages, executor, 1,
            cfg, ExitConfig(**params), http=http)
        return result, executor, messages, clears

    async def test_successful_exit_commits_only_matching_tool_and_cleans_stream(self):
        http = HTTP()
        result, tool, messages, clears = await self.measure(http)
        self.assertTrue(result["exit"]["fired"])
        self.assertFalse(result["exit"]["rolled_back"])
        self.assertTrue(result["spec_used"])
        self.assertEqual(tool.commits, 1)
        self.assertEqual(len(clears), 1)
        self.assertTrue(http.stopped)

    async def test_canonical_history_closes_call_but_preserves_raw_record(self):
        result, _, messages, _ = await self.measure(HTTP(), canonical_tool_history=True)
        self.assertTrue(messages[1]["content"].endswith("</tool_call>"))
        self.assertNotIn("<think>", messages[1]["content"])
        self.assertIn("<think>", result["baseline_text"])
        self.assertFalse(result["baseline_text"].endswith("</tool_call>"))

    async def test_no_tool_trial_is_discarded_and_original_reasoning_resumes(self):
        result, tool, messages, _ = await self.measure(HTTP("no_tool"))
        self.assertTrue(result["exit"]["rolled_back"])
        self.assertNotIn("NO_TOOL_TRIAL", json.dumps(messages))
        self.assertEqual(result["baseline_tool_args"], {"id": "a"})
        self.assertEqual(tool.commits, 1)

    async def test_truncated_trial_cannot_be_committed(self):
        result, tool, _, _ = await self.measure(HTTP("truncated"))
        self.assertTrue(result["exit"]["rolled_back"])
        self.assertTrue(result["completion_valid"])
        self.assertEqual(tool.commits, 1)

    async def test_changed_arguments_execute_canonical_call_instead_of_old_result(self):
        result, tool, _, _ = await self.measure(HTTP("different"))
        self.assertTrue(result["exit"]["fired"])
        self.assertFalse(result["spec_used"])
        self.assertEqual(tool.commits, 0)
        self.assertEqual(tool.canonical_calls[-1]["arguments"], {"id": "b"})

    async def test_tool_name_confidence_alone_cannot_trigger_exit(self):
        result, _, _, _ = await self.measure(HTTP(free_probability=.1))
        self.assertFalse(result["exit"]["fired"])

    async def test_truncated_proposal_never_dispatches(self):
        result, tool, _, _ = await self.measure(HTTP(forced_truncated=True))
        self.assertFalse(result["spec_dispatched"])
        self.assertEqual(tool.commits, 0)

    async def test_truncated_actor_fails_instead_of_committing_speculation(self):
        with self.assertRaisesRegex(RuntimeError, "Incomplete Actor"):
            await self.measure(HTTP(actor_truncated=True), enable=False)

    async def test_stable_replacement_can_exit_and_only_new_tool_commits(self):
        result, tool, _, clears = await self.measure(ChangingHTTP(), stable_checks=3,
            max_checks=6, retarget_after_stable=True)
        self.assertTrue(result["exit"]["fired"])
        self.assertEqual(result["exit"]["retarget_count"], 1)
        self.assertEqual(result["baseline_tool_args"], CALL["arguments"])
        old, new = result["speculative_executions"]
        self.assertFalse(old["committed"])
        self.assertTrue(new["committed"])
        self.assertIsNotNone(old["end_ms"])
        self.assertEqual(tool.commits, 1)
        self.assertEqual(len(clears), 1)

    async def test_locked_reference_keeps_wrong_initial_candidate(self):
        result, tool, _, _ = await self.measure(ChangingHTTP(), stable_checks=3, max_checks=6)
        self.assertFalse(result["exit"]["fired"])
        self.assertEqual(result["exit"]["retarget_count"], 0)
        self.assertEqual(tool.commits, 0)

    async def test_oscillating_or_low_probability_calls_cannot_replace_candidate(self):
        for http in (ChangingHTTP(alternate=True), ChangingHTTP(low_free=True)):
            with self.subTest(http=http):
                result, _, _, _ = await self.measure(http, stable_checks=3, max_checks=6,
                                                    retarget_after_stable=True)
                self.assertFalse(result["exit"]["fired"])
                self.assertEqual(result["exit"]["retarget_count"], 0)

    async def test_late_obsolete_tool_is_drained_without_overwriting_new_timing(self):
        result, _, _, _ = await self.measure(ChangingHTTP(), stable_checks=3, max_checks=6,
            retarget_after_stable=True, executor=SlowObsoleteExecutor())
        old, new = result["speculative_executions"]
        self.assertTrue(result["exit"]["fired"])
        self.assertGreater(old["end_ms"], new["end_ms"])
        self.assertEqual(result["spec_end_ms"], new["end_ms"])
        self.assertGreaterEqual(result["turn_end_ms"], old["end_ms"])
        self.assertFalse(old["committed"])
        self.assertTrue(new["committed"])

    async def test_actor_sampling_survives_exit_and_rollback_without_changing_probes(self):
        sampling = {"temperature": .6, "top_p": .95, "top_k": 20}
        http = HTTP("no_tool")
        result, _, _, _ = await self.measure(http, actor_sampling=sampling)
        self.assertTrue(result["exit"]["rolled_back"])
        self.assertEqual(http.stream_options["sampling"], sampling)
        sampled = [opts for _, opts in http.request_options if opts.get("sampling")]
        self.assertEqual(len(sampled), 2)  # independent continuation, then rollback
        self.assertTrue(all(opts["sampling"] == sampling for opts in sampled))
        for text, opts in http.request_options:
            if opts["max_tokens"] == 1 or text.endswith('{"name": "'):
                self.assertIsNone(opts.get("sampling"))

    async def test_paper_guard_uses_latest_probability_with_three_equal_calls(self):
        result, _, _, _ = await self.measure(SequenceHTTP((.1, .1, .99)),
            guard_policy="paper", stable_checks=3, max_checks=3)
        self.assertTrue(result["exit"]["fired"])
        self.assertEqual(len(result["checks"]), 3)
        conservative, _, _, _ = await self.measure(SequenceHTTP((.1, .1, .99)),
            stable_checks=3, max_checks=3)
        self.assertFalse(conservative["exit"]["fired"])
        low_last, _, _, _ = await self.measure(SequenceHTTP((.99, .99, .1)),
            guard_policy="paper", stable_checks=3, max_checks=3)
        self.assertFalse(low_last["exit"]["fired"])

    async def test_paper_stability_includes_calls_before_confident_dispatch(self):
        http = SequenceHTTP((.1, .1, .99), confidences=(.1, .1, 1.))
        result, executor, _, _ = await self.measure(http, guard_policy="paper",
            stable_checks=3, max_checks=3)
        self.assertTrue(result["exit"]["fired"])
        self.assertEqual((http.forced_count, http.free_count), (3, 3))
        self.assertEqual(len(result["probes"]), 3)
        self.assertEqual(executor.commits, 1)

    async def test_paper_budget_is_fifteen_total_even_without_dispatch(self):
        http = SequenceHTTP((.1,), confidences=(.1,))
        result, executor, _, _ = await self.measure(http, guard_policy="paper",
            stable_checks=3, max_checks=15)
        self.assertEqual((http.forced_count, http.free_count), (15, 15))
        self.assertEqual(len(result["checks"]), 15)
        self.assertFalse(result["spec_dispatched"])
        self.assertEqual(executor.commits, 0)

    async def test_paper_dispatch_does_not_reset_the_total_budget(self):
        http = SequenceHTTP((.1,), confidences=(.1, .1, .1, .1, 1.))
        result, _, _, _ = await self.measure(http, guard_policy="paper",
            stable_checks=3, max_checks=6)
        self.assertEqual((http.forced_count, http.free_count), (6, 6))
        self.assertTrue(result["spec_dispatched"])
        self.assertFalse(result["exit"]["fired"])

    async def test_paper_replacement_uses_the_same_guard_and_keeps_exact_commit(self):
        wrong = {"name": "lookup", "arguments": {"id": "wrong"}}
        http = SequenceHTTP((.99, .1, .1, .99), calls=(wrong, CALL))
        result, executor, _, _ = await self.measure(http, guard_policy="paper",
            stable_checks=3, max_checks=4, retarget_after_stable=True)
        self.assertTrue(result["exit"]["fired"])
        self.assertEqual(result["exit"]["retarget_count"], 1)
        self.assertEqual((http.forced_count, http.free_count), (4, 4))
        old, new = result["speculative_executions"]
        self.assertFalse(old["committed"])
        self.assertTrue(new["committed"])
        self.assertEqual(executor.commits, 1)

    async def test_deferred_low_probability_checks_do_not_generate_full_calls(self):
        http = DeferredHTTP((.1,))
        result, _, _, _ = await self.measure(http, guard_policy="paper",
            stable_checks=3, max_checks=5, defer_forced_checks=True)
        self.assertEqual((http.forced_count, http.free_count), (1, 5))
        self.assertFalse(result["exit"]["fired"])

    async def test_deferred_exit_backfills_the_actual_three_consecutive_prefixes(self):
        http = DeferredHTTP((.1, .1, .1, .99))
        result, executor, _, _ = await self.measure(http, guard_policy="paper",
            stable_checks=3, max_checks=4, defer_forced_checks=True)
        self.assertTrue(result["exit"]["fired"])
        self.assertEqual(http.forced_counts_at_free, [1, 1, 1, 1])
        self.assertEqual(http.forced_count, 4)
        self.assertEqual(result["checks"][-1]["verified_window_indices"], [1, 2, 3])
        self.assertTrue(all(c["forced_verified_at_check"] == 3 for c in result["checks"][1:]))
        self.assertEqual(executor.commits, 1)

    async def test_deferred_unchecked_prefix_or_past_mismatch_cannot_prove_stability(self):
        http = DeferredHTTP((.1, .1, .1, .99), mismatch_at=2)
        result, _, _, _ = await self.measure(http, guard_policy="paper",
            stable_checks=3, max_checks=4, defer_forced_checks=True, retarget_after_stable=True)
        self.assertFalse(result["exit"]["fired"])
        self.assertEqual(result["exit"]["retarget_count"], 0)
        self.assertEqual(http.forced_count, 3)
        self.assertNotIn("prediction", result["checks"][1])
        self.assertEqual(result["checks"][-1]["candidate_stable_count"], 1)

    async def test_adaptive_cadence_retains_three_check_exit_and_exact_commit(self):
        result, executor, _, _ = await self.measure(HTTP(), guard_policy="paper",
            stable_checks=3, max_checks=5, defer_forced_checks=True,
            check_schedule="ready_fast", fast_check_interval=2)
        self.assertTrue(result["exit"]["fired"])
        self.assertEqual(result["exit"]["stable_count"], 3)
        self.assertEqual(executor.commits, 1)
        fast = [c for c in result["checks"] if c.get("check_interval_used") == 2]
        self.assertTrue(fast)
        self.assertTrue(all(c["candidate_clear_before_wait"] and c["tool_ready_at_checkpoint"] for c in fast))


if __name__ == "__main__":
    unittest.main()
