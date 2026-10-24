"""CPU checks for early-exit commit, rollback, and truncated-output boundaries."""
import asyncio
import json
import math
import sys
import time
import unittest
from unittest.mock import patch

from test_launch_timing import Adapter, CALL, Client
from spork_core.early_exit import ExitConfig, run_early_exit_turn
from spork_core.executors import SimulatedToolExecutor
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


class HTTP:
    def __init__(self, continuation="same", free_probability=.99, forced_truncated=False,
                 actor_truncated=False):
        self.continuation = continuation
        self.free_probability = free_probability
        self.forced_truncated = forced_truncated
        self.actor_truncated = actor_truncated
        self.prompts = []
        self.stopped = False

    async def stream(self, session, prompt, state, **kwargs):
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


class ExitTests(unittest.IsolatedAsyncioTestCase):
    async def measure(self, http, *, enable=True, method="eta"):
        executor = Executor()
        messages = [{"role": "user", "content": "lookup a"}]
        clears = []

        async def clear():
            clears.append(True)
            return {"clear_status": "cleared"}

        cfg = RetryTurnConfig(max_retries=2, d3_clear=clear, enable_thinking=True,
                              baseline_max_tokens=8192, context_window_tokens=40960)
        result = await run_early_exit_turn(None, Client(), CheckedAdapter(), messages, executor, 1,
            cfg, ExitConfig(method=method, stable_checks=2, check_interval=10, max_checks=4,
                            enable_exit=enable), http=http)
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

    async def test_free_dispatch_gate_does_not_lock_to_confident_premature_call(self):
        class InitiallyWrong(HTTP):
            forced_count = 0
            free_count = 0

            async def complete(self, session, ids, **kwargs):
                result = await super().complete(session, ids, **kwargs)
                if bytes(ids).decode().endswith('{"name": "'):
                    self.forced_count += 1
                    if self.free_count <= 1:
                        result["text"] = json.dumps({"name": "premature", "arguments": {}})
                if kwargs["max_tokens"] == 1:
                    self.free_count += 1
                    result["logprobs"] = {"top_logprobs": [{"<tool_call>": math.log(.1 if self.free_count == 1 else .99)}]}
                return result

        async def check(*args, **kwargs):
            return {"call_text": "<tool_call>" + json.dumps(CALL), "finish_reason": "stop",
                    "tool_probability": .99, "d3_registered": True}

        http = InitiallyWrong()
        with patch.object(sys.modules[run_early_exit_turn.__module__], "decision_probe", new=check):
            result, executor, _, _ = await self.measure(http, method="reuse_gated")
        self.assertTrue(result["probes"][0]["dispatch_deferred"])
        self.assertTrue(result["probes"][0]["prediction_skipped"])
        self.assertEqual(http.forced_count, 1)
        self.assertTrue(result["exit"]["fired"])
        self.assertTrue(result["spec_used"])
        self.assertFalse(any(c["name"] == "premature" for c in executor.canonical_calls))


if __name__ == "__main__":
    unittest.main()
