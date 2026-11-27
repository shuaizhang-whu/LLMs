"""CPU contract and isolated-commit tests; no GPU performance assertions."""
import asyncio
import copy
import hashlib
import json
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from spork_core.executors import Tau2ToolExecutor, commit_speculative
from spork_core.successor_prefetch import (
    cancel_and_drain, rejection_reason, run_successor_pair, state_key)
from spork_core.turn_runner import RetryTurnConfig, _append_tool_messages


class Tools:
    def __init__(self, env):
        self.env = env

    def advance(self, delta):
        self.env.value += delta
        return self.env.value


class Env:
    def __init__(self):
        self.value = 0
        self.tools = Tools(self)

    def get_db_hash(self):
        return str(self.value)

    def get_user_db_hash(self):
        return None


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(text.encode())


class Adapter:
    tokenizer = Tokenizer()
    main_stop = ("</tool_call>",)

    def render_main_prompt(self, messages, **kwargs):
        return json.dumps(messages, sort_keys=True)


class Tests(unittest.IsolatedAsyncioTestCase):
    def test_each_reuse_condition_is_required(self):
        env = Env()
        call = {"name": "advance", "arguments": {"delta": 1}}
        parent = {"baseline_has_tool_call": True, "baseline_tool_name": call["name"],
                  "baseline_tool_args": call["arguments"], "tool_result_sha256": "result"}
        ticket = {"parent_call": call, "observation_sha256": "result", "parent_state": state_key(env),
                  "prompt_ids": [1, 2, 3], "generation": {"seed": 42}}
        self.assertIsNone(rejection_reason(ticket, parent, env, [1, 2, 3], {"seed": 42}))
        for key, value, reason in [
            ("parent_call", {"name": "advance", "arguments": {"delta": 2}}, "parent_call"),
            ("observation_sha256", "other", "parent_observation"),
            ("parent_state", {"agent_db": "changed", "user_db": None}, "parent_state"),
            ("prompt_ids", [1, 2, 4], "next_prompt_tokens"),
            ("generation", {"seed": 43}, "generation_settings")]:
            changed = copy.deepcopy(ticket); changed[key] = value
            self.assertEqual(rejection_reason(changed, parent, env, [1, 2, 3], {"seed": 42}), reason)

    async def fake_pair(self, parent_delta=1, different_serialization=False, parent_reply=False):
        real = Env()
        executor = Tau2ToolExecutor(real, floor_s=0)
        client = SimpleNamespace(model="mock", base_url="mock", timeout_s=10)
        messages = [{"role": "user", "content": "advance"}]
        isolated_states = []

        async def runner(session, client, adapter, messages, executor, turn, cfg, *, http=None):
            state = SimpleNamespace(finished_at=None, finish_reason=None,
                                    text="unfinished reasoning", token_ids=[turn, turn])
            if http is not None:
                http.state = state
                http.prompt_ids = adapter.tokenizer.encode(adapter.render_main_prompt(messages))
            if turn == 1:
                proposed = {"name": "advance", "arguments": {"delta": 1}}
                execution = await asyncio.to_thread(executor.execute_speculative, proposed)
                await asyncio.sleep(.06)  # Child is allowed to finish first.
                self.assertEqual(real.value, 0)
                self.assertEqual(execution.state.value, 1)
                if parent_reply:
                    state.finished_at = time.perf_counter()
                    state.finish_reason = "stop"
                    return {"turn": turn, "baseline_has_tool_call": False,
                            "baseline_text": "Please provide more information.",
                            "main_token_ids": state.token_ids, "turn_end_ms": 1.}
                call = {"name": "advance", "arguments": {"delta": parent_delta}}
                if parent_delta == 1:
                    commit_speculative(executor, execution)
                    result = execution.text
                else:
                    result, _ = executor.execute(call)
            else:
                call = {"name": "advance", "arguments": {"delta": 10}}
                result, _ = executor.execute(call)
                isolated_states.append(executor.env.value)
            state.finished_at = time.perf_counter()
            state.finish_reason = "stop"
            text = "<tool_call>\n" + json.dumps(call) + "\n"
            if turn == 1 and different_serialization:
                text = "<tool_call>\n" + json.dumps(call, separators=(",", ":")) + "\n"
            _append_tool_messages(messages, text, result)
            return {"turn": turn, "baseline_has_tool_call": True, "baseline_tool_name": call["name"],
                    "baseline_tool_args": call["arguments"], "tool_result_sha256": hashlib.sha256(result.encode()).hexdigest(),
                    "turn_end_ms": 1., "main_token_ids": state.token_ids}

        with patch("spork_core.successor_prefetch.run_timed_retry_turn", runner):
            entries, info = await run_successor_pair(None, client, Adapter(), messages, executor, 1,
                lambda messages, turn: RetryTurnConfig(), mode="raw")
        return executor, entries, info, isolated_states

    async def test_accepted_child_cannot_mutate_parent_before_commit(self):
        executor, entries, info, values = await self.fake_pair()
        self.assertTrue(info["adopted"])
        self.assertEqual(executor.env.value, 11)
        self.assertEqual(values, [11])
        self.assertEqual(len(entries), 2)

    async def test_wrong_parent_discards_child_state_and_runs_actual_next_turn(self):
        executor, entries, info, values = await self.fake_pair(parent_delta=2)
        self.assertFalse(info["adopted"])
        self.assertEqual(info["rejection"], "parent_call")
        self.assertEqual(executor.env.value, 12)
        self.assertEqual(values, [11, 12])
        self.assertEqual(info["observed_wasted_child_tokens"], 2)

    async def test_same_action_and_state_are_insufficient_when_prompt_differs(self):
        executor, _, info, values = await self.fake_pair(different_serialization=True)
        self.assertFalse(info["adopted"])
        self.assertEqual(info["rejection"], "next_prompt_tokens")
        self.assertEqual(executor.env.value, 11)
        self.assertEqual(values, [11, 11])

    async def test_parent_reply_discards_all_child_mutations(self):
        executor, entries, info, values = await self.fake_pair(parent_reply=True)
        self.assertFalse(info["adopted"])
        self.assertEqual(info["rejection"], "parent_has_no_tool")
        self.assertEqual(executor.env.value, 0)
        self.assertEqual(values, [11])
        self.assertEqual(len(entries), 1)

    async def test_cancellation_waits_for_actual_cleanup(self):
        done = []
        async def worker():
            try:
                await asyncio.sleep(10)
            finally:
                await asyncio.to_thread(time.sleep, .025)
                done.append(True)
        task = asyncio.create_task(worker())
        await asyncio.sleep(0)
        await cancel_and_drain([task])
        self.assertEqual(done, [True])
        self.assertTrue(task.done())


if __name__ == "__main__":
    unittest.main()
