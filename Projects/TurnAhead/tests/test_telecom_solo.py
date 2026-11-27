"""Exercise the actual Qwen turn loop and tau2 grader with scripted model outputs."""
import asyncio
import json
import os
import time
import unittest
from unittest.mock import patch

from spork_core.benchmarks import telecom_solo
from spork_core.prefix_replay import PrefixHttpClient
from spork_core.successor_task import run_successor_task
from spork_core.turn_runner import RetryTurnConfig
from spork_core.vllm_client import VllmClient


class TelecomSoloTest(unittest.TestCase):
    def test_ticket_to_official_reward(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(os.environ["SPORK_CODEX_MODEL"], local_files_only=True)
        scheduled, tasks = telecom_solo.task_plan("test", None)
        self.assertEqual(len(scheduled), 40)
        for basis_size in (1, 2):
            task = next(t for t in tasks.values() if len(t["evaluation_criteria"]["reward_basis"]) == basis_size)
            adapter, history, executor = telecom_solo.setup(tokenizer, task, 0.)
            self.assertIn(task["ticket"], history[0]["content"])
            calls = [{"name": a["name"], "arguments": a["arguments"]} for a in task["evaluation_criteria"]["actions"]]
            before = executor.env.get_user_db_hash()
            isolated = executor.execute_speculative(calls[0])
            self.assertEqual(executor.env.get_user_db_hash(), before)
            self.assertIsInstance(executor.fork_at(isolated.state), telecom_solo.SoloExecutor)
            calls.append({"name": "done", "arguments": {}})
            outputs = iter(calls)

            async def stream(http, session, prompt, state, **kwargs):
                state.text = "Finished thinking.</think>\n<tool_call>\n" + json.dumps(next(outputs))
                state.token_ids = tokenizer.encode(state.text, add_special_tokens=False)
                state.first_output_at = state.finished_at = time.perf_counter()
                state.finish_reason = "stop"
                state.changed.set()

            def config(messages, turn):
                return RetryTurnConfig(launch_mode="off", max_retries=0)

            with patch.object(PrefixHttpClient, "stream", stream):
                turns, trace = asyncio.run(run_successor_task(None, VllmClient("mock", "mock"),
                    adapter, history, executor, config, mode="baseline", max_turns=len(calls)+1))
            self.assertFalse(trace["turn_limit_reached"])
            self.assertEqual(turns[-1]["terminal_tool"], "done")
            record = dict(turns=turns, session_id="cpu", errors=[], wall_s=0.,
                          started_at="2026-11-27T00:00:00", ended_at="2026-11-27T00:00:00")
            self.assertEqual(telecom_solo.score(task, record)["reward"], 1.)
            turns.pop()
            self.assertEqual(telecom_solo.score(task, record)["reward"], 0.)
