"""CPU review of the independently edited spork_v2 redispatch implementation.

Only the HTTP boundary and tools are faked. The target runner is loaded without
modifying its source. Tests cover late superseded workers and commit ownership.
"""
import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT.parent / "spork_v2/spork_core/early_exit.py"
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
from test_early_exit import CheckedAdapter, Client, HTTP
from spork_core.executors import SpeculativeToolExecution
from spork_core.turn_runner import RetryTurnConfig

spec = importlib.util.spec_from_file_location("spork_core.reviewed_early_exit", TARGET)
reviewed = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = reviewed
spec.loader.exec_module(reviewed)
SOURCE_HASH = hashlib.sha256(TARGET.read_bytes()).hexdigest()


class Environment:
    def __init__(self, value="initial"):
        self.value = value

    def get_db_hash(self):
        return self.value

    def get_user_db_hash(self):
        return None


class Tools:
    def __init__(self):
        self.env = Environment()
        self.finished, self.committed, self.canonical = [], [], []
        self.last_real_wall_s = self.last_floor_sleep_s = 0.

    def execute_speculative(self, call):
        value = call["arguments"]["id"]
        duration = .08 if value == "a" else .001
        time.sleep(duration)
        self.finished.append(value)
        return SpeculativeToolExecution(value, duration, duration, 0., Environment(value))

    def commit_speculative(self, result):
        self.committed.append(result.text)
        self.env = result.state

    def execute(self, call):
        value = call["arguments"]["id"]
        self.canonical.append(value)
        self.env = Environment(value)
        return value, 0.


class SlowContinuation(HTTP):
    def __init__(self, final_id):
        super().__init__("different")
        self.final_id = final_id

    async def complete(self, session, ids, **kwargs):
        result = await super().complete(session, ids, **kwargs)
        if kwargs["max_tokens"] != 1 and bytes(ids).decode().endswith("</think>\n\n"):
            # Let the superseded old worker finish after the new tool is ready
            # but before the record is constructed.
            await asyncio.sleep(.10)
            result["text"] = '<tool_call>' + json.dumps({"name": "lookup", "arguments": {"id": self.final_id}})
            result["token_ids"] = list(result["text"].encode())
        return result


class Contracts(unittest.IsolatedAsyncioTestCase):
    async def execute(self, final_id):
        tools = Tools()

        async def check(*args, **kwargs):
            return {"call_text": '<tool_call>{"name":"lookup","arguments":{"id":"b"}}',
                    "finish_reason": "stop", "tool_probability": .99}

        async def clear():
            return {"clear_status": "cleared"}

        cfg = RetryTurnConfig(max_retries=2, d3_clear=clear, baseline_max_tokens=8192,
                              context_window_tokens=40960, enable_thinking=True)
        with patch.object(reviewed, "decision_probe", new=check):
            result = await reviewed.run_early_exit_turn(None, Client(), CheckedAdapter(),
                [{"role": "user", "content": "lookup"}], tools, 1, cfg,
                reviewed.ExitConfig(method="redo", stable_checks=3, check_interval=10, max_checks=6),
                http=SlowContinuation(final_id))
        return result, tools

    async def test_late_old_worker_cannot_replace_new_result_or_ready_timestamp(self):
        result, tools = await self.execute("b")
        self.assertEqual(result["redispatch_count"], 1)
        self.assertTrue(result["exit"]["fired"])
        self.assertTrue(result["spec_used"])
        self.assertEqual(tools.committed, ["b"])
        self.assertEqual(tools.env.value, "b")
        self.assertEqual(sorted(tools.finished), ["a", "b"])
        self.assertLessEqual(result["spec_end_ms"], result["exit"]["cut_ms"])

    async def test_formal_mismatch_commits_neither_speculative_state(self):
        result, tools = await self.execute("c")
        self.assertEqual(result["redispatch_count"], 1)
        self.assertFalse(result["spec_used"])
        self.assertEqual(tools.committed, [])
        self.assertEqual(tools.canonical, ["c"])
        self.assertEqual(tools.env.value, "c")
        self.assertEqual(sorted(tools.finished), ["a", "b"])


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Contracts)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {"target": str(TARGET), "source_sha256": SOURCE_HASH,
              "source_unchanged_during_check": hashlib.sha256(TARGET.read_bytes()).hexdigest() == SOURCE_HASH,
              "tests_run": result.testsRun, "passed": result.wasSuccessful(),
              "scope": "CPU lifecycle/commit tests only; no GPU inference or performance evidence"}
    (Path(__file__).parent / "redispatch_contract_review.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    sys.exit(0 if result.wasSuccessful() else 1)
