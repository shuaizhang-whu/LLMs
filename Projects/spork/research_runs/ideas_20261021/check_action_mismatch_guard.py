"""CPU reproduction of a tool hallucination after a stable early-exit probe."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
from test_early_exit import CheckedAdapter, Client, Executor, HTTP
from spork_core.turn_runner import RetryTurnConfig

spec = importlib.util.spec_from_file_location("spork_core.guarded_refresh_preview", HERE / "guarded_refresh_preview.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class InventedFlight(HTTP):
    async def complete(self, session, ids, **kwargs):
        result = await super().complete(session, ids, **kwargs)
        text = bytes(ids).decode()
        if kwargs["max_tokens"] != 1 and text.endswith("</think>\n\n"):
            result["text"] = '<tool_call>{"name":"get_flight_status","arguments":{"flight_number":"AA123","date":"2024-04-01"}}'
            result["token_ids"] = list(result["text"].encode())
        elif kwargs["max_tokens"] != 1 and not text.endswith('{"name": "'):
            result["text"] = 'More thought.</think>\nPlease provide the flight number and date.'
            result["token_ids"] = list(result["text"].encode())
        return result


class GuardTests(unittest.IsolatedAsyncioTestCase):
    async def run_case(self, guard, http):
        executor = Executor()
        messages = [{"role": "user", "content": "My flight was cancelled; I want compensation."}]

        async def clear():
            return {"clear_status": "cleared"}

        cfg = RetryTurnConfig(max_retries=2, d3_clear=clear, baseline_max_tokens=8192,
                              context_window_tokens=40960, enable_thinking=True)
        result = await module.run_early_exit_turn(None, Client(), CheckedAdapter(), messages, executor, 1,
            cfg, module.ExitConfig(method="eta", stable_checks=2, check_interval=10, max_checks=4,
                                   rollback_on_action_mismatch=guard), http=http)
        return result, executor, messages

    async def test_original_policy_executes_the_disagreeing_shortened_call(self):
        result, tools, _ = await self.run_case(False, InventedFlight())
        self.assertTrue(result["exit"]["fired"])
        self.assertFalse(result["exit"]["rolled_back"])
        self.assertEqual(tools.canonical_calls[-1]["name"], "get_flight_status")

    async def test_guard_rolls_back_before_the_invented_call_reaches_tools(self):
        result, tools, messages = await self.run_case(True, InventedFlight())
        self.assertTrue(result["exit"]["rolled_back"])
        self.assertEqual(result["exit"]["rollback_reason"], "candidate_disagreement")
        self.assertFalse(result["baseline_has_tool_call"])
        self.assertIn("Please provide", result["final_answer"])
        self.assertNotIn("AA123", json.dumps(messages))
        self.assertFalse(any(c["name"] == "get_flight_status" for c in tools.canonical_calls))

    async def test_guard_keeps_a_matching_early_exit(self):
        result, tools, _ = await self.run_case(True, HTTP())
        self.assertTrue(result["exit"]["fired"])
        self.assertFalse(result["exit"]["rolled_back"])
        self.assertTrue(result["spec_used"])
        self.assertEqual(tools.commits, 1)


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(GuardTests))
    (HERE / "action_mismatch_guard_cpu.json").write_text(json.dumps({"tests": result.testsRun,
        "passed": result.wasSuccessful(), "scope": "CPU control-flow reproduction only; no GPU or task-success evidence"}, indent=2) + "\n")
    sys.exit(0 if result.wasSuccessful() else 1)
