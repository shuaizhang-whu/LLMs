"""CPU orchestration checks; these do not simulate GPU speed or task success."""
import unittest
from unittest.mock import patch

from spork_core.successor_task import run_successor_task


class Tests(unittest.IsolatedAsyncioTestCase):
    async def simulate(self, max_turns=8, reply_at=99, fail_at=None):
        observed, calls, trace = [], [], {}
        async def pair(session, client, adapter, messages, executor, turn, make_cfg, *, mode, on_turn, trace):
            calls.append(("pair", turn))
            trace.update(adopted=True, observed_wasted_child_tokens=3, errors=[])
            for current in (turn, turn + 1):
                if current == fail_at:
                    raise RuntimeError("transport failure")
                on_turn({"turn": current, "baseline_has_tool_call": current != reply_at})
                if current == reply_at:
                    break
        async def single(session, client, adapter, messages, executor, turn, cfg):
            calls.append(("single", turn))
            return {"turn": turn, "baseline_has_tool_call": turn != reply_at}
        with patch("spork_core.successor_task.run_successor_pair", pair), patch("spork_core.successor_task.run_timed_retry_turn", single):
            try:
                result = await run_successor_task(None, None, None, [], None, lambda m, t: None,
                    max_turns=max_turns, on_turn=observed.append, trace=trace)
            except RuntimeError as error:
                result = error
        return result, observed, calls, trace

    async def test_disjoint_pairs_cover_all_eight_turns(self):
        result, observed, calls, trace = await self.simulate()
        self.assertEqual(calls, [("pair", 1), ("pair", 3), ("pair", 5), ("pair", 7)])
        self.assertEqual([t["turn"] for t in observed], list(range(1, 9)))
        self.assertTrue(trace["turn_limit_reached"])
        self.assertEqual(trace["adopted"], 4)
        self.assertEqual(trace["observed_wasted_child_tokens"], 12)

    async def test_final_reply_stops_future_pairs(self):
        _, observed, calls, trace = await self.simulate(reply_at=3)
        self.assertEqual(calls, [("pair", 1), ("pair", 3)])
        self.assertEqual(len(observed), 3)
        self.assertFalse(trace["turn_limit_reached"])

    async def test_odd_budget_has_no_unusable_successor(self):
        _, observed, calls, _ = await self.simulate(max_turns=5)
        self.assertEqual(calls, [("pair", 1), ("pair", 3), ("single", 5)])
        self.assertEqual(len(observed), 5)

    async def test_single_turn_budget_runs_only_original_spork(self):
        _, observed, calls, trace = await self.simulate(max_turns=1)
        self.assertEqual(calls, [("single", 1)])
        self.assertEqual(trace["adopted"], 0)

    async def test_failure_keeps_previously_committed_turns(self):
        result, observed, _, trace = await self.simulate(fail_at=4)
        self.assertIsInstance(result, RuntimeError)
        self.assertEqual([t["turn"] for t in observed], [1, 2, 3])
        self.assertEqual(len(trace["pairs"]), 2)
        self.assertIn("task_end_s", trace)


if __name__ == "__main__":
    unittest.main()
