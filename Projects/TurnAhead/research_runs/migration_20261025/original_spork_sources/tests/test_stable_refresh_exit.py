"""CPU checks for replacement eligibility and isolation of discarded writes."""
import copy
import unittest

from spork_core.stable_refresh_exit import ExitConfig, StableCandidate
from spork_core.executors import Tau2ToolExecutor, execute_speculative, commit_speculative


OLD = {"name": "lookup", "arguments": {"id": "old"}}
NEW = {"name": "lookup", "arguments": {"id": "new"}}


class CounterEnvironment:
    def __init__(self):
        self.value = 0
        self.tools = self

    def add(self, amount):
        self.value += amount
        return {"value": self.value}


class RefreshTests(unittest.TestCase):
    def eligible(self, tracker, **changes):
        args = dict(required=3, refreshes=0, limit=1, active_done=True)
        args.update(changes)
        return tracker.should_refresh(OLD, **args)

    def test_waits_for_three_consecutive_matching_candidates(self):
        tracker = StableCandidate()
        for _ in range(2):
            tracker.observe(NEW, True)
            self.assertFalse(self.eligible(tracker))
        tracker.observe(NEW, True)
        self.assertTrue(self.eligible(tracker))

    def test_argument_changes_reset_confirmation(self):
        tracker = StableCandidate()
        for call in (NEW, NEW, OLD, NEW):
            tracker.observe(call, True)
        self.assertEqual(tracker.count, 1)
        self.assertFalse(self.eligible(tracker))

    def test_no_call_or_low_probability_breaks_stability(self):
        for call, eligible in ((None, True), (NEW, False)):
            tracker = StableCandidate()
            tracker.observe(NEW, True)
            tracker.observe(NEW, True)
            tracker.observe(call, eligible)
            self.assertEqual(tracker.count, 0)
            self.assertFalse(self.eligible(tracker))

    def test_cannot_replace_busy_tool_or_exceed_budget(self):
        tracker = StableCandidate()
        for _ in range(3):
            tracker.observe(NEW, True)
        self.assertFalse(self.eligible(tracker, active_done=False))
        self.assertFalse(self.eligible(tracker, refreshes=1))
        self.assertFalse(self.eligible(tracker, limit=0))

    def test_same_active_candidate_is_not_reexecuted(self):
        tracker = StableCandidate()
        for _ in range(3):
            tracker.observe(OLD, True)
        self.assertFalse(self.eligible(tracker))

    def test_caller_mutation_cannot_rewrite_previous_evidence(self):
        tracker = StableCandidate()
        call = copy.deepcopy(NEW)
        tracker.observe(call, True)
        call["arguments"]["id"] = "changed"
        self.assertEqual(tracker.call, NEW)

    def test_replacement_starts_from_committed_not_obsolete_state(self):
        executor = Tau2ToolExecutor(CounterEnvironment(), floor_s=0)
        obsolete = execute_speculative(executor, {"name": "add", "arguments": {"amount": 100}})
        replacement = execute_speculative(executor, {"name": "add", "arguments": {"amount": 3}})
        self.assertEqual(executor.env.value, 0)
        self.assertEqual(obsolete.state.value, 100)
        self.assertEqual(replacement.state.value, 3)
        commit_speculative(executor, replacement)
        self.assertEqual(executor.env.value, 3)

    def test_invalid_configuration_rejected(self):
        for config in (ExitConfig(max_refreshes=3), ExitConfig(stable_checks=1),
                       ExitConfig(method="unknown")):
            with self.assertRaises(ValueError):
                config.validate()


if __name__ == "__main__":
    unittest.main()
