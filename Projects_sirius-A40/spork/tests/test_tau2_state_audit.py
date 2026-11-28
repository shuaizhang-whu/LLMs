import unittest

from scripts.audit_tau2_state import assess_record


class StateAuditTests(unittest.TestCase):
    def record(self, turns, errors=None, capped=False):
        return {"group_id": "retail:1", "repeat": 0, "method": "eta", "turns": turns,
                "errors": errors or [], "turn_limit_reached": capped}

    def test_failed_empty_record_is_not_scored_as_unchanged_target_state(self):
        ref = {"state": {"agent_db": "unchanged", "user_db": None}, "errors": []}
        row = assess_record(self.record([], ["length"]), ref)
        self.assertIsNone(row["db_match"])
        self.assertEqual(row["db_comparison_unavailable"], "missing_final_state")

    def test_db_match_keeps_cap_and_does_not_claim_official_success(self):
        state = {"agent_db": "same", "user_db": None}
        row = assess_record(self.record([{"environment_after": state, "baseline_has_tool_call": True}],
                                        capped=True), {"state": state, "errors": []})
        self.assertTrue(row["db_match"])
        self.assertTrue(row["turn_capped"])
        self.assertIsNone(row["official_task_success"])

    def test_reference_errors_prevent_comparing_to_partial_target(self):
        state = {"agent_db": "same", "user_db": None}
        row = assess_record(self.record([{"environment_after": state}]),
                            {"state": state, "errors": [{"error": "bad action"}]})
        self.assertIsNone(row["db_match"])
        self.assertEqual(row["db_comparison_unavailable"], "reference_action_error")

    def test_failed_reference_lookup_with_no_state_change_remains_comparable(self):
        state = {"agent_db": "same", "user_db": None}
        row = assess_record(self.record([{"environment_after": state}]),
                            {"state": state, "errors": [{"error": "User not found",
                                                         "mutating": False, "state_unchanged": True}]})
        self.assertTrue(row["db_match"])


if __name__ == "__main__":
    unittest.main()
