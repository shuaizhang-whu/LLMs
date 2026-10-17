"""HotPotQA 离线动作指标的最小回归测试。"""

import unittest

from src.metrics import Metrics


class MetricsSemanticsTest(unittest.TestCase):
    """锁定官方轨迹指标当前实际采用的匹配与计时口径。"""

    def test_first_step_is_always_skipped(self):
        normal = {"actions": ["Search[ignored]", "Finish[answer]"]}
        simulated = {"actions": [["Search[ignored]"], ["Finish[answer]"]]}

        result = Metrics.get_action_metrics(normal, simulated, k=1)

        self.assertEqual(result["general"], 1)
        self.assertNotIn("Search", result)
        self.assertEqual(result["Finish"], 1)

    def test_top1_only_checks_the_first_candidate(self):
        normal = {"actions": ["Search[ignored]", "Search[target]"]}
        simulated = {
            "actions": [
                ["Search[ignored]"],
                ["Lookup[wrong]", "Search[target]", "Finish[wrong]"],
            ]
        }

        self.assertEqual(
            Metrics.get_action_metrics(normal, simulated, k=1)["general"], 0
        )
        self.assertEqual(
            Metrics.get_action_metrics(normal, simulated, k=3)["general"], 1
        )

    def test_top3_strictly_truncates_extra_candidates(self):
        normal = {"actions": ["Search[ignored]", "Search[target]"]}
        simulated = {
            "actions": [
                ["Search[ignored]"],
                ["Lookup[a]", "Lookup[b]", "Lookup[c]", "Search[target]"],
            ]
        }

        self.assertEqual(
            Metrics.get_action_metrics(normal, simulated, k=3)["general"], 0
        )
        self.assertEqual(
            Metrics.get_action_metrics(normal, simulated, k=None)["general"], 1
        )

    def test_all_steps_audit_includes_first_action(self):
        normal = {"actions": ["Search[first]", "Finish[answer]"]}
        simulated = {"actions": [["Lookup[wrong]"], ["Finish[answer]"]]}

        paper = Metrics.get_action_metrics(normal, simulated, k=1, skip_first=True)
        audit = Metrics.get_action_metrics(normal, simulated, k=1, skip_first=False)

        self.assertEqual(paper["general"], 1)
        self.assertEqual(audit["general"], 0.5)

    def test_full_action_comparison_is_case_insensitive(self):
        self.assertEqual(Metrics.compare_action("Search[Alpha]", "search[alpha]"), 1)
        self.assertEqual(Metrics.compare_action("Search[Alpha]", "search[beta]"), 0)

    def test_empty_candidate_list_scores_zero(self):
        normal = {"actions": ["Search[ignored]", "Search[target]"]}
        simulated = {"actions": [["Search[ignored]"], []]}

        self.assertEqual(
            Metrics.get_action_metrics(normal, simulated, k=3)["general"], 0
        )

    def test_search_time_ignores_other_actions_and_counts_a_step_once(self):
        observations = {
            "actions": [
                "Search[first]",
                "Lookup[item]",
                ["Search[second]", "Search[duplicate]"],
                [],
            ],
            "time_taken": [2.0, 99.0, 4.0, 50.0],
        }

        self.assertEqual(Metrics.get_avg_time_taken(observations), 3.0)


if __name__ == "__main__":
    unittest.main()
