"""可重复 Runner 的固定样本、解析与 lossless 报告测试。"""

import json
from pathlib import Path
import tempfile
import unittest

from src.experiment_report import validate_lossless_pair
from src.reproduction_runner import FIXED_SAMPLE_IDS, ReproductionRunner, parse_sample_ids


class ReproductionRunnerTest(unittest.TestCase):
    def test_fixed_seed_sample_order_matches_paper_protocol(self):
        self.assertEqual(
            FIXED_SAMPLE_IDS,
            [
                7107, 5619, 373, 6904, 1267, 4202, 2316, 6, 1248, 7356,
                1477, 1354, 7090, 3021, 2038, 4349, 2885, 2861, 4948, 2527,
            ],
        )

    def test_sample_id_parser_preserves_order_and_rejects_duplicates(self):
        self.assertEqual(parse_sample_ids("7107, 5619"), [7107, 5619])
        with self.assertRaises(ValueError):
            parse_sample_ids("7107,7107")

    def test_action_extraction_preserves_model_output_order(self):
        text = "Thought: x. Action 1: Lookup[a], Search[b], Finish[c]"
        self.assertEqual(
            ReproductionRunner.extract_actions(text),
            ["Lookup[a]", "Search[b]", "Finish[c]"],
        )

    def test_lossless_validator_compares_actions_and_task_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            sequential = Path(tmp) / "sequential"
            online = Path(tmp) / "online"
            for root in (sequential, online):
                sample = root / "7107"
                sample.mkdir(parents=True)
                (sample / "authority_trace.json").write_text(
                    json.dumps(
                        {
                            "sample_id": 7107,
                            "steps": [{"action": "Search[Alpha]"}, {"action": "Finish[x]"}],
                        }
                    ),
                    encoding="utf-8",
                )
                (sample / "sample_metrics.json").write_text(
                    json.dumps(
                        {
                            "actual_question_idx": 7107,
                            "task": {
                                "answer": "x",
                                "ground_truth": "x",
                                "em": 1.0,
                                "f1": 1.0,
                            },
                        }
                    ),
                    encoding="utf-8",
                )

            result = validate_lossless_pair(sequential, online)

        self.assertTrue(result["passed"])


if __name__ == "__main__":
    unittest.main()
