import copy
from pathlib import Path
import tempfile
import unittest

from spork_core.evaluation_artifacts import (archive_unfinished, complete_record,
    digest, freeze_plan, planned_records, read_completed, record_path, visible_final_answers)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "runner.py"
        self.source.write_text("original source\n")
        self.output = self.root / "run"
        self.item = {"domain": "retail", "task_id": "0", "group_id": "retail:0",
                     "split": "train", "repeat": 0, "config": "spork"}
        self.plan = {"created_at": "first", "max_tokens": 8192, "schedule": [self.item],
                     "input_sha256": "inputs", "code_sha256": {"runner.py": digest(self.source)}}

    def freeze(self, plan=None):
        return freeze_plan(self.output, plan or self.plan, self.root, [self.source])

    def test_prepare_then_resume_keeps_first_manifest_and_source(self):
        first = self.freeze()
        again = copy.deepcopy(self.plan); again["created_at"] = "second"
        self.assertEqual(self.freeze(again), first)
        self.assertEqual((self.output / "code/runner.py").read_bytes(), self.source.read_bytes())

    def test_changed_budget_or_inputs_cannot_mix_into_existing_results(self):
        self.freeze()
        for key, value in [("max_tokens", 4096), ("input_sha256", "different")]:
            changed = copy.deepcopy(self.plan);changed[key] = value
            with self.assertRaisesRegex(ValueError, "Frozen"):
                self.freeze(changed)

    def test_changed_source_and_damaged_snapshot_are_rejected(self):
        self.freeze()
        self.source.write_text("changed")
        with self.assertRaises(ValueError):
            self.freeze()
        self.source.write_text("original source\n")
        (self.output / "code/runner.py").write_text("tampered snapshot")
        with self.assertRaisesRegex(ValueError, "snapshot"):
            self.freeze()

    def test_failed_run_is_completed_and_cannot_be_overwritten(self):
        manifest = self.freeze()
        path = record_path(self.output, self.item)
        record = {**self.item, "turns": [], "errors": ["generation length limit"]}
        complete_record(path, record)
        self.assertEqual(read_completed(path, self.item), record)
        self.assertEqual(planned_records(self.output, manifest)[0], [record])
        with self.assertRaisesRegex(ValueError, "overwrite"):
            complete_record(path, {**record, "errors": []})

    def test_record_corruption_or_identity_mismatch_is_visible(self):
        self.freeze();path = record_path(self.output, self.item)
        complete_record(path, {**self.item, "turns": [{}], "errors": []})
        with self.assertRaisesRegex(ValueError, "identity"):
            read_completed(path, {**self.item, "config": "refresh"})
        path.write_text("{}")
        with self.assertRaisesRegex(ValueError, "changed"):
            read_completed(path)

    def test_unsealed_partial_is_archived_instead_of_treated_as_success(self):
        self.freeze();path = record_path(self.output, self.item)
        path.parent.mkdir();path.write_text('{"incomplete":')
        self.assertIsNone(read_completed(path))
        archived = Path(archive_unfinished(path))
        self.assertEqual(archived.read_text(), '{"incomplete":')
        self.assertFalse(path.exists())

    def test_answer_comparison_excludes_reasoning_but_keeps_changed_content(self):
        def record(text):
            return {"turns": [{"baseline_has_tool_call": False, "final_answer": text}]}
        a = record("<think>long reasoning</think> Please provide the flight number.\n")
        b = record("<think>different reasoning</think>\nPlease provide the flight number.")
        self.assertEqual(visible_final_answers(a), visible_final_answers(b))
        c = record("<think>reason</think>Your refund was issued.")
        self.assertNotEqual(visible_final_answers(a), visible_final_answers(c))
        self.assertEqual(visible_final_answers({"turns": [{"baseline_has_tool_call": True}]}), [])


if __name__ == "__main__":
    unittest.main()
