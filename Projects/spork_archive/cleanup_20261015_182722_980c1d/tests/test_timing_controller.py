"""CPU-only correctness checks; all test encoders are tiny and randomly initialized."""
import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from spork_core.controller_data import (
    ControllerInputEncoder, QUALITY_NAMES, load_controller_rows, partition_controller_rows, split_by_task, visible_text_fields,
)
from spork_core.timing_controller import ControllerConfig, TimingController
from scripts.train_timing_controller import run, tiny_random_controller


def make_row(sid, task, targets=(1., -2.)):
    return {
        "id": sid, "group_id": "retail:" + task, "split": "pilot",
        "input": {
            "messages": [{"role": "system", "content": "Follow the account policy."},
                         {"role": "user", "content": "Please look up my account."},
                         {"role": "assistant", "content": "lookup account"},
                         {"role": "tool", "content": "Account found with two orders."}],
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
            "prefix_text": "<think> I need account details reasonneedle",
            "visible_generated_tokens": 32, "wait_tokens": 64,
            "prompt_token_ids": [190000], "prefix_token_ids": [180000],
        },
        "targets": dict(zip(("gain_now_s", "gain_wait_s"), targets)),
        "label_valid": True, "quality_flags": {**{k: True for k in QUALITY_NAMES},
                                                   "task_success_evaluated": False},
        "metadata": {"domain": "retail", "task_id": task, "model": "test-actor",
                     "measurement": "test-replay", "tool_floor_s": 2.},
    }


class DataTests(unittest.TestCase):
    def load(self, rows, policy="matched"):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "data.jsonl"
            p.write_text("".join(json.dumps(r) + "\n" for r in rows))
            return load_controller_rows([p], policy)

    def test_quality_selection_is_explicit_and_keeps_negative_targets(self):
        good, flagged = make_row("a", "1"), make_row("b", "2")
        flagged["quality_flags"]["wait_same_state_as_off"] = False
        kept, report = self.load([good, flagged])
        self.assertEqual([r["id"] for r in kept], ["a"])
        self.assertEqual(kept[0]["targets"]["gain_wait_s"], -2.)
        self.assertEqual(report["excluded"][0]["reason"], "outcomes_differ")
        self.assertEqual(len(self.load([good, flagged], "all-valid")[0]), 2)

    def test_duplicate_sample_ids_and_wrong_groups_are_rejected(self):
        row = make_row("a", "1")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.load([row, row])
        row["group_id"] = "retail:other"
        with self.assertRaisesRegex(ValueError, "group"):
            self.load([row])

    def test_nonfinite_labels_and_mixed_cost_settings_are_rejected(self):
        row = make_row("a", "1", (float("nan"), 1.))
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            self.load([row])
        a, b = make_row("a", "1"), make_row("b", "2")
        b["metadata"]["tool_floor_s"] = 10.
        with self.assertRaisesRegex(ValueError, "Mixed"):
            self.load([a, b])

    def test_task_groups_do_not_leak_between_splits(self):
        rows = [make_row(str(i), str(i // 2)) for i in range(8)]
        train, val, split = split_by_task(rows, .25, 42)
        self.assertFalse({r["group_id"] for r in train} & {r["group_id"] for r in val})
        self.assertEqual(len(train) + len(val), len(rows))
        self.assertEqual(split, split_by_task(rows, .25, 42)[2])

    def test_predefined_test_tasks_stay_separate(self):
        rows = [make_row(str(i), str(i)) for i in range(3)]
        for row, split in zip(rows, ("train", "validation", "test")):
            row["split"] = split
        train, validation, test, report = partition_controller_rows(rows)
        self.assertEqual([r["id"] for r in train], ["0"])
        self.assertEqual([r["id"] for r in validation], ["1"])
        self.assertEqual([r["id"] for r in test], ["2"])
        self.assertFalse(report["test_used_for_checkpoint_selection"])
        rows[-1]["group_id"] = rows[0]["group_id"]
        with self.assertRaisesRegex(ValueError, "crosses"):
            partition_controller_rows(rows)

    def test_pilot_cannot_be_mixed_with_predefined_split(self):
        rows = [make_row("a", "1"), make_row("b", "2")]
        rows[1]["split"] = "train"
        with self.assertRaisesRegex(ValueError, "mix"):
            partition_controller_rows(rows)


class ControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(7)
        self.rows = [make_row("a", "1"), make_row("b", "2", (3., 4.))]
        self.model, self.tokenizer = tiny_random_controller(
            self.rows, ControllerConfig(max_length=64, head_hidden_size=16, dropout=0.))
        self.processor = ControllerInputEncoder(self.tokenizer, 64)

    def test_future_outcomes_and_actor_ids_are_not_encoder_inputs(self):
        row = self.rows[0]
        original = self.processor.encode(row["input"])
        changed = copy.deepcopy(row)
        changed["targets"] = {"gain_now_s": 99999., "gain_wait_s": -99999.}
        changed["metadata"]["future_action"] = "FUTURE_SECRET"
        changed["input"]["future_action"] = "FUTURE_SECRET"
        changed["input"]["prompt_token_ids"] = [999999999]
        self.assertEqual(original, self.processor.encode(changed["input"]))
        self.assertNotIn("FUTURE_SECRET", " ".join(visible_text_fields(changed["input"]).values()))
        self.assertLess(max(original["input_ids"]), len(self.tokenizer))

    def test_reasoning_is_kept_when_history_is_long(self):
        inp = copy.deepcopy(self.rows[0]["input"])
        inp["messages"][0]["content"] *= 100
        inp["messages"][-1]["content"] *= 100
        encoded = self.processor.encode(inp)
        self.assertLessEqual(len(encoded["input_ids"]), 64)
        self.assertIn(self.tokenizer.convert_tokens_to_ids("reasonneedle"), encoded["input_ids"])
        self.assertLess(encoded["token_counts"]["context"]["used"], encoded["token_counts"]["context"]["original"])

    def test_reasoning_ablation_preserves_other_fields_and_checkpoint_behavior(self):
        inp = copy.deepcopy(self.rows[0]["input"])
        inp["messages"][0]["content"] *= 30
        full = self.processor.encode(inp)
        masked_processor = ControllerInputEncoder(self.tokenizer, 64, "mask_reasoning")
        masked = masked_processor.encode(inp)
        self.assertEqual(full["token_counts"], masked["token_counts"])
        self.assertEqual(full["numeric_features"], masked["numeric_features"])
        n = full["token_counts"]["reasoning"]["used"]
        self.assertEqual(full["input_ids"][:-n-1], masked["input_ids"][:-n-1])
        self.assertEqual(masked["input_ids"][-n-1:-1], [self.tokenizer.mask_token_id] * n)
        self.assertEqual(len(full["input_ids"]), len(masked["input_ids"]))
        self.model.config.input_ablation = "mask_reasoning"
        batch = masked_processor.collate([masked])
        self.model.eval()
        with torch.no_grad():
            expected = self.model(**batch).tolist()[0]
        with tempfile.TemporaryDirectory() as tmp:
            self.model.save(Path(tmp) / "ckpt", self.tokenizer)
            restored, tokenizer = TimingController.load(Path(tmp) / "ckpt")
            prediction = restored.predict_inputs([inp], tokenizer)[0]
        self.assertEqual(restored.config.input_ablation, "mask_reasoning")
        self.assertEqual([prediction["q_now_s"], prediction["q_wait_s"]], expected)

    def test_values_are_in_seconds_and_policy_ties_choose_off(self):
        self.model.fit_target_scaling([[1., 3.], [3., 7.]])
        torch.nn.init.zeros_(self.model.head[-1].weight)
        torch.nn.init.zeros_(self.model.head[-1].bias)
        batch = self.processor.collate([self.processor.encode(r["input"]) for r in self.rows])
        self.model.eval()
        values = self.model(**batch)
        torch.testing.assert_close(values, torch.tensor([[2., 5.], [2., 5.]]))
        picks = TimingController.policies(torch.tensor([[-1., -2.], [0., 0.], [2., 1.], [1., 3.]]))
        self.assertEqual(picks.tolist(), [0, 0, 1, 2])

    def test_padding_does_not_change_values(self):
        batch = self.processor.collate([self.processor.encode(r["input"]) for r in self.rows])
        self.model.eval()
        with torch.no_grad():
            expected = self.model(**batch)
            batch["input_ids"] = torch.nn.functional.pad(batch["input_ids"], (0, 3), value=self.tokenizer.pad_token_id)
            batch["attention_mask"] = torch.nn.functional.pad(batch["attention_mask"], (0, 3), value=0)
            actual = self.model(**batch)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_checkpoint_restores_scaling_tokenizer_and_predictions(self):
        self.model.fit_target_scaling([[1., 3.], [3., 7.]])
        self.model.eval()
        inputs = [r["input"] for r in self.rows]
        expected = self.model.predict_inputs(inputs, self.tokenizer)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint"
            self.model.save(path, self.tokenizer)
            restored, tokenizer = TimingController.load(path)
            actual = restored.predict_inputs(inputs, tokenizer)
            self.assertEqual(expected, actual)
            self.assertEqual(restored.predict_inputs([], tokenizer), [])
            with self.assertRaises(FileExistsError):
                restored.save(path, tokenizer)

    def test_train_entrypoint_uses_local_fixture_and_disjoint_groups(self):
        # One tiny synthetic epoch validates the actual train branch. It is a
        # CPU unit test, not a fit/evaluation on the collected pilot measurements.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            encoder = root / "local_encoder_fixture"
            self.model.encoder.save_pretrained(encoder)
            self.tokenizer.save_pretrained(encoder)
            rows = [make_row(str(i), str(i // 2), (float(i), float(-i))) for i in range(6)]
            for i, row in enumerate(rows):
                row["split"] = ("train", "validation", "test")[i // 2]
                if row["split"] == "test":
                    row["targets"] = {"gain_now_s": 99999., "gain_wait_s": -99999.}
            data = root / "synthetic.jsonl"
            data.write_text("".join(json.dumps(r) + "\n" for r in rows))
            args = argparse.Namespace(mode="train", encoder=encoder, device="cpu", cpu_threads=2,
                                      seed=42, data=[data], quality_policy="matched", validation_fraction=.5,
                                      output=root / "output", max_length=64, head_hidden_size=16, dropout=0.,
                                      encoder_lr=2e-5, head_lr=1e-3, weight_decay=.01, batch_size=2, epochs=1)
            with contextlib.redirect_stdout(io.StringIO()):
                result = run(args)
            split = json.loads((args.output / "split.json").read_text())
            self.assertFalse(set(split["train_groups"]) & set(split["validation_groups"]))
            train_targets = torch.tensor([[r["targets"]["gain_now_s"], r["targets"]["gain_wait_s"]]
                                          for r in rows if r["id"] in split["train_ids"]])
            manifest = json.loads((args.output / "manifest.json").read_text())
            self.assertEqual(manifest["target_mean_from_training_only"], train_targets.mean(0).tolist())
            restored, _ = TimingController.load(args.output / result["best_checkpoint"])
            torch.testing.assert_close(restored.target_mean, train_targets.mean(0))
            self.assertEqual(result["epochs_completed"], 1)
            self.assertEqual(result["reserved_test_rows"], 2)
            self.assertFalse(result["test_evaluated"])
            self.assertFalse(manifest["test_used_for_training_or_checkpoint_selection"])
            self.assertFalse(result["online_performance_evaluated"])


if __name__ == "__main__":
    unittest.main()
