"""Short synthetic CPU checks. No collected data or pretrained models are fit."""
import argparse
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from test_timing_controller import make_row
from scripts.run_controller_training_experiments import preflight, run
from spork_core.timing_controller import TimingController
from spork_core.controller_evaluation import (
    bootstrap_task_difference, choose_policy, fit_simple_baselines, score_policy, task_values,
)


class EvaluationTests(unittest.TestCase):
    def test_task_weighting_is_not_turn_weighting_and_regret_is_recorded(self):
        rows = [make_row("a", "1", (3., 2.)), make_row("b", "1", (3., 2.)),
                make_row("c", "2", (-1., 0.))]
        result = score_policy(rows, [1, 1, 1])
        self.assertAlmostEqual(result["offline_selected_recorded_gain_s"], 5 / 3)
        self.assertEqual(result["task_mean_recorded_gain_s"], 1.)
        self.assertEqual(result["task_regret_to_empirical_oracle_s"], .5)
        self.assertEqual(result["selected_negative_gain_rows"], 1)
        self.assertEqual(task_values(rows, [1., 3., 8.]), {"retail:1": 2., "retail:2": 8.})
        self.assertEqual(choose_policy([-1., -2.]), 0)
        self.assertEqual(choose_policy([0., 0.]), 0)

    def test_simple_baseline_ignores_validation_labels_and_future_metadata(self):
        train = [make_row("a", "1", (2., 0.)), make_row("b", "2", (3., 1.))]
        val = [make_row("c", "3", (-1., 90.))]
        expected, estimates = fit_simple_baselines(train, val)
        altered = copy.deepcopy(val)
        altered[0]["targets"] = {"gain_now_s": 900., "gain_wait_s": -999.}
        altered[0]["metadata"]["future_tool"] = "wait is the answer"
        altered[0]["id"] = "future_sample_id"
        actual, altered_estimates = fit_simple_baselines(train, altered)
        self.assertEqual(actual["policies"], expected["policies"])
        self.assertEqual(altered_estimates, estimates)
        self.assertEqual(expected["train_best_constant_policy"], "now")
        # An unseen visible tool vocabulary falls back to TRAIN means.
        altered[0]["input"]["tools"] = [{"name": "unseen"}]
        _, altered_estimates = fit_simple_baselines(train, altered)
        self.assertEqual(altered_estimates, [expected["global_training_mean_gains_s"]])

    def test_bootstrap_is_paired_at_task_level(self):
        report = bootstrap_task_difference({"a": .5, "b": .5}, repetitions=100)
        self.assertEqual((report["mean_s"], report["lower_s"], report["upper_s"]), (.5, .5, .5))
        self.assertEqual(report["tasks"], 2)
        with self.assertRaises(ValueError):
            bootstrap_task_difference({})


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fixture(self, root):
        dataset, encoder = root / "data", root / "encoder"
        batch = dataset / "batch_000"
        batch.mkdir(parents=True)
        rows = [make_row(str(i), str(i // 2), (float(i - 2), float(3 - i))) for i in range(8)]
        for i, row in enumerate(rows):
            row["split"] = "train" if i < 4 else "validation"
        def dump(path, value):
            path.write_text(json.dumps(value))
        assignment = {"tasks": [{"group_id": "retail:" + str(i), "split": "train" if i < 2 else "validation"}
                                 for i in range(4)] + [{"group_id": "retail:reserved", "split": "test"}]}
        dump(dataset / "task_split.json", assignment)
        raw = "".join(json.dumps(r) + "\n" for r in rows).encode()
        (batch / "training_rows.jsonl").write_bytes(raw)
        (dataset / "training_rows.jsonl").write_bytes(raw)
        dump(dataset / "manifest.json", {"batches": [assignment["tasks"][:4]],
             "task_split_sha256": hashlib.sha256((dataset / "task_split.json").read_bytes()).hexdigest()})
        dump(dataset / "training_data_gate.json", {"passed": True, "checks": {"synthetic_fixture": True},
             "usable_prefixes": 8, "recorded_prefixes": 8, "usable_groups": {"train": 2, "validation": 2}})
        dump(batch / "collection_report.json", {"collection_complete": True,
             "training_rows_sha256": hashlib.sha256(raw).hexdigest(), "per_prefix_audits": [
                 {"label_usable": True, "split": r["split"], "nominal_gain_intervals_s": {
                     "now_minus_wait": {"lower": -1., "upper": 1.}}} for r in rows]})
        # Match the on-disk BERT encoder interface using a tiny random fixture.
        from transformers import BertConfig, BertModel, BertTokenizer
        encoder.mkdir()
        (encoder / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nplease\nlook\nup\nmy\naccount\n")
        tokenizer = BertTokenizer(vocab_file=str(encoder / "vocab.txt"))
        tokenizer.save_pretrained(encoder)
        model = BertModel(BertConfig(vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32,
                                     num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=64))
        model.save_pretrained(encoder)
        return dataset, encoder, rows

    def test_completed_dataset_guard_detects_tampering_without_replaying(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset, encoder, rows = self.fixture(Path(tmp))
            train, val, report, _ = preflight(dataset, encoder, 64)
            self.assertEqual((len(train), len(val)), (4, 4))
            self.assertEqual(report["reserved_test_tasks"], 1)
            raw = (dataset / "training_rows.jsonl").read_bytes()
            (dataset / "training_rows.jsonl").write_bytes(raw + b"\n")
            with self.assertRaisesRegex(ValueError, "Merged"):
                preflight(dataset, encoder, 64)
            (dataset / "training_rows.jsonl").write_bytes(raw)
            (dataset / "task_split.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "split changed"):
                preflight(dataset, encoder, 64)

    def test_two_variants_train_and_summarize_with_frozen_task_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, encoder, rows = self.fixture(root)
            args = argparse.Namespace(dataset=dataset, encoder=encoder, output=root / "output", check_only=False,
                                      device="cpu", seeds=[7], epochs=2, batch_size=2, cpu_threads=2, max_length=64)
            with contextlib.redirect_stdout(io.StringIO()):
                summary = run(args)
            self.assertEqual(set(summary["variants"]), {"full", "masked_reasoning"})
            self.assertFalse(summary["test_evaluated"])
            self.assertFalse(summary["online_performance_evaluated"])
            self.assertEqual(len(summary["runs"]), 2)
            for fit in summary["runs"]:
                path = args.output / fit["directory"]
                report = json.loads((path / "training_report.json").read_text())
                metrics = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]
                selected = max(metrics, key=lambda m: (m["task_mean_recorded_gain_s"], -m["mse_seconds_squared"]))
                self.assertEqual(fit["selected_epoch"], selected["epoch"])
                model, tokenizer = TimingController.load(args.output / fit["checkpoint"])
                validation = [r for r in rows if r["split"] == "validation"]
                saved = [json.loads(line) for line in (path / report["selected_predictions"]).read_text().splitlines()]
                actual = model.predict_inputs([r["input"] for r in validation], tokenizer)
                for a, b in zip(actual, saved):
                    self.assertAlmostEqual(a["q_now_s"], b["q_now_s"], places=5)
                    self.assertAlmostEqual(a["q_wait_s"], b["q_wait_s"], places=5)
                    self.assertEqual(a["policy"], b["policy"])
            self.assertTrue((args.output / "summary.md").is_file())


if __name__ == "__main__":
    unittest.main()
