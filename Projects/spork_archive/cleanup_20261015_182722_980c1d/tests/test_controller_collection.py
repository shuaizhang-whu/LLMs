"""CPU tests: held-out task groups, label censoring and full collection resume."""
import asyncio
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from spork_core.controller_collection import aggregate_prefix, branch_errors, policy_orders, task_batches
from spork_core.prefix_replay import ReplayConfig
from scripts.repeat_prefix_pairs import dump
from test_prefix_replay import Adapter, FakeClient, TinyEnv, Tokenizer


class CollectionTokenizer(Tokenizer):
    def decode(self, ids, **kwargs):
        return bytes(ids).decode()


class CollectionClient(FakeClient):
    def __init__(self, truncate=False):
        super().__init__(finish_reason="length" if truncate else "stop")
        self.stream_count = 0

    async def complete(self, session, prompt_ids, **kwargs):
        if kwargs.get("max_tokens") == 32:
            text = "<think> I will look up the order"
            return {"text": text, "token_ids": list(text.encode()), "finish_reason": "length", "wall_s": .01}
        return await super().complete(session, prompt_ids, **kwargs)

    async def stream(self, *args, **kwargs):
        self.stream_count += 1
        await super().stream(*args, **kwargs)


def tiny_manifest(task_path):
    task = {"domain": "retail", "task_id": "1", "group_id": "retail:1", "split": "train"}
    return {"dataset_id": "unit", "model": "test-model", "tool_floor_s": 0.,
            "replay_config": asdict(ReplayConfig()), "max_turns_per_task": 2, "prefix_tokens": 32,
            "task_sources": [{"path": str(task_path), "domain": "retail"}], "batches": [[task]]}


class PlanningTests(unittest.TestCase):
    def test_tasks_never_cross_split_and_test_is_not_collected(self):
        tasks = [{"domain": d, "task_id": str(i), "group_id": f"{d}:{i}", "split": s}
                 for d in ("airline", "retail") for i, s in enumerate(["train"] * 12 + ["validation"] * 4 + ["test"] * 3)]
        batches = task_batches(tasks)
        flattened = [t for b in batches for t in b]
        self.assertEqual(len({t["group_id"] for t in flattened}), 32)
        self.assertEqual(Counter(t["split"] for t in batches[0]), {"train": 12, "validation": 4})
        self.assertFalse(any(t["split"] == "test" for t in flattened))
        self.assertEqual(batches, task_batches(tasks))

    def test_each_policy_occupies_every_position_once(self):
        orders = policy_orders("sample")
        for pos in range(3):
            self.assertEqual({order[pos] for order in orders}, {"off", "now", "wait"})

    def test_negative_labels_are_preserved_and_token_forks_reject_labels(self):
        from test_fork_diagnostic import SNAPSHOT, outcome
        inp = {**SNAPSHOT, "sample_id": "retail_1", "group_id": "retail:1", "split": "train", "domain": "retail",
               "task_id": "1", "turn": 1, "messages": [], "tools": [], "prompt_token_ids": [1], "visible_generated_tokens": 2}
        m = {"dataset_id": "test", "model": "test", "tool_floor_s": 0., "replay_config": asdict(ReplayConfig())}
        repeats = [{p: outcome("main_only") for p in ("off", "now", "wait")} for _ in range(3)]
        for values in repeats:
            for p, wall in (("off", 1.), ("now", 2.), ("wait", 3.)):
                values[p].update(policy=p, wall_s=wall)
        row, audit = aggregate_prefix(inp, repeats, m)
        self.assertTrue(row["label_valid"])
        self.assertEqual(row["targets"], {"gain_now_s": -1., "gain_wait_s": -2.})
        self.assertNotIn("targets", row["input"])
        repeats[1]["wait"]["continuation_token_ids"][0] += 1
        row, audit = aggregate_prefix(inp, repeats, m)
        self.assertFalse(row["label_valid"])
        self.assertIsNone(row["targets"]["gain_wait_s"])

    def test_quality_gate_never_accepts_a_missing_batch(self):
        from scripts.check_controller_training_data import assess
        thresholds = {"min_usable_fraction": .8, "min_precise_fraction": .8, "max_token_disagreement_prefixes": 0,
                      "first_batch_min_usable_prefixes": 2, "first_batch_min_train_groups": 1,
                      "first_batch_min_validation_groups": 1}
        audits = [{"label_usable": True, "split": split, "group_id": group,
                   "meets_initial_precision_criterion": True} for split, group in (("train", "r:1"), ("validation", "r:2"))]
        report = {"collection_complete": True, "per_prefix_audits": audits, "token_disagreement_prefixes": 0,
                  "d3_counter_totals": {"verified_injections": 1}}
        self.assertTrue(assess([report], thresholds, True)["passed"])
        report["collection_complete"] = False
        self.assertFalse(assess([report], thresholds, True)["passed"])
        report["collection_complete"] = True
        audits[1]["meets_initial_precision_criterion"] = False
        self.assertFalse(assess([report], thresholds, True)["passed"])


class CollectionTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, truncated):
        import scripts.collect_controller_dataset as script
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks = root / "tasks.json"
            dump(tasks, [{"id": "1"}])
            m = tiny_manifest(tasks)
            client = CollectionClient(truncated)
            with patch("spork_core.benchmarks.tau2.build_user_message", return_value="User"), \
                 patch("spork_core.benchmarks.tau2.load_system_prompt", return_value="System"), \
                 patch("spork_core.benchmarks.tau2.load_tools", return_value=[]), \
                 patch("spork_core.benchmarks.tau2.get_environment", side_effect=lambda _: TinyEnv()), \
                 patch("spork_core.qwen.QwenToolCallAdapter", Adapter), patch.object(script, "log", lambda *a: None):
                await script.collect_task(None, client, CollectionTokenizer(), m["batches"][0][0], m, root)
                count = client.stream_count
                await script.collect_task(None, client, CollectionTokenizer(), m["batches"][0][0], m, root)
                self.assertEqual(client.stream_count, count)
            report = script.export_batch(root, m, 0)
            self.assertTrue(report["collection_complete"])
            rows = [json.loads(line) for line in (root / "batch_000/training_rows.jsonl").read_text().splitlines()]
            if truncated:
                self.assertLessEqual(count, 3)
                self.assertEqual(len(rows), 1)
                self.assertFalse(rows[0]["label_valid"])
                self.assertIsNone(rows[0]["targets"]["gain_now_s"])
                self.assertEqual(report["task_stop_reasons"], {"reference_truncated": 1})
            else:
                self.assertEqual(count, 18)
                self.assertEqual(len(rows), 2)
                self.assertTrue(all(row["label_valid"] for row in rows))
                from spork_core.controller_data import load_controller_rows
                loaded, _ = load_controller_rows([root / "batch_000/training_rows.jsonl"])
                self.assertEqual(len(loaded), 2)
                first = script.read(root / "tasks/retail_1/turn_001/snapshot/input.json")
                second = script.read(root / "tasks/retail_1/turn_002/snapshot/input.json")
                self.assertEqual(first["environment_hash"]["agent_db"], "[]")
                self.assertNotEqual(second["environment_hash"]["agent_db"], "[]")
                self.assertEqual(second["messages"][-1]["role"], "tool")
                # Corrupt a sealed raw measurement: export must refuse it.
                raw = next((root / "tasks").rglob("result.json"))
                raw.write_text('{}')
                with self.assertRaisesRegex(ValueError, "Changed checkpoint"):
                    script.export_batch(root, m, 0)

    async def test_reference_history_three_policies_and_resume(self):
        await self.exercise(False)

    async def test_reference_truncation_is_retained_without_running_nine_branches(self):
        await self.exercise(True)


if __name__ == "__main__":
    unittest.main()
