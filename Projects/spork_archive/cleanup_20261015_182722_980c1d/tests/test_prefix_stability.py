"""Fast offline tests for repeatability gates, order balance and resume storage."""
import argparse
import asyncio
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from spork_core.prefix_stability import (
    POLICIES, QUALITY_KEYS, analyze_repetitions,
    balanced_schedule, mean_interval,
)
from scripts.repeat_prefix_pairs import completed_record, digest, dump, runtime_checks
from spork_core.prefix_replay import COUNTERS, ReplayConfig


def record(rep, now_gain=2., wait_gain=.5, match=True):
    outcomes = {}
    for p, gain in (("off", 0.), ("now", now_gain), ("wait", wait_gain)):
        outcomes[p] = {
            "policy": p, "wall_s": 10. - gain, "completion_valid": True,
            "clear_after": {"active_requests": 0, "pending_verifications": 0},
            "d3_counters": {**{k: 0 for k in COUNTERS}, "verified_injections": int(p == "now")},
            "continuation_token_ids": [1, 2, 3],
        }
    labels = {"valid": True, "gain_now_s": now_gain, "gain_wait_s": wait_gain,
              **{k: match for k in QUALITY_KEYS}}
    return {"repetition": rep, "labels": labels, "outcomes": outcomes}


def dataset():
    return {f"prefix_{i:03d}": [record(rep) for rep in range(6)] for i in range(10)}


class StabilityTests(unittest.TestCase):
    def analyze(self, data):
        return analyze_repetitions(data, sorted(data), 6)

    def test_all_order_permutations_and_positions_are_balanced(self):
        ids = ["prefix_0", "prefix_1"]
        schedule = balanced_schedule(ids, 6, 7)
        self.assertEqual(schedule, balanced_schedule(ids, 6, 7))
        for sid in ids:
            orders = [x["order"] for x in schedule if x["sample_id"] == sid]
            self.assertEqual(len(set(tuple(o) for o in orders)), 6)
            for position in range(3):
                self.assertEqual(Counter(o[position] for o in orders), {p: 2 for p in POLICIES})
        for rep in range(6):
            self.assertEqual({s["sample_id"] for s in schedule[2 * rep:2 * rep + 2]}, set(ids))
        with self.assertRaises(ValueError):
            balanced_schedule(ids, 5)

    def test_constant_valid_data_passes(self):
        result = self.analyze(dataset())
        self.assertTrue(result["passed"])
        self.assertEqual(result["observed_triples"], 60)
        self.assertEqual(result["precise_prefixes"], 10)
        self.assertEqual(result["clearly_best_policy_counts"], {"now": 10})
        self.assertEqual(result["per_prefix"][0]["gains_s"]["now"]["mean"], 2.)

    def test_noisy_gains_are_not_rescued_by_filtering(self):
        data = dataset()
        for sid in data:
            data[sid] = [record(rep, 7. if rep % 2 else -7.) for rep in range(6)]
        result = self.analyze(data)
        self.assertFalse(result["passed"])
        self.assertEqual(result["precise_prefixes"], 0)
        self.assertEqual(result["observed_triples"], 60)
        self.assertIn("label_precision_rate", result["failed_checks"])

    def test_outcome_disagreements_fail_even_when_timings_are_stable(self):
        data = dataset()
        for sid in list(data)[:2]:
            data[sid] = [record(rep, match=False) for rep in range(6)]
        result = self.analyze(data)
        self.assertFalse(result["passed"])
        self.assertEqual(result["outcome_match_fraction"], .8)
        self.assertEqual(result["valid_triples"], 60)
        self.assertIn("outcome_match_rate", result["failed_checks"])

    def test_small_near_ties_do_not_require_identical_argmax(self):
        data = dataset()
        for sid in data:
            data[sid] = [record(rep, .1 if rep % 2 else -.1, 0.) for rep in range(6)]
        result = self.analyze(data)
        self.assertTrue(result["passed"])
        self.assertEqual(result["clearly_best_policy_counts"], {})

    def test_missing_duplicates_and_corrupted_labels_fail(self):
        for mutation in ("missing", "duplicate", "label"):
            data = dataset()
            sid = sorted(data)[0]
            if mutation == "missing":
                data[sid].pop()
            elif mutation == "duplicate":
                data[sid][-1]["repetition"] = 0
            else:
                data[sid][0]["labels"]["gain_now_s"] = 123.
            self.assertFalse(self.analyze(data)["passed"], mutation)

    def test_d3_errors_and_nonfinite_times_fail(self):
        data = dataset()
        data["prefix_000"][0]["outcomes"]["now"]["d3_counters"]["accounting_errors"] = 1
        self.assertFalse(self.analyze(data)["passed"])
        data["prefix_000"][1]["outcomes"]["wait"]["wall_s"] = float("nan")
        self.assertFalse(self.analyze(data)["passed"])

    def test_intervals_require_repeats_and_keep_negative_means(self):
        self.assertIsNone(mean_interval([-3])["half_width"])
        constant = mean_interval([-3] * 6)
        self.assertEqual(constant["lower"], -3.)
        self.assertEqual(constant["upper"], -3.)
        spread = mean_interval([1, 2, 3, 4, 5, 6])
        self.assertLess(spread["lower"], 3.5)
        self.assertGreater(spread["upper"], 3.5)

    def test_completed_record_checksums_reject_silent_edits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertIsNone(completed_record(root))
            attempt = root / "attempt_000"
            attempt.mkdir()
            data = record(0)
            dump(attempt / "labels.json", data["labels"])
            for p in POLICIES:
                dump(attempt / f"{p}.json", data["outcomes"][p])
            names = ["labels.json"] + [p + ".json" for p in POLICIES]
            dump(root / "completed.json", {"attempt": attempt.name, "repetition": 0,
                                           "sha256": {n: digest(attempt / n) for n in names}})
            self.assertEqual(completed_record(root)["repetition"], 0)
            (attempt / "now.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "checksum"):
                completed_record(root)

    def test_runtime_checks_detect_changed_prefix_state_and_future_input(self):
        cfg = ReplayConfig()
        snapshot = {"prefix_text": "<think> hello", "environment_hash": {"agent_db": "a"}}
        outcomes = record(0)["outcomes"]
        for out in outcomes.values():
            out.update(config=asdict(cfg), environment_before={"agent_db": "a"},
                       main_text="<think> hello world", continuation_text=" world", probes=[])
        self.assertEqual(runtime_checks(snapshot, outcomes, cfg), [])
        outcomes["now"]["probes"] = [{"new_tokens_at_dispatch": 2, "visible_prefix_text": "<think> hello world"}]
        self.assertIn("now:saw_future_prefix", runtime_checks(snapshot, outcomes, cfg))
        outcomes["wait"]["environment_before"] = {"agent_db": "other"}
        self.assertIn("wait:different_initial_environment", runtime_checks(snapshot, outcomes, cfg))

    def test_task_split_stratifies_and_reserves_development_tasks(self):
        from scripts.prepare_controller_task_split import assign_splits
        tasks = [{"domain": d, "task_id": str(i)} for d in ("airline", "retail") for i in range(20)]
        development = {"airline:1", "retail:2"}
        split = assign_splits(tasks, development, 43)
        self.assertEqual(split, assign_splits(tasks, development, 43))
        self.assertEqual(len({t["group_id"] for t in split}), 40)
        for t in split:
            if t["group_id"] in development:
                self.assertEqual(t["split"], "train")
        for d in ("airline", "retail"):
            self.assertEqual(Counter(t["split"] for t in split if t["domain"] == d),
                             {"train": 14, "validation": 3, "test": 3})


class SavedEnvironment:
    def get_db_hash(self):
        return "unchanged-state"

    def get_user_db_hash(self):
        return None


class RepeatIntegrationTests(unittest.TestCase):
    def test_prepare_collect_analyze_and_resume_without_inference(self):
        import scripts.repeat_prefix_pairs as collector

        class Tokenizer:
            def encode(self, text, **kwargs):
                return list(text.encode())

            def decode(self, ids, **kwargs):
                return bytes(ids).decode()

            def apply_chat_template(self, *args, **kwargs):
                return "discarded warmup"

        class Adapter:
            def __init__(self, *args):
                pass

            def render_main_prompt(self, *args, **kwargs):
                return "past prompt"

        class Client:
            def __init__(self, *args):
                self.base_url = "http://test/v1"

            async def status(self, session):
                return {"has_drafter": True, "active_requests": 0, "pending_verifications": 0}

            async def complete(self, *args, **kwargs):
                return {"wall_s": .01}

        class Response:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def raise_for_status(self):
                pass

            async def json(self):
                return {"data": [{"id": "fake-model"}]}

        class Session(Response):
            def __init__(self, **kwargs):
                pass

            def get(self, *args, **kwargs):
                return Response()

        calls = []

        async def branch(session, client, adapter, snapshot, executor, policy, cfg):
            self.assertEqual(snapshot["prefix_text"], "<think> saved prefix")
            self.assertEqual(executor.env.get_db_hash(), "unchanged-state")
            calls.append(policy)
            out = record(0)["outcomes"][policy]
            out.update(config=asdict(cfg), main_text=snapshot["prefix_text"] + " suffix",
                       continuation_text=" suffix", probes=[], formal_call=None, tool_result=None)
            return out

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            sample = source / "prefix_000"
            sample.mkdir(parents=True)
            env = pickle.dumps(SavedEnvironment())
            (sample / "environment.pkl").write_bytes(env)
            inp = {"sample_id": "prefix_000", "prefix_text": "<think> saved prefix", "messages": [], "tools": [],
                   "prompt_text": "past prompt", "prompt_token_ids": list(b"past prompt"),
                   "prefix_token_ids": list(b"<think> saved prefix"), "environment_file": "environment.pkl",
                   "environment_sha256": digest(sample / "environment.pkl"),
                   "environment_hash": {"agent_db": "unchanged-state", "user_db": None}}
            dump(sample / "input.json", inp)
            dump(source / "validation_report.json", {"passed": True})
            dump(source / "manifest.json", {"args": {"model": "fake-model", "model_url": "http://test/v1",
                                                       "tool_floor_s": 2., "timeout_s": 10},
                                            "replay_config": asdict(ReplayConfig())})
            for p in POLICIES:
                dump(sample / f"{p}.json", record(0)["outcomes"][p])
            base = {"id": "old/prefix_000", "group_id": "retail:1", "split": "pilot", "input": {"prefix_text": inp["prefix_text"]},
                    "targets": {"gain_now_s": 999., "gain_wait_s": 999.}, "label_valid": True,
                    "metadata": {"raw_sample": "prefix_000", "domain": "retail", "task_id": "1"}, "quality_flags": {}}
            (source / "training_rows.jsonl").write_text(json.dumps(base) + "\n")
            args = argparse.Namespace(output=root / "repeat", source_run=source, repetitions=6, order_seed=42,
                                      model_url=None, resume=False, analyze_only=False)
            manifest = collector.prepare(args)
            self.assertEqual(manifest["planned_branch_runs"], 18)
            original_hash = digest(source / "training_rows.jsonl")
            with patch("transformers.AutoTokenizer.from_pretrained", return_value=Tokenizer()), \
                 patch("spork_core.qwen.QwenToolCallAdapter", Adapter), \
                 patch("spork_core.prefix_replay.PrefixHttpClient", Client), \
                 patch("spork_core.prefix_replay.run_prefix_branch", branch), \
                 patch("spork_core.executors.Tau2ToolExecutor", lambda env, **kw: SimpleNamespace(env=env)), \
                 patch("aiohttp.ClientSession", Session), patch("aiohttp.TCPConnector", return_value=None), \
                 patch.object(collector, "log", lambda *args: None):
                asyncio.run(collector.collect(args.output, manifest))
                self.assertEqual(len(calls), 18)
                asyncio.run(collector.collect(args.output, manifest))
                self.assertEqual(len(calls), 18, "Resume must not duplicate completed measurements")
            report = collector.analyze(args.output, manifest)
            self.assertEqual(report["observed_triples"], 6)
            self.assertFalse(report["passed"], "One prefix cannot pass the actual ten-prefix gate")
            averaged = json.loads((args.output / "averaged_rows.jsonl").read_text())
            self.assertEqual(averaged["targets"], {"gain_now_s": 2., "gain_wait_s": .5})
            self.assertEqual(averaged["metadata"]["repetitions"], 6)
            self.assertEqual(averaged["split"], "pilot")
            self.assertEqual(averaged["input"], base["input"])
            self.assertEqual(digest(source / "training_rows.jsonl"), original_hash)
            args.resume = True
            self.assertEqual(collector.prepare(args)["schedule"], manifest["schedule"])


if __name__ == "__main__":
    unittest.main()
