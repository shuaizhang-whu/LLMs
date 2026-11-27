"""CPU-only runner bookkeeping: frozen plans, failures, resume and interruption."""
import asyncio
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import run_successor_full_eval as runner
from spork_core.prefix_replay import COUNTERS
from spork_core.evaluation_artifacts import planned_records


class Tokenizer:
    def encode(self, text, **kwargs):
        return [1, 2]

    def apply_chat_template(self, messages, **kwargs):
        return "cpu test prompt"


class HTTP:
    requests = 0
    def __init__(self, *args):
        pass

    async def status(self, session):
        return {**{k: 0 for k in COUNTERS}, "active_requests": 0,
                "pending_verifications": 0, "server_config": {"batch_invariant_requested": True,
                                                              "prefix_cache_reset_available": True}}

    async def complete(self, *args, **kwargs):
        HTTP.requests += 1
        return {"token_ids": [1], "finish_reason": "length"}


class Tests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.args = SimpleNamespace(output=Path(self.tmp.name) / "run", scope="train",
            task_ids="airline:3,retail:1", configs="spork,raw", max_turns=4, repetitions=1,
            seed=42, max_tokens=8192, tool_floor=2., model="/mnt/data/zhangshuai/LLMs/Models/Qwen3-32B",
            base_url="http://127.0.0.1:8017/v1",
            task_split=runner.ROOT / "prefix_data/datasets/controller_stage3_20261008_213013_wi87e7du/task_split.json")
        self.manifest, self.tasks = runner.prepare(self.args)
        self.started = 0

    def transport(self, *, fail_second=False, interrupt=False):
        async def task(*args, on_turn, trace, **kwargs):
            self.started += 1
            trace.update(adopted=0, errors=[], turn_limit_reached=False, pairs=[])
            for turn in range(1, 5):
                on_turn({"turn": turn, "completion_valid": True,
                    "baseline_has_tool_call": turn < 4, "baseline_tool_name": "cpu_mock" if turn < 4 else None,
                    "baseline_tool_args": {}, "tool_result_sha256": "cpu",
                    "environment_after": {"agent_db": "cpu", "user_db": None},
                    "main_token_ids": [turn], "baseline_text": "cpu", "d3_status": {"clear_status": "ok"}})
                if turn == 1 and interrupt:
                    raise asyncio.CancelledError()
                if turn == 1 and fail_second and self.started == 2:
                    raise RuntimeError("synthetic transport failure")
        stack = ExitStack()
        stack.enter_context(patch.object(runner, "PrefixHttpClient", HTTP))
        stack.enter_context(patch.object(runner, "run_successor_task", task))
        stack.enter_context(patch.object(runner, "tau2_action_reward", return_value={"reward": 0.}))
        async def reset(*args):
            return {"success": True, "attempts": 1, "wall_s": 0.}
        stack.enter_context(patch.object(runner, "reset_trial_cache", reset))
        stack.enter_context(patch("transformers.AutoTokenizer.from_pretrained", return_value=Tokenizer()))
        return stack

    async def test_completed_failures_are_sealed_and_not_reexecuted(self):
        with self.transport(fail_second=True):
            await runner.execute(self.args, self.manifest, self.tasks)
        records, unsealed = planned_records(self.args.output, self.manifest)
        self.assertEqual(len(records), 4)
        self.assertEqual(sum(bool(r["errors"]) for r in records), 1)
        self.assertEqual(unsealed, [])
        argv = ["full_eval", "--output", str(self.args.output), "--scope", "train",
                "--task-ids", self.args.task_ids, "--max-turns", "4", "--run"]
        with patch.object(sys, "argv", argv), patch.object(runner, "execute", side_effect=AssertionError("must not execute")):
            runner.main()
        self.assertEqual(self.started, 4)

    async def test_recovery_is_enabled_only_for_the_explicit_recovery_arm(self):
        self.args.output = Path(self.tmp.name) / "recovery"
        self.args.configs = "spork,raw_fallback"
        manifest, tasks = runner.prepare(self.args)
        with self.transport():
            with patch.object(runner, "run_successor_task", wraps=runner.run_successor_task) as calls:
                await runner.execute(self.args, manifest, tasks)
        settings = [(c.kwargs["mode"], c.kwargs["recover_child_failure"])
                    for c in calls.call_args_list]
        self.assertEqual(settings.count(("baseline", False)), 2)
        self.assertEqual(settings.count(("raw", True)), 2)

    async def test_nospec_arm_runs_plain_turns_without_draft_cleanup(self):
        self.args.output = Path(self.tmp.name) / "nospec"
        self.args.configs = "nospec,spork"
        manifest, tasks = runner.prepare(self.args)
        seen = []
        async def task(session, client, adapter, messages, executor, make_cfg, *, mode, on_turn, trace, **kwargs):
            cfg = make_cfg(messages, 1)
            seen.append((mode, cfg.launch_mode, cfg.max_retries, cfg.d3_set is None, cfg.d3_clear is None))
            trace.update(adopted=0, errors=[], turn_limit_reached=False, pairs=[])
            plain = cfg.launch_mode == "off"
            on_turn({"turn": 1, "completion_valid": True, "baseline_has_tool_call": False,
                "main_token_ids": [1], "baseline_text": "cpu", "num_probes": 0,
                "d3_status": None if plain else {"clear_status": "ok"}})
        with self.transport(), patch.object(runner, "run_successor_task", task):
            await runner.execute(self.args, manifest, tasks)
        self.assertEqual(seen.count(("baseline", "off", 0, True, True)), 2)
        self.assertEqual(seen.count(("baseline", "original", 5, False, False)), 2)
        records, _ = planned_records(self.args.output, manifest)
        self.assertEqual([r["errors"] for r in records], [[]] * 4)

    async def test_standard_plan_refuses_a_batch_invariant_server(self):
        self.args.output = Path(self.tmp.name) / "standard"
        self.args.serving_mode = "standard"
        manifest, tasks = runner.prepare(self.args)
        self.assertEqual((manifest["serving_mode"], manifest["batch_invariant_expected"]), ("standard", False))
        with self.transport(), self.assertRaisesRegex(RuntimeError, "frozen batch mode"):
            await runner.execute(self.args, manifest, tasks)
        self.args.require_identical_trajectories = True
        self.args.output = Path(self.tmp.name) / "standard_strict"
        with self.assertRaisesRegex(ValueError, "standard kernels"):
            runner.prepare(self.args)

    async def test_interruption_is_preserved_and_is_not_a_completed_trial(self):
        with self.transport(interrupt=True):
            with self.assertRaises(asyncio.CancelledError):
                await runner.execute(self.args, self.manifest, self.tasks)
        records, _ = planned_records(self.args.output, self.manifest)
        self.assertEqual(records, [])
        attempts = list((self.args.output / "attempts").glob("*.json"))
        self.assertEqual(len(attempts), 1)
        first = attempts[0].read_bytes()
        partial = json.loads(first)
        self.assertEqual(partial["stage"], "interrupted")
        self.assertEqual(len(partial["turns"]), 1)
        with self.transport():
            await runner.execute(self.args, self.manifest, self.tasks)
        self.assertEqual(attempts[0].read_bytes(), first)
        self.assertEqual(len(planned_records(self.args.output, self.manifest)[0]), 4)

    def test_changed_budget_and_test_task_outside_train_are_rejected(self):
        self.args.max_turns = 5
        with self.assertRaisesRegex(ValueError, "Frozen"):
            runner.prepare(self.args)
        self.args.max_turns = 4
        split = json.loads(self.args.task_split.read_text())
        self.args.task_ids = next(t["group_id"] for t in split["tasks"] if t["split"] == "test")
        with self.assertRaisesRegex(ValueError, "selected split"):
            runner.prepare(self.args)

    def test_failed_later_turn_does_not_hide_completed_prefix_divergence(self):
        row = {"group_id": "airline:2", "config": "raw", "valid": False,
               "shared_completed_turns": 1, "shared_completed_tokens_equal": False}
        with self.assertRaisesRegex(RuntimeError, "Trajectory divergence"):
            runner.require_matching_trajectories({"comparisons": [row]},
                                                 {"require_identical_trajectories": True})
        row["shared_completed_tokens_equal"] = True
        runner.require_matching_trajectories({"comparisons": [row]},
                                             {"require_identical_trajectories": True})

    def test_strict_gate_rejects_discordant_failures_and_unverified_failed_output(self):
        manifest = {"require_identical_trajectories": True}
        for extra in [{"same_failure_status": False},
                      {"both_failed": True, "same_failed_generation": False},
                      {"both_failed": True, "same_failed_generation": None}]:
            row = {"group_id": "airline:4", "config": "raw", "valid": False, **extra}
            with self.subTest(extra=extra), self.assertRaisesRegex(RuntimeError, "Trajectory divergence"):
                runner.require_matching_trajectories({"comparisons": [row]}, manifest)
        row = {"group_id": "airline:4", "config": "raw", "valid": False,
               "same_failure_status": True, "both_failed": True, "same_failed_generation": True}
        runner.require_matching_trajectories({"comparisons": [row]}, manifest)


if __name__ == "__main__":
    unittest.main()
