"""CPU-only tests for diagnostic interventions, comparison and persistence."""
import argparse
import asyncio
from collections import Counter
import copy
from dataclasses import asdict
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

from spork_core.fork_diagnostic import MODES, check_branch, compare_outputs, schedule_modes, summarize
from spork_core.prefix_replay import COUNTERS, ReplayConfig
from scripts.repeat_prefix_pairs import digest, dump
from test_prefix_replay import Adapter, FakeClient, FakeResponse, TinyEnv, Tokenizer


class DecodeTokenizer(Tokenizer):
    def decode(self, ids, **kwargs):
        return "".join(chr(i) for i in ids)

    def convert_tokens_to_ids(self, value):
        return {"</think>": 1000, "<tool_call>": 1001}[value]


SNAPSHOT = {"prefix_text": "<think> fixed", "prefix_token_ids": [9, 8],
            "environment_hash": {"agent_db": "before"}}


def outcome(mode, changed=False):
    setting = MODES[mode]
    ids = [65, 67 if changed else 66, 1000, 1001, 68]
    out = {"policy": setting["policy"], "config": asdict(ReplayConfig()),
           "interventions": {k: setting[k] for k in ("enable_d3", "enable_speculative_tools")},
           "completion_valid": True, "finish_reason": "stop", "main_request_sha256": "same",
           "wall_s": 2., "main_finished_s": 1., "main_text": SNAPSHOT["prefix_text"] + " suffix",
           "continuation_text": " suffix", "continuation_token_ids": ids,
           "environment_before": SNAPSHOT["environment_hash"], "environment_after": {"agent_db": "after"},
           "formal_call": {"name": "lookup", "arguments": {"id": "a"}}, "tool_result": "result",
           "probes": [], "clear_after": {"active_requests": 0, "pending_verifications": 0},
           "d3_counters": {k: 0 for k in COUNTERS}, "spec_used": False, "spec_wasted": False,
           "token_events": [{"end_token": i + 1, "received_s": .1 * (i + 1)} for i in range(len(ids))]}
    if mode != "main_only":
        out["probes"] = [{"dispatch_s": .001, "end_s": .3, "new_tokens_at_dispatch": 0,
                          "visible_prefix_text": SNAPSHOT["prefix_text"], "dispatched": False}]
    if mode == "spork_full":
        out["d3_counters"]["verified_injections"] = 1
    return out


class ComparisonTests(unittest.TestCase):
    def test_three_repeat_order_balances_every_position(self):
        schedule = schedule_modes(["prefix_003", "prefix_009"])
        self.assertEqual(len(schedule), 6)
        for sid in ("prefix_003", "prefix_009"):
            orders = [x["order"] for x in schedule if x["sample_id"] == sid]
            for pos in range(3):
                self.assertEqual(Counter(x[pos] for x in orders), {m: 1 for m in MODES})
        self.assertEqual(schedule, schedule_modes(["prefix_003", "prefix_009"]))

    def test_first_divergence_is_before_reasoning_end(self):
        pair = compare_outputs(outcome("main_only"), outcome("prediction_only", True), DecodeTokenizer(), 32)
        self.assertEqual(pair["first_difference_token_0based"], 1)
        self.assertEqual(pair["first_difference_including_saved_prefix_0based"], 33)
        self.assertTrue(pair["before_both_reasoning_ends"])
        self.assertTrue(pair["same_formal_action"])
        self.assertEqual(pair["right"]["divergence_chunk_received_s"], .2)

    def test_equal_and_length_only_cases_are_distinguished(self):
        left = outcome("main_only")
        equal = compare_outputs(left, copy.deepcopy(left), DecodeTokenizer(), 32)
        self.assertTrue(equal["same_tokens"])
        self.assertIsNone(equal["first_difference_token_0based"])
        short = copy.deepcopy(left)
        short["continuation_token_ids"] = short["continuation_token_ids"][:2]
        diff = compare_outputs(left, short, DecodeTokenizer(), 32)
        self.assertEqual(diff["first_difference_token_0based"], 2)
        self.assertIsNone(diff["before_both_reasoning_ends"])

    def test_disabled_effects_are_checked_even_when_flags_claim_disabled(self):
        value = outcome("prediction_only")
        self.assertEqual(check_branch("prediction_only", value, SNAPSHOT, asdict(ReplayConfig())), [])
        value["probes"][0]["d3_set"] = {"status": "ok"}
        value["probes"][0]["dispatched"] = True
        issues = check_branch("prediction_only", value, SNAPSHOT, asdict(ReplayConfig()))
        self.assertIn("d3_was_not_disabled", issues)
        self.assertIn("speculative_tools_were_not_disabled", issues)

    def test_summary_does_not_claim_specific_numerical_cause(self):
        records = [{"repetition": rep, "outcomes": {mode: outcome(mode, mode != "main_only") for mode in MODES}}
                   for rep in range(3)]
        manifest = {"sample_ids": ["prefix_003"], "repetitions": 3, "replay_config": asdict(ReplayConfig())}
        report = summarize({"prefix_003": records}, {"prefix_003": SNAPSHOT}, manifest, DecodeTokenizer())
        self.assertTrue(report["pipeline_passed"], report["errors"])
        self.assertFalse(report["generation_invariance_passed"])
        self.assertEqual(report["per_prefix"][0]["observation"],
                         "divergence_observed_without_action_draft_registration_or_early_tool_execution")
        records[1]["outcomes"]["main_only"] = outcome("main_only", True)
        report = summarize({"prefix_003": records}, {"prefix_003": SNAPSHOT}, manifest, DecodeTokenizer())
        self.assertFalse(report["per_prefix"][0]["reference_tokens_stable"])
        self.assertIn("reference_itself_varies", report["per_prefix"][0]["observation"])

    def test_invariance_requires_tokens_and_result_not_just_actions(self):
        records = [{"repetition": rep, "outcomes": {mode: outcome(mode) for mode in MODES}}
                   for rep in range(3)]
        manifest = {"sample_ids": ["prefix_003"], "repetitions": 3, "replay_config": asdict(ReplayConfig())}
        def report():
            return summarize({"prefix_003": records}, {"prefix_003": SNAPSHOT}, manifest, DecodeTokenizer())
        self.assertTrue(report()["generation_invariance_passed"])
        records[0]["outcomes"]["spork_full"]["tool_result"] = "changed result"
        self.assertTrue(report()["pipeline_passed"])
        self.assertFalse(report()["generation_invariance_passed"])
        records[0]["outcomes"]["spork_full"]["tool_result"] = "result"
        records[0]["outcomes"]["prediction_only"]["continuation_token_ids"][1] = 67
        self.assertFalse(report()["generation_invariance_passed"])

    def test_invariance_requires_all_repetitions(self):
        records = [{"repetition": rep, "outcomes": {mode: outcome(mode) for mode in MODES}}
                   for rep in range(2)]
        manifest = {"sample_ids": ["prefix_003"], "repetitions": 3, "replay_config": asdict(ReplayConfig())}
        report = summarize({"prefix_003": records}, {"prefix_003": SNAPSHOT}, manifest, DecodeTokenizer())
        self.assertFalse(report["generation_invariance_passed"])

    def test_invariance_preflight_rejects_old_server_before_generation(self):
        from scripts.diagnose_prefix_forks import check_server_settings
        for status in ({}, {"server_config": {"batch_invariant_requested": False}}):
            with self.assertRaisesRegex(RuntimeError, "no generation was started"):
                check_server_settings(status, {"require_invariance": True})
        check_server_settings({"server_config": {"batch_invariant_requested": True}},
                              {"require_invariance": True})
        check_server_settings({}, {"require_invariance": False})

    def test_only_clean_truncation_can_continue(self):
        from scripts.diagnose_prefix_forks import inspect_outcome
        out = outcome("prediction_only")
        out.update(finish_reason="length", completion_valid=False)
        manifest = {"replay_config": asdict(ReplayConfig()), "continue_on_truncation": True}
        integrity, fatal = inspect_outcome("prediction_only", out, SNAPSHOT, manifest)
        self.assertFalse(fatal)
        self.assertFalse(integrity["passed"])
        self.assertFalse(integrity["use_for_completion_time_label"])
        self.assertTrue(integrity["continued_after_truncation"])
        out["d3_counters"]["accounting_errors"] = 1
        self.assertTrue(inspect_outcome("prediction_only", out, SNAPSHOT, manifest)[1])
        out["d3_counters"]["accounting_errors"] = 0
        out["error"] = "TimeoutError"
        self.assertTrue(inspect_outcome("prediction_only", out, SNAPSHOT, manifest)[1])
        del out["error"]
        manifest["continue_on_truncation"] = False
        self.assertTrue(inspect_outcome("prediction_only", out, SNAPSHOT, manifest)[1])

    def test_identical_truncations_are_collected_but_never_pass_quality(self):
        records = [{"repetition": rep, "outcomes": {mode: outcome(mode) for mode in MODES}}
                   for rep in range(3)]
        for record in records:
            for out in record["outcomes"].values():
                out.update(finish_reason="length", completion_valid=False)
        manifest = {"sample_ids": ["prefix_003"], "repetitions": 3, "replay_config": asdict(ReplayConfig())}
        report = summarize({"prefix_003": records}, {"prefix_003": SNAPSHOT}, manifest, DecodeTokenizer())
        self.assertTrue(report["collection_complete"])
        self.assertEqual(report["truncated_branch_runs"], 9)
        self.assertFalse(report["pipeline_passed"])
        self.assertFalse(report["generation_invariance_passed"])
        self.assertTrue(all(not r["use_for_completion_time_label"] for r in report["truncated_branches"]))


class DiagnosticIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_eighteen_branches_are_isolated_and_resume_does_not_rerun(self):
        await self.exercise_collection(False)

    async def test_partial_resume_reuses_truncated_branch_and_finishes_others(self):
        await self.exercise_collection(True)

    async def exercise_collection(self, truncate_and_interrupt):
        import scripts.diagnose_prefix_forks as script
        from spork_core.prefix_replay import run_prefix_branch as original_replay

        branch_calls = 0
        async def replay(*args, **kwargs):
            nonlocal branch_calls
            branch_calls += 1
            if truncate_and_interrupt and branch_calls == 5:
                raise RuntimeError("simulated interruption after a saved truncated branch")
            out = await original_replay(*args, **kwargs)
            if truncate_and_interrupt and branch_calls == 4:
                out.update(finish_reason="length", completion_valid=False)
            return out

        class Client(FakeClient):
            instances = []

            def __init__(self, base_url, model, timeout_s):
                super().__init__()
                self.base_url = base_url
                self.model = model
                self.main_requests = 0
                self.instances.append(self)

            async def stream(self, *args, **kwargs):
                self.main_requests += 1
                await super().stream(*args, **kwargs)

        class Session:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def get(self, *args, **kwargs):
                return FakeResponse(json_value={"data": [{"id": "test-model"}]})

        tokenizer = DecodeTokenizer()
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            checks = {}
            for sid in ("prefix_003", "prefix_009"):
                dest = source / "inputs" / sid
                dest.mkdir(parents=True)
                env = TinyEnv()
                (dest / "environment.pkl").write_bytes(pickle.dumps(env))
                messages = [{"role": "user", "content": "look up an order"}]
                prompt = json.dumps(messages)
                inp = {"sample_id": sid, "messages": messages, "tools": [], "prompt_text": prompt,
                       "prompt_token_ids": tokenizer.encode(prompt), "prefix_text": "<think> fixed",
                       "prefix_token_ids": tokenizer.encode("<think> fixed"),
                       "environment_hash": {"agent_db": env.get_db_hash(), "user_db": env.get_user_db_hash()}}
                dump(dest / "input.json", inp)
                checks[sid] = {"input_sha256": digest(dest / "input.json"), "environment_sha256": digest(dest / "environment.pkl")}
            dump(source / "manifest.json", {"input_checksums": checks, "model": "test-model", "model_url": "http://test/v1",
                                            "timeout_s": 10, "tool_floor_s": 0., "replay_config": asdict(ReplayConfig())})
            args = argparse.Namespace(output=Path(tmp) / "out", source_run=source, resume=False, analyze_only=False,
                                      model_url=None, repetitions=3, order_seed=43,
                                      continue_on_truncation=truncate_and_interrupt)
            manifest = script.prepare(args)
            with patch("spork_core.prefix_replay.PrefixHttpClient", Client), \
                 patch("spork_core.prefix_replay.run_prefix_branch", replay), \
                 patch("spork_core.qwen.QwenToolCallAdapter", Adapter), \
                 patch("aiohttp.ClientSession", Session), patch("aiohttp.TCPConnector", return_value=None), \
                 patch.object(script, "log", lambda *a: None):
                if truncate_and_interrupt:
                    with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                        await script.collect(args.output, manifest, tokenizer)
                    self.assertEqual(Client.instances[-1].main_requests, 4)
                    saved = list((args.output / "measurements").rglob("*_saved.json"))
                    self.assertEqual(len(saved), 4)
                    before = {p: digest(p) for p in saved}
                    await script.collect(args.output, manifest, tokenizer)
                    self.assertEqual(Client.instances[-1].main_requests, 14)
                    self.assertEqual(before, {p: digest(p) for p in saved})
                else:
                    await script.collect(args.output, manifest, tokenizer)
                    self.assertEqual(Client.instances[-1].main_requests, 18)
                await script.collect(args.output, manifest, tokenizer)
                self.assertEqual(Client.instances[-1].main_requests, 0)
            report = script.analyze(args.output, manifest, tokenizer)
            self.assertEqual(report["pipeline_passed"], not truncate_and_interrupt, report["errors"])
            self.assertTrue(report["collection_complete"])
            self.assertEqual(report["truncated_branch_runs"], int(truncate_and_interrupt))
            if truncate_and_interrupt:
                self.assertFalse(report["generation_invariance_passed"])
            self.assertEqual(report["completed_branch_runs"], 18)
            self.assertEqual(report["d3_counters_by_mode"]["prediction_only"]["set_count"], 0)
            self.assertEqual(report["d3_counters_by_mode"]["main_only"]["verified_injections"], 0)
            args.resume = True
            self.assertEqual(script.prepare(args)["schedule"], manifest["schedule"])


if __name__ == "__main__":
    unittest.main()
