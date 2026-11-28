"""CPU checks of causal full-probe drafting and unchanged ETA guard semantics."""
import asyncio
import ast
import json
from pathlib import Path
import time
import unittest

from test_launch_timing import CharTokenizer, CALL, Client
from test_early_exit import CheckedAdapter, ChangingHTTP, Executor
from spork_core.candidate_verification import forced_probe_with_draft
from spork_core.early_exit import ExitConfig, run_early_exit_turn
from spork_core.turn_runner import RetryTurnConfig


class DraftHTTP:
    def __init__(self, delay=.05, responses=None):
        self.delay = delay
        self.responses = list(responses or [{"status": "ok"}])
        self.started = asyncio.Event()
        self.calls, self.registrations = [], []
        self.cancelled = False

    async def complete(self, session, ids, **kwargs):
        self.calls.append((list(ids), kwargs))
        self.started.set()
        try:
            await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return {"text": "target correction", "logprobs": {"target": True}, "wall_s": self.delay}

    async def set_draft(self, session, request_id, anchor, candidate, tokenizer, **kwargs):
        self.registrations.append((request_id, anchor, candidate, kwargs))
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]


class ReuseHTTP(ChangingHTTP):
    def __init__(self):
        super().__init__()
        self.registrations = []

    async def complete(self, session, ids, **kwargs):
        if bytes(ids).decode().endswith('{"name": "'):
            await asyncio.sleep(.04)
        return await super().complete(session, ids, **kwargs)

    async def set_draft(self, session, request_id, anchor, candidate, tokenizer, **kwargs):
        self.registrations.append(candidate)
        return {"status": "ok", "reuse_mode": kwargs["reuse_mode"]}

    async def stream(self, session, prompt, state, **kwargs):
        full = "<think>\n" + "thinking.\n" * 240 + "</think>\n<tool_call>" + json.dumps(CALL)
        try:
            for end in range(10, len(full) + 10, 10):
                state.text = full[:end]
                state.token_ids = list(state.text.encode())
                state.changed.set()
                await asyncio.sleep(.002)
            state.finish_reason = "stop"
        finally:
            state.finished_at = time.perf_counter()
            state.changed.set()


class ForcedReuseTests(unittest.IsolatedAsyncioTestCase):
    def request(self):
        return CharTokenizer().encode('history <tool_call>old</tool_call>\n</think>\n\n<tool_call>\n{"name": "')

    async def probe(self, http, candidate=CALL, request=None):
        return await forced_probe_with_draft(None, http, CharTokenizer(), self.request() if request is None else request,
                                             candidate=candidate, max_tokens=160, seed=42,
                                             stop=("</tool_call>",), logprobs=5, reuse_mode="prefix")

    async def test_unchanged_request_and_target_correction_with_old_candidate(self):
        http = DraftHTTP()
        choice, metadata = await self.probe(http)
        self.assertEqual(choice["text"], "target correction")
        self.assertEqual(choice["logprobs"], {"target": True})
        ids, options = http.calls[0]
        self.assertEqual(ids, self.request())
        self.assertEqual({k: v for k, v in options.items() if k != "request_id"},
                         dict(max_tokens=160, seed=42, stop=("</tool_call>",), logprobs=5))
        rid, anchor, candidate, kwargs = http.registrations[0]
        self.assertEqual(rid, options["request_id"])
        self.assertEqual(bytes(ids[anchor:]).decode(), '<tool_call>\n{"name": "')
        self.assertEqual(candidate, CALL)
        self.assertTrue(metadata["d3_registered"])
        self.assertEqual(kwargs, {"reuse_mode": "prefix"})

    async def test_no_prior_candidate_and_fast_completion_do_not_register(self):
        for candidate in (None, CALL):
            http = DraftHTTP(delay=0)
            choice, metadata = await self.probe(http, candidate=candidate)
            self.assertEqual(choice["text"], "target correction")
            self.assertFalse(metadata["d3_registered"])
            self.assertEqual(http.registrations, [])

    async def test_malformed_anchor_fails_before_a_model_request(self):
        http = DraftHTTP()
        with self.assertRaises(ValueError):
            await self.probe(http, request=CharTokenizer().encode("old <tool_call>not a forced opener"))
        self.assertEqual(http.calls, [])

    async def test_not_active_registration_retries_without_reissuing_completion(self):
        http = DraftHTTP(delay=.07, responses=[{"status": "error", "error": "main_request_not_active_or_ambiguous", "matches": 0}, {"status": "ok"}])
        _, metadata = await self.probe(http)
        self.assertEqual(len(http.calls), 1)
        self.assertEqual(len(http.registrations), 2)
        self.assertTrue(metadata["d3_registered"])

    async def test_ambiguous_registration_cancels_own_request(self):
        http = DraftHTTP(delay=10, responses=[{"status": "error", "error": "main_request_not_active_or_ambiguous", "matches": 2}])
        with self.assertRaises(RuntimeError):
            await self.probe(http)
        self.assertTrue(http.cancelled)

    async def test_external_cancellation_drains_completion(self):
        http = DraftHTTP(delay=10)
        task = asyncio.create_task(self.probe(http))
        await http.started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(http.cancelled)

    async def test_paper_checks_reuse_previous_not_current_prediction(self):
        http = ReuseHTTP()
        clears = []

        async def clear():
            clears.append(True)
            return {"clear_status": "cleared"}

        cfg = RetryTurnConfig(enable_thinking=True, baseline_max_tokens=8192, context_window_tokens=40960,
                              d3_clear=clear)
        exit_cfg = ExitConfig(method="eta", stable_checks=3, check_interval=10, max_checks=15,
                              guard_policy="paper", retarget_after_stable=True,
                              reuse_forced_probe=True, probe_draft_reuse_mode="prefix")
        result = await run_early_exit_turn(None, Client(), CheckedAdapter(), [{"role": "user", "content": "lookup a"}],
                                           Executor(), 1, cfg, exit_cfg, http=http)
        predictions = result["probes"] + [c["prediction"] for c in result["checks"] if "prediction" in c]
        self.assertIsNone(predictions[0]["draft_reuse"]["candidate"])
        for before, after in zip(predictions, predictions[1:]):
            self.assertEqual(after["draft_reuse"]["candidate"], before["call"])
        self.assertEqual(len(result["checks"]), len(predictions))
        self.assertTrue(all("free" in c for c in result["checks"]))
        self.assertEqual(len(http.registrations), len(predictions) - 1)
        self.assertTrue(result["exit"]["fired"])
        self.assertEqual(result["exit"]["retarget_count"], 1)
        self.assertEqual(result["baseline_tool_args"], CALL["arguments"])
        self.assertEqual(clears, [True])

    async def test_config_supports_paper_rules_without_decision_gating(self):
        source = Path(__file__).resolve().parents[1] / "scripts/run_interactive_smoke.py"
        fn = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "exit_config")
        namespace = {"ExitConfig": ExitConfig}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), namespace)
        config = namespace["exit_config"]("eta_retarget_forced_reuse", "paper")
        self.assertEqual(config.method, "eta")
        self.assertEqual(config.max_checks, 15)
        self.assertTrue(config.reuse_forced_probe)
        self.assertFalse(namespace["exit_config"]("eta_retarget", "paper").reuse_forced_probe)
        with self.assertRaises(ValueError):
            ExitConfig(method="reuse", reuse_forced_probe=True).validate()


if __name__ == "__main__":
    unittest.main()
