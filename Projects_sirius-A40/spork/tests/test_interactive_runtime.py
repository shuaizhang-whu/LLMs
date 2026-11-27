"""Exercise the complete interactive driver using CPU model-response fixtures.

The production turn runners, official Orchestrator, isolated tool executor,
state replay, result seals and resume path run unchanged. Model transport is
replaced and all actual aiohttp requests are forbidden during this test.
"""
import asyncio
from contextlib import ExitStack, redirect_stdout
import io
from itertools import product
import json
import math
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import run_interactive_smoke as driver
from spork_core.eval_artifacts import digest, load_sealed
from spork_core.qwen import parse_baseline_tool_call
from tau2.data_model.message import AssistantMessage
from tau2.data_model.tasks import Task
from tau2.evaluator.evaluator_env import EnvironmentEvaluator
from tests.test_interactive_tau2_bridge import create_call, environment
from tests.test_launch_timing import Adapter, CharTokenizer


class DialogueAdapter(Adapter):
    def __init__(self, tokenizer, tools):
        self.tokenizer = tokenizer
        self.tools = tools

    def render_main_prompt(self, messages, **kwargs):
        return json.dumps(messages)

    def parse_main_tool_call(self, text):
        return parse_baseline_tool_call(text)

    def parse_probe_tool_call(self, text):
        call = json.loads(text)
        return call if call.get("name") else None


def stage(prompt):
    if '"role": "tool"' in prompt:
        return "done"
    if "Yes, please proceed." in prompt:
        return "tool"
    return "confirm"


class FixtureTransport:
    failures_left = 0
    streams = 0

    def __init__(self, *_):
        pass

    @staticmethod
    async def status(_):
        return {"active_requests": 0, "pending_verifications": 0,
                "accounting_errors": 0, "dropped_injections": 0, "prefix_mismatches": 0}

    async def clear(self, _):
        return {"status": "cleared"}

    async def set_draft(self, *_):
        return {"status": "ok"}

    async def stream(self, session, prompt, state, **kwargs):
        type(self).streams += 1
        phase = stage(bytes(prompt).decode())
        text = "<think>\n" + "thinking.\n" * (80 if phase == "tool" else 5) + "</think>\n"
        text += {"confirm": "May I proceed?", "tool": "<tool_call>" + json.dumps(create_call()),
                 "done": "Created it."}[phase]
        failing = type(self).failures_left > 0
        if failing:
            type(self).failures_left -= 1
        try:
            for end in range(10, len(text) + 10, 10):
                state.text = text[:end]
                state.token_ids = list(state.text.encode())
                state.first_output_at = state.first_output_at or time.perf_counter()
                state.changed.set()
                await asyncio.sleep(.001)
            state.finish_reason = "length" if failing else "stop"
        finally:
            state.finished_at = time.perf_counter()
            state.changed.set()

    async def complete(self, session, ids, **kwargs):
        prompt = bytes(ids).decode()
        phase = stage(prompt)
        finish, logs = "stop", {"confidence": 1.0}
        if kwargs["max_tokens"] == 1:
            text = "<tool_call>"
            logs = {"top_logprobs": [{"<tool_call>": math.log(.99 if phase == "tool" else .01)}]}
        elif prompt.endswith('{"name": "'):
            text = json.dumps(create_call()) if phase == "tool" else "{}"
            finish = "stop" if phase == "tool" else "length"
        else:
            text = "<tool_call>" + json.dumps(create_call())
        return {"text": text, "token_ids": list(text.encode()), "finish_reason": finish,
                "logprobs": logs, "wall_s": 0., "usage": {}}


class InteractiveRuntimeTests(unittest.TestCase):
    def test_real_turn_runners_dialogue_seals_failures_and_resume(self):
        for reference, fail_first in product(("conservative", "paper"), (False, True)):
            with self.subTest(reference=reference, fail_first=fail_first), tempfile.TemporaryDirectory() as temporary:
                args = SimpleNamespace(model="/cpu-fixture", base_url="http://unused.invalid/v1",
                    output=Path(temporary), tool_floor=0., max_tokens=8192, max_steps=20,
                    eta_reference=reference, repeats=2)
                task = Task(id="cpu", user_scenario={"instructions": "PRIVATE_USER_SCENARIO"},
                    evaluation_criteria={"actions": [{"action_id": "create", **create_call()}],
                                         "reward_basis": ["DB"]})
                manifest = {"actor_sampling": {"temperature": .6, "top_p": .95, "top_k": 20},
                    "schedule": [dict(group_id="airline:cpu", domain="airline", task_id="cpu", repeat=r,
                                      method=m) for r in range(2) for m in ("spork", "eta_retarget")]}
                seen_user = []

                def user_response(**kwargs):
                    last = kwargs["messages"][-1].content
                    seen_user.append(last)
                    answer = {"Hi! How can I help you today?": "Please create a task.",
                              "May I proceed?": "Yes, please proceed.", "Created it.": "###STOP###"}[last]
                    return AssistantMessage(role="assistant", content=answer,
                                            raw_data={"choices": [{"finish_reason": "stop"}]})

                def quality(simulation, actual_task, *_):
                    score = EnvironmentEvaluator.calculate_reward(environment, actual_task,
                                                                    simulation.messages, strict_replay=True)
                    return {"reward_info": score.model_dump(mode="json"), "error": None}

                FixtureTransport.failures_left = int(fail_first)
                FixtureTransport.streams = 0
                with ExitStack() as stack, redirect_stdout(io.StringIO()):
                    for target, replacement in [
                        ("transformers.AutoTokenizer.from_pretrained", lambda *a, **kw: CharTokenizer()),
                        ("scripts.run_interactive_smoke.QwenToolCallAdapter", DialogueAdapter),
                        ("scripts.run_interactive_smoke.load_tools", lambda _: []),
                        ("scripts.run_interactive_smoke.load_system_prompt", lambda _: "Public policy"),
                        ("scripts.run_interactive_smoke.get_environment", lambda _: environment()),
                        ("scripts.run_interactive_smoke.preflight", lambda _: {}),
                        ("scripts.run_interactive_smoke.PrefixHttpClient", FixtureTransport),
                        ("spork_core.timed_retry.PrefixHttpClient", FixtureTransport),
                        ("spork_core.early_exit.PrefixHttpClient", FixtureTransport),
                        ("tau2.user.user_simulator.generate", user_response),
                        ("scripts.run_interactive_smoke.evaluate_local", quality),
                    ]:
                        stack.enter_context(patch(target, replacement))
                    stack.enter_context(patch("aiohttp.ClientSession._request", side_effect=AssertionError("Real network forbidden")))
                    try:
                        driver.run(args, manifest, {"airline:cpu": task})
                    except RuntimeError:
                        details = [json.loads(p.read_text()) for p in (args.output / "records").glob("*.json")
                                   if not p.name.endswith(".complete.json")]
                        self.fail(json.dumps([{k: r.get(k) for k in ("method", "errors", "traceback")}
                                              | {"turn_errors": [t.get("spec_cleanup_error", t.get("worker_errors"))
                                                                  for t in r["turns"]]}
                                              for r in details], indent=2))
                    paths = sorted((args.output / "records").glob("*.json"))
                    checksums = {p.name: digest(p) for p in paths}
                    stream_count, user_count = FixtureTransport.streams, len(seen_user)
                    driver.run(args, manifest, {"airline:cpu": task})
                    self.assertEqual(FixtureTransport.streams, stream_count)
                    self.assertEqual(len(seen_user), user_count)
                    self.assertEqual(checksums, {p.name: digest(p) for p in paths})
                results = {m: load_sealed(args.output / "records" / f"airline_cpu_{m}.json")
                           for m in ("spork", "eta_retarget")}
                for mode, record in results.items():
                    if fail_first and mode == "spork":
                        self.assertTrue(record["errors"])
                        self.assertEqual(record["quality"], {})
                        self.assertEqual(record["failed_turn"]["partial_actor"]["finish_reason"], "length")
                        self.assertTrue(record["partial_messages"])
                    else:
                        self.assertEqual(record["errors"], [])
                        self.assertEqual(record["termination_reason"], "user_stop")
                        self.assertEqual(record["quality"]["reward_info"]["reward"], 1)
                        self.assertEqual(len(record["turns"]), 3)
                        self.assertTrue(record["turns"][1]["spec_used"])
                        self.assertEqual(sum(t["baseline_has_tool_call"] for t in record["turns"]), 1)
                self.assertTrue(results["eta_retarget"]["turns"][1]["exit"]["fired"])
                for mode in ("spork", "eta_retarget"):
                    second = load_sealed(args.output / "records" / f"airline_cpu_{mode}_repeat2.json")
                    self.assertEqual(second["repeat"], 1)
                    self.assertEqual(second["errors"], [])
                    self.assertEqual(second["quality"]["reward_info"]["reward"], 1)


if __name__ == "__main__":
    unittest.main()
