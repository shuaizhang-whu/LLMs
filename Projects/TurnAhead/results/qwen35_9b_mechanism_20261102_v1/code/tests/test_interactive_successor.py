"""CPU integration checks; mock only model turns, keep real pair/adoption logic."""
import asyncio
import hashlib
import json
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from spork_core.executors import commit_speculative
from spork_core.interactive_successor import SuccessorInteractiveAgent
from spork_core.interactive_tau2 import NativeToolExecutor
from spork_core.turn_runner import _append_tool_messages
from spork_core.timed_retry import ActorGenerationError
from tau2.data_model.message import ToolMessage, UserMessage


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(text.encode())


class Adapter:
    tokenizer = Tokenizer()
    main_stop = ["</tool_call>"]
    tools = []
    def render_main_prompt(self, messages, **kwargs):
        return json.dumps(messages, sort_keys=True)
    def speculative_tool_message(self, call):
        return "<tool_call>\n" + json.dumps(call) + "\n"


class Env:
    def __init__(self):
        self.value = 0
    def get_tools(self):
        return []
    def get_policy(self):
        return "Public test policy"
    def get_db_hash(self):
        return str(self.value)
    def get_user_db_hash(self):
        return None
    def get_response(self, call):
        self.value += call.arguments["delta"]
        return ToolMessage(id=call.id, role="tool", content=str(self.value), error=False)


class Tests(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.Runner()
        self.addCleanup(self.loop.close)
        async def clear(*args):
            return {"status": "cleared"}
        async def status(*args):
            return {"active_requests": 0, "pending_verifications": 0}
        self.runtime = SimpleNamespace(run=self.loop.run, session=None,
            client=SimpleNamespace(model="cpu-mock", base_url="mock", timeout_s=10, presence_penalty=1.5),
            http=SimpleNamespace(root="mock", clear=clear, status=status))
        self.cfgs = []
        self.block_child = False
        self.child_stopped = False
        self.child_reply = False
        self.parent_call = None
        self.fail_child = False
        self.fail_canonical = False
        self.infrastructure_error = False

    def agent(self, mode="raw", max_steps=100):
        env = Env()
        with patch("spork_core.interactive_successor.NativeQwen35Adapter", return_value=Adapter()):
            agent = SuccessorInteractiveAgent(env, self.runtime, Tokenizer(), mode, floor_s=0)
        agent.orchestrator = SimpleNamespace(environment=env, step_count=1,
            max_steps=max_steps, num_errors=0, max_errors=5)
        self.addCleanup(agent.close)
        return agent

    async def model_turn(self, session, client, adapter, messages, executor, turn, cfg, *, http=None):
        self.cfgs.append((turn, cfg.launch_mode, cfg.max_retries, cfg.d3_set is not None))
        ids = adapter.tokenizer.encode(adapter.render_main_prompt(messages))
        state = SimpleNamespace(text="unfinished reasoning", token_ids=[turn],
                                finished_at=None, finish_reason=None)
        if http is not None:
            http.state, http.prompt_ids = state, ids
        if turn % 2:
            call = self.parent_call or {"name": "advance", "arguments": {"delta": 1}}
            execution = await asyncio.to_thread(executor.execute_speculative, call)
            await asyncio.sleep(.03)
            commit_speculative(executor, execution)
            text = execution.text
        else:
            if self.fail_child and (http is not None or self.fail_canonical):
                if http is not None:
                    executor.env.value = 999  # Must never reach the real environment.
                error_type = RuntimeError if self.infrastructure_error else ActorGenerationError
                error = error_type("synthetic missing reasoning boundary")
                error.partial_generation = {"text": "unfinished", "token_ids": [1],
                    "prompt_token_ids": ids, "generation_settings": {"seed": cfg.seed},
                    "finish_reason": "stop"}
                raise error
            if self.block_child:
                try:
                    await asyncio.sleep(20)
                finally:
                    self.child_stopped = True
            call = {"name": "advance", "arguments": {"delta": 10}}
            text = None if self.child_reply else (await asyncio.to_thread(executor.execute, call))[0]
        has_tool = turn % 2 or not self.child_reply
        output = adapter.speculative_tool_message(call) if has_tool else "<think>private</think>Please confirm."
        if has_tool:
            _append_tool_messages(messages, output, text)
        state.finished_at, state.finish_reason = time.perf_counter(), "stop"
        return {"turn": turn, "prompt_token_ids": ids, "generation_settings": {"seed": cfg.seed},
            "main_token_ids": list(output.encode()), "baseline_text": output,
            "baseline_has_tool_call": bool(has_tool), "baseline_tool_name": call["name"] if has_tool else None,
            "baseline_tool_args": call["arguments"] if has_tool else None,
            "tool_result_sha256": hashlib.sha256(text.encode()).hexdigest() if has_tool else None,
            "environment_after": {"agent_db": executor.env.get_db_hash(), "user_db": None},
            "turn_end_ms": 1., "completion_valid": True, "d3_status": {"clear_status": "cleared"}}

    def patched(self):
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch("spork_core.successor_prefetch.run_timed_retry_turn", self.model_turn))
        stack.enter_context(patch("spork_core.interactive_successor.run_timed_retry_turn", self.model_turn))
        return stack

    def first(self, agent):
        return agent.generate_next_message(UserMessage(role="user", content="advance"),
            [{"role": "system", "content": "Public policy only"}])

    def adopt(self, agent, output):
        response, env = agent.adopt(output.tool_calls[0], agent.orchestrator.environment)
        agent.orchestrator.environment = env
        agent.orchestrator.step_count += 2
        return response

    def test_parent_handoff_does_not_wait_for_child_and_stop_drains_it(self):
        agent = self.agent()
        self.block_child = True
        with self.patched():
            started = time.perf_counter()
            output, state = self.first(agent)
            self.assertLess(time.perf_counter() - started, 1.)
            self.assertEqual(agent.orchestrator.environment.value, 0)
            self.adopt(agent, output)
            self.assertEqual(agent.orchestrator.environment.value, 1)
            agent.close()
            self.assertTrue(self.child_stopped)
            self.assertEqual(agent.orchestrator.environment.value, 1)
            self.assertEqual(len(agent.turns), 1)

    def test_native_child_write_is_adopted_once_at_its_own_boundary(self):
        agent = self.agent()
        with self.patched():
            output, state = self.first(agent)
            response = self.adopt(agent, output)
            self.assertEqual(agent.orchestrator.environment.value, 1)
            output, state = agent.generate_next_message(response, state)
            self.assertEqual(agent.orchestrator.environment.value, 1)
            self.assertEqual(output.tool_calls[0].arguments, {"delta": 10})
            self.adopt(agent, output)
            self.assertEqual(agent.orchestrator.environment.value, 11)
            self.assertEqual(agent.prefetch["pairs"][0]["native_delivery_count"], 2)
            with self.assertRaisesRegex(RuntimeError, "duplicate"):
                agent.adopt(output.tool_calls[0], agent.orchestrator.environment)

    def test_changed_native_parent_state_rejects_the_child_handoff(self):
        agent = self.agent()
        with self.patched():
            output, state = self.first(agent)
            response = self.adopt(agent, output)
            agent.orchestrator.environment.value = 99
            with self.assertRaisesRegex(RuntimeError, "starting state"):
                agent.generate_next_message(response, state)
            self.assertEqual(agent.orchestrator.environment.value, 99)

    def test_failed_speculative_child_recovers_from_committed_state_in_all_arms(self):
        self.fail_child = True
        for mode in ("raw", "raw_no_probe", "raw_one_probe"):
            with self.subTest(mode=mode), self.patched():
                self.cfgs.clear()
                agent = self.agent(mode)
                output, state = self.first(agent)
                response = self.adopt(agent, output)
                output, state = agent.generate_next_message(response, state)
                self.assertEqual(agent.orchestrator.environment.value, 1)
                self.adopt(agent, output)
                self.assertEqual(agent.orchestrator.environment.value, 11)
                pair = agent.prefetch["pairs"][0]
                self.assertTrue(pair["recovery_succeeded"])
                self.assertFalse(pair["adopted"])
                self.assertEqual(pair["failed_child_generation"]["text"], "unfinished")
                self.assertEqual([c[0] for c in self.cfgs], [1, 2, 2])
                self.assertEqual(self.cfgs[-1], (2, "original", 5, True))
                agent.close()

    def test_failed_canonical_retry_stops_task_without_another_retry(self):
        self.fail_child = self.fail_canonical = True
        agent = self.agent()
        with self.patched():
            output, state = self.first(agent)
            response = self.adopt(agent, output)
            with self.assertRaises(ActorGenerationError):
                agent.generate_next_message(response, state)
            self.assertEqual([c[0] for c in self.cfgs], [1, 2, 2])
            self.assertEqual(agent.orchestrator.environment.value, 1)
            self.assertEqual(len(agent.turns), 1)

    def test_unexpected_child_error_does_not_trigger_model_retry(self):
        self.fail_child = self.infrastructure_error = True
        agent = self.agent()
        with self.patched():
            output, state = self.first(agent)
            response = self.adopt(agent, output)
            with self.assertRaisesRegex(RuntimeError, "synthetic"):
                agent.generate_next_message(response, state)
            self.assertEqual([c[0] for c in self.cfgs], [1, 2])
            self.assertEqual(agent.orchestrator.environment.value, 1)

    def test_visible_reply_accepts_a_followup_user_after_child_cleanup(self):
        agent = self.agent()
        self.child_reply = True
        with self.patched():
            output, state = self.first(agent)
            response = self.adopt(agent, output)
            output, state = agent.generate_next_message(response, state)
            self.assertEqual(output.content, "Please confirm.")
            self.assertTrue(agent._task.done())
            self.assertNotIn("private", json.dumps(state))
            output, state = agent.generate_next_message(UserMessage(role="user", content="Yes"), state)
            self.assertEqual(len(agent.turns), 3)
            self.assertIn({"role": "user", "content": "Yes"}, state)

    def test_near_native_step_cap_does_not_launch_an_unusable_child(self):
        agent = self.agent(max_steps=3)
        with self.patched():
            output, state = self.first(agent)
            self.assertEqual(len(self.cfgs), 1)
            self.assertFalse(agent.prefetch["pairs"][0]["child_started"])

    def test_child_probe_budgets_do_not_change_parent_or_formal_tools(self):
        for mode, budget, launch in [("raw",5,"original"),("raw_one_probe",1,"original"),("raw_no_probe",0,"off")]:
            with self.subTest(mode=mode), self.patched():
                self.cfgs.clear()
                agent = self.agent(mode)
                output, state = self.first(agent)
                response = self.adopt(agent, output)
                output, _ = agent.generate_next_message(response, state)
                self.adopt(agent, output)
                self.assertEqual(self.cfgs[0], (1,"original",5,True))
                self.assertEqual(self.cfgs[1], (2,launch,budget,False))
                self.assertEqual(agent.orchestrator.environment.value, 11)
                agent.close()

    def test_real_native_orchestration_and_strict_replay_accept_successor_handoff(self):
        from pathlib import Path
        from spork_core import paths
        from spork_core.interactive_tau2 import SpeculativeOrchestrator
        from tau2.data_model.tasks import Task
        from tau2.domains.airline.environment import get_environment
        from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
        from tau2.user.user_simulator import UserSimulator

        class ScriptedUser(UserSimulator):
            def _generate_next_message(self, message, state):
                return UserMessage(role="user", content="Please help with XEWRD9." if not state.messages else "###STOP###")

        raw = json.loads((paths.TAU2_ROOT / "data/tau2/domains/airline/tasks.json").read_text())
        task = Task.model_validate(next(r for r in raw if str(r["id"]) == "13"))
        env = get_environment()
        self.parent_call = {"name":"transfer_to_human_agents","arguments":{"summary":"Needs human assistance"}}
        self.child_reply = True
        with patch("spork_core.interactive_successor.NativeQwen35Adapter", return_value=Adapter()):
            agent = SuccessorInteractiveAgent(env, self.runtime, Tokenizer(), "raw", floor_s=0)
        self.addCleanup(agent.close)
        user = ScriptedUser(llm="cpu-mock", instructions=str(task.user_scenario))
        orch = SpeculativeOrchestrator(domain="airline", agent=agent, user=user,
            environment=env, task=task, max_steps=20, seed=42, validate_communication=True)
        agent.orchestrator = orch
        with self.patched():
            simulation = orch.run()
        reward = evaluate_simulation(simulation, task, EvaluationType.ALL, False, "airline", strict_replay=True)
        self.assertEqual(reward.reward, 1.)
        self.assertEqual(len(agent.turns), 2)
        self.assertEqual(sum(isinstance(m, ToolMessage) for m in simulation.messages), 1)
        self.assertFalse(any("private" in (m.content or "") for m in simulation.messages))
        self.assertEqual(agent.prefetch["pairs"][0]["native_delivery_count"], 2)


if __name__ == "__main__":
    unittest.main()
