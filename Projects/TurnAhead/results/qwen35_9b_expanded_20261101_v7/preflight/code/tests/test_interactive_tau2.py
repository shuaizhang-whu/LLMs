"""CPU checks for conversational routing and isolated native tool commits."""
import asyncio
import copy
import hashlib
import json
from types import SimpleNamespace
import unittest

from spork_core.interactive_tau2 import (
    InteractiveAgent, LocalUserSimulator, NativeToolExecutor, SpeculativeOrchestrator, environment_key)
from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage


class FakeEnv:
    def __init__(self):
        self.value = 0

    def get_db_hash(self):
        return str(self.value)

    def get_user_db_hash(self):
        return None

    def get_response(self, call):
        self.value += call.arguments.get("amount", 1)
        return ToolMessage(id=call.id, role="tool", content=str(self.value), error=False)


class InteractiveTests(unittest.TestCase):
    def staged(self):
        env = FakeEnv()
        executor = NativeToolExecutor(copy.deepcopy(env), floor_s=0)
        call = ToolCall(id="formal", name="increment", arguments={"amount": 2})
        result = executor.execute_speculative(call.model_dump())
        executor.commit_speculative(result)
        agent = object.__new__(InteractiveAgent)
        agent.pending = (call, executor.last_response, executor.env, environment_key(env))
        agent.expected_reply = None
        return agent, env, executor, call

    def test_write_is_isolated_and_committed_once(self):
        agent, env, executor, call = self.staged()
        self.assertEqual(env.value, 0)
        reply, committed = agent.adopt(call, env)
        self.assertEqual(committed.value, 2)
        self.assertEqual(reply.content, "2")
        self.assertEqual(reply.id, call.id)
        self.assertEqual(agent.expected_reply, reply)
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            agent.adopt(call, committed)

    def test_wrong_call_or_stale_state_cannot_commit(self):
        agent, env, _, call = self.staged()
        with self.assertRaisesRegex(RuntimeError, "changed"):
            agent.adopt(call.model_copy(update={"arguments": {"amount": 3}}), env)
        self.assertIsNotNone(agent.pending)
        env.value = 1
        with self.assertRaisesRegex(RuntimeError, "changed"):
            agent.adopt(call, env)

    def test_superseded_branch_never_contaminates_new_branch(self):
        env = FakeEnv()
        executor = NativeToolExecutor(env, floor_s=0)
        old = executor.execute_speculative({"name": "increment", "arguments": {"amount": 100}})
        new = executor.execute_speculative({"name": "increment", "arguments": {"amount": 2}})
        executor.commit_speculative(new)
        self.assertEqual(old.state.value, 100)
        self.assertEqual(env.value, 0)
        self.assertEqual(executor.env.value, 2)

    def test_native_error_flag_and_output_are_preserved(self):
        from tau2.domains.airline.environment import get_environment
        executor = NativeToolExecutor(get_environment(), floor_s=0)
        result = executor.execute_speculative({"name": "get_flight_status",
            "arguments": {"flight_number": "NOT_A_REAL_FLIGHT", "date": "2024-05-15"}})
        executor.commit_speculative(result)
        self.assertTrue(executor.last_response.error)
        self.assertEqual(result.text, executor.last_response.content)

    def test_visible_reply_allows_later_user_confirmation(self):
        agent = object.__new__(InteractiveAgent)
        agent.orchestrator = SimpleNamespace(environment=FakeEnv())
        agent.pending = agent.expected_reply = None
        agent.floor_s = 0
        agent.turns = []
        agent.on_turn = lambda _: None
        agent.runtime = SimpleNamespace(run=asyncio.run)

        async def fake_turn(state, executor):
            return {"turn": len(agent.turns) + 1, "baseline_has_tool_call": False,
                    "baseline_text": "<think>PRIVATE REASONING</think>Please confirm.",
                    "completion_valid": True, "d3_status": {"clear_status": "cleared"}}

        agent._turn = fake_turn
        state = [{"role": "system", "content": "Public policy only"}]
        output, state = agent.generate_next_message(UserMessage(role="user", content="Cancel it"), state)
        self.assertIsInstance(output, AssistantMessage)
        self.assertEqual(output.content, "Please confirm.")
        self.assertNotIn("PRIVATE", str(state))
        agent.generate_next_message(UserMessage(role="user", content="Yes, confirmed"), state)
        self.assertEqual(len(agent.turns), 2)
        self.assertIn({"role": "user", "content": "Yes, confirmed"}, state)

    def test_mismatched_tool_reply_is_rejected(self):
        agent = object.__new__(InteractiveAgent)
        agent.pending = None
        agent.expected_reply = ToolMessage(id="right", role="tool", content="2")
        with self.assertRaisesRegex(RuntimeError, "differs"):
            agent.generate_next_message(ToolMessage(id="wrong", role="tool", content="2"), [])

    def test_local_tokenizer_user_request_is_json_serializable(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "/mnt/data/zhangshuai/LLMs/Models/Qwen3-32B", local_files_only=True)

        response = "###STOP###"

        async def complete(session, ids, **kwargs):
            self.assertIsInstance(ids, list)
            self.assertTrue(all(isinstance(i, int) for i in ids))
            json.dumps({"prompt": ids, **kwargs})
            return {"text": response, "finish_reason": "stop", "wall_s": 0.}

        runtime = SimpleNamespace(client=SimpleNamespace(model="cpu-mock"), session=None,
                                  run=asyncio.run, http=SimpleNamespace(complete=complete))
        user = LocalUserSimulator(runtime, tokenizer, "You want to cancel your reservation.")
        output, state = user.generate_next_message(
            AssistantMessage(role="assistant", content="How can I help?"), user.get_init_state())
        self.assertTrue(user.is_stop(output))
        self.assertEqual(state.messages[-1], output)
        thinking_user = LocalUserSimulator(runtime, tokenizer, "Your task is complete.", thinking=True)
        response = "private user reasoning</think>###STOP###"
        output, _ = thinking_user.generate_next_message(
            AssistantMessage(role="assistant", content="Goodbye."), thinking_user.get_init_state())
        self.assertEqual(output.content, "###STOP###")
        response = "###STOP###"
        with self.assertRaisesRegex(RuntimeError, "reasoning was not closed"):
            thinking_user.generate_next_message(
                AssistantMessage(role="assistant", content="Goodbye."), thinking_user.get_init_state())

    def test_native_orchestration_and_strict_evaluation_accept_handoff(self):
        from spork_core import paths
        from spork_core.turn_runner import _append_tool_messages
        from tau2.data_model.tasks import Task
        from tau2.domains.airline.environment import get_environment
        from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
        from tau2.user.user_simulator import UserSimulator
        raw = json.loads((paths.TAU2_ROOT / "data/tau2/domains/airline/tasks.json").read_text())
        task = Task.model_validate(next(t for t in raw if str(t["id"]) == "13"))

        class ScriptedUser(UserSimulator):
            def _generate_next_message(self, message, state):
                return UserMessage(role="user", content=("Please help with XEWRD9."
                    if not state.messages else "###STOP###"))

        env = get_environment()
        agent = InteractiveAgent(env, SimpleNamespace(run=asyncio.run), None, "spork", floor_s=0)

        async def fake_turn(state, executor):
            base = {"turn": len(agent.turns) + 1, "completion_valid": True,
                    "d3_status": {"clear_status": "cleared"}}
            if agent.turns:
                return {**base, "baseline_has_tool_call": False,
                        "baseline_text": "<think>hidden</think>You have been transferred."}
            call = {"name": "transfer_to_human_agents", "arguments": {"summary": "Needs human assistance"}}
            result, _ = executor.execute(call)
            text = "<tool_call>" + json.dumps(call) + "</tool_call>"
            _append_tool_messages(state, text, result)
            return {**base, "baseline_has_tool_call": True, "baseline_tool_name": call["name"],
                    "baseline_tool_args": call["arguments"], "baseline_text": text,
                    "tool_result_sha256": hashlib.sha256(result.encode()).hexdigest()}

        agent._turn = fake_turn
        user = ScriptedUser(llm="cpu-mock", instructions=str(task.user_scenario))
        orchestrator = SpeculativeOrchestrator(domain="airline", agent=agent, user=user,
            environment=env, task=task, seed=42, max_steps=10, validate_communication=True)
        agent.orchestrator = orchestrator
        result = orchestrator.run()
        reward = evaluate_simulation(result, task, EvaluationType.ALL, False, "airline", strict_replay=True)
        self.assertEqual(reward.reward, 1.)
        self.assertEqual(len(agent.turns), 2)
        self.assertEqual(sum(isinstance(m, ToolMessage) for m in result.messages), 1)
        self.assertFalse(any("hidden" in (m.content or "") for m in result.messages))


if __name__ == "__main__":
    unittest.main()
