"""CPU integration checks with the official Orchestrator and UserSimulator."""
from copy import deepcopy
import hashlib
import json
import unittest
from unittest.mock import patch

from scripts.interactive_tau2_bridge import (
    CommittedToolEnvironment, InteractiveToolExecutor, SporkInteractiveAgent, db_hashes,
)
from tau2.agent.base_agent import AgentError
from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage
from tau2.data_model.simulation import TerminationReason
from tau2.data_model.tasks import Task
from tau2.domains.mock.data_model import MockDB, User
from tau2.domains.mock.environment import get_environment
from tau2.evaluator.evaluator_env import EnvironmentEvaluator
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.user.user_simulator import UserSimulator


def environment(**kwargs):
    return get_environment(db=MockDB(tasks={}, users={
        "u": User(user_id="u", name="Test User", tasks=[])}), **kwargs)


def create_call(title="confirmed"):
    return {"name": "create_task", "arguments": {"user_id": "u", "title": title}}


def turn_record(executor, call=None, text="Done."):
    return {"completion_valid": True, "main_finish_reason": "stop",
            "baseline_has_tool_call": call is not None,
            "baseline_tool_name": call["name"] if call else None,
            "baseline_tool_args": call["arguments"] if call else None,
            "baseline_text": text, "final_answer": text,
            "tool_result_sha256": (hashlib.sha256(executor.pending.response.content.encode()).hexdigest()
                                   if call else None)}


class InteractiveBridgeTests(unittest.TestCase):
    def test_official_dialogue_confirms_then_commits_once_and_replays_strictly(self):
        executor = InteractiveToolExecutor(environment(), floor_s=0)
        relay = CommittedToolEnvironment(executor)
        seen_actor, seen_user = [], []

        def run_turn(history, tools, turn):
            seen_actor.append(deepcopy(history))
            if turn == 1:
                return turn_record(tools, text="<think>PRIVATE_ACTOR_THOUGHT</think>May I create it?")
            if turn == 2:
                wrong = tools.execute_speculative(create_call("discard this prediction"))
                self.assertEqual(len(tools.env.tools.db.tasks), 0)
                self.assertEqual(len(wrong.state.env.tools.db.tasks), 1)
                right = tools.execute_speculative(create_call())
                tools.commit_speculative(right)
                return turn_record(tools, create_call(), '<think>thought</think><tool_call>raw junk')
            self.assertEqual(sum(m["role"] == "tool" for m in history), 1)
            return turn_record(tools, text="<think>private</think>Created it.")

        agent = SporkInteractiveAgent(executor, "Public domain policy", run_turn)
        user = UserSimulator(llm="cpu-test-only", instructions="PRIVATE_USER_SCENARIO; confirm only when asked.")
        task = Task(id="bridge-test", user_scenario={"instructions": "PRIVATE_USER_SCENARIO"},
                    evaluation_criteria={"actions": [{"action_id": "create", **create_call()}],
                                         "reward_basis": ["DB"]})
        replies = iter(["Please create a task.", "Yes, please do.", "###STOP###"])

        def user_model(**kwargs):
            seen_user.append(kwargs["messages"])
            return AssistantMessage(role="assistant", content=next(replies))

        with patch("tau2.user.user_simulator.generate", side_effect=user_model) as model:
            simulation = Orchestrator(domain="mock", agent=agent, user=user, environment=relay,
                                      task=task, seed=42, max_steps=20).run()
        self.assertEqual(model.call_count, 3)
        self.assertEqual(simulation.termination_reason, TerminationReason.USER_STOP)
        self.assertEqual(len(executor.env.tools.db.tasks), 1)
        self.assertEqual(executor.env.tools.db.tasks["task_1"].title, "confirmed")
        self.assertIsNone(executor.pending)
        self.assertNotIn("PRIVATE_USER_SCENARIO", json.dumps(seen_actor))
        self.assertIn("PRIVATE_USER_SCENARIO", seen_user[0][0].content)
        self.assertNotIn("PRIVATE_ACTOR_THOUGHT", " ".join(m.content or "" for m in simulation.messages))
        self.assertTrue(any(m["content"] == "Yes, please do." for m in seen_actor[1]))
        Orchestrator.validate_message_history(simulation.messages)
        score = EnvironmentEvaluator.calculate_reward(environment, task, simulation.messages,
                                                        strict_replay=True)
        self.assertTrue(score.db_check.db_match)

    def test_relay_rejects_mismatched_or_duplicate_action_and_preserves_error(self):
        executor = InteractiveToolExecutor(environment(), floor_s=0)
        relay = CommittedToolEnvironment(executor)
        call = {"name": "create_task", "arguments": {"user_id": "missing", "title": "x"}}
        executor.execute(call)
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            relay.get_response(ToolCall(id="wrong", **create_call()))
        response = relay.get_response(ToolCall(id="correct", **call))
        self.assertTrue(response.error)
        self.assertEqual(response.id, "correct")
        self.assertIn("not found", response.content)
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            relay.get_response(ToolCall(id="duplicate", **call))
        self.assertEqual(len(executor.env.tools.db.tasks), 0)

    def test_stale_speculation_cannot_replace_new_environment(self):
        executor = InteractiveToolExecutor(environment(), floor_s=0)
        speculative = executor.execute_speculative(create_call("stale"))
        executor.env.get_response(ToolCall(**create_call("external change")))
        state = db_hashes(executor.env)
        with self.assertRaisesRegex(RuntimeError, "stale environment"):
            executor.commit_speculative(speculative)
        self.assertEqual(db_hashes(executor.env), state)
        self.assertIsNone(executor.pending)

    def test_text_reply_cannot_hide_committed_tool(self):
        executor = InteractiveToolExecutor(environment(), floor_s=0)

        def bad_turn(history, tools, turn):
            tools.execute(create_call())
            return turn_record(tools, text="No action taken.")

        agent = SporkInteractiveAgent(executor, "policy", bad_turn)
        with self.assertRaisesRegex(AgentError, "committed despite a text-only"):
            agent.generate_next_message(UserMessage(role="user", content="Hello"), agent.get_init_state())

    def test_incomplete_actor_turn_is_not_a_normal_reply(self):
        executor = InteractiveToolExecutor(environment(), floor_s=0)
        agent = SporkInteractiveAgent(executor, "policy", lambda *_: {
            "completion_valid": False, "main_finish_reason": "length"})
        with self.assertRaisesRegex(AgentError, "Incomplete Actor"):
            agent.generate_next_message(UserMessage(role="user", content="Hello"), agent.get_init_state())

    def test_failed_generation_keeps_partial_evidence_and_elapsed_time(self):
        executor = InteractiveToolExecutor(environment(), floor_s=0)

        def failed_turn(*_):
            error = RuntimeError("length")
            error.partial_actor = {"finish_reason": "length", "token_ids": [1, 2]}
            raise error

        agent = SporkInteractiveAgent(executor, "policy", failed_turn)
        with self.assertRaises(AgentError):
            agent.generate_next_message(UserMessage(role="user", content="Hello"), agent.get_init_state())
        self.assertEqual(agent.failed_turn["partial_actor"]["finish_reason"], "length")
        self.assertEqual(agent.turns, [])
        self.assertEqual(len(agent.turn_wall_s), 1)


if __name__ == "__main__":
    unittest.main()
