"""Ticket-only telecom workload, evaluated with the bundled tau2 half-duplex grader."""
import json
import time

from spork_core import paths
from spork_core.executors import Tau2ToolExecutor
from spork_core.qwen import QwenToolCallAdapter

DATA = paths.TAU2_ROOT / "data/tau2/domains/telecom"


def task_plan(split, ids):
    split_ids = json.loads((DATA / "split_tasks.json").read_text())[split]
    tasks = json.loads((DATA / "tasks.json").read_text())
    by_id = {t["id"]: (index, t) for index, t in enumerate(tasks)}
    selected, material = [], {}
    for original in split_ids:
        index, task = by_id[original]
        group = f"telecom:{index}"
        if ids and group not in ids.split(","):
            continue
        selected.append(dict(domain="telecom", task_id=str(index), group_id=group,
                             official_task_id=original, split=split))
        material[group] = task
    if not selected or (ids and set(ids.split(",")) != set(material)):
        raise ValueError(f"Unknown telecom task in {split}: {ids}")
    return selected, material


class SoloAdapter(QwenToolCallAdapter):
    terminal_tool = "done"

    def parse_probe_tool_call(self, text):
        call = super().parse_probe_tool_call(text)
        return None if call and call["name"] == self.terminal_tool else call


class SoloExecutor(Tau2ToolExecutor):
    def _execute_on_env(self, env, tool_call):
        from tau2.data_model.message import ToolCall
        start = time.perf_counter()
        response = env.get_response(ToolCall(**tool_call))
        real = time.perf_counter() - start
        padding = max(0., self.floor_s - real)
        if padding:
            time.sleep(padding)
        return response.content, time.perf_counter() - start, real, padding


def setup(tokenizer, raw_task, floor):
    from tau2.agent.llm_agent import LLMSoloAgent
    from tau2.data_model.tasks import Task
    from tau2.domains.telecom.environment import get_environment
    task = Task.model_validate(raw_task)
    env = get_environment(solo_mode=True)
    init = task.initial_state
    env.set_state(init.initialization_data, init.initialization_actions, init.message_history or [])
    agent = LLMSoloAgent(env.get_tools() + env.get_user_tools(), env.get_policy(), task, "Qwen3-32B")
    adapter = SoloAdapter(tokenizer, [tool.openai_schema for tool in agent.tools])
    return adapter, [{"role": "system", "content": agent.system_prompt}], SoloExecutor(env, floor)


def score(raw_task, record):
    from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage
    from tau2.data_model.simulation import SimulationRun, TerminationReason
    from tau2.data_model.tasks import Task
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
    messages = []
    reason = TerminationReason.MAX_STEPS
    for t in record["turns"]:
        if t["baseline_has_tool_call"]:
            call = ToolCall(id=str(t["turn"]), name=t["baseline_tool_name"], arguments=t["baseline_tool_args"])
            messages.extend([AssistantMessage(role="assistant", tool_calls=[call]),
                             ToolMessage(role="tool", id=call.id, content=t["tool_observation"])])
        else:
            stopped = t.get("terminal_tool") == "done"
            reason = TerminationReason.AGENT_STOP if stopped else TerminationReason.AGENT_ERROR
            messages.append(AssistantMessage(role="assistant", content="###STOP###" if stopped else t["final_answer"]))
    if record["errors"]:
        reason = TerminationReason.AGENT_ERROR
    simulation = SimulationRun(id=record["session_id"], task_id=raw_task["id"], messages=messages,
                               termination_reason=reason, duration=record["wall_s"],
                               start_time=record["started_at"], end_time=record["ended_at"])
    reward = evaluate_simulation(simulation, Task.model_validate(raw_task), EvaluationType.ALL,
                                 solo_mode=True, domain="telecom")
    return {"termination": reason.value, **reward.model_dump(mode="json")}
