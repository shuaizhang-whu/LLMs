"""SPORK bridge for the local tau2 Orchestrator; no model requests on import.

The existing turn runners execute the canonical tool before returning. The
orchestrator must relay that result, rather than execute the action a second
time. This separate adapter is not imported by the fixed-user experiments.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import time
from typing import Callable

from spork_core.executors import SpeculativeToolExecution, Tau2ToolExecutor, tool_call_cache_key
from spork_core.paths import add_runtime_paths
from spork_core.turn_runner import _append_tool_messages

add_runtime_paths()

from tau2.agent.base_agent import AgentError, HalfDuplexAgent
from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
from tau2.utils.utils import get_now


def db_hashes(env) -> tuple:
    return env.get_db_hash(), env.get_user_db_hash()


@dataclass
class PreparedTool:
    call_key: str
    env: object
    response: ToolMessage
    before: tuple
    committed: bool = False


class InteractiveToolExecutor(Tau2ToolExecutor):
    """Execute through tau2's API and retain exactly one committed response.

    Unlike the legacy fixed-user executor, this preserves tau2's complete
    response serialization and error flag, so strict official replay can check
    the trajectory. Predictions still run on isolated environment copies.
    """

    def __init__(self, env, floor_s: float = 2.0):
        super().__init__(env, floor_s=floor_s)
        self.pending: PreparedTool | None = None

    def _prepare(self, env, call: dict) -> SpeculativeToolExecution:
        before = db_hashes(env)
        started = time.perf_counter()
        response = env.get_response(ToolCall(name=call["name"], arguments=call.get("arguments") or {}))
        real = time.perf_counter() - started
        delay = max(0.0, self.floor_s - real)
        if delay:
            time.sleep(delay)
        wall = time.perf_counter() - started
        prepared = PreparedTool(tool_call_cache_key(call), env, response, before)
        return SpeculativeToolExecution(response.content, wall, real, max(0.0, wall - real), prepared)

    def execute_speculative(self, tool_call: dict) -> SpeculativeToolExecution:
        return self._prepare(deepcopy(self.env), tool_call)

    def _publish(self, execution: SpeculativeToolExecution) -> None:
        prepared = execution.state
        if self.pending is not None or prepared.committed:
            raise RuntimeError("A committed tool response must be consumed exactly once")
        prepared.committed = True
        self.env = prepared.env
        self.pending = prepared
        self.last_real_wall_s = execution.real_wall_s
        self.last_floor_sleep_s = execution.artificial_sleep_s

    def execute(self, tool_call: dict) -> tuple[str, float]:
        if self.pending is not None:
            raise RuntimeError("Unconsumed canonical tool response")
        execution = self._prepare(self.env, tool_call)
        self._publish(execution)
        return execution.text, execution.wall_s

    def commit_speculative(self, execution: SpeculativeToolExecution) -> None:
        if db_hashes(self.env) != execution.state.before:
            raise RuntimeError("Speculative tool result was produced from a stale environment")
        self._publish(execution)

    def response_for(self, call: ToolCall, *, consume: bool) -> ToolMessage:
        prepared = self.pending
        expected = tool_call_cache_key({"name": call.name, "arguments": call.arguments})
        if call.requestor != "assistant" or prepared is None or prepared.call_key != expected:
            raise RuntimeError("Orchestrator tool call does not match the committed SPORK action")
        # Orchestrator sorts the trajectory by timestamp. Speculative execution
        # can finish before the Actor emits the call; stamp its delivery here.
        response = prepared.response.model_copy(deep=True, update={"id": call.id, "timestamp": get_now()})
        if consume:
            self.pending = None
        return response


class CommittedToolEnvironment:
    """Delegate state to SPORK's current environment and relay executed tools."""

    def __init__(self, executor: InteractiveToolExecutor):
        self.executor = executor

    def __getattr__(self, name):
        return getattr(self.executor.env, name)

    def get_response(self, call: ToolCall) -> ToolMessage:
        if call.requestor == "assistant":
            return self.executor.response_for(call, consume=True)
        if self.executor.pending is not None:
            raise RuntimeError("User tool cannot interleave with an unconsumed agent tool")
        return self.executor.env.get_response(call)


@dataclass
class InteractiveState:
    messages: list[dict]
    waiting_tool: ToolMessage | None = None


class SporkInteractiveAgent(HalfDuplexAgent[InteractiveState]):
    """Adapt an existing SPORK turn callback to the official half-duplex loop.

    ``run_turn(messages, executor, turn)`` runs one complete existing turn and
    returns its record. An external driver owns the HTTP session/event loop.
    Only public dialogue enters the Actor; scenario instructions belong to the
    official UserSimulator, which the orchestrator calls on every text reply.
    """

    def __init__(self, executor: InteractiveToolExecutor, system_prompt: str, run_turn: Callable):
        super().__init__(executor.env.get_tools(), executor.env.get_policy())
        self.executor = executor
        self.system_prompt = system_prompt
        self.run_turn = run_turn
        self.turns: list[dict] = []
        self.turn_wall_s: list[float] = []
        self.errors: list[str] = []
        self.failed_turn: dict | None = None
        self.seed = 42

    def set_seed(self, seed: int):
        self.seed = seed

    def get_init_state(self, message_history=None) -> InteractiveState:
        messages = [{"role": "system", "content": self.system_prompt}]
        for msg in message_history or []:
            if not isinstance(msg, (AssistantMessage, UserMessage)) or msg.is_tool_call():
                raise ValueError("Interactive bridge currently supports text-only initial history")
            messages.append({"role": msg.role, "content": msg.content})
        return InteractiveState(messages)

    def generate_next_message(self, message, state: InteractiveState):
        try:
            return self._generate(message, state)
        except Exception as error:
            self.errors.append(f"{type(error).__name__}: {error}")
            raise AgentError(self.errors[-1]) from error

    def _generate(self, message, state: InteractiveState):
        if isinstance(message, ToolMessage):
            expected = state.waiting_tool
            if (expected is None or self.executor.pending is not None
                    or any(getattr(message, k) != getattr(expected, k)
                           for k in ("id", "content", "requestor", "error"))):
                raise ValueError("Unexpected tool response in agent dialogue")
            # The turn runner already appended this observation to Actor history.
            state.waiting_tool = None
        elif isinstance(message, UserMessage) and not message.is_tool_call():
            if state.waiting_tool is not None or self.executor.pending is not None:
                raise ValueError("User reply arrived before the canonical tool response")
            state.messages.append({"role": "user", "content": message.content})
        else:
            raise ValueError("Expected one public user message or canonical tool response")

        started = time.perf_counter()
        working_history = deepcopy(state.messages)
        try:
            entry = self.run_turn(working_history, self.executor, len(self.turns) + 1)
        except Exception as error:
            self.failed_turn = {"turn": len(self.turns) + 1,
                                "error": f"{type(error).__name__}: {error}",
                                "partial_actor": getattr(error, "partial_actor", None)}
            raise
        finally:
            self.turn_wall_s.append(time.perf_counter() - started)
        self.turns.append(entry)
        if not entry.get("completion_valid") or entry.get("main_finish_reason") != "stop":
            raise ValueError("Incomplete Actor turn")
        if entry.get("worker_errors") or entry.get("spec_cleanup_error"):
            raise ValueError("Tool cleanup failed")
        if entry["baseline_has_tool_call"]:
            call = ToolCall(id=f"spork-turn-{len(self.turns)}", name=entry["baseline_tool_name"],
                            arguments=entry["baseline_tool_args"] or {})
            response = self.executor.response_for(call, consume=False)
            if hashlib.sha256(response.content.encode()).hexdigest() != entry["tool_result_sha256"]:
                raise ValueError("Recorded tool result differs from committed response")
            # Use only the executed action and the official full observation.
            _append_tool_messages(state.messages, entry["baseline_text"], response.content,
                                  canonical_call={"name": call.name, "arguments": call.arguments})
            state.waiting_tool = response
            return AssistantMessage(role="assistant", tool_calls=[call]), state
        if self.executor.pending is not None:
            raise ValueError("Tool was committed despite a text-only Actor reply")
        text = entry.get("final_answer", entry["baseline_text"]).rsplit("</think>", 1)[-1].strip()
        if not text or "<think>" in text:
            raise ValueError("Empty reply or unfinished reasoning cannot be sent to the user")
        state.messages.append({"role": "assistant", "content": text})
        return AssistantMessage(role="assistant", content=text), state
