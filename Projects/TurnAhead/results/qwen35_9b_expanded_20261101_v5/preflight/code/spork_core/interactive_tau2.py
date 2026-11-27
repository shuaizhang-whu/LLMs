"""SPORK turn runners inside tau2's native conversational orchestrator.

All tool calls run on isolated environments. The native orchestrator adopts the
chosen environment exactly once when it receives the Actor's actual ToolCall.
User prompts, role flipping, stopping rules and evaluation remain tau2's. The
local Qwen user is an experimental simulator, not the published benchmark user.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import time
import uuid

import aiohttp

from . import paths
from .executors import SpeculativeToolExecution, Tau2ToolExecutor
from .prefix_replay import PrefixHttpClient, same_call
from .qwen import QwenToolCallAdapter
from .stable_refresh_exit import ExitConfig, run_early_exit_turn
from .timed_retry import run_timed_retry_turn
from .turn_runner import RetryTurnConfig
from .vllm_client import VllmClient

paths.add_runtime_paths()
from tau2.agent.base_agent import HalfDuplexAgent
from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT
from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.user.user_simulator import UserSimulator
from tau2.utils.llm_utils import to_litellm_messages
from tau2.utils.utils import get_now


def environment_key(env):
    return (env.get_db_hash(), env.get_user_db_hash())


class LocalRuntime:
    """One persistent HTTP session, shared sequentially by Actor and user."""

    def __init__(self, base_url, model):
        self.loop = asyncio.Runner()
        self.client = VllmClient(base_url, model, timeout_s=900)
        self.http = PrefixHttpClient(base_url, model, timeout_s=900)
        self.session = self.run(self._open())

    async def _open(self):
        return aiohttp.ClientSession(trust_env=False)

    def run(self, coro):
        return self.loop.run(coro)

    def close(self):
        self.run(self.session.close())
        self.loop.close()


class NativeToolExecutor(Tau2ToolExecutor):
    """Use native serialization, errors and synchronization on every branch."""

    def __init__(self, env, floor_s):
        super().__init__(env, floor_s=floor_s)
        self.last_response = None

    def fork_at(self, state):
        return NativeToolExecutor(copy.deepcopy(state), self.floor_s)

    def adopt_successor(self, other):
        if not isinstance(other, NativeToolExecutor):
            raise TypeError("Native successor must preserve native tool semantics")
        super().adopt_successor(other)
        self.last_response = other.last_response

    def _execute_native(self, env, call):
        started = time.perf_counter()
        response = env.get_response(ToolCall(
            id="pending", name=call["name"], arguments=call.get("arguments") or {},
            requestor="assistant"))
        real_s = time.perf_counter() - started
        sleep_s = max(0., self.floor_s - real_s)
        if sleep_s:
            time.sleep(sleep_s)
        execution = SpeculativeToolExecution(
            text=response.content, wall_s=time.perf_counter() - started,
            real_wall_s=real_s, artificial_sleep_s=sleep_s, state=env)
        execution.native_response = response
        return execution

    def execute(self, call):
        result = self._execute_native(self.env, call)
        self.commit_speculative(result)
        return result.text, result.wall_s

    def execute_speculative(self, call):
        return self._execute_native(copy.deepcopy(self.env), call)

    def commit_speculative(self, execution):
        super().commit_speculative(execution)
        self.last_response = execution.native_response


class InteractiveAgent(HalfDuplexAgent):
    """Expose one SPORK decision per native agent step, with deferred handoff."""

    def __init__(self, env, runtime, tokenizer, method, *, seed=42, max_tokens=8192,
                 floor_s=2., on_turn=None):
        super().__init__(env.get_tools(), env.get_policy())
        self.runtime, self.tokenizer, self.method = runtime, tokenizer, method
        self.seed, self.max_tokens, self.floor_s = seed, max_tokens, floor_s
        self.adapter = QwenToolCallAdapter(tokenizer, [t.openai_schema for t in self.tools])
        self.orchestrator = None
        self.pending = self.expected_reply = None
        self.turns = []
        self.on_turn = on_turn or (lambda _: None)

    def set_seed(self, seed):
        self.seed = seed

    def get_init_state(self, message_history=None):
        prompt = SYSTEM_PROMPT.format(agent_instruction=AGENT_INSTRUCTION,
                                      domain_policy=self.domain_policy)
        return [{"role": "system", "content": prompt}] + to_litellm_messages(message_history or [])

    async def _turn(self, state, executor):
        runtime = self.runtime
        request_id = "interactive-" + uuid.uuid4().hex
        prompt = self.adapter.render_main_prompt(state, enable_thinking=True)
        prompt_len = len(self.tokenizer.encode(prompt, add_special_tokens=False))

        async def register(call, _prompt):
            return await runtime.http.set_draft(runtime.session, request_id, prompt_len,
                                                call, self.tokenizer)

        async def clear():
            info = await runtime.http.clear(runtime.session)
            return {**await runtime.http.status(runtime.session), "clear_status": info["status"]}

        cfg = RetryTurnConfig(
            max_retries=5, confidence_threshold=.9, retry_token_step=50,
            probe_max_tokens=160, baseline_max_tokens=self.max_tokens,
            context_window_tokens=40960, enable_thinking=True, seed=self.seed,
            d3_set=register, d3_clear=clear, main_request_id=request_id,
            launch_mode="original")
        args = (runtime.session, runtime.client, self.adapter, state, executor,
                len(self.turns) + 1, cfg)
        if self.method == "spork":
            return await run_timed_retry_turn(*args)
        return await run_early_exit_turn(*args, ExitConfig(method=self.method))

    def generate_next_message(self, message, state):
        started = time.perf_counter()
        if self.pending is not None:
            raise RuntimeError("Previous formal call was not handed to the environment")
        if isinstance(message, ToolMessage):
            if self.expected_reply is None or message != self.expected_reply:
                raise RuntimeError("Orchestrator tool reply differs from adopted result")
            self.expected_reply = None  # Already appended by the SPORK turn runner.
        elif isinstance(message, UserMessage) and not message.is_tool_call():
            if self.expected_reply is not None:
                raise RuntimeError("Missing tool reply before the next user message")
            state.append({"role": "user", "content": message.content})
        else:
            raise ValueError("Only single tool replies and text user messages are supported")
        before = environment_key(self.orchestrator.environment)
        executor = NativeToolExecutor(copy.deepcopy(self.orchestrator.environment), self.floor_s)
        entry = self.runtime.run(self._turn(state, executor))
        entry["actor_step_wall_s"] = time.perf_counter() - started
        if (not entry.get("completion_valid") or entry.get("worker_errors")
                or entry.get("spec_cleanup_error")
                or (entry.get("d3_status") or {}).get("clear_status") not in ("ok", "cleared")):
            raise RuntimeError("Actor turn cleanup/completion check failed")
        if entry["baseline_has_tool_call"]:
            call = ToolCall(id="call-" + uuid.uuid4().hex,
                            name=entry["baseline_tool_name"],
                            arguments=entry["baseline_tool_args"], requestor="assistant")
            response = executor.last_response
            if response is None or hashlib.sha256(response.content.encode()).hexdigest() != entry["tool_result_sha256"]:
                raise RuntimeError("No matching native tool result for formal handoff")
            self.pending = (call, response, executor.env, before)
            output = AssistantMessage(role="assistant", content=None, tool_calls=[call])
        else:
            visible = entry["baseline_text"].split("</think>")[-1].strip()
            if not visible:
                raise RuntimeError("Empty visible Actor reply")
            state.append({"role": "assistant", "content": visible})
            output = AssistantMessage(role="assistant", content=visible)
        self.turns.append(entry)
        self.on_turn(entry)
        return output, state

    def adopt(self, call, env):
        if self.pending is None:
            raise RuntimeError("Unexpected or duplicate formal tool execution")
        expected, response, new_env, before = self.pending
        if (call.id != expected.id or call.requestor != "assistant"
                or not same_call(call.model_dump(), expected.model_dump())
                or before != environment_key(env)):
            raise RuntimeError("Call or starting environment changed before commit")
        self.pending = None
        response = response.model_copy(update={"id": call.id, "timestamp": get_now()})
        self.expected_reply = response
        return response, new_env


class SpeculativeOrchestrator(Orchestrator):
    """Native routing with an exact, one-time adoption at the tool boundary."""

    def _execute_tool_calls(self, tool_calls):
        if tool_calls and all(c.requestor == "assistant" for c in tool_calls):
            if len(tool_calls) != 1:
                raise ValueError("SPORK supports one tool call per decision")
            response, self.environment = self.agent.adopt(tool_calls[0], self.environment)
            self.num_errors += int(response.error)
            return [response]
        return super()._execute_tool_calls(tool_calls)


class LocalUserSimulator(UserSimulator):
    """Native user prompt/history/termination with a bounded local completion."""

    def __init__(self, runtime, tokenizer, instructions, *, max_tokens=512, thinking=False):
        super().__init__(llm=runtime.client.model, instructions=instructions, tools=None)
        self.runtime, self.tokenizer, self.max_tokens = runtime, tokenizer, max_tokens
        self.thinking = thinking
        self.calls = []
        self.seed = 42

    def set_seed(self, seed):
        self.seed = seed

    def _generate_next_message(self, message, state):
        if not isinstance(message, AssistantMessage) or message.is_tool_call():
            raise ValueError("Local user adapter supports text-only airline/retail users")
        if message.has_content():
            state.messages.append(message)
        messages = to_litellm_messages(state.system_messages + state.flip_roles())
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False,
                add_generation_prompt=True, enable_thinking=self.thinking)
        # transformers versions differ in the default tokenize=True return type.
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) + self.max_tokens > 40960:
            raise ValueError("User simulator context budget exceeded")
        choice = self.runtime.run(self.runtime.http.complete(self.runtime.session, ids,
            max_tokens=self.max_tokens, seed=self.seed, stop=["<|im_end|>"]))
        self.calls.append(choice)
        if choice["finish_reason"] != "stop":
            raise RuntimeError("User simulator output truncated")
        if self.thinking and "</think>" not in choice["text"]:
            raise RuntimeError("User simulator reasoning was not closed")
        text = choice["text"].split("</think>")[-1].strip()
        if not text or "<tool_call>" in text:
            raise RuntimeError("Invalid user simulator response")
        return UserMessage(role="user", content=text)
