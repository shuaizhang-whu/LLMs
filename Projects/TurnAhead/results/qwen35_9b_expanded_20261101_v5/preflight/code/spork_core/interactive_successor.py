"""Exact-input successor pairs at native tau2 tool/participant boundaries.

The native orchestrator receives each formal message in order. Speculative
work stays on a private executor until that boundary is reached. A persistent
event loop lets the full parent finish and hand off its result while the child
continues. No child can survive into a user-simulator request or finalization.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import time
import uuid

import aiohttp

from .interactive_tau2 import (
    InteractiveAgent, LocalRuntime, LocalUserSimulator, NativeToolExecutor,
    environment_key)
from .prefix_replay import PrefixHttpClient
from .qwen35_native import NativeQwen35Adapter
from .successor_prefetch import cancel_and_drain, run_successor_pair
from .timed_retry import run_timed_retry_turn
from .turn_runner import RetryTurnConfig

from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
from tau2.utils.llm_utils import to_litellm_messages

CONFIGS = ("spork", "raw", "raw_no_probe", "raw_one_probe")
USER_PRESENCE_PENALTY = 1.5
USER_THINKING = False
USER_SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20,
                 "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0}


class UserHttpClient(PrefixHttpClient):
    """Declared Qwen3.5 non-thinking user sampling, separate from the Actor."""
    def body(self, prompt_ids, **kwargs):
        return {**super().body(prompt_ids, **kwargs), **USER_SAMPLING}


def validate_user_completion(choice, *, thinking):
    if choice["finish_reason"] != "stop":
        raise RuntimeError("User generation truncated")
    if thinking and "</think>" not in choice["text"]:
        raise RuntimeError("User reasoning was not closed")
    if not thinking and "<think>" in choice["text"]:
        raise RuntimeError("Non-thinking user emitted a reasoning block")
    text = choice["text"].split("</think>")[-1].strip()
    if not text or "<tool_call>" in text:
        raise RuntimeError("Invalid user response")
    return text


class InteractiveRuntime(LocalRuntime):
    def __init__(self, base_url, model, presence_penalty=1.5):
        super().__init__(base_url, model)
        self.client.presence_penalty = self.http.presence_penalty = presence_penalty
        self.user_http = UserHttpClient(base_url, model, 900,
                                         presence_penalty=USER_PRESENCE_PENALTY)
        # The simulator's repetition control must not silently change scoring.
        self.judge_http = PrefixHttpClient(base_url, model, 900, presence_penalty=0.)

    async def _open(self):
        return aiohttp.ClientSession(trust_env=False,
                                     connector=aiohttp.TCPConnector(force_close=True))


class SuccessorInteractiveAgent(InteractiveAgent):
    def __init__(self, env, runtime, tokenizer, method, **kwargs):
        if method not in CONFIGS:
            raise ValueError(method)
        super().__init__(env, runtime, tokenizer, method, **kwargs)
        self.adapter = NativeQwen35Adapter(tokenizer, [t.openai_schema for t in self.tools])
        self._task = None
        self._events = []
        self._changed = None
        self._first_turn = None
        self.prefetch = {"pairs": []}
        self.close_wall_s = 0.
        self.cleanup_errors = []
        self.abandoned_entries = []

    def make_cfg(self, state, turn):
        runtime = self.runtime
        rid = "interactive-successor-" + uuid.uuid4().hex
        prompt = self.adapter.render_main_prompt(state, enable_thinking=True)
        plen = len(self.tokenizer.encode(prompt, add_special_tokens=False))

        async def register(call, _prompt):
            ids = self.tokenizer.encode(self.adapter.draft_body(call), add_special_tokens=False)
            async with runtime.session.post(runtime.http.root + "/spork/set_tokens",
                    json={"request_id": rid, "prompt_len": plen, "draft_tokens": ids},
                    timeout=30) as response:
                response.raise_for_status()
                return await response.json()

        async def clear():
            result = await runtime.http.clear(runtime.session)
            return {**await runtime.http.status(runtime.session), "clear_status": result["status"]}

        return RetryTurnConfig(max_retries=5, confidence_threshold=.9, retry_token_step=50,
            probe_max_tokens=160, baseline_max_tokens=self.max_tokens, context_window_tokens=40960,
            enable_thinking=True, seed=self.seed, d3_set=register, d3_clear=clear,
            main_request_id=rid, launch_mode="original")

    def _pair_allowed(self):
        # A native agent decision and its tool response consume separate steps.
        # Close to a cap, run only the current canonical turn.
        orch = self.orchestrator
        return (orch.max_steps - orch.step_count >= 4
                and orch.num_errors + 1 < orch.max_errors)

    async def _start(self, state):
        if self._task is not None:
            if not self._task.done():
                raise RuntimeError("Previous pair is still running")
            self._task.result()
        if self._events:
            raise RuntimeError("Previous pair still has undelivered entries")
        self._changed = asyncio.Event()
        self._first_turn = len(self.turns) + 1
        work_state = copy.deepcopy(state)
        executor = NativeToolExecutor(copy.deepcopy(self.orchestrator.environment), self.floor_s)
        before = environment_key(executor.env)
        pair = {"first_turn": self._first_turn, "native_delivery_count": 0}
        self.prefetch["pairs"].append(pair)

        def capture(entry):
            nonlocal before
            self._events.append({"entry": copy.deepcopy(entry), "state": copy.deepcopy(work_state),
                "environment": copy.deepcopy(executor.env), "before": before,
                "response": copy.deepcopy(executor.last_response), "pair": pair})
            before = environment_key(executor.env)
            self._changed.set()

        async def work():
            if self._pair_allowed():
                pair["kind"] = "successor_pair"
                await run_successor_pair(self.runtime.session, self.runtime.client,
                    self.adapter, work_state, executor, self._first_turn, self.make_cfg,
                    mode="baseline" if self.method == "spork" else self.method,
                    on_turn=capture, trace=pair)
            else:
                pair.update(kind="native_budget_single", child_started=False, adopted=False)
                entry = await run_timed_retry_turn(self.runtime.session, self.runtime.client,
                    self.adapter, work_state, executor, self._first_turn,
                    self.make_cfg(work_state, self._first_turn))
                capture(entry)

        self._task = asyncio.create_task(work(), name="native-successor-pair")

    async def _next(self, state):
        if self._task is None or (self._task.done() and not self._events):
            await self._start(state)
        while not self._events:
            if self._task.done():
                self._task.result()
                raise RuntimeError("Pair completed without an Actor entry")
            waiter = asyncio.create_task(self._changed.wait())
            try:
                await asyncio.wait([self._task, waiter], return_when=asyncio.FIRST_COMPLETED)
            finally:
                if not waiter.done():
                    waiter.cancel()
                await asyncio.gather(waiter, return_exceptions=True)
        event = self._events.pop(0)
        if not self._events:
            self._changed.clear()
        entry = event["entry"]
        # Finish cleanup before a user response or the end of the pair. The
        # first tool call may be handed off while its isolated child is live.
        if (not entry["baseline_has_tool_call"] or entry["turn"] > self._first_turn
                or event["pair"]["kind"] == "native_budget_single"):
            await self._task
        return event

    def generate_next_message(self, message, state):
        started = time.perf_counter()
        if self.pending is not None:
            raise RuntimeError("Previous tool call has not been committed")
        if isinstance(message, ToolMessage):
            if self.expected_reply is None or message != self.expected_reply:
                raise RuntimeError("Native tool reply differs from committed result")
            self.expected_reply = None
        elif isinstance(message, UserMessage) and not message.is_tool_call():
            if self.expected_reply is not None or self._events or (self._task and not self._task.done()):
                raise RuntimeError("A speculative child reached a user boundary")
            state.append({"role": "user", "content": message.content})
        else:
            raise ValueError("Only text users and single native tool replies are supported")
        event = self.runtime.run(self._next(state))
        entry = event["entry"]
        actual_ids = self.tokenizer.encode(
            self.adapter.render_main_prompt(state, enable_thinking=True), add_special_tokens=False)
        if (entry["turn"] != len(self.turns) + 1
                or entry["prompt_token_ids"] != actual_ids
                or event["before"] != environment_key(self.orchestrator.environment)):
            raise RuntimeError("Native handoff changed the complete prompt or starting state")
        if (not entry.get("completion_valid") or entry.get("worker_errors")
                or entry.get("spec_cleanup_error")
                or (entry.get("d3_status") or {}).get("clear_status") not in ("ok", "cleared")):
            raise RuntimeError("Actor completion or D3 cleanup failed")
        new_state = event["state"]
        if entry["baseline_has_tool_call"]:
            call = ToolCall(id="call-" + uuid.uuid4().hex, requestor="assistant",
                name=entry["baseline_tool_name"], arguments=entry["baseline_tool_args"])
            response = event["response"]
            if response is None or hashlib.sha256(response.content.encode()).hexdigest() != entry["tool_result_sha256"]:
                raise RuntimeError("Missing native response for formal handoff")
            self.pending = (call, response, event["environment"], event["before"])
            output = AssistantMessage(role="assistant", content=None, tool_calls=[call])
        else:
            visible = entry["baseline_text"].split("</think>")[-1].strip()
            if not visible:
                raise RuntimeError("Empty visible reply")
            new_state.append({"role": "assistant", "content": visible})
            output = AssistantMessage(role="assistant", content=visible)
        entry["actor_step_wall_s"] = time.perf_counter() - started
        event["pair"]["native_delivery_count"] += 1
        self.turns.append(entry)
        self.on_turn(entry)
        return output, new_state

    async def _close(self):
        task = self._task
        if task is not None:
            await cancel_and_drain([task])
        for event in self._events:
            entry = event["entry"]
            self.abandoned_entries.append({"turn": entry["turn"],
                "received_tokens": len(entry.get("main_token_ids", [])),
                "environment_after": entry.get("environment_after")})
        self._events.clear()
        self._task = None

    def close(self):
        started = time.perf_counter()
        try:
            self.runtime.run(self._close())
        except Exception as exc:
            self.cleanup_errors.append(f"{type(exc).__name__}: {exc}")
            raise
        finally:
            self.close_wall_s += time.perf_counter() - started

    def stop(self, last_message=None, state=None):
        self.close()


class TracedLocalUser(LocalUserSimulator):
    """Native user instructions/role flipping, with full request provenance."""
    def prompt_token_ids(self, state):
        messages = to_litellm_messages(state.system_messages + state.flip_roles())
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=self.thinking)
        return self.tokenizer.encode(prompt, add_special_tokens=False)

    def _generate_next_message(self, message, state):
        started = time.perf_counter()
        if not isinstance(message, AssistantMessage) or message.is_tool_call():
            raise ValueError("Only text airline/retail users are supported")
        if message.has_content():
            state.messages.append(message)
        ids = self.prompt_token_ids(state)
        if len(ids) + self.max_tokens > 40960:
            raise ValueError("User context budget exceeded")
        http = self.runtime.user_http
        settings = {k: v for k, v in http.body(ids, max_tokens=self.max_tokens,
            seed=self.seed, stop=["<|im_end|>"]).items() if k not in {"prompt", "request_id"}}
        choice = {"prompt_token_ids": ids, "generation_settings": settings,
                  "text": "", "token_ids": [], "finish_reason": None}
        self.calls.append(choice)
        try:
            result = self.runtime.run(http.complete(self.runtime.session, ids,
                max_tokens=self.max_tokens, seed=self.seed, stop=["<|im_end|>"]))
            choice.update(result)
            choice.update(prompt_token_ids=ids, generation_settings=settings)
        except BaseException as exc:
            choice["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            choice["participant_wall_s"] = time.perf_counter() - started
        text = validate_user_completion(choice, thinking=self.thinking)
        return UserMessage(role="user", content=text)
