"""Two-turn SPORK pipeline with full-Actor successor inference and exact reuse.

The parent Actor always completes normally. A child executes on an independent
copy of the speculative post-parent environment. Its results are adopted only
after the actual parent action, observation, state and entire child prompt agree.
There is no training and no future trace input. Concurrency determinism still
requires empirical verification on the serving backend.
"""
from __future__ import annotations

import asyncio
import copy
from dataclasses import replace
import hashlib
import json
import time

from .candidate_verification import decision_prefix, decision_probe
from .executors import Tau2ToolExecutor
from .deferred_tools import DeferredTools
from .prefix_replay import PrefixHttpClient, formal_action_started, same_call
from .timed_retry import ActorGenerationError, run_timed_retry_turn
from .turn_runner import _append_tool_messages


def state_key(env):
    return {"agent_db": env.get_db_hash(), "user_db": env.get_user_db_hash()}


def generation_key(cfg, client, adapter):
    return {"model": client.model, "base_url": client.base_url, "seed": cfg.seed,
            "max_tokens": cfg.baseline_max_tokens, "context_window": cfg.context_window_tokens,
            "thinking": cfg.enable_thinking, "stop": list(adapter.main_stop),
            "temperature": 0.0, "presence_penalty": getattr(client, "presence_penalty", 0.0)}


def rejection_reason(ticket, parent, env, prompt_ids, settings):
    actual_call = {"name": parent.get("baseline_tool_name"),
                   "arguments": parent.get("baseline_tool_args")}
    if not parent.get("baseline_has_tool_call") or not same_call(actual_call, ticket["parent_call"]):
        return "parent_call"
    if parent.get("tool_result_sha256") != ticket["observation_sha256"]:
        return "parent_observation"
    if state_key(env) != ticket["parent_state"]:
        return "parent_state"
    if list(prompt_ids) != ticket["prompt_ids"]:
        return "next_prompt_tokens"
    if settings != ticket["generation"]:
        return "generation_settings"
    return None


class TrackedHTTP(PrefixHttpClient):
    def __init__(self, client):
        super().__init__(client.base_url, client.model, client.timeout_s,
                         presence_penalty=getattr(client, "presence_penalty", 0.0))
        self.state = None
        self.prompt_ids = None
        self.calls = []

    async def stream(self, session, prompt_ids, state, **kwargs):
        self.state, self.prompt_ids = state, list(prompt_ids)
        return await super().stream(session, prompt_ids, state, **kwargs)

    async def complete(self, session, prompt_ids, **kwargs):
        call = {"prompt_tokens": len(prompt_ids), "budget": kwargs.get("max_tokens")}
        self.calls.append(call)
        started = time.perf_counter()
        try:
            result = await super().complete(session, prompt_ids, **kwargs)
            call.update(usage=result.get("usage"), finish_reason=result["finish_reason"])
            return result
        finally:
            call["wall_s"] = time.perf_counter() - started


class ResultObserver:
    """Delegate tool semantics; notify the event loop after isolated work ends."""
    def __init__(self, executor, loop, callback):
        self.executor, self.loop, self.callback = executor, loop, callback

    def __getattr__(self, key):
        return getattr(self.executor, key)

    def execute_speculative(self, call):
        result = self.executor.execute_speculative(call)
        self.loop.call_soon_threadsafe(self.callback, copy.deepcopy(call), result)
        return result


async def cancel_and_drain(tasks):
    """Do not abandon a child runner while it is draining real tool threads."""
    tasks = [t for t in tasks if t is not None]
    for task in tasks:
        if not task.done():
            task.cancel()
    if not tasks:
        return
    async def drain():
        return await asyncio.gather(*tasks, return_exceptions=True)
    cleanup = asyncio.create_task(drain())
    interrupted = False
    while not cleanup.done():
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            interrupted = True
    cleanup.result()
    if interrupted:
        raise asyncio.CancelledError


async def run_successor_pair(session, client, adapter, messages, executor, first_turn,
                             make_cfg, *, mode="confirmed", on_turn=None, trace=None,
                             recover_child_failure=False):
    """Run at most two canonical turns; measure elapsed pair time externally.

    Modes: baseline (no prefetch), raw (result-ready prefetch), confirmed (one
    unforced call confirmation before spending a full successor request).
    Both speculative tool execution and cancellation cost remain inside the pair.
    """
    if mode not in {"baseline", "raw", "confirmed", "serial", "actor_only", "raw_no_probe", "raw_one_probe"}:
        raise ValueError(mode)
    on_turn = on_turn or (lambda _: None)
    started = time.perf_counter()
    elapsed = lambda: time.perf_counter() - started
    before_messages = copy.deepcopy(messages)
    parent_http, child_http = TrackedHTTP(client), TrackedHTTP(client)
    confirmation_http = TrackedHTTP(client)
    ticket = {}
    info = trace if trace is not None else {}
    info.update(mode=mode, candidate_received=False, child_started=False,
                adopted=False, admission=None, rejection=None, errors=[])
    parent_task = admission_task = child_task = None
    deferred_tools = None
    closed = False
    entries = []

    def launch_child(phase):
        nonlocal child_task, deferred_tools
        child_executor = ticket["executor"]
        if mode == "actor_only":
            deferred_tools = DeferredTools(child_executor)
            child_executor = deferred_tools
        info.update(child_started=True, child_start_s=elapsed(), admission="launched",
                    child_launch_phase=phase)
        child_task = asyncio.create_task(run_timed_retry_turn(
            session, client, adapter, ticket["messages"], child_executor,
            first_turn + 1, ticket["cfg"], http=child_http))

    def parent_still_reasoning():
        state = parent_http.state
        return (state is not None and state.finished_at is None
                and "</think>" not in state.text and not formal_action_started(state.text)
                and parent_task is not None and not parent_task.done() and not parent_task.cancelling())

    async def admit(call, execution):
        nonlocal child_task
        info["candidate_received"] = True
        info["candidate_ready_s"] = elapsed()
        info["candidate_call"] = call
        try:
            if execution.state is None or not parent_still_reasoning():
                info["admission"] = "parent_already_finished_or_no_isolated_state"
                return
            if mode == "confirmed":
                state = parent_http.state
                ids, text = list(state.token_ids), state.text
                cfg = make_cfg(before_messages, first_turn)
                prefix = decision_prefix(adapter.tokenizer, parent_http.prompt_ids, ids, text)
                check = await decision_probe(session, confirmation_http, adapter.tokenizer, prefix,
                    use_d3=False, threshold=.9, max_tokens=cfg.probe_max_tokens, seed=cfg.seed)
                check["visible_parent_tokens"] = len(ids)
                info["confirmation"] = check
                confirmed_call = adapter.parse_main_tool_call(check.get("call_text") or "")
                if check.get("finish_reason") != "stop" or not same_call(confirmed_call, call):
                    info["admission"] = "confirmation_did_not_match"
                    return
            if not parent_still_reasoning():
                info["admission"] = "parent_finished_during_confirmation"
                return
            child_messages = copy.deepcopy(before_messages)
            text = (adapter.speculative_tool_message(call)
                    if hasattr(adapter, "speculative_tool_message") else
                    "<tool_call>\n" + json.dumps(call, ensure_ascii=False) + "\n")
            _append_tool_messages(child_messages, text, execution.text)
            prompt = adapter.render_main_prompt(child_messages, enable_thinking=True)
            prompt_ids = adapter.tokenizer.encode(prompt, add_special_tokens=False)
            # Child mutations must not affect the candidate parent environment.
            child_executor = executor.fork_at(execution.state)
            cfg = make_cfg(child_messages, first_turn + 1)
            ticket.update(parent_call=call,
                observation_sha256=hashlib.sha256(execution.text.encode()).hexdigest(),
                parent_state=state_key(execution.state), prompt_ids=prompt_ids,
                generation=generation_key(cfg, client, adapter), messages=child_messages,
                executor=child_executor)
            # The current D3 clear is server-wide. An uncommitted child uses
            # ordinary Target generation and D1/D2 without its own D3 state.
            cfg = replace(cfg, d3_set=None, d3_clear=None)
            if mode == "raw_no_probe":
                # Keep full Target inference and formal isolated tool execution;
                # remove only the child's additional D1/D2 prediction requests.
                cfg = replace(cfg, launch_mode="off", max_retries=0)
            elif mode == "raw_one_probe":
                cfg = replace(cfg, max_retries=1)
            info["child_prediction_policy"] = cfg.launch_mode
            info["child_prediction_budget"] = cfg.max_retries
            ticket["cfg"] = cfg
            info.update(child_prompt_sha256=hashlib.sha256(json.dumps(prompt_ids).encode()).hexdigest(),
                        candidate_parent_state=ticket["parent_state"],
                        candidate_observation_sha256=ticket["observation_sha256"])
            if mode == "serial":
                info["admission"] = "deferred_until_parent_validation"
            else:
                launch_child("before_parent_completion")
        except Exception as exc:
            info["admission"] = "error"
            info["errors"].append(f"admission: {type(exc).__name__}: {exc}")
        finally:
            info["admission_end_s"] = elapsed()

    def candidate_ready(call, execution):
        nonlocal admission_task
        if not closed and admission_task is None and mode != "baseline":
            admission_task = asyncio.create_task(admit(call, execution))

    try:
        cfg = make_cfg(messages, first_turn)
        parent_executor = (executor if mode == "baseline" else
                           ResultObserver(executor, asyncio.get_running_loop(), candidate_ready))
        parent_task = asyncio.create_task(run_timed_retry_turn(session, client, adapter,
            messages, parent_executor, first_turn, cfg, http=parent_http))
        parent = await parent_task
        info["parent_finished_s"] = elapsed()
        entries.append(parent)
        on_turn(parent)
        # No overlap remains if confirmation has not launched the child yet.
        if admission_task is not None and not admission_task.done():
            info["admission"] = "parent_finished_before_child_launch"
            await cancel_and_drain([admission_task])
        elif admission_task is not None:
            await admission_task
        if not parent["baseline_has_tool_call"]:
            info["rejection"] = "parent_has_no_tool"
            return entries, info
        next_started = time.perf_counter()
        next_cfg = make_cfg(messages, first_turn + 1)
        prompt = adapter.render_main_prompt(messages, enable_thinking=True)
        actual_ids = adapter.tokenizer.encode(prompt, add_special_tokens=False)
        has_candidate = (child_task is not None or
                         (mode == "serial" and info["admission"] == "deferred_until_parent_validation"))
        reason = (rejection_reason(ticket, parent, executor.env, actual_ids,
                    generation_key(next_cfg, client, adapter)) if has_candidate else "not_launched")
        info["rejection"] = reason
        if reason is None:
            if mode == "serial":
                launch_child("after_parent_validation")
            if deferred_tools is not None:
                deferred_tools.allow()
            try:
                child = await child_task
            except ActorGenerationError as exc:
                if not recover_child_failure:
                    raise
                # A failed isolated branch must never replace committed state.
                # CancelledError intentionally propagates instead of retrying.
                info.update(rejection="child_failure", recovery_attempted=True,
                            recovery_succeeded=False,
                            recovery_reason=f"{type(exc).__name__}: {exc}")
                if getattr(exc, "partial_generation", None) is not None:
                    info["failed_child_generation"] = exc.partial_generation
                if deferred_tools is not None:
                    deferred_tools.discard()
                await cancel_and_drain([child_task])
                info["rejection_cleanup_finished_s"] = elapsed()
                child = await run_timed_retry_turn(session, client, adapter,
                    messages, executor, first_turn + 1, next_cfg)
                info["recovery_succeeded"] = True
            else:
                # Verify again at commit; awaiting the child must not expose it to
                # any mutation of the real parent environment or conversation.
                if rejection_reason(ticket, parent, executor.env,
                        adapter.tokenizer.encode(adapter.render_main_prompt(messages, enable_thinking=True), add_special_tokens=False),
                        generation_key(next_cfg, client, adapter)) is not None:
                    raise RuntimeError("Reuse contract changed while awaiting child")
                executor.adopt_successor(ticket["executor"])
                messages[:] = ticket["messages"]
                info["adopted"] = True
                child["prefetch_compute_turn_end_ms"] = child["turn_end_ms"]
                if next_cfg.d3_clear is not None:
                    child["d3_status"] = await next_cfg.d3_clear()
                child["turn_end_ms"] = (time.perf_counter() - next_started) * 1000
                child["timing_scope"] = "Canonical child wait/adoption; original compute duration retained separately"
                child["prefetch_d3_disabled"] = True
        else:
            if deferred_tools is not None:
                deferred_tools.discard()
            await cancel_and_drain([child_task])
            info["rejection_cleanup_finished_s"] = elapsed()
            child = await run_timed_retry_turn(session, client, adapter, messages, executor,
                                               first_turn + 1, next_cfg)
        entries.append(child)
        on_turn(child)
        return entries, info
    finally:
        closed = True
        if deferred_tools is not None and not info["adopted"]:
            # Release blocked real worker threads before waiting for cancellation.
            deferred_tools.discard()
        await cancel_and_drain([parent_task, admission_task, child_task])
        if deferred_tools is not None:
            info["deferred_tools"] = {"wait_s": deferred_tools.wait_s,
                                     "blocked_calls": deferred_tools.blocked_calls,
                                     "executed_calls": deferred_tools.executed_calls}
        state = child_http.state
        info.update(pair_end_s=elapsed(),
            observed_child_tokens=len(state.token_ids) if state else 0,
            child_finish_reason=state.finish_reason if state else None,
            observed_wasted_child_tokens=(len(state.token_ids) if state and not info["adopted"] else 0),
            token_accounting_scope="Received stream tokens; cancelled GPU work can exceed this lower bound",
            parent_prediction_requests=parent_http.calls,
            confirmation_requests=confirmation_http.calls,
            child_prediction_requests=child_http.calls)
        if state and not info["adopted"]:
            # Kept for offline comparison with the turn that replaced this child.
            info["discarded_child_token_ids"] = list(state.token_ids)
        if child_task is not None and child_task.done() and not child_task.cancelled():
            error = child_task.exception()
            if error is not None:
                info["child_error"] = f"{type(error).__name__}: {error}"
            else:
                child_result = child_task.result()
                info["child_output"] = {k: child_result.get(k) for k in (
                    "baseline_has_tool_call", "baseline_tool_name", "baseline_tool_args",
                    "tool_result_sha256", "environment_after", "completion_valid")}
        if ticket.get("executor") is not None:
            # Also covers a tool that finished while its child runner was
            # cancelled. This is the isolated state, never a second commit.
            info["child_environment_observed"] = state_key(ticket["executor"].env)
        info["pair_end_s"] = elapsed()
