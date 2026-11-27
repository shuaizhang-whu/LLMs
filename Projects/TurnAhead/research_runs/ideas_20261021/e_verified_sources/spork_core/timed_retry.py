"""Live first-launch timing, with the collection-time D2/D3 policy.

The legacy SPORK runner dispatches here only for an explicit launch_mode.
All modes share this transport, retry rule, strict tool gate and cleanup.
Prediction requests deliberately have no automatic HTTP retries, matching
prefix_replay. A late prediction is still awaited, as it was during label
collection; its delay must not be misreported as Actor generation time.
"""
from __future__ import annotations

import asyncio
import hashlib
import time
from typing import TYPE_CHECKING

from .executors import commit_speculative, execute_speculative
from .launch_timing import choose_first_launch, validate_launch_settings
from .prefix_replay import (
    PrefixHttpClient, StreamState, formal_action_started, prediction_prompt_ids,
    same_call, wait_for_tokens,
)
from .tool_calls import args_exact_match

if TYPE_CHECKING:
    from .turn_runner import RetryTurnConfig


class _TurnResources:
    """Own every background task until the real work has stopped.

    Cancelling an asyncio wrapper cannot stop a tool/BERT worker thread.
    Await thread tasks through shield, then drain them in __aexit__, even on
    failure/cancellation. HTTP streaming can be cancelled and closed normally.
    """

    def __init__(self, clear):
        self.main_task = None
        self.workers = []
        self.clear = clear
        self.d3_status = None
        self.cleanup_ms = 0.0
        self.tool_cleanup_ms = 0.0
        self.worker_errors = []

    async def __aenter__(self):
        return self

    def start_worker(self, kind, func, *args):
        task = asyncio.create_task(asyncio.to_thread(func, *args), name=f"spork-{kind}")
        self.workers.append((kind, task))
        return task

    async def run_controller(self, func, *args):
        return await asyncio.shield(self.start_worker("controller", func, *args))

    async def _close(self):
        start = time.perf_counter()
        if self.main_task is not None:
            if not self.main_task.done():
                self.main_task.cancel()
            await asyncio.gather(self.main_task, return_exceptions=True)
        tool_start = time.perf_counter()
        if self.workers:
            outcomes = await asyncio.gather(*(t for _, t in self.workers), return_exceptions=True)
            self.worker_errors = [f"{kind}: {type(out).__name__}: {out}"
                                  for (kind, _), out in zip(self.workers, outcomes)
                                  if isinstance(out, BaseException)]
        self.tool_cleanup_ms = (time.perf_counter() - tool_start) * 1000
        # No Actor or tool from this turn may survive into the next turn.
        # The sidecar clear is server-wide; callers must use a dedicated server.
        if self.clear is not None:
            self.d3_status = await self.clear()
            if (not isinstance(self.d3_status, dict)
                    or self.d3_status.get("error")
                    or self.d3_status.get("clear_status", self.d3_status.get("status"))
                    not in ("ok", "cleared")):
                raise RuntimeError(f"D3 cleanup failed: {self.d3_status}")
        self.cleanup_ms = (time.perf_counter() - start) * 1000

    async def __aexit__(self, exc_type, exc, traceback):
        cleanup = asyncio.create_task(self._close(), name="spork-turn-cleanup")
        interrupted = False
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                # A second cancellation must not abandon real worker threads.
                interrupted = True
            except Exception:
                break
        try:
            cleanup.result()
        except Exception as cleanup_error:
            if exc is None:
                raise
            exc.add_note(f"Also failed to clean up SPORK: {cleanup_error}")
        if interrupted and exc is None:
            raise asyncio.CancelledError
        return False


def _validate(adapter, cfg):
    validate_launch_settings(cfg.launch_mode, cfg.launch_controller,
                             cfg.controller_prefix_tokens, cfg.controller_wait_tokens)
    if (getattr(adapter, "uses_chat_api", False)
            or adapter.name != "qwen3_tool_call" or not cfg.enable_thinking):
        raise ValueError("Launch timing requires the Qwen3 completion adapter with thinking enabled")
    if (cfg.continue_after_dispatch or cfg.hybrid_name_loose_first
            or cfg.min_tokens_first_probe or cfg.theta_decay_per_retry):
        raise ValueError("Launch timing uses the collected D2 policy: fixed confidence, "
                         "one dispatched tool, and no additional first-launch delay")
    if (cfg.max_retries < 0 or cfg.retry_token_step <= 0 or cfg.probe_max_tokens <= 0
            or cfg.baseline_max_tokens <= 0 or cfg.context_window_tokens <= 0
            or not 0 <= cfg.confidence_threshold <= 1 or cfg.probe_timeout_s < 0):
        raise ValueError("Invalid timing runner budgets or confidence threshold")


async def _predict(session, http, adapter, prompt_ids, visible_ids, visible_text, cfg):
    ids = prediction_prompt_ids(adapter.tokenizer, prompt_ids, visible_ids, visible_text)
    budget = min(cfg.probe_max_tokens, cfg.context_window_tokens - len(ids))
    if budget <= 0:
        raise ValueError("The prediction prompt fills the context window")
    request = http.complete(session, ids, max_tokens=budget, stop=adapter.probe_stop,
                            logprobs=5, seed=cfg.seed)
    if cfg.probe_timeout_s > 0:
        choice = await asyncio.wait_for(request, timeout=cfg.probe_timeout_s)
    else:
        choice = await request
    # Match collection-time parsing for prediction truncation. These candidates
    # can only be committed against a separately completed, exact Actor call.
    logprobs = adapter.extract_probe_logprobs(choice.get("logprobs"))
    call = adapter.parse_probe_tool_call(choice.get("text", ""))
    return call, adapter.span_confidence(logprobs, skip_first=True), logprobs


async def run_timed_retry_turn(session, client, adapter, messages, executor, turn,
                              cfg: RetryTurnConfig):
    """Execute one live turn; only the first launch decision is learned."""
    from .turn_runner import _append_tool_messages, _fork_point

    _validate(adapter, cfg)  # Fail before starting any Actor request.
    start = time.perf_counter()

    def elapsed_ms(at=None):
        return ((time.perf_counter() if at is None else at) - start) * 1000

    prompt = adapter.render_main_prompt(messages, enable_thinking=True)
    prompt_ids = adapter.tokenizer.encode(prompt, add_special_tokens=False)
    budget = min(cfg.baseline_max_tokens, cfg.context_window_tokens - len(prompt_ids))
    if budget <= 0:
        raise ValueError("The Actor prompt fills the context window")
    http = PrefixHttpClient(client.base_url, client.model, client.timeout_s)
    state = StreamState()
    probes = []
    spec_task = spec_call = committed_retry = None
    spec_dispatch_ms = spec_finished_ms = None
    d3_count = 0

    async with _TurnResources(cfg.d3_clear) as resources:
        resources.main_task = asyncio.create_task(http.stream(
            session, prompt_ids, state, max_tokens=budget, seed=cfg.seed,
            stop=adapter.main_stop, request_id=cfg.main_request_id,
        ), name="spork-actor")
        main = resources.main_task
        first_text, first_count, trace = await choose_first_launch(
            state, main, adapter.tokenizer, messages, adapter.tools,
            mode=cfg.launch_mode, controller=cfg.launch_controller,
            prefix_tokens=cfg.controller_prefix_tokens, wait_tokens=cfg.controller_wait_tokens,
            run_controller=resources.run_controller,
        )
        last_count = 0
        for retry in range(cfg.max_retries if first_text is not None else 0):
            if retry and not await wait_for_tokens(
                    state, main, last_count + cfg.retry_token_step, snap=cfg.snap_to_sentence):
                break
            if main.done() or state.finished_at is not None or formal_action_started(state.text):
                break
            text = first_text if retry == 0 else state.text
            count = first_count if retry == 0 else len(state.token_ids)
            visible_ids = list(state.token_ids[:count])
            last_count = len(state.token_ids)
            dispatch = elapsed_ms()
            p = {"retry_idx": retry, "tokens_at_dispatch": count,
                 "main_tokens_available_at_dispatch": last_count,
                 "observed_prefix_len": len(text), "probe_dispatch_ms": dispatch,
                 "effective_theta": cfg.confidence_threshold, "dispatched": False,
                 "probe_error": None}
            probes.append(p)
            call = confidence = None
            logprobs = {}
            try:
                call, confidence, logprobs = await _predict(
                    session, http, adapter, prompt_ids, visible_ids, text, cfg)
            except Exception as error:
                p["probe_error"] = f"{type(error).__name__}: {error}"
            p.update(probe_end_ms=elapsed_ms(), confidence=confidence,
                     probe_wall_s=(elapsed_ms() - dispatch) / 1000,
                     probe_tool_name=call["name"] if call else None,
                     probe_tool_args=call.get("arguments") if call else None,
                     probe_parse_ok=call is not None, probe_logprobs=logprobs)
            if call is not None and cfg.d3_set is not None and not main.done():
                try:
                    info = await cfg.d3_set(call, prompt)
                    p["d3_set"] = info
                    if not isinstance(info, dict) or info.get("status") != "ok" or info.get("error"):
                        raise RuntimeError(f"D3 registration failed: {info}")
                    d3_count += 1
                except Exception as error:
                    p["d3_set_error"] = f"{type(error).__name__}: {error}"
            # Do not dispatch a new speculative tool after a failed/truncated Actor.
            if main.done():
                await main
                if state.finish_reason != "stop":
                    break
            if call is not None and confidence is not None and confidence >= cfg.confidence_threshold:
                spec_call, committed_retry = call, retry
                spec_dispatch_ms = elapsed_ms()
                p["dispatched"] = True

                def execute_candidate(call=call):
                    nonlocal spec_finished_ms
                    try:
                        return execute_speculative(executor, call)
                    finally:
                        spec_finished_ms = elapsed_ms()

                spec_task = resources.start_worker("speculative_tool", execute_candidate)
                break

        await main
        if state.finish_reason != "stop":
            raise RuntimeError(f"Incomplete Actor generation: finish_reason={state.finish_reason!r}")
        main_observed_ms = elapsed_ms()
        main_end_ms = elapsed_ms(state.finished_at)
        main_call = adapter.parse_main_tool_call(state.text)
        if formal_action_started(state.text) and main_call is None:
            raise RuntimeError("Actor produced an unparsable tool call")
        has_tool = main_call is not None
        gate_match = same_call(main_call, spec_call)
        spec_used = False
        spec_wall_s = tool_real_wall_s = overlap_ms = 0.0
        tool_result = ""
        canonical_dispatch_ms = canonical_end_ms = None
        if gate_match:
            execution = await asyncio.shield(spec_task)
            commit_speculative(executor, execution)
            tool_result = execution.text
            spec_wall_s, tool_real_wall_s = execution.wall_s, execution.real_wall_s
            spec_used = True
            overlap_ms = min(spec_wall_s * 1000, max(0, main_end_ms - spec_dispatch_ms))
        elif has_tool:
            canonical_dispatch_ms = elapsed_ms()
            tool_result, _ = await asyncio.shield(resources.start_worker(
                "canonical_tool", executor.execute, main_call))
            tool_real_wall_s = executor.last_real_wall_s
            canonical_end_ms = elapsed_ms()
        ready_ms = elapsed_ms()
        if has_tool:
            _append_tool_messages(messages, state.text, tool_result)
        hit_name = bool(spec_call and main_call and spec_call.get("name") == main_call.get("name"))
        entry = {
            "turn": turn, "fork_point": _fork_point(turn), "mode": "spork_retry",
            "gate_variant": "args_exact_strict", "max_retries": cfg.max_retries,
            "confidence_threshold": cfg.confidence_threshold, "continue_after_dispatch": False,
            "main_max_tokens": budget,
            "main_first_token_ms": elapsed_ms(state.first_output_at) if state.first_output_at else None,
            "main_end_ms": main_end_ms, "main_result_observed_ms": main_observed_ms,
            "main_result_wait_ms": max(0, main_observed_ms - main_end_ms),
            "tool_result_ready_ms": ready_ms, "first_token_timeout": False,
            "baseline_has_tool_call": has_tool,
            "baseline_tool_name": main_call["name"] if has_tool else None,
            "baseline_tool_args": main_call.get("arguments") if has_tool else None,
            "baseline_text": state.text, **adapter.token_metrics(state.text),
            "num_probes": len(probes), "committed_retry_idx": committed_retry,
            "committed_probe_name": spec_call["name"] if spec_call else None,
            "committed_probe_args": spec_call.get("arguments") if spec_call else None,
            "hit_name": hit_name,
            "hit_args_exact": bool(hit_name and args_exact_match(spec_call.get("arguments"), main_call.get("arguments"))),
            "gate_match": gate_match, "spec_dispatched": spec_task is not None,
            "spec_used": spec_used, "spec_wasted": spec_task is not None and not spec_used,
            "superseded_specs": 0, "d3_set_count": d3_count,
            "spec_dispatch_ms": spec_dispatch_ms, "spec_wall_s": spec_wall_s,
            "canonical_tool_dispatch_ms": canonical_dispatch_ms,
            "canonical_tool_end_ms": canonical_end_ms,
            "real_tool_wall_s": tool_real_wall_s, "real_overlap_ms": overlap_ms,
            "probes": probes, "launch_timing": trace,
            "main_finish_reason": state.finish_reason, "completion_valid": True,
            "main_token_ids": list(state.token_ids),
            "tool_result_sha256": hashlib.sha256(tool_result.encode()).hexdigest() if has_tool else None,
            "timing_scope": "live turn; monotonic client clock; turn_end includes cleanup",
        }
        if has_tool:
            entry.update(tool_result_len=len(tool_result), tool_result_preview=adapter.preview(tool_result))
        else:
            entry["final_answer"] = state.text.strip()
        if hasattr(executor, "env"):
            entry["environment_after"] = {"agent_db": executor.env.get_db_hash(),
                                          "user_db": executor.env.get_user_db_hash()}

    entry.update(turn_end_ms=elapsed_ms(), spec_end_ms=spec_finished_ms,
                 spec_cleanup_ms=resources.tool_cleanup_ms, cleanup_ms=resources.cleanup_ms,
                 d3_status=resources.d3_status)
    if spec_task is not None and not spec_task.cancelled() and spec_task.exception() is None:
        # Misses still consumed real resources. Record their cost even though
        # the formal result came from the canonical executor.
        execution = spec_task.result()
        entry.update(spec_wall_s=execution.wall_s,
                     spec_real_tool_wall_s=execution.real_wall_s,
                     spec_artificial_sleep_s=execution.artificial_sleep_s)
    tool_errors = [e for e in resources.worker_errors if not e.startswith("controller:")]
    if tool_errors:
        entry["spec_cleanup_error"] = "; ".join(tool_errors)
    return entry
