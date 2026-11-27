"""Experimental ETA-style exit with a bounded replacement of a stale candidate.

This changes the Actor's reasoning policy. Exact speculative-tool commit does
not make it lossless relative to the original full-reasoning Actor. The default
SPORK runner is deliberately untouched; callers opt into this experiment.

Attribution: ETA supplies stable-call/free-probe/tool-readiness guards and the
no-tool rollback. SPORK supplies D1/D2/D3, isolated execution and the idea of
candidate supersession. The hypothesis is that replacing a stale dispatched
call unlocks ETA's dispatch-match guard and saves reasoning, beyond tool overlap.
Initial runner adapted from the existing spork_v2/early_exit.py snapshot
dee01dd57e9b809bfe52f11d210779773f1ee4f7f152d775ec1494aeea29b96c.
This module is independent of that concurrently developed experiment.
"""
from __future__ import annotations

import asyncio
import copy
from dataclasses import asdict, dataclass
import hashlib
import time

from .candidate_verification import decision_prefix, tool_probability
from .executors import commit_speculative, execute_speculative
from .prefix_replay import (PrefixHttpClient, StreamState, formal_action_started,
                            prediction_prompt_ids, same_call, wait_for_tokens)
from .timed_retry import _TurnResources
from .turn_runner import _append_tool_messages, _fork_point


@dataclass(frozen=True)
class ExitConfig:
    method: str = "refresh"  # eta, refresh (stable), latest (simple baseline)
    stable_checks: int = 3
    check_interval: int = 50
    max_checks: int = 15
    max_refreshes: int = 1
    tool_probability_threshold: float = .9
    continuation_max_tokens: int = 512
    enable_exit: bool = True

    def validate(self):
        if self.method not in {"eta", "refresh", "latest"}:
            raise ValueError(self.method)
        if not (2 <= self.stable_checks <= self.max_checks <= 20
                and self.check_interval > 0 and self.continuation_max_tokens > 0
                and 0 <= self.max_refreshes <= 2
                and 0 < self.tool_probability_threshold <= 1):
            raise ValueError("Invalid early-exit configuration")


class StableCandidate:
    """Confirm consecutive complete candidates using only past checks."""

    def __init__(self):
        self.call = None
        self.count = 0

    def observe(self, call, eligible):
        if not eligible or call is None:
            self.call, self.count = None, 0
        else:
            self.count = self.count + 1 if same_call(call, self.call) else 1
            self.call = copy.deepcopy(call)
        return self.count

    def should_refresh(self, active_call, *, required, refreshes, limit, active_done):
        return (self.call is not None and self.count >= required
                and not same_call(self.call, active_call)
                and refreshes < limit and active_done)


async def _forced(session, http, adapter, prompt, ids, text, cfg):
    request = prediction_prompt_ids(adapter.tokenizer, prompt, ids, text)
    choice = await http.complete(session, request, max_tokens=min(
        cfg.probe_max_tokens, cfg.context_window_tokens - len(request)),
        seed=cfg.seed, stop=adapter.probe_stop, logprobs=5)
    logs = adapter.extract_probe_logprobs(choice.get("logprobs"))
    call = adapter.parse_probe_tool_call(choice["text"])
    if choice["finish_reason"] != "stop":
        call = None  # Never treat a length-truncated proposal as an action.
    return {"call": call, "confidence": adapter.span_confidence(logs, skip_first=True),
            "wall_s": choice["wall_s"], "usage": choice.get("usage", {}),
            "text": choice["text"], "finish_reason": choice["finish_reason"]}


async def _free(session, http, tokenizer, prompt, ids, text, cfg):
    prefix = decision_prefix(tokenizer, prompt, ids, text)
    choice = await http.complete(session, prefix, max_tokens=1, seed=cfg.seed, logprobs=20)
    return {"tool_probability": tool_probability(choice), "wall_s": choice["wall_s"],
            "usage": choice.get("usage", {}), "token_ids": choice["token_ids"]}


async def _wait_cut(state, main, tool, armed_count):
    """Wait for real tool readiness and a boundary, with a 30-token grace."""
    while True:
        state.changed.clear()
        if main.done() or "</think>" in state.text or formal_action_started(state.text):
            return False
        if tool.done() and (state.text.endswith(("\n", ". ", ".\n"))
                            or len(state.token_ids) >= armed_count + 30):
            # Propagate a speculative tool failure instead of accepting it.
            tool.result()
            return True
        await state.changed.wait()


async def run_early_exit_turn(session, client, adapter, messages, executor, turn,
                              cfg, exit_cfg: ExitConfig, *, http=None):
    exit_cfg.validate()
    if adapter.name != "qwen3_tool_call" or not cfg.enable_thinking:
        raise ValueError("The experiment currently supports thinking Qwen3 only")
    if not callable(getattr(executor, "execute_speculative", None)):
        raise ValueError("An isolated speculative executor is required")
    start = time.perf_counter()
    elapsed = lambda: (time.perf_counter() - start) * 1000
    prompt = adapter.render_main_prompt(messages, enable_thinking=True)
    prompt_ids = adapter.tokenizer.encode(prompt, add_special_tokens=False)
    budget = min(cfg.baseline_max_tokens, cfg.context_window_tokens - len(prompt_ids))
    if budget <= 0:
        raise ValueError("Actor prompt fills context")
    http = http or PrefixHttpClient(client.base_url, client.model, client.timeout_s)
    state = StreamState()
    checks, proposals, speculations = [], [], []
    spec_task = spec_call = None
    spec_dispatch_ms = None
    d3_count = 0
    cut = {"fired": False, "rolled_back": False, "stable_count": 0, "refresh_count": 0}
    stable = StableCandidate()
    refresh_stable = StableCandidate()
    speculation_tasks = []
    active_job = None
    output_ids, output_text, finish_reason = [], "", None
    continuation = rollback = None
    async with _TurnResources(cfg.d3_clear) as resources:
        def dispatch(call):
            job = {"call": copy.deepcopy(call), "dispatch_ms": elapsed(),
                   "end_ms": None, "superseded": False, "committed": False}
            speculations.append(job)

            def execute():
                try:
                    execution = execute_speculative(executor, job["call"])
                    job.update(wall_s=execution.wall_s, real_wall_s=execution.real_wall_s,
                               artificial_sleep_s=execution.artificial_sleep_s,
                               result_sha256=hashlib.sha256(execution.text.encode()).hexdigest())
                    return execution
                finally:
                    job["end_ms"] = elapsed()

            task = resources.start_worker("speculative_tool", execute)
            speculation_tasks.append((task, job))
            return task, job

        main = resources.main_task = asyncio.create_task(http.stream(
            session, prompt_ids, state, max_tokens=budget, seed=cfg.seed,
            stop=adapter.main_stop, request_id=cfg.main_request_id))
        last_count = 0
        # One total check budget, as in ETA Algorithm 1. Call stability and
        # the current free-call probability are separate conditions (Eq. 5).
        for index in range(exit_cfg.max_checks):
            target = 1 if index == 0 else last_count + exit_cfg.check_interval
            if not await wait_for_tokens(state, main, target, snap=index > 0):
                break
            if "</think>" in state.text:
                break
            ids, text = list(state.token_ids), state.text
            last_count = len(ids)
            check = {"index": index, "kind": exit_cfg.method,
                     "tokens_at_dispatch": last_count, "dispatch_ms": elapsed()}
            prediction = await _forced(session, http, adapter, prompt_ids, ids, text, cfg)
            proposals.append(prediction)
            check["prediction"] = prediction
            if main.done() or "</think>" in state.text or formal_action_started(state.text):
                check.update(main_finished_during_check=True, end_ms=elapsed())
                checks.append(check)
                break
            free = await _free(session, http, adapter.tokenizer, prompt_ids, ids, text, cfg)
            call, probability = prediction["call"], free["tool_probability"]
            check.update(prediction=prediction, free=free)
            eligible = (probability >= exit_cfg.tool_probability_threshold and call is not None
                        and prediction["confidence"] is not None
                        and prediction["confidence"] >= cfg.confidence_threshold)
            stable.observe(call, call is not None)
            refresh_stable.observe(call, eligible)
            check["candidate_stable_count"] = stable.count
            if main.done() or "</think>" in state.text or formal_action_started(state.text):
                check.update(main_finished_during_check=True, end_ms=elapsed())
                checks.append(check)
                break
            if spec_task is None and call is not None and prediction["confidence"] is not None \
                    and prediction["confidence"] >= cfg.confidence_threshold:
                spec_call = copy.deepcopy(call)
                spec_task, active_job = dispatch(call)
                spec_dispatch_ms = active_job["dispatch_ms"]
                if cfg.d3_set:
                    info = await cfg.d3_set(call, prompt)
                    check["d3_set"] = info
                    if info.get("status") == "ok":
                        d3_count += 1
                    elif info.get("error") != "main_request_not_active_or_ambiguous":
                        raise RuntimeError(f"Main D3 registration failed: {info}")
            required = 1 if exit_cfg.method == "latest" else exit_cfg.stable_checks
            can_refresh = refresh_stable.should_refresh(spec_call, required=required,
                refreshes=cut["refresh_count"], limit=exit_cfg.max_refreshes,
                active_done=spec_task is not None and spec_task.done())
            check["refresh_eligible"] = can_refresh
            if exit_cfg.method in {"refresh", "latest"} and can_refresh:
                # Never adopt the previous speculative environment. Every new
                # candidate executes against a fresh copy of committed state.
                active_job["superseded"] = True
                spec_call = copy.deepcopy(refresh_stable.call)
                spec_task, active_job = dispatch(spec_call)
                spec_dispatch_ms = active_job["dispatch_ms"]
                cut["refresh_count"] += 1
                check["refreshed"] = True
            agrees = same_call(call, spec_call) and probability >= exit_cfg.tool_probability_threshold
            cut["stable_count"] = stable.count if same_call(call, spec_call) else 0
            check.update(call=call, tool_probability=probability, agrees=agrees,
                         stable_count=cut["stable_count"], end_ms=elapsed())
            checks.append(check)
            if (agrees and spec_task is not None and cut["stable_count"] >= exit_cfg.stable_checks
                    and exit_cfg.enable_exit and len(state.token_ids) < budget - 1):
                if await _wait_cut(state, main, spec_task, len(state.token_ids)):
                    cut.update(fired=True, cut_tokens=len(state.token_ids), cut_ms=elapsed())
                    main.cancel()
                    await asyncio.gather(main, return_exceptions=True)
                break

        if cut["fired"]:
            frozen_ids, frozen_text = list(state.token_ids), state.text
            prefix = decision_prefix(adapter.tokenizer, prompt_ids, frozen_ids, frozen_text)
            continuation = await http.complete(session, prefix, max_tokens=min(
                exit_cfg.continuation_max_tokens, budget - len(frozen_ids),
                cfg.context_window_tokens - len(prefix)),
                seed=cfg.seed, stop=adapter.main_stop)
            output_text = frozen_text + "\n</think>\n\n" + continuation["text"]
            call = adapter.parse_main_tool_call(output_text)
            mismatched_exit = exit_cfg.method != "eta" and not same_call(call, spec_call)
            if call is None or continuation["finish_reason"] != "stop" or mismatched_exit:
                # Resume the original, unclosed reasoning prefix. The trial
                # continuation never reaches messages or tool execution.
                cut["rolled_back"] = True
                cut["rollback_reason"] = ("candidate_mismatch" if mismatched_exit and call is not None
                                          else "no_complete_tool_call")
                rollback = await http.complete(session, prompt_ids + frozen_ids,
                    max_tokens=min(budget - len(frozen_ids),
                                   cfg.context_window_tokens - len(prompt_ids) - len(frozen_ids)),
                    seed=cfg.seed, stop=adapter.main_stop)
                output_text = frozen_text + rollback["text"]
                output_ids = frozen_ids + rollback["token_ids"]
                finish_reason = rollback["finish_reason"]
            else:
                output_ids = frozen_ids + adapter.tokenizer.encode("\n</think>\n\n", add_special_tokens=False) + continuation["token_ids"]
                finish_reason = continuation["finish_reason"]
        else:
            await main
            output_text, output_ids, finish_reason = state.text, list(state.token_ids), state.finish_reason
        if finish_reason != "stop":
            raise RuntimeError(f"Incomplete Actor output: {finish_reason}")
        main_end_ms = elapsed()
        call = adapter.parse_main_tool_call(output_text)
        if formal_action_started(output_text) and call is None:
            raise RuntimeError("Actor emitted an invalid tool call")
        # Every fork was created from the same committed environment state.
        # Keep obsolete results available in case the full Actor selects one.
        reusable = next(((task, job) for task, job in speculation_tasks
                         if same_call(call, job["call"])), None)
        matched = reusable is not None
        result_text = ""
        if matched:
            reusable_task, committed_job = reusable
            execution = await asyncio.shield(reusable_task)
            commit_speculative(executor, execution)
            committed_job["committed"] = True
            result_text = execution.text
        elif call is not None:
            result_text, _ = await asyncio.shield(resources.start_worker("canonical_tool", executor.execute, call))
        if call is not None:
            _append_tool_messages(messages, output_text, result_text)
        record = {"turn": turn, "fork_point": _fork_point(turn), "mode": "early_exit_" + exit_cfg.method,
                  "exit_config": asdict(exit_cfg), "exit": cut, "checks": checks, "probes": proposals,
                  "baseline_has_tool_call": call is not None,
                  "baseline_tool_name": call["name"] if call else None,
                  "baseline_tool_args": call.get("arguments") if call else None,
                  "baseline_text": output_text, "main_token_ids": output_ids,
                  "main_finish_reason": finish_reason, "completion_valid": True,
                  "initial_main_observed_tokens": len(state.token_ids),
                  "continuation": continuation, "rollback": rollback,
                  "main_end_ms": main_end_ms, "tool_result_ready_ms": elapsed(),
                  "spec_dispatched": spec_task is not None, "spec_used": matched,
                  "spec_wasted": any(not job["committed"] for job in speculations),
                  "spec_dispatch_ms": spec_dispatch_ms,
                  "spec_end_ms": active_job["end_ms"] if active_job else None,
                  "speculations": speculations,
                  "d3_set_count": d3_count, "tool_result_sha256": hashlib.sha256(result_text.encode()).hexdigest() if call else None,
                  "tool_result_preview": adapter.preview(result_text),
                  "environment_after": {"agent_db": executor.env.get_db_hash(), "user_db": executor.env.get_user_db_hash()},
                  **adapter.token_metrics(output_text),
                  "scope": "Approximate reasoning policy. Exact tool commit is only against the emitted shortened Actor call."}
        if call is None:
            record["final_answer"] = output_text
    record.update(turn_end_ms=elapsed(), cleanup_ms=resources.cleanup_ms,
                  d3_status=resources.d3_status, worker_errors=resources.worker_errors,
                  spec_end_ms=active_job["end_ms"] if active_job else None,
                  speculative_tool_calls=len(speculations))
    return record
