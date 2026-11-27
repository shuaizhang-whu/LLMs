"""Experimental ETA-style early exit with candidate-reusing decision probes.

This changes the Actor's reasoning policy. Exact speculative-tool commit does
not make it lossless relative to the original full-reasoning Actor. The default
SPORK runner is deliberately untouched; callers opt into this experiment.

Attribution: ETA supplies stable-call/free-probe/tool-readiness guards and the
no-tool rollback. SPORK supplies D1/D2/D3 and isolated tool execution. Reusing an
existing call to accelerate repeated unforced checks is the hypothesis tested
here, motivated by ToolSpec's reuse and verify-rather-than-regenerate methods.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
import hashlib
import time

from .candidate_verification import decision_prefix, decision_probe, tool_probability
from .executors import commit_speculative, execute_speculative
from .prefix_replay import (PrefixHttpClient, StreamState, formal_action_started,
                            prediction_prompt_ids, same_call, wait_for_tokens)
from .timed_retry import _TurnResources
from .turn_runner import _append_tool_messages, _fork_point


@dataclass(frozen=True)
class ExitConfig:
    method: str = "reuse"  # eta, decision (ablation), reuse, redo, reuse_gated
    stable_checks: int = 3
    check_interval: int = 50
    max_checks: int = 8
    tool_probability_threshold: float = .9
    continuation_max_tokens: int = 512
    enable_exit: bool = True

    def validate(self):
        if self.method not in {"eta", "decision", "reuse", "redo", "reuse_gated"}:
            raise ValueError(self.method)
        if not (2 <= self.stable_checks <= self.max_checks <= 20
                and self.check_interval > 0 and self.continuation_max_tokens > 0
                and 0 < self.tool_probability_threshold <= 1):
            raise ValueError("Invalid early-exit configuration")


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
    checks, proposals = [], []
    spec_task = spec_call = None
    spec_dispatch_ms = spec_end_ms = None
    d3_count = 0
    dispatch_generation = 0
    redispatch_count = 0
    redispatch_log = []
    call_history: list[dict | None] = []
    cut = {"fired": False, "rolled_back": False, "stable_count": 0}
    output_ids, output_text, finish_reason = [], "", None
    continuation = rollback = None
    async with _TurnResources(cfg.d3_clear) as resources:
        main = resources.main_task = asyncio.create_task(http.stream(
            session, prompt_ids, state, max_tokens=budget, seed=cfg.seed,
            stop=adapter.main_stop, request_id=cfg.main_request_id))

        def start_speculative(target_call):
            """Dispatch, or supersede with, a speculative tool execution.

            A worker thread cannot be stopped once started. A superseded
            execution is therefore never awaited or committed, and
            _TurnResources drains it at cleanup. Only the current generation
            may update spec_end_ms.
            """
            nonlocal spec_task, spec_dispatch_ms, dispatch_generation
            dispatch_generation += 1
            generation = dispatch_generation

            def execute():
                nonlocal spec_end_ms
                try:
                    return execute_speculative(executor, target_call)
                finally:
                    if generation == dispatch_generation:
                        spec_end_ms = elapsed()

            spec_dispatch_ms = elapsed()
            spec_task = resources.start_worker("speculative_tool", execute)
            return spec_task

        last_count = 0
        # Original first-token launch and name-confidence D2 dispatch.
        for index in range(cfg.max_retries):
            target = 1 if index == 0 else last_count + cfg.retry_token_step
            if not await wait_for_tokens(state, main, target, snap=index > 0):
                break
            if "</think>" in state.text:
                break
            ids, text = list(state.token_ids), state.text
            last_count = len(ids)
            free = None
            if exit_cfg.method == "reuse_gated":
                # Use ETA's unforced signal before paying for a complete
                # prediction (CTES-style pre-draft admission, without training).
                # It concerns this shortened context, not final tool necessity.
                free = await _free(session, http, adapter.tokenizer, prompt_ids, ids, text, cfg)
                if free["tool_probability"] < exit_cfg.tool_probability_threshold:
                    proposals.append({"call": None, "confidence": None,
                        "tokens_at_dispatch": last_count, "end_ms": elapsed(),
                        "dispatch_free_probe": free, "dispatch_deferred": True,
                        "prediction_skipped": True})
                    continue
                if main.done() or "</think>" in state.text or formal_action_started(state.text):
                    break
            proposal = await _forced(session, http, adapter, prompt_ids, ids, text, cfg)
            proposal.update(tokens_at_dispatch=last_count, end_ms=elapsed())
            if free is not None:
                proposal["dispatch_free_probe"] = free
            proposals.append(proposal)
            call, conf = proposal["call"], proposal["confidence"]
            if main.done():
                break
            if call and cfg.d3_set:
                info = await cfg.d3_set(call, prompt)
                proposal["d3_set"] = info
                if info.get("status") == "ok":
                    d3_count += 1
                elif info.get("error") != "main_request_not_active_or_ambiguous":
                    raise RuntimeError(f"Main D3 registration failed: {info}")
            if call is None or conf is None or conf < cfg.confidence_threshold:
                continue
            spec_call = call
            start_speculative(call)
            call_history.append(call)
            if free is None:
                free = await _free(session, http, adapter.tokenizer, prompt_ids, ids, text, cfg)
            cut["stable_count"] = int(free["tool_probability"] >= exit_cfg.tool_probability_threshold)
            checks.append({"kind": "initial_forced_and_free", "tokens_at_dispatch": last_count,
                           "call": call, "free": free, "end_ms": elapsed(),
                           "stable_count": cut["stable_count"]})
            break

        for index in range(1, exit_cfg.max_checks if spec_task is not None else 0):
            if not await wait_for_tokens(state, main, last_count + exit_cfg.check_interval, snap=True):
                break
            if "</think>" in state.text:
                break
            ids, text = list(state.token_ids), state.text
            last_count = len(ids)
            check = {"index": index, "kind": exit_cfg.method,
                     "tokens_at_dispatch": last_count, "dispatch_ms": elapsed()}
            if exit_cfg.method == "eta":
                prediction = await _forced(session, http, adapter, prompt_ids, ids, text, cfg)
                free = await _free(session, http, adapter.tokenizer, prompt_ids, ids, text, cfg)
                call, probability = prediction["call"], free["tool_probability"]
                check.update(prediction=prediction, free=free)
            else:
                prefix = decision_prefix(adapter.tokenizer, prompt_ids, ids, text)
                prediction = await decision_probe(
                    session, http, adapter.tokenizer, prefix, candidate=spec_call,
                    use_d3=exit_cfg.method in {"reuse", "redo", "reuse_gated"},
                    threshold=exit_cfg.tool_probability_threshold,
                    max_tokens=cfg.probe_max_tokens, seed=cfg.seed)
                call = adapter.parse_main_tool_call(prediction["call_text"] or "")
                if prediction.get("finish_reason") != "stop":
                    call = None
                probability = prediction["tool_probability"]
                check["prediction"] = prediction

            if exit_cfg.method == "redo":
                # ETA's stability is defined over consecutive probe predictions.
                # When the stable call differs from the one already dispatched,
                # supersede the stale execution instead of forfeiting the exit.
                call_history.append(call)
                window = call_history[-exit_cfg.stable_checks:]
                stable_k = (len(window) == exit_cfg.stable_checks
                            and window[0] is not None
                            and all(same_call(c, window[0]) for c in window[1:]))
                redispatched = stable_k and not same_call(call, spec_call)
                if redispatched:
                    previous_call = spec_call
                    spec_call = call
                    start_speculative(call)
                    redispatch_count += 1
                    redispatch_log.append({"index": index, "tokens": last_count,
                                           "previous_call": previous_call, "call": call,
                                           "end_ms": elapsed()})
                    check["redispatch"] = True
                    check["previous_call"] = previous_call
                agrees = (same_call(call, spec_call)
                          and probability >= exit_cfg.tool_probability_threshold)
                cut["stable_count"] = (exit_cfg.stable_checks if redispatched else
                                       (cut["stable_count"] + 1 if agrees else 0))
            else:
                agrees = (same_call(call, spec_call)
                          and probability >= exit_cfg.tool_probability_threshold)
                cut["stable_count"] = cut["stable_count"] + 1 if agrees else 0
            check.update(call=call, tool_probability=probability, agrees=agrees,
                         stable_count=cut["stable_count"], end_ms=elapsed())
            checks.append(check)
            if cut["stable_count"] >= exit_cfg.stable_checks and exit_cfg.enable_exit:
                if await _wait_cut(state, main, spec_task, len(state.token_ids)):
                    cut.update(fired=True, cut_tokens=len(state.token_ids), cut_ms=elapsed())
                    main.cancel()
                    await asyncio.gather(main, return_exceptions=True)
                break

        if cut["fired"]:
            frozen_ids, frozen_text = list(state.token_ids), state.text
            prefix = decision_prefix(adapter.tokenizer, prompt_ids, frozen_ids, frozen_text)
            continuation = await http.complete(session, prefix, max_tokens=min(
                exit_cfg.continuation_max_tokens, cfg.context_window_tokens - len(prefix)),
                seed=cfg.seed, stop=adapter.main_stop)
            output_text = frozen_text + "\n</think>\n\n" + continuation["text"]
            call = adapter.parse_main_tool_call(output_text)
            if call is None or continuation["finish_reason"] != "stop":
                # Resume the original, unclosed reasoning prefix. The trial
                # continuation never reaches messages or tool execution.
                cut["rolled_back"] = True
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
        matched = same_call(call, spec_call)
        result_text = ""
        if matched:
            execution = await asyncio.shield(spec_task)
            commit_speculative(executor, execution)
            result_text = execution.text
        elif call is not None:
            result_text, _ = await asyncio.shield(resources.start_worker("canonical_tool", executor.execute, call))
        if call is not None:
            _append_tool_messages(messages, output_text, result_text)
        record = {"turn": turn, "fork_point": _fork_point(turn), "mode": "early_exit_" + exit_cfg.method,
                  "exit_config": asdict(exit_cfg), "exit": cut, "checks": checks, "probes": proposals,
                  "redispatch_count": redispatch_count, "redispatch_log": redispatch_log,
                  "baseline_has_tool_call": call is not None,
                  "baseline_tool_name": call["name"] if call else None,
                  "baseline_tool_args": call.get("arguments") if call else None,
                  "baseline_text": output_text, "main_token_ids": output_ids,
                  "main_finish_reason": finish_reason, "completion_valid": True,
                  "initial_main_observed_tokens": len(state.token_ids),
                  "continuation": continuation, "rollback": rollback,
                  "main_end_ms": main_end_ms, "tool_result_ready_ms": elapsed(),
                  "spec_dispatched": spec_task is not None, "spec_used": matched,
                  "spec_wasted": spec_task is not None and not matched,
                  "spec_dispatch_ms": spec_dispatch_ms, "spec_end_ms": spec_end_ms,
                  "d3_set_count": d3_count, "tool_result_sha256": hashlib.sha256(result_text.encode()).hexdigest() if call else None,
                  "tool_result_preview": adapter.preview(result_text),
                  "environment_after": {"agent_db": executor.env.get_db_hash(), "user_db": executor.env.get_user_db_hash()},
                  **adapter.token_metrics(output_text),
                  "scope": "Approximate reasoning policy. Exact tool commit is only against the emitted shortened Actor call."}
        if call is None:
            record["final_answer"] = output_text
    record.update(turn_end_ms=elapsed(), cleanup_ms=resources.cleanup_ms,
                  d3_status=resources.d3_status, worker_errors=resources.worker_errors)
    return record
