"""Experimental ETA-style early exit with bounded replacement and draft reuse.

This changes the Actor's reasoning policy. Exact speculative-tool commit does
not make it lossless relative to the original full-reasoning Actor. The default
SPORK runner is deliberately untouched; callers opt into this experiment.

Attribution: ETA supplies stable-call/free-probe/tool-readiness guards and the
no-tool rollback. SPORK supplies D1/D2/D3 and isolated tool execution. Reusing an
existing call to accelerate repeated unforced checks is the hypothesis tested
here, motivated by ToolSpec's reuse and verify-rather-than-regenerate methods.
An optional stable replacement reopens speculation after the first dispatch,
with at most one additional isolated execution; neither result is committed
until it matches the independently emitted Actor action.

Deferred checks change the evaluation order of ETA's conjunction: a low free
probability postpones full calls, while a high one triggers verification of the
last k frozen prefixes. This keeps the stability evidence requirement, but the
different request timing can change the online Actor trajectory.
"""
from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import time

from .candidate_verification import decision_prefix, decision_probe, forced_probe_with_draft, tool_probability
from .draft_reuse import REUSE_MODES
from .executors import commit_speculative, execute_speculative
from .prefix_replay import (PrefixHttpClient, StreamState, formal_action_started,
                            prediction_prompt_ids, same_call, wait_for_tokens)
from .timed_retry import _TurnResources
from .turn_runner import _append_tool_messages, _fork_point


@dataclass(frozen=True)
class ExitConfig:
    method: str = "reuse"  # eta, decision (ablation), reuse
    stable_checks: int = 3
    check_interval: int = 50
    max_checks: int = 8
    tool_probability_threshold: float = .9
    continuation_max_tokens: int = 512
    enable_exit: bool = True
    retarget_after_stable: bool = False
    probe_draft_reuse_mode: str = "single"
    reuse_forced_probe: bool = False
    defer_forced_checks: bool = False
    check_schedule: str = "standard"  # post-dispatch: standard, fixed_fast, ready_fast
    fast_check_interval: int = 16
    guard_policy: str = "conservative"  # paper: Eq. (4)-(5), one total probe budget

    def validate(self):
        if self.method not in {"eta", "decision", "reuse"}:
            raise ValueError(self.method)
        if self.probe_draft_reuse_mode not in REUSE_MODES:
            raise ValueError(self.probe_draft_reuse_mode)
        if self.guard_policy not in {"conservative", "paper"}:
            raise ValueError(self.guard_policy)
        if self.guard_policy == "paper" and self.method != "eta":
            raise ValueError("Paper guards require forced and free probes at every check")
        if self.reuse_forced_probe and self.method != "eta":
            raise ValueError("Forced-probe reuse requires ETA's full forced/free checks")
        if self.defer_forced_checks and (self.method != "eta" or self.guard_policy != "paper"):
            raise ValueError("Deferred checks require ETA's consecutive-call guard")
        if self.check_schedule not in {"standard", "fixed_fast", "ready_fast"}:
            raise ValueError(self.check_schedule)
        if self.check_schedule != "standard" and not (
                self.guard_policy == "paper" and self.method == "eta"
                and 0 < self.fast_check_interval < self.check_interval):
            raise ValueError("Short post-dispatch cadence requires ETA and a shorter positive interval")
        if not (2 <= self.stable_checks <= self.max_checks <= 20
                and self.check_interval > 0 and self.continuation_max_tokens > 0
                and 0 < self.tool_probability_threshold <= 1):
            raise ValueError("Invalid early-exit configuration")


async def _forced(session, http, adapter, prompt, ids, text, cfg, *, draft=None, reuse_mode=None):
    request = prediction_prompt_ids(adapter.tokenizer, prompt, ids, text)
    generation = dict(max_tokens=min(
        cfg.probe_max_tokens, cfg.context_window_tokens - len(request)),
        seed=cfg.seed, stop=adapter.probe_stop, logprobs=5)
    reuse = None
    if reuse_mode is None:
        choice = await http.complete(session, request, **generation)
    else:
        choice, reuse = await forced_probe_with_draft(
            session, http, adapter.tokenizer, request, candidate=draft,
            reuse_mode=reuse_mode, **generation)
    logs = adapter.extract_probe_logprobs(choice.get("logprobs"))
    call = adapter.parse_probe_tool_call(choice["text"])
    if choice["finish_reason"] != "stop":
        call = None  # Never treat a length-truncated proposal as an action.
    result = {"call": call, "confidence": adapter.span_confidence(logs, skip_first=True),
              "wall_s": reuse["wall_s"] if reuse is not None else choice["wall_s"],
              "usage": choice.get("usage", {}), "text": choice["text"],
              "finish_reason": choice["finish_reason"]}
    if reuse is not None:
        result["draft_reuse"] = reuse
    return result


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


async def _wait_next_check(state, main, tool, last_count, config, *, candidate_clear):
    """Choose a checkpoint using only the last observed call/probability and tool readiness.

    All schedules retain the same sentence-boundary rule and 30-token grace.
    Ready-fast rechecks readiness as the Actor advances, so a tool completing
    during the wait can shorten the target. Closer checks are a policy change;
    three nearby matching calls do not establish unchanged task quality.
    """
    while True:
        state.changed.clear()
        if (main.done() or state.finished_at is not None or "</think>" in state.text
                or formal_action_started(state.text)):
            return None
        fast = config.check_schedule == "fixed_fast" or (
            config.check_schedule == "ready_fast" and candidate_clear and tool.done())
        interval = config.fast_check_interval if fast else config.check_interval
        target = last_count + interval
        boundary = state.text.endswith(("\n", ". ", ".\n"))
        if len(state.token_ids) >= target and (boundary or len(state.token_ids) >= target + 30):
            return interval
        await state.changed.wait()


async def run_early_exit_turn(session, client, adapter, messages, executor, turn,
                              cfg, exit_cfg: ExitConfig, *, http=None):
    exit_cfg.validate()
    paper_guard = exit_cfg.guard_policy == "paper"
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
    executions = []
    active_execution = None
    stable_call, stable_count = None, 0
    draft_call = None
    previous_forced_call = None
    # Keep the actual observed prefixes, including low-probability checkpoints.
    # A skipped prediction is unknown, never evidence of a stable action.
    recent_checks = deque(maxlen=exit_cfg.stable_checks)
    d3_count = 0
    cut = {"fired": False, "rolled_back": False, "stable_count": 0, "retarget_count": 0}
    output_ids, output_text, finish_reason = [], "", None
    continuation = rollback = None
    async with _TurnResources(cfg.d3_clear) as resources:
        def dispatch(call, reason):
            execution = {"call": call, "reason": reason, "dispatch_ms": elapsed(),
                         "end_ms": None, "committed": False}
            executions.append(execution)

            def execute():
                try:
                    return execute_speculative(executor, call)
                finally:
                    execution["end_ms"] = elapsed()

            return resources.start_worker("speculative_tool", execute), execution

        main = resources.main_task = asyncio.create_task(http.stream(
            session, prompt_ids, state, max_tokens=budget, seed=cfg.seed,
            stop=adapter.main_stop, request_id=cfg.main_request_id, sampling=cfg.actor_sampling))

        async def arm_exit():
            if await _wait_cut(state, main, spec_task, len(state.token_ids)):
                cut.update(fired=True, cut_tokens=len(state.token_ids), cut_ms=elapsed())
                main.cancel()
                await asyncio.gather(main, return_exceptions=True)

        last_count = 0
        armed = False
        # Original first-token launch and name-confidence D2 dispatch.
        # The paper uses one budget including every pre-dispatch probe, and
        # counts equal calls independently of the preceding free probabilities.
        initial_limit = exit_cfg.max_checks if paper_guard else cfg.max_retries
        for index in range(initial_limit):
            interval = exit_cfg.check_interval if paper_guard else cfg.retry_token_step
            target = 1 if index == 0 else last_count + interval
            if not await wait_for_tokens(state, main, target, snap=index > 0):
                break
            if "</think>" in state.text:
                break
            ids, text = list(state.token_ids), state.text
            last_count = len(ids)
            proposal = await _forced(session, http, adapter, prompt_ids, ids, text, cfg,
                draft=previous_forced_call,
                reuse_mode=exit_cfg.probe_draft_reuse_mode if exit_cfg.reuse_forced_probe else None)
            previous_forced_call = proposal["call"]
            proposal.update(tokens_at_dispatch=last_count, end_ms=elapsed())
            proposals.append(proposal)
            call, conf = proposal["call"], proposal["confidence"]
            if main.done():
                break
            if paper_guard:
                free = await _free(session, http, adapter.tokenizer, prompt_ids, ids, text, cfg)
                stable_count = stable_count + 1 if same_call(call, stable_call) else int(call is not None)
                stable_call = call
                checks.append({"index": index, "kind": "initial_forced_and_free",
                               "proposal_index": len(proposals) - 1,
                               "tokens_at_dispatch": last_count, "call": call, "free": free,
                               "end_ms": elapsed(), "stable_count": stable_count})
                if exit_cfg.defer_forced_checks:
                    recent_checks.append((checks[-1], ids, text))
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
            spec_call = draft_call = call
            spec_task, active_execution = dispatch(call, "initial")
            if paper_guard:
                cut["stable_count"] = stable_count
                agrees = free["tool_probability"] >= exit_cfg.tool_probability_threshold
                checks[-1].update(agrees=agrees)
                if stable_count >= exit_cfg.stable_checks and agrees and exit_cfg.enable_exit:
                    armed = True
                    await arm_exit()
            else:
                stable_call = call
                free = await _free(session, http, adapter.tokenizer, prompt_ids, ids, text, cfg)
                cut["stable_count"] = int(free["tool_probability"] >= exit_cfg.tool_probability_threshold)
                stable_count = cut["stable_count"]
                checks.append({"kind": "initial_forced_and_free", "tokens_at_dispatch": last_count,
                               "call": call, "free": free, "end_ms": elapsed(),
                               "stable_count": cut["stable_count"]})
            break

        first_check = len(checks) if paper_guard else 1
        for index in range(first_check, exit_cfg.max_checks if spec_task is not None and not armed else 0):
            previous = checks[-1]
            previous_probability = previous.get("tool_probability", previous.get("free", {}).get("tool_probability", 0.))
            candidate_clear = (same_call(previous.get("call"), spec_call)
                               and previous_probability >= exit_cfg.tool_probability_threshold)
            interval = await _wait_next_check(state, main, spec_task, last_count, exit_cfg,
                                             candidate_clear=candidate_clear)
            if interval is None:
                break
            if "</think>" in state.text:
                break
            ids, text = list(state.token_ids), state.text
            last_count = len(ids)
            check = {"index": index, "kind": exit_cfg.method,
                     "tokens_at_dispatch": last_count, "dispatch_ms": elapsed(),
                     "check_interval_used": interval, "candidate_clear_before_wait": candidate_clear,
                     "tool_ready_at_checkpoint": spec_task.done()}
            if exit_cfg.defer_forced_checks:
                free = await _free(session, http, adapter.tokenizer, prompt_ids, ids, text, cfg)
                probability = free["tool_probability"]
                check.update(free=free, call=None, forced_deferred=True)
                recent_checks.append((check, ids, text))
                stable_call, stable_count = None, 0
                if probability >= exit_cfg.tool_probability_threshold:
                    # Evaluate the conjunction only when it can enable an exit.
                    # Start with the latest call, then verify older frozen
                    # prefixes backwards, stopping at the first disagreement.
                    # Each full prediction is cached at its own checkpoint.
                    for prior, prior_ids, prior_text in reversed(recent_checks):
                        if main.done() or "</think>" in state.text:
                            break
                        if "prediction" not in prior and "proposal_index" not in prior:
                            prediction = await _forced(
                                session, http, adapter, prompt_ids, prior_ids, prior_text, cfg,
                                draft=previous_forced_call,
                                reuse_mode=exit_cfg.probe_draft_reuse_mode if exit_cfg.reuse_forced_probe else None)
                            previous_forced_call = prediction["call"]
                            prior.update(prediction=prediction, call=prediction["call"],
                                         forced_verified_at_check=index, forced_end_ms=elapsed())
                        candidate = prior["call"]
                        if candidate is None or (stable_count and not same_call(candidate, stable_call)):
                            break
                        stable_call = candidate
                        stable_count += 1
                call = check["call"]
                check["verified_window_indices"] = [
                    entry[0]["index"] for entry in list(recent_checks)[-stable_count:]
                ] if stable_count else []
            elif exit_cfg.method == "eta":
                prediction = await _forced(session, http, adapter, prompt_ids, ids, text, cfg,
                    draft=previous_forced_call,
                    reuse_mode=exit_cfg.probe_draft_reuse_mode if exit_cfg.reuse_forced_probe else None)
                previous_forced_call = prediction["call"]
                free = await _free(session, http, adapter.tokenizer, prompt_ids, ids, text, cfg)
                call, probability = prediction["call"], free["tool_probability"]
                check.update(prediction=prediction, free=free)
            else:
                prefix = decision_prefix(adapter.tokenizer, prompt_ids, ids, text)
                prediction = await decision_probe(
                    session, http, adapter.tokenizer, prefix, candidate=draft_call,
                    use_d3=exit_cfg.method == "reuse", threshold=exit_cfg.tool_probability_threshold,
                    max_tokens=cfg.probe_max_tokens, seed=cfg.seed,
                    reuse_mode=exit_cfg.probe_draft_reuse_mode)
                call = adapter.parse_main_tool_call(prediction["call_text"] or "")
                if prediction.get("finish_reason") != "stop":
                    call = None
                probability = prediction["tool_probability"]
                check["prediction"] = prediction
            agrees = same_call(call, spec_call) and probability >= exit_cfg.tool_probability_threshold
            if not paper_guard:
                cut["stable_count"] = cut["stable_count"] + 1 if agrees else 0
            if exit_cfg.defer_forced_checks:
                pass  # Count only the explicitly verified consecutive window above.
            elif call is not None and (paper_guard or probability >= exit_cfg.tool_probability_threshold):
                stable_count = stable_count + 1 if same_call(call, stable_call) else 1
                stable_call = call
                if exit_cfg.retarget_after_stable:
                    # Only an already observed call supplies the next probe's
                    # draft; target verification still generates its decision.
                    draft_call = call
            else:
                stable_call, stable_count = None, 0
            if paper_guard:
                cut["stable_count"] = stable_count if same_call(call, spec_call) else 0
            if (exit_cfg.retarget_after_stable and cut["retarget_count"] == 0
                    and stable_count >= exit_cfg.stable_checks
                    and probability >= exit_cfg.tool_probability_threshold
                    and not same_call(stable_call, spec_call)
                    and not main.done() and "</think>" not in state.text
                    and not formal_action_started(state.text)):
                # At most one replacement. Both executions use isolated state;
                # the obsolete worker stays owned and is drained on cleanup.
                check["previous_dispatched_call"] = spec_call
                spec_call = stable_call
                spec_task, active_execution = dispatch(spec_call, "stable_replacement")
                cut["retarget_count"] += 1
                cut["stable_count"] = stable_count
                check["retargeted"] = True
                agrees = True
            check.update(call=call, tool_probability=probability, agrees=agrees,
                         stable_count=cut["stable_count"], candidate_stable_count=stable_count,
                         end_ms=elapsed())
            checks.append(check)
            if cut["stable_count"] >= exit_cfg.stable_checks and agrees and exit_cfg.enable_exit:
                await arm_exit()
                break

        if cut["fired"]:
            frozen_ids, frozen_text = list(state.token_ids), state.text
            prefix = decision_prefix(adapter.tokenizer, prompt_ids, frozen_ids, frozen_text)
            continuation = await http.complete(session, prefix, max_tokens=min(
                exit_cfg.continuation_max_tokens, cfg.context_window_tokens - len(prefix)),
                seed=cfg.seed, stop=adapter.main_stop, sampling=cfg.actor_sampling)
            output_text = frozen_text + "\n</think>\n\n" + continuation["text"]
            call = adapter.parse_main_tool_call(output_text)
            if call is None or continuation["finish_reason"] != "stop":
                # Resume the original, unclosed reasoning prefix. The trial
                # continuation never reaches messages or tool execution.
                cut["rolled_back"] = True
                rollback = await http.complete(session, prompt_ids + frozen_ids,
                    max_tokens=min(budget - len(frozen_ids),
                                   cfg.context_window_tokens - len(prompt_ids) - len(frozen_ids)),
                    seed=cfg.seed, stop=adapter.main_stop, sampling=cfg.actor_sampling)
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
            error = RuntimeError(f"Incomplete Actor output: {finish_reason}")
            error.partial_actor = {"turn": turn, "text": output_text,
                "token_ids": output_ids, "finish_reason": finish_reason,
                "exit": cut, "probes": proposals, "checks": checks, "completion_valid": False}
            raise error
        main_end_ms = elapsed()
        call = adapter.parse_main_tool_call(output_text)
        if formal_action_started(output_text) and call is None:
            raise RuntimeError("Actor emitted an invalid tool call")
        matched = same_call(call, spec_call)
        result_text = ""
        if matched:
            execution = await asyncio.shield(spec_task)
            commit_speculative(executor, execution)
            active_execution["committed"] = True
            result_text = execution.text
        elif call is not None:
            result_text, _ = await asyncio.shield(resources.start_worker("canonical_tool", executor.execute, call))
        if call is not None:
            _append_tool_messages(messages, output_text, result_text,
                                  canonical_call=call if cfg.canonical_tool_history else None)
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
                  "spec_wasted": spec_task is not None and not matched,
                  "spec_dispatch_ms": active_execution["dispatch_ms"] if active_execution else None,
                  "spec_end_ms": active_execution["end_ms"] if active_execution else None,
                  "speculative_executions": executions,
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
