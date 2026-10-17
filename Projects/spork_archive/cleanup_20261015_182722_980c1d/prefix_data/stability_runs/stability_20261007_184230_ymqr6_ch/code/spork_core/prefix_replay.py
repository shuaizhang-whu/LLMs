"""Paired, one-turn prefix replay for learning the first SPORK launch time.

This is an opt-in collection path. The existing benchmark runner is unchanged.
Unlike the legacy HTTP callback, this path counts server-returned token IDs,
not SSE messages. All branch times include request/queue/prefill time after
the common prefix has been restored; they are replay measurements, not online
end-to-end speedups. A branch never receives another branch's continuation.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

import aiohttp

from .executors import commit_speculative, execute_speculative
from .tool_calls import args_exact_match


@dataclass
class ReplayConfig:
    delay_tokens: int = 64
    max_tokens: int = 8192
    context_window: int = 40960
    max_retries: int = 5
    retry_token_step: int = 50
    confidence_threshold: float = 0.90
    probe_max_tokens: int = 160
    snap_to_sentence: bool = True
    seed: int = 42


@dataclass
class StreamState:
    text: str = ""
    token_ids: list[int] = field(default_factory=list)
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    first_output_at: float | None = None
    finished_at: float | None = None
    finish_reason: str | None = None
    usage: dict = field(default_factory=dict)


class PrefixHttpClient:
    """vLLM completion API with exact token IDs and no hidden request retries."""

    def __init__(self, base_url: str, model: str, timeout_s: float = 600):
        self.base_url = base_url.rstrip("/")
        self.root = self.base_url.removesuffix("/v1")
        self.model = model
        self.timeout_s = timeout_s

    def body(self, prompt_ids, *, max_tokens, seed, stop=(), stream=False,
             request_id=None, logprobs=None):
        body = dict(model=self.model, prompt=prompt_ids, max_tokens=max_tokens,
                    temperature=0.0, seed=seed, stop=list(stop), stream=stream,
                    return_token_ids=True)
        if request_id is not None:
            body["request_id"] = request_id
        if logprobs is not None:
            body["logprobs"] = logprobs
        if stream:
            body["stream_options"] = {"include_usage": True}
        return body

    async def complete(self, session, prompt_ids, **kwargs):
        body = self.body(prompt_ids, **kwargs)
        t0 = time.perf_counter()
        async with session.post(self.base_url + "/completions", json=body,
                                timeout=aiohttp.ClientTimeout(total=self.timeout_s)) as r:
            if r.status != 200:
                raise RuntimeError(f"completion HTTP {r.status}: {(await r.text())[:1200]}")
            result = await r.json()
        choice = result["choices"][0]
        if choice.get("token_ids") is None:
            raise RuntimeError("vLLM did not return token_ids; exact prefix replay is unavailable")
        return {**choice, "usage": result.get("usage", {}),
                "wall_s": time.perf_counter() - t0}

    async def stream(self, session, prompt_ids, state: StreamState, **kwargs):
        body = self.body(prompt_ids, stream=True, **kwargs)
        completed = False
        try:
            async with session.post(self.base_url + "/completions", json=body,
                                    timeout=aiohttp.ClientTimeout(total=self.timeout_s)) as r:
                if r.status != 200:
                    raise RuntimeError(f"stream HTTP {r.status}: {(await r.text())[:1200]}")
                async for raw in r.content:
                    line = raw.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        completed = True
                        break
                    chunk = json.loads(payload)
                    if chunk.get("error"):
                        raise RuntimeError(str(chunk["error"]))
                    if chunk.get("usage"):
                        state.usage = chunk["usage"]
                    for choice in chunk.get("choices") or []:
                        ids = choice.get("token_ids")
                        text = choice.get("text") or ""
                        if text and ids is None:
                            raise RuntimeError("stream text arrived without token_ids")
                        if ids:
                            if state.first_output_at is None:
                                state.first_output_at = time.perf_counter()
                            state.token_ids.extend(ids)
                        state.text += text
                        if choice.get("finish_reason") is not None:
                            state.finish_reason = choice["finish_reason"]
                        state.changed.set()
            if not completed or state.finish_reason is None:
                raise RuntimeError("incomplete completion stream")
        finally:
            state.finished_at = time.perf_counter()
            state.changed.set()

    async def status(self, session):
        async with session.get(self.root + "/spork/status", timeout=30) as r:
            r.raise_for_status()
            return await r.json()

    async def clear(self, session):
        async with session.post(self.root + "/spork/clear", json={}, timeout=30) as r:
            r.raise_for_status()
            result = await r.json()
        if result.get("status") not in ("ok", "cleared"):
            raise RuntimeError(f"D3 clear failed: {result}")
        return result

    async def set_draft(self, session, request_id, prompt_len, call, tokenizer):
        body = "\n" + json.dumps(call, ensure_ascii=False)
        ids = tokenizer.encode(body, add_special_tokens=False)
        payload = dict(request_id=request_id, prompt_len=prompt_len, draft_tokens=ids)
        async with session.post(self.root + "/spork/set_tokens", json=payload, timeout=30) as r:
            r.raise_for_status()
            return await r.json()


def same_call(a: dict | None, b: dict | None) -> bool:
    return bool(a and b and a.get("name") == b.get("name")
                and args_exact_match(a.get("arguments"), b.get("arguments")))


def prediction_prompt_ids(tokenizer, prompt_ids, visible_ids, visible_text):
    """Preserve the exact visible token prefix, then force the SPORK call opener."""
    suffix = '\n\n<tool_call>\n{"name": "'
    if "</think>" not in visible_text:
        suffix = "\n</think>" + suffix
    return list(prompt_ids) + list(visible_ids) + tokenizer.encode(suffix, add_special_tokens=False)


def formal_action_started(text: str) -> bool:
    return "<tool_call>" in text


async def wait_for_tokens(state, main_task, target, *, snap=False):
    """Do not launch on a wall-clock timeout before the token target is reached."""
    while True:
        state.changed.clear()
        if main_task.done() or formal_action_started(state.text):
            return False
        count = len(state.token_ids)
        boundary = state.text.endswith(("\n", ". ", ".\n"))
        if count >= target and (not snap or boundary or count >= target + 30):
            return True
        # The stream's finally block signals errors and completion too.
        await state.changed.wait()


COUNTERS = ("set_count", "inject_count", "draft_token_count", "verified_injections",
            "verified_draft_tokens", "accepted_draft_tokens", "dropped_injections",
            "prefix_mismatches", "accounting_errors")


def counter_delta(before, after):
    return {k: after.get(k, 0) - before.get(k, 0) for k in COUNTERS}


async def run_prefix_branch(session, client, adapter, snapshot, executor, policy, cfg):
    """Run one independent off/now/wait continuation; return all outcomes.

    snapshot contains only the past. The caller supplies a freshly restored
    executor and performs a discarded warmup outside this measurement.
    D2: up to 5 predictions, name-span confidence, +50-token sentence retries.
    D3: register every parseable prediction before the tool-execution gate.
    The first confident tool is dispatched; strict name+args gates its commit.
    """
    if policy not in ("off", "now", "wait"):
        raise ValueError(policy)
    prompt_ids = snapshot["prompt_token_ids"] + snapshot["prefix_token_ids"]
    budget = min(cfg.max_tokens - len(snapshot["prefix_token_ids"]),
                 cfg.context_window - len(prompt_ids))
    if budget <= 0:
        raise ValueError("prefix fills the model context")
    await client.clear(session)
    before = await client.status(session)
    state = StreamState()
    request_id = "prefix-" + uuid.uuid4().hex
    probes = []
    spec_task = None
    spec_call = None
    main_task = None
    result: dict[str, Any] = {"policy": policy, "request_id": request_id,
                              "config": asdict(cfg), "probes": probes}
    t0 = time.perf_counter()
    try:
        main_task = asyncio.create_task(client.stream(
            session, prompt_ids, state, max_tokens=budget, seed=cfg.seed,
            stop=adapter.main_stop, request_id=request_id))
        # Both requests can start from the saved prefix; "now" does not wait
        # for a new Actor token or accidentally observe that token as input.
        await asyncio.sleep(0)
        last_count = 0
        if policy != "off":
            for retry in range(cfg.max_retries):
                if retry == 0:
                    if policy == "wait" and not await wait_for_tokens(
                            state, main_task, cfg.delay_tokens):
                        result["first_launch_cancelled"] = True
                        break
                elif not await wait_for_tokens(state, main_task,
                                               last_count + cfg.retry_token_step,
                                               snap=cfg.snap_to_sentence):
                    break
                if main_task.done() or formal_action_started(state.text):
                    break
                # Freeze precisely the bytes/IDs visible when this prediction starts.
                available_count = len(state.token_ids)
                if retry == 0 and policy == "now":
                    new_ids, new_text = [], ""
                else:
                    new_ids, new_text = list(state.token_ids), state.text
                last_count = available_count
                visible = snapshot["prefix_text"] + new_text
                visible_ids = snapshot["prefix_token_ids"] + new_ids
                probe_ids = prediction_prompt_ids(adapter.tokenizer,
                                                 snapshot["prompt_token_ids"],
                                                 visible_ids, visible)
                p = {"retry_idx": retry, "new_tokens_at_dispatch": len(new_ids),
                     "main_tokens_available_at_dispatch": available_count,
                     "total_visible_generated_tokens": len(visible_ids),
                     "visible_prefix_text": visible,
                     "dispatch_s": time.perf_counter() - t0}
                probes.append(p)
                probe_budget = min(cfg.probe_max_tokens, cfg.context_window - len(probe_ids))
                if probe_budget <= 0:
                    p["error"] = "prediction_context_full"
                    continue
                choice = await client.complete(session, probe_ids, max_tokens=probe_budget,
                                               stop=adapter.probe_stop, logprobs=5, seed=cfg.seed)
                call = adapter.parse_probe_tool_call(choice.get("text", ""))
                confidence = adapter.span_confidence(
                    adapter.extract_probe_logprobs(choice.get("logprobs")), skip_first=True)
                p.update(end_s=time.perf_counter() - t0, prediction=choice,
                         call=call, confidence=confidence, dispatched=False)
                # A prediction that finishes late is still an observed cost.
                if call is not None and not main_task.done():
                    p["d3_set"] = await client.set_draft(
                        session, request_id, len(prompt_ids), call, adapter.tokenizer)
                if call is not None and confidence is not None and confidence >= cfg.confidence_threshold:
                    spec_call = call
                    p["dispatched"] = True
                    p["tool_dispatch_s"] = time.perf_counter() - t0
                    spec_task = asyncio.create_task(asyncio.to_thread(
                        execute_speculative, executor, call))
                    break
        await main_task
        full_text = snapshot["prefix_text"] + state.text
        main_call = adapter.parse_main_tool_call(full_text)
        used = same_call(main_call, spec_call)
        tool_result = None
        tool_wall = real_wall = floor_wall = 0.0
        if used:
            execution = await spec_task
            commit_speculative(executor, execution)
            tool_result = execution.text
            tool_wall, real_wall, floor_wall = (
                execution.wall_s, execution.real_wall_s, execution.artificial_sleep_s)
        elif main_call is not None:
            result["canonical_tool_dispatch_s"] = time.perf_counter() - t0
            tool_result, tool_wall = await asyncio.to_thread(executor.execute, main_call)
            real_wall, floor_wall = executor.last_real_wall_s, executor.last_floor_sleep_s
        ready = time.perf_counter()
        result.update(
            wall_s=ready - t0, main_text=full_text,
            continuation_text=state.text, continuation_token_ids=list(state.token_ids),
            finish_reason=state.finish_reason, usage=state.usage,
            first_output_s=(state.first_output_at - t0 if state.first_output_at else None),
            main_finished_s=state.finished_at - t0,
            formal_call=main_call, tool_result=tool_result,
            tool_wall_s=tool_wall, tool_real_s=real_wall, tool_artificial_sleep_s=floor_wall,
            spec_used=used, spec_wasted=spec_task is not None and not used,
            completion_valid=state.finish_reason == "stop",
        )
    except Exception as exc:
        result.update(error=f"{type(exc).__name__}: {exc}", completion_valid=False)
    finally:
        cleanup_start = time.perf_counter()
        if main_task is not None and not main_task.done():
            main_task.cancel()
        if main_task is not None:
            await asyncio.gather(main_task, return_exceptions=True)
        # Cancelling asyncio.to_thread does not stop its underlying tool thread.
        # Drain it, including misses/no-tool turns, before another branch starts.
        if spec_task is not None:
            spec_results = await asyncio.gather(spec_task, return_exceptions=True)
            if isinstance(spec_results[0], BaseException):
                result["cleanup_tool_error"] = str(spec_results[0])
        after = await client.status(session)
        result["d3_status_after"] = after
        cleared = await client.clear(session)
        result["clear_after"] = cleared
        # Clearing a still-pending draft increments dropped_injections; include
        # that cleanup cost in the accounting instead of losing it between pairs.
        result["d3_counters"] = counter_delta(before, {**after, **cleared})
        result["cleanup_s"] = time.perf_counter() - cleanup_start
    return result


def paired_labels(outcomes):
    """Keep raw negative gains and outcome disagreements, not just successful hits."""
    valid = (set(outcomes) == {"off", "now", "wait"}
             and all(r.get("completion_valid") and not r.get("error") for r in outcomes.values()))
    labels = {"valid": valid, "measurement": "warmed_prefix_replay_request_to_result",
              "quality_evaluated": False}
    if not valid:
        labels["reason"] = "missing_failed_or_token_limited_branch"
        return labels
    off = outcomes["off"]
    labels.update(gain_now_s=off["wall_s"] - outcomes["now"]["wall_s"],
                  gain_wait_s=off["wall_s"] - outcomes["wait"]["wall_s"])
    for p in ("now", "wait"):
        a, b = off.get("formal_call"), outcomes[p].get("formal_call")
        labels[p + "_same_action_as_off"] = (a is None and b is None) or same_call(a, b)
        labels[p + "_same_result_as_off"] = off.get("tool_result") == outcomes[p].get("tool_result")
    labels["best_observed_policy"] = max(
        (("off", 0.0), ("now", labels["gain_now_s"]), ("wait", labels["gain_wait_s"])),
        key=lambda item: item[1])[0]
    return labels
