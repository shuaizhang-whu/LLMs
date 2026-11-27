"""Compose exact-input successor pairs across a bounded SPORK trajectory."""
from __future__ import annotations

import time

from .successor_prefetch import run_successor_pair
from .timed_retry import run_timed_retry_turn


async def run_successor_task(session, client, adapter, messages, executor, make_cfg,
                             *, mode="raw", max_turns=8, on_turn=None, trace=None):
    """Keep all completed turns, stopping on a visible reply or the shared cap.

    A pair never speculates recursively. After it is consumed or discarded, the
    next pair starts from the actual committed history and environment. With
    one turn left, use ordinary SPORK instead of computing an unusable child.
    """
    if mode not in {"baseline", "raw", "confirmed", "serial", "actor_only"} or max_turns < 1:
        raise ValueError("Invalid task mode or turn budget")
    trace = trace if trace is not None else {}
    trace.update(mode=mode, pairs=[], adopted=0, errors=[], observed_wasted_child_tokens=0)
    entries = []
    started = time.perf_counter()

    def accept(entry):
        if entry["turn"] != len(entries) + 1:
            raise RuntimeError("Canonical turn order changed")
        if entries and not entries[-1]["baseline_has_tool_call"]:
            raise RuntimeError("A turn was committed after a visible final reply")
        entries.append(entry)
        if on_turn is not None:
            on_turn(entry)

    try:
        while len(entries) < max_turns:
            turn = len(entries) + 1
            pair = {"first_turn": turn, "start_offset_s": time.perf_counter() - started}
            trace["pairs"].append(pair)
            if max_turns - len(entries) == 1:
                pair.update(kind="single_remaining_turn", child_started=False, adopted=False)
                entry = await run_timed_retry_turn(session, client, adapter, messages,
                                                    executor, turn, make_cfg(messages, turn))
                accept(entry)
            else:
                pair["kind"] = "successor_pair"
                await run_successor_pair(session, client, adapter, messages, executor,
                    turn, make_cfg, mode=mode, on_turn=accept, trace=pair)
            pair["end_offset_s"] = time.perf_counter() - started
            if entries and not entries[-1]["baseline_has_tool_call"]:
                break
        return entries, trace
    finally:
        trace.update(task_end_s=time.perf_counter() - started,
            adopted=sum(bool(p.get("adopted")) for p in trace["pairs"]),
            observed_wasted_child_tokens=sum(p.get("observed_wasted_child_tokens", 0) for p in trace["pairs"]),
            errors=[e for p in trace["pairs"] for e in p.get("errors", [])],
            turn_limit_reached=bool(entries and len(entries) == max_turns and entries[-1]["baseline_has_tool_call"]))
