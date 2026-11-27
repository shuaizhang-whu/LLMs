#!/usr/bin/env python3
"""Measure how much concurrent generations slow each other down on the running server.

Default prints the plan only. With --run, m = 1..4 streaming requests are sent at once
(first-turn prompts of distinct tasks of one domain). Each stream's decode time is
compared with the same prompt generated alone, over the same token positions and only
while all m streams were generating.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spork_core.benchmarks.tau2 import load_system_prompt, load_tools
from spork_core.evaluation_artifacts import write_json
from spork_core.prefix_replay import PrefixHttpClient, StreamState
from spork_core.qwen import QwenToolCallAdapter
from spork_core.qwen35_native import NativeQwen35Adapter
from spork_core.gemma4 import Gemma4Adapter
from spork_unified_eval import load_tau2_tasks
from scripts.run_successor_full_eval import evaluation_session, reset_trial_cache

MIN_TOKENS = 32


def token_times(state):
    """Arrival time of every generated token; tokens of one chunk share a time."""
    times, done = [], 0
    for event in state.token_events:
        times += [event["received_at"]] * (event["end_token"] - done)
        done = event["end_token"]
    return times


async def generate(session, http, prompts, args):
    states = [StreamState(record_token_timing=True) for _ in prompts]
    await asyncio.gather(*(http.stream(session, ids, state, max_tokens=args.max_tokens, seed=args.seed)
                           for ids, state in zip(prompts, states)))
    return [token_times(state) for state in states]


def compare(together, alone):
    """Per stream: ms/token and slowdown inside the window shared by all streams."""
    start, end = max(t[0] for t in together), min(t[-1] for t in together)
    rows = []
    for times, solo in zip(together, alone):
        inside = [i for i, t in enumerate(times) if start <= t <= end and i < len(solo)]
        if len(inside) <= MIN_TOKENS:
            continue
        lo, hi = inside[0], inside[-1]
        rows.append({"tokens": hi - lo, "ms_per_token": 1e3 * (times[hi] - times[lo]) / (hi - lo),
                     "slowdown": (times[hi] - times[lo]) / (solo[hi] - solo[lo])})
    return rows


async def measure(args, prompts, extra_body=None):
    http = PrefixHttpClient(args.base_url, args.model, 900, extra_body=extra_body)
    report = {"model": args.model, "max_tokens": args.max_tokens, "repetitions": args.repetitions,
              "started_at": datetime.now().isoformat(), "rounds": []}
    async with evaluation_session() as session:
        status = await http.status(session)
        if status.get("active_requests") != 0:
            raise RuntimeError("The server is busy; run this on an idle server")
        report["server_config"] = status.get("server_config")
        await reset_trial_cache(session, http)
        for ids in prompts:
            await http.complete(session, ids, max_tokens=1, seed=args.seed)
        for repetition in range(args.repetitions):
            alone = [(await generate(session, http, [ids], args))[0] for ids in prompts]
            solo = [1e3 * (t[-1] - t[0]) / (len(t) - 1) for t in alone if len(t) > MIN_TOKENS]
            report["rounds"].append({"repetition": repetition, "streams": 1, "ms_per_token": solo})
            for m in range(2, len(prompts) + 1):
                rows = compare(await generate(session, http, prompts[:m], args), alone[:m])
                report["rounds"].append({"repetition": repetition, "streams": m, "rows": rows})
                print(f"  repetition {repetition}, {m} streams: slowdown "
                      + ", ".join(f'{r["slowdown"]:.3f}' for r in rows), flush=True)
    summary = {1: {"ms_per_token": statistics.median(x for r in report["rounds"] if r["streams"] == 1
                                                     for x in r["ms_per_token"]), "slowdown": 1.0}}
    for m in range(2, len(prompts) + 1):
        rows = [row for r in report["rounds"] if r["streams"] == m for row in r["rows"]]
        if rows:
            summary[m] = {"ms_per_token": statistics.median(r["ms_per_token"] for r in rows),
                          "slowdown": statistics.median(r["slowdown"] for r in rows), "samples": len(rows)}
    report["summary"] = summary
    write_json(args.output, report)
    for m, row in summary.items():
        print(f'{m} streams: {row["ms_per_token"]:.1f} ms/token, slowdown {row["slowdown"]:.3f}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="/mnt/data/zhangshuai/LLMs/Models/Qwen3-32B")
    p.add_argument("--model-family", choices=["qwen3", "qwen35", "gemma4"], default="qwen3")
    p.add_argument("--base-url", default="http://127.0.0.1:8017/v1")
    p.add_argument("--domain", choices=["airline", "retail"], default="retail")
    p.add_argument("--max-tokens", type=int, default=320)
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run", action="store_true")
    args = p.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    adapter = {"qwen35": NativeQwen35Adapter, "gemma4": Gemma4Adapter}.get(
        args.model_family, QwenToolCallAdapter)(tok, load_tools(args.domain))
    tasks = [t for t in load_tau2_tasks(args.seed) if t["domain"] == args.domain][:4]
    prompts = [tok.encode(adapter.render_main_prompt(
        [{"role": "system", "content": load_system_prompt(args.domain)},
         {"role": "user", "content": t["user_message"]}], enable_thinking=True), add_special_tokens=False)
        for t in tasks]
    print(f'Prepared {len(prompts)} prompts ({[len(x) for x in prompts]} tokens), '
          f'{args.repetitions} repetitions, up to {args.max_tokens} tokens per stream', flush=True)
    if args.run:
        asyncio.run(measure(args, prompts, getattr(adapter, "request_fields", None)))


if __name__ == "__main__":
    main()
