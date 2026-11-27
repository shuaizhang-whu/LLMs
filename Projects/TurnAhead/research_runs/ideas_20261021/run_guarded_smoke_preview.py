#!/usr/bin/env python3
"""Bounded SPORK / ETA-style / stable-candidate-refresh experiments.

This reuses the existing fixed-user-message harness. It measures real online
time and action/state agreement, NOT official interactive tau2 task success.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime
import hashlib
import fcntl
import json
from pathlib import Path
import statistics
import sys
import time
import traceback
import uuid
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spork_core.stable_refresh_exit import ExitConfig, run_early_exit_turn
from spork_core.prefix_replay import PrefixHttpClient, counter_delta
from spork_core.qwen import QwenToolCallAdapter
from spork_core.timed_retry import run_timed_retry_turn
from spork_core.turn_runner import RetryTurnConfig
from spork_core.vllm_client import VllmClient


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def signature(record):
    return [{key: t.get(key) for key in ("baseline_has_tool_call", "baseline_tool_name",
             "baseline_tool_args", "tool_result_sha256", "environment_after")}
             for t in record["turns"]]


def final_answers(record):
    return [t.get("baseline_text", "").rsplit("</think>", 1)[-1].strip()
            for t in record["turns"] if not t.get("baseline_has_tool_call")]


def analyze(records, modes):
    groups = {}
    for record in records:
        groups.setdefault((record["group_id"], record["repeat"]), {})[record["method"]] = record
    comparisons = {mode: [] for mode in modes if mode != "spork"}
    for group, values in groups.items():
        reference = values.get("spork")
        if reference is None:
            continue
        for mode in comparisons:
            other = values.get(mode)
            if other is None:
                continue
            valid = not reference["errors"] and not other["errors"]
            comparisons[mode].append({"group_id": group[0], "repeat": group[1], "valid": valid,
                "same_action_results_state": valid and signature(reference) == signature(other),
                "same_final_answer": valid and final_answers(reference) == final_answers(other),
                "baseline_s": reference["wall_s"], "method_s": other["wall_s"],
                "saved_s": reference["wall_s"] - other["wall_s"]})
    totals = {}
    for method in modes:
        rows = [r for r in records if r["method"] == method]
        good = [r for r in rows if not r["errors"]]
        totals[method] = {"completed": len(rows), "failed": len(rows) - len(good),
                          "mean_wall_s": statistics.mean(r["wall_s"] for r in good) if good else None,
                          "exits": sum(t.get("exit", {}).get("fired", False) for r in good for t in r["turns"]),
                          "rollbacks": sum(t.get("exit", {}).get("rolled_back", False) for r in good for t in r["turns"]),
                          "refreshes": sum(t.get("exit", {}).get("refresh_count", 0) for r in good for t in r["turns"]),
                          "turns": sum(len(r["turns"]) for r in good)}
    eta_pairs = []
    for group, values in groups.items():
        if "eta" in values and "refresh" in values:
            base, changed = values["eta"], values["refresh"]
            eta_pairs.append({"group_id": group[0], "repeat": group[1],
                "valid": not base["errors"] and not changed["errors"],
                "same_action_results_state": signature(base) == signature(changed),
                "same_final_answer": final_answers(base) == final_answers(changed),
                "eta_s": base["wall_s"], "refresh_s": changed["wall_s"],
                "saved_s": base["wall_s"] - changed["wall_s"]})
    return {"runs": len(records), "totals": totals, "comparisons": comparisons,
            "refresh_vs_eta": eta_pairs,
            "failures": [{"group_id": r["group_id"], "method": r["method"], "errors": r["errors"]}
                         for r in records if r["errors"]],
            "scope": "Training-only small online smoke. Fixed user message, capped turns. No official task success claim.",
            "comparison_rule": "Report every pair including mismatches and failures; no selection by positive speedup."}


async def run(args):
    import aiohttp
    from transformers import AutoTokenizer
    from spork_core.benchmarks.tau2 import (build_user_message, get_environment,
                                           load_system_prompt, load_tasks, load_tools)
    from spork_core.executors import Tau2ToolExecutor
    args.output.mkdir(parents=True, exist_ok=False)
    split = json.loads(args.task_split.read_text())
    selected = []
    for domain in ("airline", "retail"):
        pool = sorted((t for t in split["tasks"] if t["domain"] == domain and t["split"] == "train"),
                      key=lambda t: int(t["task_id"]))
        selected.extend(pool[args.task_offset:args.task_offset + args.per_domain])
    schedule = []
    for repeat in range(args.repetitions):
        for index, task in enumerate(selected):
            shift = (index + repeat) % len(args.modes)
            order = args.modes[shift:] + args.modes[:shift]
            for method in order:
                schedule.append({**task, "repeat": repeat, "method": method})
    sources = [Path(__file__), ROOT / "spork_core/stable_refresh_exit.py", ROOT / "spork_core/candidate_verification.py",
               ROOT / "spork_core/timed_retry.py", ROOT / "spork_core/prefix_replay.py",
               ROOT / "spork_core/vllm029_d3.py", ROOT / "scripts/serve_qwen32_candidate_checks.sh"]
    manifest = {"purpose": "early_exit_live_smoke", "created_at": datetime.now().isoformat(),
                "model": args.model, "base_url": args.base_url, "tasks": selected,
                "schedule": schedule, "max_turns": args.max_turns, "seed": 42,
                "max_tokens": args.max_tokens, "tool_floor_s": args.tool_floor,
                "selection": "numeric train IDs in each domain; fixed task_offset; no test or validation tasks",
                "task_offset": args.task_offset,
                "test_tasks_used": False, "quality_scope": "fixed user message; action/state checks only",
                "exit_config": asdict(ExitConfig(stable_checks=args.stable_checks,
                    max_checks=args.max_checks, max_refreshes=args.max_refreshes)),
                "code_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    dump(args.output / "manifest.json", manifest)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    adapters = {d: QwenToolCallAdapter(tok, load_tools(d)) for d in ("airline", "retail")}
    tasks = {f"{d}:{t['id']}": t for d in adapters for t in load_tasks(d)}
    client = VllmClient(args.base_url, args.model, 900)
    http = PrefixHttpClient(args.base_url, args.model, 900)
    records = []
    # Control-plane calls are separated by multi-second dumps; do not let
    # aiohttp reuse a connection that uvicorn's 5 s keep-alive already closed.
    async with aiohttp.ClientSession(
            trust_env=False,
            connector=aiohttp.TCPConnector(force_close=True)) as session:
        initial_status = await http.status(session)
        if (initial_status.get("server_config", {}).get("batch_invariant_requested") is not True
                or not initial_status.get("has_drafter")
                or initial_status.get("active_requests") != 0):
            raise RuntimeError("An idle batch-invariant D3 server is required")
        dump(args.output / "server_start.json", initial_status)
        for position, item in enumerate(schedule):
            adapter = adapters[item["domain"]]
            task = tasks[item["group_id"]]
            messages = [{"role": "system", "content": load_system_prompt(item["domain"])},
                        {"role": "user", "content": build_user_message(task)}]
            executor = Tau2ToolExecutor(get_environment(item["domain"]), floor_s=args.tool_floor)
            prompt_ids = tok.encode(adapter.render_main_prompt(messages, enable_thinking=True), add_special_tokens=False)
            await http.clear(session)
            await http.complete(session, prompt_ids, max_tokens=1, seed=42)
            before = await http.status(session)
            record = {**item, "turns": [], "errors": []}
            path = args.output / "records" / f"{item['domain']}_{item['task_id']}_r{item['repeat']}_{item['method']}.json"
            print(f"[{position+1}/{len(schedule)}] {item['group_id']} {item['method']} r{item['repeat']}", flush=True)
            started = time.perf_counter()
            try:
                for turn in range(1, args.max_turns + 1):
                    request_id = "exit-smoke-" + uuid.uuid4().hex
                    main_prompt = adapter.render_main_prompt(messages, enable_thinking=True)
                    prompt_len = len(tok.encode(main_prompt, add_special_tokens=False))

                    async def register(call, _prompt, rid=request_id, plen=prompt_len):
                        return await http.set_draft(session, rid, plen, call, tok)

                    async def clear():
                        result = await http.clear(session)
                        return {**await http.status(session), "clear_status": result["status"]}

                    cfg = RetryTurnConfig(max_retries=5, confidence_threshold=.9, retry_token_step=50,
                        probe_max_tokens=160, baseline_max_tokens=args.max_tokens,
                        context_window_tokens=40960, enable_thinking=True, seed=42,
                        d3_set=register, d3_clear=clear, main_request_id=request_id, launch_mode="original")
                    if item["method"] == "spork":
                        entry = await run_timed_retry_turn(session, client, adapter, messages, executor, turn, cfg)
                    elif item["method"] == "off":
                        entry = await run_early_exit_turn(session, client, adapter, messages, executor, turn, cfg,
                            ExitConfig(method="eta", stable_checks=args.stable_checks,
                                max_checks=args.max_checks, max_refreshes=args.max_refreshes,
                                enable_exit=False))
                    else:
                        entry = await run_early_exit_turn(session, client, adapter, messages, executor, turn, cfg,
                            ExitConfig(method="refresh" if item["method"].startswith("refresh") else "eta",
                                stable_checks=args.stable_checks, max_checks=args.max_checks,
                                max_refreshes=args.max_refreshes,
                                rollback_on_action_mismatch=item["method"].endswith("_guarded")))
                    record["turns"].append(entry)
                    dump(args.output / "in_progress.json", record)
                    if not entry["baseline_has_tool_call"]:
                        break
            except Exception as error:
                record["errors"].append(f"{type(error).__name__}: {error}")
                record["traceback"] = traceback.format_exc()
            record["wall_s"] = time.perf_counter() - started
            try:
                after = await http.status(session)
                delta = counter_delta(before, after)
            except Exception as status_error:
                after = {"status_error": str(status_error)}
                delta = {key: 0 for key in before if key in ("accounting_errors", "dropped_injections")}
                record["errors"].append("server_unavailable")
            record.update(d3_counters=delta, status_after=after,
                          turn_limit_reached=bool(record["turns"] and len(record["turns"]) == args.max_turns
                                                  and record["turns"][-1]["baseline_has_tool_call"]))
            for field in ("accounting_errors", "dropped_injections"):
                if delta.get(field):
                    record["errors"].append(field)
            # Prefix mismatch is a valid draft miss, but is reported explicitly.
            for t in record["turns"]:
                if t.get("worker_errors") or t.get("spec_cleanup_error"):
                    record["errors"].append("tool_cleanup_error")
                if not t.get("completion_valid"):
                    record["errors"].append("invalid_completion")
                d3 = t.get("d3_status") or {}
                if d3.get("error") or d3.get("clear_status", d3.get("status")) not in ("ok", "cleared"):
                    record["errors"].append("d3_cleanup_error")
            if after.get("active_requests") != 0 or after.get("pending_verifications") != 0:
                record["errors"].append("residual_d3_state")
            dump(path, record)
            records.append(record)
            dump(args.output / "summary.json", analyze(records, args.modes))
            print(f"  {record['wall_s']:.3f}s; {len(record['turns'])} turns; errors={record['errors']}", flush=True)
            if record["errors"]:
                raise RuntimeError("Smoke failure saved; inspect before starting another run")
        dump(args.output / "server_end.json", await http.status(session))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="/mnt/data/zhangshuai/LLMs/Models/Qwen3-32B")
    p.add_argument("--base-url", default="http://127.0.0.1:8017/v1")
    p.add_argument("--task-split", type=Path, default=ROOT / "prefix_data/datasets/controller_stage3_20261008_213013_wi87e7du/task_split.json")
    p.add_argument("--per-domain", type=int, default=1)
    p.add_argument("--task-offset", type=int, default=0)
    p.add_argument("--max-turns", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--repetitions", type=int, default=1)
    p.add_argument("--tool-floor", type=float, default=2.0)
    p.add_argument("--stable-checks", type=int, default=3)
    p.add_argument("--max-checks", type=int, default=15)
    p.add_argument("--max-refreshes", type=int, default=1)
    p.add_argument("--modes", nargs="+", choices=("spork", "eta", "refresh", "off", "eta_guarded", "refresh_guarded"), default=["spork", "eta", "refresh"])
    args = p.parse_args()
    ExitConfig(stable_checks=args.stable_checks, max_checks=args.max_checks,
               max_refreshes=args.max_refreshes).validate()
    if (not 1 <= args.per_domain <= 4 or not 1 <= args.repetitions <= 3
            or not 1 <= args.max_turns <= 8 or args.task_offset < 0):
        p.error("Small smoke limits: <=4 tasks/domain, <=3 repeats, <=8 turns")
    if "spork" not in args.modes or len(set(args.modes)) != len(args.modes):
        p.error("Use unique modes including the SPORK reference")
    port = urlparse(args.base_url).port or 80
    locks = [ROOT / "results" / f"controller_online_{port}.lock"]
    # Honor the sibling workspace's lock when it owns the same server. In the
    # spork_v2 copy the sibling is the original spork tree, not spork_v2 itself.
    sibling_name = "spork" if ROOT.name == "spork_v2" else "spork_v2"
    sibling = ROOT.parent / sibling_name / "results"
    if sibling.is_dir():
        locks.append(sibling / f"controller_online_{port}.lock")
    locks = sorted(set(locks))
    with ExitStack() as stack:
        for path in sorted(locks):
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = stack.enter_context(path.open("a+"))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        asyncio.run(run(args))


if __name__ == "__main__":
    main()
