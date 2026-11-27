#!/usr/bin/env python3
"""Bounded Qwen3.5-9B SPORK / ETA / stable-refresh smoke.

Fixed user messages, native Qwen3.5 XML tool calls, no D3 sidecar. Measures
real online wall time and action/state agreement with the SPORK reference; this
is not official interactive tau2 task success.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime
import fcntl
import hashlib
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
from spork_core.prefix_replay import PrefixHttpClient
from spork_core.qwen35_native import NativeQwen35Adapter
from spork_core.stable_refresh_exit import ExitConfig, run_early_exit_turn
from spork_core.timed_retry import run_timed_retry_turn
from spork_core.turn_runner import RetryTurnConfig
from spork_core.vllm_client import VllmClient


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def signature(record):
    return [{key: t.get(key) for key in (
        "baseline_has_tool_call", "baseline_tool_name", "baseline_tool_args",
        "tool_result_sha256", "environment_after")} for t in record["turns"]]


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
            comparisons[mode].append({
                "group_id": group[0], "repeat": group[1], "valid": valid,
                "same_action_results_state": valid and signature(reference) == signature(other),
                "same_final_answer": valid and final_answers(reference) == final_answers(other),
                "baseline_s": reference["wall_s"], "method_s": other["wall_s"],
                "saved_s": reference["wall_s"] - other["wall_s"],
                "refresh_blocked": sum(t.get("exit", {}).get("refresh_blocked", 0)
                                       for t in other["turns"]),
            })
    totals = {}
    for method in modes:
        rows = [r for r in records if r["method"] == method]
        good = [r for r in rows if not r["errors"]]
        totals[method] = {
            "completed": len(rows), "failed": len(rows) - len(good),
            "mean_wall_s": statistics.mean(r["wall_s"] for r in good) if good else None,
            "exits": sum(t.get("exit", {}).get("fired", False) for r in good for t in r["turns"]),
            "rollbacks": sum(t.get("exit", {}).get("rolled_back", False) for r in good for t in r["turns"]),
            "refreshes": sum(t.get("exit", {}).get("refresh_count", 0) for r in good for t in r["turns"]),
            "refresh_blocked": sum(t.get("exit", {}).get("refresh_blocked", 0) for r in good for t in r["turns"]),
            "turns": sum(len(r["turns"]) for r in good)}
    eta_pairs = []
    for group, values in groups.items():
        if "eta" in values and "refresh" in values:
            base, changed = values["eta"], values["refresh"]
            eta_pairs.append({
                "group_id": group[0], "repeat": group[1],
                "valid": not base["errors"] and not changed["errors"],
                "same_action_results_state": signature(base) == signature(changed),
                "same_final_answer": final_answers(base) == final_answers(changed),
                "eta_s": base["wall_s"], "refresh_s": changed["wall_s"],
                "saved_s": base["wall_s"] - changed["wall_s"]})
    return {"runs": len(records), "totals": totals, "comparisons": comparisons,
            "refresh_vs_eta": eta_pairs,
            "failures": [{"group_id": r["group_id"], "method": r["method"], "errors": r["errors"]}
                         for r in records if r["errors"]],
            "scope": "Qwen3.5-9B training-only smoke; fixed user message; no official task success claim.",
            "comparison_rule": "Report every pair including mismatches and failures; no selection by positive speedup."}


async def run(args):
    import aiohttp
    from transformers import AutoTokenizer
    from spork_core.benchmarks.tau2 import (build_user_message, get_environment,
                                           load_system_prompt, load_tasks, load_tools)
    from spork_core.executors import Tau2ToolExecutor

    args.output.mkdir(parents=True, exist_ok=False)
    split = json.loads(args.task_split.read_text())
    if args.task_ids:
        wanted = {x.strip() for x in args.task_ids.split(",") if x.strip()}
        selected = [t for t in split["tasks"] if t["group_id"] in wanted]
        missing = wanted - {t["group_id"] for t in selected}
        if missing:
            raise ValueError(f"Unknown task ids: {sorted(missing)}")
    else:
        selected = []
        for domain in ("airline", "retail"):
            pool = sorted((t for t in split["tasks"]
                           if t["domain"] == domain and t["split"] == "train"),
                          key=lambda t: int(t["task_id"]))
            selected.extend(pool[args.task_offset:args.task_offset + args.per_domain])
    schedule = []
    for repeat in range(args.repetitions):
        for index, task in enumerate(selected):
            shift = (index + repeat) % len(args.modes)
            order = args.modes[shift:] + args.modes[:shift]
            for method in order:
                schedule.append({**task, "repeat": repeat, "method": method})

    sources = [Path(__file__), ROOT / "spork_core/stable_refresh_exit.py",
               ROOT / "spork_core/candidate_verification.py",
               ROOT / "spork_core/timed_retry.py", ROOT / "spork_core/prefix_replay.py",
               ROOT / "spork_core/qwen.py", ROOT / "spork_core/qwen35_native.py",
               ROOT / "spork_core/vllm_client.py",
               ROOT / "scripts/serve_qwen35_9b.sh"]
    manifest = {
        "purpose": "qwen35_refresh_smoke", "created_at": datetime.now().isoformat(),
        "model": args.model, "model_family": "qwen35", "base_url": args.base_url,
        "selection": args.task_ids or f"per-domain {args.per_domain} offset {args.task_offset}",
        "test_tasks_used": any(t["split"] == "test" for t in selected),
        "tasks": selected, "schedule": schedule, "max_turns": args.max_turns,
        "max_tokens": args.max_tokens, "seed": args.seed, "tool_floor_s": args.tool_floor,
        "presence_penalty": args.presence_penalty,
        "d3_enabled": False, "refresh_policy": args.refresh_policy,
        "probe_mode": args.probe_mode, "check_interval": args.check_interval,
        "exit_config": asdict(ExitConfig(stable_checks=args.stable_checks,
            max_checks=args.max_checks, max_refreshes=args.max_refreshes,
            refresh_policy=args.refresh_policy, probe_mode=args.probe_mode,
            check_interval=args.check_interval,
            answer_exit=("answer" in args.modes),
            answer_min_tokens=args.answer_min_tokens,
            answer_interval=args.answer_interval,
            answer_max_tokens=args.answer_max_tokens,
            answer_stable_checks=args.answer_stable_checks)),
        "quality_scope": "fixed user message; action/state checks only",
        "code_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sources if p.is_file()},
    }
    dump(args.output / "manifest.json", manifest)

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    adapters = {d: NativeQwen35Adapter(tok, load_tools(d)) for d in ("airline", "retail")}
    tasks = {f"{d}:{t['id']}": t for d in adapters for t in load_tasks(d)}
    client = VllmClient(args.base_url, args.model, 900)
    client.presence_penalty = args.presence_penalty
    client.main_stop_token_ids = (248058,)
    http = PrefixHttpClient(args.base_url, args.model, 900,
                            presence_penalty=args.presence_penalty)
    records = []
    async with aiohttp.ClientSession(
            trust_env=False,
            connector=aiohttp.TCPConnector(force_close=True)) as session:
        async with session.get(client.models_url, timeout=30) as response:
            response.raise_for_status()
            server_start = await response.json()
        if not any(m.get("id") == args.model for m in server_start.get("data", [])):
            raise RuntimeError(f"Server does not expose {args.model}: {server_start}")
        dump(args.output / "server_start.json", server_start)
        for position, item in enumerate(schedule):
            adapter = adapters[item["domain"]]
            task = tasks[item["group_id"]]
            messages = [{"role": "system", "content": load_system_prompt(item["domain"])},
                        {"role": "user", "content": build_user_message(task)}]
            executor = Tau2ToolExecutor(get_environment(item["domain"]), floor_s=args.tool_floor)
            prompt_ids = tok.encode(
                adapter.render_main_prompt(messages, enable_thinking=True),
                add_special_tokens=False)
            await http.complete(session, prompt_ids, max_tokens=1, seed=args.seed)
            record = {**item, "turns": [], "errors": []}
            path = args.output / "records" / (
                f'{item["domain"]}_{item["task_id"]}_r{item["repeat"]}_{item["method"]}.json')
            print(f'[{position + 1}/{len(schedule)}] {item["group_id"]} {item["method"]} '
                  f'r{item["repeat"]}', flush=True)
            started = time.perf_counter()
            try:
                for turn in range(1, args.max_turns + 1):
                    cfg = RetryTurnConfig(
                        max_retries=5, confidence_threshold=.9, retry_token_step=50,
                        probe_max_tokens=320, baseline_max_tokens=args.max_tokens,
                        context_window_tokens=40960, enable_thinking=True, seed=args.seed,
                        d3_set=None, d3_clear=None, main_request_id=None,
                        launch_mode="original")
                    if item["method"] == "spork":
                        entry = await run_timed_retry_turn(
                            session, client, adapter, messages, executor, turn, cfg)
                    elif item["method"] == "off":
                        entry = await run_early_exit_turn(
                            session, client, adapter, messages, executor, turn, cfg,
                            ExitConfig(method="eta", stable_checks=args.stable_checks,
                                       max_checks=args.max_checks,
                                       max_refreshes=args.max_refreshes,
                                       refresh_policy=args.refresh_policy,
                                       probe_mode=args.probe_mode,
                                       check_interval=args.check_interval,
                                       enable_exit=False))
                    elif item["method"] == "answer":
                        entry = await run_early_exit_turn(
                            session, client, adapter, messages, executor, turn, cfg,
                            ExitConfig(method="eta", stable_checks=args.stable_checks,
                                       max_checks=args.max_checks,
                                       max_refreshes=args.max_refreshes,
                                       refresh_policy=args.refresh_policy,
                                       probe_mode=args.probe_mode,
                                       check_interval=args.check_interval,
                                       answer_exit=True,
                                       answer_min_tokens=args.answer_min_tokens,
                                       answer_interval=args.answer_interval,
                                       answer_max_tokens=args.answer_max_tokens,
                                       answer_stable_checks=args.answer_stable_checks))
                    else:
                        entry = await run_early_exit_turn(
                            session, client, adapter, messages, executor, turn, cfg,
                            ExitConfig(method=item["method"], stable_checks=args.stable_checks,
                                       max_checks=args.max_checks,
                                       max_refreshes=args.max_refreshes,
                                       refresh_policy=args.refresh_policy,
                                       probe_mode=args.probe_mode,
                                       check_interval=args.check_interval))
                    record["turns"].append(entry)
                    dump(args.output / "in_progress.json", record)
                    if not entry["baseline_has_tool_call"]:
                        break
            except Exception as error:
                record["errors"].append(f"{type(error).__name__}: {error}")
                record["traceback"] = traceback.format_exc()
            record["wall_s"] = time.perf_counter() - started
            record["turn_limit_reached"] = bool(
                record["turns"] and len(record["turns"]) == args.max_turns
                and record["turns"][-1].get("baseline_has_tool_call"))
            for turn in record["turns"]:
                if turn.get("worker_errors") or turn.get("spec_cleanup_error"):
                    record["errors"].append("tool_cleanup_error")
                if not turn.get("completion_valid"):
                    record["errors"].append("invalid_completion")
            dump(path, record)
            records.append(record)
            dump(args.output / "summary.json", analyze(records, args.modes))
            print(f'  {record["wall_s"]:.3f}s; {len(record["turns"])} turns; '
                  f'errors={record["errors"]}', flush=True)
            if record["errors"] and not args.keep_going:
                raise RuntimeError("Smoke failure saved; inspect before starting another run")
        async with session.get(client.models_url, timeout=30) as response:
            dump(args.output / "server_end.json", await response.json())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="/mnt/data/zhangshuai/LLMs/Models/Qwen3.5-9B")
    p.add_argument("--base-url", default="http://127.0.0.1:8027/v1")
    p.add_argument("--task-split", type=Path,
                   default=ROOT / "prefix_data/datasets/controller_stage3_20261008_213013_wi87e7du/task_split.json")
    p.add_argument("--per-domain", type=int, default=1)
    p.add_argument("--task-offset", type=int, default=0)
    p.add_argument("--task-ids", default="",
                   help="comma-separated group ids, e.g. airline:2,airline:5")
    p.add_argument("--max-turns", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--repetitions", type=int, default=1)
    p.add_argument("--tool-floor", type=float, default=2.0)
    p.add_argument("--stable-checks", type=int, default=3)
    p.add_argument("--max-checks", type=int, default=15)
    p.add_argument("--max-refreshes", type=int, default=1)
    p.add_argument("--presence-penalty", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--refresh-policy", choices=("read_only", "block_read_write", "all"),
                   default="read_only")
    p.add_argument("--probe-mode", choices=("full", "name"), default="name")
    p.add_argument("--check-interval", type=int, default=50)
    p.add_argument("--answer-min-tokens", type=int, default=300)
    p.add_argument("--answer-interval", type=int, default=500)
    p.add_argument("--answer-max-tokens", type=int, default=512)
    p.add_argument("--answer-stable-checks", type=int, default=2)
    p.add_argument("--keep-going", action="store_true")
    p.add_argument("--modes", nargs="+",
                   choices=("spork", "eta", "refresh", "off", "answer"),
                   default=["spork", "eta", "refresh", "off"])
    args = p.parse_args()
    ExitConfig(stable_checks=args.stable_checks, max_checks=args.max_checks,
               max_refreshes=args.max_refreshes,
               refresh_policy=args.refresh_policy, probe_mode=args.probe_mode,
               check_interval=args.check_interval,
               answer_exit=("answer" in args.modes),
               answer_min_tokens=args.answer_min_tokens,
               answer_interval=args.answer_interval,
               answer_max_tokens=args.answer_max_tokens,
               answer_stable_checks=args.answer_stable_checks).validate()
    if (not 1 <= args.per_domain <= 4 or not 1 <= args.repetitions <= 3
            or not 1 <= args.max_turns <= 8 or args.task_offset < 0):
        p.error("Small smoke limits: <=4 tasks/domain, <=3 repeats, <=8 turns")
    if "spork" not in args.modes or len(set(args.modes)) != len(args.modes):
        p.error("Use unique modes including the SPORK reference")
    port = urlparse(args.base_url).port or 80
    lock = ROOT / "results" / f"controller_online_{port}.lock"
    with ExitStack() as stack:
        lock.parent.mkdir(parents=True, exist_ok=True)
        handle = stack.enter_context(lock.open("a+"))
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        asyncio.run(run(args))


if __name__ == "__main__":
    main()
