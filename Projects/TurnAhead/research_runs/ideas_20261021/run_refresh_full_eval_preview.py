#!/usr/bin/env python3
"""Full 155-task evaluation for SPORK / ETA-exit / stable-refresh / off.

This is the formal-validation runner. It is NOT a smoke script: it runs every
selected task for up to --max-turns turns, resumes completed records, and keeps
going after a task-level failure (a dirty D3/server state still stops the run).

Quality is the tau2 ACTION reward on the existing fixed-user-message harness,
not the official interactive tau2 task success. Report that distinction.

Example (launch in tmux by the user):
    PYTHONPATH=. python scripts/run_refresh_full_eval.py \
        --output results/refresh_full_20261021 \
        --configs spork,eta,refresh,off --scope all \
        --max-turns 8 --repetitions 1
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
import statistics
import sys
import time
import traceback
import uuid
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spork_core.stable_refresh_exit import ExitConfig, run_early_exit_turn  # noqa: E402
from spork_core.prefix_replay import PrefixHttpClient, counter_delta, COUNTERS  # noqa: E402
from spork_core.qwen import QwenToolCallAdapter  # noqa: E402
from spork_core.timed_retry import run_timed_retry_turn  # noqa: E402
from spork_core.turn_runner import RetryTurnConfig  # noqa: E402
from spork_core.vllm_client import VllmClient  # noqa: E402
from spork_core.evaluation_artifacts import (freeze_plan, read_completed, complete_record,
    archive_unfinished, planned_records, visible_final_answers)  # noqa: E402


def dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def signature(record):
    return [{k: t.get(k) for k in (
        "baseline_has_tool_call", "baseline_tool_name", "baseline_tool_args",
        "tool_result_sha256", "environment_after")} for t in record["turns"]]


def final_answers(record):
    return visible_final_answers(record)


def select_tasks(split, scope, domains=("airline", "retail")):
    wanted = {"all": {"train", "validation", "test"}}.get(scope, {scope})
    tasks = [t for t in split["tasks"] if t["domain"] in domains and t["split"] in wanted]
    return sorted(tasks, key=lambda t: (t["domain"], int(t["task_id"])))


def make_schedule(tasks, configs, repetitions):
    schedule = []
    for repeat in range(repetitions):
        for index, task in enumerate(tasks):
            shift = (index + repeat) % len(configs)
            order = configs[shift:] + configs[:shift]
            for config in order:
                schedule.append({**task, "repeat": repeat, "config": config})
    return schedule


def analyze(output: Path, manifest):
    records, unfinished = planned_records(output, manifest)
    configs = manifest["configs"]
    groups = {}
    for record in records:
        groups.setdefault((record["group_id"], record["repeat"]), {})[record["config"]] = record

    report = {"expected_runs": len(manifest["schedule"]), "finished_runs": len(records),
              "experiment_complete": len(records) == len(manifest["schedule"]),
              "unsealed_records": unfinished,
              "final_answer_comparison": "Exact visible reply, excluding reasoning; unobserved replies are explicitly marked",
              "scope": manifest["scope"], "configs": configs, "splits": {},
              "comparisons": {}, "quality_by_initial_match": {},
              "quality_scope": "fixed user message; ACTION reward, not official interactive tau2 success"}
    for split_name in ("all", "train", "validation", "test"):
        subset = [r for r in records if split_name == "all" or r["split"] == split_name]
        rows = {}
        for config in configs:
            chosen = [r for r in subset if r["config"] == config]
            good = [r for r in chosen if not r["errors"]]
            walls = sorted(r["wall_s"] for r in good)
            rewards = [r["action_reward"]["reward"] for r in good
                       if isinstance(r.get("action_reward"), dict)]
            rows[config] = {
                "completed": len(chosen), "failed": len(chosen) - len(good),
                "mean_wall_s": statistics.mean(walls) if walls else None,
                "p50_wall_s": walls[len(walls) // 2] if walls else None,
                "p95_wall_s": walls[min(len(walls) - 1, int(0.95 * len(walls)))] if walls else None,
                "action_reward_mean": statistics.mean(rewards) if rewards else None,
                "turns": sum(len(r["turns"]) for r in good),
                "exits": sum(t.get("exit", {}).get("fired", False) for r in good for t in r["turns"]),
                "refreshes": sum(t.get("exit", {}).get("refresh_count", 0) for r in good for t in r["turns"]),
                "rollbacks": sum(t.get("exit", {}).get("rolled_back", False) for r in good for t in r["turns"]),
                "turn_limit": sum(bool(r.get("turn_limit_reached")) for r in good),
                "failures": [r["group_id"] for r in chosen if r["errors"]],
            }
        report["splits"][split_name] = rows

    for split_name in ("all", "train", "validation", "test"):
        comparisons = {}
        for config in configs:
            if config == "spork":
                continue
            pairs = []
            for (group_id, repeat), by_config in groups.items():
                ref, other = by_config.get("spork"), by_config.get(config)
                if ref is None or other is None:
                    continue
                split_ok = split_name == "all" or ref["split"] == split_name
                if not split_ok:
                    continue
                valid = not ref["errors"] and not other["errors"]
                pairs.append({
                    "group_id": group_id, "split": ref["split"], "repeat": repeat, "valid": valid,
                    "initial_match": bool(ref["turns"] and ref["turns"][0].get("spec_used")),
                    "same_action_results_state": valid and signature(ref) == signature(other),
                    "same_final_answer": final_answers(ref) == final_answers(other),
                    "final_answer_observed": bool(final_answers(ref) or final_answers(other)),
                    "baseline_s": ref["wall_s"], "method_s": other["wall_s"],
                    "saved_s": ref["wall_s"] - other["wall_s"],
                    "refreshes": sum(t.get("exit", {}).get("refresh_count", 0) for t in other["turns"]),
                })
            comparisons[config] = pairs
        report["comparisons"][split_name] = comparisons

    for config, pairs in report["comparisons"].get("all", {}).items():
        rows = {}
        for matched in (True, False):
            subset = [p for p in pairs if p["valid"] and p["initial_match"] is matched]
            if subset:
                rows["matched" if matched else "unmatched"] = {
                    "n": len(subset),
                    "mean_saved_s": statistics.mean(p["saved_s"] for p in subset),
                    "quality_ok": sum(bool(p["same_action_results_state"]) for p in subset),
                    "refreshes": sum(p["refreshes"] for p in subset),
                }
        report["quality_by_initial_match"][config] = rows

    report["failures"] = [{"group_id": r["group_id"], "config": r["config"],
                           "split": r["split"], "errors": r["errors"]}
                          for r in records if r["errors"]]
    report["comparison_rule"] = "Every pair is reported; no selection by positive speedup."
    dump(output / "summary.json", report)
    lines = ["# Stable-refresh full evaluation", "",
             f"records: {len(records)}; configs: {', '.join(configs)}; scope: {manifest['scope']}", ""]
    for split_name, rows in report["splits"].items():
        lines += [f"## {split_name}", "",
                  "| config | done/failed | mean s | P50 | P95 | ACTION | exits | refreshes | rollbacks |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for config, row in rows.items():
            fmt = lambda v: "-" if v is None else f"{v:.3f}"
            lines.append(f"| {config} | {row['completed']}/{row['failed']} | {fmt(row['mean_wall_s'])} | "
                         f"{fmt(row['p50_wall_s'])} | {fmt(row['p95_wall_s'])} | "
                         f"{fmt(row['action_reward_mean'])} | {row['exits']} | {row['refreshes']} | {row['rollbacks']} |")
        lines.append("")
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    return report


async def run(args):
    import aiohttp
    from transformers import AutoTokenizer
    from spork_core.benchmarks.tau2 import (get_environment, load_system_prompt,
                                           load_tools)
    from spork_core.executors import Tau2ToolExecutor
    from spork_unified_eval import load_tau2_tasks, tau2_action_reward

    split = json.loads(args.task_split.read_text())
    selected = select_tasks(split, args.scope)
    if not selected:
        raise ValueError("No tasks selected")
    task_map = {f'{t["domain"]}:{t["task_id"]}': t for t in load_tau2_tasks(args.seed)}
    missing = [t["group_id"] for t in selected if t["group_id"] not in task_map]
    if missing:
        raise ValueError(f"Tasks missing from local tau2 loader: {missing[:5]}")

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    if not configs or len(set(configs)) != len(configs) or set(configs) - {"spork", "eta", "refresh", "off"}:
        raise ValueError("Use unique configs from spork, eta, refresh, off")
    for required in ("spork",):
        if required not in configs:
            raise ValueError("The spork reference is required")
    schedule = make_schedule(selected, configs, args.repetitions)
    sources = [Path(__file__), ROOT / "spork_core/stable_refresh_exit.py",
               ROOT / "spork_core/candidate_verification.py", ROOT / "spork_core/timed_retry.py",
               ROOT / "spork_core/prefix_replay.py", ROOT / "spork_core/turn_runner.py",
               ROOT / "spork_core/evaluation_artifacts.py", ROOT / "spork_core/executors.py",
               ROOT / "spork_core/qwen.py", ROOT / "spork_core/launch_timing.py",
               ROOT / "spork_core/vllm029_d3.py", ROOT / "launch_vllm_spork_http.py",
               ROOT / "scripts/serve_qwen32_candidate_checks.sh"]
    manifest = {
        "purpose": "stable_refresh_full_eval", "created_at": datetime.now().isoformat(),
        "model": args.model, "base_url": args.base_url, "scope": args.scope,
        "max_tokens": args.max_tokens, "context_window": 40960,
        "exit_parameters": {k: v for k, v in asdict(ExitConfig(
            stable_checks=args.stable_checks, max_checks=args.max_checks,
            max_refreshes=args.max_refreshes)).items() if k not in {"method", "enable_exit"}},
        "config_semantics": {"spork": "unchanged full-reasoning SPORK",
            "eta": "ETA-style exit with the common exit_parameters",
            "refresh": "stable candidate replacement plus the same exit rule",
            "off": "ETA instrumentation and speculation retained; only early exit disabled"},
        "measurement": "Each task starts after the same discarded one-token prompt warmup; measured wall includes turn cleanup",
        "task_split_sha256": hashlib.sha256(args.task_split.read_bytes()).hexdigest(),
        "task_material_sha256": hashlib.sha256(json.dumps(
            {t["group_id"]: task_map[t["group_id"]] for t in selected},
            sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        "policy_tools_sha256": {d: hashlib.sha256(json.dumps(
            [load_system_prompt(d), load_tools(d)], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            for d in ("airline", "retail")},
        "model_config_sha256": {name: hashlib.sha256((Path(args.model) / name).read_bytes()).hexdigest()
            for name in ("config.json", "tokenizer_config.json", "generation_config.json")
            if (Path(args.model) / name).is_file()},
        "seed": args.seed, "configs": configs, "max_turns": args.max_turns,
        "repetitions": args.repetitions, "tool_floor_s": args.tool_floor,
        "stable_checks": args.stable_checks, "max_checks": args.max_checks,
        "max_refreshes": args.max_refreshes, "tasks": selected, "schedule": schedule,
        "quality_scope": "fixed user message; ACTION reward, not official interactive tau2 success",
        "code_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sources},
    }
    manifest = freeze_plan(args.output, manifest, ROOT, sources)
    print(f"{len(selected)} tasks x {len(configs)} configs x {args.repetitions} repeats = {len(schedule)} runs")
    if args.dry_run:
        print("dry run; schedule written to", args.output / "manifest.json")
        return

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    adapters = {d: QwenToolCallAdapter(tok, load_tools(d)) for d in ("airline", "retail")}
    policies = {d: load_system_prompt(d) for d in adapters}
    client = VllmClient(args.base_url, args.model, 900)
    http = PrefixHttpClient(args.base_url, args.model, 900)
    async with aiohttp.ClientSession(
            trust_env=False,
            connector=aiohttp.TCPConnector(force_close=True)) as session:
        status = await http.status(session)
        if (status.get("server_config", {}).get("batch_invariant_requested") is not True
                or not status.get("has_drafter") or status.get("active_requests") != 0
                or status.get("pending_verifications") != 0):
            raise RuntimeError("An idle batch-invariant D3 server is required")
        dump(args.output / "server_start.json", status)
        started_all = time.perf_counter()
        completed = 0
        for position, item in enumerate(schedule):
            path = args.output / "records" / (
                f'{item["domain"]}_{item["task_id"]}_r{item["repeat"]}_{item["config"]}.json')
            if read_completed(path, item) is not None:
                continue  # Completed failures are preserved, not silently retried.
            previous_unsealed = archive_unfinished(path)
            adapter = adapters[item["domain"]]
            task = task_map[item["group_id"]]
            messages = [{"role": "system", "content": policies[item["domain"]]},
                        {"role": "user", "content": task["user_message"]}]
            executor = Tau2ToolExecutor(get_environment(item["domain"]), floor_s=args.tool_floor)
            record = {**item, "turns": [], "errors": [], "previous_unsealed": previous_unsealed}
            print(f'[{position + 1}/{len(schedule)}] {item["split"]} {item["group_id"]} {item["config"]}', flush=True)
            initial_ids = tok.encode(adapter.render_main_prompt(messages, enable_thinking=True), add_special_tokens=False)
            await http.complete(session, initial_ids, max_tokens=1, seed=args.seed)
            status = await http.status(session)
            if status.get("active_requests") != 0 or status.get("pending_verifications") != 0:
                raise RuntimeError("Server is not idle after discarded warmup")
            started = time.perf_counter()
            try:
                for turn in range(1, args.max_turns + 1):
                    request_id = "refresh-full-" + uuid.uuid4().hex
                    main_prompt = adapter.render_main_prompt(messages, enable_thinking=True)
                    prompt_len = len(tok.encode(main_prompt, add_special_tokens=False))

                    async def register(call, _prompt, rid=request_id, plen=prompt_len):
                        return await http.set_draft(session, rid, plen, call, tok)

                    async def clear():
                        result = await http.clear(session)
                        return {**await http.status(session), "clear_status": result["status"]}

                    cfg = RetryTurnConfig(
                        max_retries=5, confidence_threshold=.9, retry_token_step=50,
                        probe_max_tokens=160, baseline_max_tokens=args.max_tokens,
                        context_window_tokens=40960, enable_thinking=True, seed=args.seed,
                        d3_set=register, d3_clear=clear, main_request_id=request_id,
                        launch_mode="original")
                    if item["config"] == "spork":
                        entry = await run_timed_retry_turn(
                            session, client, adapter, messages, executor, turn, cfg)
                    else:
                        exit_cfg = ExitConfig(
                            method="refresh" if item["config"] == "refresh" else "eta",
                            stable_checks=args.stable_checks, max_checks=args.max_checks,
                            max_refreshes=args.max_refreshes,
                            enable_exit=item["config"] != "off")
                        entry = await run_early_exit_turn(
                            session, client, adapter, messages, executor, turn, cfg, exit_cfg)
                    record["turns"].append(entry)
                    dump(args.output / "in_progress.json", record)
                    if not entry["baseline_has_tool_call"]:
                        break
            except Exception as error:
                record["errors"].append(f"{type(error).__name__}: {error}")
                record["traceback"] = traceback.format_exc()
            record["wall_s"] = time.perf_counter() - started
            record["turn_limit_reached"] = bool(
                len(record["turns"]) == args.max_turns
                and record["turns"] and record["turns"][-1].get("baseline_has_tool_call"))
            try:
                after = await http.status(session)
                record["d3_counters"] = counter_delta(status, after)
                record["status_after"] = after
                status = after
            except Exception as error:
                record["errors"].append(f"server_status: {type(error).__name__}: {error}")
                after = {}
            for key in ("accounting_errors", "dropped_injections"):
                if record.get("d3_counters", {}).get(key):
                    record["errors"].append(key)
            if any(key not in after for key in COUNTERS):
                record["errors"].append("missing_d3_counters")
            if any(value < 0 for value in record.get("d3_counters", {}).values()):
                record["errors"].append("d3_counter_reset")
            if after.get("active_requests") != 0 or after.get("pending_verifications") != 0:
                record["errors"].append("residual_d3_state")
            if not record["turns"] and not record["errors"]:
                record["errors"].append("empty_task")
            for entry in record["turns"]:
                if not entry.get("completion_valid") or entry.get("worker_errors") or entry.get("spec_cleanup_error"):
                    record["errors"].append("turn_completion_or_cleanup_error")
                if (entry.get("d3_status") or {}).get("clear_status") not in ("ok", "cleared"):
                    record["errors"].append("unconfirmed_d3_cleanup")
            try:
                record["action_reward"] = (tau2_action_reward(task.get("raw_task") or {}, record["turns"])
                                           if not record["errors"] else None)
            except Exception as score_error:
                record["action_reward"] = None
                record["errors"].append(f"reward_evaluation: {type(score_error).__name__}: {score_error}")
            record["completed_at"] = datetime.now().isoformat()
            complete_record(path, record)
            completed += 1
            mean_s = (time.perf_counter() - started_all) / completed
            print(f'  {record["wall_s"]:.1f}s turns={len(record["turns"])} '
                  f'errors={record["errors"]} ETA≈{mean_s * (len(schedule) - position - 1) / 3600:.1f}h', flush=True)
            analyze(args.output, manifest)
            if any(e in record["errors"] for e in ("accounting_errors", "dropped_injections",
                    "missing_d3_counters", "d3_counter_reset", "residual_d3_state", "unconfirmed_d3_cleanup")):
                raise RuntimeError("Server/D3 integrity failed; sealed result saved before stopping")
        dump(args.output / "server_end.json", await http.status(session))
        save_stats = {"completed": completed, "total": len(schedule),
                      "wall_s": time.perf_counter() - started_all}
        dump(args.output / "run_stats.json", save_stats)
        analyze(args.output, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="/mnt/data/zhangshuai/LLMs/Models/Qwen3-32B")
    parser.add_argument("--base-url", default="http://127.0.0.1:8017/v1")
    parser.add_argument("--task-split", type=Path,
                        default=ROOT / "prefix_data/datasets/controller_stage3_20261008_213013_wi87e7du/task_split.json")
    parser.add_argument("--scope", choices=("all", "train", "validation", "test"), default="all")
    parser.add_argument("--configs", default="spork,eta,refresh,off")
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--tool-floor", type=float, default=2.0)
    parser.add_argument("--stable-checks", type=int, default=3)
    parser.add_argument("--max-checks", type=int, default=15)
    parser.add_argument("--max-refreshes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    ExitConfig(stable_checks=args.stable_checks, max_checks=args.max_checks,
               max_refreshes=args.max_refreshes).validate()
    if args.repetitions < 1 or args.max_turns < 1:
        parser.error("repetitions and max-turns must be positive")
    port = urlparse(args.base_url).port or 80
    if args.dry_run:
        asyncio.run(run(args))
        return
    # Several checked-out SPORK copies may share the same local serving port.
    locks = {(ROOT / "results" / f"controller_online_{port}.lock").resolve()}
    for sibling in ROOT.parent.iterdir():
        if sibling.name.startswith("spork") and (sibling / "spork_core").is_dir():
            locks.add((sibling / "results" / f"controller_online_{port}.lock").resolve())
    with ExitStack() as stack:
        for path in sorted(set(locks)):
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = stack.enter_context(path.open("a+"))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        asyncio.run(run(args))


if __name__ == "__main__":
    main()
