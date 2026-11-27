#!/usr/bin/env python3
"""Train-only two-turn online comparison of full SPORK and successor prefetch.

Fixed user inputs isolate inference/state reuse before interactive integration.
This runner makes no task-success claim. Default invocation only freezes a plan.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import ExitStack
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

from spork_core.evaluation_artifacts import (
    archive_unfinished, complete_record, digest, freeze_plan, planned_records,
    read_completed, record_path, visible_final_answers, write_json)
from spork_core.executors import Tau2ToolExecutor
from spork_core.prefix_replay import COUNTERS, PrefixHttpClient, counter_delta
from spork_core.qwen import QwenToolCallAdapter
from spork_core.successor_prefetch import run_successor_pair
from spork_core.turn_runner import RetryTurnConfig
from spork_core.vllm_client import VllmClient
from spork_core.benchmarks.tau2 import get_environment, load_system_prompt, load_tools
from spork_unified_eval import load_tau2_tasks


def signature(record):
    return [{k: t.get(k) for k in ("baseline_has_tool_call", "baseline_tool_name", "baseline_tool_args",
                                  "tool_result_sha256", "environment_after")} for t in record["turns"]]


def analyze(output, manifest):
    records, unfinished = planned_records(output, manifest)
    groups = {}
    for row in records:
        groups.setdefault((row["group_id"], row["repeat"]), {})[row["config"]] = row
    comparisons = []
    for (task, repeat), methods in groups.items():
        base = methods.get("spork")
        if base is None:
            continue
        for mode, row in methods.items():
            if mode == "spork":
                continue
            good = not base["errors"] and not row["errors"]
            comparisons.append({"group_id": task, "repeat": repeat, "config": mode,
                "valid": good, "baseline_s": base["wall_s"], "method_s": row["wall_s"],
                "saved_s": base["wall_s"] - row["wall_s"],
                "same_actions_results_state": good and signature(base) == signature(row),
                "same_actor_tokens": good and [t["main_token_ids"] for t in base["turns"]]
                    == [t["main_token_ids"] for t in row["turns"]],
                "visible_answer_observed": bool(visible_final_answers(base) or visible_final_answers(row)),
                "same_visible_answers": visible_final_answers(base) == visible_final_answers(row),
                "prefetch": row["prefetch"]})
    totals = {}
    for mode in manifest["configs"]:
        chosen = [r for r in records if r["config"] == mode]
        good = [r for r in chosen if not r["errors"]]
        totals[mode] = {"recorded": len(chosen), "failed": len(chosen) - len(good),
                        "mean_s": statistics.mean(r["wall_s"] for r in good) if good else None,
                        "adopted": sum(r["prefetch"].get("adopted", False) for r in chosen),
                        "observed_wasted_child_tokens": sum(r["prefetch"].get("observed_wasted_child_tokens", 0) for r in chosen)}
    report = {"expected": len(manifest["schedule"]), "recorded": len(records),
              "complete": len(records) == len(manifest["schedule"]), "unsealed": unfinished,
              "totals": totals, "comparisons": comparisons,
              "failures": [{"group_id": r["group_id"], "config": r["config"], "errors": r["errors"]} for r in records if r["errors"]],
              "scope": manifest["scope"]}
    write_json(output / "summary.json", report)
    return report


def plan(args):
    split = json.loads(args.task_split.read_text())
    split_map = {t["group_id"]: t for t in split["tasks"]}
    tasks = [split_map[g] for g in args.task_ids.split(",")]
    if len(tasks) != len({t["group_id"] for t in tasks}) or any(t["split"] != "train" for t in tasks):
        raise ValueError("Select unique training tasks only")
    modes = args.configs.split(",")
    if "spork" not in modes or len(modes) != len(set(modes)) or set(modes) - {"spork", "raw", "confirmed"}:
        raise ValueError("Use unique modes from spork, raw, confirmed, including spork")
    schedule = []
    for repeat in range(args.repetitions):
        for i, task in enumerate(tasks):
            shift = (i + repeat) % len(modes)
            schedule.extend({**task, "repeat": repeat, "config": mode}
                            for mode in modes[shift:] + modes[:shift])
    task_map = {f'{t["domain"]}:{t["task_id"]}': t for t in load_tau2_tasks(args.seed)}
    sources = sorted(set([Path(__file__), ROOT / "spork_unified_eval.py", ROOT / "launch_vllm_spork_http.py",
                          ROOT / "scripts/serve_qwen32_candidate_checks.sh"]
                         + list((ROOT / "spork_core").rglob("*.py"))
                         + list((ROOT / "coder/scripts").glob("tau2_tool_parser.py"))))
    manifest = {
        "purpose": "full_actor_successor_prefetch_smoke", "created_at": datetime.now().isoformat(),
        "tasks": tasks, "schedule": schedule, "configs": modes, "repetitions": args.repetitions,
        "model": args.model, "base_url": args.base_url, "seed": args.seed,
        "max_tokens": args.max_tokens, "max_turns": 2, "tool_floor_s": args.tool_floor,
        "context_window": 40960, "confirmation_threshold": .9,
        "sampling": "greedy; same target model/seed/stop rules, parent full reasoning",
        "child_d3": "Disabled while uncommitted; original D1/D2 retained. Parent and fallback use D3.",
        "selection": "Default: first eligible training task per domain; includes stale-candidate and matching-candidate cases observed in prior runs. No validation/test tasks.",
        "scope": "Two live fixed-user turns, no user simulator. Tests acceleration and exact Actor/action/state agreement, not task success.",
        "measurement": "Pair wall includes all admission, rejected work and cleanup. One discarded token warmup on initial prompt before each pair.",
        "task_split_sha256": digest(args.task_split),
        "task_material_sha256": hashlib.sha256(json.dumps([task_map[t["group_id"]] for t in tasks], sort_keys=True).encode()).hexdigest(),
        "policy_tools_sha256": {d: hashlib.sha256(json.dumps([load_system_prompt(d), load_tools(d)], sort_keys=True).encode()).hexdigest() for d in {t["domain"] for t in tasks}},
        "model_config_sha256": {f: digest(Path(args.model) / f) for f in ["config.json", "tokenizer_config.json", "generation_config.json"]},
        "code_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources},
    }
    return freeze_plan(args.output, manifest, ROOT, sources), task_map


async def run(args, manifest, task_map):
    import aiohttp
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    client = VllmClient(args.base_url, args.model, 900)
    http = PrefixHttpClient(args.base_url, args.model, 900)
    async with aiohttp.ClientSession(trust_env=False) as session:
        status = await http.status(session)
        if (not status.get("server_config", {}).get("batch_invariant_requested")
                or status.get("active_requests") != 0 or status.get("pending_verifications") != 0):
            raise RuntimeError("An unoccupied batch-invariant D3 server is required")
        write_json(args.output / "server_start.json", status)
        for index, item in enumerate(manifest["schedule"]):
            path = record_path(args.output, item)
            if read_completed(path, item) is not None:
                continue
            record = {**item, "turns": [], "errors": [], "prefetch": {},
                      "previous_unsealed": archive_unfinished(path)}
            task = task_map[item["group_id"]]
            adapter = QwenToolCallAdapter(tok, load_tools(item["domain"]))
            messages = [{"role": "system", "content": load_system_prompt(item["domain"])},
                        {"role": "user", "content": task["user_message"]}]
            executor = Tau2ToolExecutor(get_environment(item["domain"]), floor_s=args.tool_floor)

            def make_cfg(current_messages, turn):
                prompt = adapter.render_main_prompt(current_messages, enable_thinking=True)
                prompt_len = len(tok.encode(prompt, add_special_tokens=False))
                request_id = "successor-" + uuid.uuid4().hex
                async def register(call, _prompt):
                    return await http.set_draft(session, request_id, prompt_len, call, tok)
                async def clear():
                    result = await http.clear(session)
                    return {**await http.status(session), "clear_status": result["status"]}
                return RetryTurnConfig(max_retries=5, confidence_threshold=.9, retry_token_step=50,
                    probe_max_tokens=160, baseline_max_tokens=args.max_tokens, context_window_tokens=40960,
                    enable_thinking=True, seed=args.seed, d3_set=register, d3_clear=clear,
                    main_request_id=request_id, launch_mode="original")

            def checkpoint(entry):
                record["turns"].append(entry)
                write_json(args.output / "in_progress.json", record)
                print(f'  turn {entry["turn"]}: {entry.get("baseline_tool_name") or "reply"}', flush=True)

            ids = tok.encode(adapter.render_main_prompt(messages, enable_thinking=True), add_special_tokens=False)
            await http.complete(session, ids, max_tokens=1, seed=args.seed)
            status = await http.status(session)
            print(f'[{index + 1}/{len(manifest["schedule"])}] {item["group_id"]} {item["config"]}', flush=True)
            started = time.perf_counter()
            try:
                await run_successor_pair(session, client, adapter, messages, executor, 1, make_cfg,
                    mode="baseline" if item["config"] == "spork" else item["config"],
                    on_turn=checkpoint, trace=record["prefetch"])
                record["errors"].extend(record["prefetch"].get("errors", []))
            except Exception as exc:
                record["errors"].append(f"{type(exc).__name__}: {exc}")
                record["traceback"] = traceback.format_exc()
            record["wall_s"] = time.perf_counter() - started
            try:
                after = await http.status(session)
                record["d3_counters"] = counter_delta(status, after)
                dirty = (any(k not in after for k in COUNTERS)
                         or any(v < 0 for v in record["d3_counters"].values())
                         or any(record["d3_counters"].get(k) for k in ("accounting_errors", "dropped_injections"))
                         or after.get("active_requests") != 0 or after.get("pending_verifications") != 0)
            except Exception as exc:
                dirty = True
                record["d3_counters"] = {}
                record["errors"].append(f"server_status: {type(exc).__name__}: {exc}")
            if dirty:
                record["errors"].append("server_integrity")
            for t in record["turns"]:
                if not t.get("completion_valid") or t.get("spec_cleanup_error"):
                    record["errors"].append("turn_completion_or_cleanup")
                if (t.get("d3_status") or {}).get("clear_status") not in ("ok", "cleared"):
                    record["errors"].append("unconfirmed_d3_cleanup")
            record["completed_at"] = datetime.now().isoformat()
            complete_record(path, record)
            analyze(args.output, manifest)
            print(json.dumps({"wall_s": record["wall_s"], "admission": record["prefetch"].get("admission"),
                              "adopted": record["prefetch"].get("adopted"), "rejection": record["prefetch"].get("rejection"),
                              "errors": record["errors"]}), flush=True)
            if dirty:
                raise RuntimeError("Server integrity failure; sealed outcome retained")
        write_json(args.output / "server_end.json", await http.status(session))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--task-ids", default="airline:2,retail:0")
    p.add_argument("--configs", default="spork,raw,confirmed")
    p.add_argument("--task-split", type=Path, default=ROOT.parent / "spork/prefix_data/datasets/controller_stage3_20261008_213013_wi87e7du/task_split.json")
    p.add_argument("--model", default="/mnt/data/zhangshuai/LLMs/Models/Qwen3-32B")
    p.add_argument("--base-url", default="http://127.0.0.1:8017/v1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repetitions", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--tool-floor", type=float, default=2.)
    p.add_argument("--run", action="store_true")
    args = p.parse_args()
    if not 1 <= args.repetitions <= 3 or args.max_tokens <= 0 or args.tool_floor < 0:
        p.error("Use 1-3 repeats, positive generation budget and nonnegative tool floor")
    manifest, task_map = plan(args)
    print(f'Frozen {len(manifest["schedule"])} pairs at {args.output}', flush=True)
    if not args.run:
        return
    if all(read_completed(record_path(args.output, item), item) is not None for item in manifest["schedule"]):
        analyze(args.output, manifest)
        return
    port = urlparse(args.base_url).port or 80
    with ExitStack() as stack:
        for sibling in sorted(ROOT.parent.glob("spork*")):
            if (sibling / "spork_core").is_dir():
                lock = sibling / "results" / f"controller_online_{port}.lock"
                lock.parent.mkdir(parents=True, exist_ok=True)
                fcntl.flock(stack.enter_context(lock.open("a+")).fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        asyncio.run(run(args, manifest, task_map))


if __name__ == "__main__":
    main()
