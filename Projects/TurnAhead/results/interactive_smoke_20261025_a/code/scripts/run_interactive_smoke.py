#!/usr/bin/env python3
"""Bounded, train-only interactive tau2 comparison; prepare unless --run.

Uses the native orchestrator and evaluator with a local Qwen user simulator.
This differs from the published simulator/model setup. All failures and task
limits are retained. Actor time and user-simulator time are reported separately.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime
import fcntl
import json
from pathlib import Path
import statistics
import sys
import time
import traceback
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spork_core import paths
from spork_core.evaluation_artifacts import (
    archive_unfinished, complete_record, digest, freeze_plan, planned_records,
    read_completed, record_path, write_json)
from spork_core.interactive_tau2 import (
    InteractiveAgent, LocalRuntime, LocalUserSimulator, SpeculativeOrchestrator)
from spork_core.prefix_replay import COUNTERS, counter_delta
from spork_core.stable_refresh_exit import ExitConfig

from tau2.data_model.tasks import Task
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.registry import registry


def summarize(output, manifest):
    records, unsealed = planned_records(output, manifest)
    by_method = {}
    for mode in manifest["configs"]:
        rows = [r for r in records if r["config"] == mode]
        good = [r for r in rows if not r["errors"]]
        by_method[mode] = {
            "recorded": len(rows), "failed": len(rows) - len(good),
            "reward_sum": sum(r["reward_info"]["reward"] for r in good),
            "mean_actor_s": statistics.mean(r["actor_wall_s"] for r in good) if good else None,
            "mean_interaction_s": statistics.mean(r["wall_s"] for r in good) if good else None,
            "user_stop_count": sum(r.get("termination_reason") == "user_stop" for r in good),
            "terminations": [r.get("termination_reason") for r in rows],
        }
    result = {"expected": len(manifest["schedule"]), "recorded": len(records),
              "complete": len(records) == len(manifest["schedule"]),
              "unsealed": unsealed, "methods": by_method,
              "scope": manifest["scope"]}
    write_json(output / "summary.json", result)
    return result


def make_plan(args):
    split = json.loads(args.task_split.read_text())
    by_id = {t["group_id"]: t for t in split["tasks"]}
    selected = [by_id[k] for k in args.task_ids.split(",")]
    if (len(selected) != len({t["group_id"] for t in selected})
            or any(t["split"] != "train" or t["domain"] not in {"airline", "retail"} for t in selected)):
        raise ValueError("Select unique airline/retail training tasks only")
    configs = args.configs.split(",")
    if set(configs) - {"spork", "eta", "refresh"} or len(configs) != len(set(configs)) or "spork" not in configs:
        raise ValueError("Use unique configs from spork, eta, refresh, including spork")
    schedule = []
    for repeat in range(args.repetitions):
        for index, task in enumerate(selected):
            shift = (index + repeat) % len(configs)
            schedule.extend({**task, "config": c, "repeat": repeat}
                            for c in configs[shift:] + configs[:shift])
    sources = sorted(set([Path(__file__), ROOT / "launch_vllm_spork_http.py",
                          ROOT / "scripts/serve_qwen32_candidate_checks.sh"]
                         + list((ROOT / "spork_core").rglob("*.py"))
                         + list((ROOT / "tau2-bench/src/tau2").rglob("*.py"))))
    data = [p for domain in {t["domain"] for t in selected}
            for p in (paths.TAU2_ROOT / "data/tau2/domains" / domain).rglob("*") if p.is_file()]
    data += [p for p in (paths.TAU2_ROOT / "data/tau2/user_simulator").rglob("*") if p.is_file()]
    data += [args.task_split]
    model_files = [Path(args.model) / f for f in ["config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"]]
    manifest = {
        "purpose": "interactive_train_smoke", "created_at": datetime.now().isoformat(),
        "scope": "Native tau2 interaction/evaluator; local Qwen3-32B user, temperature 0, thinking off. Training diagnostics, not published benchmark reproduction.",
        "model": args.model, "base_url": args.base_url, "configs": configs,
        "tasks": selected, "schedule": schedule, "seed": args.seed,
        "max_steps": args.max_steps, "max_errors": 5, "max_tokens": args.max_tokens,
        "user_max_tokens": args.user_max_tokens, "context_window": 40960,
        "timeout_s": args.timeout_s, "tool_floor_s": args.tool_floor,
        "exit_parameters": asdict(ExitConfig()),
        "evaluation": "EvaluationType.ALL with strict native tool replay and task reward_basis",
        "actor_input": "Policy, tool schemas, visible user turns, actual tool observations. No task instructions or evaluation criteria.",
        "measurement": "No cross-participant overlap; Actor and user share one server sequentially. Actor step includes branch cleanup; simulation wall includes user generation; evaluation is timed separately.",
        "selection_reason": "airline:13 diagnoses missed transfer in fixed-message train run; retail:3 had refresh opportunity. Neither is an unbiased validation sample.",
        "input_sha256": {str(p): digest(p) for p in sorted(set(data + model_files))},
        "code_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources},
    }
    return freeze_plan(args.output, manifest, ROOT, sources)


def idle_status(runtime):
    status = runtime.run(runtime.http.status(runtime.session))
    if (any(k not in status for k in COUNTERS) or status.get("active_requests") != 0
            or status.get("pending_verifications") != 0
            or not status.get("server_config", {}).get("batch_invariant_requested")):
        raise RuntimeError("An idle batch-invariant D3 server is required")
    return status


def run(args, manifest):
    from loguru import logger
    from transformers import AutoTokenizer
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    runtime = LocalRuntime(args.base_url, args.model)
    try:
        status = idle_status(runtime)
        write_json(args.output / "server_start.json", status)
        for index, item in enumerate(manifest["schedule"]):
            path = record_path(args.output, item)
            if read_completed(path, item) is not None:
                continue
            record = {**item, "turns": [], "errors": [],
                      "previous_unsealed": archive_unfinished(path)}
            raw = json.loads((paths.TAU2_ROOT / f'data/tau2/domains/{item["domain"]}/tasks.json').read_text())
            task = Task.model_validate(next(t for t in raw if str(t["id"]) == str(item["task_id"])))
            if "NL_ASSERTION" in str(task.evaluation_criteria.reward_basis).upper():
                raise ValueError("This bounded local smoke must not invoke a remote LLM judge")
            env = registry.get_env_constructor(item["domain"])()
            if env.user_tools is not None and env.get_user_tools():
                raise ValueError("User-side tools are outside this adapter's scope")
            agent = InteractiveAgent(env, runtime, tokenizer, item["config"],
                seed=args.seed, max_tokens=args.max_tokens, floor_s=args.tool_floor)
            user = LocalUserSimulator(runtime, tokenizer, str(task.user_scenario), max_tokens=args.user_max_tokens)
            orchestrator = SpeculativeOrchestrator(domain=item["domain"], agent=agent,
                user=user, environment=env, task=task, max_steps=args.max_steps,
                max_errors=5, seed=args.seed + item["repeat"],
                validate_communication=True, timeout=args.timeout_s)
            agent.orchestrator = orchestrator

            def checkpoint(entry):
                record["turns"] = agent.turns
                record["user_calls"] = user.calls
                write_json(args.output / "in_progress.json", record)
                print(f'  actor {entry["turn"]}: {entry.get("baseline_tool_name") or "reply"}; '
                      f'{entry["actor_step_wall_s"]:.1f}s', flush=True)

            agent.on_turn = checkpoint
            print(f'[{index + 1}/{len(manifest["schedule"])}] {item["group_id"]} {item["config"]}', flush=True)
            status = idle_status(runtime)
            started = time.perf_counter()
            try:
                simulation = orchestrator.run()
                record["wall_s"] = time.perf_counter() - started
                record["termination_reason"] = str(simulation.termination_reason.value)
                # Save the entire conversation before evaluation, including capped runs.
                write_json(args.output / "trajectories" / path.name, simulation.model_dump(mode="json"))
                eval_start = time.perf_counter()
                reward = evaluate_simulation(simulation, task, EvaluationType.ALL,
                                             solo_mode=False, domain=item["domain"], strict_replay=True)
                record["evaluation_wall_s"] = time.perf_counter() - eval_start
                record["reward_info"] = reward.model_dump(mode="json")
            except Exception as exc:
                record["errors"].append(f"{type(exc).__name__}: {exc}")
                record["traceback"] = traceback.format_exc()
                record.setdefault("wall_s", time.perf_counter() - started)
                write_json(args.output / "trajectories" / path.name,
                           {"partial": True, "messages": [m.model_dump(mode="json") for m in orchestrator.trajectory]})
            record["turns"] = agent.turns
            record["user_calls"] = user.calls
            record["actor_wall_s"] = sum(t["actor_step_wall_s"] for t in agent.turns)
            record["user_wall_s"] = sum(c["wall_s"] for c in user.calls)
            dirty = False
            try:
                after = idle_status(runtime)
                record["d3_counters"] = counter_delta(status, after)
                if any(v < 0 for v in record["d3_counters"].values()) or any(
                        record["d3_counters"].get(k) for k in ("accounting_errors", "dropped_injections")):
                    raise RuntimeError("D3 accounting/drop/reset failure")
            except Exception as exc:
                dirty = True
                record["errors"].append(f"server_integrity: {exc}")
            record["completed_at"] = datetime.now().isoformat()
            complete_record(path, record)
            report = summarize(args.output, manifest)
            print(json.dumps({"wall_s": record["wall_s"], "reward": record.get("reward_info", {}).get("reward"),
                              "termination": record.get("termination_reason"), "errors": record["errors"]}), flush=True)
            if dirty or record["errors"]:
                raise RuntimeError("Smoke stopped after a sealed failure; inspect before another run")
        write_json(args.output / "server_end.json", idle_status(runtime))
        print(json.dumps(report, ensure_ascii=False), flush=True)
    finally:
        runtime.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--task-ids", default="airline:13,retail:3")
    p.add_argument("--configs", default="spork,eta,refresh")
    p.add_argument("--task-split", type=Path, default=ROOT.parent / "spork/prefix_data/datasets/controller_stage3_20261008_213013_wi87e7du/task_split.json")
    p.add_argument("--model", default="/mnt/data/zhangshuai/LLMs/Models/Qwen3-32B")
    p.add_argument("--base-url", default="http://127.0.0.1:8017/v1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repetitions", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--user-max-tokens", type=int, default=512)
    p.add_argument("--timeout-s", type=float, default=1800.)
    p.add_argument("--tool-floor", type=float, default=2.)
    p.add_argument("--run", action="store_true")
    args = p.parse_args()
    if min(args.repetitions, args.max_steps, args.max_tokens, args.user_max_tokens, args.timeout_s) <= 0 or args.tool_floor < 0:
        p.error("Positive budgets and nonnegative tool floor are required")
    manifest = make_plan(args)
    print(f'Frozen {len(manifest["schedule"])} runs at {args.output}', flush=True)
    if not args.run:
        return
    port = urlparse(args.base_url).port or 80
    with ExitStack() as stack:
        for sibling in sorted(ROOT.parent.glob("spork*")):
            if not (sibling / "spork_core").is_dir():
                continue
            lock = sibling / "results" / f"controller_online_{port}.lock"
            lock.parent.mkdir(parents=True, exist_ok=True)
            fcntl.flock(stack.enter_context(lock.open("a+")).fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args, manifest)


if __name__ == "__main__":
    main()
