#!/usr/bin/env python3
"""Full frozen-controller evaluation. GPU inference is user-launched in tmux.

All six policies run on fresh environments, adjacent within each task, with
counterbalanced policy order. Completed results (including failures) are sealed
and never silently retried. The same output directory resumes interrupted work.
The fixed-user-message tau2 harness reports ACTION reward, not official
interactive tau2 success. Eight rounds are a declared cap, not guaranteed task
completion. Development and reserved test groups are always reported separately.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime
import fcntl
import json
from pathlib import Path
import random
import statistics
import sys
import time
from types import SimpleNamespace
from urllib.parse import urlparse
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.check_controller_online import CONFIGS, checksum, get_status, read  # noqa: E402
from spork_core.prefix_replay import COUNTERS  # noqa: E402

SOURCE = ROOT / "results/controller_online_20261017_014410_6288ae"
SIGNATURE_FIELDS = ("baseline_has_tool_call", "baseline_tool_name", "baseline_tool_args",
                    "tool_result_sha256", "environment_after", "main_token_ids")
DECISION_STATUSES = {"ready", "disabled", "reasoning_ended_before_decision",
                     "main_finished_or_formal_action_before_decision",
                     "main_finished_or_formal_action_during_decision",
                     "main_finished_or_formal_action_while_waiting"}


def dump(path, value):
    """Atomic writes; sealed results are checked before this is called."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def log(message):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


@contextmanager
def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another evaluator holds {path}") from error
        yield


def make_schedule(tasks, repetitions, seed):
    """Six-order Williams design, cycled independently within each split.

    With incomplete multiples of six, position counts differ by at most one.
    Each task's six modes remain adjacent; no mode gets every task first.
    """
    randomizer = random.Random(seed)
    base = (0, 1, 5, 2, 4, 3)
    schedule = []
    positions = Counter()
    for repeat in range(repetitions):
        ordered = list(tasks)
        randomizer.shuffle(ordered)
        for task in ordered:
            shift = positions[task["split"]] % len(CONFIGS)
            positions[task["split"]] += 1
            for position, index in enumerate(base):
                config = CONFIGS[(index + shift) % len(CONFIGS)]
                schedule.append({**task, "repeat": repeat, "position": position,
                                 "config": config,
                                 "result": f"records/repeat_{repeat:03d}/{task['domain']}_{task['task_id']}/{config}.json"})
    return schedule


def prepare(args):
    output = args.output.resolve()
    path = output / "manifest.json"
    if path.exists():
        if read(output / "manifest_sha256.json")["sha256"] != checksum(path):
            raise ValueError("The frozen experiment manifest changed")
        manifest = read(path)
        if manifest.get("purpose") != "full_frozen_spork_controller_evaluation":
            raise ValueError("This output directory belongs to a different experiment")
        if not args.analyze_only:
            settings = {"scope": args.scope, "repetitions": args.repetitions,
                        "max_turns": args.max_turns, "seed": args.seed,
                        "order_seed": args.order_seed, "cpu_threads": args.cpu_threads,
                        "model_url": args.model_url}
            if any(manifest[key] != value for key, value in settings.items()):
                raise ValueError("Resume settings differ from the frozen manifest")
            for name, digest in manifest["code_sha256"].items():
                if checksum(ROOT / name) != digest:
                    raise ValueError(f"Code changed since preparation: {name}")
            for name, digest in manifest["asset_sha256"].items():
                if checksum(name) != digest:
                    raise ValueError(f"Frozen task/checkpoint asset changed: {name}")
        return manifest
    if args.analyze_only:
        raise ValueError("No experiment manifest to analyze")
    if min(args.repetitions, args.max_turns, args.cpu_threads) <= 0:
        raise ValueError("Repetitions, max_turns and cpu_threads must be positive")
    if output.exists() and any(p.name != "run.lock" for p in output.iterdir()):
        raise ValueError("Choose a fresh output directory, or resume a prepared experiment")
    source = args.source.resolve()
    report, evidence = read(source / "online_check_report.json"), read(source / "manifest.json")
    if not report.get("passed"):
        raise ValueError("The controller integration check did not pass")
    # Use exactly the serving implementation and checkpoints that passed online.
    for name, digest in evidence["code_sha256"].items():
        if checksum(ROOT / name) != digest:
            raise ValueError(f"Serving code differs from the passed integration check: {name}")
    data = Path(evidence["dataset"])
    training = read(Path(evidence["training"]) / "manifest.json")
    if (checksum(data / "training_rows.jsonl") != training["data_sha256"]
            or checksum(data / "task_split.json") != training["task_split_sha256"]):
        raise ValueError("Training data or task split changed")
    split = read(data / "task_split.json")
    if split["task_counts"] != {"train": 109, "validation": 23, "test": 23}:
        raise ValueError("Expected the existing 155-task split")
    tasks = [t for t in split["tasks"] if args.scope == "all" or t["split"] == "test"]
    if len({t["group_id"] for t in tasks}) != len(tasks):
        raise ValueError("Duplicate task groups")
    assets = {str(data / "task_split.json"): checksum(data / "task_split.json")}
    for task_source in split["sources"]:
        if checksum(task_source["path"]) != task_source["sha256"]:
            raise ValueError("tau2 task files changed")
        assets[task_source["path"]] = task_source["sha256"]
    for checkpoint in evidence["checkpoints"].values():
        directory = Path(checkpoint["path"])
        if checksum(directory / "controller.safetensors") != checkpoint["weight_sha256"]:
            raise ValueError("Selected controller weights changed")
        for asset in directory.rglob("*"):
            if asset.is_file():
                assets[str(asset)] = checksum(asset)
    code = {**evidence["code_sha256"],
            "scripts/run_controller_full_eval.py": checksum(Path(__file__)),
            "scripts/run_controller_full_eval.sh": checksum(ROOT / "scripts/run_controller_full_eval.sh"),
            "spork_core/benchmarks/tau2.py": checksum(ROOT / "spork_core/benchmarks/tau2.py")}
    manifest = {
        "purpose": "full_frozen_spork_controller_evaluation", "schema_version": 1,
        "source_integration": str(source), "created_at": datetime.now().isoformat(),
        "scope": args.scope, "repetitions": args.repetitions, "max_turns": args.max_turns,
        "seed": args.seed, "order_seed": args.order_seed, "cpu_threads": args.cpu_threads,
        "model_url": args.model_url, "model": evidence["model"],
        "configs": list(CONFIGS), "checkpoints": evidence["checkpoints"],
        "tasks": tasks, "task_counts": dict(Counter(t["split"] for t in tasks)),
        "schedule": make_schedule(tasks, args.repetitions, args.order_seed),
        "tool_floor_s": evidence["tool_floor_s"], "context_window": 40960,
        "completion_max_tokens": 8192, "prefix_tokens": 32, "wait_tokens": 64,
        "d2": {"max_retries": 5, "retry_token_step": 50, "confidence_threshold": .90,
               "probe_max_tokens": 160, "snap_to_sentence": True},
        "forced_conclusion": False, "code_sha256": code, "asset_sha256": assets,
        "primary_split": "test", "weights_updated": False,
        "measurement": "Live task wall time, including BERT and per-turn D3/tool cleanup; "
                       "model loading and warmup excluded. One execution per task/policy/repetition.",
        "quality_scope": "Existing fixed-user-message tau2 harness and ACTION reward. "
                         "No official interactive tau2 success claim. Turn-cap hits reported explicitly.",
        "timing_spork_scope": evidence["timing_spork_scope"],
        "failure_policy": "Retain and seal failed completed attempts; never silently select a faster rerun. "
                          "An interrupted, unsealed attempt is retained and restarted on resume.",
    }
    output.mkdir(parents=True, exist_ok=True)
    for name in code:
        target = output / "code" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    dump(output / "task_manifest.json", tasks)
    dump(path, manifest)
    dump(output / "manifest_sha256.json", {"sha256": checksum(path)})
    return manifest


def load_sealed(path):
    marker = path.with_suffix(".complete.json")
    if not marker.exists():
        return None
    if not path.exists() or read(marker)["sha256"] != checksum(path):
        raise ValueError(f"Completed result changed: {path}")
    return read(path)


def record_errors(turns, before, after):
    errors = []
    if not turns:
        errors.append("empty_task")
    for turn in turns:
        if turn.get("error") or not turn.get("completion_valid"):
            errors.append("invalid_completion")
        if any(k not in turn for k in SIGNATURE_FIELDS) or not turn.get("main_token_ids"):
            errors.append("missing_outcome_evidence")
        trace = turn.get("launch_timing", {})
        if (trace.get("status") not in DECISION_STATUSES
                or trace.get("error") or turn.get("spec_cleanup_error")):
            errors.append("controller_or_tool_error")
        d3 = turn.get("d3_status") or {}
        if d3.get("error") or d3.get("clear_status") not in ("ok", "cleared"):
            errors.append("d3_cleanup_error")
        for prediction in turn.get("probes", []):
            registered = prediction.get("d3_set")
            if (prediction.get("probe_error") or prediction.get("d3_set_error")
                    or (registered is not None
                        and (registered.get("status") != "ok" or registered.get("error")))):
                errors.append("prediction_or_d3_error")
    if any(k not in before or k not in after for k in COUNTERS):
        errors.append("missing_d3_counters")
    delta = {k: after.get(k, 0) - before.get(k, 0) for k in COUNTERS}
    if any(v < 0 for v in delta.values()):
        errors.append("d3_counter_reset")
    if any(delta[k] for k in ("accounting_errors", "prefix_mismatches", "dropped_injections")):
        errors.append("d3_integrity_error")
    if after.get("active_requests") != 0 or after.get("pending_verifications") != 0:
        errors.append("server_not_idle")
    return sorted(set(errors)), delta


def same_outcome(left, right):
    def signature(record):
        return [{k: turn.get(k) for k in SIGNATURE_FIELDS} for turn in record["turns"]]
    return signature(left) == signature(right) and left.get("action_reward") == right.get("action_reward")


def analyze(output, manifest):
    records = []
    for item in manifest["schedule"]:
        record = load_sealed(output / item["result"])
        if record is not None:
            if any(record[k] != item[k] for k in ("group_id", "split", "config", "repeat")):
                raise ValueError("Result identity disagrees with schedule")
            records.append(record)
    grouped = defaultdict(dict)
    for r in records:
        grouped[(r["group_id"], r["repeat"])][r["config"]] = r
    matched, mismatches, incomplete = set(), [], []
    for identity, modes in grouped.items():
        if set(modes) != set(CONFIGS) or any(r["errors"] for r in modes.values()):
            incomplete.append({"group_id": identity[0], "repeat": identity[1]})
            continue
        differences = [name for name, r in modes.items() if not same_outcome(r, modes["timing_off"])]
        if differences:
            mismatches.append({"group_id": identity[0], "repeat": identity[1], "configs": differences})
        else:
            matched.add(identity)
    report = {
        "experiment_complete": len(records) == len(manifest["schedule"]),
        "expected_runs": len(manifest["schedule"]), "finished_runs": len(records),
        "failed_runs": [{k: r[k] for k in ("group_id", "repeat", "config", "errors")}
                        for r in records if r["errors"]],
        "quality_mismatches": mismatches, "incomplete_or_failed_comparisons": incomplete,
        "primary_split": "test", "splits": {},
        "d3_counters": {k: sum(r.get("d3_counters", {}).get(k, 0) for r in records) for k in COUNTERS},
        "quality_scope": manifest["quality_scope"], "measurement": manifest["measurement"],
        "comparison_rule": "Paired speed uses the same successful, outcome-matched task/repetition set "
                           "for all six policies. All failures and mismatches remain reported. "
                           "A single repetition does not estimate per-task timing variability.",
    }
    for split in ("all", "train", "validation", "test"):
        subset = [r for r in records if split == "all" or r["split"] == split]
        if not subset:
            continue
        split_report = {}
        for config in CONFIGS:
            runs = [r for r in subset if r["config"] == config]
            good = [r for r in runs if not r["errors"]]
            pairs = [r for r in good if (r["group_id"], r["repeat"]) in matched]
            calls = [t["launch_timing"]["controller_inference_ms"] for r in good for t in r["turns"]
                     if t.get("launch_timing", {}).get("controller_calls")]
            rewards = [r["action_reward"]["reward"] for r in good if r.get("action_reward") is not None]
            row = {"attempted_runs": len(runs), "failed_runs": len(runs) - len(good),
                   "successful_run_mean_wall_s": statistics.mean(r["wall_s"] for r in good) if good else None,
                   "turn_cap_hits": sum(r["turn_limit_reached"] for r in good),
                   "action_reward_scored_runs": len(rewards),
                   "action_reward_mean": statistics.mean(rewards) if rewards else None,
                   "controller_calls": len(calls),
                   "controller_mean_ms": statistics.mean(calls) if calls else None,
                   "policies": dict(Counter(t["launch_timing"].get("policy") for r in good for t in r["turns"])),
                   "paired_runs": len(pairs), "paired_tasks": len({r["group_id"] for r in pairs}),
                   "paired_mean_wall_s": statistics.mean(r["wall_s"] for r in pairs) if pairs else None,
                   "comparisons": {}}
            for reference in ("timing_off", "timing_spork", "timing_wait"):
                base = [grouped[(r["group_id"], r["repeat"])][reference]["wall_s"] for r in pairs]
                target = [r["wall_s"] for r in pairs]
                row["comparisons"][reference] = {
                    "mean_saved_s": statistics.mean(a - b for a, b in zip(base, target)) if pairs else None,
                    "speedup_ratio": sum(base) / sum(target) if pairs and sum(target) > 0 else None,
                    "latency_reduction_fraction": 1 - sum(target) / sum(base) if pairs and sum(base) > 0 else None,
                }
            split_report[config] = row
        report["splits"][split] = split_report
    report["controller_coverage"] = {
        name: report["splits"].get("all", {}).get(name, {}).get("controller_calls", 0) > 0
        for name in ("timing_bert", "timing_masked")}
    report["integrity_passed"] = (report["experiment_complete"] and not report["failed_runs"]
                                  and not mismatches and all(report["controller_coverage"].values())
                                  and report["d3_counters"]["verified_injections"] > 0)
    dump(output / "summary.json", report)
    lines = ["# SPORK 控制器完整实验", "",
             f"已完成 {len(records)}/{len(manifest['schedule'])} 次任务运行；"
             f"失败 {len(report['failed_runs'])} 次；动作或轨迹不一致 {len(mismatches)} 个任务组。",
             "", "主要泛化结果使用 test 划分。速度比较仅使用六组均成功且轨迹一致的相同任务集合；"
             "失败、差异和达到轮数上限的次数均保留在 summary.json 中。",
             "", "每个任务每种策略运行一次；本次不估计单任务的重复运行波动。"
             "质量指标为当前固定用户消息流程的 ACTION reward，非官方交互式 τ² 成功率。"]
    def cell(value, precision=3):
        return "—" if value is None else f"{value:.{precision}f}"
    for split in ("test", "validation", "train", "all"):
        if split not in report["splits"]:
            continue
        lines += ["", f"## {split}", "",
                  f"| 策略 | 完成/失败 | 配对数 | 配对平均耗时(s) | 比 SPORK 节省(s) | ACTION均值 | 达到{manifest['max_turns']}轮上限 |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for name, row in report["splits"][split].items():
            lines.append(f"| {name} | {row['attempted_runs']}/{row['failed_runs']} | {row['paired_runs']} | "
                         f"{cell(row['paired_mean_wall_s'])} | "
                         f"{cell(row['comparisons']['timing_spork']['mean_saved_s'])} | "
                         f"{cell(row['action_reward_mean'])} | {row['turn_cap_hits']} |")
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    return report


async def execute(output, manifest):
    import aiohttp
    from transformers import AutoTokenizer
    from spork_core.benchmarks.tau2 import get_environment, load_system_prompt, load_tools
    from spork_core.executors import Tau2ToolExecutor
    from spork_core.launch_timing import LaunchController, visible_input
    from spork_core.qwen import QwenToolCallAdapter
    from spork_core.turn_runner import TurnConfig
    from spork_core.vllm_client import VllmClient
    from spork_unified_eval import load_tau2_tasks, run_task, tau2_action_reward

    task_map = {f'{t["domain"]}:{t["task_id"]}': t for t in load_tau2_tasks(manifest["seed"])}
    if any(t["group_id"] not in task_map for t in manifest["tasks"]):
        raise ValueError("A frozen task is missing from the local tau2 loader")
    tokenizer = AutoTokenizer.from_pretrained(manifest["model"], local_files_only=True)
    controllers = {}
    for name, variant in (("timing_bert", "full"), ("timing_masked", "masked_reasoning")):
        controllers[name] = LaunchController(manifest["checkpoints"][variant]["path"], manifest["cpu_threads"])
    d2 = manifest["d2"]
    args = SimpleNamespace(
        d2_max_retries=d2["max_retries"], d2_confidence_threshold=d2["confidence_threshold"],
        d2_retry_token_step=d2["retry_token_step"], d2_snap_to_sentence=d2["snap_to_sentence"],
        d2_probe_max_tokens=d2["probe_max_tokens"], d2_probe_timeout_s=0.,
        d2_theta_decay_per_retry=0., d2_min_tokens_first_probe=0,
        d2_continue_after_dispatch=False, no_forced_conclusion=True, _launch_controllers=controllers,
    )
    cfg = TurnConfig(enable_thinking=True, baseline_max_tokens=manifest["completion_max_tokens"],
                     context_window_tokens=manifest["context_window"], seed=manifest["seed"])
    adapters = {d: QwenToolCallAdapter(tokenizer, load_tools(d)) for d in ("airline", "retail")}
    policies = {d: load_system_prompt(d) for d in adapters}
    client = VllmClient(base_url=manifest["model_url"], model=manifest["model"], timeout_s=600)
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    session_dir = output / "sessions" / session_id
    get_status(manifest)  # Require the same idle batch-invariant D3 server.
    async with aiohttp.ClientSession(trust_env=False) as session:
        # Warmup is part of starting this formal run, never part of timed tasks.
        warm = visible_input([{"role": "user", "content": "Warmup"}], [], "<think>\n", 32, 64)
        for controller in controllers.values():
            await asyncio.to_thread(controller.predict, warm)
        for _ in range(2):
            await client.complete(session, "Say OK.", max_tokens=32, logprobs=None, seed=manifest["seed"])
        dump(session_dir / "status_start.json", get_status(manifest))
        started = time.perf_counter()
        new_runs = 0
        for index, item in enumerate(manifest["schedule"]):
            path = output / item["result"]
            if load_sealed(path) is not None:
                continue
            task = task_map[item["group_id"]]
            if path.exists():
                # Preserve an interrupted write that never received a completion seal.
                backup = path.parent / "attempts" / (path.stem + "_" + uuid.uuid4().hex + "_unsealed.json")
                backup.parent.mkdir(parents=True, exist_ok=True)
                path.replace(backup)
            previous = sorted((path.parent / "attempts").glob(path.stem + "_*.json"))
            attempt = path.parent / "attempts" / (path.stem + "_" + uuid.uuid4().hex + ".json")
            dump(attempt, {"session": session_id, "started_at": datetime.now().isoformat(),
                           "schedule_item": item, "status": "started"})
            before = get_status(manifest)
            executor = Tau2ToolExecutor(get_environment(item["domain"]), floor_s=manifest["tool_floor_s"])
            messages = [{"role": "system", "content": policies[item["domain"]]},
                        {"role": "user", "content": task["user_message"]}]
            log(f"[{index + 1}/{len(manifest['schedule'])}] {item['split']} {item['group_id']} "
                f"{item['config']} repeat={item['repeat'] + 1}")
            start = time.perf_counter()
            turns, error = [], None
            try:
                turns = await run_task(args, session, client, adapters[item["domain"]], executor,
                                       messages, item["config"], cfg, max_turns=manifest["max_turns"])
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            wall = time.perf_counter() - start
            try:
                after = get_status(manifest, require_idle=False)
            except Exception as exc:
                after = {"status_error": f"{type(exc).__name__}: {exc}"}
            errors, delta = record_errors(turns, before, after)
            if error:
                errors.append(error)
            record = {**item, "session": session_id, "wall_s": wall, "turns": turns,
                      "errors": errors, "d3_counters": delta, "status_after": after,
                      "turn_limit_reached": bool(len(turns) == manifest["max_turns"]
                                                  and turns[-1].get("baseline_has_tool_call")),
                      "action_reward": tau2_action_reward(task.get("raw_task") or {}, turns) if not error else None,
                      "previous_unsealed_attempts": [str(p.relative_to(output)) for p in previous],
                      "completed_at": datetime.now().isoformat()}
            dump(path, record)
            dump(path.with_suffix(".complete.json"), {"sha256": checksum(path)})
            dump(attempt, {"session": session_id, "schedule_item": item,
                           "status": "finished", "wall_s": wall, "errors": errors})
            new_runs += 1
            mean_wall = (time.perf_counter() - started) / new_runs
            log(f"  {wall:.1f}s, {len(turns)} turns, errors={len(errors)}, "
                f"ETA≈{mean_wall * (len(manifest['schedule']) - index - 1) / 3600:.1f}h")
            if (index + 1) % (10 * len(CONFIGS)) == 0 or errors:
                analyze(output, manifest)
            if (after.get("active_requests") != 0 or after.get("pending_verifications") != 0
                    or any(k in errors for k in ("d3_counter_reset", "d3_integrity_error", "d3_cleanup_error"))):
                raise RuntimeError("Server/D3 state is not clean. Result saved; inspect before resuming.")
        dump(session_dir / "status_end.json", get_status(manifest))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--scope", choices=("all", "test"), default="all")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--order-seed", type=int, default=20261017)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--model-url", default="http://127.0.0.1:8016/v1")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    with lock(output / "run.lock"):
        manifest = prepare(args)
        log(f"完整实验目录：{output}")
        log(f"{len(manifest['tasks'])} tasks × {len(CONFIGS)} modes × {manifest['repetitions']} "
            f"repeat(s) = {len(manifest['schedule'])} runs; max_turns={manifest['max_turns']}")
        if args.prepare_only:
            log("配置已准备，尚未执行模型推理。")
            return
        pending = any(load_sealed(output / item["result"]) is None for item in manifest["schedule"])
        if not args.analyze_only and pending:
            port = urlparse(manifest["model_url"]).port or 80
            with lock(ROOT / "results" / f"controller_online_{port}.lock"):
                asyncio.run(execute(output, manifest))
        report = analyze(output, manifest)
        log(f"完成 {report['finished_runs']}/{report['expected_runs']}，"
            f"失败 {len(report['failed_runs'])}，结果不一致 {len(report['quality_mismatches'])}")
        log(f"按划分汇总：{output / 'summary.json'}")
        if not report["integrity_passed"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
