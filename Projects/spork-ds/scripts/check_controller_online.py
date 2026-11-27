#!/usr/bin/env python3
"""Small real SPORK integration check; user launches this in tmux.

Two preassigned validation tasks, at most two turns, six first-launch modes.
This is a functionality/overhead check, not a final task-success experiment.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spork_core.prefix_replay import COUNTERS  # noqa: E402

DEFAULT_TRAINING = ROOT / "controller_runs/training_compare_e200_20261016_165218_ifqRYU"
CONFIGS = ("timing_off", "timing_spork", "timing_now", "timing_wait", "timing_bert", "timing_masked")
REFERENCE = "timing_off"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(training, output, model_url):
    training = Path(training).resolve()
    m = read(training / "manifest.json")
    data = Path(m["args"]["dataset"])
    split = read(data / "task_split.json")
    if checksum(data / "task_split.json") != m["task_split_sha256"]:
        raise ValueError("The frozen task assignment changed")
    if checksum(data / "training_rows.jsonl") != m["data_sha256"]:
        raise ValueError("The frozen training data changed")
    epochs = m["epochs_per_fit"]
    if type(epochs) is not int or epochs <= 0:
        raise ValueError("Training manifest has an invalid epoch count")
    # Select by task ID before looking at predictions, gains, or online results.
    tasks = [min((t for t in split["tasks"] if t["split"] == "validation" and t["domain"] == domain),
                 key=lambda t: int(t["task_id"])) for domain in ("airline", "retail")]
    checkpoints = {}
    for variant, ablation in (("full", "none"), ("masked_reasoning", "mask_reasoning")):
        run = training / f"{variant}_seed_42"
        report = read(run / "training_report.json")
        if report["epochs_completed"] != epochs:
            raise ValueError(f"Expected a completed {epochs}-epoch fit")
        checkpoint = (run / report["best_checkpoint"]).resolve()
        if checkpoint.parent != run:
            raise ValueError("Selected checkpoint must belong to its training run")
        if read(checkpoint / "controller_config.json")["config"]["input_ablation"] != ablation:
            raise ValueError("Checkpoint input variant mismatch")
        checkpoints[variant] = {"path": str(checkpoint), "weight_sha256": checksum(checkpoint / "controller.safetensors"),
                                "seed": 42, "epoch": report["selected_validation_metrics"]["epoch"]}
    dataset = read(data / "manifest.json")
    if dataset["prefix_tokens"] != 32 or dataset["replay_config"]["delay_tokens"] != 64:
        raise ValueError("This integration expects the collected 32-token decision and 64-token wait")
    manifest = {"purpose": "small_live_spork_controller_integration_check", "training": str(training),
                "dataset": str(data), "model": dataset["model"], "model_url": model_url,
                "tool_floor_s": dataset["tool_floor_s"], "configs": list(CONFIGS), "tasks": tasks,
                "max_turns": 2, "prefix_tokens": 32, "wait_tokens": 64, "checkpoints": checkpoints,
                "quality_reference": REFERENCE,
                "timing_spork_scope": "Native first-chunk launch strategy on the common exact-token transport; "
                                      "not a byte-for-byte replay of the legacy HTTP runner",
                "training_epochs": epochs,
                "test_tasks_used": False, "final_online_performance_experiment": False,
                "checkpoint_rule": "seed 42 fixed in advance; epoch chosen by the existing validation-gain rule",
                "code_sha256": {str(p.relative_to(ROOT)): checksum(p) for p in (
                    ROOT / "spork_core/launch_timing.py", ROOT / "spork_core/turn_runner.py",
                    ROOT / "spork_core/timed_retry.py", ROOT / "spork_core/prefix_replay.py",
                    ROOT / "spork_core/timing_controller.py", ROOT / "spork_core/controller_data.py",
                    ROOT / "spork_core/executors.py", ROOT / "spork_core/qwen.py",
                    ROOT / "spork_core/vllm029_d3.py", ROOT / "launch_vllm_spork_http.py",
                    ROOT / "spork_unified_eval.py", Path(__file__))}}
    output.mkdir(parents=True, exist_ok=False)
    write(output / "task_manifest.json", tasks)
    write(output / "manifest.json", manifest)
    return manifest


def get_status(manifest, require_idle=True):
    root = manifest["model_url"].rstrip("/").removesuffix("/v1")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(root + "/v1/models", timeout=15) as response:
        models = json.load(response)
    if manifest["model"] not in [m["id"] for m in models["data"]]:
        raise ValueError("The server serves a different Actor")
    with opener.open(root + "/spork/status", timeout=30) as response:
        status = json.load(response)
    if (not status.get("has_drafter")
            or status.get("server_config", {}).get("batch_invariant_requested") is not True
            or (require_idle and (status.get("active_requests") != 0
                                  or status.get("pending_verifications") != 0))):
        raise ValueError("An idle batch-invariant D3 server is required")
    return status


def analyze(output, manifest, before, after):
    summaries = list((output / "eval").glob("**/summary.json"))
    if len(summaries) != 1:
        raise ValueError("Expected one completed evaluator summary")
    parent = summaries[0].parent
    records = {}
    errors = []
    for config in CONFIGS:
        records[config] = {}
        for path in sorted((parent / config).glob("*.json")):
            record = read(path)
            group = f'{record["domain"]}:{record["task_id"]}'
            if group in records[config]:
                errors.append(f"{config}/{group}: duplicate task results")
            records[config][group] = record
    groups = [t["group_id"] for t in manifest["tasks"]]
    checks, costs = {}, {}
    signature_fields = ("baseline_has_tool_call", "baseline_tool_name", "baseline_tool_args",
                        "tool_result_sha256", "environment_after", "main_token_ids")

    def signature(turn):
        return {key: turn.get(key) for key in signature_fields}

    for config, data in records.items():
        if set(data) != set(groups):
            errors.append(f"{config}: missing/extra task results")
        entries = [t for r in data.values() for t in r.get("turns", [])]
        traces = [t["launch_timing"] for t in entries if t.get("launch_timing")]
        for task, result in data.items():
            turns = result.get("turns", [])
            if result.get("error") or not turns:
                errors.append(f"{config}/{task}: empty/failed task")
            for t in turns:
                if t.get("error") or not t.get("completion_valid", False):
                    errors.append(f"{config}/{task}: invalid completion")
                if any(key not in t for key in signature_fields) or not t.get("main_token_ids"):
                    errors.append(f"{config}/{task}: missing outcome evidence")
                trace = t.get("launch_timing", {})
                if not trace or trace.get("status") not in {
                    "ready", "disabled", "reasoning_ended_before_decision",
                    "main_finished_or_formal_action_before_decision",
                    "main_finished_or_formal_action_during_decision",
                    "main_finished_or_formal_action_while_waiting",
                }:
                    errors.append(f"{config}/{task}: missing/failed timing decision")
                if trace.get("error") or t.get("spec_cleanup_error"):
                    errors.append(f"{config}/{task}: controller/tool cleanup error")
                d3 = t.get("d3_status") or {}
                if d3.get("error") or d3.get("clear_status") not in ("ok", "cleared"):
                    errors.append(f"{config}/{task}: D3 cleanup not confirmed")
                for prediction in t.get("probes", []):
                    registration = prediction.get("d3_set")
                    if (prediction.get("probe_error") or prediction.get("d3_set_error")
                            or (registration is not None
                                and (registration.get("status") != "ok" or registration.get("error")))):
                        errors.append(f"{config}/{task}: prediction/D3 error")
            ref = records[REFERENCE].get(task)
            if ref is None:
                continue
            same = [signature(t) for t in turns] == [signature(t) for t in ref.get("turns", [])]
            checks[f"{config}/{task}"] = same
            if not same:
                errors.append(f"{config}/{task}: trajectory/outcome differs from speculation off")
        measurements = [t["controller_inference_ms"] for t in traces if t.get("controller_calls")]
        costs[config] = {"controller_calls": len(measurements),
                         "controller_inference_mean_ms": statistics.mean(measurements) if measurements else None,
                         "task_wall_s": {k: r.get("wall_s") for k, r in data.items()},
                         "turn_wall_ms": [t.get("turn_end_ms") for t in entries],
                         "tool_result_ready_ms": [t.get("tool_result_ready_ms") for t in entries],
                         "cleanup_ms": [t.get("cleanup_ms") for t in entries],
                         "policies": dict(Counter(t.get("policy") for t in traces)),
                         "statuses": dict(Counter(t.get("status") for t in traces))}
    for name in ("timing_bert", "timing_masked"):
        if costs[name]["controller_calls"] == 0:
            errors.append(f"{name}: controller was not exercised")
    if any(k not in before or k not in after for k in COUNTERS):
        errors.append("Server did not return complete D3 counters")
    delta = {k: after.get(k, 0) - before.get(k, 0) for k in COUNTERS}
    if any(v < 0 for v in delta.values()):
        errors.append("D3 counters reset during the check")
    if not delta["verified_injections"] or any(delta[k] for k in ("accounting_errors", "prefix_mismatches", "dropped_injections")):
        errors.append("D3 integrity/coverage check failed")
    if after.get("active_requests") != 0 or after.get("pending_verifications") != 0:
        errors.append("Server did not finish cleanly")
    report = {"passed": not errors, "errors": errors, "outcome_checks": checks, "controller_costs": costs,
              "quality_reference": REFERENCE,
              "d3_counter_delta": delta, "eval_summary": read(summaries[0]),
              "scope": "Two validation tasks, max two turns; correctness and serving overhead only. "
                       "No final held-out speedup or official interactive tau2 success claim."}
    write(output / "online_check_report.json", report)
    print(json.dumps({"passed": report["passed"], "errors": errors, "controller_costs": costs}, ensure_ascii=False, indent=2))
    return report


def run(args):
    output = args.output.resolve() if args.output else ROOT / "results" / (
        "controller_online_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
    manifest = prepare(args.training, output, args.model_url)
    print(f"在线检查目录：{output}", flush=True)
    if args.prepare_only:
        print("仅准备配置，尚未调用 Actor 或运行在线检查。", flush=True)
        return
    before = get_status(manifest)
    write(output / "status_before.json", before)
    cmd = [sys.executable, str(ROOT / "spork_unified_eval.py"),
           "--model-url", manifest["model_url"], "--model-name", manifest["model"],
           "--tokenizer-path", manifest["model"], "--benchmark", "tau2", "--n", "2", "--seed", "42",
           "--configs", ",".join(CONFIGS), "--enable-thinking", "--max-tokens", "40960",
           "--completion-max-tokens", "8192", "--max-turns", "2", "--no-forced-conclusion",
           "--task-manifest", str(output / "task_manifest.json"), "--tau2-floor", str(manifest["tool_floor_s"]),
           "--warmup-requests", "2", "--warmup-max-tokens", "32", "--d2-max-retries", "5",
           "--d2-retry-token-step", "50", "--d2-confidence-threshold", "0.90", "--d2-probe-max-tokens", "160",
           "--controller-checkpoint", manifest["checkpoints"]["full"]["path"],
           "--controller-masked-checkpoint", manifest["checkpoints"]["masked_reasoning"]["path"],
           "--controller-cpu-threads", "4", "--results-root", str(output / "eval")]
    write(output / "command.json", cmd)
    completed = subprocess.run(cmd, cwd=ROOT, check=False)
    if completed.returncode:
        write(output / "online_check_report.json", {
            "passed": False, "errors": [f"Evaluator exited with code {completed.returncode}"],
            "scope": "Incomplete integration check; inspect the launch log.",
        })
        raise SystemExit(completed.returncode)
    after = get_status(manifest, require_idle=False)
    write(output / "status_after.json", after)
    report = analyze(output, manifest, before, after)
    print(f"检查报告：{output / 'online_check_report.json'}", flush=True)
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-url", default="http://127.0.0.1:8016/v1")
    parser.add_argument("--prepare-only", action="store_true")
    run(parser.parse_args())
