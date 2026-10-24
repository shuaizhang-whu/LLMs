#!/usr/bin/env python3
"""Bounded, paired SPORK D3 smoke test; never starts or restarts a server.

Default: prepare and check the existing dedicated endpoint, with no inference.
--execute runs two train tasks, at most two turns, against the same first-token
SPORK policy. Full experiments and official interactive tau2 remain separate.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import sys
import time
from types import SimpleNamespace
from urllib.parse import urlparse
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spork_core.paths import QWEN3_32B_MODEL, resolve_artifact_path
from spork_core.draft_reuse import REUSE_MODES
from scripts.check_controller_online import get_status
from scripts.run_controller_full_eval import dump, lock, record_errors, same_outcome


def prepare(args, output):
    split = resolve_artifact_path(args.dataset) / "task_split.json"
    tasks = json.loads(split.read_text())["tasks"]
    chosen = [min((t for t in tasks if t["split"] == "train" and t["domain"] == domain),
                  key=lambda t: int(t["task_id"])) for domain in ("airline", "retail")]
    modes = args.modes.split(",")
    if len(set(modes)) != len(modes) or any(m not in REUSE_MODES for m in modes) or "single" not in modes:
        raise ValueError("Modes must be unique and include the single-injection baseline")
    schedule = []
    for repeat in range(args.repeats):
        for index, task in enumerate(chosen):
            shift = (index + repeat) % len(modes)
            for mode in modes[shift:] + modes[:shift]:
                schedule.append({**task, "repeat": repeat, "reuse_mode": mode})
    output.mkdir(parents=True, exist_ok=False)
    sources = list((ROOT / "spork_core").glob("*.py")) + [
        ROOT / "spork_core/benchmarks/tau2.py", ROOT / "spork_unified_eval.py",
        ROOT / "launch_vllm_spork_http.py", Path(__file__),
        ROOT / "scripts/check_controller_online.py", ROOT / "scripts/run_controller_full_eval.py"]
    hashes = {}
    for path in sources:
        relative = path.relative_to(ROOT)
        hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
        dest = output / "code" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    manifest = dict(purpose="probe_reuse_online_smoke", tasks=chosen, modes=modes, schedule=schedule,
                    model=args.model_name, model_url=args.model_url, tokenizer=str(args.tokenizer.resolve()),
                    max_turns=args.max_turns, seed=42, tool_floor_s=2., code_sha256=hashes,
                    task_split_sha256=hashlib.sha256(split.read_bytes()).hexdigest(),
                    quality_scope="Fixed-user-message harness; short capped traces, not official tau2 success.",
                    measurement="Live task time including tool wait and cleanup, excluding model load and warmup.",
                    first_launch="SPORK first token for every mode; no BERT")
    dump(output / "manifest.json", manifest)
    return manifest


def preflight(manifest):
    errors = []
    tokenizer = Path(manifest["tokenizer"])
    if not (tokenizer / "tokenizer.json").is_file():
        errors.append(f"Tokenizer directory missing: {tokenizer}")
    try:
        status = get_status(manifest)
        supported = status.get("server_config", {}).get("d3_reuse_modes", ["single"])
        if not set(manifest["modes"]).issubset(supported):
            errors.append("Server does not expose the requested D3 reuse modes")
    except Exception as error:
        status = None
        errors.append(f"Endpoint preflight failed: {type(error).__name__}: {error}")
    return dict(passed=not errors, errors=errors, status=status, inference_started=False)


def summarize(records, manifest):
    errors, pairs = [], []
    for task in manifest["tasks"]:
        repeats = sorted({r["repeat"] for r in records if r["group_id"] == task["group_id"]})
        for repeat in repeats:
            group = {r["reuse_mode"]: r for r in records
                     if r["group_id"] == task["group_id"] and r["repeat"] == repeat}
            if set(group) != set(manifest["modes"]) or any(r["errors"] for r in group.values()):
                errors.append(f"Incomplete/failed pair: {task['group_id']}/{repeat}")
                continue
            if any(not same_outcome(r, group["single"]) for r in group.values()):
                errors.append(f"Outcome mismatch: {task['group_id']}/{repeat}")
                continue
            pairs.append(group)
    if len(records) != len(manifest["schedule"]):
        errors.append("Schedule incomplete")
    timing = {}
    for mode in manifest["modes"]:
        timing[mode] = dict(paired_runs=len(pairs),
                           mean_wall_s=statistics.mean(p[mode]["wall_s"] for p in pairs) if pairs else None,
                           mean_saved_vs_single_s=statistics.mean(p["single"]["wall_s"] - p[mode]["wall_s"]
                                                                  for p in pairs) if pairs else None)
    return dict(integrity_passed=not errors and bool(pairs), errors=errors, timing=timing,
                speedup_established=False,
                note="A short smoke test cannot establish stable speedup or generalization; all failures retained.")


async def execute(manifest, output):
    import aiohttp
    from transformers import AutoTokenizer
    from spork_core.benchmarks.tau2 import get_environment, load_system_prompt, load_tools
    from spork_core.executors import Tau2ToolExecutor
    from spork_core.qwen import QwenToolCallAdapter
    from spork_core.turn_runner import TurnConfig
    from spork_core.vllm_client import VllmClient
    from spork_unified_eval import load_tau2_tasks, run_task, tau2_action_reward

    tokenizer = AutoTokenizer.from_pretrained(manifest["tokenizer"], local_files_only=True)
    task_map = {f'{t["domain"]}:{t["task_id"]}': t for t in load_tau2_tasks(manifest["seed"])}
    adapters = {d: QwenToolCallAdapter(tokenizer, load_tools(d)) for d in ("airline", "retail")}
    policies = {d: load_system_prompt(d) for d in adapters}
    client = VllmClient(base_url=manifest["model_url"], model=manifest["model"], timeout_s=600)
    cfg = TurnConfig(enable_thinking=True, baseline_max_tokens=8192, context_window_tokens=40960, seed=42)
    args = SimpleNamespace(d2_max_retries=5, d2_confidence_threshold=.90, d2_retry_token_step=50,
                           d2_snap_to_sentence=True, d2_probe_max_tokens=160, d2_probe_timeout_s=0.,
                           d2_theta_decay_per_retry=0., d2_min_tokens_first_probe=0,
                           d2_continue_after_dispatch=False, no_forced_conclusion=True, _launch_controllers={})
    records = []
    async with aiohttp.ClientSession(trust_env=False) as session:
        for _ in range(2):
            await client.complete(session, "Say OK.", max_tokens=32, logprobs=None, seed=42)
        for index, item in enumerate(manifest["schedule"]):
            args.d3_reuse_mode = item["reuse_mode"]
            task = task_map[item["group_id"]]
            messages = [{"role": "system", "content": policies[item["domain"]]},
                        {"role": "user", "content": task["user_message"]}]
            executor = Tau2ToolExecutor(get_environment(item["domain"]), floor_s=manifest["tool_floor_s"])
            before = get_status(manifest)
            path = output / "records" / f"{index:03d}.json"
            dump(path, {**item, "status": "started"})
            print(f"[{index+1}/{len(manifest['schedule'])}] {item['group_id']} {args.d3_reuse_mode}", flush=True)
            start = time.perf_counter()
            turns, failure = [], None
            try:
                turns = await run_task(args, session, client, adapters[item["domain"]], executor,
                                       messages, "timing_spork", cfg, max_turns=manifest["max_turns"])
            except Exception as error:
                failure = f"{type(error).__name__}: {error}"
            wall = time.perf_counter() - start
            try:
                after = get_status(manifest, require_idle=False)
            except Exception as error:
                after = {"status_error": str(error)}
            errors, delta = record_errors(turns, before, after)
            if failure:
                errors.append(failure)
            if any(t.get("d3_reuse_mode") != args.d3_reuse_mode for t in turns):
                errors.append("Wrong reuse mode in turn record")
            record = {**item, "wall_s": wall, "turns": turns, "errors": errors,
                      "action_reward": tau2_action_reward(task.get("raw_task") or {}, turns) if not failure else None,
                      "d3_counters": delta, "status_after": after, "status": "finished"}
            dump(path, record)
            dump(path.with_suffix(".complete.json"), {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            records.append(record)
            dump(output / "summary.json", summarize(records, manifest))
            if errors:
                raise RuntimeError(f"Smoke failed; result retained: {errors}")
    report = summarize(records, manifest)
    dump(output / "summary.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["integrity_passed"]:
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-url", default="http://127.0.0.1:8016/v1")
    parser.add_argument("--model-name", default=str(QWEN3_32B_MODEL))
    parser.add_argument("--tokenizer", type=Path, default=QWEN3_32B_MODEL)
    parser.add_argument("--dataset", default="prefix_data/datasets/controller_stage3_20261008_213013_wi87e7du")
    parser.add_argument("--modes", default="single,prefix,suffix")
    parser.add_argument("--repeats", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--max-turns", type=int, choices=[1, 2], default=2)
    parser.add_argument("--output", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve() if args.output else ROOT / "results" / (
        "probe_reuse_online_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
    manifest = prepare(args, output)
    print(f"Smoke output: {output}", flush=True)
    if args.prepare_only:
        return
    port = urlparse(manifest["model_url"]).port or 80
    with lock(ROOT / "results" / f"controller_online_{port}.lock"):
        report = preflight(manifest)
        dump(output / "preflight.json", report)
        if not report["passed"]:
            print(json.dumps(report, ensure_ascii=False, indent=2))
            raise SystemExit(2)
        if args.execute:
            asyncio.run(execute(manifest, output))
        else:
            print("Preflight passed. No inference requested; use --execute for the bounded smoke test.")


if __name__ == "__main__":
    main()
