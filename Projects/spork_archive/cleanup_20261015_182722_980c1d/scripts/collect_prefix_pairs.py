#!/usr/bin/env python3
"""Collect independent off/now/wait continuations of ten saved tau2 prefixes.

Run against a dedicated, user-started SPORK vLLM server. The collector never
starts a model server. It writes inputs (past only), outcomes and labels to
separate files and saves complete local environment snapshots. No training.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time

import aiohttp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spork_core.benchmarks.tau2 import (  # noqa: E402
    build_user_message, get_environment, load_system_prompt, load_tasks, load_tools,
)
from spork_core.executors import Tau2ToolExecutor  # noqa: E402
from spork_core.prefix_replay import (  # noqa: E402
    COUNTERS, PrefixHttpClient, ReplayConfig, paired_labels, run_prefix_branch,
)
from spork_core.qwen import QwenToolCallAdapter  # noqa: E402
from spork_core.turn_runner import _append_tool_messages  # noqa: E402


def dump(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def log(message):
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def environment_hash(env):
    return {"agent_db": env.get_db_hash(), "user_db": env.get_user_db_hash()}


def candidate_tasks(seed, limit):
    # Interleave domains; order is independent of later timing/hit outcomes.
    rng = random.Random(seed)
    pools = {d: load_tasks(d) for d in ("airline", "retail")}
    for pool in pools.values():
        rng.shuffle(pool)
    tasks = []
    for i in range(max(map(len, pools.values()))):
        for domain, pool in pools.items():
            if i < len(pool):
                tasks.append((domain, pool[i]))
    return tasks[:limit]


def make_summary(rows, requested, failures, skipped):
    totals = {k: 0 for k in COUNTERS}
    branch_errors = []
    cleanup_errors = []
    for row in rows:
        for policy, result in row["outcomes"].items():
            for k in totals:
                totals[k] += result.get("d3_counters", {}).get(k, 0)
            if result.get("error"):
                branch_errors.append({"sample_id": row["sample_id"], "policy": policy,
                                      "error": result["error"]})
            if result.get("cleanup_tool_error"):
                cleanup_errors.append({"sample_id": row["sample_id"], "policy": policy,
                                       "error": result["cleanup_tool_error"]})
            clear = result.get("clear_after", {})
            if clear.get("active_requests", 0) or clear.get("pending_verifications", 0):
                cleanup_errors.append({"sample_id": row["sample_id"], "policy": policy,
                                       "error": "D3 state remained after clear"})
    valid = [row for row in rows if row["labels"].get("valid")]
    histogram = {p: sum(r["labels"]["best_observed_policy"] == p for r in valid)
                 for p in ("off", "now", "wait")}
    return {
        "requested_prefixes": requested, "collected_prefixes": len(rows),
        "branch_runs": sum(len(row["outcomes"]) for row in rows),
        "valid_label_pairs": len(valid), "skipped_prefix_candidates": skipped,
        "failures": failures, "branch_errors": branch_errors, "cleanup_errors": cleanup_errors,
        "d3_counter_totals": totals,
        "best_observed_policy_counts": histogram,
        "action_disagreements": sum(not r["labels"].get(p + "_same_action_as_off", False)
                                    for r in valid for p in ("now", "wait")),
        "pipeline_passed": (len(rows) == requested and len(valid) == requested
                            and not failures and not branch_errors and not cleanup_errors
                            and not totals["accounting_errors"] and not totals["dropped_injections"]),
        "d3_verification_exercised": totals["verified_injections"] > 0,
        "note": "One repetition per prefix. Replay feasibility data, not a trained policy or speedup claim. "
                "All branches use the same ngram-enabled server. Tool latency floor is artificial. "
                "Action/result/state equality are diagnostics, not official tau2 task success.",
    }


async def collect(args, tokenizer, tasks):
    run_dir = args.output.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "manifest.json").exists():
        raise ValueError("Output already contains a run; choose a new directory")
    client = PrefixHttpClient(args.model_url, args.model, timeout_s=args.timeout_s)
    cfg = ReplayConfig(delay_tokens=args.delay_tokens, max_tokens=args.max_tokens,
                       context_window=args.context_window, seed=args.seed)
    packages = {n: version(n) for n in ("vllm", "torch", "transformers", "aiohttp", "tau2")}
    code_files = [Path(__file__), ROOT / "spork_core/prefix_replay.py",
                  ROOT / "spork_core/qwen.py", ROOT / "spork_core/executors.py",
                  ROOT / "spork_core/vllm029_d3.py", ROOT / "launch_vllm_spork_http.py",
                  ROOT / "scripts/serve_qwen32_prefix_collection.sh",
                  ROOT / "scripts/collect_prefixes_smoke.sh"]
    manifest = {
        "schema_version": 1, "purpose": "first_launch_timing_prefix_replay_smoke",
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "replay_config": asdict(cfg), "packages": packages,
        "candidate_tasks": [{"domain": d, "task_id": t["id"] if "id" in t else t["task_id"]}
                            for d, t in tasks],
        "code_sha256": {str(p.relative_to(ROOT)): sha256(p.read_bytes()) for p in code_files},
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "sample_timing": "Warmed replay request submission to formal result availability. "
                         "Prefix generation, discarded warmup, state restoration and final draining are separate.",
        "token_counting": "server token_ids, never SSE message count",
        "http_connection_policy": "fresh_connection_per_request_no_automatic_retries",
        "history_policy": "Only the off branch advances the reference history for the next turn.",
        "policy_scope": "One first-launch decision; same D2 confidence/retry parameters and D3 after launch. "
                        "Strict first wait in real tokens; cancel pending launch on formal call or completion. "
                        "Retries use real tokens and sentence/+30 boundary; no timeout-triggered early launch.",
        "training": False,
    }
    dump(run_dir / "manifest.json", manifest)
    # Save the exact uncommitted working version as well as hashes.
    code_dir = run_dir / "code"
    for p in code_files:
        target = code_dir / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
    rows, failures, skipped = [], [], []
    branch_order_rng = random.Random(args.seed + 1009)
    # tau2 imports and snapshot construction can block the event loop longer
    # than uvicorn's keep-alive timeout. A fresh local connection avoids reusing
    # a socket the server closed in the meantime, without silently retrying a
    # timed request and contaminating its latency label.
    connector = aiohttp.TCPConnector(force_close=True)
    async with aiohttp.ClientSession(trust_env=False, connector=connector) as session:
        async with session.get(client.base_url + "/models", timeout=15) as r:
            r.raise_for_status()
            models = await r.json()
        if args.model not in [m["id"] for m in models.get("data", [])]:
            raise RuntimeError(f"Wrong model served: {models}")
        status = await client.status(session)
        if not status.get("has_drafter"):
            raise RuntimeError("D3/ngram server required; use serve_qwen32_prefix_collection.sh")
        if status.get("active_requests", 0) or status.get("pending_verifications", 0):
            raise RuntimeError("D3 has active state; use a dedicated idle server")
        dump(run_dir / "preflight.json", {"models": models, "status": status})
        # Discarded warmup, outside all samples and reference histories.
        warm_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Briefly explain how to look up an order."}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True)
        warm = tokenizer.encode(warm_text, add_special_tokens=False)
        await client.complete(session, warm, max_tokens=128, seed=args.seed)
        for domain, task in tasks:
            if len(rows) >= args.prefixes:
                break
            task_id = str(task.get("id", task.get("task_id")))
            tools = load_tools(domain)
            adapter = QwenToolCallAdapter(tokenizer, tools)
            messages = [{"role": "system", "content": load_system_prompt(domain)},
                        {"role": "user", "content": build_user_message(task)}]
            reference_env = get_environment(domain)
            if task.get("initial_state"):
                raise ValueError("This SPORK-harness collector requires tasks without initial_state overrides")
            for turn in range(1, args.max_turns_per_task + 1):
                if len(rows) >= args.prefixes:
                    break
                prompt = adapter.render_main_prompt(messages, enable_thinking=True)
                prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
                if len(prompt_ids) + args.prefix_tokens >= args.context_window:
                    skipped.append(dict(domain=domain, task_id=task_id, turn=turn, reason="context_full"))
                    break
                log(f"capture {domain}/{task_id} turn={turn}: {args.prefix_tokens} tokens")
                capture = await client.complete(session, prompt_ids, max_tokens=args.prefix_tokens,
                                                stop=adapter.main_stop, seed=args.seed)
                prefix_ids = capture["token_ids"]
                prefix_text = capture["text"]
                if (len(prefix_ids) != args.prefix_tokens or capture.get("finish_reason") != "length"
                        or "<tool_call>" in prefix_text or "</think>" in prefix_text
                        or "<think>" not in prefix_text):
                    skipped.append(dict(domain=domain, task_id=task_id, turn=turn,
                                        reason="not_an_unfinished_thinking_prefix", capture=capture))
                    break
                sample_id = f"prefix_{len(rows):03d}"
                sample_dir = run_dir / sample_id
                sample_dir.mkdir()
                # Snapshots are written/read only by this trusted local collector.
                # Do not load .pkl files from untrusted sources.
                env_bytes = pickle.dumps(reference_env, protocol=5)
                (sample_dir / "environment.pkl").write_bytes(env_bytes)
                state_hash = environment_hash(reference_env)
                snapshot = dict(sample_id=sample_id, domain=domain, task_id=task_id, turn=turn,
                                messages=json.loads(json.dumps(messages)), tools=tools,
                                prompt_text=prompt, prompt_token_ids=prompt_ids,
                                prefix_text=prefix_text, prefix_token_ids=prefix_ids,
                                visible_generated_tokens=len(prefix_ids),
                                environment_file="environment.pkl",
                                environment_sha256=sha256(env_bytes), environment_hash=state_hash)
                dump(sample_dir / "input.json", snapshot)
                dump(sample_dir / "capture.json", capture)
                order = ["off", "now", "wait"]
                branch_order_rng.shuffle(order)
                outcomes = {}
                off_executor = None
                dump(sample_dir / "order.json", order)
                for policy in order:
                    log(f"{sample_id} {domain}/{task_id} turn={turn} policy={policy}")
                    restored_at = time.perf_counter()
                    restored_env = pickle.loads(env_bytes)
                    restored_hash = environment_hash(restored_env)
                    if restored_hash != state_hash:
                        raise RuntimeError("Environment changed during snapshot restore")
                    executor = Tau2ToolExecutor(restored_env, floor_s=args.tool_floor_s)
                    restore_s = time.perf_counter() - restored_at
                    # Same exact prefix warmup before EACH policy; discard its output.
                    warmup = await client.complete(session, prompt_ids + prefix_ids,
                                                   max_tokens=1, seed=args.seed)
                    result = await run_prefix_branch(session, client, adapter, snapshot,
                                                     executor, policy, cfg)
                    result.update(environment_before=restored_hash,
                                  environment_after=environment_hash(executor.env),
                                  restore_s=restore_s, discarded_warmup_s=warmup["wall_s"])
                    dump(sample_dir / f"{policy}.json", result)
                    outcomes[policy] = result
                    if policy == "off":
                        off_executor = executor
                    log(f"{sample_id} {policy}: wall={result.get('wall_s')} "
                        f"valid={result.get('completion_valid')} error={result.get('error')}")
                labels = paired_labels(outcomes)
                labels["now_same_state_as_off"] = outcomes["now"]["environment_after"] == outcomes["off"]["environment_after"]
                labels["wait_same_state_as_off"] = outcomes["wait"]["environment_after"] == outcomes["off"]["environment_after"]
                dump(sample_dir / "labels.json", labels)
                rows.append(dict(sample_id=sample_id, domain=domain, task_id=task_id,
                                 turn=turn, order=order, labels=labels, outcomes=outcomes))
                # One compact row per prefix. Future outcomes are explicitly labels.
                with (run_dir / "pairs.jsonl").open("a") as f:
                    f.write(json.dumps(dict(sample_id=sample_id,
                                            input_file=f"{sample_id}/input.json", labels=labels),
                                       ensure_ascii=False) + "\n")
                dump(run_dir / "smoke_report.json", make_summary(rows, args.prefixes, failures, skipped))
                off = outcomes["off"]
                if not off.get("completion_valid") or not off.get("formal_call"):
                    break
                _append_tool_messages(messages, off["main_text"], off["tool_result"])
                reference_env = off_executor.env
        dump(run_dir / "status_after.json", await client.status(session))
    report = make_summary(rows, args.prefixes, failures, skipped)
    dump(run_dir / "smoke_report.json", report)
    log(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["pipeline_passed"]:
        raise RuntimeError("Collection did not pass; inspect smoke_report.json and per-branch files")
    return report


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-url", default="http://127.0.0.1:8016/v1")
    ap.add_argument("--model", default="/mnt/data/zhangshuai/LLMs/Models/Qwen3-32B")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--prefixes", type=int, default=10)
    ap.add_argument("--prefix-tokens", type=int, default=32)
    ap.add_argument("--delay-tokens", type=int, default=64)
    ap.add_argument("--max-turns-per-task", type=int, default=2)
    ap.add_argument("--max-candidate-tasks", type=int, default=30)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--context-window", type=int, default=40960)
    ap.add_argument("--tool-floor-s", type=float, default=2.0)
    ap.add_argument("--timeout-s", type=float, default=600)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    for name in ("prefixes", "prefix_tokens", "delay_tokens", "max_turns_per_task",
                 "max_candidate_tasks", "max_tokens", "context_window", "timeout_s"):
        if getattr(args, name) <= 0:
            ap.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_tokens <= args.prefix_tokens:
        ap.error("--max-tokens must exceed --prefix-tokens")
    if args.tool_floor_s < 0:
        ap.error("--tool-floor-s cannot be negative")
    return args


def main():
    args = parse_args()
    prior_manifest = (args.output / "manifest.json").exists()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tasks = candidate_tasks(args.seed, args.max_candidate_tasks)
    try:
        asyncio.run(collect(args, tokenizer, tasks))
    except Exception as exc:
        if args.output.exists() and not prior_manifest:
            dump(args.output / "fatal_error.json", {"error": f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    main()
