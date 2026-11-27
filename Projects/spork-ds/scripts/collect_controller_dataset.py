#!/usr/bin/env python3
"""Staged off/now/wait data collection. Inference is launched by the user.

--prepare-only and --analyze-only are offline. No training or server startup.
Frozen task splits, independent environments, per-branch resume, all failures
retained. Only repetition-zero off advances the reference task history.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import asdict
from datetime import datetime
import fcntl
from importlib.metadata import version
import json
from pathlib import Path
import pickle
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.repeat_prefix_pairs import digest, dump, read, log  # noqa: E402
from spork_core.controller_collection import aggregate_prefix, branch_errors, policy_orders, task_batches  # noqa: E402
from spork_core.prefix_replay import ReplayConfig  # noqa: E402

DEFAULT_SPLIT = ROOT / "prefix_data/stability_runs/stability_20261007_184230_ymqr6_ch/formal_task_split.json"
DEFAULT_EVIDENCE = ROOT / "prefix_data/diagnostic_runs/batch_invariance_continue_20261008_200116_01vkjcmo/fork_diagnostic_report.json"


def code_files():
    names = ["scripts/collect_controller_dataset.py", "scripts/run_controller_collection.sh",
             "scripts/repeat_prefix_pairs.py", "spork_core/controller_collection.py", "spork_core/prefix_replay.py",
             "spork_core/prefix_stability.py", "spork_core/qwen.py", "spork_core/tool_calls.py",
             "spork_core/executors.py", "spork_core/turn_runner.py", "spork_core/benchmarks/tau2.py",
             "spork_core/vllm029_d3.py", "launch_vllm_spork_http.py", "scripts/serve_qwen32_prefix_collection.sh",
             "scripts/serve_qwen32_prefix_invariant.sh", "scripts/check_controller_training_data.py",
             "scripts/run_controller_stage3.sh"]
    return [ROOT / name for name in names]


def seal(directory, names):
    dump(directory / "completed.json", {"sha256": {n: digest(directory / n) for n in names}})


def verified(directory):
    path = directory / "completed.json"
    if not path.exists():
        return False
    for name, checksum in read(path)["sha256"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts or digest(directory / name) != checksum:
            raise ValueError(f"Changed checkpoint: {directory / name}")
    return True


def branch_dir(turn, rep, policy):
    return turn / "measurements" / f"repeat_{rep:03d}" / policy


def saved_result(directory):
    if not verified(directory):
        return None
    index = read(directory / "result_index.json")
    relative = Path(index["result"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe saved result path")
    return read(directory / relative)


def prepare(args):
    output = args.output.resolve()
    path = output / "manifest.json"
    if path.exists():
        m = read(path)
        if m.get("purpose") != "staged_controller_prefix_collection":
            raise ValueError("Wrong dataset purpose")
        if not args.analyze_only:
            for name, checksum in m["code_sha256"].items():
                if digest(ROOT / name) != checksum:
                    raise ValueError(f"Code changed after preparation: {name}")
            for source in m["task_sources"]:
                if digest(Path(source["path"])) != source["sha256"]:
                    raise ValueError("Task source changed")
            if digest(output / "task_split.json") != m["task_split_sha256"]:
                raise ValueError("Frozen task split changed")
        return m
    if args.analyze_only:
        raise ValueError("No prepared dataset")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty new output directory")
    split = read(args.task_split)
    evidence = read(args.evidence)
    if not evidence.get("collection_complete") or not evidence.get("per_prefix"):
        raise ValueError("The bounded invariance check must be complete")
    if any(not x["reference_tokens_stable"] or any(x["token_differences_by_pair"].values())
           for x in evidence["per_prefix"]):
        raise ValueError("Observed token forks remain in the bounded check")
    if any(not e.endswith(":failed_or_truncated_completion") for e in evidence["errors"]):
        raise ValueError("Unresolved diagnostic integrity errors")
    for source in split["sources"]:
        if digest(Path(source["path"])) != source["sha256"]:
            raise ValueError("Tasks differ from the frozen task split")
    batches = task_batches(split["tasks"])
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.task_split, output / "task_split.json")
    shutil.copyfile(args.evidence, output / "invariance_evidence.json")
    m = {
        "schema_version": 1, "purpose": "staged_controller_prefix_collection", "dataset_id": output.name,
        "created_at": datetime.now().isoformat(), "model": "/mnt/data/zhangshuai/LLMs/Models/Qwen3-32B",
        "model_url": "http://127.0.0.1:8016/v1", "timeout_s": 600, "tool_floor_s": 2.,
        "replay_config": asdict(ReplayConfig()), "prefix_tokens": 32, "repetitions": 3,
        "max_turns_per_task": 2, "batches": batches,
        "task_split_sha256": digest(output / "task_split.json"), "task_sources": split["sources"],
        "reserved_test_tasks": [t for t in split["tasks"] if t["split"] == "test"],
        "code_sha256": {str(p.relative_to(ROOT)): digest(p) for p in code_files()},
        "packages": {n: version(n) for n in ("vllm", "torch", "transformers", "aiohttp", "tau2")},
        "start_condition": "Bounded concurrent-generation forks no longer observed. Earlier ten-prefix/six-repeat "
                           "label gate is NOT claimed passed; this first data batch includes label validation.",
        "collection_scope": "Existing SPORK fixed-user-message airline/retail harness; not official interactive tau2 task success.",
        "reference_policy": "Only the first off repetition advances history. If it truncates, retain recorded failures, "
                            "mark remaining comparisons unobserved and stop that task; never impute zero gains.",
        "precision_policy": "Three position-balanced repeats, all measured gains retained; inspect nominal intervals "
                            "before training. No automatic training, no test-task selection or collection.",
        "server_condition": "All policies use the same batch-invariant server and artificial two-second tool floor.",
        "training_gate": {"min_usable_fraction": .8, "min_precise_fraction": .8,
                          "max_token_disagreement_prefixes": 0,
                          "first_batch_min_usable_prefixes": 20, "first_batch_min_train_groups": 8,
                          "first_batch_min_validation_groups": 3, "final_min_usable_prefixes": 100,
                          "final_min_train_groups": 30, "final_min_validation_groups": 8,
                          "interpretation": "Fixed engineering checks for the first controller fit; "
                                            "three-repeat intervals are not a statistical guarantee or the previous six-repeat test."},
    }
    for p in code_files():
        target = output / "code" / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
    dump(path, m)
    return m


def state_hash(env):
    return {"agent_db": env.get_db_hash(), "user_db": env.get_user_db_hash()}


def check_server(status):
    if (status.get("server_config", {}).get("batch_invariant_requested") is not True
            or not status.get("has_drafter") or status.get("active_requests") != 0
            or status.get("pending_verifications") != 0):
        raise RuntimeError("An idle batch-invariant D3 server is required")


async def measure(session, client, adapter, inp, env_bytes, turn, rep, policy, manifest):
    from spork_core.executors import Tau2ToolExecutor
    from spork_core.prefix_replay import run_prefix_branch
    directory = branch_dir(turn, rep, policy)
    result = saved_result(directory)
    if result is not None:
        errors = branch_errors(policy, result, inp, manifest["replay_config"])
        if errors:
            raise ValueError(f"Invalid saved branch: {directory}: {errors}")
        return result
    directory.mkdir(parents=True, exist_ok=True)
    attempt = directory / f"attempt_{len(list(directory.glob('attempt_*'))):03d}"
    attempt.mkdir()
    restored_at = time.perf_counter()
    env = pickle.loads(env_bytes)  # Local snapshot, validated by its checkpoint checksum.
    before = state_hash(env)
    if before != inp["environment_hash"]:
        raise ValueError("Environment restore mismatch")
    executor = Tau2ToolExecutor(env, floor_s=manifest["tool_floor_s"])
    restore_s = time.perf_counter() - restored_at
    warm = await client.complete(session, inp["prompt_token_ids"] + inp["prefix_token_ids"],
                                 max_tokens=1, seed=manifest["replay_config"]["seed"])
    enabled = policy != "off"
    result = await run_prefix_branch(session, client, adapter, inp, executor, policy,
                                     ReplayConfig(**manifest["replay_config"]), enable_d3=enabled,
                                     enable_speculative_tools=enabled, record_token_timing=True)
    result.update(environment_before=before, environment_after=state_hash(executor.env),
                  restore_s=restore_s, discarded_warmup_s=warm["wall_s"])
    dump(attempt / "result.json", result)
    errors = branch_errors(policy, result, inp, manifest["replay_config"])
    dump(attempt / "integrity.json", {"errors": errors, "truncated": result.get("finish_reason") == "length"})
    if errors:
        raise RuntimeError(f"Branch integrity failure at {attempt}: {errors}")
    names = [f"{attempt.name}/result.json", f"{attempt.name}/integrity.json", "result_index.json"]
    if rep == 0 and policy == "off":
        (attempt / "environment_after.pkl").write_bytes(pickle.dumps(executor.env, protocol=5))
        names.append(f"{attempt.name}/environment_after.pkl")
    dump(directory / "result_index.json", {"result": f"{attempt.name}/result.json"})
    seal(directory, names)
    log(f"{inp['sample_id']} r{rep + 1} {policy}: {result['finish_reason']}, "
        f"{len(result['continuation_token_ids'])} tokens, {result['wall_s']:.3f}s")
    return result


async def collect_task(session, client, tokenizer, task, manifest, output):
    from spork_core.benchmarks.tau2 import build_user_message, get_environment, load_system_prompt, load_tools
    from spork_core.qwen import QwenToolCallAdapter
    from spork_core.turn_runner import _append_tool_messages
    directory = output / "tasks" / f"{task['domain']}_{task['task_id']}"
    if verified(directory):
        return
    directory.mkdir(parents=True, exist_ok=True)
    data_path = next(Path(s["path"]) for s in manifest["task_sources"] if s["domain"] == task["domain"])
    raw_task = next(t for t in read(data_path) if str(t.get("id", t.get("task_id"))) == task["task_id"])
    if raw_task.get("initial_state"):
        raise ValueError("Unsupported task initial_state override")
    tools = load_tools(task["domain"])
    adapter = QwenToolCallAdapter(tokenizer, tools)
    messages = [{"role": "system", "content": load_system_prompt(task["domain"])},
                {"role": "user", "content": build_user_message(raw_task)}]
    env = get_environment(task["domain"])
    reason, prefixes = "turn_limit", []
    for turn_no in range(1, manifest["max_turns_per_task"] + 1):
        turn = directory / f"turn_{turn_no:03d}"
        snapshot_dir = turn / "snapshot"
        prompt = adapter.render_main_prompt(messages, enable_thinking=True)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(prompt_ids) + manifest["prefix_tokens"] >= manifest["replay_config"]["context_window"]:
            reason = "context_full"
            break
        if not verified(snapshot_dir):
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            capture_path = snapshot_dir / "capture.json"
            if capture_path.exists():
                # Interrupted capture remains evidence, not a silently overwritten retry.
                raise RuntimeError(f"Unsealed capture at {snapshot_dir}; inspect before resuming")
            log(f"Capture {task['group_id']} turn {turn_no}")
            capture = await client.complete(session, prompt_ids, max_tokens=manifest["prefix_tokens"],
                                            stop=adapter.main_stop, seed=manifest["replay_config"]["seed"])
            dump(capture_path, capture)
            sid = f"{task['domain']}_{task['task_id']}_turn_{turn_no:03d}"
            inp = {**task, "sample_id": sid, "turn": turn_no, "messages": messages, "tools": tools,
                   "prompt_text": prompt, "prompt_token_ids": prompt_ids,
                   "prefix_text": capture["text"], "prefix_token_ids": capture["token_ids"],
                   "visible_generated_tokens": len(capture["token_ids"]), "environment_hash": state_hash(env)}
            dump(snapshot_dir / "input.json", inp)
            (snapshot_dir / "environment.pkl").write_bytes(pickle.dumps(env, protocol=5))
            seal(snapshot_dir, ["capture.json", "input.json", "environment.pkl"])
        inp, capture = read(snapshot_dir / "input.json"), read(snapshot_dir / "capture.json")
        if (inp["messages"] != messages or inp["prompt_token_ids"] != prompt_ids or inp["prompt_text"] != prompt
                or inp["environment_hash"] != state_hash(env)
                or tokenizer.decode(inp["prefix_token_ids"], skip_special_tokens=False) != inp["prefix_text"]):
            raise ValueError("Saved decision prefix does not reproduce")
        if (len(inp["prefix_token_ids"]) != manifest["prefix_tokens"] or capture["finish_reason"] != "length"
                or "<think>" not in inp["prefix_text"] or "</think>" in inp["prefix_text"] or "<tool_call>" in inp["prefix_text"]):
            reason = "no_decision_at_32_thinking_tokens"
            break
        env_bytes = (snapshot_dir / "environment.pkl").read_bytes()
        orders = policy_orders(inp["sample_id"])
        repetitions, stop_reference = [], False
        for rep, order in enumerate(orders):
            outcomes = {}
            for policy in order:
                outcomes[policy] = await measure(session, client, adapter, inp, env_bytes, turn, rep, policy, manifest)
                if rep == 0 and policy == "off" and not outcomes[policy]["completion_valid"]:
                    stop_reference = True
                    break
            repetitions.append(outcomes)
            if stop_reference:
                break
        row, audit = aggregate_prefix(inp, repetitions, manifest)
        audit["unobserved_branches_after_reference_failure"] = 9 - audit["recorded_branch_runs"] if stop_reference else 0
        dump(turn / "row.json", row)
        dump(turn / "label_audit.json", audit)
        seal(turn, ["row.json", "label_audit.json", "snapshot/completed.json"] + [
            str(branch_dir(turn, rep, p).relative_to(turn) / "completed.json")
            for rep, values in enumerate(repetitions) for p in values])
        prefixes.append(str(turn.relative_to(directory)))
        log(f"{inp['sample_id']}: label_usable={audit['label_usable']}, "
            f"precise={audit['meets_initial_precision_criterion']}, truncated={audit['truncated_branch_runs']}")
        off = repetitions[0]["off"]
        if stop_reference or not off.get("formal_call"):
            reason = "reference_truncated" if stop_reference else "reference_answer_finished"
            break
        _append_tool_messages(messages, off["main_text"], off["tool_result"])
        off_dir = branch_dir(turn, 0, "off")
        saved = off_dir / read(off_dir / "result_index.json")["result"]
        env = pickle.loads((saved.parent / "environment_after.pkl").read_bytes())
        if state_hash(env) != off["environment_after"]:
            raise ValueError("Saved reference state mismatch")
    dump(directory / "task_report.json", {**task, "stop_reason": reason, "prefixes": prefixes,
                                          "task_success_evaluated": False})
    seal(directory, ["task_report.json"] + [p + "/completed.json" for p in prefixes])


def export_batch(output, manifest, batch_index):
    rows, audits, reports, counters = [], [], [], Counter()
    for task in manifest["batches"][batch_index]:
        directory = output / "tasks" / f"{task['domain']}_{task['task_id']}"
        if not verified(directory):
            continue
        report = read(directory / "task_report.json")
        reports.append(report)
        for name in report["prefixes"]:
            turn = directory / name
            if not verified(turn) or not verified(turn / "snapshot"):
                raise ValueError("Incomplete sealed prefix")
            inp = read(turn / "snapshot/input.json")
            repetitions = []
            for rep in range(3):
                values = {p: saved_result(branch_dir(turn, rep, p)) for p in ("off", "now", "wait")}
                values = {p: value for p, value in values.items() if value is not None}
                if values:
                    for p, value in values.items():
                        errors = branch_errors(p, value, inp, manifest["replay_config"])
                        if errors:
                            raise ValueError(f"Saved branch integrity failure: {errors}")
                        counters.update(value.get("d3_counters", {}))
                    repetitions.append(values)
            row, audit = aggregate_prefix(inp, repetitions, manifest)
            if row != read(turn / "row.json"):
                raise ValueError("Labels differ from saved raw outcomes")
            rows.append(row)
            audits.append(audit)
    destination = output / f"batch_{batch_index:03d}"
    destination.mkdir(exist_ok=True)
    temp = destination / "training_rows.jsonl.tmp"
    temp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    temp.replace(destination / "training_rows.jsonl")
    good = [r for r in rows if r["label_valid"]]
    report = {
        "batch": batch_index, "planned_tasks": len(manifest["batches"][batch_index]), "completed_tasks": len(reports),
        "collection_complete": len(reports) == len(manifest["batches"][batch_index]),
        "captured_prefixes": len(rows), "usable_label_prefixes": len(good),
        "usable_label_fraction": len(good) / len(rows) if rows else 0.,
        "recorded_branches": sum(a["recorded_branch_runs"] for a in audits),
        "truncated_branches": sum(a["truncated_branch_runs"] for a in audits),
        "token_disagreement_prefixes": sum(not a["all_observed_tokens_equal"] for a in audits),
        "precise_usable_prefixes": sum(a["label_usable"] and a["meets_initial_precision_criterion"] for a in audits),
        "usable_groups_by_split": {s: len({r["group_id"] for r in good if r["split"] == s}) for s in ("train", "validation")},
        "task_stop_reasons": dict(Counter(r["stop_reason"] for r in reports)),
        "best_mean_policy_counts": dict(Counter(max({"off": 0., "now": r["targets"]["gain_now_s"],
                                                        "wait": r["targets"]["gain_wait_s"]}, key=lambda p: {
                                                            "off": 0., "now": r["targets"]["gain_now_s"],
                                                            "wait": r["targets"]["gain_wait_s"]}[p]) for r in good)),
        "training_started": False, "test_tasks_used": False, "per_prefix_audits": audits,
        "d3_counter_totals": dict(counters), "training_rows_sha256": digest(destination / "training_rows.jsonl"),
        "scope": "Initial three-repeat data batch, not an automatic training gate or online speedup result. "
                 "Selection on completed/matching outputs limits coverage; failures remain in this report and JSONL.",
    }
    dump(destination / "collection_report.json", report)
    return report


async def collect(output, manifest, batch_index, tokenizer):
    import aiohttp
    from spork_core.prefix_replay import PrefixHttpClient
    client = PrefixHttpClient(manifest["model_url"], manifest["model"], manifest["timeout_s"])
    destination = output / f"batch_{batch_index:03d}"
    destination.mkdir(exist_ok=True)
    async with aiohttp.ClientSession(trust_env=False, connector=aiohttp.TCPConnector(force_close=True)) as session:
        async with session.get(client.base_url + "/models", timeout=15) as response:
            response.raise_for_status()
            models = await response.json()
        if manifest["model"] not in [x["id"] for x in models.get("data", [])]:
            raise RuntimeError("Wrong served model")
        before = await client.status(session)
        check_server(before)
        dump(destination / f"preflight_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json", {"models": models, "status": before})
        warm = tokenizer.apply_chat_template([{"role": "user", "content": "Explain how to look up an order."}],
                                             tokenize=False, add_generation_prompt=True, enable_thinking=True)
        await client.complete(session, tokenizer.encode(warm, add_special_tokens=False), max_tokens=128, seed=42)
        for task in manifest["batches"][batch_index]:
            await collect_task(session, client, tokenizer, task, manifest, output)
            report = export_batch(output, manifest, batch_index)
            log(f"Batch {batch_index}: {report['completed_tasks']}/{report['planned_tasks']} tasks, "
                f"{report['usable_label_prefixes']}/{report['captured_prefixes']} usable prefixes")
        after = await client.status(session)
        dump(destination / "status_after.json", after)
        check_server(after)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--task-split", type=Path, default=DEFAULT_SPLIT)
    ap.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    ap.add_argument("--batch", type=int, default=0)
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument("--prepare-only", action="store_true")
    modes.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    args.output = args.output.resolve()
    manifest = prepare(args)
    if not 0 <= args.batch < len(manifest["batches"]):
        ap.error("Batch index is out of range")
    log(f"Dataset: {args.output}; batch {args.batch}, {len(manifest['batches'][args.batch])} tasks")
    if args.prepare_only:
        return
    with (args.output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not args.analyze_only:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(manifest["model"], local_files_only=True)
            try:
                asyncio.run(collect(args.output, manifest, args.batch, tokenizer))
            except Exception as exc:
                dump(args.output / f"failure_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json",
                     {"batch": args.batch, "error": f"{type(exc).__name__}: {exc}"})
                raise
        report = export_batch(args.output, manifest, args.batch)
        print(json.dumps({k: v for k, v in report.items() if k != "per_prefix_audits"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
