#!/usr/bin/env python3
"""Frozen multi-turn evaluation of SPORK versus exact-input successor reuse.

Default: prepare 155 tasks x 2 methods x 8 maximum turns, without GPU requests.
Pass --scope train --task-ids ... for bounded development checks.
Pass --run only when ready to execute in tmux. Completed failures are retained.
Quality is bounded-trajectory agreement and optional fixed-message ACTION reward,
not official interactive tau2 success.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import ExitStack
from datetime import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback
import uuid
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spork_core import paths
from spork_core.benchmarks.tau2 import get_environment, load_system_prompt, load_tools
from spork_core.evaluation_artifacts import (archive_unfinished, complete_record,
    digest, freeze_plan, planned_records, read_completed, record_path,
    visible_final_answers, write_json)
from spork_core.executors import Tau2ToolExecutor
from spork_core.prefix_replay import COUNTERS, PrefixHttpClient, counter_delta
from spork_core.qwen import QwenToolCallAdapter
from spork_core.qwen35_native import NativeQwen35Adapter
from spork_core.successor_task import run_successor_task
from spork_core.turn_runner import RetryTurnConfig
from spork_core.vllm_client import VllmClient
from spork_unified_eval import load_tau2_tasks, tau2_action_reward
from scripts.run_successor_smoke import signature


def evaluation_session():
    """Do not reuse a socket after synchronous environment/data preparation."""
    import aiohttp
    return aiohttp.ClientSession(trust_env=False,
                                connector=aiohttp.TCPConnector(force_close=True))


async def reset_trial_cache(session, http):
    """Clear previous-trial prefix blocks without interrupting live requests."""
    started = time.perf_counter()
    result = None
    for attempt in range(20):
        async with session.post(http.root + "/spork/reset_prefix_cache", json={}, timeout=30) as response:
            response.raise_for_status()
            result = await response.json()
        if result.get("success") is True:
            return {"success": True, "attempts": attempt + 1, "wall_s": time.perf_counter() - started}
        await asyncio.sleep(.05)
    raise RuntimeError(f"Prefix cache is still held by live work; reset refused: {result}")


def analyze(output, manifest):
    records, unfinished = planned_records(output, manifest)
    groups = {}
    for record in records:
        groups.setdefault((record["group_id"], record["repeat"]), {})[record["config"]] = record
    report = {"expected": len(manifest["schedule"]), "recorded": len(records),
              "complete": len(records) == len(manifest["schedule"]), "unsealed": unfinished,
              "scope": manifest["quality_scope"], "splits": {}, "comparisons": []}
    for split in ("all", "train", "validation", "test"):
        subset = [r for r in records if split == "all" or r["split"] == split]
        totals = {}
        for mode in manifest["configs"]:
            chosen = [r for r in subset if r["config"] == mode]
            good = [r for r in chosen if not r["errors"]]
            totals[mode] = {"recorded": len(chosen), "failed": len(chosen) - len(good),
                "mean_s": statistics.mean(r["wall_s"] for r in good) if good else None,
                "adopted_pairs": sum(r["prefetch"].get("adopted", 0) for r in chosen),
                "early_children_started": sum(
                    p.get("child_launch_phase") == "before_parent_completion"
                    for r in chosen for p in r["prefetch"].get("pairs", [])),
                "turn_limits": sum(bool(r.get("turn_limit_reached")) for r in chosen)}
        paired = [g for g in groups.values() if set(g) == set(manifest["configs"])
                  and not any(r["errors"] for r in g.values())
                  and (split == "all" or g["spork"]["split"] == split)]
        matched = {"n": len(paired), "mean_s": {
            c: statistics.mean(g[c]["wall_s"] for g in paired) if paired else None
            for c in manifest["configs"]}}
        report["splits"][split] = {"totals": totals, "common_error_free_pairs": matched}
    for (group, repeat), methods in groups.items():
        base = methods.get("spork")
        if base is None:
            continue
        for mode, row in methods.items():
            if mode == "spork":
                continue
            hashes = base.get("initial_prompt_sha256"), row.get("initial_prompt_sha256")
            same_initial = (hashes[0] == hashes[1]) if all(hashes) else None
            good = not base["errors"] and not row["errors"] and same_initial is not False
            answers = visible_final_answers(base), visible_final_answers(row)
            shared_turns = min(len(base["turns"]), len(row["turns"]))
            shared_tokens_equal = all(a["main_token_ids"] == b["main_token_ids"]
                for a, b in zip(base["turns"], row["turns"]))
            shared_actions_equal = signature(base)[:shared_turns] == signature(row)[:shared_turns]
            report["comparisons"].append({
                "group_id": group, "split": base["split"], "repeat": repeat, "config": mode,
                "valid": good, "same_session": base.get("session_id") == row.get("session_id"),
                "same_initial_prompt": same_initial,
                "shared_completed_turns": shared_turns,
                "shared_completed_tokens_equal": shared_tokens_equal if shared_turns else None,
                "shared_completed_actions_equal": shared_actions_equal if shared_turns else None,
                "baseline_s": base["wall_s"], "method_s": row["wall_s"],
                "saved_s": base["wall_s"] - row["wall_s"],
                "same_actor_tokens": good and [t["main_token_ids"] for t in base["turns"]]
                    == [t["main_token_ids"] for t in row["turns"]],
                "same_actions_results_state": good and signature(base) == signature(row),
                "visible_answer_observed": bool(answers[0] or answers[1]),
                "same_visible_answer": (answers[0] == answers[1]) if answers[0] or answers[1] else None,
                "same_turn_limit": base.get("turn_limit_reached") == row.get("turn_limit_reached"),
                "adopted_pairs": row["prefetch"].get("adopted", 0),
                "observed_wasted_child_tokens": row["prefetch"].get("observed_wasted_child_tokens", 0)})
    report["failures"] = [{"group_id": r["group_id"], "config": r["config"], "errors": r["errors"]}
                          for r in records if r["errors"]]
    report["comparison_rule"] = "All scheduled pairs retained, including slowdowns, mismatches and failures."
    write_json(output / "summary.json", report)
    return report


def prepare(args):
    split = json.loads(args.task_split.read_text())
    selected = [t for t in split["tasks"] if t["domain"] in {"airline", "retail"}
                and (args.scope == "all" or t["split"] == args.scope)]
    if args.task_ids:
        wanted = args.task_ids.split(",")
        by_id = {t["group_id"]: t for t in selected}
        if len(wanted) != len(set(wanted)) or any(t not in by_id for t in wanted):
            raise ValueError("Task IDs must be unique and belong to the selected split")
        selected = [by_id[t] for t in wanted]
    else:
        selected.sort(key=lambda t: (t["domain"], int(t["task_id"])))
    configs = args.configs.split(",")
    if (not selected or "spork" not in configs or len(configs) != len(set(configs))
            or set(configs) - {"spork", "raw", "raw_fallback", "confirmed", "serial", "actor_only"}):
        raise ValueError("Select unique configs from spork,raw,raw_fallback,confirmed,serial,actor_only including spork")
    family = getattr(args, "model_family", "qwen3")
    reference_gdn = family == "qwen35" and os.environ.get("SPORK_Q35_REFERENCE_GDN", "0") == "1"
    parser_root = (Path(importlib.util.find_spec("vllm").origin).parent
                   if family == "qwen35" else None)
    parser_sources = sorted((parser_root / "parser").rglob("*.py")) if parser_root else []
    if family == "qwen35" and "confirmed" in configs:
        raise ValueError("The legacy free-confirmation probe has not been ported to Qwen3.5")
    schedule = []
    for repeat in range(args.repetitions):
        for index, task in enumerate(selected):
            shift = (index + repeat) % len(configs)
            schedule.extend({**task, "repeat": repeat, "config": c}
                            for c in configs[shift:] + configs[:shift])
    task_map = {f'{t["domain"]}:{t["task_id"]}': t for t in load_tau2_tasks(args.seed)}
    sources = sorted(set([Path(__file__), ROOT / "scripts/run_successor_smoke.py",
        ROOT / "spork_unified_eval.py", ROOT / "tau2_tool_parser.py", ROOT / "launch_vllm_spork_http.py",
        ROOT / "scripts/serve_qwen32_candidate_checks.sh", ROOT / "scripts/codex_env.sh",
        ROOT / "scripts/run_codex_full_eval.sh", ROOT / "scripts/check_codex_runtime.py"]
        + list(ROOT.glob("*.py"))
        + list((ROOT / "spork_core").rglob("*.py"))
        + list((ROOT / "tau2-bench/src/tau2").rglob("*.py"))
        + list((ROOT / "coder/scripts").glob("tau2_tool_parser.py"))
        + ([ROOT / "scripts" / name for name in (
            "qwen35_env.sh", "serve_qwen35_9b.sh", "run_qwen35_experiments.sh",
            "check_qwen35_experiment.py")] if family == "qwen35" else [])))
    data_files = [p for d in {t["domain"] for t in selected}
                  for p in (paths.TAU2_ROOT / "data/tau2/domains" / d).rglob("*") if p.is_file()]
    data_files.append(args.task_split)
    manifest = {
        "purpose": "multi_turn_successor_evaluation", "created_at": datetime.now().isoformat(),
        "model_family": family,
        "native_parser": "vllm.parser.qwen3.Qwen3Parser with local schema/type validation" if family == "qwen35" else None,
        "external_parser_root": str(parser_root) if parser_root else None,
        "external_parser_source_sha256": {str(p.relative_to(parser_root)): digest(p) for p in parser_sources},
        "gdn_execution": ({"decode_kernel": os.environ.get("VLLM_GDN_DECODE_KERNEL", "cuda"),
            "packed_recurrent_decode": os.environ.get("VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE", "1"),
            "reference_core": reference_gdn}
            if family == "qwen35" else None),
        "batch_invariant_expected": family != "qwen35" or reference_gdn,
        "determinism_scope": ("Experimental worker-local GDN shared recurrence with native speculative-state addressing; "
            "scoped kernel checks passed, full-model exact-trajectory gate required; not upstream or universal support"
            if reference_gdn else "GDN backend does not support batch-invariant mode; "
            "greedy plus empirical exact-trajectory gate, not a backend guarantee"
            if family == "qwen35" else "batch-invariant backend requested"),
        "serving_profile": ({"visible_devices": os.environ.get("SPORK_Q35_DEVICES", "2,3"),
            "tensor_parallel_size": os.environ.get("SPORK_Q35_TP", "2"),
            "max_model_len": "40960", "max_num_seqs": "4",
            "max_num_batched_tokens": os.environ.get("SPORK_Q35_BATCH_TOKENS", "4096"),
            "block_size": os.environ.get("SPORK_Q35_BLOCK_SIZE", "640"),
            "hf_overrides": json.dumps({"head_dtype": os.environ.get("SPORK_Q35_HEAD_DTYPE", "model")}, separators=(",", ":")),
            "enforce_eager": reference_gdn,
            "compilation_config": '{"mode":0,"cudagraph_mode":0}' if reference_gdn else None} if family == "qwen35" else None),
        "history_policy": (NativeQwen35Adapter.history_policy if family == "qwen35"
                           else "existing_Qwen3_SPORK_history"),
        "require_identical_trajectories": getattr(args, "require_identical_trajectories", False),
        "config_semantics": {
            "spork": "Original full-reasoning SPORK with D1/D2/D3",
            "raw": "Early complete next Actor and isolated tool execution",
            "raw_fallback": "Raw plus at most one canonical SPORK attempt after a failed isolated child; extra attempt and cleanup remain timed; no stochastic distribution guarantee",
            "serial": "Build the same candidate ticket, validate first, then run its child without D3; misses use canonical SPORK",
            "actor_only": "Early next Actor; all child tool execution waits for parent validation",
            "confirmed": "Legacy Qwen3-only extra free confirmation before early child launch"},
        "scope": args.scope, "tasks": selected, "schedule": schedule, "configs": configs,
        "seed": args.seed, "repetitions": args.repetitions, "max_turns": args.max_turns,
        "presence_penalty": getattr(args, "presence_penalty", 0.0),
        "model": args.model, "base_url": args.base_url, "max_tokens": args.max_tokens,
        "tool_floor_s": args.tool_floor, "context_window": 40960,
        "strategy": "Disjoint full-Actor pairs; exact parent action/observation/state/next-prompt/generation matching; no early reasoning exit.",
        "child_d3": "Uncommitted child uses D1/D2 and ordinary Target generation; D3 state is disabled in child.",
        "quality_scope": "Fixed user messages and bounded trajectories. ACTION is not official interactive tau2 success.",
        "measurement": "Whole-task wall includes admission, rejection and thread cleanup; one discarded initial-prompt token before timing each task.",
        "http_transport": "Fresh TCP connection per request; no automatic generation retries",
        "prefix_cache_policy": "Reset local prefix cache before every task/config/repeat, then one identical initial-prompt token warmup. Within-task sharing remains enabled.",
        "task_material_sha256": hashlib.sha256(json.dumps([task_map[t["group_id"]] for t in selected], sort_keys=True).encode()).hexdigest(),
        "input_sha256": {str(p): digest(p) for p in sorted(set(data_files))},
        "model_config_sha256": {f: digest(Path(args.model) / f)
            for f in ("config.json", "tokenizer_config.json", "generation_config.json")
            if (Path(args.model) / f).is_file()},
        "absent_optional_model_files": [f for f in ("generation_config.json",)
            if not (Path(args.model) / f).is_file()],
        "code_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources}}
    frozen = freeze_plan(args.output, manifest, ROOT, sources)
    for source in parser_sources:
        target = args.output / "external_code" / "vllm" / source.relative_to(parser_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and digest(target) != digest(source):
            raise ValueError("Frozen external parser changed")
        if not target.exists():
            target.write_bytes(source.read_bytes())
    return frozen, task_map


async def execute(args, manifest, task_map):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    client = VllmClient(args.base_url, args.model, 900)
    http = PrefixHttpClient(args.base_url, args.model, 900)
    client.presence_penalty = http.presence_penalty = manifest.get("presence_penalty", 0.0)
    session_id = uuid.uuid4().hex
    async with evaluation_session() as session:
        status = await http.status(session)
        if (status.get("server_config", {}).get("batch_invariant_requested")
                != manifest.get("batch_invariant_expected", True)
                or status.get("active_requests") != 0 or status.get("pending_verifications") != 0):
            raise RuntimeError("A dedicated D3 server matching the frozen batch mode with clean draft state is required")
        if not status.get("server_config", {}).get("prefix_cache_reset_available"):
            raise RuntimeError("Restart using the updated launcher with /spork/reset_prefix_cache before this evaluation")
        if (manifest.get("gdn_execution") or {}).get("reference_core"):
            from spork_core.gdn_invariant_worker import verify_runtime_status
            verify_runtime_status(status, manifest["code_sha256"])
        write_json(args.output / "server_sessions" / (session_id + ".json"),
                   {"started_at": datetime.now().isoformat(), "status": status})
        for position, item in enumerate(manifest["schedule"]):
            path = record_path(args.output, item)
            if read_completed(path, item) is not None:
                continue
            # Domain data and schemas can be read afresh by each task loader.
            for source, checksum in manifest["input_sha256"].items():
                if digest(source) != checksum:
                    raise RuntimeError(f"Frozen benchmark data changed: {source}")
            record = {**item, "session_id": session_id, "turns": [], "errors": [], "prefetch": {},
                      "previous_unsealed": archive_unfinished(path)}
            partial = args.output / "attempts" / (path.stem + "_" + session_id + ".json")
            record["attempt_path"] = str(partial)
            domain = item["domain"]
            task = task_map[item["group_id"]]
            adapter_cls = NativeQwen35Adapter if getattr(args, "model_family", "qwen3") == "qwen35" else QwenToolCallAdapter
            adapter = adapter_cls(tok, load_tools(domain))
            messages = [{"role": "system", "content": load_system_prompt(domain)},
                        {"role": "user", "content": task["user_message"]}]
            executor = Tau2ToolExecutor(get_environment(domain), floor_s=args.tool_floor)

            def make_cfg(current_messages, turn):
                prompt = adapter.render_main_prompt(current_messages, enable_thinking=True)
                plen = len(tok.encode(prompt, add_special_tokens=False))
                rid = "successor-full-" + uuid.uuid4().hex
                async def register(call, _prompt):
                    if isinstance(adapter, NativeQwen35Adapter):
                        ids = tok.encode(adapter.draft_body(call), add_special_tokens=False)
                        async with session.post(http.root + "/spork/set_tokens",
                                json={"request_id": rid, "prompt_len": plen, "draft_tokens": ids},
                                timeout=30) as response:
                            response.raise_for_status()
                            return await response.json()
                    return await http.set_draft(session, rid, plen, call, tok)
                async def clear():
                    result = await http.clear(session)
                    return {**await http.status(session), "clear_status": result["status"]}
                return RetryTurnConfig(max_retries=5, confidence_threshold=.9, retry_token_step=50,
                    probe_max_tokens=160, baseline_max_tokens=args.max_tokens, context_window_tokens=40960,
                    enable_thinking=True, seed=args.seed, d3_set=register, d3_clear=clear,
                    main_request_id=rid, launch_mode="original")

            def checkpoint(entry):
                record["turns"].append(entry)
                write_json(partial, record)
                write_json(args.output / "in_progress.json", record)
                print(f'  turn {entry["turn"]}: {entry.get("baseline_tool_name") or "reply"}', flush=True)

            ids = tok.encode(adapter.render_main_prompt(messages, enable_thinking=True), add_special_tokens=False)
            record["initial_prompt_sha256"] = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
            record["initial_prompt_tokens"] = len(ids)
            record["stage"] = "prefix_cache_reset"
            write_json(partial, record)
            record["prefix_cache_reset"] = await reset_trial_cache(session, http)
            record["stage"] = "warmup"
            write_json(partial, record)
            await http.complete(session, ids, max_tokens=1, seed=args.seed)
            status = await http.status(session)
            print(f'[{position + 1}/{len(manifest["schedule"])}] {item["split"]} {item["group_id"]} {item["config"]}', flush=True)
            started = time.perf_counter()
            record["stage"] = "running"
            try:
                await run_successor_task(session, client, adapter, messages, executor, make_cfg,
                    mode=("baseline" if item["config"] == "spork" else
                          "raw" if item["config"] == "raw_fallback" else item["config"]),
                    recover_child_failure=item["config"] == "raw_fallback",
                    max_turns=args.max_turns, on_turn=checkpoint, trace=record["prefetch"])
                record["errors"].extend(record["prefetch"].get("errors", []))
            except asyncio.CancelledError:
                record.update(stage="interrupted", wall_s=time.perf_counter() - started)
                record["errors"].append("interrupted; incomplete attempt retained, not sealed as a completed trial")
                write_json(partial, record)
                write_json(args.output / "in_progress.json", record)
                raise
            except Exception as exc:
                record["errors"].append(f"{type(exc).__name__}: {exc}")
                record["traceback"] = traceback.format_exc()
                if getattr(exc, "partial_generation", None) is not None:
                    record["failed_generation"] = exc.partial_generation
            record["wall_s"] = time.perf_counter() - started
            record["turn_limit_reached"] = record["prefetch"].get("turn_limit_reached", False)
            dirty = False
            try:
                after = await http.status(session)
                record["d3_counters"] = counter_delta(status, after)
                if (any(k not in after for k in COUNTERS)
                        or any(v < 0 for v in record["d3_counters"].values())
                        or any(record["d3_counters"].get(k) for k in ("accounting_errors", "dropped_injections"))
                        or after.get("active_requests") != 0 or after.get("pending_verifications") != 0):
                    raise RuntimeError("Residual, dropped or reset D3 state")
            except Exception as exc:
                dirty = True
                record.setdefault("d3_counters", {})
                record["errors"].append(f"server_integrity: {type(exc).__name__}: {exc}")
            for turn in record["turns"]:
                if not turn.get("completion_valid") or turn.get("spec_cleanup_error"):
                    record["errors"].append("turn_completion_or_cleanup")
                if (turn.get("d3_status") or {}).get("clear_status") not in ("ok", "cleared"):
                    dirty = True
                    record["errors"].append("unconfirmed_d3_cleanup")
            if not record["turns"] and not record["errors"]:
                record["errors"].append("empty_task")
            try:
                record["action_reward"] = (tau2_action_reward(task.get("raw_task") or {}, record["turns"])
                                           if not record["errors"] else None)
            except Exception as exc:
                record["errors"].append(f"action_reward: {type(exc).__name__}: {exc}")
            record["completed_at"] = datetime.now().isoformat()
            record["stage"] = "completed"
            write_json(partial, record)
            complete_record(path, record)
            report = analyze(args.output, manifest)
            print(json.dumps({"wall_s": record["wall_s"], "turns": len(record["turns"]),
                              "adopted_pairs": record["prefetch"].get("adopted", 0),
                              "turn_limit": record["turn_limit_reached"], "errors": record["errors"]}), flush=True)
            if dirty:
                raise RuntimeError("Server/cleanup integrity failed; outcome was sealed before stopping")
            require_matching_trajectories(report, manifest)
        write_json(args.output / "server_sessions" / (session_id + "_end.json"), await http.status(session))


def require_matching_trajectories(report, manifest):
    if not manifest.get("require_identical_trajectories"):
        return
    bad = [r for r in report["comparisons"] if (
        r.get("same_initial_prompt") is False
        or r.get("shared_completed_tokens_equal") is False
        or r.get("shared_completed_actions_equal") is False
        or (r["valid"] and (not r["same_actor_tokens"] or not r["same_actions_results_state"]
            or r["same_visible_answer"] is False or not r["same_turn_limit"])))]
    if bad:
        raise RuntimeError("Trajectory divergence retained; stop before further runs: "
                           + ", ".join(f'{r["group_id"]}/{r["config"]}' for r in bad))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--scope", choices=["all", "train", "validation", "test"], default="all")
    p.add_argument("--task-ids", help="Optional explicit IDs restricted to the chosen split")
    p.add_argument("--configs", default="spork,raw")
    p.add_argument("--max-turns", type=int, default=8)
    p.add_argument("--repetitions", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--tool-floor", type=float, default=2.)
    p.add_argument("--presence-penalty", type=float, default=0.0)
    p.add_argument("--model", default="/mnt/data/zhangshuai/LLMs/Models/Qwen3-32B")
    p.add_argument("--model-family", choices=["qwen3", "qwen35"], default="qwen3")
    p.add_argument("--require-identical-trajectories", action="store_true")
    p.add_argument("--base-url", default="http://127.0.0.1:8017/v1")
    p.add_argument("--task-split", type=Path, default=ROOT / "prefix_data/datasets/controller_stage3_20261008_213013_wi87e7du/task_split.json")
    p.add_argument("--run", action="store_true")
    args = p.parse_args()
    if not -2 <= args.presence_penalty <= 2:
        p.error("presence_penalty must be between -2 and 2")
    if min(args.max_turns, args.repetitions, args.max_tokens) < 1 or args.tool_floor < 0:
        p.error("Positive budgets and a nonnegative tool floor are required")
    manifest, task_map = prepare(args)
    print(f'Prepared {len(manifest["tasks"])} tasks, {len(manifest["schedule"])} runs, up to {args.max_turns} turns each: {args.output}', flush=True)
    if not args.run:
        return
    if (args.output / "records").is_dir():
        require_matching_trajectories(analyze(args.output, manifest), manifest)
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
        asyncio.run(execute(args, manifest, task_map))


if __name__ == "__main__":
    main()
