#!/usr/bin/env python3
"""Diagnose two saved prefixes with solo, prediction-only and full-SPORK runs.

The user launches collection in tmux. --prepare-only / --analyze-only are CPU
operations. This script does not start a vLLM server or train a controller.
"""
from __future__ import annotations

import argparse
import asyncio
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
from scripts.repeat_prefix_pairs import digest, dump, log, read, snapshot_for  # noqa: E402
from spork_core.fork_diagnostic import MODES, check_branch, schedule_modes, summarize  # noqa: E402

DEFAULT_SOURCE = ROOT / "prefix_data/stability_runs/stability_20261007_184230_ymqr6_ch"


def source_files():
    return [Path(__file__), ROOT / "scripts/repeat_prefix_pairs.py", ROOT / "spork_core/fork_diagnostic.py",
            ROOT / "spork_core/prefix_replay.py", ROOT / "spork_core/qwen.py", ROOT / "spork_core/tool_calls.py",
            ROOT / "spork_core/executors.py", ROOT / "spork_core/vllm029_d3.py", ROOT / "launch_vllm_spork_http.py",
            ROOT / "scripts/serve_qwen32_prefix_collection.sh", ROOT / "scripts/run_prefix_fork_diagnostic.sh",
            ROOT / "scripts/serve_qwen32_prefix_invariant.sh", ROOT / "scripts/run_prefix_invariance_check.sh"]


def prepare(args):
    output = args.output.resolve()
    if (output / "manifest.json").exists():
        if not args.resume and not args.analyze_only:
            raise ValueError("Existing diagnostic: use --resume or a new directory")
        manifest = read(output / "manifest.json")
        if manifest["purpose"] != "two_prefix_generation_fork_diagnostic":
            raise ValueError("Not a generation-fork diagnostic directory")
        if not args.analyze_only:
            for name, checksum in manifest["code_sha256"].items():
                if digest(ROOT / name) != checksum:
                    raise ValueError(f"Code changed after preparation: {name}; prepare a new run")
            if args.repetitions != manifest["repetitions"]:
                raise ValueError("Cannot change repetitions on resume")
            if args.model_url and args.model_url.rstrip("/") != manifest["model_url"]:
                raise ValueError("Cannot change model URL on resume")
            if bool(getattr(args, "require_invariance", False)) != manifest.get("require_invariance", False):
                raise ValueError("Cannot change the invariance requirement on resume")
        return manifest
    if args.analyze_only:
        raise ValueError("No existing diagnostic to analyze")
    if output.exists() and any(output.iterdir()):
        raise ValueError("New output directory must be empty")
    source = args.source_run.resolve()
    previous = read(source / "manifest.json")
    ids = ["prefix_003", "prefix_009"]
    schedule = schedule_modes(ids, args.repetitions, args.order_seed)
    output.mkdir(parents=True, exist_ok=True)
    checksums = {}
    for sid in ids:
        snapshot, env_bytes = snapshot_for(source, previous, sid)
        destination = output / "inputs" / sid
        destination.mkdir(parents=True)
        # Copy byte-for-byte: do not recapture or generate new decision prefixes.
        shutil.copyfile(source / "inputs" / sid / "input.json", destination / "input.json")
        (destination / "environment.pkl").write_bytes(env_bytes)
        checksums[sid] = {"input_sha256": digest(destination / "input.json"),
                          "environment_sha256": digest(destination / "environment.pkl")}
    manifest = {
        "schema_version": 1, "purpose": "two_prefix_generation_fork_diagnostic",
        "require_invariance": bool(getattr(args, "require_invariance", False)),
        "created_at": datetime.now().isoformat(), "source_run": str(source),
        "sample_ids": ids, "repetitions": args.repetitions, "order_seed": args.order_seed,
        "schedule": schedule, "input_checksums": checksums, "modes": MODES,
        "model": previous["model"], "model_url": (args.model_url or previous["model_url"]).rstrip("/"),
        "timeout_s": previous["timeout_s"], "tool_floor_s": previous["tool_floor_s"],
        "replay_config": previous["replay_config"], "planned_branch_runs": 3 * len(schedule),
        "packages": {name: version(name) for name in ("vllm", "torch", "transformers", "aiohttp", "tau2")},
        "code_sha256": {str(p.relative_to(ROOT)): digest(p) for p in source_files()},
        "comparison": "First prediction launches immediately from the saved prefix in both prediction modes. "
                      "Same Actor request, seed=42, temperature=0, ngram-enabled server, discarded warmup and "
                      "fresh environment in every mode. The prediction-only mode stops at the same confidence "
                      "gate as full SPORK but never registers action drafts or starts speculative tools.",
        "measurement_scope": "Diagnostic client traces and replay timings, not training labels or online speedup.",
    }
    for p in source_files():
        destination = output / "code" / p.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, destination)
    dump(output / "manifest.json", manifest)
    return manifest


def completed_triplet(root):
    marker = root / "completed.json"
    if not marker.exists():
        return None
    saved = read(marker)
    attempt = saved["attempt"]
    if Path(attempt).name != attempt or not attempt.startswith("attempt_"):
        raise ValueError("Invalid attempt path")
    directory = root / attempt
    for name, checksum in saved["sha256"].items():
        if Path(name).name != name or digest(directory / name) != checksum:
            raise ValueError(f"Saved diagnostic measurement changed: {directory / name}")
    return {"repetition": saved["repetition"],
            "outcomes": {mode: read(directory / f"{mode}.json") for mode in MODES}}


def check_server_settings(status, manifest):
    if manifest.get("require_invariance"):
        cfg = status.get("server_config", {})
        if cfg.get("batch_invariant_requested") is not True:
            raise RuntimeError("Batch-invariant server required. Stop the old server and run "
                               "scripts/serve_qwen32_prefix_invariant.sh in tmux; no generation was started.")


async def collect(output, manifest, tokenizer):
    import aiohttp
    from spork_core.executors import Tau2ToolExecutor
    from spork_core.prefix_replay import PrefixHttpClient, ReplayConfig, run_prefix_branch
    from spork_core.qwen import QwenToolCallAdapter

    cfg = ReplayConfig(**manifest["replay_config"])
    client = PrefixHttpClient(manifest["model_url"], manifest["model"], manifest["timeout_s"])
    connector = aiohttp.TCPConnector(force_close=True)
    async with aiohttp.ClientSession(trust_env=False, connector=connector) as session:
        async with session.get(client.base_url + "/models", timeout=15) as response:
            response.raise_for_status()
            models = await response.json()
        if manifest["model"] not in [m["id"] for m in models.get("data", [])]:
            raise RuntimeError("Wrong model at the diagnostic endpoint")
        before = await client.status(session)
        if not before.get("has_drafter") or before.get("active_requests") or before.get("pending_verifications"):
            raise RuntimeError("Dedicated idle D3/ngram server required")
        dump(output / f"preflight_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json",
             {"models": models, "d3_status": before})
        check_server_settings(before, manifest)
        text = tokenizer.apply_chat_template([{"role": "user", "content": "Explain how to look up an order."}],
                                             tokenize=False, add_generation_prompt=True, enable_thinking=True)
        await client.complete(session, tokenizer.encode(text, add_special_tokens=False), max_tokens=128, seed=cfg.seed)
        done = 0
        for item in manifest["schedule"]:
            sid, rep = item["sample_id"], item["repetition"]
            directory = output / "measurements" / sid / f"repeat_{rep:03d}"
            if completed_triplet(directory) is not None:
                done += 1
                continue
            snapshot, environment_bytes = snapshot_for(output, manifest, sid)
            adapter = QwenToolCallAdapter(tokenizer, snapshot["tools"])
            rendered = adapter.render_main_prompt(snapshot["messages"], enable_thinking=True)
            if (rendered != snapshot["prompt_text"]
                    or tokenizer.encode(rendered, add_special_tokens=False) != snapshot["prompt_token_ids"]
                    or tokenizer.decode(snapshot["prefix_token_ids"], skip_special_tokens=False) != snapshot["prefix_text"]):
                raise ValueError("Saved prompt/prefix no longer reproduces with the local tokenizer")
            directory.mkdir(parents=True, exist_ok=True)
            attempt = directory / f"attempt_{len(list(directory.glob('attempt_*'))):03d}"
            attempt.mkdir()
            dump(attempt / "order.json", item)
            outcomes = {}
            for mode in item["order"]:
                log(f"{done + 1}/{len(manifest['schedule'])} {sid} repeat={rep + 1} mode={mode}")
                restored_at = time.perf_counter()
                environment = pickle.loads(environment_bytes)  # trusted local snapshot with checked checksum
                state = {"agent_db": environment.get_db_hash(), "user_db": environment.get_user_db_hash()}
                if state != snapshot["environment_hash"]:
                    raise RuntimeError("Restored state differs from the saved prefix")
                executor = Tau2ToolExecutor(environment, floor_s=manifest["tool_floor_s"])
                restore_s = time.perf_counter() - restored_at
                warm = await client.complete(session, snapshot["prompt_token_ids"] + snapshot["prefix_token_ids"],
                                             max_tokens=1, seed=cfg.seed)
                setting = MODES[mode]
                result = await run_prefix_branch(session, client, adapter, snapshot, executor,
                                                 setting["policy"], cfg, enable_d3=setting["enable_d3"],
                                                 enable_speculative_tools=setting["enable_speculative_tools"],
                                                 record_token_timing=True)
                result.update(mode=mode, environment_before=state,
                              environment_after={"agent_db": executor.env.get_db_hash(),
                                                 "user_db": executor.env.get_user_db_hash()},
                              restore_s=restore_s, discarded_warmup_s=warm["wall_s"])
                dump(attempt / f"{mode}.json", result)
                problems = check_branch(mode, result, snapshot, manifest["replay_config"])
                dump(attempt / f"{mode}_integrity.json", {"passed": not problems, "errors": problems})
                if problems:
                    raise RuntimeError(f"Diagnostic integrity failure at {attempt / (mode + '.json')}: {problems}")
                outcomes[mode] = result
                log(f"{sid} {mode}: tokens={len(result['continuation_token_ids'])}, wall_s={result['wall_s']:.3f}, "
                    f"action={(result['formal_call'] or {}).get('name', '(no tool)')}")
            if len({out["main_request_sha256"] for out in outcomes.values()}) != 1:
                raise RuntimeError("The three modes sent different Actor requests")
            names = ["order.json"] + [mode + suffix for mode in MODES for suffix in (".json", "_integrity.json")]
            dump(directory / "completed.json", {"attempt": attempt.name, "repetition": rep,
                                               "sha256": {name: digest(attempt / name) for name in names}})
            done += 1
            dump(output / "progress.json", {"completed_triples": done, "planned_triples": len(manifest["schedule"])})
        after = await client.status(session)
        dump(output / "status_after.json", after)
        check_server_settings(after, manifest)
        if after.get("active_requests") or after.get("pending_verifications"):
            raise RuntimeError("D3 state remains active after the diagnostic")


def analyze(output, manifest, tokenizer):
    samples, snapshots = {}, {}
    for sid in manifest["sample_ids"]:
        snapshots[sid], _ = snapshot_for(output, manifest, sid)
        samples[sid] = []
        for rep in range(manifest["repetitions"]):
            record = completed_triplet(output / "measurements" / sid / f"repeat_{rep:03d}")
            if record is not None:
                samples[sid].append(record)
    report = summarize(samples, snapshots, manifest, tokenizer)
    dump(output / "fork_diagnostic_report.json", report)
    brief = {"pipeline_passed": report["pipeline_passed"],
             "generation_invariance_passed": report["generation_invariance_passed"],
             "completed_branch_runs": report["completed_branch_runs"],
             "errors": report["errors"], "per_prefix": [
                 {k: p[k] for k in ("sample_id", "reference_tokens_stable", "observation", "token_differences_by_pair")}
                 for p in report["per_prefix"]], "report": str(output / "fork_diagnostic_report.json")}
    print(json.dumps(brief, indent=2, ensure_ascii=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--model-url")
    parser.add_argument("--repetitions", type=int, choices=(3, 6), default=3)
    parser.add_argument("--order-seed", type=int, default=43)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--require-invariance", action="store_true",
                        help="Require the batch-invariant server setting and identical outputs across modes")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare-only", action="store_true")
    modes.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    manifest = prepare(args)
    log(f"Diagnostic output: {args.output}; planned {manifest['planned_branch_runs']} branches")
    if args.prepare_only:
        return
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(manifest["model"], local_files_only=True)
    if not args.analyze_only:
        # Prevent accidental duplicate invocations against the same run directory.
        with (args.output / "run.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                asyncio.run(collect(args.output, manifest, tokenizer))
            except Exception as exc:
                dump(args.output / f"failure_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json",
                     {"error": f"{type(exc).__name__}: {exc}"})
                analyze(args.output, manifest, tokenizer)
                raise
    report = analyze(args.output, manifest, tokenizer)
    passed = report["pipeline_passed"] and (
        not manifest.get("require_invariance") or report["generation_invariance_passed"])
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
