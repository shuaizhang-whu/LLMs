#!/usr/bin/env python3
"""Prepare/run a bounded two-Actor serving preflight on saved train prompts.

This is a hardware/transport experiment, NOT a causal Agent rollout. The second
prompt is a recorded later-turn input. It cannot establish online speedup.
Default invocation only prepares inputs. The experiment owner passes --run.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spork_core.prefix_replay import PrefixHttpClient, StreamState


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def prepare(args):
    manifest_path = args.output / "manifest.json"
    sources = [Path(__file__), ROOT / "spork_core/prefix_replay.py"]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (manifest["code_sha256"] != hashes or manifest["max_tokens"] != args.max_tokens
                or manifest["repetitions"] != args.repetitions or manifest["base_url"] != args.base_url
                or manifest["model"] != args.model):
            raise ValueError("Prepared configuration/source changed; choose a new output directory")
        return manifest, json.loads((args.output / "inputs.json").read_text())
    groups = {}
    for line in (args.dataset / "training_rows.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["split"] == "train":
            groups.setdefault(row["group_id"], {})[row["metadata"]["turn"]] = row
    inputs = []
    for domain in ("airline", "retail"):
        eligible = sorted((g for g, rows in groups.items() if g.startswith(domain + ":") and {1, 2} <= set(rows)),
                          key=lambda g: int(g.split(":")[1]))
        if not eligible:
            raise ValueError(f"No pair of recorded train inputs for {domain}")
        group = eligible[0]
        pair = {"group_id": group, "prompts": []}
        for turn in (1, 2):
            row = groups[group][turn]
            pair["prompts"].append({"sample_id": row["id"], "turn": turn,
                "token_ids": row["input"]["prompt_token_ids"] + row["input"]["prefix_token_ids"]})
        inputs.append(pair)
    manifest = {"purpose": "two_actor_hardware_preflight", "created_at": datetime.now().isoformat(),
                "model": args.model, "base_url": args.base_url, "max_tokens": args.max_tokens,
                "repetitions": args.repetitions, "groups": [p["group_id"] for p in inputs],
                "test_tasks_used": False, "selection": "smallest train task ID with turns 1 and 2 in each domain",
                "code_sha256": hashes,
                "scope": "Saved prompts, no environment or tools. Measures concurrency cost and greedy token equivalence only.",
                "not_causal_agent_rollout": True}
    args.output.mkdir(parents=True, exist_ok=True)
    dump(manifest_path, manifest)
    dump(args.output / "inputs.json", inputs)
    for source in sources:
        target = args.output / "code" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    return manifest, inputs


async def measure(session, http, pair, mode, max_tokens):
    states = [StreamState(), StreamState()]
    starts = [None, None]
    tasks = []
    whole_start = time.perf_counter()

    async def request(index):
        starts[index] = time.perf_counter()
        await http.stream(session, pair["prompts"][index]["token_ids"], states[index],
                          max_tokens=max_tokens, seed=42, request_id="overlap-" + uuid.uuid4().hex)

    try:
        if mode == "serial":
            for i in (0, 1):
                tasks.append(asyncio.create_task(request(i)))
                await tasks[-1]
        else:
            tasks.append(asyncio.create_task(request(0)))
            while states[0].first_output_at is None and not tasks[0].done():
                states[0].changed.clear()
                await states[0].changed.wait()
            if tasks[0].done():
                await tasks[0]
            # Stagger the second prefill until the first request has populated
            # the shared prefix cache, as a successor launch would do.
            tasks.append(asyncio.create_task(request(1)))
            await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    wall = time.perf_counter() - whole_start
    responses = []
    for i, state in enumerate(states):
        if state.finish_reason not in {"stop", "length"} or state.first_output_at is None:
            raise RuntimeError("Incomplete or empty benchmark response")
        responses.append({"wall_s": state.finished_at - starts[i],
                          "first_token_s": state.first_output_at - starts[i],
                          "decode_s": state.finished_at - state.first_output_at,
                          "start_offset_s": starts[i] - whole_start,
                          "finish_offset_s": state.finished_at - whole_start,
                          "token_ids": state.token_ids, "finish_reason": state.finish_reason,
                          "usage": state.usage})
    overlap = max(0., min(s.finished_at for s in states) - max(starts))
    return {"mode": mode, "wall_s": wall, "overlap_s": overlap, "responses": responses}


async def run(args, manifest, inputs):
    import aiohttp
    if (args.output / "summary.json").exists() or list((args.output / "records").glob("*.json")):
        raise ValueError("Do not overwrite measured samples; choose a new output directory")
    http = PrefixHttpClient(args.base_url, args.model, 300)
    rows = []
    async with aiohttp.ClientSession(trust_env=False, connector=aiohttp.TCPConnector(force_close=True)) as session:
        before = await http.status(session)
        if before.get("active_requests") != 0 or before.get("pending_verifications") != 0:
            raise RuntimeError("D3 has active state; another experiment may be using this server")
        dump(args.output / "server_before.json", before)
        for pair in inputs:
            for repeat in range(args.repetitions):
                record = {"group_id": pair["group_id"], "repeat": repeat}
                order = ("serial", "parallel") if repeat % 2 == 0 else ("parallel", "serial")
                for mode in order:
                    for prompt in pair["prompts"]:
                        await http.complete(session, prompt["token_ids"], max_tokens=1, seed=42)
                    record[mode] = await measure(session, http, pair, mode, args.max_tokens)
                record["tokens_equal"] = [record["serial"]["responses"][i]["token_ids"]
                                           == record["parallel"]["responses"][i]["token_ids"] for i in (0, 1)]
                record["pair_speedup"] = record["serial"]["wall_s"] / record["parallel"]["wall_s"]
                record["request_slowdown"] = [record["parallel"]["responses"][i]["wall_s"] /
                                               record["serial"]["responses"][i]["wall_s"] for i in (0, 1)]
                rows.append(record)
                dump(args.output / "records" / f"{pair['group_id'].replace(':', '_')}_r{repeat}.json", record)
                print(json.dumps({k: record[k] for k in ("group_id", "repeat", "tokens_equal", "pair_speedup", "request_slowdown")}), flush=True)
        dump(args.output / "server_after.json", await http.status(session))
    dump(args.output / "summary.json", {"records": rows, "scope": manifest["scope"],
                                        "not_causal_agent_rollout": True})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataset", type=Path, default=ROOT / "prefix_data/datasets/controller_stage3_20261008_213013_wi87e7du")
    p.add_argument("--base-url", default="http://127.0.0.1:8017/v1")
    p.add_argument("--model", default="/mnt/data/zhangshuai/LLMs/Models/Qwen3-32B")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--repetitions", type=int, default=2)
    p.add_argument("--run", action="store_true")
    args = p.parse_args()
    if not 32 <= args.max_tokens <= 512 or not 1 <= args.repetitions <= 3:
        p.error("Bounded preflight: 32-512 tokens and 1-3 repetitions")
    manifest, inputs = prepare(args)
    if not args.run:
        print(f"Prepared {len(inputs)} prompt pairs; GPU requests have not been started: {args.output}")
        return
    with (ROOT / "results/controller_online_8017.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            asyncio.run(run(args, manifest, inputs))
        except Exception as error:
            dump(args.output / "failure.json", {"error": str(error), "traceback": traceback.format_exc()})
            raise


if __name__ == "__main__":
    main()
