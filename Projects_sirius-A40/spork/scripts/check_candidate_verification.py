#!/usr/bin/env python3
"""Bounded GPU preflight for candidate checking; no tools or benchmark writes.

Uses only preselected training prefixes and no future labels. This checks the
HTTP scoring implementation and its cost, not online Agent speed or quality.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import traceback
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spork_core.candidate_verification import decision_prefix, score_candidate
from spork_core.prefix_replay import PrefixHttpClient, prediction_prompt_ids, same_call
from spork_core.qwen import QwenToolCallAdapter
from spork_core.paths import QWEN3_32B_MODEL


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def select_rows(dataset, per_domain):
    rows = [json.loads(line) for line in (dataset / "training_rows.jsonl").read_text().splitlines()]
    selected = []
    for domain in ("airline", "retail"):
        pool = sorted((r for r in rows if r["split"] == "train"
                       and r["metadata"]["domain"] == domain and r["metadata"]["turn"] == 1),
                      key=lambda r: int(r["metadata"]["task_id"]))
        selected.extend(pool[:per_domain])
    return selected


async def run(args):
    import aiohttp
    from transformers import AutoTokenizer
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows = select_rows(args.dataset, args.per_domain)
    manifest = {"purpose": "candidate_verification_cost_and_alignment_preflight",
                "model": args.model, "base_url": args.base_url,
                "sample_ids": [r["id"] for r in rows], "repetitions": args.repetitions,
                "selection": "smallest numeric training task IDs, first turn, per domain",
                "test_tasks_used": False, "labels_used": False,
                "code_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (Path(__file__), ROOT / "spork_core/candidate_verification.py")}}
    dump(output / "manifest.json", manifest)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    client = PrefixHttpClient(args.base_url, args.model, 180)
    results = []
    async with aiohttp.ClientSession(trust_env=False) as session:
        for row in rows:
            inp = row["input"]
            adapter = QwenToolCallAdapter(tokenizer, inp["tools"])
            prefix = decision_prefix(tokenizer, inp["prompt_token_ids"],
                                     inp["prefix_token_ids"], inp["prefix_text"])
            forced_ids = prediction_prompt_ids(tokenizer, inp["prompt_token_ids"],
                                               inp["prefix_token_ids"], inp["prefix_text"])
            # Warm only past input; never use the recorded off/now/wait outcomes.
            await client.complete(session, prefix, max_tokens=1, seed=42)
            forced = await client.complete(session, forced_ids, max_tokens=160,
                                           seed=42, stop=adapter.probe_stop, logprobs=5)
            call = adapter.parse_probe_tool_call(forced["text"])
            if call is None or forced["finish_reason"] != "stop":
                raise RuntimeError(f"No complete candidate for {row['id']}")
            candidate_text = '<tool_call>\n{"name": "' + forced["text"]
            if not candidate_text.rstrip().endswith("</tool_call>"):
                candidate_text += "</tool_call>"
            candidate_ids = tokenizer.encode(candidate_text, add_special_tokens=False)
            record = {"sample_id": row["id"], "group_id": row["group_id"],
                      "candidate": call, "candidate_text": candidate_text,
                      "forced_probe_wall_s": forced["wall_s"], "pairs": []}
            dump(output / (row["metadata"]["raw_sample"] + ".json"), record)
            for repeat in range(args.repetitions):
                order = ("generate", "score") if repeat % 2 == 0 else ("score", "generate")
                pair = {"repeat": repeat, "order": order}
                for method in order:
                    if method == "score":
                        pair[method] = await score_candidate(session, client, prefix, candidate_ids)
                    else:
                        generated = await client.complete(session, prefix, max_tokens=160,
                                                          seed=42, stop=adapter.main_stop)
                        generated_call = adapter.parse_main_tool_call(generated["text"])
                        pair[method] = {"text": generated["text"], "token_ids": generated["token_ids"],
                                        "finish_reason": generated["finish_reason"],
                                        "call": generated_call, "same_call": same_call(call, generated_call),
                                        "wall_s": generated["wall_s"], "usage": generated["usage"]}
                    dump(output / "in_progress.json", {"sample_id": row["id"], "pair": pair})
                # A greedy score must imply the independently generated prefix.
                generated_ids = pair["generate"]["token_ids"]
                compared = min(len(generated_ids), len(candidate_ids))
                pair["score_agrees_with_generation"] = (
                    not pair["score"]["all_greedy"]
                    or (compared > 0 and pair["generate"]["finish_reason"] == "stop"
                        and pair["generate"]["same_call"]
                        and generated_ids[:compared] == candidate_ids[:compared]))
                record["pairs"].append(pair)
                dump(output / (row["metadata"]["raw_sample"] + ".json"), record)
                print(json.dumps({"sample": row["group_id"], "repeat": repeat,
                                  "score_s": pair["score"]["wall_s"],
                                  "generate_s": pair["generate"]["wall_s"],
                                  "greedy_fraction": pair["score"]["greedy_fraction"],
                                  "all_greedy": pair["score"]["all_greedy"],
                                  "same_call": pair["generate"]["same_call"],
                                  "alignment_ok": pair["score_agrees_with_generation"]}), flush=True)
            wrong = {**call, "name": "__unavailable_tool__"}
            wrong_ids = tokenizer.encode("<tool_call>\n" + json.dumps(wrong, ensure_ascii=False)
                                         + "\n</tool_call>", add_special_tokens=False)
            record["negative_control"] = await score_candidate(session, client, prefix, wrong_ids)
            record["negative_control_rejected"] = not record["negative_control"]["all_greedy"]
            dump(output / (row["metadata"]["raw_sample"] + ".json"), record)
            results.append(record)
        dump(output / "summary.json", {"completed_samples": len(results), "results": results,
                                       "scope": "cost/alignment preflight only; no Agent latency or success claim"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "prefix_data/datasets/controller_stage3_20261008_213013_wi87e7du")
    parser.add_argument("--model", default=str(QWEN3_32B_MODEL))
    parser.add_argument("--base-url", default="http://127.0.0.1:8017/v1")
    parser.add_argument("--per-domain", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.per_domain <= 4 or not 1 <= args.repetitions <= 3:
        parser.error("This preflight is bounded to 1-4 prefixes/domain and 1-3 repeats")
    try:
        asyncio.run(run(args))
    except Exception as error:
        if args.output.is_dir() and not (args.output / "failure.json").exists():
            dump(args.output / "failure.json", {"error": f"{type(error).__name__}: {error}",
                                                "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
