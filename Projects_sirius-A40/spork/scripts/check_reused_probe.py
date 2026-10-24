#!/usr/bin/env python3
"""Measure existing-candidate D3 reuse on training prefixes, without tool calls."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
from pathlib import Path
import sys
import json
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.check_candidate_verification import dump, select_rows
from spork_core.candidate_verification import decision_prefix, decision_probe
from spork_core.prefix_replay import PrefixHttpClient, counter_delta, same_call
from spork_core.qwen import QwenToolCallAdapter
from spork_core.paths import QWEN3_32B_MODEL


async def run(args):
    import aiohttp
    from transformers import AutoTokenizer
    args.output.mkdir(parents=True, exist_ok=False)
    rows = select_rows(args.dataset, args.per_domain)
    manifest = {"purpose": "reused_probe_preflight", "model": args.model,
                "base_url": args.base_url, "samples": [r["id"] for r in rows],
                "repetitions": args.repetitions, "test_tasks_used": False,
                "selection": "smallest numeric train IDs, first turn, each domain",
                "candidate_source": "live warmup at same visible prefix; not a future label",
                "code_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (Path(__file__), ROOT / "spork_core/candidate_verification.py")}}
    dump(args.output / "manifest.json", manifest)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    http = PrefixHttpClient(args.base_url, args.model, 180)
    results = []
    async with aiohttp.ClientSession(trust_env=False) as session:
        for row in rows:
            inp = row["input"]
            adapter = QwenToolCallAdapter(tok, inp["tools"])
            prefix = decision_prefix(tok, inp["prompt_token_ids"], inp["prefix_token_ids"], inp["prefix_text"])
            seed_probe = await decision_probe(session, http, tok, prefix)
            candidate = adapter.parse_main_tool_call(seed_probe["call_text"] or "")
            if candidate is None:
                raise RuntimeError(f"Warmup had no tool call: {row['id']}")
            record = {"sample_id": row["id"], "candidate": candidate, "seed_probe": seed_probe, "pairs": []}
            for rep in range(args.repetitions):
                pair = {"repeat": rep}
                for mode in (("plain", "d3") if rep % 2 == 0 else ("d3", "plain")):
                    await http.clear(session)
                    before = await http.status(session)
                    probe = await decision_probe(session, http, tok, prefix,
                                                 candidate=candidate, use_d3=mode == "d3")
                    probe["call"] = adapter.parse_main_tool_call(probe["call_text"] or "")
                    await http.clear(session)
                    after = await http.status(session)
                    probe["d3_counters"] = counter_delta(before, after)
                    pair[mode] = probe
                    dump(args.output / "in_progress.json", {"sample_id": row["id"], "pair": pair})
                pair["same_action"] = same_call(pair["plain"]["call"], pair["d3"]["call"])
                pair["same_tokens"] = (pair["plain"].get("continuation_token_ids")
                                       == pair["d3"].get("continuation_token_ids"))
                pair["saved_s"] = pair["plain"]["wall_s"] - pair["d3"]["wall_s"]
                record["pairs"].append(pair)
                dump(args.output / (row["metadata"]["raw_sample"] + ".json"), record)
                print(json.dumps({"sample": row["group_id"], "repeat": rep,
                                  "plain_s": pair["plain"]["wall_s"], "d3_s": pair["d3"]["wall_s"],
                                  "same_action": pair["same_action"], "same_tokens": pair["same_tokens"],
                                  "d3_counters": pair["d3"]["d3_counters"]}), flush=True)
            results.append(record)
        dump(args.output / "summary.json", {"results": results, "completed_samples": len(results),
                                            "scope": "probe cost and exactness only; no online Agent speed claim"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "prefix_data/datasets/controller_stage3_20261008_213013_wi87e7du")
    parser.add_argument("--model", default=str(QWEN3_32B_MODEL))
    parser.add_argument("--base-url", default="http://127.0.0.1:8017/v1")
    parser.add_argument("--per-domain", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.per_domain <= 4 or not 1 <= args.repetitions <= 3:
        parser.error("Bounded preflight permits 1-4 prefixes/domain and 1-3 repetitions")
    try:
        asyncio.run(run(args))
    except Exception as e:
        if args.output.is_dir():
            dump(args.output / "failure.json", {"error": str(e), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
