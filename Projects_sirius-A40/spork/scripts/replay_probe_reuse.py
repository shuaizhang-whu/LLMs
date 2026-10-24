#!/usr/bin/env python3
"""Short CPU trace experiment. Counts decode steps; does NOT measure GPU speedup.

Uses only train tasks, the first two recorded live SPORK turns, and their saved
prompt IDs. No test/validation trace is used to tune a retrieval rule. Both arms
retain the original prompt-lookup ngram proposer (vLLM 0.29, min=2, max=20).
"""
from __future__ import annotations

import argparse
from array import array
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spork_core.draft_reuse import ProbeDraft, REUSE_MODES
from spork_core.paths import resolve_artifact_path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_json(path, marker):
    values = json.loads(marker.read_text())["sha256"]
    expected = values if isinstance(values, str) else values[str(path.relative_to(marker.parent))]
    if sha(path) != expected:
        raise ValueError(f"Seal mismatch: {path}")
    return json.loads(path.read_text())


def ngram_proposal(tokens, limit, min_n=2, max_n=20):
    """CPU reference of vLLM 0.29: longest suffix, earliest prior occurrence.

    A packed uint32 search avoids Python loops over the full prompt. Check byte
    alignment so a match cannot begin in the middle of a token. Prior matches
    may overlap the suffix, as in vLLM's implementation.
    """
    packed = array("I", tokens).tobytes()
    width = array("I").itemsize
    for n in range(min(max_n, len(tokens) - 1), min_n - 1, -1):
        pattern = packed[-n * width:]
        pos = packed.find(pattern, 0, len(packed) - width)
        while pos >= 0:
            if pos % width == 0:
                start = pos // width + n
                return tokens[start:start + limit]
            pos = packed.find(pattern, pos + 1, len(packed) - width)
    return []


def replay(prompt, target, draft, mode, block=20):
    registration_start = time.perf_counter_ns()
    cursor = ProbeDraft(draft, mode=mode) if mode != "ngram" else None
    registration_cpu_us = (time.perf_counter_ns() - registration_start) / 1000
    emitted, steps, accepted, proposed = [], 0, 0, 0
    by_source = Counter()
    cpu_ns = []
    first_probe_acceptance = None
    while len(emitted) < len(target):
        fallback = ngram_proposal(prompt + emitted, block)
        start = time.perf_counter_ns()
        candidate = cursor.propose(emitted, block) if cursor else None
        cpu_ns.append(time.perf_counter_ns() - start)
        tokens = candidate.tokens if candidate else fallback
        source = candidate.source if candidate else "ngram"
        by_source[source] += bool(tokens)
        remaining = target[len(emitted):]
        keep = 0
        for a, b in zip(tokens, remaining):
            if a != b:
                break
            keep += 1
        if candidate and first_probe_acceptance is None:
            first_probe_acceptance = keep
        accepted += keep
        proposed += len(tokens)
        # The verifier emits a correction/bonus token after its accepted prefix.
        emitted.extend(remaining[:min(keep + 1, len(remaining))])
        steps += 1
    assert emitted == target
    return dict(decode_steps=steps, accepted_draft_tokens=accepted,
                proposed_tokens=proposed, sources=dict(by_source),
                first_probe_acceptance=first_probe_acceptance,
                registration_cpu_us=registration_cpu_us,
                proposer_cpu_us_median=statistics.median(cpu_ns) / 1000,
                trace_reconstruction_exact=True)


def load_cases(run, dataset, tokenizer):
    cases, excluded = [], []
    for file in sorted((run / "records/repeat_000").glob("*/timing_spork.json")):
        record = checked_json(file, file.with_suffix(".complete.json"))
        if record.get("split") != "train":
            continue
        if record.get("errors") or not record.get("turns"):
            excluded.append(dict(task=record["group_id"], reason="failed_original_run"))
            continue
        for turn in record["turns"][:2]:
            identity = dict(task=record["group_id"], turn=turn["turn"])
            snapshot = dataset / "tasks" / file.parent.name / f"turn_{turn['turn']:03d}/snapshot"
            if not (snapshot / "input.json").exists():
                excluded.append({**identity, "reason": "no_saved_prompt"})
                continue
            inp = checked_json(snapshot / "input.json", snapshot / "completed.json")
            if inp["split"] != "train":
                raise ValueError("Input split disagrees with live record")
            ids = turn["main_token_ids"]
            if ids[:len(inp["prefix_token_ids"])] != inp["prefix_token_ids"]:
                raise ValueError(f"Saved prefix differs: {identity}")
            probes = [p for p in turn["probes"] if p.get("d3_set", {}).get("status") == "ok"]
            if 151657 not in ids or not probes:
                excluded.append({**identity, "reason": "no_tool_boundary_or_registered_probe"})
                continue
            probe = probes[-1]
            injected = turn["d3_status"].get("last_inject_info") or {}
            if (injected.get("engine_request_id") != probe["d3_set"]["engine_request_id"]
                    or injected.get("body_tokens_already_generated") != 0):
                excluded.append({**identity, "reason": "no_confirmed_probe_at_body_start"})
                continue
            start = ids.index(151657) + 1
            target = ids[start:]
            body = "\n" + json.dumps({"name": probe["probe_tool_name"],
                                       "arguments": probe["probe_tool_args"]}, ensure_ascii=False)
            draft = tokenizer.encode(body, add_special_tokens=False).ids
            if len(draft) != probe["d3_set"]["n_draft_tokens"]:
                raise ValueError(f"Draft tokenizer mismatch: {identity}")
            text = tokenizer.decode(target, skip_special_tokens=False)
            if tokenizer.encode(text, add_special_tokens=False).ids != target:
                raise ValueError(f"Recorded body does not roundtrip: {identity}")
            verified = turn["d3_status"]["last_verification"]
            prompt = inp["prompt_token_ids"] + ids[:start]
            cases.append({**identity, "prompt": prompt, "target": target, "draft": draft,
                          "recorded_first_acceptance": verified["accepted_draft_tokens"],
                          "original_actor_tokens": len(ids),
                          "source": str(file.relative_to(ROOT)), "source_sha256": sha(file),
                          "input_sha256": sha(snapshot / "input.json")})
    return cases, excluded


def load_collected_cases(dataset, tokenizer):
    """Independent trace source: first now-repeat of every valid train prefix."""
    cases, excluded = [], []
    rows = [json.loads(line) for line in (dataset / "training_rows.jsonl").read_text().splitlines()]
    for row in rows:
        if row["split"] != "train" or not row["label_valid"]:
            continue
        identity = dict(task=row["group_id"], turn=row["metadata"]["turn"])
        base = dataset / "tasks" / row["group_id"].replace(":", "_") / f"turn_{identity['turn']:03d}"
        inp = checked_json(base / "snapshot/input.json", base / "snapshot/completed.json")
        measurement = base / "measurements/repeat_000/now"
        index = checked_json(measurement / "result_index.json", measurement / "completed.json")
        file = measurement / index["result"]
        record = checked_json(file, measurement / "completed.json")
        ids = inp["prefix_token_ids"] + record["continuation_token_ids"]
        probes = [p for p in record["probes"] if p.get("d3_set", {}).get("status") == "ok"]
        if not record["completion_valid"] or 151657 not in ids or not probes:
            excluded.append({**identity, "reason": "no_valid_tool_with_probe"})
            continue
        probe = probes[-1]
        status = record["d3_status_after"]
        injected = status.get("last_inject_info") or {}
        if (injected.get("engine_request_id") != probe["d3_set"]["engine_request_id"]
                or injected.get("body_tokens_already_generated") != 0):
            excluded.append({**identity, "reason": "no_confirmed_probe_at_body_start"})
            continue
        draft = tokenizer.encode("\n" + json.dumps(probe["call"], ensure_ascii=False), add_special_tokens=False).ids
        start = ids.index(151657) + 1
        target = ids[start:]
        if len(draft) != probe["d3_set"]["n_tokens"]:
            raise ValueError(f"Collected draft tokenizer mismatch: {identity}")
        if tokenizer.encode(tokenizer.decode(target, skip_special_tokens=False), add_special_tokens=False).ids != target:
            raise ValueError(f"Collected body tokenizer mismatch: {identity}")
        cases.append({**identity, "prompt": inp["prompt_token_ids"] + ids[:start],
                      "target": target, "draft": draft,
                      "recorded_first_acceptance": status["last_verification"]["accepted_draft_tokens"],
                      "original_actor_tokens": len(ids),
                      "source": str(file.relative_to(ROOT)), "source_sha256": sha(file),
                      "input_sha256": sha(base / "snapshot/input.json")})
    return cases, excluded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="results/controller_full_20261017_021939_Wid7sh")
    parser.add_argument("--dataset", default="prefix_data/datasets/controller_stage3_20261008_213013_wi87e7du")
    parser.add_argument("--tokenizer", type=Path, required=True,
                        help="Tokenizer JSON only; weights are never loaded")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-source", choices=["live", "collected"], default="live")
    args = parser.parse_args()
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    run, dataset = resolve_artifact_path(args.run), resolve_artifact_path(args.dataset)
    cases, excluded = (load_cases(run, dataset, tokenizer) if args.trace_source == "live"
                       else load_collected_cases(dataset, tokenizer))
    rows = []
    for case in cases:
        results = {mode: replay(case["prompt"], case["target"], case["draft"], mode)
                   for mode in ("ngram", *REUSE_MODES)}
        if results["single"]["first_probe_acceptance"] != case["recorded_first_acceptance"]:
            raise ValueError(f"Replay disagrees with recorded D3 acceptance: {case['task']}/{case['turn']}")
        rows.append({k: v for k, v in case.items() if k not in ("prompt", "target", "draft")} |
                    {"tool_body_tokens": len(case["target"]), "probe_tokens": len(case["draft"]),
                     "results": results})
    summary = {}
    for mode in ("ngram", *REUSE_MODES):
        summary[mode] = {key: sum(r["results"][mode][key] for r in rows)
                         for key in ("decode_steps", "accepted_draft_tokens", "proposed_tokens")}
        summary[mode]["wins_vs_single"] = sum(r["results"][mode]["decode_steps"] < r["results"]["single"]["decode_steps"] for r in rows)
        summary[mode]["losses_vs_single"] = sum(r["results"][mode]["decode_steps"] > r["results"]["single"]["decode_steps"] for r in rows)
    report = dict(kind="cpu_fixed_trace_replay_not_gpu_timing", cases=len(rows),
                  trace_source=args.trace_source,
                  selection=("all available train tasks; first two live SPORK turns" if args.trace_source == "live"
                             else "all valid train prefixes; first now repeat only") + "; confirmed D3 registration at body start",
                  limitations=["Recorded greedy continuation is the verifier oracle; output equality is by construction.",
                               "Decode-step counts omit variable GPU verification cost, contention and scheduling.",
                               "Only tool-call bodies are replayed; no end-to-end speedup or task-quality claim.",
                               "Tokenizer is validated against recorded body IDs, draft lengths and first-block acceptance; no model weights loaded."],
                  run_manifest_sha256=sha(run / "manifest.json"),
                  tokenizer_sha256=sha(args.tokenizer),
                  code_sha256={str(p.relative_to(ROOT)): sha(p) for p in
                               (Path(__file__), ROOT / "spork_core/draft_reuse.py")},
                  summary=summary, rows=rows, excluded=excluded)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({"cases": len(rows), "excluded": len(excluded), "summary": summary,
                      "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
