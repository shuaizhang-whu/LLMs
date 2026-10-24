#!/usr/bin/env python3
"""CPU token-opportunity replay, NOT an online speed/quality experiment.

Keeps recorded Actor tokens and SSE/registration times fixed. It deliberately
does not simulate vLLM's background ngram proposals, batching or GPU costs.
Consequently accepted tokens here are NOT incremental gains over real SPORK.
Only train/validation data are permitted; original test tasks stay untouched.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spork_core.draft_reuse import DraftBank, MODES


def simulate(target, prefix_len, events, registrations, mode, min_suffix):
    bank = DraftBank(mode, min_suffix=min_suffix)
    ends = [e["end_token"] for e in events]
    if ends != sorted(ends) or not ends or ends[-1] != len(target) - prefix_len:
        raise ValueError("Incomplete/non-monotone token event coverage")
    pos, registered, body_start = prefix_len, 0, None
    for index, token in enumerate(target[:prefix_len]):
        if token == 151657:
            body_start = index + 1
        elif token == 151658:
            body_start = None
    stats = Counter()
    while pos < len(target):
        visible_time = 0.0 if pos == prefix_len else events[bisect.bisect_left(ends, pos - prefix_len)]["received_s"]
        while registered < len(registrations) and registrations[registered][0] <= visible_time:
            bank.add(registrations[registered][1])
            registered += 1
        # Only visible tokens determine whether we are inside a tool body.
        if pos and target[pos - 1] == 151657:
            body_start = pos
        elif pos and target[pos - 1] == 151658:
            body_start = None
        proposal = bank.propose(target[body_start:pos], 20) if body_start is not None else None
        accepted = 0
        if proposal and proposal.tokens:
            stats["injections"] += 1
            stats["proposed_tokens"] += len(proposal.tokens)
            stats[proposal.kind + "_injections"] += 1
            # Future target tokens are used ONLY by this offline verifier.
            for draft, actual in zip(proposal.tokens, target[pos:]):
                if draft != actual:
                    break
                accepted += 1
            stats["accepted_tokens"] += accepted
            stats[proposal.kind + "_accepted_tokens"] += accepted
            stats["zero_accept_injections"] += int(accepted == 0)
        # Rescan accepted/correction tokens for a closing tool marker.
        advance = min(accepted + 1, len(target) - pos)
        for index in range(pos, pos + advance - 1):
            if target[index] == 151657:
                body_start = index + 1
            elif target[index] == 151658:
                body_start = None
        pos += advance
    return dict(stats)


def sealed_result(directory):
    marker = json.loads((directory / "completed.json").read_text())
    for relative, expected in marker["sha256"].items():
        if hashlib.sha256((directory / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Seal mismatch: {directory / relative}")
    index = json.loads((directory / "result_index.json").read_text())
    return json.loads((directory / index["result"]).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "prefix_data/datasets/controller_stage3_20261008_213013_wi87e7du")
    parser.add_argument("--tokenizer", type=Path, required=True, help="Qwen3 tokenizer.json (no model weights needed)")
    parser.add_argument("--output", type=Path, required=True, help="New JSON file; refuses to overwrite")
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--limit", type=int, default=0, help="0 = all valid prefixes in the selected split")
    parser.add_argument("--min-suffix", type=int, default=7)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; retain it and choose a new path")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    source = args.dataset / "training_rows.jsonl"
    rows = [json.loads(line) for line in source.open()]
    rows = [r for r in rows if r["split"] == args.split and r["label_valid"]]
    rows.sort(key=lambda r: r["id"])
    if args.limit:
        rows = rows[:args.limit]
    output, totals, skipped = [], defaultdict(Counter), Counter()
    started = time.perf_counter()
    for row in rows:
        meta = row["metadata"]
        folder = args.dataset / "tasks" / f'{meta["domain"]}_{meta["task_id"]}' / f'turn_{int(meta["turn"]):03d}'
        prefix = row["input"]["prefix_token_ids"]
        if tokenizer.decode(prefix, skip_special_tokens=False) != row["input"]["prefix_text"]:
            raise ValueError("Tokenizer does not reproduce the saved prefix")
        for policy in ("now", "wait"):
            result = sealed_result(folder / "measurements/repeat_000" / policy)
            if not result["completion_valid"] or result["finish_reason"] != "stop":
                skipped["incomplete"] += 1
                continue
            registrations = []
            for probe in result["probes"]:
                status = probe.get("d3_set", {})
                if status.get("status") != "ok":
                    continue
                call = probe["call"]
                body = "\n" + json.dumps({"name": call["name"], "arguments": call.get("arguments", {})}, ensure_ascii=False)
                tokens = tokenizer.encode(body, add_special_tokens=False).ids
                if len(tokens) != status["n_tokens"]:
                    raise ValueError("Draft encoding differs from the sealed registration")
                registrations.append((probe["d3_registration_end_s"], tokens))
            registrations.sort(key=lambda pair: pair[0])
            target = prefix + result["continuation_token_ids"]
            modes = {mode: simulate(target, len(prefix), result["token_events"], registrations,
                                    mode, args.min_suffix) for mode in MODES}
            record = {"id": row["id"], "group_id": row["group_id"], "split": args.split,
                      "policy": policy, "actor_tokens": len(target),
                      "has_formal_tool_call": result["formal_call"] is not None,
                      "registered_drafts": len(registrations), "modes": modes}
            output.append(record)
            for mode, counts in modes.items():
                totals[mode].update(counts)
    report = {"kind": "frozen_teacher_trace_token_opportunities_not_online_benchmark",
              "limitations": ["Recorded timing is fixed; no counterfactual contention or scheduling.",
                              "Background ngram drafts are omitted; counts are not incremental SPORK gains.",
                              "Greedy token equality replay does not test live rejection sampling or task quality."],
              "split": args.split, "prefixes": len(rows), "traces": len(output),
              "tasks": len({r["group_id"] for r in output}),
              "tool_traces": sum(r["has_formal_tool_call"] for r in output),
              "actor_tokens": sum(r["actor_tokens"] for r in output),
              "min_suffix": args.min_suffix, "elapsed_cpu_wall_s": time.perf_counter() - started,
              "dataset_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "tokenizer_sha256": hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
              "code_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (Path(__file__), ROOT / "spork_core/draft_reuse.py")},
              "totals": dict(totals), "skipped": dict(skipped), "records": output}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k not in ("records", "code_sha256")}, indent=2))


if __name__ == "__main__":
    main()
