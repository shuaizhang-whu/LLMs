#!/usr/bin/env python3
"""Audit saved paired-prefix data and export explicit input/target JSONL.

No inference, environment unpickling, training or changes to raw samples.
The root index retains failed runs and points at the latest validated batch.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spork_core.prefix_replay import COUNTERS, paired_labels  # noqa: E402
from spork_core.qwen import parse_baseline_tool_call  # noqa: E402
from spork_core.turn_runner import _append_tool_messages  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def validate_run(run, tokenizer=None):
    run = Path(run).resolve()
    manifest = read(run / "manifest.json")
    report = read(run / "smoke_report.json")
    errors, rows = [], []
    totals = {k: 0 for k in COUNTERS}
    histories = {}

    def check(condition, detail):
        if not condition:
            errors.append(detail)

    for relative, expected in manifest["code_sha256"].items():
        path = run / "code" / relative
        check(path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == expected,
              f"code snapshot hash: {relative}")
    samples = sorted(run.glob("prefix_*/input.json"))
    check(len(samples) == manifest["args"]["prefixes"], "prefix count differs from request")
    check(not (run / "fatal_error.json").exists(), "run has fatal_error.json")
    expected_prefix_tokens = manifest["args"]["prefix_tokens"]
    for path in samples:
        sample_dir = path.parent
        sid = sample_dir.name
        inp = read(path)
        cap = read(sample_dir / "capture.json")
        outcomes = {p: read(sample_dir / f"{p}.json") for p in ("off", "now", "wait")}
        labels = read(sample_dir / "labels.json")
        group = f'{inp["domain"]}:{inp["task_id"]}'
        check(inp["sample_id"] == sid, f"{sid}: sample ID mismatch")
        check(inp["prefix_token_ids"] == cap["token_ids"], f"{sid}: prefix IDs differ from server capture")
        check(inp["prefix_text"] == cap["text"], f"{sid}: prefix text differs from capture")
        check(len(inp["prefix_token_ids"]) == expected_prefix_tokens, f"{sid}: wrong prefix length")
        check(inp["visible_generated_tokens"] == expected_prefix_tokens, f"{sid}: wrong visible count")
        if tokenizer is not None:
            rendered = tokenizer.apply_chat_template(
                inp["messages"], tools=inp["tools"], tokenize=False,
                add_generation_prompt=True, enable_thinking=True)
            check(rendered == inp["prompt_text"], f"{sid}: prompt differs from saved history and tools")
            check(tokenizer.encode(rendered, add_special_tokens=False) == inp["prompt_token_ids"],
                  f"{sid}: saved prompt token IDs do not reproduce")
            check(tokenizer.decode(inp["prefix_token_ids"], skip_special_tokens=False) == inp["prefix_text"],
                  f"{sid}: prefix text and token IDs disagree")
        check(cap["finish_reason"] == "length", f"{sid}: capture already terminated")
        check("</think>" not in inp["prefix_text"] and "<tool_call>" not in inp["prefix_text"],
              f"{sid}: capture is past the intended reasoning decision point")
        snapshot_path = sample_dir / inp["environment_file"]
        check(hashlib.sha256(snapshot_path.read_bytes()).hexdigest() == inp["environment_sha256"],
              f"{sid}: environment snapshot checksum")
        check(sorted(read(sample_dir / "order.json")) == ["now", "off", "wait"],
              f"{sid}: branch order does not cover the three policies")
        if inp["turn"] > 1:
            previous = histories.get((group, inp["turn"] - 1))
            check(previous is not None and previous == inp["messages"],
                  f"{sid}: history does not follow the preceding off branch")
        for policy, out in outcomes.items():
            prefix = f"{sid}/{policy}"
            check(out["policy"] == policy, f"{prefix}: wrong policy")
            check(out["config"] == manifest["replay_config"], f"{prefix}: changed config")
            check(not out.get("error") and out.get("completion_valid"), f"{prefix}: execution incomplete")
            check(out.get("finish_reason") == "stop", f"{prefix}: truncated/failed main output")
            check(math.isfinite(out["wall_s"]) and out["wall_s"] >= out["main_finished_s"] >= 0,
                  f"{prefix}: invalid timing order")
            check(out["environment_before"] == inp["environment_hash"], f"{prefix}: different initial state")
            check(out["main_text"] == inp["prefix_text"] + out["continuation_text"],
                  f"{prefix}: continuation lost or duplicated the prefix")
            check(parse_baseline_tool_call(out["main_text"]) == out["formal_call"],
                  f"{prefix}: recorded action differs from main text")
            tail = out["main_text"].rsplit("</think>", 1)[-1]
            check("<tool_call>" not in tail or out["formal_call"] is not None,
                  f"{prefix}: malformed formal tool call")
            clear = out["clear_after"]
            check(not clear.get("active_requests", 0) and not clear.get("pending_verifications", 0),
                  f"{prefix}: residual D3 state")
            for k in totals:
                totals[k] += out["d3_counters"].get(k, 0)
            for p in out["probes"]:
                check(out["main_text"].startswith(p["visible_prefix_text"]),
                      f"{prefix}: prediction saw text outside its actual Actor continuation")
                check(p["total_visible_generated_tokens"] == expected_prefix_tokens + p["new_tokens_at_dispatch"],
                      f"{prefix}: visible token count is inconsistent")
            if policy == "off":
                check(not out["probes"] and not out["spec_used"], f"{prefix}: off branch speculated")
                check(all(out["d3_counters"].get(k, 0) == 0 for k in COUNTERS),
                      f"{prefix}: D3 activity in off branch")
            elif out["probes"]:
                first = out["probes"][0]
                if policy == "now":
                    check(first["visible_prefix_text"] == inp["prefix_text"]
                          and first["new_tokens_at_dispatch"] == 0, f"{prefix}: now saw future text")
                else:
                    check(first["new_tokens_at_dispatch"] >= manifest["replay_config"]["delay_tokens"],
                          f"{prefix}: wait launched before the token threshold")
        expected_labels = paired_labels(outcomes)
        for key, value in expected_labels.items():
            check(labels.get(key) == value, f"{sid}: label mismatch {key}")
        for p in ("now", "wait"):
            state_equal = outcomes[p]["environment_after"] == outcomes["off"]["environment_after"]
            check(labels[p + "_same_state_as_off"] == state_equal, f"{sid}: state equality flag {p}")
        # Store the off branch as reference, irrespective of which timing won.
        next_messages = copy.deepcopy(inp["messages"])
        if outcomes["off"]["formal_call"] is not None:
            _append_tool_messages(next_messages, outcomes["off"]["main_text"], outcomes["off"]["tool_result"])
            histories[(group, inp["turn"])] = next_messages
        outcome_flags = {k: v for k, v in labels.items() if "_same_" in k}
        rows.append({
            "id": run.name + "/" + sid, "group_id": group, "split": "pilot",
            "input": {
                "messages": inp["messages"], "tools": inp["tools"],
                "prefix_text": inp["prefix_text"],
                "prompt_token_ids": inp["prompt_token_ids"], "prefix_token_ids": inp["prefix_token_ids"],
                "visible_generated_tokens": inp["visible_generated_tokens"],
                "wait_tokens": manifest["replay_config"]["delay_tokens"],
            },
            "targets": {k: labels.get(k) for k in ("gain_now_s", "gain_wait_s")},
            "label_valid": labels["valid"],
            "quality_flags": {**outcome_flags, "task_success_evaluated": False},
            "metadata": {"domain": inp["domain"], "task_id": inp["task_id"], "turn": inp["turn"],
                         "model": manifest["args"]["model"], "raw_sample": sid,
                         "measurement": labels["measurement"], "repetitions": 1,
                         "tool_floor_s": manifest["args"]["tool_floor_s"]},
        })
    check(totals == report["d3_counter_totals"], "D3 aggregate differs from branch records")
    check(totals["verified_injections"] > 0, "no actual D3 verification exercised")
    check(totals["accounting_errors"] == 0 and totals["dropped_injections"] == 0,
          "D3 verification/accounting failure")
    check(report["pipeline_passed"], "collector pipeline gate did not pass")
    check(report["valid_label_pairs"] == len(rows), "not every prefix has a valid label pair")
    compact = [json.loads(line) for line in (run / "pairs.jsonl").read_text().splitlines()]
    check(len(compact) == len(rows), "pairs.jsonl count mismatch")
    for r in compact:
        check(r["labels"] == read(run / r["sample_id"] / "labels.json"), "pairs.jsonl label mismatch")
    check(not read(run / "status_after.json").get("active_requests", 0), "server state remains active")
    check(not read(run / "status_after.json").get("pending_verifications", 0), "verification remains pending")
    return rows, {
        "passed": not errors, "errors": errors, "prefixes": len(rows),
        "branch_runs": 3 * len(rows), "label_pairs": sum(row["label_valid"] for row in rows),
        "task_groups": len({row["group_id"] for row in rows}),
        "outcome_consistent_pairs": sum(all(v for k, v in row["quality_flags"].items()
                                           if "_same_" in k) for row in rows),
        "d3_counter_totals": totals,
        "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "tokenizer_checked": tokenizer is not None,
        "scope": "Storage integrity, past-only prefix construction, isolated initial states, trigger rules, "
                 "recorded actions, exact label recomputation and exercised D3 verification. "
                 "Does not certify task success or statistical reliability of one-shot latency differences.",
    }


def rebuild_index(root):
    root = Path(root).resolve()
    runs = []
    for path in sorted((root / "runs").iterdir()):
        if not path.is_dir():
            continue
        row = {"run_id": path.name, "path": str(path.relative_to(root)), "status": "unvalidated"}
        if (path / "fatal_error.json").exists():
            row.update(status="failed", error=read(path / "fatal_error.json")["error"])
        if (path / "validation_report.json").exists():
            report = read(path / "validation_report.json")
            row.update(status="validated" if report["passed"] else "validation_failed",
                       prefixes=report["prefixes"], label_pairs=report["label_pairs"],
                       task_groups=report["task_groups"],
                       outcome_consistent_pairs=report["outcome_consistent_pairs"])
            if report["passed"]:
                row["training_rows"] = str((path / "training_rows.jsonl").relative_to(root))
        runs.append(row)
    good = [row for row in runs if row["status"] == "validated"]
    write(root / "index.json", {"schema_version": 1, "runs": runs,
                                "latest_validated_run": good[-1]["run_id"] if good else None,
                                "validated_prefixes": sum(r["prefixes"] for r in good)})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path)
    ap.add_argument("--data-root", type=Path, default=ROOT / "prefix_data")
    args = ap.parse_args()
    passed = True
    if args.run is not None:
        run = args.run.resolve()
        if run.parent != (args.data_root.resolve() / "runs"):
            ap.error("--run must be an immediate child of --data-root/runs")
        from transformers import AutoTokenizer
        model = read(run / "manifest.json")["args"]["model"]
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        rows, report = validate_run(run, tokenizer=tokenizer)
        write(run / "validation_report.json", report)
        passed = report["passed"]
        if passed:
            temp = run / "training_rows.jsonl.tmp"
            temp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
            temp.replace(run / "training_rows.jsonl")
        print(json.dumps(report, ensure_ascii=False, indent=2))
    rebuild_index(args.data_root)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
