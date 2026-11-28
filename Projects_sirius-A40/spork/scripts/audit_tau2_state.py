#!/usr/bin/env python3
"""CPU-only DB-state diagnostic for sealed fixed-user SPORK trajectories.

Replay the task's reference actions using the local tau2 environment API and
compare their final DB hashes with recorded hashes. This is not a full tau2
reward: no user simulation, communication judge or policy judge is invoked.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spork_core.eval_artifacts import digest, load_sealed
from spork_core.paths import TAU2_ROOT, add_runtime_paths


def state_hashes(env):
    return {"agent_db": env.get_db_hash(), "user_db": env.get_user_db_hash()}


def assess_record(record, reference):
    turns = record.get("turns", [])
    actual = turns[-1].get("environment_after") if turns else None
    # A failed read can be intentional (e.g. try a new zip, then the old zip).
    # Keep it visible without rejecting the otherwise well-defined target DB.
    reference_ambiguous = any(e.get("mutating", True) or not e.get("state_unchanged", False)
                              for e in reference["errors"])
    comparable = (actual is not None and "agent_db" in actual
                  and "user_db" in actual and not reference_ambiguous)
    final = turns[-1] if turns and not turns[-1].get("baseline_has_tool_call") else None
    return {
        "group_id": record["group_id"], "repeat": record["repeat"], "method": record["method"],
        "runtime_errors": record["errors"], "turn_capped": bool(record.get("turn_limit_reached")),
        "recorded_final_state": actual, "db_match": actual == reference["state"] if comparable else None,
        "db_comparison_unavailable": (None if comparable else
                                      "reference_action_error" if reference_ambiguous else "missing_final_state"),
        "action_reference_coverage": record.get("action_criteria_progress"),
        "last_reply_for_manual_review": (final.get("final_answer", final.get("baseline_text", ""))
                                         .split("</think>")[-1].strip() if final else None),
        "official_task_success": None,
    }


def audit(directory):
    add_runtime_paths()
    from tau2.data_model.tasks import Task
    from spork_core.benchmarks.tau2 import get_environment, load_tasks

    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if digest(manifest_path) != json.loads((directory / "manifest_sha256.json").read_text())["sha256"]:
        raise ValueError("Manifest hash mismatch")
    for name, expected in manifest["data_sha256"].items():
        if digest(TAU2_ROOT / name) != expected:
            raise ValueError(f"Task/environment data changed: {name}")
    for name, expected in manifest["code_sha256"].items():
        if digest(directory / "code" / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    paths = [p for p in sorted((directory / "records").glob("*.json"))
             if not p.name.endswith(".complete.json")]
    records = []
    for path in paths:
        record = load_sealed(path)
        if record is None:
            raise ValueError(f"Unsealed record: {path}")
        records.append(record)
    tasks = {f"{domain}:{task['id']}": task for domain in ("airline", "retail")
             for task in load_tasks(domain)}
    references = {}
    for group in sorted({r["group_id"] for r in records}):
        task = Task(**tasks[group])
        env = get_environment(group.split(":")[0])
        initial = task.initial_state
        env.set_state(initialization_data=initial.initialization_data if initial else None,
                      initialization_actions=initial.initialization_actions if initial else None,
                      message_history=(initial.message_history or []) if initial else [])
        initial_hashes = state_hashes(env)
        errors = []
        criteria = task.evaluation_criteria
        # Same reference-state construction as EnvironmentEvaluator. Retain
        # exceptions, distinguishing failed reads from ambiguous failed writes.
        for action in criteria.actions or []:
            before_action = state_hashes(env)
            try:
                env.make_tool_call(action.name, requestor=action.requestor, **action.arguments)
            except Exception as error:
                errors.append({"action_id": action.action_id, "name": action.name,
                               "error": f"{type(error).__name__}: {error}",
                               "mutating": env._is_mutating_tool(action.name),
                               "state_unchanged": state_hashes(env) == before_action})
        state = state_hashes(env)
        references[group] = {"state": state, "initial_state": initial_hashes, "errors": errors,
                             "reference_changes_db": state != initial_hashes,
                             "reward_basis": [x.value for x in criteria.reward_basis],
                             "communicate_info": criteria.communicate_info,
                             "nl_assertions": criteria.nl_assertions}
    rows = [assess_record(r, references[r["group_id"]]) for r in records]
    totals = {}
    for method in sorted({r["method"] for r in rows}):
        selected = [r for r in rows if r["method"] == method]
        totals[method] = {
            "records": len(selected), "db_comparable": sum(r["db_match"] is not None for r in selected),
            "db_matches": sum(r["db_match"] is True for r in selected),
            "runtime_failures": sum(bool(r["runtime_errors"]) for r in selected),
            "turn_capped": sum(r["turn_capped"] for r in selected),
            "uncapped_error_free_db_matches": sum(r["db_match"] is True and not r["runtime_errors"]
                                                  and not r["turn_capped"] for r in selected),
            "action_all_matched": sum((r["action_reference_coverage"] or {}).get("reward") == 1
                                      for r in selected),
        }
    return {
        "source": str(directory.resolve()), "manifest_sha256": digest(manifest_path),
        "record_sha256": {p.name: digest(p) for p in paths},
        "audit_source_sha256": digest(Path(__file__)),
        "current_tau2_source_sha256": {str(p.relative_to(TAU2_ROOT)): digest(p)
                                      for p in sorted((TAU2_ROOT / "src/tau2").rglob("*.py"))},
        "reference_reward_basis_counts": dict(Counter(
            "+".join(r["reward_basis"]) for r in references.values())),
        "totals": totals, "references": references, "records": rows,
        "scope": "DB hash comparison on existing fixed-user records; no new model requests or latency measurements.",
        "limitations": [
            "ACTION is a diagnostic, not these tasks' configured reward basis.",
            "An unchanged reference DB may match without performing required lookups or giving a useful reply.",
            "Matching DB state does not prove user confirmation, policy compliance, communication correctness or official success.",
            "Communication and NL assertions have not been evaluated. Fixed-user runs do not contain real user responses.",
            "Failures and turn-capped runs remain separate; absent final state is unavailable, not a match.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new path to preserve previous evidence")
    report = audit(args.results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "totals": report["totals"],
                      "reference_errors": {g: r["errors"] for g, r in report["references"].items() if r["errors"]}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
