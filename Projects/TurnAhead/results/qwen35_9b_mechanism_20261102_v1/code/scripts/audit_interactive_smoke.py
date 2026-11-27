#!/usr/bin/env python3
"""Read-only audit of sealed interactive runs; never re-score or rerun them."""
import argparse
import hashlib
import json
from pathlib import Path


def audit(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    findings, rows, pending = [], [], []
    for relative, expected in manifest["code_sha256"].items():
        path = directory / "code" / relative
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            findings.append({"source": relative, "error": "code_snapshot_mismatch"})
    for item in manifest["schedule"]:
        name = f'{item["domain"]}_{item["task_id"]}_r{item["repeat"]}_{item["config"]}.json'
        path = directory / "records" / name
        seal = path.with_name(name + ".seal")
        if not seal.exists():
            pending.append(name)
            continue
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != json.loads(seal.read_text())["sha256"]:
            findings.append({"record": name, "error": "record_seal_mismatch"})
            continue
        record = json.loads(path.read_text())
        if any(record.get(k) != item.get(k) for k in ("group_id", "split", "repeat", "config")):
            findings.append({"record": name, "error": "record_identity_mismatch"})
        row = {k: record.get(k) for k in ("group_id", "config", "repeat", "wall_s", "actor_wall_s",
                                           "user_wall_s", "termination_reason", "errors", "d3_counters")}
        reward = record.get("reward_info") or {}
        actions = reward.get("action_checks") or []
        row.update(native_reward=reward.get("reward"), reward_basis=reward.get("reward_basis"),
                   matched_required_actions=sum(bool(a["action_match"]) for a in actions),
                   checked_required_actions=len(actions),
                   exits=sum(t.get("exit", {}).get("fired", False) for t in record["turns"]),
                   refreshes=sum(t.get("exit", {}).get("refresh_count", 0) for t in record["turns"]))
        trajectory = directory / "trajectories" / name
        if trajectory.exists() and not record["errors"]:
            simulation = json.loads(trajectory.read_text())
            messages = simulation["messages"]
            assistant = [m for m in messages if m["role"] == "assistant"]
            users = [m["content"] for m in messages if m["role"] == "user"]
            tools = [m for m in messages if m["role"] == "tool"]
            row["trajectory_sha256"] = hashlib.sha256(trajectory.read_bytes()).hexdigest()
            row["user_messages"] = len(users)
            row["tool_errors"] = sum(bool(m.get("error")) for m in tools)
            if len(assistant) != len(record["turns"]) + 1:
                findings.append({"record": name, "error": "actor_message_count"})
            for message, turn in zip(assistant[1:], record["turns"]):
                if turn["baseline_has_tool_call"]:
                    calls = message.get("tool_calls") or []
                    if (len(calls) != 1 or calls[0]["name"] != turn["baseline_tool_name"]
                            or calls[0]["arguments"] != turn["baseline_tool_args"]):
                        findings.append({"record": name, "turn": turn["turn"], "error": "formal_call_mismatch"})
                    elif not any(m["id"] == calls[0]["id"] and hashlib.sha256(m["content"].encode()).hexdigest()
                                 == turn["tool_result_sha256"] for m in tools):
                        findings.append({"record": name, "turn": turn["turn"], "error": "tool_result_mismatch"})
                elif message.get("content") != turn["baseline_text"].split("</think>")[-1].strip():
                    findings.append({"record": name, "turn": turn["turn"], "error": "visible_answer_mismatch"})
            expected_users = [u["text"].split("</think>")[-1].strip() for u in record["user_calls"]]
            if users != expected_users:
                findings.append({"record": name, "error": "user_message_mismatch"})
        elif not record["errors"]:
            findings.append({"record": name, "error": "missing_trajectory"})
        rows.append(row)
    return {"directory": str(directory.resolve()), "expected": len(manifest["schedule"]),
            "recorded": len(rows), "pending": pending, "findings": findings, "rows": rows,
            "scope": manifest["scope"],
            "limits": ["Selected training diagnostics, not a held-out success estimate.",
                       "Native reward follows task reward_basis; action checks remain separate.",
                       "No semantic judgment of NL assertions is inferred from numeric reward.",
                       "Trajectory hashes are recorded by this audit; generated content is cross-checked against sealed records."]}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = audit(args.results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("recorded", "expected", "findings", "rows")}, ensure_ascii=False))
