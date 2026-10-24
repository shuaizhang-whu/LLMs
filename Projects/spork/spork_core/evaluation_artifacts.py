"""Frozen plans and sealed records for resumable research evaluations."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import uuid


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def visible_final_answers(record):
    """Visible replies only; raw thinking and tool-turn count are separate data."""
    replies = []
    for turn in record.get("turns", []):
        if not turn.get("baseline_has_tool_call") and turn.get("final_answer") is not None:
            text = turn.get("baseline_text", turn["final_answer"])
            replies.append(text.split("</think>")[-1].strip())
    return replies


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def freeze_plan(output, candidate, root, sources):
    """Prepare once or require an identical plan, inputs, and implementation."""
    output, root = Path(output), Path(root)
    candidate = dict(candidate)
    candidate["artifact_schema"] = 1
    expected = {str(Path(p).relative_to(root)): digest(p) for p in sources}
    if candidate["code_sha256"] != expected:
        raise ValueError("Source changed while constructing the plan")
    path = output / "manifest.json"
    if path.exists():
        existing = json.loads(path.read_text())
        stable = lambda m: {k: v for k, v in m.items() if k != "created_at"}
        if stable(existing) != stable(candidate):
            raise ValueError("Frozen configuration, inputs, or source changed; use a new output directory")
        for relative, checksum in expected.items():
            snapshot = output / "code" / relative
            if not snapshot.is_file() or digest(snapshot) != checksum:
                raise ValueError(f"Missing or modified code snapshot: {relative}")
        return existing
    if output.exists() and any(output.iterdir()):
        raise ValueError("Nonempty output directory has no frozen manifest")
    output.mkdir(parents=True, exist_ok=True)
    for source in sources:
        relative = Path(source).relative_to(root)
        content = Path(source).read_bytes()
        if hashlib.sha256(content).hexdigest() != expected[str(relative)]:
            raise ValueError(f"Source changed during snapshot: {relative}")
        target = output / "code" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    write_json(path, candidate)
    return candidate


def seal_path(path):
    path = Path(path)
    # A non-.json suffix prevents legacy '*.json' discovery from reading seals
    # as if they were completed task records.
    return path.with_name(path.name + ".seal")


def read_completed(path, item=None):
    """Return both successes and failures; never silently substitute a rerun."""
    path = Path(path)
    seal = seal_path(path)
    if not seal.exists():
        return None
    if not path.is_file():
        raise ValueError(f"Sealed record is missing: {path}")
    expected = json.loads(seal.read_text())
    if digest(path) != expected["sha256"]:
        raise ValueError(f"Sealed record changed: {path}")
    record = json.loads(path.read_text())
    if not isinstance(record.get("errors"), list) or not isinstance(record.get("turns"), list):
        raise ValueError(f"Malformed completed record: {path}")
    if not record["errors"] and not record["turns"]:
        raise ValueError(f"Empty run cannot be a successful record: {path}")
    if item is not None:
        for key in ("group_id", "split", "repeat", "config"):
            if record.get(key) != item.get(key):
                raise ValueError(f"Record identity changed ({key}): {path}")
    return record


def archive_unfinished(path):
    """Preserve an interrupted, unsealed record before a fresh attempt."""
    path = Path(path)
    if seal_path(path).exists():
        raise ValueError("A completed record must not be archived or rerun")
    if not path.exists():
        return None
    target = path.parent / "attempts" / (path.stem + "_" + uuid.uuid4().hex + "_unsealed.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    path.replace(target)
    return str(target)


def complete_record(path, record):
    path = Path(path)
    if path.exists() or seal_path(path).exists():
        raise ValueError(f"Refusing to overwrite an existing attempt: {path}")
    write_json(path, record)
    write_json(seal_path(path), {"sha256": digest(path), "sealed_at": datetime.now().isoformat()})


def record_path(output, item):
    return Path(output) / "records" / f"{item['domain']}_{item['task_id']}_r{item['repeat']}_{item['config']}.json"


def planned_records(output, manifest):
    records, unfinished = [], []
    for item in manifest["schedule"]:
        path = record_path(output, item)
        record = read_completed(path, item)
        if record is not None:
            records.append(record)
        elif path.exists():
            unfinished.append(str(path))
    return records, unfinished
