"""Small immutable manifests and completion seals for long local evaluations."""
import hashlib
import json
from pathlib import Path
import uuid


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def freeze_manifest(output, proposed, *, resume=False):
    path = Path(output) / "manifest.json"
    seal = Path(output) / "manifest_sha256.json"
    if resume:
        previous = json.loads(path.read_text())
        if json.loads(seal.read_text())["sha256"] != digest(path):
            raise ValueError("Frozen manifest was modified")
        keys = (set(previous) | set(proposed)) - {"created_at"}
        changed = sorted(k for k in keys if previous.get(k) != proposed.get(k))
        if changed:
            raise ValueError(f"Resume would change the experiment: {changed}")
        return previous
    if path.exists() or seal.exists():
        raise FileExistsError("Refusing to overwrite an existing manifest")
    dump(path, proposed)
    dump(seal, {"sha256": digest(path)})
    return proposed


def load_sealed(path):
    path = Path(path)
    seal = path.with_suffix(".complete.json")
    if not seal.exists():
        return None
    if not path.is_file() or json.loads(seal.read_text())["sha256"] != digest(path):
        raise ValueError(f"Completed result was modified: {path}")
    return json.loads(path.read_text())


def seal_record(path, record):
    path = Path(path)
    if path.exists() or path.with_suffix(".complete.json").exists():
        raise FileExistsError(f"Refusing to replace a result: {path}")
    dump(path, record)
    dump(path.with_suffix(".complete.json"), {"sha256": digest(path)})


def archive_unsealed(path):
    path = Path(path)
    if path.with_suffix(".complete.json").exists():
        raise ValueError(f"Cannot archive a sealed result: {path}")
    if path.exists():
        backup = path.parent / "attempts" / f"{path.stem}_{uuid.uuid4().hex}_unsealed.json"
        backup.parent.mkdir(parents=True, exist_ok=True)
        path.replace(backup)
        return backup
    return None
