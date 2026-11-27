"""A bounded simulator-only gate before expensive interactive validation."""
from __future__ import annotations

import copy
from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace

from .evaluation_artifacts import digest, write_json
from .interactive_successor import (
    InteractiveRuntime, TracedLocalUser, USER_PRESENCE_PENALTY, validate_user_completion)
from .prefix_replay import PrefixHttpClient
from tau2.orchestrator.orchestrator import DEFAULT_FIRST_AGENT_MESSAGE


def initial_user(runtime, tokenizer, task, seed):
    if task.initial_state and task.initial_state.message_history:
        raise ValueError("Opening-user diagnostic requires an empty message history")
    user = TracedLocalUser(runtime, tokenizer, str(task.user_scenario),
                           max_tokens=4096, thinking=True)
    user.set_seed(seed)
    return user, user.get_init_state(), copy.deepcopy(DEFAULT_FIRST_AGENT_MESSAGE)


def prepare_usercheck(args, tasks):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    stub = SimpleNamespace(client=SimpleNamespace(model=args.model))
    user, state, message = initial_user(stub, tokenizer, tasks["airline:2"], args.seed)
    state.messages.append(message)
    ids = user.prompt_token_ids(state)
    http = PrefixHttpClient(args.base_url, args.model, 900, presence_penalty=USER_PRESENCE_PENALTY)
    settings = {k:v for k,v in http.body(ids, max_tokens=4096, seed=args.seed,
        stop=["<|im_end|>"]).items() if k not in {"prompt", "request_id"}}
    result = {"task": "airline:2", "expected_runs": 2, "prompt_token_ids": ids,
        "generation_settings": settings, "thinking": True,
        "preflight_manifest_sha256": digest(args.output_root / "preflight/manifest.json"),
        "scope": "Opening user generation, closure and same-seed repeatability only; not full simulator semantic fidelity or Actor performance."}
    p = args.output_root / "usercheck/input.json"
    if p.exists() and json.loads(p.read_text()) != result:
        raise ValueError("Frozen simulator diagnostic changed; use a new run tag")
    if not p.exists():
        write_json(p, result)
    return result


def load_record(path):
    seal = path.with_suffix(".json.seal")
    if not path.exists() and not seal.exists():
        return None
    if not path.is_file() or not seal.is_file() or digest(path) != json.loads(seal.read_text())["sha256"]:
        raise ValueError("User diagnostic record is unsealed or modified; inspect it before proceeding")
    return json.loads(path.read_text())


def check_usercheck(args, spec):
    directory = args.output_root / "usercheck"
    records = [load_record(directory / f"records/repeat_{i}.json") for i in range(2)]
    if any(r is None for r in records):
        raise ValueError("Complete the two-run usercheck before interactive preflight")
    signatures = []
    for i, r in enumerate(records):
        if r["repeat"] != i or r["input_sha256"] != digest(directory / "input.json") or r["errors"]:
            raise ValueError(f"Retained usercheck failure or identity mismatch: repeat {i}")
        call = r["call"]
        if (call["prompt_token_ids"] != spec["prompt_token_ids"]
                or call["generation_settings"] != spec["generation_settings"] or not call.get("token_ids")):
            raise ValueError("User diagnostic did not use the actual frozen request")
        visible = validate_user_completion(call, thinking=True)
        if visible != r["visible_text"]:
            raise ValueError("User diagnostic visible text changed")
        if r["server_after"].get("active_requests") != 0 or r["server_after"].get("pending_verifications") != 0:
            raise ValueError("User diagnostic left active server work")
        signatures.append({k:call[k] for k in
            ("prompt_token_ids", "generation_settings", "token_ids", "text", "finish_reason")})
    if signatures[0] != signatures[1]:
        raise ValueError("Same-seed user diagnostic outputs diverged")
    report = {"passed": True, "recorded": 2, "input_sha256": digest(directory / "input.json"),
        "completion_tokens": [len(r["call"]["token_ids"]) for r in records],
        "visible_text": records[0]["visible_text"], "scope": spec["scope"],
        "actor_requests": 0, "tool_executions": 0}
    write_json(directory / "stage_check.json", report)
    return report


def run_usercheck(args, spec, tasks, *, idle, reset_cache, check_server):
    directory = args.output_root / "usercheck"
    previous = [load_record(directory / f"records/repeat_{i}.json") for i in range(2)]
    if any(r is not None and r["errors"] for r in previous):
        raise RuntimeError("A failed usercheck is sealed; inspect it instead of silently rerunning")
    if all(r is not None for r in previous):
        return check_usercheck(args, spec)
    check_server()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    runtime = InteractiveRuntime(args.base_url, args.model)
    try:
        for i, old in enumerate(previous):
            if old is not None:
                continue
            idle(runtime)
            reset = runtime.run(reset_cache(runtime.session, runtime.http))
            user, state, message = initial_user(runtime, tokenizer, tasks[spec["task"]], args.seed)
            row = {"repeat": i, "errors": [], "input_sha256": digest(directory / "input.json"),
                   "prefix_cache_reset": reset, "visible_text": None}
            print(f'[usercheck {i+1}/2] {spec["task"]}; user presence_penalty={USER_PRESENCE_PENALTY}', flush=True)
            try:
                reply, _ = user.generate_next_message(message, state)
                row["visible_text"] = reply.content
                if user.is_stop(reply):
                    raise RuntimeError("User stopped at the greeting before the task began")
            except Exception as exc:
                row["errors"].append(f"{type(exc).__name__}: {exc}")
            row["call"] = user.calls[0] if user.calls else None
            try:
                row["server_after"] = idle(runtime)
            except Exception as exc:
                row["errors"].append(f"server_integrity: {exc}")
            row["completed_at"] = datetime.now().astimezone().isoformat()
            path = directory / f"records/repeat_{i}.json"
            if path.exists() or path.with_suffix(".json.seal").exists():
                raise ValueError("Refusing to overwrite user diagnostic evidence")
            write_json(path, row)
            write_json(path.with_suffix(".json.seal"), {"sha256": digest(path)})
            print(json.dumps({"finish_reason": (row["call"] or {}).get("finish_reason"),
                "tokens": len((row["call"] or {}).get("token_ids", [])),
                "visible_text": row["visible_text"], "errors": row["errors"]}, ensure_ascii=False), flush=True)
            if row["errors"]:
                raise RuntimeError("User-only check failed; no Actor or expanded evaluation was started")
    finally:
        runtime.close()
    return check_usercheck(args, spec)
