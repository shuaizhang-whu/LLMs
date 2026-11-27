#!/usr/bin/env python3
"""Replay two user-fidelity failures; no Actor or tools are executed.

This is a diagnostic comparison of reasoning off/on under identical visible
histories and native user instructions. Expectations are never prompt inputs.
"""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spork_core import paths
from spork_core.evaluation_artifacts import digest, write_json
from spork_core.interactive_tau2 import LocalRuntime
from tau2.data_model.tasks import Task
from tau2.user.user_simulator import UserSimulator


def prepare(args):
    if args.output.exists():
        raise ValueError("Use a new output directory; diagnostic samples are immutable")
    source = args.source / "trajectories/airline_2_r0_eta.json"
    messages = json.loads(source.read_text())["messages"]
    task_path = paths.TAU2_ROOT / "data/tau2/domains/airline/tasks.json"
    task = Task.model_validate(next(t for t in json.loads(task_path.read_text()) if str(t["id"]) == "2"))
    system = UserSimulator(llm=args.model, instructions=str(task.user_scenario), tools=None).system_prompt
    visible = [{"role": "system", "content": system}]
    cases = []
    users = 0
    closing_added = False
    for message in messages:
        if message["role"] == "assistant" and not message.get("tool_calls"):
            visible.append({"role": "user", "content": message["content"]})
            if not closing_added and re.search(r"have a (wonderful|great|fantastic) day", message["content"], re.I):
                cases.append({"case": "closing", "messages": list(visible),
                              "expectation_for_analysis_only": "Emit native STOP after the resolved conversation and farewells"})
                closing_added = True
        elif message["role"] == "user":
            if users == 2:
                cases.append({"case": "unknown_flight_details", "messages": list(visible),
                              "expectation_for_analysis_only": "Do not invent a flight number or date; scenario says reservation details are unknown",
                              "observed_bad_reply": message["content"]})
            visible.append({"role": "assistant", "content": message["content"]})
            users += 1
    if {c["case"] for c in cases} != {"closing", "unknown_flight_details"}:
        raise ValueError("Diagnostic source lacks the two expected situations")
    args.output.mkdir(parents=True)
    sources = [Path(__file__), ROOT / "spork_core/interactive_tau2.py",
               ROOT / "spork_core/prefix_replay.py"]
    manifest = {"purpose": "user_simulator_fidelity_diagnostic", "model": args.model,
                "base_url": args.base_url, "seed": 42, "modes": {"plain": 512, "thinking": 2048},
                "source_sha256": {str(p): digest(p) for p in [source, task_path]},
                "code_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources},
                "scope": "Two selected failure histories, one generation per mode, no claim of general simulator reliability"}
    write_json(args.output / "manifest.json", manifest)
    write_json(args.output / "inputs.json", cases)
    for source_path in sources:
        dest = args.output / "code" / source_path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(source_path.read_bytes())
    return cases, manifest


def run(args, cases, manifest):
    from loguru import logger
    from transformers import AutoTokenizer
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    runtime = LocalRuntime(args.base_url, args.model)
    try:
        status = runtime.run(runtime.http.status(runtime.session))
        if status.get("active_requests") != 0 or status.get("pending_verifications") != 0:
            raise RuntimeError("Another request is active")
        rows = []
        for case in cases:
            for mode, budget in manifest["modes"].items():
                prompt = tok.apply_chat_template(case["messages"], tokenize=False,
                    add_generation_prompt=True, enable_thinking=mode == "thinking")
                ids = tok.encode(prompt, add_special_tokens=False)
                print(case["case"], mode, flush=True)
                choice = runtime.run(runtime.http.complete(runtime.session, ids,
                    max_tokens=budget, seed=42, stop=["<|im_end|>"]))
                text = choice["text"].split("</think>")[-1].strip()
                row = {"case": case["case"], "mode": mode, "choice": choice,
                       "prompt_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                       "visible_response": text,
                       "complete": choice["finish_reason"] == "stop" and (mode == "plain" or "</think>" in choice["text"])}
                # STOP is a protocol check. Groundedness is inspected from the
                # saved source instructions and full visible response, not a
                # substring test for the single previously invented flight.
                row["has_stop_marker"] = "###STOP###" in text
                rows.append(row)
                write_json(args.output / f'{case["case"]}_{mode}.json', row)
                print(json.dumps({k: row[k] for k in ("case", "mode", "visible_response", "complete", "has_stop_marker")}, ensure_ascii=False), flush=True)
        write_json(args.output / "summary.json", {"rows": rows, "scope": manifest["scope"]})
    finally:
        runtime.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="/mnt/data/zhangshuai/LLMs/Models/Qwen3-32B")
    p.add_argument("--base-url", default="http://127.0.0.1:8017/v1")
    p.add_argument("--run", action="store_true")
    args = p.parse_args()
    cases, manifest = prepare(args)
    if args.run:
        port = urlparse(args.base_url).port or 80
        with ExitStack() as stack:
            for sibling in sorted(ROOT.parent.glob("spork*")):
                if (sibling / "spork_core").is_dir():
                    handle = stack.enter_context((sibling / "results" / f"controller_online_{port}.lock").open("a+"))
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(args, cases, manifest)
