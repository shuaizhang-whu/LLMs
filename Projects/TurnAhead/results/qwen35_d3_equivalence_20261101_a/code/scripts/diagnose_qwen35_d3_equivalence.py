#!/usr/bin/env python3
"""Compare arbitrary D3 drafts on identical saved training input; no tools run."""
import argparse
import asyncio
from contextlib import ExitStack
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spork_core.evaluation_artifacts import digest, freeze_plan, write_json
from spork_core.prefix_replay import PrefixHttpClient, StreamState, counter_delta
from spork_core.qwen35_native import NativeQwen35Adapter
from spork_core.benchmarks.tau2 import load_tools
from scripts.run_successor_full_eval import evaluation_session, reset_trial_cache


async def execute(args, prompt, adapter, drafts):
    http = PrefixHttpClient(args.base_url, args.model, 600, presence_penalty=args.presence_penalty)
    records = []
    async with evaluation_session() as session:
        write_json(args.output / "server_start.json", await http.status(session))
        for repeat in range(2):
            for name, draft in drafts.items():
                await http.clear(session)
                await reset_trial_cache(session, http)
                await http.complete(session, prompt, max_tokens=1, seed=42)
                before = await http.status(session)
                state, rid = StreamState(), "d3-equivalence-" + uuid.uuid4().hex
                task = asyncio.create_task(http.stream(session, prompt, state,
                    max_tokens=512, seed=42, stop=adapter.main_stop, request_id=rid))
                registration = None
                try:
                    if draft is not None:
                        while not state.token_ids and not task.done():
                            state.changed.clear()
                            if not state.token_ids and not task.done():
                                try:
                                    await asyncio.wait_for(state.changed.wait(), .1)
                                except asyncio.TimeoutError:
                                    pass
                        if not task.done():
                            async with session.post(http.root + "/spork/set_tokens", json={
                                    "request_id": rid, "prompt_len": len(prompt),
                                    "draft_tokens": draft}, timeout=30) as response:
                                response.raise_for_status()
                                registration = await response.json()
                            if registration.get("status") != "ok":
                                raise RuntimeError(f"D3 registration failed: {registration}")
                    await task
                finally:
                    if not task.done():
                        task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                    await http.clear(session)
                after = await http.status(session)
                row = {"draft_case": name, "repeat": repeat,
                    "text": state.text, "token_ids": list(state.token_ids),
                    "finish_reason": state.finish_reason,
                    "parsed_call": adapter.parse_main_tool_call(state.text),
                    "registration": registration, "counters": counter_delta(before, after),
                    "last_verification": after.get("last_verification")}
                write_json(args.output / "records" / f"{name}_r{repeat}.json", row)
                records.append(row)
                print(json.dumps({k: row[k] for k in ["draft_case", "repeat", "finish_reason",
                    "parsed_call", "counters"]}, ensure_ascii=False), flush=True)
        baseline = next(r for r in records if r["draft_case"] == "off")
        comparisons = [{"draft_case": r["draft_case"], "repeat": r["repeat"],
            "same_tokens_as_off": r["token_ids"] == baseline["token_ids"],
            "tokens": len(r["token_ids"]),
            "first_difference": next((i for i, (a, b) in enumerate(zip(r["token_ids"], baseline["token_ids"]))
                                      if a != b), min(len(r["token_ids"]), len(baseline["token_ids"]))),
            "injections": r["counters"].get("verified_injections", 0)} for r in records]
        write_json(args.output / "summary.json", {"comparisons": comparisons,
            "all_tokens_equal": all(r["same_tokens_as_off"] for r in comparisons),
            "scope": "D3 correctness audit on one saved training prompt; not an online agent speed test"})
        write_json(args.output / "server_end.json", await http.status(session))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, default=ROOT / "results/qwen35_9b_20261101_gdnref_ablation/records/airline_4_r0_serial.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="/mnt/data/zhangshuai/LLMs/Models/Qwen3.5-9B")
    p.add_argument("--base-url", default="http://127.0.0.1:8027/v1")
    p.add_argument("--presence-penalty", type=float, default=1.5)
    p.add_argument("--run", action="store_true")
    args = p.parse_args()
    if (args.output / "records").exists():
        raise ValueError("Use a new immutable audit output directory")
    record = json.loads(args.reference.read_text())
    generation = record["failed_generation"]
    prompt = generation["prompt_token_ids"]
    assert hashlib.sha256(json.dumps(prompt).encode()).hexdigest() == generation["prompt_sha256"]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    adapter = NativeQwen35Adapter(tokenizer, load_tools("airline"))
    calls = {"correct": {"name": "get_reservation_details", "arguments": {"reservation_id": "NM1VX1"}},
             "wrong": {"name": "get_user_details", "arguments": {"user_id": "CANARY_DO_NOT_ACCEPT"}}}
    drafts = {"off": None, **{k: tokenizer.encode(adapter.draft_body(v), add_special_tokens=False)
                            for k, v in calls.items()}}
    sources = sorted(set([Path(__file__), ROOT / "scripts/run_successor_full_eval.py"]
        + list(ROOT.glob("*.py")) + list((ROOT / "spork_core").rglob("*.py"))))
    manifest = {"purpose": "D3 on/off and wrong-draft invariance audit",
        "model": args.model, "base_url": args.base_url, "seed": 42,
        "max_tokens": 512, "presence_penalty": args.presence_penalty,
        "reference": str(args.reference), "reference_sha256": digest(args.reference),
        "prompt_sha256": generation["prompt_sha256"], "drafts": drafts,
        "scope": "Saved training input and deliberately supplied drafts; no tools execute and no acceleration claim",
        "code_sha256": {str(f.relative_to(ROOT)): digest(f) for f in sources}}
    freeze_plan(args.output, manifest, ROOT, sources)
    write_json(args.output / "input.json", {"prompt_token_ids": prompt, "drafts": drafts})
    print("Prepared audit:", args.output, flush=True)
    if args.run:
        with ExitStack() as stack:
            for root in sorted(ROOT.parent.glob("spork*")):
                if (root / "spork_core").is_dir():
                    path = root / "results/controller_online_8027.lock"
                    fcntl.flock(stack.enter_context(path.open("a+")).fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            asyncio.run(execute(args, prompt, adapter, drafts))


if __name__ == "__main__":
    main()
