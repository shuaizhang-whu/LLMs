#!/usr/bin/env python3
"""Serving diagnostic on saved TRAIN prompts; not a causal Agent speed test."""
import argparse
import asyncio
from contextlib import ExitStack
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spork_core.benchmarks.tau2 import get_environment, load_system_prompt, load_tools
from spork_core.evaluation_artifacts import digest, freeze_plan, write_json
from spork_core.executors import Tau2ToolExecutor
from spork_core.prefix_replay import PrefixHttpClient
from spork_core.qwen35_native import NativeQwen35Adapter
from spork_core.turn_runner import _append_tool_messages
from scripts.run_successor_full_eval import evaluation_session, reset_trial_cache
from spork_unified_eval import load_tau2_tasks
from time import perf_counter


class DiagnosticHTTP(PrefixHttpClient):
    def __init__(self, *args, presence_penalty=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.presence_penalty = presence_penalty

    def body(self, *args, **kwargs):
        result = super().body(*args, **kwargs)
        result["presence_penalty"] = self.presence_penalty
        return result


def build(args):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    adapter = NativeQwen35Adapter(tokenizer, load_tools("airline"))
    reference = json.loads(args.reference.read_text())
    task = next(t for t in load_tau2_tasks(42) if t["domain"] == "airline" and str(t["task_id"]) == "2")
    messages = [{"role": "system", "content": load_system_prompt("airline")},
                {"role": "user", "content": task["user_message"]}]
    first = tokenizer.encode(adapter.render_main_prompt(messages, enable_thinking=True), add_special_tokens=False)
    assert hashlib.sha256(json.dumps(first).encode()).hexdigest() == reference["initial_prompt_sha256"]
    parent = reference["turns"][0]
    call = {"name": parent["baseline_tool_name"], "arguments": parent["baseline_tool_args"]}
    executor = Tau2ToolExecutor(get_environment("airline"), floor_s=0)
    observation, _ = executor.execute(call)
    assert hashlib.sha256(observation.encode()).hexdigest() == parent["tool_result_sha256"]
    _append_tool_messages(messages, parent["baseline_text"], observation)
    second = tokenizer.encode(adapter.render_main_prompt(messages, enable_thinking=True), add_special_tokens=False)
    probes = {}
    for name, count in [("probe_early", 1), ("probe_late", 205)]:
        visible = parent["main_token_ids"][:count]
        probes[name] = adapter.prediction_prompt_ids(first, visible, tokenizer.decode(visible))
        legacy_suffix = "\n</think>\n\n<tool_call>\n<function="
        probes[name + "_legacy"] = first + visible + tokenizer.encode(legacy_suffix, add_special_tokens=False)
    inputs = {"parent": first, "next": second, **probes}
    sources = sorted(set([Path(__file__), ROOT / "scripts/run_successor_full_eval.py",
        ROOT / "scripts/run_successor_smoke.py"] + list(ROOT.glob("*.py"))
        + list((ROOT / "spork_core").rglob("*.py"))))
    manifest = {"purpose": "qwen35_serving_diagnostic", "model": args.model,
                "base_url": args.base_url, "seed": 42, "max_tokens": 512,
                "presence_penalty": args.presence_penalty, "cases": args.cases,
                "mixed": args.mixed, "main_logprobs": args.main_logprobs,
                "background_case": args.background_case,
                "reference": str(args.reference), "reference_sha256": digest(args.reference),
                "input_sha256": hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest(),
                "code_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources},
                "scope": "Saved training prompts, real server; no online acceleration claim"}
    freeze_plan(args.output, manifest, ROOT, sources)
    write_json(args.output / "inputs.json", inputs)
    return inputs, adapter, reference


async def run(args, inputs, adapter, reference):
    http = DiagnosticHTTP(args.base_url, args.model, 600, presence_penalty=args.presence_penalty)
    output = args.output
    rows = []
    async with evaluation_session() as session:
        write_json(output / "server_start.json", await http.status(session))
        async def request(case, warm, repeat, mixed=False):
            await reset_trial_cache(session, http)
            if warm:
                await http.complete(session, inputs[case], max_tokens=1, seed=42)
            target = asyncio.create_task(http.complete(session, inputs[case],
                max_tokens=160 if case.startswith("probe") else 512, seed=42,
                stop=adapter.probe_stop if case.startswith("probe") else adapter.main_stop,
                logprobs=5 if case.startswith("probe") else (
                    None if args.main_logprobs < 0 else args.main_logprobs),
                request_id="diag-" + uuid.uuid4().hex))
            background = None
            if mixed:
                await asyncio.sleep(.3)
                background_case = (("next" if case == "parent" else "parent")
                                   if args.background_case == "other_actor" else args.background_case)
                background = asyncio.create_task(http.complete(session, inputs[background_case],
                    max_tokens=160 if background_case.startswith("probe") else 512,
                    seed=42, stop=adapter.probe_stop if background_case.startswith("probe") else adapter.main_stop,
                    logprobs=5 if background_case.startswith("probe") else None,
                    request_id="diag-background-" + uuid.uuid4().hex))
            try:
                response = await target
                background_response = await background if background else None
            finally:
                for future in [target, background]:
                    if future is not None and not future.done():
                        future.cancel()
                await asyncio.gather(*(f for f in [target, background] if f is not None),
                                     return_exceptions=True)
            row = {"case": case, "warmup": warm, "repeat": repeat, "mixed": mixed,
                   "response": response, "background_response": background_response}
            if case.startswith("probe"):
                row["parsed_call"] = (adapter.parse_main_tool_call(
                    "<tool_call>\n<function=" + response["text"]) if case.endswith("_legacy")
                    else adapter.parse_probe_tool_call(response["text"]))
            else:
                row["parsed_call"] = adapter.parse_main_tool_call(response["text"])
                old = reference["turns"][0 if case == "parent" else 1]["main_token_ids"]
                ids = response["token_ids"]
                row["same_reference_tokens"] = old == ids
                row["first_reference_difference"] = next(
                    (i for i, (a, b) in enumerate(zip(old, ids)) if a != b), min(len(old), len(ids)))
            write_json(output / "records" / f"{case}_warm{int(warm)}_mix{int(mixed)}_r{repeat}.json", row)
            rows.append(row)
            print(json.dumps({k: row[k] for k in row if k not in {"response", "background_response"}}, ensure_ascii=False), flush=True)
            print("finish", response["finish_reason"], "tokens", len(response["token_ids"]),
                  "seconds", round(response["wall_s"], 3), flush=True)
        for case in [x for x in args.cases.split(",") if not x.startswith("probe")]:
            for warm in [False, True]:
                for repeat in range(2):
                    await request(case, warm, repeat)
            if args.mixed:
                for repeat in range(2):
                    await request(case, True, repeat, mixed=True)
        for case in [x for x in args.cases.split(",") if x.startswith("probe")]:
            await request(case, False, 0)
        write_json(output / "summary.json", {"rows": [{
            **{k: v for k, v in r.items() if k not in {"response", "background_response"}},
            "finish_reason": r["response"]["finish_reason"],
            "tokens": len(r["response"]["token_ids"]),
            "text": r["response"]["text"],
            "wall_s": r["response"]["wall_s"]} for r in rows],
            "repeated_tokens_equal": {f"{case}_warm{int(warm)}": (
                selected[0]["response"]["token_ids"] == selected[1]["response"]["token_ids"])
                for case in ["parent", "next"] for warm in [False, True]
                if len(selected := [r for r in rows if r["case"] == case and r["warmup"] == warm and not r["mixed"]]) == 2},
            "solo_vs_mixed_tokens_equal": {case: (
                solo[0]["response"]["token_ids"] == mixed[0]["response"]["token_ids"]
                == mixed[1]["response"]["token_ids"])
                for case in ["parent", "next"]
                if (solo := [r for r in rows if r["case"] == case and r["warmup"] and not r["mixed"]])
                and len(mixed := [r for r in rows if r["case"] == case and r["mixed"]]) == 2},
            "cold_vs_warm_tokens_equal": {case: (
                cold[0]["response"]["token_ids"] == warm[0]["response"]["token_ids"])
                for case in ["parent", "next"]
                if (cold := [r for r in rows if r["case"] == case and not r["warmup"] and not r["mixed"]])
                and (warm := [r for r in rows if r["case"] == case and r["warmup"] and not r["mixed"]])},
            "scope": "Serving diagnostic only; saved future prompts are not used in online evaluation"})
        write_json(output / "server_end.json", await http.status(session))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, default=ROOT / "results/qwen35_9b_20261101_preflight/records/airline_2_r0_spork.json")
    p.add_argument("--model", default="/mnt/data/zhangshuai/LLMs/Models/Qwen3.5-9B")
    p.add_argument("--base-url", default="http://127.0.0.1:8027/v1")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cases", default="parent,next,probe_early,probe_late")
    p.add_argument("--mixed", action="store_true")
    p.add_argument("--presence-penalty", type=float, default=0)
    p.add_argument("--main-logprobs", type=int, default=5)
    p.add_argument("--background-case", choices=["probe_early", "other_actor"], default="probe_early")
    p.add_argument("--run", action="store_true")
    args = p.parse_args()
    if (args.output / "records").exists():
        raise ValueError("Diagnostic attempts are immutable; select a new output directory")
    inputs, adapter, reference = build(args)
    print("Prepared saved train-prompt diagnostic:", args.output, flush=True)
    if args.run:
        with ExitStack() as stack:
            for root in sorted(ROOT.parent.glob("spork*")):
                if (root / "spork_core").is_dir():
                    path = root / "results/controller_online_8027.lock"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    fcntl.flock(stack.enter_context(path.open("a+")).fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            asyncio.run(run(args, inputs, adapter, reference))


if __name__ == "__main__":
    main()
