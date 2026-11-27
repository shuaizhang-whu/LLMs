#!/usr/bin/env python3
"""Gemma 4 feasibility on the running server, before any formal run (about 10-30 minutes).

Default prints the plan only. With --run, on first-turn prompts of a few train tasks:
  1. server configuration (serving mode, loaded sources);
  2. batch dependence: one prompt alone, then together with three others, then alone again;
  3. full first turns: thought channel, call format and parse, thought length, decode speed;
  4. probes at the first token and inside the thought: parse rate, confidence, agreement
     with the final call;
  5. exact-input commit preconditions: the next input rendered from the model's own output
     equals the one rendered from the parsed call, and the D3 draft equals the generated call.
Writes one JSON report and prints a short verdict. Generation requests only; no tools run.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
from datetime import datetime
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spork_core.benchmarks.tau2 import load_system_prompt, load_tools
from spork_core.evaluation_artifacts import write_json
from spork_core.gemma4 import CALL_OPEN_TOKEN_ID, Gemma4Adapter
from spork_core.prefix_replay import PrefixHttpClient, StreamState, same_call
from spork_unified_eval import load_tau2_tasks
from scripts.run_successor_full_eval import evaluation_session, reset_trial_cache

DEFAULT_TASKS = "retail:51,retail:61,airline:2,airline:3"
THRESHOLD = 0.9


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


async def generate(session, http, ids, args, max_tokens, stop=()):
    state = StreamState()
    started = time.perf_counter()
    await http.stream(session, ids, state, max_tokens=max_tokens, seed=args.seed, stop=list(stop))
    return state, started


async def batch_dependence(session, http, prompts, args):
    alone, _ = await generate(session, http, prompts[0], args, args.det_tokens)
    together = [StreamState() for _ in prompts]
    await asyncio.gather(*(http.stream(session, ids, state, max_tokens=args.det_tokens, seed=args.seed)
                           for ids, state in zip(prompts, together)))
    again, _ = await generate(session, http, prompts[0], args, args.det_tokens)
    a, b, c = alone.token_ids, together[0].token_ids, again.token_ids
    return {"tokens": len(a), "alone_equals_batched": a == b, "alone_repeatable": a == c,
            "first_divergence_batched": first_divergence(a, b),
            "first_divergence_repeated": first_divergence(a, c)}


async def first_turn(session, http, adapter, ids, args):
    state, started = await generate(session, http, ids, args, args.max_tokens, adapter.main_stop)
    text = state.text
    call = adapter.parse_main_tool_call(text)
    try:
        adapter.validate_actor_completion(text)
        completion_error = None
    except RuntimeError as error:
        completion_error = str(error)
    tokens = len(state.token_ids)
    decode_s = (state.finished_at - state.first_output_at) if state.first_output_at else None
    return {"finish_reason": state.finish_reason, "tokens": tokens,
            "thought_tokens": adapter.token_metrics(text)["cot_tokens"],
            "starts_with_thought_channel": text.startswith("<|channel>thought"),
            "thought_closed": adapter.reasoning_closed(text), "call": call,
            "call_tool_known": (call["name"] in {t["function"]["name"] for t in adapter.tools}) if call else None,
            "visible_answer": None if call else adapter.visible_answer(text),
            "completion_error": completion_error,
            "first_token_s": (state.first_output_at - started) if state.first_output_at else None,
            "ms_per_token": 1e3 * decode_s / (tokens - 1) if decode_s and tokens > 1 else None,
            "text": text, "token_ids": list(state.token_ids)}


async def probes(session, http, adapter, prompt_ids, turn, args):
    generated, thought = turn["token_ids"], turn["thought_tokens"]
    positions = sorted({1, max(1, thought // 4), max(1, thought // 2), max(1, 3 * thought // 4)})
    rows = []
    for k in positions:
        if k > len(generated):
            continue
        visible = generated[:k]
        text = adapter.tokenizer.decode(visible, skip_special_tokens=False)
        ids = adapter.prediction_prompt_ids(prompt_ids, visible, text)
        started = time.perf_counter()
        choice = await http.complete(session, ids, max_tokens=160, stop=adapter.probe_stop,
                                     logprobs=5, seed=args.seed)
        call = adapter.parse_probe_tool_call(choice.get("text", ""))
        confidence = adapter.span_confidence(adapter.extract_probe_logprobs(choice.get("logprobs")),
                                             skip_first=True)
        rows.append({"visible_tokens": k, "probe_text": choice.get("text"),
                     "finish_reason": choice.get("finish_reason"), "parsed": call is not None,
                     "call": call, "confidence": confidence,
                     "would_dispatch": confidence is not None and confidence >= THRESHOLD,
                     "matches_final_call": same_call(call, turn["call"]),
                     "wall_s": time.perf_counter() - started})
    return rows


def commit_preconditions(adapter, messages, turn):
    call = turn["call"]
    if call is None:
        return None
    observation = '{"status": "observation used only to render the next input"}'
    def next_input(assistant_text):
        history = copy.deepcopy(messages) + [{"role": "assistant", "content": assistant_text},
                                             {"role": "tool", "content": observation}]
        return adapter.tokenizer.encode(adapter.render_main_prompt(history, enable_thinking=True),
                                        add_special_tokens=False)
    real, anchored = next_input(turn["text"]), next_input(adapter.speculative_tool_message(call))
    draft = adapter.tokenizer.encode(adapter.draft_body(call), add_special_tokens=False)
    generated = turn["token_ids"]
    body = (generated[generated.index(CALL_OPEN_TOKEN_ID) + 1:]
            if CALL_OPEN_TOKEN_ID in generated else None)
    return {"next_input_identical": real == anchored, "next_input_tokens": len(real),
            "first_divergence": first_divergence(real, anchored),
            "draft_tokens": len(draft), "draft_matches_generated_call": bool(body) and body[:len(draft)] == draft}


async def run(args, cases, tok):
    http = PrefixHttpClient(args.base_url, args.model, 900, extra_body=Gemma4Adapter.request_fields)
    report = {"started_at": datetime.now().isoformat(), "model": args.model, "tasks": [c["group_id"] for c in cases],
              "seed": args.seed, "max_tokens": args.max_tokens, "det_tokens": args.det_tokens,
              "request_fields": Gemma4Adapter.request_fields}
    async with evaluation_session() as session:
        status = await http.status(session)
        if status.get("active_requests") != 0:
            raise RuntimeError("The server is busy; run this on an idle server")
        report["server_config"] = status.get("server_config")
        mode = "invariant" if (status.get("server_config") or {}).get("batch_invariant_requested") else "standard"
        report["serving_mode"] = mode
        print(f"Server: {mode} kernels, project root {(status.get('server_config') or {}).get('project_root')}", flush=True)
        await reset_trial_cache(session, http)
        for case in cases:
            await http.complete(session, case["prompt_ids"], max_tokens=1, seed=args.seed)
        report["batch_dependence"] = await batch_dependence(session, http, [c["prompt_ids"] for c in cases], args)
        print("Batch dependence:", {k: v for k, v in report["batch_dependence"].items()}, flush=True)
        report["cases"] = []
        for case in cases:
            turn = await first_turn(session, http, case["adapter"], case["prompt_ids"], args)
            row = {"group_id": case["group_id"], "prompt_tokens": len(case["prompt_ids"]), "first_turn": turn}
            print(f'{case["group_id"]}: {turn["tokens"]} tokens ({turn["thought_tokens"]} before the call), '
                  f'{turn["ms_per_token"] or 0:.1f} ms/token, call={turn["call"] and turn["call"]["name"]}, '
                  f'error={turn["completion_error"]}', flush=True)
            if turn["call"] is not None:
                row["probes"] = await probes(session, http, case["adapter"], case["prompt_ids"], turn, args)
                row["commit_preconditions"] = commit_preconditions(case["adapter"], case["messages"], turn)
            report["cases"].append(row)
        report["final_status"] = await http.status(session)
    report["verdict"] = verdict(report)
    output = args.output or ROOT / "results" / f"gemma4_feasibility_{mode}_{datetime.now():%Y%m%d_%H%M%S}.json"
    write_json(output, report)
    print(f"Report: {output}")
    for key, value in report["verdict"].items():
        print(f"  {key}: {value}")
    if report["verdict"]["deterministic"] is False:
        raise SystemExit(f"Batch-invariance check failed. See {output}. Use the standard serving experiment.")


def verdict(report):
    turns = [c["first_turn"] for c in report["cases"]]
    probes_ = [p for c in report["cases"] for p in c.get("probes", [])]
    commits = [c["commit_preconditions"] for c in report["cases"] if c.get("commit_preconditions")]
    speeds = [t["ms_per_token"] for t in turns if t["ms_per_token"]]
    dependence = report["batch_dependence"]
    return {
        "format_ok": all(t["finish_reason"] == "stop" and t["completion_error"] is None for t in turns),
        "calls_parsed": f'{sum(t["call"] is not None for t in turns)}/{len(turns)} first turns ended in a parsed call',
        "unknown_tools": sum(t["call_tool_known"] is False for t in turns),
        "deterministic": (dependence["alone_equals_batched"] and dependence["alone_repeatable"]
                          if report["serving_mode"] == "invariant" else "not expected under standard kernels"),
        "next_input_identical": f'{sum(c["next_input_identical"] for c in commits)}/{len(commits)}',
        "d3_draft_matches": f'{sum(c["draft_matches_generated_call"] for c in commits)}/{len(commits)}',
        "probes_parsed": f'{sum(p["parsed"] for p in probes_)}/{len(probes_)}',
        "probes_matching_final_call": f'{sum(p["matches_final_call"] for p in probes_)}/{len(probes_)}',
        "probes_dispatchable": f'{sum(p["would_dispatch"] for p in probes_)}/{len(probes_)} at confidence >= {THRESHOLD}',
        "median_ms_per_token": round(statistics.median(speeds), 1) if speeds else None,
        "median_tokens_before_call": statistics.median(t["thought_tokens"] for t in turns) if turns else None,
    }


def main():
    port = os.environ.get("SPORK_CHECK_PORT", "8017")
    model_root = os.environ.get("SPORK_MODEL_ROOT", str(ROOT.parents[1] / "Models"))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=os.environ.get("SPORK_GEMMA_MODEL", model_root + "/gemma-4-31B-it"))
    p.add_argument("--base-url", default=f"http://127.0.0.1:{port}/v1")
    p.add_argument("--task-ids", default=DEFAULT_TASKS)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--det-tokens", type=int, default=384)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path)
    p.add_argument("--run", action="store_true")
    args = p.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tasks = {f'{t["domain"]}:{t["task_id"]}': t for t in load_tau2_tasks(args.seed)}
    cases = []
    for group_id in args.task_ids.split(","):
        task, domain = tasks[group_id], group_id.split(":")[0]
        adapter = Gemma4Adapter(tok, load_tools(domain))
        messages = [{"role": "system", "content": load_system_prompt(domain)},
                    {"role": "user", "content": task["user_message"]}]
        ids = tok.encode(adapter.render_main_prompt(messages, enable_thinking=True), add_special_tokens=False)
        cases.append({"group_id": group_id, "adapter": adapter, "messages": messages, "prompt_ids": ids})
    print(f"Prepared {len(cases)} first-turn prompts ({[len(c['prompt_ids']) for c in cases]} tokens) for {args.model}",
          flush=True)
    if args.run:
        asyncio.run(run(args, cases, tok))


if __name__ == "__main__":
    main()
