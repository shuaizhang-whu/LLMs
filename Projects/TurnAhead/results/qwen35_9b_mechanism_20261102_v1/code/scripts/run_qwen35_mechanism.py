#!/usr/bin/env python3
"""Bounded saved-prefix diagnostics. Default prepares on CPU, --run uses HTTP.

Oracle D3 drafts and synthetic concurrent requests are used ONLY in the stop
boundary diagnostic. Pair timing uses the unmodified online SPORK/successor
code, native simulated tools and restored historical state, with no oracle.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback
import uuid
from contextlib import ExitStack

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spork_core import paths
from spork_core.evaluation_artifacts import digest, freeze_plan, write_json
from spork_core.expanded_validation import verify_frozen
from spork_core.interactive_successor import InteractiveRuntime, SuccessorInteractiveAgent
from spork_core.interactive_tau2 import NativeToolExecutor
from spork_core.prefix_replay import PrefixHttpClient, StreamState, counter_delta
from spork_core.successor_prefetch import run_successor_pair, cancel_and_drain
from spork_core.turn_runner import _append_tool_messages
from scripts.check_qwen35_experiment import server_check
from scripts.run_successor_full_eval import reset_trial_cache
from tau2.data_model.message import AssistantMessage
from tau2.data_model.tasks import Task
from tau2.registry import registry

CASES = (("gain", "airline", "2", 1),
         ("adopted_slow", "retail", "28", 20),
         ("candidate_miss", "retail", "28", 22))
MODES = ("spork", "raw", "raw_no_probe")
BOUNDARY = ("off_solo", "on_solo", "off_concurrent", "on_concurrent")


def sealed_read(path):
    if digest(path) != json.loads(path.with_name(path.name + ".seal").read_text())["sha256"]:
        raise ValueError(f"Record seal mismatch: {path}")
    return json.loads(path.read_text())


def seal(path, row):
    if path.exists():
        raise ValueError(f"Refusing to overwrite {path}")
    write_json(path, row)
    write_json(path.with_name(path.name + ".seal"), {"sha256": digest(path)})


def restore(source, tokenizer, domain, task_id, first_turn):
    """Rebuild real tool state and exact Actor messages; never invoke a model."""
    record_path = source / f"records/{domain}_{task_id}_r0_spork.json"
    record = sealed_read(record_path)
    if record["errors"]:
        raise ValueError("A diagnostic prefix must come from a completed trajectory")
    trajectory_path = source / record["trajectory_path"]
    if digest(trajectory_path) != record["trajectory_sha256"]:
        raise ValueError("Historical trajectory changed")
    trajectory = json.loads(trajectory_path.read_text())["messages"]
    tasks = json.loads((paths.TAU2_ROOT / f"data/tau2/domains/{domain}/tasks.json").read_text())
    task = Task.model_validate(next(t for t in tasks if str(t["id"]) == task_id))
    if task.initial_state is not None:
        raise ValueError("These fixed diagnostic cases require a fresh native initial state")
    env = registry.get_env_constructor(domain)(solo_mode=False)
    env.set_state(initialization_data=None, initialization_actions=None, message_history=[])
    agent = SuccessorInteractiveAgent(env, None, tokenizer, "spork")
    state = agent.get_init_state([AssistantMessage(role="assistant", content=trajectory[0]["content"])])
    executor = NativeToolExecutor(env, 0.)
    index = 0
    for message in trajectory[1:]:
        if message["role"] == "user":
            state.append({"role": "user", "content": message["content"]})
        elif message["role"] == "assistant":
            turn = record["turns"][index]
            actual = tokenizer.encode(agent.adapter.render_main_prompt(state, enable_thinking=True), add_special_tokens=False)
            if actual != turn["prompt_token_ids"]:
                raise ValueError(f"Restored prompt mismatch: {domain}:{task_id} turn {index+1}")
            if index + 1 == first_turn:
                return agent, state, executor.env, record
            if turn["baseline_has_tool_call"]:
                call = {"name": turn["baseline_tool_name"], "arguments": turn["baseline_tool_args"]}
                text, _ = executor.execute(call)
                if hashlib.sha256(text.encode()).hexdigest() != turn["tool_result_sha256"]:
                    raise ValueError("Restored tool observation mismatch")
                _append_tool_messages(state, turn["baseline_text"], text)
            else:
                state.append({"role": "assistant", "content": turn["baseline_text"].split("</think>")[-1].strip()})
            if {"agent_db": executor.env.get_db_hash(), "user_db": executor.env.get_user_db_hash()} != turn["environment_after"]:
                raise ValueError("Restored environment state mismatch")
            index += 1
    raise ValueError("Requested prefix not found")


def token_boundary(ids, tool_close, eos, valid_tool):
    """Report, never erase, suffix tokens after a validated tool stop boundary."""
    if valid_tool and tool_close in ids:
        end = ids.index(tool_close) + 1
        return {"through_stop": ids[:end], "suffix": ids[end:],
                "suffix_is_only_eos": all(t == eos for t in ids[end:])}
    return {"through_stop": ids, "suffix": [], "suffix_is_only_eos": True}


class ChunkHTTP(PrefixHttpClient):
    """Capture original stream chunks, including stop_reason, for diagnosis."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chunks = []

    async def stream(self, session, prompt_ids, state, **kwargs):
        import aiohttp
        started = time.perf_counter()
        done = False
        try:
            async with session.post(self.base_url + "/completions",
                    json=self.body(prompt_ids, stream=True, **kwargs),
                    timeout=aiohttp.ClientTimeout(total=self.timeout_s)) as response:
                response.raise_for_status()
                async for raw in response.content:
                    line = raw.decode().strip()
                    if not line.startswith("data:"):
                        continue
                    if line[5:].strip() == "[DONE]":
                        done = True
                        break
                    chunk = json.loads(line[5:])
                    self.chunks.append({"elapsed_s": time.perf_counter()-started, "payload": chunk})
                    if chunk.get("error"):
                        raise RuntimeError(str(chunk["error"]))
                    if chunk.get("usage"):
                        state.usage = chunk["usage"]
                    for choice in chunk.get("choices", []):
                        if choice.get("text") and choice.get("token_ids") is None:
                            raise RuntimeError("Missing stream token IDs")
                        if choice.get("token_ids"):
                            state.first_output_at = state.first_output_at or time.perf_counter()
                            state.token_ids.extend(choice["token_ids"])
                        state.text += choice.get("text") or ""
                        if choice.get("finish_reason") is not None:
                            state.finish_reason = choice["finish_reason"]
                        state.changed.set()
            if not done or state.finish_reason is None:
                raise RuntimeError("Incomplete SSE stream")
        finally:
            state.finished_at = time.perf_counter()
            state.changed.set()


async def first_token(task, state):
    while not state.token_ids and not task.done():
        state.changed.clear()
        try:
            await asyncio.wait_for(state.changed.wait(), .1)
        except asyncio.TimeoutError:
            pass
    if task.done():
        await task


async def idle(runtime):
    status = await runtime.http.status(runtime.session)
    if status.get("active_requests") != 0 or status.get("pending_verifications") != 0:
        raise RuntimeError("Server has active/unverified requests")
    return status


async def cold_start(runtime, warm_prompt):
    await idle(runtime)
    await runtime.http.clear(runtime.session)
    await reset_trial_cache(runtime.session, runtime.http)
    await runtime.http.complete(runtime.session, warm_prompt, max_tokens=1, seed=42)
    return await idle(runtime)


async def boundary_trial(runtime, inputs, mode, close_id, eos_id):
    target = ChunkHTTP(runtime.client.base_url, runtime.client.model, 900, presence_penalty=1.5)
    load = ChunkHTTP(runtime.client.base_url, runtime.client.model, 900, presence_penalty=1.5)
    state, other = StreamState(), StreamState()
    jobs = []
    row = {"errors": [], "kind": "boundary", "config": mode}
    before = await cold_start(runtime, inputs["warmup"])
    started = time.perf_counter()
    competitor_start = target_start = None
    try:
        if mode.endswith("concurrent"):
            competitor_start = time.perf_counter()
            job = asyncio.create_task(load.stream(runtime.session, inputs["load_prompt"], other,
                max_tokens=512, seed=42, stop=["</tool_call>"]))
            jobs.append(job)
            await first_token(job, other)
        rid = "boundary-" + uuid.uuid4().hex
        target_start = time.perf_counter()
        job = asyncio.create_task(target.stream(runtime.session, inputs["prompt"], state,
            max_tokens=8192, seed=42, stop=["</tool_call>"], request_id=rid))
        jobs.append(job)
        if mode.startswith("on_"):
            await first_token(job, state)
            if not job.done():
                async with runtime.session.post(runtime.http.root + "/spork/set_tokens",
                        json={"request_id": rid, "prompt_len": len(inputs["prompt"]),
                              "draft_tokens": inputs["oracle_draft"]}, timeout=30) as response:
                    response.raise_for_status()
                    row["registration"] = await response.json()
        await job
    except Exception as exc:
        row["errors"].append(f"{type(exc).__name__}: {exc}")
        row["traceback"] = traceback.format_exc()
    finally:
        target_end = state.finished_at or time.perf_counter()
        load_end = other.finished_at or target_end
        row["target_s"] = target_end-target_start if target_start else None
        row["concurrent_overlap_s"] = max(0., min(target_end, load_end)-max(target_start or target_end, competitor_start or target_end))
        await cancel_and_drain(jobs)
        await runtime.http.clear(runtime.session)
    after = await idle(runtime)
    if before.get("server_config") != after.get("server_config"):
        row["errors"].append("Server configuration changed during trial")
    row.update(wall_s=time.perf_counter()-started, text=state.text, token_ids=state.token_ids,
        finish_reason=state.finish_reason, chunks=target.chunks, load_chunks=load.chunks,
        load_tokens=len(other.token_ids), d3_counters=counter_delta(before, after))
    return row


async def pair_trial(runtime, tokenizer, restored, mode):
    old_agent, saved_messages, saved_env, reference = restored
    env, messages = copy.deepcopy(saved_env), copy.deepcopy(saved_messages)
    agent = SuccessorInteractiveAgent(env, runtime, tokenizer, mode)
    prompt = tokenizer.encode(agent.adapter.render_main_prompt(messages, enable_thinking=True), add_special_tokens=False)
    first = next(t["turn"] for t in reference["turns"] if t["prompt_token_ids"] == prompt)
    warm = agent.get_init_state()
    warm.append({"role": "user", "content": "Warmup."})
    warm_ids = tokenizer.encode(agent.adapter.render_main_prompt(warm, enable_thinking=True), add_special_tokens=False)
    before = await cold_start(runtime, warm_ids)
    trace, entries = {}, []
    row = {"kind": "pair", "config": mode, "errors": [], "first_turn": first}
    started = time.perf_counter()
    try:
        await run_successor_pair(runtime.session, runtime.client, agent.adapter, messages,
            NativeToolExecutor(env, 2.), first, agent.make_cfg,
            mode="baseline" if mode == "spork" else mode,
            on_turn=lambda t: entries.append(copy.deepcopy(t)), trace=trace, recover_child_failure=True)
    except Exception as exc:
        row["errors"].append(f"{type(exc).__name__}: {exc}")
        row["traceback"] = traceback.format_exc()
        row["failed_generation"] = getattr(exc, "partial_generation", None)
    finally:
        row["wall_s"] = time.perf_counter()-started
    after = await idle(runtime)
    if before.get("server_config") != after.get("server_config"):
        row["errors"].append("Server configuration changed during trial")
    row.update(turns=entries, trace=trace, d3_counters=counter_delta(before, after),
        parent_finished_s=trace.get("parent_finished_s"),
        parent_generation_s=entries[0]["main_end_ms"]/1000 if entries else None,
        post_parent_s=max(0., row["wall_s"]-trace["parent_finished_s"]) if "parent_finished_s" in trace else None)
    row["errors"].extend(trace.get("errors", []))
    return row


def summarize(output, plan, adapter, close_id, eos_id):
    rows = [sealed_read(output / "records" / (item["id"]+".json"))
            for item in plan["schedule"] if (output / "records" / (item["id"]+".json.seal")).exists()]
    report = {"expected": len(plan["schedule"]), "recorded": len(rows),
              "complete": len(rows)==len(plan["schedule"]), "boundary": [], "pairs": [],
              "scope": plan["scope"], "failures": [{"id":r["id"],"errors":r["errors"]} for r in rows if r["errors"]]}
    for rep in range(3):
        br = [r for r in rows if r["kind"] == "boundary" and r["repeat"] == rep]
        base = next((r for r in br if r["config"] == "off_solo"), None)
        if base:
            for r in br:
                call = adapter.parse_main_tool_call(r["text"])
                basecall = adapter.parse_main_tool_call(base["text"])
                x = token_boundary(r["token_ids"], close_id, eos_id, call is not None)
                y = token_boundary(base["token_ids"], close_id, eos_id, basecall is not None)
                report["boundary"].append({"id":r["id"], "raw_tokens_equal":r["token_ids"]==base["token_ids"],
                    "through_stop_equal":x["through_stop"]==y["through_stop"],
                    "suffix":x["suffix"], "suffix_is_only_eos":x["suffix_is_only_eos"],
                    "text_equal":r["text"]==base["text"], "call_equal":call==basecall,
                    "finish_reason":r["finish_reason"], "injections":r["d3_counters"]["inject_count"],
                    "concurrent_overlap_s":r["concurrent_overlap_s"]})
        for case, *_ in CASES:
            group = [r for r in rows if r["kind"]=="pair" and r["case"]==case and r["repeat"]==rep]
            base = next((r for r in group if r["config"]=="spork"),None)
            if not base:
                continue
            fields = ("prompt_token_ids", "generation_settings", "baseline_text", "baseline_has_tool_call",
                      "baseline_tool_name", "baseline_tool_args", "tool_result_sha256", "environment_after")
            for r in group:
                same = len(r["turns"])==len(base["turns"])==2 and not r["errors"] and not base["errors"]
                same = same and all(all(a.get(k)==b.get(k) for k in fields) for a,b in zip(r["turns"],base["turns"]))
                raw_equal = [t["main_token_ids"] for t in r["turns"]]==[t["main_token_ids"] for t in base["turns"]]
                report["pairs"].append({"id":r["id"],"case":case,"config":r["config"],"repeat":rep,
                    "same_session":r["session_id"]==base["session_id"],
                    "same_requests_text_actions_state":same, "raw_tokens_equal":raw_equal,
                    "wall_s":r["wall_s"], "saved_s":base["wall_s"]-r["wall_s"],
                    "parent_extra_s":r["parent_finished_s"]-base["parent_finished_s"] if r["parent_finished_s"] is not None and base["parent_finished_s"] is not None else None,
                    "post_parent_s":r["post_parent_s"], "child_adopted":r["trace"].get("adopted"),
                    "rejection":r["trace"].get("rejection"),
                    "wasted_child_tokens":r["trace"].get("observed_wasted_child_tokens"),
                    "child_predictions":len(r["trace"].get("child_prediction_requests", []))})
    report["timing_by_case"] = []
    for case, *_ in CASES:
        for mode in MODES[1:]:
            rr=[r for r in report["pairs"] if r["case"]==case and r["config"]==mode
                and r["same_requests_text_actions_state"] and r["raw_tokens_equal"] and r["same_session"]]
            report["timing_by_case"].append({"case":case,"config":mode,"strict_pairs":len(rr),
                "saved_s_each_repeat":[r["saved_s"] for r in rr],
                "mean_saved_s":statistics.mean(r["saved_s"] for r in rr) if rr else None})
    write_json(output / "summary.json", report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--source",type=Path,default=ROOT/"results/qwen35_9b_expanded_20261101_v7/expanded")
    parser.add_argument("--model",default=os.environ.get("SPORK_CODEX_MODEL"))
    parser.add_argument("--base-url",default="http://127.0.0.1:8027/v1")
    parser.add_argument("--run",action="store_true")
    args=parser.parse_args()
    args.source,args.output=args.source.resolve(),args.output.resolve()
    old_manifest=json.loads((args.source/"manifest.json").read_text())
    verify_frozen(args.source,old_manifest,ROOT,current=False)
    if args.model != old_manifest["model"] or args.base_url != old_manifest["base_url"]:
        raise ValueError("Use the same 9B model/service as v7")
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    restored={name:restore(args.source,tokenizer,d,t,n) for name,d,t,n in CASES}
    boundary=restore(args.source,tokenizer,"retail","55",2)
    agent,state,env,ref=boundary
    turn=ref["turns"][1]
    call={"name":turn["baseline_tool_name"],"arguments":turn["baseline_tool_args"]}
    warm=agent.get_init_state()+[{"role":"user","content":"Warmup."}]
    inputs={"prompt":turn["prompt_token_ids"],
        "oracle_draft":tokenizer.encode(agent.adapter.draft_body(call),add_special_tokens=False),
        "load_prompt":restored["gain"][3]["turns"][0]["prompt_token_ids"],
        "warmup":tokenizer.encode(agent.adapter.render_main_prompt(warm,enable_thinking=True),add_special_tokens=False)}
    close_id=tokenizer.convert_tokens_to_ids("</tool_call>")
    eos_id=tokenizer.convert_tokens_to_ids("<|im_end|>")
    schedule=[]
    for rep in range(3):
        for mode in BOUNDARY[rep:]+BOUNDARY[:rep]:
            schedule.append({"kind":"boundary","config":mode,"repeat":rep,"id":f"boundary_{mode}_r{rep}"})
    for rep in range(3):
        for i,(case,*_) in enumerate(CASES):
            shift=(rep+i)%len(MODES)
            for mode in MODES[shift:]+MODES[:shift]:
                schedule.append({"kind":"pair","case":case,"config":mode,"repeat":rep,"id":f"{case}_{mode}_r{rep}"})
    sources=sorted(set(list((ROOT/"spork_core").rglob("*.py"))+list(ROOT.glob("*.py"))
        +list((ROOT/"scripts").glob("*.py"))+list((ROOT/"scripts").glob("*.sh"))
        +list((ROOT/"tests").glob("*.py"))+list((paths.TAU2_ROOT/"src/tau2").rglob("*.py"))))
    reference_files=[args.source/"manifest.json"]
    for _,d,t,_ in [*CASES,("boundary","retail","55",2)]:
        f=args.source/f"records/{d}_{t}_r0_spork.json"
        reference_files.extend([f,f.with_name(f.name+".seal"),args.source/sealed_read(f)["trajectory_path"]])
    plan={"schedule":schedule,"model":args.model,"base_url":args.base_url,"seed":42,
        "cases":CASES,"boundary_inputs":inputs,"source":str(args.source),
        "reference_sha256":{str(f):digest(f) for f in reference_files},
        "scope":"Post-hoc selected saved TRAIN prefixes. 12 stop-boundary diagnostics with oracle drafts/synthetic load; 27 two-turn native-tool replays with live predictions. 3 same-seed timing repeats, no user/judge generation, no task-success or general speedup claim. Cold prefix cache and one fixed warmup token per trial, outside timing. No previous results modified.",
        "code_sha256":{str(f.relative_to(ROOT)):digest(f) for f in sources}}
    # JSON-normalize tuple constants before resumable manifest comparison.
    plan=json.loads(json.dumps(plan))
    freeze_plan(args.output,plan,ROOT,sources)
    summarize(args.output,plan,agent.adapter,close_id,eos_id)
    print(json.dumps({"output":str(args.output),"boundary_trials":12,"two_turn_trials":27,
        "restored_prefixes":4,"generation_requests":"enabled" if args.run else 0}),flush=True)
    if not args.run:
        return
    with ExitStack() as stack:
        for sibling in sorted(ROOT.parent.glob("spork*")):
            if (sibling/"spork_core").is_dir():
                lock=sibling/"results/controller_online_8027.lock"
                lock.parent.mkdir(exist_ok=True)
                fcntl.flock(stack.enter_context(lock.open("a+")).fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        server_check()
        runtime=InteractiveRuntime(args.base_url,args.model)
        session_id=uuid.uuid4().hex
        try:
            write_json(args.output/f"sessions/{session_id}_start.json",runtime.run(idle(runtime)))
            for i,item in enumerate(schedule):
                path=args.output/"records"/(item["id"]+".json")
                if path.with_name(path.name+".seal").exists():
                    sealed_read(path)
                    continue
                if path.exists():
                    raise ValueError(f"Unsealed record requires inspection: {path}")
                print(f'[{i+1}/{len(schedule)}] {item["id"]}',flush=True)
                row=runtime.run(boundary_trial(runtime,inputs,item["config"],close_id,eos_id)
                    if item["kind"]=="boundary" else pair_trial(runtime,tokenizer,restored[item["case"]],item["config"]))
                row.update(item,session_id=session_id)
                seal(path,row)
                report=summarize(args.output,plan,agent.adapter,close_id,eos_id)
                print(json.dumps({"id":item["id"],"wall_s":row["wall_s"],"errors":row["errors"]}),flush=True)
                if row["errors"] or any(row["d3_counters"].get(k,0) for k in ("accounting_errors","dropped_injections")):
                    raise RuntimeError("Diagnostic error saved; inspect before continuing")
            write_json(args.output/f"sessions/{session_id}_end.json",runtime.run(idle(runtime)))
        finally:
            runtime.close()


if __name__=="__main__":
    main()
