#!/usr/bin/env python3
"""Prepare or run fixed interactive Qwen3.5-9B development validation."""
from __future__ import annotations

import argparse
import asyncio
import copy
from contextlib import ExitStack
from datetime import datetime
import fcntl
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
import uuid
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spork_core import paths
from spork_core.evaluation_artifacts import (
    archive_unfinished, complete_record, digest, freeze_plan, read_completed,
    record_path, write_json)
from spork_core.expanded_validation import (
    CONFIGS, OLD_TASKS, analyze, check_results, normalized_messages, select_tasks, verify_frozen)
from spork_core.interactive_successor import InteractiveRuntime, SuccessorInteractiveAgent, TracedLocalUser, USER_PRESENCE_PENALTY, USER_SAMPLING
from spork_core.interactive_tau2 import SpeculativeOrchestrator, environment_key
from spork_core.local_tau2_judge import LocalNLJudge
from spork_core.prefix_replay import COUNTERS, counter_delta
from spork_core.user_generation_check import prepare_usercheck, run_usercheck
from scripts.run_successor_full_eval import reset_trial_cache
from scripts.check_qwen35_experiment import local_check, server_check
from tau2.data_model.tasks import Task
from tau2.registry import registry


WARMUP_USER_TEXT = "Warmup."


def warmup_prompt_ids(agent):
    """A valid, discarded request; never mutate the real conversation state.

    Qwen3.5's chat template rejects system-only histories. A fixed user message
    warms the same policy/tool prefix for every arm without using task answers
    or generating the actual user's first message outside the measured run.
    """
    state = copy.deepcopy(agent.get_init_state())
    state.append({"role": "user", "content": WARMUP_USER_TEXT})
    prompt = agent.adapter.render_main_prompt(state, enable_thinking=True)
    return agent.tokenizer.encode(prompt, add_special_tokens=False)


def check_prompt_templates(model):
    """CPU-only preparation must exercise the actual native chat template."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    lengths = {}
    for domain in ("airline", "retail"):
        env = registry.get_env_constructor(domain)(solo_mode=False)
        agent = SuccessorInteractiveAgent(env, None, tokenizer, "spork")
        ids = warmup_prompt_ids(agent)
        if not ids or len(ids) + 1 > 40960:
            raise ValueError(f"Invalid warmup context for {domain}")
        lengths[domain] = len(ids)
    return {"warmup_prompt_tokens": lengths, "generation_requests": 0}


def protocol(args):
    selected = select_tasks(json.loads(args.task_split.read_text()), args.selection_seed)
    fresh = {domain: next(r["group_id"] for r in selected if r["domain"] == domain
                          and r["cohort"] == "new24") for domain in ("airline", "retail")}
    return {"purpose": "expanded_interactive_development", "selection_seed": args.selection_seed,
        "selection": "Previous eight IDs plus 12 SHA256-ranked new training IDs per domain; no outcome-based selection.",
        "tasks": selected, "preflight_ids": ["airline:2", "retail:3", fresh["airline"], fresh["retail"]],
        "configs": list(CONFIGS), "repetitions": args.repetitions, "seed": args.seed,
        "config_semantics": {
            "spork": "Full reasoning SPORK D1/D2/D3; no early successor",
            "raw": "Full early successor; child up to five prediction attempts; child D3 off",
            "raw_no_probe": "Same early successor, zero child predictions; formal child tool execution retained",
            "raw_one_probe": "Same early successor, at most one child prediction; no child D2 retry"},
        "max_steps": args.max_steps, "timeout_s": args.timeout_s,
        "actor_max_tokens": 8192, "actor_presence_penalty": 1.5,
        "user_max_tokens": 4096, "user_thinking": True, "user_presence_penalty": USER_PRESENCE_PENALTY,
        "user_sampling": USER_SAMPLING,
        "judge_max_tokens": 4096, "judge_thinking": False, "judge_presence_penalty": 0.,
        "tool_floor_s": 2., "model": args.model, "base_url": args.base_url,
        "model_choice": "Keep the verified 9B/v4 eager runtime for within-model expansion; Qwen3-32B is a later cross-model/architecture study.",
        "quality": "Native tau2 orchestration and ALL/strict_replay scoring. Local 9B user and local 9B NL judge, not the published simulator or judge setup.",
        "warmup_user_text": WARMUP_USER_TEXT,
        "measurement": "Fresh prefix cache per trial; one discarded token from policy/tools plus the same fixed synthetic user message outside timing. Warmup never enters the real conversation. Interaction includes user calls; Agent pipeline equals interaction minus full measured user-generation calls. Scoring follows timing; only exact judge inputs reuse sealed judgments.",
        "usercheck": "Two initial-user generations on airline:2 with the same frozen input and seed. Require natural completion, closed thinking, nonempty reply and exact repeatability before Actor runs. This does not establish full simulator fidelity.",
        "go_no_go": "Pass usercheck first. All four preflight tasks must finish naturally with continued user interaction, exact trajectories, native scoring, clean D3, and exercised successor adoption before expanded runs.",
        "scope": "32 training tasks (16 airline,16 retail); report previous8 and new24 separately. Actor is greedy; user uses declared general-thinking sampling. Seed42 repeats test reproducibility, not independent policy samples. No holdout or official leaderboard claim."}


def prepare(args, stage):
    plan = protocol(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    plan_path = args.output_root / "protocol.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Protocol changed; select a fresh SPORK_Q35_EXPANDED_TAG")
    if not plan_path.exists():
        write_json(plan_path, plan)
    selected = plan["tasks"] if stage == "expanded" else [r for r in plan["tasks"] if r["group_id"] in plan["preflight_ids"]]
    repetitions = args.repetitions if stage == "expanded" else 1
    schedule = []
    for repeat in range(repetitions):
        for i, task in enumerate(selected):
            shift = (i + repeat) % len(CONFIGS)
            order = CONFIGS[shift:] + CONFIGS[:shift]
            schedule.extend({**task, "repeat": repeat, "config": c} for c in order)
    sources = sorted(set(list(ROOT.glob("*.py")) + list((ROOT / "spork_core").rglob("*.py"))
        + list((ROOT / "scripts").glob("*.py")) + list((ROOT / "scripts").glob("*.sh"))
        + list((ROOT / "tests").glob("*.py")) + list((paths.TAU2_ROOT / "src/tau2").rglob("*.py"))))
    inputs = [p for d in ("airline", "retail") for p in
              (paths.TAU2_ROOT / "data/tau2/domains" / d).rglob("*") if p.is_file()]
    inputs += [p for p in (paths.TAU2_ROOT / "data/tau2/user_simulator").rglob("*") if p.is_file()]
    inputs += [args.task_split, plan_path]
    parser_root = Path(importlib.util.find_spec("vllm").origin).parent
    parser_sources = sorted((parser_root / "parser").rglob("*.py"))
    manifest = {**plan, "created_at": datetime.now().astimezone().isoformat(), "stage": stage,
        "tasks": selected, "schedule": schedule, "repetitions": repetitions,
        "task_split": str(args.task_split), "max_errors": 5,
        "model_family": "qwen35", "require_identical_trajectories": True,
        "serving_profile": "qwen35_9b_gdn_reference_v4",
        "software_versions": {n: importlib.metadata.version(n) for n in
             ("vllm", "torch", "transformers", "tokenizers", "aiohttp", "pydantic")},
        "code_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources},
        "input_sha256": {str(p.resolve()): digest(p) for p in sorted(set(inputs))},
        "external_parser_root": str(parser_root),
        "external_parser_source_sha256": {str(p.relative_to(parser_root)): digest(p) for p in parser_sources},
        "model_config_sha256": {n: digest(Path(args.model) / n) for n in
            ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json") if (Path(args.model) / n).is_file()},
        "absent_optional_model_files": [n for n in ("generation_config.json",) if not (Path(args.model) / n).is_file()]}
    output = args.output_root / stage
    frozen = freeze_plan(output, manifest, ROOT, sources)
    for source in parser_sources:
        target = output / "external_code/vllm" / source.relative_to(parser_root)
        if target.exists() and digest(target) != digest(source):
            raise ValueError("External parser snapshot changed")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
    # Validate all task schemas and native scoring bases without generating text.
    task_map = {}
    for d in ("airline", "retail"):
        raw = json.loads((paths.TAU2_ROOT / f"data/tau2/domains/{d}/tasks.json").read_text())
        task_map.update({f'{d}:{r["id"]}': Task.model_validate(r) for r in raw})
    for item in selected:
        task = task_map[item["group_id"]]
        bases = {x.value for x in task.evaluation_criteria.reward_basis}
        if bases - {"DB", "ENV_ASSERTION", "ACTION", "COMMUNICATE", "NL_ASSERTION"}:
            raise ValueError(f"Unsupported native reward basis: {bases}")
    return frozen, task_map


def idle(runtime):
    status = runtime.run(runtime.http.status(runtime.session))
    if (status.get("active_requests") != 0 or status.get("pending_verifications") != 0
            or any(k not in status for k in COUNTERS)):
        raise RuntimeError("Dedicated server has active or unverified work")
    return status


def run_stage(args, manifest, tasks):
    output = args.output_root / manifest["stage"]
    verify_frozen(output, manifest, ROOT)
    report = analyze(output, manifest)
    if report["gate_errors"]:
        raise RuntimeError(f'Retained failure/divergence requires inspection: {report["gate_errors"]}')
    if report["complete"]:
        return check_results(output, ROOT)
    server_check()
    from loguru import logger
    from transformers import AutoTokenizer
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    runtime = InteractiveRuntime(args.base_url, args.model)
    session_id = uuid.uuid4().hex
    try:
        initial_status = idle(runtime)
        write_json(output / "server_sessions" / (session_id + "_start.json"), initial_status)
        for index, item in enumerate(manifest["schedule"]):
            path = record_path(output, item)
            if read_completed(path, item) is not None:
                continue
            for f, h in manifest["input_sha256"].items():
                if digest(f) != h:
                    raise ValueError(f"Input changed during the run: {f}")
            record = {**item, "session_id": session_id, "turns": [], "errors": [],
                "user_calls": [], "judge_calls": [], "reward_info": None,
                "previous_unsealed": archive_unfinished(path)}
            partial = output / "attempts" / (path.stem + "_" + session_id + ".json")
            record["attempt_path"] = str(partial)
            task = tasks[item["group_id"]]
            env = registry.get_env_constructor(item["domain"])(solo_mode=False)
            agent = SuccessorInteractiveAgent(env, runtime, tokenizer, item["config"],
                seed=args.seed, max_tokens=8192, floor_s=2.)
            user = TracedLocalUser(runtime, tokenizer, str(task.user_scenario),
                                  max_tokens=4096, thinking=True)
            orch = SpeculativeOrchestrator(domain=item["domain"], agent=agent, user=user,
                environment=env, task=task, max_steps=args.max_steps, max_errors=5,
                seed=args.seed, validate_communication=True, timeout=args.timeout_s)
            agent.orchestrator = orch
            judge = LocalNLJudge(runtime, tokenizer, output / "judge_cache", seed=args.seed)

            def checkpoint(entry):
                record.update(turns=agent.turns, user_calls=user.calls, prefetch=agent.prefetch)
                write_json(partial, record)
                write_json(output / "in_progress.json", record)
                print(f'  actor {entry["turn"]}: {entry.get("baseline_tool_name") or "reply"}', flush=True)

            agent.on_turn = checkpoint
            idle(runtime)
            record["prefix_cache_reset"] = runtime.run(reset_trial_cache(runtime.session, runtime.http))
            ids = warmup_prompt_ids(agent)
            runtime.run(runtime.http.complete(runtime.session, ids, max_tokens=1, seed=args.seed))
            before = idle(runtime)
            if before.get("server_config") != initial_status.get("server_config"):
                raise RuntimeError("Serving identity/configuration changed within the session")
            record["stage"] = "interaction"
            write_json(partial, record)
            print(f'[{index+1}/{len(manifest["schedule"])}] {manifest["stage"]} {item["group_id"]} r{item["repeat"]} {item["config"]}', flush=True)
            started = time.perf_counter()
            simulation = None
            interrupted = None
            try:
                simulation = orch.run()
                record["termination_reason"] = simulation.termination_reason.value
            except (KeyboardInterrupt, asyncio.CancelledError) as exc:
                interrupted = exc
                record["errors"].append(f"interrupted: {type(exc).__name__}")
            except Exception as exc:
                record["errors"].append(f"interaction: {type(exc).__name__}: {exc}")
                record["traceback"] = traceback.format_exc()
                if getattr(exc, "partial_generation", None) is not None:
                    record["failed_generation"] = exc.partial_generation
            finally:
                try:
                    agent.close()
                except Exception as exc:
                    record["errors"].append(f"cleanup: {type(exc).__name__}: {exc}")
                record["wall_s"] = time.perf_counter() - started
            record.update(turns=agent.turns, user_calls=user.calls, prefetch=agent.prefetch,
                abandoned_entries=agent.abandoned_entries, cleanup_errors=agent.cleanup_errors,
                final_environment=list(environment_key(orch.environment)))
            record["errors"].extend(agent.cleanup_errors)
            record["errors"].extend(e for pair in agent.prefetch["pairs"] for e in pair.get("errors", []))
            record["actor_wall_s"] = sum(t["actor_step_wall_s"] for t in agent.turns) + agent.close_wall_s
            record["user_wall_s"] = sum(c["participant_wall_s"] for c in user.calls)
            record["agent_pipeline_s"] = record["wall_s"] - record["user_wall_s"]
            if record["agent_pipeline_s"] < 0:
                record["errors"].append("negative Agent pipeline time")
            trajectory = (simulation.model_dump(mode="json") if simulation is not None else
                          {"partial": True, "messages": [m.model_dump(mode="json") for m in orch.trajectory]})
            record["trajectory_path"] = "trajectories/" + path.name
            write_json(output / record["trajectory_path"], trajectory)
            record["trajectory_sha256"] = digest(output / record["trajectory_path"])
            record["normalized_messages"] = normalized_messages(trajectory["messages"])
            if interrupted is not None:
                record.update(stage="interrupted", interrupted_at=datetime.now().astimezone().isoformat())
                write_json(partial, record)
                write_json(output / "in_progress.json", record)
                raise interrupted
            # Judge calls and deterministic native replay are outside measured interaction latency.
            record["stage"] = "evaluation"
            eval_start = time.perf_counter()
            if simulation is not None and not record["errors"]:
                try:
                    record["reward_info"] = judge.evaluate(simulation, task, item["domain"]).model_dump(mode="json")
                except Exception as exc:
                    record["errors"].append(f"evaluation: {type(exc).__name__}: {exc}")
                    record["evaluation_traceback"] = traceback.format_exc()
            record["evaluation_wall_s"] = time.perf_counter() - eval_start
            record["judge_calls"] = judge.calls
            try:
                after = idle(runtime)
                record["d3_counters"] = counter_delta(before, after)
                if any(v < 0 for v in record["d3_counters"].values()):
                    raise RuntimeError("D3 counters reset during a trial")
            except Exception as exc:
                record["errors"].append(f"server_integrity: {exc}")
            record.update(stage="completed", completed_at=datetime.now().astimezone().isoformat())
            write_json(partial, record)
            complete_record(path, record)
            report = analyze(output, manifest)
            print(json.dumps({"agent_pipeline_s": record["agent_pipeline_s"],
                "interaction_s": record["wall_s"], "reward": (record["reward_info"] or {}).get("reward"),
                "termination": record.get("termination_reason"), "errors": record["errors"]}), flush=True)
            if report["gate_errors"]:
                raise RuntimeError(f'Sealed outcome needs inspection; stopping: {report["gate_errors"]}')
        write_json(output / "server_sessions" / (session_id + "_end.json"), idle(runtime))
    finally:
        runtime.close()
    return check_results(output, ROOT)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--stage", choices=["prepare", "usercheck", "preflight", "expanded", "all", "check"], default="prepare")
    p.add_argument("--model", default=os.environ.get("SPORK_CODEX_MODEL", str(ROOT.parents[1] / "Models/Qwen3.5-9B")))
    p.add_argument("--base-url", default="http://127.0.0.1:8027/v1")
    p.add_argument("--task-split", type=Path, default=ROOT / "prefix_data/datasets/controller_stage3_20261008_213013_wi87e7du/task_split.json")
    p.add_argument("--selection-seed", type=int, default=20261101)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--timeout-s", type=float, default=3600.)
    p.add_argument("--run", action="store_true")
    args = p.parse_args()
    args.output_root = args.output_root.resolve()
    if os.environ.get("SPORK_Q35_REFERENCE_GDN") != "1":
        p.error("Use scripts/run_qwen35_expanded.sh with the verified reference-GDN profile")
    if min(args.repetitions, args.max_steps, args.timeout_s) <= 0:
        p.error("Positive experiment budgets required")
    if args.stage == "check":
        print(json.dumps(check_results(args.output_root / "expanded", ROOT), indent=2))
        return
    template_check = check_prompt_templates(args.model)
    print(json.dumps(template_check, ensure_ascii=False), flush=True)
    # Always freeze both plans before starting; expansion cannot silently change
    # tasks or settings after seeing preflight results.
    plans = {stage: prepare(args, stage) for stage in ("preflight", "expanded")}
    user_spec = prepare_usercheck(args, plans["preflight"][1])
    print(json.dumps({"output": str(args.output_root), "stage": args.stage, "usercheck_runs": 2,
        "preflight_runs": len(plans["preflight"][0]["schedule"]),
        "expanded_runs": len(plans["expanded"][0]["schedule"]), "model": args.model,
        "generation_requests": "enabled" if args.run else 0}, ensure_ascii=False), flush=True)
    if not args.run or args.stage == "prepare":
        local_check()
        return
    with ExitStack() as stack:
        port = urlparse(args.base_url).port or 80
        for sibling in sorted(ROOT.parent.glob("spork*")):
            if (sibling / "spork_core").is_dir():
                lock = sibling / "results" / f"controller_online_{port}.lock"
                lock.parent.mkdir(parents=True, exist_ok=True)
                fcntl.flock(stack.enter_context(lock.open("a+")).fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        # A user-generation failure must not be discovered by repeatedly
        # starting the entire interactive experiment.
        verify_frozen(args.output_root / "preflight", plans["preflight"][0], ROOT)
        print(json.dumps(run_usercheck(args, user_spec, plans["preflight"][1],
            idle=idle, reset_cache=reset_trial_cache, check_server=server_check), ensure_ascii=False), flush=True)
        if args.stage in {"preflight", "all"}:
            run_stage(args, *plans["preflight"])
        if args.stage in {"expanded", "all"}:
            check_results(args.output_root / "preflight", ROOT)
            run_stage(args, *plans["expanded"])


if __name__ == "__main__":
    main()
