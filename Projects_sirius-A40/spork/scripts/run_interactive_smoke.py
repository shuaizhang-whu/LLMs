#!/usr/bin/env python3
"""Manual local pilot: SPORK turns inside the official tau2 dialogue loop.

This is a diagnostic on previously inspected training tasks. The same local
Qwen model supplies Actor, user simulation and the assertion judge; its scores
are not comparable to published tau2 results using different model settings.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
import fcntl
import hashlib
from importlib.metadata import version
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback
from urllib.parse import urlparse
from urllib.request import ProxyHandler, build_opener
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# The user simulator uses the OpenAI-compatible SDK through LiteLLM. Keep
# loopback requests local even when the shell has HTTP(S)_PROXY configured.
_local_no_proxy = ",".join(filter(None, [os.environ.get("NO_PROXY", ""),
                                       os.environ.get("no_proxy", ""), "127.0.0.1,localhost,::1"]))
os.environ["NO_PROXY"] = os.environ["no_proxy"] = _local_no_proxy
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

from spork_core.early_exit import ExitConfig, run_early_exit_turn
from spork_core.eval_artifacts import archive_unsealed, digest, dump, freeze_manifest, load_sealed, seal_record
from spork_core.paths import QWEN3_32B_MODEL, TAU2_ROOT, add_runtime_paths
from spork_core.prefix_replay import PrefixHttpClient, counter_delta
from spork_core.qwen import QwenToolCallAdapter
from spork_core.timed_retry import run_timed_retry_turn
from spork_core.turn_runner import RetryTurnConfig
from spork_core.vllm_client import VllmClient

add_runtime_paths()

from scripts.interactive_tau2_bridge import CommittedToolEnvironment, InteractiveToolExecutor, SporkInteractiveAgent
from spork_core.benchmarks.tau2 import get_environment, load_system_prompt, load_tasks, load_tools
from tau2.data_model.simulation import TerminationReason
from tau2.data_model.message import UserMessage
from tau2.data_model.tasks import RewardType, Task
from tau2.evaluator import evaluator_nl_assertions as nl_module
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.user.user_simulator import UserSimulator
from tau2.user.user_simulator_base import UserError

NORMAL_ENDS = {TerminationReason.USER_STOP, TerminationReason.AGENT_STOP}
METHODS = ("spork", "eta", "eta_retarget", "decision_retarget", "reuse_retarget",
           "eta_retarget_forced_reuse", "eta_retarget_deferred_reuse",
           "eta_retarget_deferred_reuse_fast", "eta_retarget_deferred_reuse_adaptive")


def local_llm_args(base_url, max_tokens):
    return {"api_base": base_url, "api_key": "EMPTY", "temperature": 0.0,
            "seed": 42, "max_tokens": max_tokens, "timeout": 900, "num_retries": 0,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}


class CheckedUserSimulator(UserSimulator):
    """Keep the official user policy while rejecting incomplete user replies."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.wall_s = []
        self.last_response = None

    def generate_next_message(self, message, state):
        started = time.perf_counter()
        try:
            reply, state = super().generate_next_message(message, state)
            self.last_response = reply.model_dump(mode="json")
            choices = (reply.raw_data or {}).get("choices") or []
            if not choices or choices[0].get("finish_reason") != "stop":
                raise UserError("User simulation did not finish normally")
            return reply, state
        finally:
            self.wall_s.append(time.perf_counter() - started)


@contextmanager
def local_assertion_judge(model, kwargs, traces, *, trajectory=None, assertions=()):
    """Route the official judge prompt only to the explicitly configured server.

    This process runs one simulation at a time. Restore the module hook even
    on errors, retain raw judge output, and never use the upstream cloud default.
    """
    original = nl_module.generate

    def generate_local(**request):
        if trajectory is not None:
            # The upstream judge renders only message.content, which omits
            # structured tool names/arguments. Give this local diagnostic the
            # complete public trajectory, without raw model data or reasoning.
            public_messages = [message.model_dump(mode="json", exclude_none=True,
                include={"role", "content", "tool_calls", "id", "requestor", "error"})
                for message in trajectory]
            prompt = ("conversation (ordered public messages, including committed tool calls):\n"
                      + json.dumps(public_messages, ensure_ascii=False)
                      + "\n\nexpectedOutcomes:\n" + json.dumps(list(assertions), ensure_ascii=False))
            request["messages"] = [request["messages"][0], UserMessage(role="user", content=prompt)]
        request.update(model="openai/" + model, **kwargs)
        reply = original(**request)
        traces.append(reply.model_dump(mode="json"))
        choices = (reply.raw_data or {}).get("choices") or []
        if not choices or choices[0].get("finish_reason") != "stop":
            raise ValueError("Assertion judge did not finish normally")
        return reply

    nl_module.generate = generate_local
    try:
        yield
    finally:
        nl_module.generate = original


def evaluate_local(simulation, task, domain, model, base_url):
    traces = []
    started = time.perf_counter()
    criteria = task.evaluation_criteria
    assertions = list(criteria.nl_assertions or []) if criteria else []
    diagnostic = {"expected": len(assertions), "all_met": None,
                  "included_in_reward": bool(criteria and RewardType.NL_ASSERTION in criteria.reward_basis)}
    try:
        judge_args = local_llm_args(base_url, 4096)
        judge_args["response_format"] = {"type": "json_object"}
        with local_assertion_judge(model, judge_args, traces,
                                   trajectory=simulation.messages, assertions=assertions):
            # Airline tasks can have NL assertions outside their reward_basis.
            # Evaluate them as diagnostics while retaining the configured score.
            reward = evaluate_simulation(simulation, task, EvaluationType.ALL_WITH_NL_ASSERTIONS, False,
                                         domain, strict_replay=True)
        if simulation.termination_reason in NORMAL_ENDS and criteria:
            expected = Counter(assertions)
            received = Counter(check.nl_assertion for check in reward.nl_assertions or [])
            if expected != received:
                raise ValueError("Judge omitted, duplicated or changed required assertions")
            if assertions:
                diagnostic["all_met"] = all(check.met for check in reward.nl_assertions)
        return {"reward_info": reward.model_dump(mode="json"), "error": None,
                "assertion_diagnostic": diagnostic, "judge_responses": traces,
                "wall_s": time.perf_counter() - started}
    except Exception as error:
        return {"reward_info": None, "error": f"{type(error).__name__}: {error}",
                "assertion_diagnostic": diagnostic, "judge_responses": traces,
                "wall_s": time.perf_counter() - started}


def exit_config(method, reference="conservative"):
    schedule = "ready_fast" if method.endswith("_adaptive") else "fixed_fast" if method.endswith("_fast") else "standard"
    method = method.removesuffix("_adaptive").removesuffix("_fast")
    deferred = method.endswith("_deferred_reuse")
    reuse_forced = deferred or method.endswith("_forced_reuse")
    method = method.removesuffix("_deferred_reuse").removesuffix("_forced_reuse")
    config = ExitConfig(method=method.removesuffix("_retarget"), stable_checks=3,
                      max_checks=15 if reference == "paper" else 8, guard_policy=reference,
                      retarget_after_stable=method.endswith("_retarget"), probe_draft_reuse_mode="prefix",
                      reuse_forced_probe=reuse_forced, defer_forced_checks=deferred,
                      check_schedule=schedule, fast_check_interval=16)
    config.validate()
    return config


def preflight(args):
    opener = build_opener(ProxyHandler({}))
    root = args.base_url.removesuffix("/v1")
    with opener.open(root + "/health", timeout=5):
        pass
    with opener.open(args.base_url + "/models", timeout=5) as response:
        models = json.load(response)
    with opener.open(root + "/spork/status", timeout=5) as response:
        status = json.load(response)
    if args.model not in {item["id"] for item in models["data"]}:
        raise ValueError("Served model differs from the frozen model path")
    if (not status.get("has_drafter") or status.get("active_requests") != 0
            or status.get("pending_verifications") != 0
            or status.get("server_config", {}).get("batch_invariant_requested") is not True):
        raise ValueError("An idle batch-invariant D3 server is required")
    return status


def prepare(args):
    reference = getattr(args, "eta_reference", "conservative")
    for method in args.modes:
        if method != "spork":
            exit_config(method, reference)
    split = json.loads(args.task_split.read_text())
    train = {t["group_id"] for t in split["tasks"] if t["split"] == "train"}
    if not set(args.tasks) <= train:
        raise ValueError("This pilot only accepts training tasks from the frozen split")
    tasks = {f"{d}:{t['id']}": Task(**t) for d in ("airline", "retail") for t in load_tasks(d)}
    for key in args.tasks:
        if tasks[key].initial_state is not None:
            raise ValueError("This first interactive pilot requires tasks without initial history")
    schedule = []
    for repeat in range(args.repeats):
        for index, group in enumerate(args.tasks):
            shift = (index + repeat) % len(args.modes)
            for method in args.modes[shift:] + args.modes[:shift]:
                domain, task_id = group.split(":")
                schedule.append(dict(group_id=group, domain=domain, task_id=task_id, method=method, repeat=repeat))
    source_paths = [Path(__file__), ROOT / "scripts/interactive_tau2_bridge.py",
                    ROOT / "scripts/run_interactive_smoke_a40.sh", ROOT / "scripts/spork_env.sh",
                    ROOT / "scripts/run_forced_reuse_eval_a40.sh",
                    ROOT / "launch_vllm_spork_http.py", ROOT / "scripts/serve_qwen_a40.sh",
                    ROOT / "tau2_tool_parser.py", *sorted((ROOT / "spork_core").rglob("*.py")),
                    *sorted((TAU2_ROOT / "src/tau2").rglob("*.py"))]
    sources = {("tau2-bench/" + str(p.relative_to(TAU2_ROOT)) if p.is_relative_to(TAU2_ROOT)
                else str(p.relative_to(ROOT))): p for p in source_paths}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as snapshot:
        for name, path in sorted(sources.items()):
            member = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            snapshot.writestr(member, path.read_bytes())
    archive = buffer.getvalue()
    data = [p for d in ("domains/airline", "domains/retail", "user_simulator")
            for p in (TAU2_ROOT / "data/tau2" / d).rglob("*")
            if p.is_file() and p.suffix in {".json", ".toml", ".md"}]
    manifest = {"purpose": "interactive_tau2_quality_pilot", "created_at": datetime.now().isoformat(),
                "model": args.model, "base_url": args.base_url, "seed": 42,
                "tasks": args.tasks, "schedule": schedule, "repeats": args.repeats, "max_steps": args.max_steps,
                "max_tokens": args.max_tokens, "tool_floor_s": args.tool_floor,
                "actor_sampling": {"temperature": .6, "top_p": .95, "top_k": 20},
                "user_model": "openai/" + args.model, "user_args": local_llm_args(args.base_url, 2048),
                "judge_model": "openai/" + args.model, "judge_max_tokens": 4096,
                "eta_reference": reference,
                "exit_configs": {m: asdict(exit_config(m, reference)) for m in args.modes if m != "spork"},
                "task_selection": "Previously inspected train tasks selected to diagnose loops and quality; not a generalization sample",
                "quality_scope": "Official dialogue and reward components with a local Qwen user and judge; requires manual transcript review",
                "assertion_scope": "Check all dataset NL assertions, including those outside reward_basis; preserve the configured reward and report assertions separately. Judge sees ordered public messages with committed tool calls.",
                "timing_scope": "Simulation includes user generation; Actor callback sums exclude user and judge; quality evaluation timed separately",
                "monetary_cost_scope": "Not measured; upstream library may emit zero for an unpriced local model. Do not interpret those fields as resource cost.",
                "tool_protocol": "Official full tool responses, canonical history, isolated predictions, single canonical commit",
                "versions": {n: version(n) for n in ("torch", "transformers", "vllm", "litellm")},
                "task_split_sha256": digest(args.task_split),
                "model_config_sha256": {n: digest(Path(args.model) / n) for n in
                    ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json")},
                "data_sha256": {str(p.relative_to(TAU2_ROOT)): digest(p) for p in data},
                "code_sha256": {n: digest(p) for n, p in sources.items()},
                "code_archive_sha256": hashlib.sha256(archive).hexdigest()}
    if args.resume and not (args.output / "manifest_sha256.json").is_file():
        raise ValueError("Resume requires an existing sealed manifest")
    # The shell launcher creates run.log before this process prepares the run.
    if not args.resume and args.output.exists():
        artifacts = {p.name for p in args.output.iterdir()} - {"run.log", "exit_code.txt"}
        if artifacts:
            raise FileExistsError("Output already contains experiment data; use --resume or a new directory")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = freeze_manifest(args.output, manifest, resume=args.resume)
    code_path = args.output / "code_snapshot.zip"
    if args.resume:
        if digest(code_path) != manifest["code_archive_sha256"]:
            raise ValueError("Frozen code snapshot was changed")
    else:
        code_path.write_bytes(archive)
        (args.output / "task_split.json").write_bytes(args.task_split.read_bytes())
    return manifest, tasks


def summarize(records):
    methods = {}
    for method in sorted({r["method"] for r in records}):
        rows = [r for r in records if r["method"] == method]
        rewards = [r["quality"]["reward_info"]["reward"] for r in rows
                   if r.get("quality", {}).get("reward_info") is not None]
        assertions = [r.get("quality", {}).get("assertion_diagnostic", {}).get("all_met") for r in rows]
        methods[method] = {"runs": len(rows), "runtime_failures": sum(bool(r["errors"]) for r in rows),
                           "normal_dialogue_ends": sum(r["termination_reason"] in NORMAL_ENDS for r in rows),
                           "quality_evaluated": len(rewards), "local_reward_passes": sum(v == 1 for v in rewards),
                           "assertions_evaluated": sum(v is not None for v in assertions),
                           "assertions_passes": sum(v is True for v in assertions)}
    return {"methods": methods, "records": [
        {k: r.get(k) for k in ("group_id", "method", "wall_s", "actor_callback_s", "user_s", "termination_reason", "errors")}
        | {"local_reward": (r.get("quality", {}).get("reward_info") or {}).get("reward"),
           "local_assertions_pass": r.get("quality", {}).get("assertion_diagnostic", {}).get("all_met")}
        for r in records], "scope": "Diagnostic interactive pilot; do not compare its latency directly to fixed-user experiments"}


def run(args, manifest, tasks):
    import aiohttp
    from transformers import AutoTokenizer
    from loguru import logger
    logger.remove()
    # The local model has no API price. Suppress only this known bookkeeping
    # message, retaining all generation, parsing, tool, and evaluator errors.
    def log_filter(record):
        return not (record["name"] == "tau2.utils.llm_utils"
                    and record["function"] == "get_response_cost"
                    and "This model isn't mapped yet." in record["message"]
                    and f"model={args.model}," in record["message"])
    logger.add(sys.stderr, level="ERROR", filter=log_filter)
    print("本地模型未配置 API 价格：费用不统计；结果中的 cost=0 不代表实际资源成本为零。", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    adapters = {d: QwenToolCallAdapter(tok, load_tools(d)) for d in ("airline", "retail")}
    client = VllmClient(args.base_url, args.model, 900)
    http = PrefixHttpClient(args.base_url, args.model, 900)
    records = []

    async def make_session():
        return aiohttp.ClientSession(trust_env=False, connector=aiohttp.TCPConnector(force_close=True))

    with asyncio.Runner() as loop:
        session = loop.run(make_session())
        try:
            for index, item in enumerate(manifest["schedule"]):
                suffix = f"_repeat{item['repeat'] + 1}" if item["repeat"] else ""
                path = args.output / "records" / f"{item['domain']}_{item['task_id']}_{item['method']}{suffix}.json"
                sealed = load_sealed(path)
                if sealed is not None:
                    records.append(sealed)
                    continue
                archive_unsealed(path)
                before = preflight(args)
                task, adapter = tasks[item["group_id"]], adapters[item["domain"]]
                executor = InteractiveToolExecutor(get_environment(item["domain"]), floor_s=args.tool_floor)

                async def actor_turn(messages, tool_executor, turn):
                    prompt = adapter.render_main_prompt(messages, enable_thinking=True)
                    plen = len(tok.encode(prompt, add_special_tokens=False))
                    rid = "interactive-" + uuid.uuid4().hex

                    async def register(call, _prompt):
                        return await http.set_draft(session, rid, plen, call, tok)

                    async def clear():
                        cleared = await http.clear(session)
                        return {**await http.status(session), "clear_status": cleared["status"]}

                    cfg = RetryTurnConfig(max_retries=5, confidence_threshold=.9, retry_token_step=50,
                        probe_max_tokens=160, baseline_max_tokens=args.max_tokens, context_window_tokens=40960,
                        enable_thinking=True, seed=42, d3_set=register, d3_clear=clear, main_request_id=rid,
                        launch_mode="original", actor_sampling=manifest["actor_sampling"], canonical_tool_history=True)
                    if item["method"] == "spork":
                        entry = await run_timed_retry_turn(session, client, adapter, messages, tool_executor, turn, cfg)
                    else:
                        entry = await run_early_exit_turn(session, client, adapter, messages, tool_executor, turn, cfg,
                                                          exit_config(item["method"], getattr(args, "eta_reference", "conservative")))
                    entry["input_prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
                    dump(args.output / "in_progress.json", {**item, "turns": agent.turns + [entry],
                         "public_messages": [m.model_dump(mode="json") for m in orchestrator.get_messages()]})
                    return entry

                agent = SporkInteractiveAgent(executor, load_system_prompt(item["domain"]),
                            lambda messages, tools, turn: loop.run(actor_turn(messages, tools, turn)))
                user = CheckedUserSimulator(llm="openai/" + args.model, instructions=str(task.user_scenario),
                                            llm_args=local_llm_args(args.base_url, 2048))
                orchestrator = Orchestrator(domain=item["domain"], agent=agent, user=user,
                    environment=CommittedToolEnvironment(executor), task=task, seed=42, max_steps=args.max_steps)
                print(f"[{index+1}/{len(manifest['schedule'])}] {item['group_id']} {item['method']} "
                      f"repeat={item['repeat'] + 1}/{args.repeats}", flush=True)
                started = time.perf_counter()
                record = {**item, "errors": [], "simulation": None, "quality": {},
                          "server_status_before": before}
                try:
                    simulation = orchestrator.run()
                    record["simulation"] = simulation.model_dump(mode="json")
                except Exception as error:
                    record["errors"].append(f"{type(error).__name__}: {error}")
                    record["traceback"] = traceback.format_exc()
                    record["partial_messages"] = [m.model_dump(mode="json") for m in orchestrator.get_messages()]
                    record["last_user_response"] = user.last_response
                record.update(wall_s=time.perf_counter() - started, turns=agent.turns, failed_turn=agent.failed_turn,
                              actor_callback_s=sum(agent.turn_wall_s), user_s=sum(user.wall_s),
                              termination_reason=orchestrator.termination_reason.value if orchestrator.termination_reason else None,
                              final_state={"agent_db": executor.env.get_db_hash(), "user_db": executor.env.get_user_db_hash()})
                try:
                    after = loop.run(http.status(session))
                    record["d3_counters"] = delta = counter_delta(before, after)
                    record["server_status_after"] = after
                    if any(delta.get(k) for k in ("accounting_errors", "dropped_injections")):
                        record["errors"].append("D3 accounting or dropped injection")
                    if after.get("active_requests") != 0 or after.get("pending_verifications") != 0:
                        record["errors"].append("Residual server requests or verifications")
                except Exception as error:
                    record["errors"].append(f"Server status: {error}")
                if record["simulation"] is not None and not record["errors"]:
                    record["quality"] = evaluate_local(simulation, task, item["domain"], args.model, args.base_url)
                seal_record(path, record)
                records.append(record)
                dump(args.output / "summary.json", summarize(records))
                reward = (record["quality"].get("reward_info") or {}).get("reward")
                print(f"  {record['wall_s']:.2f}s; {len(agent.turns)} Actor turns; local_reward={reward}; "
                      f"errors={record['errors']}; quality_error={record['quality'].get('error')}", flush=True)
                length_only = (len(record["errors"]) == 1
                               and ((agent.failed_turn or {}).get("partial_actor") or {}).get("finish_reason") == "length")
                if record["errors"] and not length_only:
                    raise RuntimeError("Failure sealed; inspect it before resuming")
            dump(args.output / "summary.json", summarize(records))
        finally:
            loop.run(session.close())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=str(QWEN3_32B_MODEL))
    parser.add_argument("--base-url", default="http://127.0.0.1:8017/v1")
    parser.add_argument("--task-split", type=Path, default=ROOT / "prefix_data/datasets/controller_stage3_20261008_213013_wi87e7du/task_split.json")
    parser.add_argument("--tasks", nargs="+", default=["airline:2", "airline:48", "retail:39", "retail:76"])
    parser.add_argument("--modes", nargs="+", choices=METHODS, default=["spork", "eta", "eta_retarget"])
    parser.add_argument("--repeats", type=int, default=1, help="Repeat the paired schedule with balanced order and the same Actor seed")
    parser.add_argument("--eta-reference", choices=("conservative", "paper"), default="conservative",
                        help="paper: 15 total forced/free probes, equal calls plus latest free probability")
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--tool-floor", type=float, default=2.)
    preparation = parser.add_mutually_exclusive_group()
    preparation.add_argument("--prepare-only", action="store_true")
    preparation.add_argument("--check-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.eta_reference == "paper" and set(args.modes) - {"spork", "eta", "eta_retarget", "eta_retarget_forced_reuse", "eta_retarget_deferred_reuse", "eta_retarget_deferred_reuse_fast", "eta_retarget_deferred_reuse_adaptive"}:
        parser.error("Paper guards require consecutive verified forced calls plus the latest free probability")
    args.model = str(Path(args.model).resolve())
    args.base_url = args.base_url.rstrip("/")
    parsed = urlparse(args.base_url)
    if parsed.hostname not in {"localhost", "127.0.0.1", "::1"} or parsed.path != "/v1":
        parser.error("This pilot requires an explicitly configured local /v1 server")
    if (not 1 <= len(args.tasks) <= 8 or len(set(args.tasks)) != len(args.tasks)
            or len(set(args.modes)) != len(args.modes) or len(args.modes) < 2 or not 1 <= args.repeats <= 10
            or not 1 <= args.max_steps <= 100 or not 1 <= args.max_tokens <= 8192 or args.tool_floor < 0):
        parser.error("Invalid small-pilot configuration")
    if args.prepare_only:
        manifest, _ = prepare(args)
        print(f"Prepared {len(manifest['schedule'])} runs; no model requests sent")
        return
    lock_path = ROOT / "results" / f"controller_online_{parsed.port or 80}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.exit(2, "Another SPORK experiment owns this server; no run started.\n")
        preflight(args)
        if args.check_only:
            print("Local model service and experiment lock are available; no generation requested")
            return
        manifest, tasks = prepare(args)
        run(args, manifest, tasks)


if __name__ == "__main__":
    main()
