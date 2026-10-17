"""HotPotQA 可重复 baseline 与保守在线投机执行器。

该执行器与 ``runner.py`` 的历史入口并存：旧入口继续用于核验仓库附带结果，
本模块则为新实验提供固定样本、隔离的 paired 轨迹、逐角色 token 统计和真正
并发但语义保守的 online 模式。
"""

from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
import random
import re
import statistics
import time
from pathlib import Path

from . import constants, environment, wrappers
from .llm_client import LLMClient
from .metrics import Metrics
from .prompts import PromptTemplates
from .utils import Utils


FIXED_SAMPLE_IDS = [
    7107,
    5619,
    373,
    6904,
    1267,
    4202,
    2316,
    6,
    1248,
    7356,
    1477,
    1354,
    7090,
    3021,
    2038,
    4349,
    2885,
    2861,
    4948,
    2527,
]


def parse_sample_ids(value):
    """解析逗号分隔样本编号，并在保持顺序的同时拒绝重复与越界。"""
    if value is None:
        return None
    ids = []
    seen = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        index = int(item)
        if not 0 <= index < constants.num:
            raise ValueError(f"sample id {index} is outside [0, {constants.num - 1}]")
        if index in seen:
            raise ValueError(f"duplicate sample id: {index}")
        ids.append(index)
        seen.add(index)
    if not ids:
        raise ValueError("--sample-ids must contain at least one integer")
    return ids


def _slug(value):
    """把模型名或 run id 转成不会逃逸实验目录的短名称。"""
    value = value.strip().replace("/", "_")
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value)
    return value.strip("._-") or "unnamed"


def _mean(values):
    return statistics.fmean(values) if values else None


class ReproductionRunner:
    """运行 sequential、paired 或保守 online HotPotQA 实验。"""

    def __init__(
        self,
        model_name,
        guess_model_name,
        execution_mode,
        sample_ids=None,
        seed=248,
        top_k=3,
        run_id=None,
        num_steps=8,
        temperature=None,
        authority_trace_root=None,
        to_print_output=True,
    ):
        if execution_mode not in {"sequential", "paired", "online"}:
            raise ValueError(f"unsupported execution mode: {execution_mode}")
        if top_k not in {1, 3}:
            raise ValueError("top_k must be 1 or 3")
        if num_steps < 2:
            raise ValueError("num_steps must be at least 2")

        self.model_name = model_name
        self.guess_model_name = guess_model_name
        self.execution_mode = execution_mode
        self.seed = int(seed)
        self.top_k = int(top_k)
        self.num_steps = int(num_steps)
        self.temperature = constants.temperature if temperature is None else float(temperature)
        self.sample_ids = list(sample_ids) if sample_ids is not None else self._default_sample_ids()
        self.run_id = _slug(run_id or time.strftime("manual_%Y%m%d_%H%M%S"))
        self.to_print_output = to_print_output
        self.current_index = None
        self.authority_trace_root = (
            Path(authority_trace_root).expanduser().resolve()
            if authority_trace_root
            else None
        )

        self.actor_llm = LLMClient(
            model_name=model_name,
            temperature=self.temperature,
            max_tokens=constants.max_output_tokens,
            top_p=constants.top_p,
            seed=self.seed,
        )
        self.candidate_llm = None
        self.observation_llm = None
        if execution_mode != "sequential":
            self.candidate_llm = LLMClient(
                model_name=model_name,
                temperature=self.temperature,
                max_tokens=constants.max_output_tokens,
                top_p=constants.top_p,
                seed=self.seed,
            )
            self.observation_llm = LLMClient(
                model_name=guess_model_name,
                temperature=constants.guess_temperature,
                max_tokens=constants.max_guess_output_tokens,
                top_p=constants.guess_top_p,
                seed=self.seed,
            )

        self.env = self._get_env()
        self.sim_env = (
            self._get_env(guess_llm=self.observation_llm)
            if execution_mode == "paired"
            else None
        )
        self.base_traj_path = str(
            Path("reproduction")
            / "experiments"
            / self.run_id
            / f"{execution_mode}_top{self.top_k}"
            / f"actor_{_slug(model_name)}"
            / f"speculator_{_slug(guess_model_name)}"
        )
        Path(self.base_traj_path).mkdir(parents=True, exist_ok=True)
        self.webthink_prompt = self._build_prompt()

    def _default_sample_ids(self):
        """保持论文 seed 的固定顺序，默认截取配置中的样本数。"""
        ids = list(range(constants.num))
        random.Random(self.seed).shuffle(ids)
        return ids[: constants.n_samples_to_run]

    @staticmethod
    def _get_env(guess_llm=None):
        wiki_env = environment.WikiEnv(guess_llm=guess_llm)
        wrapped = wrappers.HotPotQAWrapper(wiki_env, split="dev")
        wrapped = wrappers.LoggingWrapper(wrapped)
        return wrappers.HistoryWrapper(wrapped, obs_format="history")

    @staticmethod
    def _build_prompt():
        examples = Utils.read_json(
            os.path.join(constants.prompts_folder, constants.prompt_file)
        ).get("webthink_simple6")
        return Utils.join_prompt(
            PromptTemplates.REACT_INSTRUCTION,
            examples,
            PromptTemplates.PROMPT_INSTRUCTION,
        )

    def log(self, *args, save_log=True):
        text = " ".join(str(arg) for arg in args).strip()
        if self.to_print_output:
            print(text)
        if save_log and self.current_index is not None:
            Utils.append_file(
                text,
                os.path.join(self.base_traj_path, str(self.current_index), "log.txt"),
            )

    @staticmethod
    def extract_actions(text):
        """按模型输出中的原始出现顺序提取 Search/Lookup/Finish。"""
        return re.findall(
            r"(?i)(?:search|lookup|finish)\[[^\]]+\]",
            text,
        )

    @staticmethod
    def canonical_action(action):
        """只规范化动作名称大小写，参数保持原样用于严格命中。"""
        action = action.strip()
        bracket = action.find("[")
        if bracket < 0:
            return action
        return action[:bracket].lower() + action[bracket:]

    @staticmethod
    def action_name(action):
        bracket = action.find("[")
        return action[:bracket].lower() if bracket >= 0 else None

    @staticmethod
    def search_entity(action):
        canonical = ReproductionRunner.canonical_action(action)
        if canonical.startswith("search[") and canonical.endswith("]"):
            return canonical[len("search[") : -1]
        return None

    @staticmethod
    def separate_thought_and_actions(step_index, response):
        marker = f"Thought {step_index}: "
        thought = response.split(marker, 1)[-1].strip() if marker in response else response.strip()
        return thought, ReproductionRunner.extract_actions(response)

    def generate_actions(
        self,
        client,
        step_index,
        running_prompt,
        num_actions,
        max_retries,
        role_label,
    ):
        """调用指定角色客户端并返回按原文顺序截断的候选动作。"""
        response = client.call(
            running_prompt
            + PromptTemplates.ACTION_GUESS_PROMPT.format(
                i=step_index,
                num_guesses=num_actions,
            )
        )
        thought, actions = self.separate_thought_and_actions(step_index, response)
        attempts = 0
        while not actions and attempts < max_retries:
            attempts += 1
            self.log(
                f"  [Retry {attempts}/{max_retries}] {role_label} returned no valid action"
            )
            response = client.call(
                running_prompt
                + PromptTemplates.RETRY_PROMPT.format(
                    attempt=attempts,
                    role=constants.agent_role,
                    num_guesses=num_actions,
                )
            )
            actions = self.extract_actions(response)
        if not actions:
            raise ValueError(f"{role_label} returned no valid action after retries")
        return thought, actions[:num_actions], attempts

    def _load_replay_trace(self, sample_id):
        if self.authority_trace_root is None:
            return None
        path = self.authority_trace_root / str(sample_id) / "authority_trace.json"
        if not path.exists():
            raise FileNotFoundError(f"authority replay trace not found: {path}")
        with path.open(encoding="utf-8") as handle:
            trace = json.load(handle)
        if int(trace["sample_id"]) != int(sample_id):
            raise ValueError(f"authority trace sample mismatch in {path}")
        return trace["steps"]

    def _authority_action(self, step_index, running_prompt, replay_steps):
        if replay_steps is not None:
            if step_index > len(replay_steps):
                raise ValueError(
                    f"authority replay has only {len(replay_steps)} steps, requested {step_index}"
                )
            record = replay_steps[step_index - 1]
            return record["thought"], record["action"], 0, True
        thought, actions, retries = self.generate_actions(
            self.actor_llm,
            step_index,
            running_prompt,
            num_actions=1,
            max_retries=constants.max_agent_retries,
            role_label="authority actor",
        )
        return thought, actions[0], retries, False

    @staticmethod
    def _empty_traj(prompt):
        return {
            "prompt": prompt,
            "observations": [],
            "thoughts": [],
            "actions": [],
            "time_taken": [],
            "environment": [],
        }

    @staticmethod
    def _append_traj(traj, thought, action, observation, elapsed, environment_meta=None):
        traj["thoughts"].append(thought)
        traj["actions"].append(action)
        traj["observations"].append(observation)
        traj["time_taken"].append(elapsed)
        traj["environment"].append(dict(environment_meta or {}))

    def _step(self, env, action, simulate=False):
        started = time.perf_counter()
        obs, reward, done, info = env.step(
            self.canonical_action(action),
            step_type="simulate" if simulate else "wiki",
        )
        elapsed = time.perf_counter() - started
        metadata = {}
        if not simulate and self.action_name(action) == "search":
            metadata = dict(env.get_wrapper_attr("last_search_metadata") or {})
        metadata.update(
            {
                "action_type": self.action_name(action),
                "wall_clock_seconds": elapsed,
                "simulate": bool(simulate),
            }
        )
        return obs.replace("\\n", "").replace("\n", " "), reward, done, info, elapsed, metadata

    @staticmethod
    def _stats_snapshot(clients):
        return {
            name: client.get_stats()
            for name, client in clients.items()
            if client is not None
        }

    @staticmethod
    def _stats_delta(clients, before):
        return {
            name: client.stats_since(before[name])
            for name, client in clients.items()
            if client is not None
        }

    def _clients(self):
        return {
            "actor_authority": self.actor_llm,
            "actor_speculative_candidates": self.candidate_llm,
            "observation_speculator": self.observation_llm,
        }

    def _reset_sample(self, sample_id, paired=False):
        question = self.env.reset(seed=self.seed, idx=sample_id)
        if paired:
            simulated_question = self.sim_env.reset(seed=self.seed, idx=sample_id)
            if simulated_question != question:
                raise AssertionError("paired environments reset to different questions")
        actual_idx = int(self.env.get_wrapper_attr("data_idx"))
        if actual_idx != int(sample_id):
            raise AssertionError(
                f"requested sample {sample_id}, environment selected {actual_idx}"
            )
        return question

    def _finish_if_needed(self, info, done):
        if done:
            return info, False
        _, _, _, info, _, _ = self._step(self.env, "Finish[]")
        return info, True

    def _run_sequential(self, sample_id):
        question = self._reset_sample(sample_id)
        running_prompt = self.webthink_prompt + question + "\n"
        normal = self._empty_traj(running_prompt)
        simulated = self._empty_traj(running_prompt)
        replay_steps = self._load_replay_trace(sample_id)
        authority_trace = []
        info = {}
        done = False
        model_finished = False

        for step_index in range(1, self.num_steps):
            thought, action, retries, replayed = self._authority_action(
                step_index,
                running_prompt,
                replay_steps,
            )
            obs, _, done, info, elapsed, env_meta = self._step(self.env, action)
            self._append_traj(normal, thought, action, obs, elapsed, env_meta)
            authority_trace.append(
                {
                    "step": step_index,
                    "thought": thought,
                    "action": action,
                    "retries": retries,
                    "replayed": replayed,
                    "observation_sha256": hashlib.sha256(obs.encode()).hexdigest(),
                }
            )
            running_prompt += PromptTemplates.NEXT_STEP_PROMPT.format(
                i=step_index,
                thought=thought,
                action=action,
                obs=obs,
            )
            self.log(
                f"  [Sequential] step={step_index} action={action} "
                f"env={elapsed:.3f}s"
            )
            if done:
                model_finished = True
                break

        info, forced_finish = self._finish_if_needed(info, done)
        return {
            "question": question,
            "normal": normal,
            "simulated": simulated,
            "info": info,
            "authority_trace": authority_trace,
            "online_events": [],
            "model_finished": model_finished,
            "forced_finish": forced_finish,
        }

    def _run_paired(self, sample_id):
        question = self._reset_sample(sample_id, paired=True)
        running_prompt = self.webthink_prompt + question + "\n"
        simulated_prompt = running_prompt
        normal = self._empty_traj(running_prompt)
        simulated = self._empty_traj(running_prompt)
        replay_steps = self._load_replay_trace(sample_id)
        authority_trace = []
        info = {}
        done = False
        model_finished = False

        for step_index in range(1, self.num_steps):
            thought, action, retries, replayed = self._authority_action(
                step_index,
                running_prompt,
                replay_steps,
            )
            sim_thought, sim_actions, sim_retries = self.generate_actions(
                self.candidate_llm,
                step_index,
                simulated_prompt,
                num_actions=self.top_k,
                max_retries=constants.max_guess_retries,
                role_label="speculative candidate actor",
            )

            obs, _, done, info, elapsed, env_meta = self._step(self.env, action)
            sim_obs, _, _, _, sim_elapsed, sim_env_meta = self._step(
                self.sim_env,
                action,
                simulate=True,
            )
            self._append_traj(normal, thought, action, obs, elapsed, env_meta)
            self._append_traj(
                simulated,
                sim_thought,
                sim_actions,
                sim_obs,
                sim_elapsed,
                sim_env_meta,
            )
            authority_trace.append(
                {
                    "step": step_index,
                    "thought": thought,
                    "action": action,
                    "retries": retries,
                    "replayed": replayed,
                    "observation_sha256": hashlib.sha256(obs.encode()).hexdigest(),
                    "candidate_retries": sim_retries,
                }
            )
            running_prompt += PromptTemplates.NEXT_STEP_PROMPT.format(
                i=step_index,
                thought=thought,
                action=action,
                obs=obs,
            )
            simulated_prompt += PromptTemplates.NEXT_STEP_PROMPT.format(
                i=step_index,
                thought=thought,
                action=action,
                obs=sim_obs,
            )
            self.log(
                f"  [Paired] step={step_index} action={action} "
                f"candidates={sim_actions} real_env={elapsed:.3f}s sim_env={sim_elapsed:.3f}s"
            )
            if done:
                model_finished = True
                break

        info, forced_finish = self._finish_if_needed(info, done)
        return {
            "question": question,
            "normal": normal,
            "simulated": simulated,
            "info": info,
            "authority_trace": authority_trace,
            "online_events": [],
            "model_finished": model_finished,
            "forced_finish": forced_finish,
        }

    def _predict_observation(self, action):
        entity = self.search_entity(action)
        if entity is None:
            raise ValueError("online observation prediction is only valid for Search")
        response = self.observation_llm.call(
            PromptTemplates.GUESS_STEP_PROMPT.format(entity)
        )
        return environment.WikiEnv.get_page_obs(response)

    def _build_online_branch(
        self,
        step_index,
        thought,
        action,
        running_prompt,
        search_executor,
    ):
        """预测当前 Search observation，生成下一步候选并预取安全 Search。"""
        branch_started = time.perf_counter()
        predicted_obs = self._predict_observation(action)
        predicted_prompt = running_prompt + PromptTemplates.NEXT_STEP_PROMPT.format(
            i=step_index,
            thought=thought,
            action=action,
            obs=predicted_obs,
        )
        candidate_thought, candidates, retries = self.generate_actions(
            self.candidate_llm,
            step_index + 1,
            predicted_prompt,
            num_actions=self.top_k,
            max_retries=constants.max_guess_retries,
            role_label="online speculative candidate actor",
        )
        prefetches = {}
        for candidate in candidates:
            canonical = self.canonical_action(candidate)
            entity = self.search_entity(canonical)
            if entity is None or canonical in prefetches:
                continue
            submitted = time.perf_counter()
            prefetches[canonical] = {
                "future": search_executor.submit(
                    environment.WikiEnv.fetch_search_result,
                    entity,
                ),
                "submitted_at": submitted,
            }
        return {
            "source_step": step_index,
            "predicted_observation": predicted_obs,
            "candidate_thought": candidate_thought,
            "candidates": candidates,
            "candidate_retries": retries,
            "prefetches": prefetches,
            "branch_seconds": time.perf_counter() - branch_started,
        }

    @staticmethod
    def _discard_branch(branch):
        if not branch:
            return
        for item in branch.get("prefetches", {}).values():
            item["future"].cancel()

    def _consume_online_branch(self, branch_future, action):
        event = {
            "candidate_ready": False,
            "action_prediction_hit": False,
            "strict_hit": False,
            "completed_hit": False,
            "unfinished_hit": False,
            "hidden_seconds": 0.0,
            "wait_seconds": 0.0,
            "candidates": [],
        }
        if branch_future is None or not branch_future.done():
            return event, None
        branch = branch_future.result()
        event["candidate_ready"] = True
        event["source_step"] = branch["source_step"]
        event["branch_seconds"] = branch["branch_seconds"]
        event["candidates"] = branch["candidates"]
        canonical = self.canonical_action(action)
        prefetch = branch["prefetches"].get(canonical)
        if canonical not in [self.canonical_action(item) for item in branch["candidates"]]:
            self._discard_branch(branch)
            return event, branch
        event["action_prediction_hit"] = True
        if prefetch is None:
            self._discard_branch(branch)
            return event, branch
        event["strict_hit"] = True

        event["completed_hit"] = prefetch["future"].done()
        event["unfinished_hit"] = not event["completed_hit"]
        wait_started = time.perf_counter()
        result = prefetch["future"].result()
        wait_seconds = time.perf_counter() - wait_started
        hidden_seconds = max(0.0, result["fetch_seconds"] - wait_seconds)
        event["wait_seconds"] = wait_seconds
        event["hidden_seconds"] = hidden_seconds
        self.env.get_wrapper_attr("install_prefetched_search")(
            canonical,
            result,
            {
                "completed_hit": event["completed_hit"],
                "unfinished_hit": event["unfinished_hit"],
                "wait_seconds": wait_seconds,
                "hidden_seconds": hidden_seconds,
                "prefetch_age_seconds": time.perf_counter() - prefetch["submitted_at"],
            },
        )
        for key, item in branch["prefetches"].items():
            if key != canonical:
                item["future"].cancel()
        return event, branch

    def _run_online(self, sample_id):
        question = self._reset_sample(sample_id)
        running_prompt = self.webthink_prompt + question + "\n"
        normal = self._empty_traj(running_prompt)
        simulated = self._empty_traj(running_prompt)
        replay_steps = self._load_replay_trace(sample_id)
        authority_trace = []
        online_events = []
        info = {}
        done = False
        model_finished = False
        prior_branch_future = None
        cleanup_started = None

        search_executor = ThreadPoolExecutor(
            max_workers=max(4, self.top_k + 1),
            thread_name_prefix="hotpot-search",
        )
        branch_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="hotpot-branch",
        )
        try:
            for step_index in range(1, self.num_steps):
                thought, action, retries, replayed = self._authority_action(
                    step_index,
                    running_prompt,
                    replay_steps,
                )
                event, consumed_branch = self._consume_online_branch(
                    prior_branch_future,
                    action,
                )
                event.update(
                    {
                        "step": step_index,
                        "authority_action": action,
                        "action_type": self.action_name(action),
                    }
                )
                simulated_actions = event["candidates"]
                self._append_traj(
                    simulated,
                    "",
                    simulated_actions,
                    "",
                    event.get("branch_seconds", 0.0),
                    {
                        "candidate_ready": event["candidate_ready"],
                        "strict_hit": event["strict_hit"],
                    },
                )

                current_branch_future = None
                if self.action_name(action) == "search":
                    current_branch_future = branch_executor.submit(
                        self._build_online_branch,
                        step_index,
                        thought,
                        action,
                        running_prompt,
                        search_executor,
                    )

                tool_future = search_executor.submit(
                    self._step,
                    self.env,
                    action,
                    False,
                )
                obs, _, done, info, elapsed, env_meta = tool_future.result()
                self._append_traj(normal, thought, action, obs, elapsed, env_meta)
                event["environment"] = env_meta
                online_events.append(event)
                authority_trace.append(
                    {
                        "step": step_index,
                        "thought": thought,
                        "action": action,
                        "retries": retries,
                        "replayed": replayed,
                        "observation_sha256": hashlib.sha256(obs.encode()).hexdigest(),
                    }
                )
                running_prompt += PromptTemplates.NEXT_STEP_PROMPT.format(
                    i=step_index,
                    thought=thought,
                    action=action,
                    obs=obs,
                )
                self.log(
                    f"  [Online] step={step_index} action={action} "
                    f"ready={event['candidate_ready']} hit={event['strict_hit']} "
                    f"hidden={event['hidden_seconds']:.3f}s env={elapsed:.3f}s"
                )
                prior_branch_future = current_branch_future
                if done:
                    model_finished = True
                    break

            info, forced_finish = self._finish_if_needed(info, done)
        finally:
            cleanup_started = time.perf_counter()
            if prior_branch_future is not None:
                try:
                    branch = prior_branch_future.result()
                    self._discard_branch(branch)
                except Exception as error:
                    self.log(f"  [Online cleanup warning] {error}", save_log=False)
            branch_executor.shutdown(wait=True, cancel_futures=True)
            search_executor.shutdown(wait=True, cancel_futures=True)
        return {
            "question": question,
            "normal": normal,
            "simulated": simulated,
            "info": info,
            "authority_trace": authority_trace,
            "online_events": online_events,
            "model_finished": model_finished,
            "forced_finish": forced_finish,
            "cleanup_seconds": time.perf_counter() - cleanup_started,
        }

    @staticmethod
    def _trajectory_hash(authority_trace):
        payload = json.dumps(
            [step["action"] for step in authority_trace],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _save_sample(self, sample_id, result, usage, sample_wall_seconds):
        sample_dir = Path(self.base_traj_path) / str(sample_id)
        sample_dir.mkdir(parents=True, exist_ok=True)
        normal = result["normal"]
        simulated = result["simulated"]
        prediction = (
            Metrics.summarize_action_predictions(normal, simulated, self.top_k)
            if self.execution_mode != "sequential"
            else None
        )
        info = result["info"]
        actual_idx = int(info.get("question_idx", sample_id))
        if actual_idx != int(sample_id):
            raise AssertionError(
                f"saved sample {sample_id} completed as dataset index {actual_idx}"
            )

        task = {
            "em": float(info.get("em", 0)),
            "f1": float(info.get("f1", 0)),
            "reward": float(info.get("reward", 0)),
            "answer": info.get("answer"),
            "ground_truth": info.get("gt_answer"),
            "model_finished": result["model_finished"],
            "forced_finish": result["forced_finish"],
        }
        environment_steps = normal["environment"]
        sample_metrics = {
            "sample_id": int(sample_id),
            "actual_question_idx": actual_idx,
            "question": result["question"],
            "execution_mode": self.execution_mode,
            "top_k": self.top_k,
            "task": task,
            "prediction": prediction,
            "usage": usage,
            "latency": {
                "sample_wall_seconds": sample_wall_seconds,
                "online_cleanup_seconds": result.get("cleanup_seconds", 0.0),
                "environment_wall_seconds": sum(normal["time_taken"]),
                "environment_by_action": {
                    action_name: sum(
                        item["wall_clock_seconds"]
                        for item in environment_steps
                        if item["action_type"] == action_name
                    )
                    for action_name in ("search", "lookup", "finish")
                },
                "hidden_seconds": sum(
                    event.get("hidden_seconds", 0.0)
                    for event in result["online_events"]
                ),
            },
            "online": {
                "eligible_steps": sum(
                    event["action_type"] == "search"
                    for event in result["online_events"]
                ),
                "candidate_ready": sum(
                    event.get("candidate_ready", False)
                    for event in result["online_events"]
                ),
                "strict_hits": sum(
                    event.get("strict_hit", False)
                    for event in result["online_events"]
                ),
                "completed_hits": sum(
                    event.get("completed_hit", False)
                    for event in result["online_events"]
                ),
                "unfinished_hits": sum(
                    event.get("unfinished_hit", False)
                    for event in result["online_events"]
                ),
            },
            "authority_action_sha256": self._trajectory_hash(result["authority_trace"]),
        }
        authority_trace = {
            "sample_id": int(sample_id),
            "question": result["question"],
            "steps": result["authority_trace"],
            "action_sha256": sample_metrics["authority_action_sha256"],
        }

        Utils.save_json(normal, str(sample_dir / "normalobs.json"))
        Utils.save_json(simulated, str(sample_dir / "simobs.json"))
        legacy_shaped_metrics = {}
        if prediction is not None:
            rates = prediction["paper"][f"top{self.top_k}"]["rates"]
            legacy_shaped_metrics = {
                key: value
                for key, value in rates.items()
                if value is not None
            }
        Utils.save_json(legacy_shaped_metrics, str(sample_dir / "metrics.json"))
        Utils.save_json(sample_metrics, str(sample_dir / "sample_metrics.json"))
        Utils.save_json(authority_trace, str(sample_dir / "authority_trace.json"))
        Utils.save_json(result["online_events"], str(sample_dir / "online_events.json"))
        Utils.save_json(info, str(sample_dir / "result.json"))
        return sample_metrics

    def _aggregate(self, samples):
        task_em = [item["task"]["em"] for item in samples]
        task_f1 = [item["task"]["f1"] for item in samples]
        prediction_items = [
            item["prediction"] for item in samples if item["prediction"] is not None
        ]
        usage = {}
        for role in self._clients():
            role_values = [item["usage"].get(role) for item in samples]
            role_values = [value for value in role_values if value is not None]
            if not role_values:
                continue
            usage[role] = {
                key: sum(value.get(key, 0) for value in role_values)
                for key in (
                    "calls",
                    "failed_calls",
                    "prompt_tokens",
                    "output_tokens",
                    "reasoning_tokens",
                    "total_tokens",
                    "latency_seconds",
                )
            }
        online_eligible = sum(item["online"]["eligible_steps"] for item in samples)
        online_hits = sum(item["online"]["strict_hits"] for item in samples)
        summary = {
            "run_id": self.run_id,
            "execution_mode": self.execution_mode,
            "actor_model": self.model_name,
            "speculator_model": self.guess_model_name,
            "seed": self.seed,
            "top_k": self.top_k,
            "num_steps": self.num_steps,
            "sample_ids": self.sample_ids,
            "n_samples": len(samples),
            "task": {
                "em_macro": _mean(task_em),
                "f1_macro": _mean(task_f1),
                "model_completion_rate": _mean(
                    [float(item["task"]["model_finished"]) for item in samples]
                ),
            },
            "prediction": (
                Metrics.aggregate_prediction_summaries(prediction_items, self.top_k)
                if prediction_items
                else None
            ),
            "usage": usage,
            "latency": {
                "sample_wall_seconds_macro": _mean(
                    [item["latency"]["sample_wall_seconds"] for item in samples]
                ),
                "environment_wall_seconds_macro": _mean(
                    [item["latency"]["environment_wall_seconds"] for item in samples]
                ),
                "hidden_seconds_total": sum(
                    item["latency"]["hidden_seconds"] for item in samples
                ),
            },
            "online": {
                "eligible_steps": online_eligible,
                "candidate_ready": sum(
                    item["online"]["candidate_ready"] for item in samples
                ),
                "strict_hits": online_hits,
                "completed_hits": sum(
                    item["online"]["completed_hits"] for item in samples
                ),
                "unfinished_hits": sum(
                    item["online"]["unfinished_hits"] for item in samples
                ),
                "speculative_hit_rate": (
                    online_hits / online_eligible if online_eligible else None
                ),
            },
        }
        return summary

    def _write_sample_csv(self, samples):
        path = Path(self.base_traj_path) / "samples.csv"
        fields = [
            "sample_id",
            "em",
            "f1",
            "model_finished",
            "sample_wall_seconds",
            "environment_wall_seconds",
            "hidden_seconds",
            "online_hits",
            "authority_action_sha256",
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for item in samples:
                writer.writerow(
                    {
                        "sample_id": item["sample_id"],
                        "em": item["task"]["em"],
                        "f1": item["task"]["f1"],
                        "model_finished": item["task"]["model_finished"],
                        "sample_wall_seconds": item["latency"]["sample_wall_seconds"],
                        "environment_wall_seconds": item["latency"][
                            "environment_wall_seconds"
                        ],
                        "hidden_seconds": item["latency"]["hidden_seconds"],
                        "online_hits": item["online"]["strict_hits"],
                        "authority_action_sha256": item["authority_action_sha256"],
                    }
                )

    def run(self, skip_done=False):
        """按显式样本列表串行运行，并持续保存逐样本结果。"""
        all_samples = []
        run_started = time.perf_counter()
        for sample_id in self.sample_ids:
            self.current_index = int(sample_id)
            sample_dir = Path(self.base_traj_path) / str(sample_id)
            metrics_path = sample_dir / "sample_metrics.json"
            if skip_done and metrics_path.exists():
                with metrics_path.open(encoding="utf-8") as handle:
                    all_samples.append(json.load(handle))
                self.log(f"[SKIP] sample {sample_id} already complete", save_log=False)
                continue
            sample_dir.mkdir(parents=True, exist_ok=True)
            Utils.delete_file(str(sample_dir / "log.txt"))

            self.log("=" * 72)
            self.log(
                f"sample={sample_id} mode={self.execution_mode} "
                f"top_k={self.top_k} actor={self.model_name} "
                f"speculator={self.guess_model_name}"
            )
            clients = self._clients()
            before = self._stats_snapshot(clients)
            sample_started = time.perf_counter()
            if self.execution_mode == "sequential":
                result = self._run_sequential(sample_id)
            elif self.execution_mode == "paired":
                result = self._run_paired(sample_id)
            else:
                result = self._run_online(sample_id)
            sample_wall_seconds = time.perf_counter() - sample_started
            usage = self._stats_delta(clients, before)
            sample_metrics = self._save_sample(
                sample_id,
                result,
                usage,
                sample_wall_seconds,
            )
            all_samples.append(sample_metrics)
            self.log(
                f"[DONE] sample={sample_id} em={sample_metrics['task']['em']} "
                f"f1={sample_metrics['task']['f1']:.4f} "
                f"wall={sample_wall_seconds:.2f}s"
            )
            summary = self._aggregate(all_samples)
            summary["run_wall_seconds_so_far"] = time.perf_counter() - run_started
            Utils.save_json(summary, os.path.join(self.base_traj_path, "summary.json"))
            self._write_sample_csv(all_samples)

        summary = self._aggregate(all_samples)
        summary["run_wall_seconds"] = time.perf_counter() - run_started
        Utils.save_json(summary, os.path.join(self.base_traj_path, "summary.json"))
        self._write_sample_csv(all_samples)
        manifest = {
            "run_id": self.run_id,
            "execution_mode": self.execution_mode,
            "actor_model": self.model_name,
            "speculator_model": self.guess_model_name,
            "seed": self.seed,
            "top_k": self.top_k,
            "num_steps": self.num_steps,
            "sample_ids": self.sample_ids,
            "authority_trace_root": (
                str(self.authority_trace_root) if self.authority_trace_root else None
            ),
            "base_traj_path": self.base_traj_path,
        }
        Utils.save_json(manifest, os.path.join(self.base_traj_path, "manifest.json"))
        return summary
