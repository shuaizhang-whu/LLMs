"""HotPotQA ReAct 轨迹采集与投机动作评估的运行器。

本模块同时记录两类数据：

* ``normal_trajectory_dict`` 保存 Actor 在真实 Wikipedia 环境中的动作与 observation；
* ``sim_trajectory_dict`` 保存同一历史下生成的 Top-K 候选动作，以及 Guess LLM
  针对真实动作生成的模拟 observation。

需要注意，当前实现用于采集轨迹并离线比较动作命中率和单次环境调用耗时，并不是
在线并发投机流水线。一次 ``webthink`` 迭代中的模型调用与环境调用均按代码顺序串行
完成；记录的 ``time_taken`` 也只覆盖对应的 ``env.step``，不包含 Actor 生成动作或
候选动作的耗时。
"""

import os
import time
import json
import random
import re
import requests
from os.path import join

from . import constants
from .utils import Utils
from .metrics import Metrics
from .prompts import PromptTemplates
from .llm_client import LLMClient
from . import environment
from . import wrappers


class HotPotQARun:
    """组织 HotPotQA 样本运行、双轨迹落盘和逐样本动作指标计算。

    ``model_name`` 对应 Actor：它既生成正常轨迹的单个动作，也在开启模拟时根据
    ``ACTION_GUESS_PROMPT`` 生成 Top-K 候选动作。``guess_model_name`` 对应
    ``WikiEnv`` 内部的 Guess LLM，仅用于为 Search 动作生成模拟页面内容。
    """

    def __init__(self, model_name="gemini", guess_model_name="gemini", to_print_output=True):
        """初始化 Actor 客户端、带包装器的环境和实验输出路径。"""

        # Actor 客户端既承担正常动作生成，也承担候选动作生成；Guess LLM 在 WikiEnv
        # 内另行初始化，因此两个模型名代表不同职责，而不是两个并发 worker。
        self.llm = LLMClient(
            model_name=model_name,
            temperature=constants.temperature,
            max_tokens=constants.max_output_tokens,
            top_p=constants.top_p,
        )
        self.model_name = model_name
        self.guess_model_name = guess_model_name
        self.env = self._get_env()
        self.simulation_observations_dict = {}
        self.current_index = None
        self.base_traj_path = self.recalc_base_traj_path()
        self.to_print_output = to_print_output

    def recalc_base_traj_path(self):
        """根据 Actor、Top-K 数量和 Guess 模型重新计算本次轨迹根目录。"""

        # 只取模型名最后一个路径片段，避免 provider/组织前缀进入目录层级。
        btp = (
            "./run_metrics/agent_"
            + self.model_name.split('/')[-1]
            + "_top" + str(constants.guess_num_actions)
            + "/trajs_" + self.guess_model_name.split("/")[-1]
        )
        self.base_traj_path = btp
        return btp

    def _get_env(self):
        """构造 WikiEnv 及数据集、日志、历史记录包装器链。"""

        # 包装顺序由内到外为：真实/模拟 Wiki 环境 -> HotPotQA 奖励与数据集
        # -> Gym 轨迹日志 -> 本实验使用的 normal/sim 双轨迹字典。
        env = environment.WikiEnv(guess_model_name=self.guess_model_name)
        env = wrappers.HotPotQAWrapper(env, split="dev")
        env = wrappers.LoggingWrapper(env)
        env = wrappers.HistoryWrapper(env, obs_format="history")
        return env

    def log(self, *args, save_log=True):
        """按配置打印日志，并可追加到当前样本目录的 ``log.txt``。"""

        text = " ".join(str(a) for a in args).strip() + "\n"
        if self.to_print_output:
            print(text, end="")
        if save_log:
            log_path = join(self.base_traj_path, str(self.current_index), "log.txt")
            Utils.append_file(text, log_path)

    def step(self, env, action, simulate=False):
        """执行一次环境动作并返回该次 ``env.step`` 的耗时。

        ``simulate=False`` 时调用真实 Wiki 路径，仅对 ``requests`` 超时最多重试十次；
        ``simulate=True`` 时调用 ``step_type=\"simulate\"``，Search 会转而调用 Guess
        LLM 生成页面。这里没有提交线程、协程或 future，两种调用均同步阻塞。

        模拟分支返回的是 ``env.sim_obs`` 而不是本次 ``env.step`` 的局部 ``obs``；按
        当前 ``WikiEnv`` 实现，``sim_obs`` 只在模拟 Search 时刷新，Lookup/Finish 时
        可能仍是上一次模拟 Search 的值。该口径按原实现保留，需在结果审计中区分。

        返回的秒数从紧邻 ``env.step`` 的两个 ``perf_counter`` 读数相减得到，因此不
        包含调用本方法之前的 Actor/候选动作生成时间。真实路径连续十次超时后当前
        实现会自然返回 ``None``，由上层后续解包暴露失败。
        """

        if simulate:
            start = time.perf_counter()
            obs, r, done, info = env.step(action, step_type="simulate")
            end = time.perf_counter()
            return env.sim_obs, r, done, info, end - start

        attempts = 0
        while attempts < 10:
            try:
                start = time.perf_counter()
                obs, r, done, info = env.step(action)
                end = time.perf_counter()
                return obs, r, done, info, end - start
            except requests.exceptions.Timeout:
                attempts += 1

    def extract_action(self, action_string):
        """从文本中按 Search、Lookup、Finish 的优先级提取首个合法形态动作。"""

        search_pattern = r'[Ss]earch\[[^\]]+\]'
        lookup_pattern = r'[Ll]ookup\[[^\]]+\]'
        finish_pattern = r'[Ff]inish\[[^\]]+\]'
        for pattern in [search_pattern, lookup_pattern, finish_pattern]:
            output = re.search(pattern, action_string)
            if output:
                return output.group()
        return None

    def extract_actions(self, action_string):
        """提取全部动作，并按动作类型而非原文出现顺序拼接结果。

        返回顺序固定为所有 Search、所有 Lookup、所有 Finish。因而下游将列表第一项
        当作 Top-1 时，其含义是当前解析规则下的第一项，不一定是原始文本最先出现的
        动作。这是现有指标口径的一部分，本方法不在此处重排或修正模型输出。
        """

        search_pattern = r'[Ss]earch\[[^\]]+\]'
        lookup_pattern = r'[Ll]ookup\[[^\]]+\]'
        finish_pattern = r'[Ff]inish\[[^\]]+\]'
        outputs = []
        for pattern in [search_pattern, lookup_pattern, finish_pattern]:
            outputs += re.findall(pattern, action_string)
        return outputs

    def action_lowercase(self, action):
        """仅将动作名称首字母小写，保留方括号内参数的原始大小写。"""

        index = action.find("[")
        if index == -1:
            return action[0].lower() + action[1:]
        else:
            return action[:index].lower() + action[index:]

    def separate_thought_and_action(self, i, thought_action):
        """解析指定步的单个 Thought/Action 文本；当前主流程未调用此方法。"""

        action_condition = f"Action {i}: " in thought_action
        thought_condition = f"Thought {i}: " in thought_action
        thought, action = None, None
        if thought_condition and action_condition:
            thought, action = thought_action.strip().split(f"Action {i}: ")
            thought = thought.strip().split(f"Thought {i}: ")[1]
        elif thought_condition:
            thought = thought_action.strip().split(f"Thought {i}: ")[1]
            action = None
        elif action_condition:
            action = thought_action.strip().split(f"Action {i}: ")[1]
            thought = "Let me do the action " + action
        return thought, action

    def separate_thought_and_actions(self, i, thought_action):
        """解析指定步的 thought，并用 ``extract_actions`` 收集候选动作列表。"""

        try:
            split = thought_action.split(f"Thought {i}: ")
            thought = split[0].strip() if len(split) == 1 else split[1].strip()
        except IndexError:
            thought = thought_action.strip()
        actions = self.extract_actions(thought_action)
        return thought, actions

    def generate_thought_actions(self, i, running_prompt, n_calls_badcalls, num_actions=1, max_retries=1):
        """同步调用 Actor，生成当前步的思考文本和一个或多个候选动作。

        ``n_calls_badcalls`` 是由调用方传入的二元可变计数器：索引 0 统计本方法发起的
        LLM 调用总数，索引 1 仅统计首次未解析出动作后的重试调用。``num_actions`` 只
        被写入 prompt 请求模型给出相应数量；解析后不会在代码层强制截断或去重。
        """

        # 首次请求无论最终是否解析成功，都计入总调用数。
        n_calls_badcalls[0] += 1

        thought_action = self.llm.call(
            running_prompt + PromptTemplates.ACTION_GUESS_PROMPT.format(i=i, num_guesses=num_actions),
            stop=None,
        )
        thought, actions = self.separate_thought_and_actions(i, thought_action)

        retry_attempt = 1
        while not actions and retry_attempt <= max_retries:
            self.log(f"  [Retry {retry_attempt}/{max_retries}] No actions found, retrying...")
            n_calls_badcalls[0] += 1
            n_calls_badcalls[1] += 1
            temp_actions = self.llm.call(
                running_prompt + PromptTemplates.RETRY_PROMPT.format(
                    attempt=retry_attempt, role=constants.agent_role, num_guesses=num_actions
                )
            )
            actions = self.extract_actions(temp_actions)
            retry_attempt += 1

        if not actions:
            raise ValueError("Action not found in LLM output after all retries")

        return thought, actions

    def webthink(self, idx=None, prompt=None, to_print=True, n=8, simulate=False):
        """运行一个 HotPotQA 样本并在内存中形成正常/模拟双轨迹。

        当 ``simulate=True`` 时，每一步严格按以下顺序同步执行：

        1. Actor 基于正常历史生成一个真实动作；
        2. 同一个 Actor 基于模拟历史生成 Top-K 候选动作；
        3. 在 ``self.env`` 上执行真实动作并记录真实 observation；
        4. 在同一个 ``self.env`` 上针对该真实动作执行模拟路径，并记录 Guess LLM
           生成的 observation。

        候选动作只用于保存和后续命中率计算，并没有在此方法中实际执行。正常路径与
        模拟路径复用同一个有状态环境对象；当前 ``WikiEnv`` 的模拟 Search 会更新
        ``page``、``obs`` 等字段，所以后续 Lookup 可能读取到最近一次模拟调用留下的
        状态。这是复现实验时需要单独核验的状态耦合，不在轨迹采集阶段隐式修正。

        ``idx`` 会传给最外层包装器；当前包装器链中 ``HistoryWrapper.reset`` 又以
        ``idx=None`` 调用下层，因此日志目录编号与实际抽取题目的对应关系也应由结果
        审计确认。
        """

        done = False
        running_prompt = prompt
        question = self.env.reset(idx=idx)
        running_prompt += question + "\n"
        self.env.normal_trajectory_dict["prompt"] = running_prompt

        if to_print:
            self.log(f"\n{'='*60}")
            self.log(f"  Question [{idx}]: {question}")
            self.log(f"{'='*60}")

        if simulate:
            sim_running_prompt = running_prompt
            self.env.sim_trajectory_dict["prompt"] = sim_running_prompt

        n_calls_badcalls = [0, 0]

        for i in range(1, n):
            if to_print:
                self.log(f"\n--- Step {i}/{n-1} ---")
            try:
                # 先生成正常轨迹的 Actor 动作；actions[0] 被作为本步真实执行动作。
                thought, actions = self.generate_thought_actions(
                    i, running_prompt, n_calls_badcalls, max_retries=constants.max_agent_retries
                )
                action = actions[0]
                if simulate:
                    # 随后同步生成候选动作。这里尚未执行真实环境调用，也没有启动并发任务。
                    sim_thought, sim_actions = self.generate_thought_actions(
                        i, sim_running_prompt, n_calls_badcalls,
                        num_actions=constants.guess_num_actions,
                        max_retries=constants.max_guess_retries,
                    )
                    # 模拟历史先对齐到本步动作执行前的正常历史；模拟 observation 会在
                    # 本轮末尾追加，使下一轮候选生成看到“正常历史 + 模拟 observation”。
                    sim_running_prompt = running_prompt
            except ValueError as e:
                self.log(f"[ERROR] {e}", save_log=False)
                continue

            # 正常轨迹：在真实 Wiki 路径执行 Actor 的第一候选，并只计 env.step 耗时。
            obs, r, done, info, normal_traj_time_taken = self.step(self.env, self.action_lowercase(action))
            obs = obs.replace('\\n', '')
            self.env.update_traj_dict_records(thought, action, obs, normal_traj_time_taken, False)
            next_step_string = PromptTemplates.NEXT_STEP_PROMPT.format(i=i, thought=thought, action=action, obs=obs)
            running_prompt += next_step_string
            if to_print:
                self.log(f"  [Normal]  Thought: {thought}")
                self.log(f"            Action:  {action}")
                self.log(f"            Obs:     {obs[:200]}{'...' if len(obs) > 200 else ''}")
                self.log(f"            Time:    {normal_traj_time_taken:.3f}s")

            if simulate:
                # 模拟轨迹并不执行 sim_actions；它把同一个真实 action 交给 simulate
                # 路径，Search 时由 Guess LLM 生成替代 observation。此调用发生在真实
                # env.step 返回之后，因此两个 time_taken 不能直接证明端到端并发加速。
                sim_obs, sim_r, sim_done, sim_info, sim_traj_time_taken = self.step(
                    self.env, self.action_lowercase(action), simulate=True
                )
                sim_obs = sim_obs.replace('\n', '')
                # 指标所比较的是这里保存的 sim_actions 与正常轨迹 action；模拟环境的
                # reward/done/info 仅接收而未参与本轮终止或最终任务得分判断。
                self.env.update_traj_dict_records(sim_thought, sim_actions, sim_obs, sim_traj_time_taken, True)
                # 模拟 prompt 沿用正常 thought/action，只替换 observation；候选动作文本
                # 本身保存在 sim_trajectory_dict，不写入下一步 prompt。
                next_sim_step_string = PromptTemplates.NEXT_STEP_PROMPT.format(
                    i=i, thought=thought, action=action, obs=sim_obs
                )
                if to_print:
                    self.log(f"  [Sim]     Thought: {sim_thought}")
                    self.log(f"            Actions: {sim_actions}")
                    self.log(f"            Obs:     {sim_obs[:200]}{'...' if len(sim_obs) > 200 else ''}")
                    self.log(f"            Time:    {sim_traj_time_taken:.3f}s")
                sim_running_prompt += next_sim_step_string

            if done:
                break

        if not done:
            # 达到步数上限时强制 Finish，但该补充动作不会写入 normal/sim 轨迹字典。
            obs, r, done, info, time_taken = self.step(self.env, "finish[]")
        if to_print:
            self.log(f"\n  Result: em={info.get('em', 'N/A')}  f1={info.get('f1', 'N/A')}")
        info.update({'n_calls': n_calls_badcalls[0], 'n_badcalls': n_calls_badcalls[1], 'traj': running_prompt})
        return info

    def run(self, webthink_simulate=False, skip_done=False):
        """按固定随机顺序运行样本，保存轨迹、日志和逐样本动作指标。

        ``webthink_simulate`` 控制是否额外采集候选动作和模拟 observation；
        ``skip_done`` 只以样本目录下是否已有 ``log.txt`` 判断是否跳过。此方法自身不
        创建并发执行器，样本之间和每个样本内部均按循环顺序执行。
        """

        # 错误类型在运行时导入；即使使用其他 provider，执行 run() 仍要求此模块可导入。
        from google.genai.errors import ClientError, ServerError

        # 使用局部 Random 实例固定样本编号的洗牌，不改变全局 random 状态。
        idxs = list(range(constants.num))
        random.Random(constants.random_seed).shuffle(idxs)

        # 组合 ReAct 总说明、few-shot 示例和题目提示前缀，作为每个样本的共同初始 prompt。
        webthink_examples = Utils.read_json(
            join(constants.prompts_folder, constants.prompt_file)
        ).get('webthink_simple6')
        webthink_prompt = Utils.join_prompt(
            PromptTemplates.REACT_INSTRUCTION, webthink_examples, PromptTemplates.PROMPT_INSTRUCTION
        )

        rewards = []
        infos = []
        old_time = time.time()

        for i in idxs[:constants.n_samples_to_run]:
            self.current_index = i
            current_dir_path = join(self.base_traj_path, str(self.current_index))

            log_path = join(current_dir_path, "log.txt")
            if skip_done:
                if os.path.exists(log_path):
                    self.log(f"[SKIP] Index {i} already done", save_log=False)
                    continue

            Utils.delete_file(log_path)

            try:
                # 单个样本运行结束后才进入下一个样本，没有跨样本并发。
                info = self.webthink(
                    i, prompt=webthink_prompt, to_print=True,
                    n=constants.n_steps_to_run, simulate=webthink_simulate,
                )

            except ClientError as e:
                self.log(f"[ERROR] Client error, sleeping {constants.client_error_sleep_time}s: {e}", save_log=False)
                Utils.delete_dir(current_dir_path, nested=True)
                time.sleep(constants.client_error_sleep_time)
                continue

            except ServerError as e:
                self.log(f"[ERROR] Server error, sleeping {constants.server_error_sleep_time}s: {e}", save_log=False)
                Utils.delete_dir(current_dir_path, nested=True)
                time.sleep(constants.server_error_sleep_time)
                continue

            except ZeroDivisionError:
                self.log("[ERROR] ZeroDivisionError, skipping sample", save_log=False)
                Utils.delete_dir(current_dir_path, nested=True)
                continue

            except Exception as e:
                Utils.delete_dir(current_dir_path, nested=True)
                raise e

            rewards.append(info['em'])
            infos.append(info)

            # 这是从整个 run 开始至今的平均墙钟时间，分母仅为成功加入 rewards 的样本；
            # 因此此前失败、跳过或等待重试的耗时仍可能留在分子中。
            avg_reward = sum(rewards) / len(rewards)
            avg_time = (time.time() - old_time) / len(rewards)
            self.log(
                f"\n  [Progress] {len(rewards)} samples done | "
                f"Total reward: {sum(rewards)} | "
                f"Avg reward: {avg_reward:.4f} | "
                f"Avg time/sample: {avg_time:.1f}s"
            )

            # normalobs/simobs 保存原始双轨迹；metrics.json 是基于二者逐动作精确匹配
            # 得到的单样本聚合指标，不是在线 speculative hit 事件日志。
            normal_observations_dict = self.env.normal_trajectory_dict
            sim_observations_dict = self.env.sim_trajectory_dict
            Utils.save_json(normal_observations_dict, join(current_dir_path, "normalobs.json"))
            Utils.save_json(sim_observations_dict, join(current_dir_path, "simobs.json"))
            try:
                metric_dict = Metrics.get_action_metrics(
                    normal_observations_dict, sim_observations_dict, sparse=False
                )
            except ZeroDivisionError:
                self.log("[WARN] ZeroDivisionError in metrics, skipping this sample", save_log=False)
                Utils.delete_dir(current_dir_path, nested=True)
                continue
            self.log(f"  [Metrics]  idx={self.current_index}  {metric_dict}", save_log=False)
            Utils.save_json(metric_dict, join(current_dir_path, "metrics.json"))

        # 将 LoggingWrapper 累积的 Gym 轨迹写入其独立的 trajs 文件。
        self.env.write()
