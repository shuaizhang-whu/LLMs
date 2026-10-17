"""为 WikiEnv 增加数据集、轨迹日志和历史 prompt 语义的 Gym wrappers。

HotPotQA runner 中实际的包装顺序（从外到内）是：
``HistoryWrapper -> LoggingWrapper -> HotPotQAWrapper -> WikiEnv``。
每一层都把大部分属性/调用转发给内层环境，但 reset 参数和轨迹状态在各层
有各自语义；尤其需要核对 ``idx`` 是否完整传到底层数据集 wrapper。
"""

import json
import os
import gymnasium as gym
import numpy as np
import re
import string
from collections import Counter


DATA_DIR = "data"
HOTPOTQA_SPLIT_FILE = {
    "train": "hotpot_train_v1.1_simplified.json",
    "dev": "hotpot_dev_v1_simplified.json",
    "test": "hotpot_test_v1_simplified.json",
}

FEVER_SPLIT_FILE = {
    "train": "train.jsonl",
    "dev": "paper_dev.jsonl",
}


class HistoryWrapper(gym.Wrapper):
    """把 LoggingWrapper 的逐步轨迹格式化为下一轮 ReAct prompt。

    ``normal_trajectory_dict`` 与 ``sim_trajectory_dict`` 只是 runner 用来导出
    两类记录的 Python 容器；它们并不复制或隔离最内层 WikiEnv 的页面状态。
    """

    def __init__(self, env, obs_format, prompt=None):
        super().__init__(env)
        assert obs_format in ["obs", "history"]
        if obs_format == "history":
            assert hasattr(self.env, "traj")
        self.obs_format = obs_format
        self.prompt = prompt if prompt is not None else ""
        self.normal_trajectory_dict = self.get_empty_traj_dict()
        self.sim_trajectory_dict = self.get_empty_traj_dict()

    def get_empty_traj_dict(self):
        """创建一份空的可序列化轨迹记录，避免两个轨迹共享 list。"""
        return {"prompt": "", "observations": [], "thoughts": [], "actions": [], "time_taken": []}

    def step(self, action, step_type="wiki"):
        """原样向 wrapper 链内层转发 action 与真实/模拟步骤类型。"""
        return self.env.step(action, step_type)

    def observation(self, obs):
        """返回单条 observation，或将 LoggingWrapper 轨迹拼成完整历史。"""
        if self.obs_format == "obs":
            return obs
        elif self.obs_format == "history":
            observation = self.env.traj["observations"][0] + "\n"
            for i, (o, a) in enumerate(zip(self.env.traj["observations"][1:], self.env.traj["actions"]), 1):
                observation += f"Action {i}: {a}\nObservation {i}: {o}\n\n"
            return self.prompt + observation

    def update_traj_dict_records(self, thought, action, observation, time, sim=False):
        """把一次模型决策写入正常轨迹或模拟轨迹的导出字典。

        这里记录的是调用方传入的 thought/action/observation/耗时，并不会推进
        ``LoggingWrapper.traj``；真正的环境 step 会另行维护后者。
        """
        if sim:
            self.sim_trajectory_dict["thoughts"].append(thought)
            self.sim_trajectory_dict["actions"].append(action)
            self.sim_trajectory_dict["observations"].append(observation)
            self.sim_trajectory_dict["time_taken"].append(time)
        else:
            self.normal_trajectory_dict["thoughts"].append(thought)
            self.normal_trajectory_dict["actions"].append(action)
            self.normal_trajectory_dict["observations"].append(observation)
            self.normal_trajectory_dict["time_taken"].append(time)

    def observation_dict(self, obs=None):
        """以字典形式提取当前历史；当前实现不会填充 ``thoughts`` 字段。"""
        obs_dict = {}
        obs_dict["prompt"] = self.prompt
        obs_dict["observations"] = []
        obs_dict["actions"] = []
        obs_dict["thoughts"] = []
        if self.obs_format == "obs":
            return obs
        elif self.obs_format == "history":
            for i, (o, a) in enumerate(zip(self.env.traj["observations"][1:], self.env.traj["actions"]), 1):
                obs_dict["observations"].append(o)
                obs_dict["actions"].append(a)
        return obs_dict

    def reset(self, seed=None, return_info=False, options=None, idx=None):
        """重置 wrapper 链和两份导出轨迹。

        ``idx``、随机种子和 Gym reset 选项必须原样传给内层 wrapper。旧实现
        把这些值硬编码为 ``None``，导致日志目录编号和实际问题不一致。
        """
        output = self.env.reset(
            seed=seed,
            return_info=return_info,
            options=options,
            idx=idx,
        )
        self.normal_trajectory_dict = self.get_empty_traj_dict()
        self.sim_trajectory_dict = self.get_empty_traj_dict()
        return output


def normalize_answer(s):
    """按 HotPotQA 常用规则做小写化、去标点/冠词及空白归一化。"""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    """计算归一化 token 级 F1、precision 与 recall。"""
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


class HotPotQAWrapper(gym.Wrapper):
    """为 WikiEnv 绑定 HotPotQA 样本、答案评分和题目元数据。"""

    def __init__(self, env, split):
        super().__init__(env)
        data_file = f"{DATA_DIR}/{HOTPOTQA_SPLIT_FILE[split]}"
        with open(data_file, encoding="utf-8") as handle:
            self.data = json.load(handle)
        self.data = [(d['question'], d['answer']) for d in self.data]
        self.data_idx = 0
        self.split = split

    def reset(self, seed=None, return_info=False, options=None, idx=None):
        """重置环境并选择一个数据样本。

        若传入 ``idx`` 则严格使用指定样本，否则通过 NumPy 随机状态抽样。
        """
        self.env.reset(seed=seed, return_info=return_info, options=options)
        # 这次空 action 调用的异常会被吞掉，随后环境再次 reset；保留该流程
        # 是为了忠实说明原实现，不代表空 action 属于有效任务步骤。
        try:
            self.env.step('')
        except:
            pass
        self.env.reset(seed=seed, return_info=return_info, options=options)
        self.data_idx = int(np.random.randint(len(self.data))) if idx is None else int(idx)
        if not 0 <= self.data_idx < len(self.data):
            raise IndexError(
                f"HotPotQA sample index {self.data_idx} is outside [0, {len(self.data) - 1}]"
            )
        observation = f"Question: {self.data[self.data_idx][0]}"
        info = self._get_info()
        return (observation, info) if return_info else observation

    def _get_info(self):
        """汇总 WikiEnv 状态与当前 HotPotQA 问题元数据。"""
        return {
            # Gymnasium 1.x 不再通过 Wrapper.__getattr__ 隐式转发未知属性；
            # 使用公开的 get_wrapper_attr 向内查找 WikiEnv 状态。
            "steps": self.get_wrapper_attr("steps"),
            "answer": self.get_wrapper_attr("answer"),
            "question": self.data[self.data_idx][0],
            "hotpot_split": self.split,
        }

    def get_reward(self, info):
        """以归一化 exact match 计算终局 0/1 奖励。"""
        if info['answer'] is not None:
            # 变量名 pred/gt 与其数据来源相反，但这里只做对称的相等比较，
            # 不影响 exact-match 数值。
            pred = normalize_answer(self.data[self.data_idx][1])
            gt = normalize_answer(info['answer'])
            score = (pred == gt)
            return int(score)
        return 0

    def get_metrics(self, info):
        """返回终局 EM/F1；未提交 finish 答案时各项均为零。"""
        if info['answer'] is not None:
            pred = normalize_answer(self.data[self.data_idx][1])
            gt = normalize_answer(info['answer'])
            em = (pred == gt)
            f1 = f1_score(pred, gt)[0]
            return {'reward': em, 'em': em, 'f1': f1}
        return {'reward': 0, 'em': 0, 'f1': 0}

    def step(self, action, step_type="wiki"):
        """执行底层 action，并在 episode 结束时补充答案与评测指标。"""
        obs, _, done, info = self.env.step(action, step_type)
        reward = self.get_reward(info)
        if done:
            obs = f"Episode finished, reward = {reward}\n"
            info.update({"gt_answer": self.data[self.data_idx][1], "question_idx": self.data_idx})
            info.update(self.get_metrics(info))
        return obs, reward, done, info

    def __len__(self):
        return len(self.data)


class FeverWrapper(gym.Wrapper):
    """WikiEnv 的 FEVER 数据集适配层；HotPotQA 主 runner 当前不使用它。"""

    def __init__(self, env, split):
        super().__init__(env)
        data_path = f"./data/{FEVER_SPLIT_FILE[split]}"
        with open(data_path, "r") as json_file:
            json_list = list(json_file)
        data = []
        for json_str in json_list:
            json_str = json.loads(json_str)
            label = json_str["label"]
            claim = json_str["claim"]
            data.append((claim, label))
        self.data = data
        self.data_idx = 0
        self.split = split

    def reset(self, seed=None, return_info=False, options=None, idx=None):
        """重置环境，并按 ``idx`` 指定或随机选择一条 FEVER claim。"""
        self.env.reset(seed=seed, return_info=return_info, options=options)
        try:
            self.env.step('')
        except:
            pass
        self.env.reset(seed=seed, return_info=return_info, options=options)
        self.data_idx = int(np.random.randint(len(self.data))) if idx is None else idx
        observation = f"Claim: {self.data[self.data_idx][0]}"
        info = self._get_info()
        return (observation, info) if return_info else observation

    def _get_info(self):
        """汇总 WikiEnv 状态与当前 FEVER claim 元数据。"""
        return {
            "steps": self.get_wrapper_attr("steps"),
            "answer": self.get_wrapper_attr("answer"),
            "question": self.data[self.data_idx][0],
            "fever_split": self.split,
        }

    def get_reward(self, info):
        """按归一化标签 exact match 返回终局 0/1 奖励。"""
        if info['answer'] is not None:
            label = normalize_answer(self.data[self.data_idx][1])
            pred = normalize_answer(info['answer'])
            if label == pred:
                return 1
        return 0

    def step(self, action):
        """执行底层 action，并在结束时附加 FEVER 评测字段。"""
        obs, _, done, info = self.env.step(action)
        reward = self.get_reward(info)
        if done:
            obs = f"Episode finished, reward = {reward}\n"
            info.update({"gt_answer": self.data[self.data_idx][1], "question_idx": self.data_idx})
            info.update({'em': reward, 'reward': reward, 'f1': reward})
        return obs, reward, done, info

    def __len__(self):
        return len(self.data)


class LoggingWrapper(gym.Wrapper):
    """记录每个 episode 的 observation/action 序列并导出 JSON。

    ``traj`` 是当前 episode，``trajs`` 是已收束 episode 的列表。旧轨迹在
    下一次 reset 或最终 write 时才由 ``update_record`` 移入列表。
    """

    def __init__(self, env, folder="trajs", file_id=None):
        super().__init__(env)
        self.trajs = []
        self.traj = {"observations": [], "actions": []}
        self.folder = folder
        self.file_id = np.random.randint(0, 10000000) if file_id is None else file_id
        self.file_path = f"{self.folder}/{self.file_id}.json"
        # 注意原实现固定创建 "trajs"，并非上面的 self.folder；若传入其他
        # folder，调用方需要保证目标目录已经存在。
        os.makedirs("trajs", exist_ok=True)

    def __len__(self):
        return len(self.env.data)

    def reset(self, seed=None, return_info=False, options=None, idx=None):
        """封存上一条非空轨迹，转发样本索引，并初始化新轨迹首条 observation。"""
        self.update_record()
        output = self.env.reset(seed=seed, return_info=return_info, options=options, idx=idx)
        observation = output[0] if return_info else output
        self.traj = {"observations": [observation], "actions": []}
        return output

    def step(self, action, step_type="wiki"):
        """转发 action，并把实际返回 observation 与 action 追加到当前轨迹。"""
        obs, reward, done, info = self.env.step(action, step_type)
        self.traj["observations"].append(obs)
        self.traj["actions"].append(action)
        if done:
            self.traj.update(info)
        return obs, reward, done, info

    def update_record(self):
        """把非空的当前 episode 移入 ``trajs``，随后清空工作区。"""
        if self.traj.get("observations"):
            self.trajs.append(self.traj)
            self.traj = {"observations": [], "actions": []}

    def write(self):
        """封存当前 episode，并把全部轨迹覆盖写入本实例的 JSON 文件。"""
        self.update_record()
        with open(self.file_path, "w") as f:
            json.dump(self.trajs, f)
            print(f"Saved trajs to trajs/{self.file_id}.json")

    def close(self):
        """关闭 wrapper 时持久化轨迹。"""
        self.write()
