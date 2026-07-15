"""HotPotQA 实验的文件、轨迹文本与表格转换工具。

本模块包含多项会覆盖或递归删除实验文件的辅助函数。尤其是
``process_obs``、``cleanup_trajs`` 和各类 ``save_*`` 调用，运行前应确认路径
指向副本或新的结果目录，避免改写仓库附带轨迹。
"""

import os
import json
import shutil
import re
import pandas as pd

from . import constants


class Utils:
    """集中提供 JSON/文本 I/O、轨迹清理与指标表格转换。"""

    @staticmethod
    def read_json(path):
        """以系统默认文本编码读取 JSON 文件。"""
        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def save_json(obj, path, delete_prev_file=False):
        """将对象格式化写入 JSON 文件。

        ``delete_prev_file=True`` 会先显式删除旧文件；即使为 False，随后使用
        ``"w"`` 打开也会截断同名文件。此函数不会自动创建父目录。
        """
        if os.path.exists(path) and delete_prev_file:
            os.remove(path)
        with open(path, "w") as f:
            json.dump(obj, f, indent=4)

    @staticmethod
    def read_file(path):
        """以 UTF-8 编码读取完整文本文件。"""
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def save_file(string, path, delete_prev_file=False):
        """创建父目录并以 UTF-8 覆盖写入文本。

        与 ``save_json`` 一样，``"w"`` 模式本身就会覆盖同名文件；删除参数只
        决定是否在写入前额外执行一次删除。
        """
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
        if os.path.exists(path) and delete_prev_file:
            os.remove(path)
        with open(path, "w", encoding="utf-8") as f:
            f.write(string)

    @staticmethod
    def join_prompt(*args):
        """按传入顺序直接拼接所有提示词片段，不添加分隔符。"""
        output = ""
        for arg in args:
            output += arg
        return output

    @staticmethod
    def append_file(string, path):
        """创建父目录，并向 UTF-8 文本末尾追加一行。"""
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(string + "\n")

    @staticmethod
    def delete_file(path):
        """若路径存在，则删除该文件。"""
        if os.path.exists(path):
            os.remove(path)

    @staticmethod
    def delete_dir(path, nested=False):
        """删除目录；仅当 ``nested=True`` 时递归删除非空目录。"""
        if os.path.isdir(path):
            try:
                os.rmdir(path)
            except OSError as e:
                if nested:
                    shutil.rmtree(path)
                else:
                    raise e

    @staticmethod
    def check_dir(dir_path):
        """打印样本目录中缺失的预期轨迹文件，不修改目录。"""
        for file_name in constants.trajectory_filenames:
            path = os.path.join(dir_path, file_name)
            if not os.path.exists(path):
                print(f"{dir_path} does not have {file_name} file")

    @staticmethod
    def check_all_dirs(base_path):
        """对基础路径下的每个条目执行轨迹文件完整性检查。"""
        for direc in os.listdir(base_path):
            Utils.check_dir(os.path.join(base_path, direc))

    @staticmethod
    def extract_ith_step_info(i, text, trajectory_type):
        """用固定日志格式提取指定编号步骤的 Thought/Action/Observation。

        正则要求标签、空格和换行与日志模板完全一致；未匹配时三个字段均返回
        None。
        """
        pattern = rf"{trajectory_type.upper()} TRAJECTORY:\n Thought {i}: (.*?)\nAction {i}: (.*?)\nObservation {i}: (.*?)\n"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            thought = match.group(1).strip()
            action = match.group(2).strip()
            observation = match.group(3).strip()
            return {"thought": thought, "action": action, "observation": observation}
        else:
            return {"thought": None, "action": None, "observation": None}

    @staticmethod
    def process_obs(base_path):
        """按日志中的首步位置裁剪轨迹，并覆盖原 JSON 文件。

        此操作具有破坏性：成功处理时会重写 ``normalobs.json`` 和
        ``simobs.json``，新对象只保留 prompt、actions、thoughts、observations，
        原文件中的 ``time_taken`` 等其他字段不会被复制。失败目录列表写到当前
        工作目录下的 ``./not_processed.txt``，该文件同样会被覆盖。
        """
        not_processed = []
        for direc in os.listdir(base_path):
            log_path = os.path.join(base_path, direc, "log.txt")
            normal_obs_path = os.path.join(base_path, direc, "normalobs.json")
            sim_obs_path = os.path.join(base_path, direc, "simobs.json")
            logs = Utils.read_file(log_path)
            normal_obs = Utils.read_json(normal_obs_path)
            sim_obs = Utils.read_json(sim_obs_path)
            new_normal_obs = {}
            new_normal_obs["prompt"] = normal_obs["prompt"]
            new_sim_obs = {}
            new_sim_obs["prompt"] = sim_obs["prompt"]
            try:
                normal_first_steps = Utils.extract_ith_step_info(1, logs, "normal")
                sim_first_steps = Utils.extract_ith_step_info(1, logs, "simulation")
                # list.index 选择首个完全相等的元素；重复内容可能使裁剪点提前。
                new_normal_obs["actions"] = normal_obs["actions"][normal_obs["actions"].index(normal_first_steps["action"]):]
                new_normal_obs["thoughts"] = normal_obs["thoughts"][normal_obs["thoughts"].index(normal_first_steps["thought"]):]
                new_normal_obs["observations"] = normal_obs["observations"][normal_obs["observations"].index(normal_first_steps["observation"]):]
                new_sim_obs["actions"] = sim_obs["actions"][sim_obs["actions"].index(sim_first_steps["action"]):]
                new_sim_obs["thoughts"] = sim_obs["thoughts"][sim_obs["thoughts"].index(sim_first_steps["thought"]):]
                new_sim_obs["observations"] = sim_obs["observations"][sim_obs["observations"].index(sim_first_steps["observation"]):]
                assert len(new_normal_obs["thoughts"]) == len(new_normal_obs["actions"]) == len(new_normal_obs["observations"]), "lengths of thoughts, actions and observations are unequal for normal obs"
                assert len(new_sim_obs["thoughts"]) == len(new_sim_obs["actions"]) == len(new_sim_obs["observations"]), "lengths of thoughts, actions and observations are unequal for sim obs"
            except Exception as e:
                print(e)
                not_processed.append(direc)
                continue

            # 两个原始轨迹文件会先删除再写入，应避免直接对唯一原始结果运行。
            Utils.save_json(new_normal_obs, normal_obs_path, delete_prev_file=True)
            Utils.save_json(new_sim_obs, sim_obs_path, delete_prev_file=True)
        print(len(not_processed))
        Utils.save_file(str(not_processed), "./not_processed.txt")

    @staticmethod
    def cleanup_trajs(trajs_folder):
        """递归删除缺少任一必需文件的样本目录。"""
        for direc in os.listdir(trajs_folder):
            datapoint_path = os.path.join(trajs_folder, direc)
            if Utils.is_dirty_traj(datapoint_path):
                print(f"Cleaning up {datapoint_path}")
                Utils.delete_dir(datapoint_path, nested=True)

    @staticmethod
    def is_dirty_traj(datapoint_path):
        """仅按四个必需文件是否存在，判断样本轨迹是否不完整。"""
        files = ["metrics.json", "normalobs.json", "simobs.json", "log.txt"]
        try:
            for file_name in files:
                assert os.path.exists(os.path.join(datapoint_path, file_name)), f"{file_name} is missing"
        except AssertionError:
            return True
        return False

    @staticmethod
    def convert_json_to_csv(json_data):
        """将 Actor→Guess 模型的嵌套聚合字典转换为 DataFrame。

        本函数只构造内存中的表，不写磁盘。各模型记录若拥有不同指标键，列表
        长度可能不一致，并会由 pandas 在构造 DataFrame 时报告错误。
        """
        df_dict = {}
        df_dict["agent model"] = []
        df_dict["guess model"] = []
        for agent_model in json_data.keys():
            for guess_model in json_data[agent_model].keys():
                df_dict["agent model"].append(agent_model)
                df_dict["guess model"].append(guess_model)
                for key in json_data[agent_model][guess_model].keys():
                    if key in df_dict.keys():
                        df_dict[key].append(json_data[agent_model][guess_model][key])
                    else:
                        df_dict[key] = [json_data[agent_model][guess_model][key]]
        df = pd.DataFrame.from_dict(df_dict)
        return df

    @staticmethod
    def convert_json_to_csv2(json_path, csv_path):
        """将扁平 JSON 记录列表转为 CSV，并覆盖 ``csv_path``。"""
        json_data = Utils.read_json(json_path)
        csv_dict = {}
        for data in json_data:
            for key in data.keys():
                if key not in csv_dict.keys():
                    csv_dict[key] = [data[key]]
                else:
                    csv_dict[key].append(data[key])
        df = pd.DataFrame.from_dict(csv_dict)
        df.to_csv(csv_path, index=False)

    @staticmethod
    def avg(my_list):
        """返回列表算术平均值；空列表返回 None。"""
        if len(my_list) == 0:
            return None
        return sum(my_list) / len(my_list)
