"""HotPotQA 轨迹的动作命中率与 Search 耗时统计。

本模块只消费已经保存的 ``normalobs.json``、``simobs.json`` 和
``metrics.json``。这里的“Top-1/Top-3”描述的是预测动作与真实动作的离线匹配，
Search 时间也只是轨迹中记录的 Search 步骤耗时；它们都不是在线投机流水线的
端到端 wall-clock 指标。
"""

import os

from .utils import Utils


class Metrics:
    """计算单条轨迹指标，并在多个样本目录上进行等权聚合。"""

    @staticmethod
    def get_action_metrics(normal_dict, sim_dict, k=None, sparse=False):
        """按动作类型计算一条轨迹的平均命中率。

        参数：
            normal_dict: 含 ``actions`` 的真实轨迹字典。
            sim_dict: 含 ``actions`` 的模拟轨迹字典；每一步可以是字符串或候选列表。
            k: 仅 ``k == 1`` 时只比较首个候选；其他值（包括 3 和 None）都会
                检查候选列表中的全部动作，因此 Top-3 口径依赖输入列表本身只有
                三项，而不是在此处切片得到前三项。
            sparse: 为 True 时仅比较 Search/Lookup/Finish 等动作类型；为 False
                时比较转为小写后的完整动作字符串（包括参数）。

        返回：
            ``general`` 以及各真实动作类型的平均命中率。索引 0 被固定跳过，
            因而首步不参与本函数的任何准确率统计。
        """
        normal_actions = normal_dict["actions"]
        sim_actions = sim_dict["actions"]
        metric_dict = {"general": 0}
        count_dict = {"general": 0}

        # zip 会以较短列表为准；这里不会主动检查两条轨迹长度是否一致。
        for idx, (na, sa) in enumerate(zip(normal_actions, sim_actions)):
            if idx == 0:
                # 仓库指标口径：无条件忽略轨迹的第一个动作。
                continue

            score = Metrics._score_action_pair(na, sa, k, sparse)
            metric_dict["general"] += score
            count_dict["general"] += 1

            # 分动作类型的分母由真实动作决定；无法解析类型的动作仍计入 general。
            na_name = Metrics.get_action_name(na)
            if na_name is None:
                continue
            if na_name not in metric_dict:
                metric_dict[na_name] = 0
                count_dict[na_name] = 0
            metric_dict[na_name] += score
            count_dict[na_name] += 1

        if count_dict["general"] == 0:
            return metric_dict

        for key in metric_dict:
            metric_dict[key] /= count_dict[key]
        return metric_dict

    @staticmethod
    def _score_action_pair(na, sa, k, sparse):
        """给一个真实动作与一个预测值计分，返回 0 或 1。

        预测值为列表时，Top-1 只看列表首项；其余模式执行“任一候选命中”。
        """
        if isinstance(sa, list):
            if not sa:
                return 0
            if k == 1:
                return Metrics.compare_action(na, sa[0], sparse)
            else:
                return Metrics.compare_actions(na, sa, sparse)
        else:
            return Metrics.compare_action(na, sa, sparse)

    @staticmethod
    def get_actions_metric(dict1, dict2, sparse=False):
        """比较两条等长动作序列并返回逐步平均命中率。

        与 :meth:`get_action_metrics` 不同，此函数不跳过首步，而且要求两条动作
        序列长度完全相同。
        """
        actions1 = dict1["actions"]
        actions2 = dict2["actions"]
        assert len(actions1) == len(actions2), "Different number of actions taken in wiki and guess"
        n = len(actions1)
        score = 0
        for a1, a2 in zip(actions1, actions2):
            score += Metrics.compare_action(a1, a2, sparse)
        return score / n

    @staticmethod
    def compare_actions(na, simulated_actions_list, sparse=False):
        """判断候选列表中是否至少有一个动作命中真实动作。"""
        total_score = 0
        for sa in simulated_actions_list:
            score = Metrics.compare_action(na, sa, sparse)
            total_score += score
        # 候选只要命中一次就记 1；候选数量和重复命中次数不增加分数。
        return 1 if total_score > 0 else 0

    @staticmethod
    def compare_action(action1, action2, sparse=False):
        """比较两个动作；可选择只比较动作类型或比较完整字符串。"""
        action1 = action1.lower()
        action2 = action2.lower()
        if sparse:
            return 1 if Metrics.get_action_name(action1) == Metrics.get_action_name(action2) else 0
        else:
            return 1 if action1 == action2 else 0

    @staticmethod
    def get_action_name(full_action):
        """返回左方括号前的动作类型；没有左方括号时返回 None。"""
        full_action = full_action.strip()
        ind = full_action.find('[')
        if ind < 0:
            return None
        else:
            return full_action[:ind]

    @staticmethod
    def get_avg_time_taken(obs_dict):
        """计算一条轨迹内 Search 步骤的平均记录耗时。

        非 Search 动作不计入分子和分母。若某一步包含动作候选列表，只要列表中
        至少有一个 Search 候选，该步骤就计一次，并使用该步骤唯一的
        ``time_taken`` 值；这不是每个候选各自的执行耗时。
        """
        avg_time = 0
        n = 0
        for i, t in enumerate(obs_dict["time_taken"]):
            if isinstance(obs_dict["actions"][i], str):
                if "search[" in obs_dict["actions"][i].lower():
                    n += 1
                    avg_time += t
            elif isinstance(obs_dict["actions"][i], list):
                for action in obs_dict["actions"][i]:
                    if "search[" in action.lower():
                        n += 1
                        avg_time += t
                        break
        if n == 0:
            return 0
        return avg_time / n

    @staticmethod
    def get_action_specific_avg_metric(dir_path, get_time=False):
        """从样本子目录的已有指标文件计算跨样本平均值。

        每个成功读取 ``metrics.json`` 的样本权重相同，而不是按该样本的动作数
        加权。启用 ``get_time`` 后，Search 时间同样先在样本内求平均，再对样本
        等权求平均。缺失指标文件的目录会被打印并跳过。
        """
        avg_metrics_dict = {
            "general": 0, "Search": 0, "Lookup": 0, "Finish": 0,
            "normal_avg_searchtime": 0, "sim_avg_searchtime": 0,
        }
        metric_names = ["general", "Search", "Lookup", "Finish"]
        n_metrics = 0
        n_times = 0
        for direc in os.listdir(dir_path):
            try:
                metrics_dict = Utils.read_json(os.path.join(dir_path, direc, "metrics.json"))
                if get_time:
                    normal_obs_dict = Utils.read_json(os.path.join(dir_path, direc, "normalobs.json"))
                    sim_obs_dict = Utils.read_json(os.path.join(dir_path, direc, "simobs.json"))
                    normal_avg_searchtime = Metrics.get_avg_time_taken(normal_obs_dict)
                    try:
                        sim_avg_searchtime = Metrics.get_avg_time_taken(sim_obs_dict)
                    except AttributeError:
                        sim_avg_searchtime = 0
                    avg_metrics_dict["normal_avg_searchtime"] += normal_avg_searchtime
                    avg_metrics_dict["sim_avg_searchtime"] += sim_avg_searchtime
                    n_times += 1
            except FileNotFoundError as e:
                print("Filenotfound", os.path.join(dir_path, direc))
                print(e)
                continue
            # 缺失的动作类型按 0 累加；聚合分母仍是所有成功读取的样本数。
            for metric in metric_names:
                avg_metrics_dict[metric] += metrics_dict.get(metric, 0)
            n_metrics += 1

        for key in metric_names:
            try:
                avg_metrics_dict[key] /= n_metrics
            except ZeroDivisionError:
                avg_metrics_dict[key] = None

        if n_times > 0:
            avg_metrics_dict["normal_avg_searchtime"] /= n_times
            avg_metrics_dict["sim_avg_searchtime"] /= n_times

        return avg_metrics_dict, n_metrics

    @staticmethod
    def recalculate_metrics(base_traj_path):
        """从每个样本的真实/模拟轨迹重算并覆盖 ``metrics.json``。

        注意：若目标指标文件已存在，会先删除再写入。默认 ``k=None`` 使用候选
        列表“任一命中”的口径，并不同时生成独立的 Top-1 指标。
        """
        for folder_name in os.listdir(base_traj_path):
            save_dir = os.path.join(base_traj_path, str(folder_name))
            normal_observations_dict = Utils.read_json(os.path.join(save_dir, "normalobs.json"))
            sim_observations_dict = Utils.read_json(os.path.join(save_dir, "simobs.json"))
            metrics_file_path = os.path.join(save_dir, "metrics.json")
            if os.path.exists(metrics_file_path):
                os.remove(metrics_file_path)
            metric_dict = Metrics.get_action_metrics(normal_observations_dict, sim_observations_dict, sparse=False)
            Utils.save_json(metric_dict, metrics_file_path)

    @staticmethod
    def cum_metrics(base_traj_path):
        """收集每个样本的 Top-1、任一候选命中率和 Search 平均耗时。

        返回值保留逐样本列表，供后续计算均值、标准差或标准误。这里传入
        ``k=3`` 时不会截断候选列表，所谓 Top-3 仍由保存轨迹的候选数量保证。
        """
        avg_metrics_dict = {}
        n_metrics = 0
        metric_names = ["general", "Search", "Lookup", "Finish"]
        for metric in metric_names:
            avg_metrics_dict[metric + "_top1"] = []
            avg_metrics_dict[metric + "_top3"] = []
        avg_metrics_dict["normal_avg_searchtime"] = []
        avg_metrics_dict["sim_avg_searchtime"] = []

        for folder_name in os.listdir(base_traj_path):
            save_dir = os.path.join(base_traj_path, str(folder_name))
            normal_observations_dict = Utils.read_json(os.path.join(save_dir, "normalobs.json"))
            sim_observations_dict = Utils.read_json(os.path.join(save_dir, "simobs.json"))
            metric_dict_top1 = Metrics.get_action_metrics(
                normal_observations_dict, sim_observations_dict, k=1, sparse=False
            )
            metric_dict_top3 = Metrics.get_action_metrics(
                normal_observations_dict, sim_observations_dict, k=3, sparse=False
            )
            for metric in metric_names:
                avg_metrics_dict[metric + "_top1"].append(metric_dict_top1.get(metric, 0))
                avg_metrics_dict[metric + "_top3"].append(metric_dict_top3.get(metric, 0))
            avg_metrics_dict["normal_avg_searchtime"].append(Metrics.get_avg_time_taken(normal_observations_dict))
            avg_metrics_dict["sim_avg_searchtime"].append(Metrics.get_avg_time_taken(sim_observations_dict))
            n_metrics += 1

        return avg_metrics_dict, n_metrics
