"""将 HotPotQA 已聚合指标绘制为柱状图。

绘图函数不负责读取或验证轨迹，而是接收调用方已经组织好的字典/列表。所有
输出路径均相对当前工作目录，保存时会自动创建父目录，并在写入前删除同名图片；
调用方仍需确认不会覆盖需要保留的图。
"""

import matplotlib.pyplot as plt
import os
import numpy as np

from . import constants
from .utils import Utils


class Grapher:
    """为不同 Actor/Guess 模型组合生成时间与动作命中率对比图。"""

    @staticmethod
    def _prepare_output_path(filename):
        """创建图片父目录，并保持原实现“先删除同名图再保存”的行为。"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        if os.path.exists(filename):
            os.remove(filename)

    @staticmethod
    def graph_agent_times(data, agent, save_graph=True):
        """绘制指定 Actor 下各 Guess 模型的平均 Search 时间。

        ``data`` 需采用 ``data[actor][guess_model][metric_name]`` 的嵌套结构，
        并包含 ``normal_avg_searchtime`` 与 ``sim_avg_searchtime``。这两个值是
        Search-only 的聚合耗时，而不是完整任务的端到端延迟。
        """
        agent_keys = data[agent].keys()
        guess_models = [key.split('/')[-1] for key in agent_keys]
        normal_times = []
        sim_times = []
        for gm in agent_keys:
            normal_times.append(data[agent][gm]["normal_avg_searchtime"])
            sim_times.append(data[agent][gm]["sim_avg_searchtime"])

        assert len(normal_times) == len(sim_times) == len(guess_models)
        width = 0.35
        plt.figure(figsize=(10, 5))
        # 两组柱使用同一横坐标而非左右错开，后绘制的 normal 柱可能遮住 sim 柱。
        plt.bar(guess_models, sim_times, width=width, label='Sim Avg Search Time')
        plt.bar(guess_models, normal_times, width=width, label='Normal Avg Search Time')
        plt.ylabel('Time (seconds)')
        plt.title(f'Comparison of Normal vs Sim Avg Search Time for Agent: {agent}')
        plt.xticks(rotation=0)
        plt.legend()
        plt.tight_layout()
        if save_graph:
            # 文件名依赖全局 guess_num_actions，且同名旧图会被直接删除。
            filename = f"./run_metrics/images/AgentTimes_{agent.split('/')[-1]}_top{constants.guess_num_actions}.png"
            Grapher._prepare_output_path(filename)
            plt.savefig(filename, dpi=300)
        plt.close()

    @staticmethod
    def graph_metric_comparison(data, save_graph=True):
        """从扁平聚合记录绘制各 Actor 的 Top-1/Top-3 命中率。

        ``data`` 应是字典列表，每项至少包含 ``agent_model``、``guess_model``、
        ``general_top1`` 与 ``general_top3``。函数只为代码中固定的三种 Actor
        建图；空值命中率会在绘图时按 0 处理。
        """
        # 该白名单同时决定 good_dict 的键；出现其他 Actor 名会触发 KeyError。
        agents = ["google/gemini-2.5-flash", "openai/gpt-4", "openai/gpt-5"]
        good_dict = {}
        width = 0.35
        for agent in agents:
            good_dict[agent] = {}

        for d in data:
            agent = d["agent_model"]
            good_dict[agent][d["guess_model"]] = d

        for agent in good_dict.keys():
            agent_name = agent.split('/')[-1]
            guess_models = []
            top1_accuracies = []
            top3_accuracies = []
            for gm in good_dict[agent].keys():
                top1 = good_dict[agent][gm]["general_top1"]
                top3 = good_dict[agent][gm]["general_top3"]
                if top1 is None:
                    top1 = 0
                if top3 is None:
                    top3 = 0
                guess_models.append(gm.split('/')[-1])
                top1_accuracies.append(top1)
                top3_accuracies.append(top3)

            plt.figure(figsize=(10, 5))
            x = np.arange(len(guess_models))
            plt.bar(x - width / 2, top3_accuracies, width=width, label='Top 3 Accuracies')
            plt.bar(x + width / 2, top1_accuracies, width=width, label='Top 1 Accuracies')
            plt.ylabel('Accuracy')
            plt.title(f'Comparison of Top 1 and 3 Accuracies for Agent: {agent_name}')
            plt.xticks(x, guess_models, rotation=0)
            plt.legend()
            plt.tight_layout()
            if save_graph:
                # 输出路径相对进程当前目录；已有同名文件会先被删除。
                filename = f"./run_metrics/images/AgentComparison_{agent_name}.png"
                Grapher._prepare_output_path(filename)
                plt.savefig(filename, dpi=300)
            plt.close()

    @staticmethod
    def graph_metric3(data, save_graph=True):
        """用逐样本指标绘制带标准误差条的 Top-1/Top-3 百分比图。

        ``data`` 需采用 ``data[actor][guess_model][metric_name]`` 的嵌套结构，
        其中 ``general_top1`` 和 ``general_top3`` 都必须是逐样本数值列表。
        图中误差条由样本标准差（``ddof=1``）除以样本数平方根得到，实际表示
        标准误；仅一个样本时 NumPy 会产生 ``nan``。
        """
        agents = ["google/gemini-2.5-flash", "openai/gpt-4", "openai/gpt-5"]
        width = 0.4

        for agent in agents:
            agent_name = agent.split('/')[-1]
            top1_accuracies = []
            top3_accuracies = []
            top1_stds = []
            top3_stds = []
            guess_models = []
            for gm in data[agent].keys():
                top1_list = data[agent][gm]["general_top1"]
                top3_list = data[agent][gm]["general_top3"]
                top1_std = np.std(top1_list, ddof=1)
                top3_std = np.std(top3_list, ddof=1)
                top1_metric = Utils.avg(top1_list)
                top3_metric = Utils.avg(top3_list)
                # 将样本标准差换算成标准误，随后连同均值一起转换为百分比。
                top1_std /= np.sqrt(len(top1_list))
                top3_std /= np.sqrt(len(top3_list))
                top1_accuracies.append(top1_metric)
                top3_accuracies.append(top3_metric)
                top1_stds.append(top1_std)
                top3_stds.append(top3_std)
                guess_models.append(gm.split('/')[-1])

            top3_percentages = [v * 100 for v in top3_accuracies]
            top1_percentages = [v * 100 for v in top1_accuracies]
            top1_std_percentages = [v * 100 for v in top1_stds]
            top3_std_percentages = [v * 100 for v in top3_stds]

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.margins(y=0.2)
            x = np.arange(len(guess_models))
            bars1 = ax.bar(x - width / 2, top3_percentages, width=width, yerr=top3_std_percentages, capsize=5, label='Top 3 Accuracies', alpha=0.6)
            bars2 = ax.bar(x + width / 2, top1_percentages, width=width, yerr=top1_std_percentages, capsize=5, label='Top 1 Accuracies', alpha=0.6)
            ax.set_ylabel('Accuracy', fontsize=20)
            ax.set_xticks(x)
            ax.set_xticklabels(guess_models, rotation=0, fontsize=20)
            ax.legend(fontsize=16)
            ax.grid(True, axis="y", linestyle="--", alpha=0.5)
            ax.bar_label(bars1, labels=[f"{v:.2f}%" for v in top3_percentages], padding=3, fontsize=18)
            ax.bar_label(bars2, labels=[f"{v:.2f}%" for v in top1_percentages], padding=3, fontsize=18)
            plt.tight_layout()

            if save_graph:
                # 固定文件名会覆盖此前针对同一 Actor 生成的同名图片。
                filename = f"./run_metrics/images/AgentComparisonProper2_{agent_name}new2.png"
                Grapher._prepare_output_path(filename)
                plt.savefig(filename, dpi=300)
            plt.close()
