"""HotPotQA 子项目的命令行入口。

这个文件负责三类工作：启动轨迹采集、汇总已有轨迹的指标、根据汇总文件绘图。
轨迹采集需要构造 ``HotPotQARun`` 和模型客户端；纯指标与绘图入口则延迟导入
Runner，因此在传入 ``--norun`` 后不需要 API Key，也不会初始化任何模型客户端。
"""

import os
import json
import argparse
from os.path import join

from src import constants
from src.utils import Utils
from src.metrics import Metrics
from src.grapher import Grapher


AGENT_MODEL_NAMES = [
    "google/gemini-2.5-flash",
    "openai/gpt-4",
    "openai/gpt-5",
]

# 指标汇总使用的固定模型矩阵。这里的名称同时决定 run_metrics 下的目录名；
# 它不会根据命令行中单独传入的 --modelname/--guessmodelname 自动扩展。
GUESS_MODEL_NAMES = [
    "openai/gpt-5-nano",
    "openai/gpt-4.1-nano",
    "openai/gpt-3.5-turbo",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-flash-lite",
]


def get_base_traj_path(agent_model, guess_model):
    """根据模型名生成与 ``HotPotQARun`` 完全一致的轨迹目录。"""
    return join(
        "./run_metrics",
        f"agent_{agent_model.split('/')[-1]}_top{constants.guess_num_actions}",
        f"trajs_{guess_model.split('/')[-1]}",
    )


def compute_metrics(runner=None, save=False):
    """读取逐样本 ``metrics.json``，按 Actor/Guess 模型组合求宏平均。

    该函数不会从 normalobs/simobs 重新计算逐步命中；它只汇总已经保存的
    metrics.json，并附带统计两类 Search step 的平均耗时。``runner`` 仅为兼容
    原有位置参数而保留，离线计算不再读取或修改 Runner。
    """
    all_avg_metrics = {}
    for agent in AGENT_MODEL_NAMES:
        all_avg_metrics[agent] = {}
        for guess_model in GUESS_MODEL_NAMES:
            # 路径约定：run_metrics/agent_<actor>_topK/trajs_<guess_model>。
            base_traj_path = get_base_traj_path(agent, guess_model)
            if not os.path.exists(base_traj_path):
                print(f"Path {base_traj_path} does not exist. Skipping...")
                continue
            avg_metrics_dict, n_samples = Metrics.get_action_specific_avg_metric(
                base_traj_path, get_time=True
            )
            print(
                f"AVERAGE METRICS for agent {agent} using guess model {guess_model}:\n",
                json.dumps(avg_metrics_dict, indent=4),
                f"\nfor {n_samples} observations",
            )
            all_avg_metrics[agent][guess_model] = avg_metrics_dict
    metrics_df = Utils.convert_json_to_csv(all_avg_metrics)
    if save:
        # 以固定文件名写入；重复运行会覆盖同名聚合文件。
        Utils.save_json(
            all_avg_metrics,
            join("./run_metrics", f"all_avg_metrics_top{constants.guess_num_actions}.json"),
        )
        Utils.save_file(
            metrics_df.to_csv(index=False),
            join("./run_metrics", f"all_avg_metrics_top{constants.guess_num_actions}.csv"),
        )


def compute_cumulative_metrics(runner=None, save=False):
    """从 normalobs/simobs 重新计算每个样本的 Top-1 与 Top-3 指标列表。

    函数名中的 cumulative 指“保留逐样本列表供后续统计”，不是按时间累加。
    Top-1/Top-3 的具体匹配语义由 ``Metrics.cum_metrics`` 定义。``runner`` 仅为
    兼容原有位置参数而保留，离线计算不再读取或修改 Runner。
    """
    all_avg_metrics = {}
    for agent in AGENT_MODEL_NAMES:
        all_avg_metrics[agent] = {}
        for guess_model in GUESS_MODEL_NAMES:
            base_traj_path = get_base_traj_path(agent, guess_model)
            if not os.path.exists(base_traj_path):
                print(f"Path {base_traj_path} does not exist. Skipping...")
                continue
            avg_metrics_dict, n_samples = Metrics.cum_metrics(base_traj_path)
            print(
                f"AVERAGE METRICS for agent {agent} using guess model {guess_model}:\n",
                json.dumps(avg_metrics_dict, indent=4),
                f"\nfor {n_samples} observations",
            )
            all_avg_metrics[agent][guess_model] = avg_metrics_dict
    if save:
        Utils.save_json(all_avg_metrics, join("./run_metrics", "list_metrics_top1_3.json"))


def main():
    """解析 CLI 参数，并按“运行、指标、绘图”的顺序执行所选操作。"""
    parser = argparse.ArgumentParser(description="HotPotQA Speculative Actions Runner")
    parser.add_argument("--getmetric", action="store_true", help="Compute and print average metrics")
    parser.add_argument("--getmetric2", action="store_true", help="Compute and print cumulative metrics")
    parser.add_argument("--savemetrics", action="store_true", help="Save computed metrics to disk")
    parser.add_argument("--graph", action="store_true", help="Plot agent time comparison graphs")
    parser.add_argument("--graph2", action="store_true", help="Plot top-1 vs top-3 comparison graphs")
    parser.add_argument("--graph3", action="store_true", help="Plot detailed metric comparison graphs")
    # 这两个选项使用 store_false：未传参数时变量为 True，显式传入后才变为 False。
    # 因而遗漏 --norun 会真正启动实验，而不是只处理已有结果。
    parser.add_argument("--noprint", action="store_false", help="Suppress command line output")
    parser.add_argument("--norun", action="store_false", help="Skip running the experiment")
    parser.add_argument("--modelname", default=constants.openrouter_model_name, help="Agent model name")
    parser.add_argument("--guessmodelname", default=constants.openrouter_guess_model_name, help="Guess model name")
    parser.add_argument("--cleanuptrajs", action="store_true", help="Clean up incomplete trajectories")
    args = parser.parse_args()

    if args.norun:
        # HotPotQARun 会进一步导入并初始化模型客户端，只允许在真实运行分支中构造。
        from src.runner import HotPotQARun

        runner = HotPotQARun(
            model_name=args.modelname,
            guess_model_name=args.guessmodelname,
            to_print_output=args.noprint,
        )
        if args.cleanuptrajs:
            Utils.cleanup_trajs(runner.base_traj_path)
        # webthink_simulate=True 表示同时采集 normal/sim 两套轨迹；runner 内部仍按
        # 串行顺序执行，并不在这里创建 Actor/工具调用的在线并发流水线。
        runner.run(webthink_simulate=True, skip_done=True)
        Utils.cleanup_trajs(runner.base_traj_path)
    elif args.cleanuptrajs:
        # 仅清理已有目录时不应为获得路径而初始化任何模型客户端。
        Utils.cleanup_trajs(get_base_traj_path(args.modelname, args.guessmodelname))

    if args.getmetric:
        compute_metrics(save=args.savemetrics)

    if args.getmetric2:
        compute_cumulative_metrics(save=args.savemetrics)

    if args.graph:
        data = Utils.read_json(
            join("./run_metrics", f"all_avg_metrics_top{constants.guess_num_actions}.json")
        )
        for agent in data.keys():
            Grapher.graph_agent_times(data, agent=agent)

    if args.graph2:
        data = Utils.read_json(join("./run_metrics", "comparision_top1_top3.json"))
        Grapher.graph_metric_comparison(data)

    if args.graph3:
        data = Utils.read_json(join("./run_metrics", "list_metrics_top1_3.json"))
        Grapher.graph_metric3(data)


if __name__ == "__main__":
    main()
