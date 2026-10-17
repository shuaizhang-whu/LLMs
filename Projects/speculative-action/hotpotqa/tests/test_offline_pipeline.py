"""仓库附带 HotPotQA 结果与纯离线 CLI 的集成回归测试。"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import run
from src.grapher import Grapher
from src.metrics import Metrics


HOTPOTQA_ROOT = Path(__file__).resolve().parents[1]
API_KEY_NAMES = (
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "OPENROUTER_API_KEY",
)


class OfflineCliTest(unittest.TestCase):
    """证明指标入口不加载 Runner，并保持官方 CLI 的输出矩阵。"""

    def test_importing_cli_does_not_import_runner_or_llm_client(self):
        # 在独立解释器中检查 import run，避免其他测试预先导入 llm_client 后
        # 污染当前进程的 sys.modules，使结果依赖 unittest 的模块加载顺序。
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; import run; "
                    "assert 'src.runner' not in sys.modules; "
                    "assert 'src.llm_client' not in sys.modules"
                ),
            ],
            cwd=HOTPOTQA_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_base_traj_path_matches_existing_naming_rule(self):
        self.assertEqual(
            run.get_base_traj_path("openai/gpt-4", "google/gemini-2.5-flash"),
            "./run_metrics/agent_gpt-4_top3/trajs_gemini-2.5-flash",
        )

    def test_graph_output_parent_is_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "run_metrics" / "images" / "figure.png"

            Grapher._prepare_output_path(str(output_path))

            self.assertTrue(output_path.parent.is_dir())
            self.assertFalse(output_path.exists())

    def test_metric_commands_run_without_keys_or_network(self):
        for metric_flag in ("--getmetric", "--getmetric2"):
            with self.subTest(metric_flag=metric_flag), tempfile.TemporaryDirectory() as tmp:
                env = os.environ.copy()
                for key_name in API_KEY_NAMES:
                    env.pop(key_name, None)
                env.update(
                    {
                        "PYTHONDONTWRITEBYTECODE": "1",
                        "MPLCONFIGDIR": tmp,
                        # 若离线代码意外发起 HTTP 请求，应立即连接失败而非访问外网。
                        "HTTP_PROXY": "http://127.0.0.1:9",
                        "HTTPS_PROXY": "http://127.0.0.1:9",
                        "ALL_PROXY": "http://127.0.0.1:9",
                        "NO_PROXY": "",
                    }
                )

                completed = subprocess.run(
                    [sys.executable, "run.py", "--norun", metric_flag],
                    cwd=HOTPOTQA_ROOT,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )

                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertEqual(completed.stdout.count("AVERAGE METRICS"), 12)
                self.assertEqual(completed.stdout.count("does not exist. Skipping"), 3)


class AttachedResultsTest(unittest.TestCase):
    """逐样本核验附带 Top-3 轨迹，没有重写任何源文件。"""

    def test_all_complete_top3_samples_match_saved_metrics(self):
        existing_pairs = 0
        missing_pairs = 0
        sample_count = 0
        candidate_lengths = {}

        for actor in run.AGENT_MODEL_NAMES:
            for speculator in run.GUESS_MODEL_NAMES:
                relative_path = run.get_base_traj_path(actor, speculator)
                trajectory_root = HOTPOTQA_ROOT / relative_path
                if not trajectory_root.exists():
                    missing_pairs += 1
                    continue

                existing_pairs += 1
                for sample_dir in sorted(path for path in trajectory_root.iterdir() if path.is_dir()):
                    with (sample_dir / "normalobs.json").open(encoding="utf-8") as handle:
                        normal = json.load(handle)
                    with (sample_dir / "simobs.json").open(encoding="utf-8") as handle:
                        simulated = json.load(handle)
                    with (sample_dir / "metrics.json").open(encoding="utf-8") as handle:
                        saved = json.load(handle)

                    # 仓库附带 metrics.json 来自旧版“检查全部候选”的兼容口径。
                    recalculated = Metrics.get_action_metrics(normal, simulated, k=None)
                    self.assertEqual(recalculated.keys(), saved.keys(), sample_dir)
                    for metric_name, expected in saved.items():
                        self.assertAlmostEqual(
                            recalculated[metric_name], expected, places=12, msg=sample_dir
                        )

                    top1 = Metrics.get_action_metrics(normal, simulated, k=1)
                    strict_top3 = Metrics.get_action_metrics(normal, simulated, k=3)
                    for metric_name in ("general", "Search", "Lookup", "Finish"):
                        self.assertLessEqual(
                            top1.get(metric_name, 0),
                            strict_top3.get(metric_name, 0),
                            sample_dir,
                        )
                        self.assertLessEqual(
                            strict_top3.get(metric_name, 0),
                            recalculated.get(metric_name, 0),
                            sample_dir,
                        )
                    for candidates in simulated["actions"]:
                        if isinstance(candidates, list):
                            length = len(candidates)
                            candidate_lengths[length] = candidate_lengths.get(length, 0) + 1
                    sample_count += 1

        self.assertEqual(existing_pairs, 12)
        self.assertEqual(missing_pairs, 3)
        self.assertEqual(sample_count, 48)
        self.assertEqual(candidate_lengths, {1: 16, 2: 1, 3: 165, 4: 1})


if __name__ == "__main__":
    unittest.main()
