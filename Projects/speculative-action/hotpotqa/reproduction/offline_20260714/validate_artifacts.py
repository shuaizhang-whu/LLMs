"""独立核验本目录记录的离线复现产物，不修改任何结果文件。"""

import csv
import json
from pathlib import Path
import re
import sys

import matplotlib.image as mpimg


HOTPOTQA_ROOT = Path(__file__).resolve().parents[2]
REPRO_ROOT = Path(__file__).resolve().parent
RUN_METRICS_ROOT = HOTPOTQA_ROOT / "run_metrics"
sys.path.insert(0, str(HOTPOTQA_ROOT))

from src.metrics import Metrics  # noqa: E402


def read_json(path):
    """读取一个 UTF-8 JSON 文件。"""
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    """验证聚合结构、逐样本指标、图片、网络跟踪和源文件哈希。"""
    aggregate = read_json(RUN_METRICS_ROOT / "all_avg_metrics_top3.json")
    cumulative = read_json(RUN_METRICS_ROOT / "list_metrics_top1_3.json")
    verification = read_json(REPRO_ROOT / "verification.json")
    with (RUN_METRICS_ROOT / "all_avg_metrics_top3.csv").open(newline="") as handle:
        csv_rows = list(csv.DictReader(handle))

    pairs = [(actor, guess) for actor, guesses in aggregate.items() for guess in guesses]
    assert len(pairs) == len(csv_rows) == 12
    csv_by_pair = {(row["agent model"], row["guess model"]): row for row in csv_rows}

    sample_count = 0
    metric_mismatches = 0
    top1_top3_violations = 0
    candidate_lengths = {}
    max_aggregate_error = 0.0
    for actor, guess in pairs:
        values = cumulative[actor][guess]
        for metric in ("general", "Search", "Lookup", "Finish"):
            top1_values = values[f"{metric}_top1"]
            top3_values = values[f"{metric}_top3"]
            top1_top3_violations += sum(
                top1 > top3 + 1e-12
                for top1, top3 in zip(top1_values, top3_values)
            )
            expected = sum(top3_values) / len(top3_values)
            max_aggregate_error = max(
                max_aggregate_error,
                abs(aggregate[actor][guess][metric] - expected),
            )

        csv_row = csv_by_pair[(actor, guess)]
        for metric_name, metric_value in aggregate[actor][guess].items():
            assert abs(float(csv_row[metric_name]) - metric_value) < 1e-12

        trajectory_root = (
            RUN_METRICS_ROOT
            / f"agent_{actor.split('/')[-1]}_top3"
            / f"trajs_{guess.split('/')[-1]}"
        )
        for sample_dir in sorted(path for path in trajectory_root.iterdir() if path.is_dir()):
            normal = read_json(sample_dir / "normalobs.json")
            simulated = read_json(sample_dir / "simobs.json")
            saved = read_json(sample_dir / "metrics.json")
            recalculated = Metrics.get_action_metrics(normal, simulated, k=3)
            if recalculated.keys() != saved.keys() or any(
                abs(recalculated[name] - saved[name]) >= 1e-12 for name in saved
            ):
                metric_mismatches += 1
            for candidates in simulated["actions"]:
                length = len(candidates)
                candidate_lengths[length] = candidate_lengths.get(length, 0) + 1
            sample_count += 1

    assert sample_count == 48
    assert metric_mismatches == 0
    assert top1_top3_violations == 0
    assert max_aggregate_error < 1e-12
    assert candidate_lengths == {1: 16, 2: 1, 3: 165, 4: 1}

    images = sorted((RUN_METRICS_ROOT / "images").glob("*.png"))
    assert len(images) == 6
    for image in images:
        assert image.stat().st_size > 0
        assert mpimg.imread(image).size > 0

    before = (REPRO_ROOT / "source_sha256_before.txt").read_bytes()
    after = (REPRO_ROOT / "source_sha256_after.txt").read_bytes()
    assert before == after
    assert len(before.splitlines()) == 376

    network_trace = (REPRO_ROOT / "network_trace.log").read_text(encoding="utf-8")
    assert not re.search(r"(?:socket|connect|sendto|recvfrom)\(", network_trace)
    assert verification["tests"]["run"] == 11

    print("aggregate_pairs=12")
    print("csv_rows=12")
    print("complete_top3_samples=48")
    print("saved_metric_mismatches=0")
    print("top1_gt_top3_violations=0")
    print(f"max_aggregate_error={max_aggregate_error:.1f}")
    print("decoded_png_files=6")
    print("network_syscalls_observed=0")
    print("source_hash_manifests_identical=true")
    print("source_files_hashed=376")


if __name__ == "__main__":
    main()
