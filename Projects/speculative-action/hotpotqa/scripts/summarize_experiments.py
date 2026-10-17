"""命令行生成本地实验对照图和可选 lossless 审计。"""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment_report import generate_comparison, validate_lossless_pair
from src.utils import Utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", required=True, help="Comma-separated run roots")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sequential-root")
    parser.add_argument("--online-root")
    args = parser.parse_args()

    run_dirs = [item.strip() for item in args.run_dirs.split(",") if item.strip()]
    comparison = generate_comparison(run_dirs, args.output_dir)
    print(json.dumps(comparison, indent=2))
    if bool(args.sequential_root) != bool(args.online_root):
        parser.error("--sequential-root and --online-root must be provided together")
    if args.sequential_root:
        lossless = validate_lossless_pair(args.sequential_root, args.online_root)
        output_path = Path(args.output_dir) / "lossless_validation.json"
        Utils.save_json(lossless, str(output_path))
        print(json.dumps(lossless, indent=2))
        if not lossless["passed"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
