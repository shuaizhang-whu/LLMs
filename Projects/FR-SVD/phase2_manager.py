#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
import threading
from dataclasses import dataclass, asdict
from datetime import datetime
from glob import glob
from typing import Dict, List, Optional, Tuple


REPO_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE2_LOG_DIR = os.path.join(REPO_DIR, "result", "phase2")
RESULT_SUMMARY_FILE = os.path.join(PHASE2_LOG_DIR, "RESULT_SUMMARY.md")
ROUND_STATE_FILE = os.path.join(PHASE2_LOG_DIR, "CURRENT_ROUND_STATE.json")
PYTHON_BIN = "/data/zhangs/anaconda3/envs/compress/bin/python"
SCRIPT_PATH = os.path.join(REPO_DIR, "FR_SVD_v1.3.py")
MODEL_PATH = "/data/zhangs/LLMs/Models/jeffwan/llama-7b-hf"
PPL_RE = re.compile(r"PPL after pruning: \{'wikitext2': ([0-9.]+)\}")
AVG_BETA_RE = re.compile(r"平均 beta:\s*([0-9.]+)")
AVG_DH_RE = re.compile(r"平均 D/H:\s*([0-9.]+)")
RUNTIME_RE = re.compile(r"Completed successfully in (\d+) minutes")


@dataclass
class Experiment:
    name: str
    rho: float
    alpha_min: float
    alpha_max: float
    r1_ratio: float
    lambda_psi: float
    nsamples: int
    gpu: int
    priority: str
    expected_gain: str
    expected_minutes: int = 18
    fisher_mode: str = "none"
    seed: int = 3
    ratio: float = 0.2
    eps: float = 1e-4
    model_seq_len: int = 2048


def alpha_tag(value: float) -> str:
    s = f"{value:.2f}".rstrip("0").rstrip(".")
    return s.replace(".", "")


def format_exp_name(nsamples: int, rho: float, alpha_min: float, alpha_max: float,
                    r1_ratio: float, lambda_psi: float) -> str:
    lam = f"{lambda_psi:.0e}".replace("+0", "").replace("+", "")
    lam = lam.replace("e-0", "e-")
    return (
        f"ns{nsamples}_rho{rho:.2f}_a{alpha_tag(alpha_min)}_{alpha_tag(alpha_max)}"
        f"_r1_{r1_ratio:.2f}_lam{lam}"
    )


def build_round_plan() -> List[Experiment]:
    specs = [
        (512, 0.41, 0.45, 0.55, 0.95, 1e-5, 6, "P0", "直接验证 rho 左邻域是否优于 0.42"),
        (512, 0.42, 0.46, 0.54, 0.95, 1e-5, 7, "P0", "围绕当前最优做更窄 alpha exploitation"),
        (512, 0.43, 0.45, 0.55, 0.95, 1e-5, 6, "P0", "验证 rho 右邻域的单调性"),
        (512, 0.44, 0.45, 0.55, 0.95, 1e-5, 7, "P1", "检验 0.44 是否存在次优盆地"),
        (512, 0.42, 0.44, 0.56, 0.94, 1e-5, 6, "P1", "测试 r1=0.94 的稳定带宽"),
        (512, 0.43, 0.46, 0.54, 0.94, 1e-5, 7, "P1", "联合微调 rho/r1，复核最优区间稳定性"),
        (256, 0.42, 0.46, 0.54, 0.95, 1e-5, 6, "P2", "对照 nsamples 敏感性，确认 512 收益真实"),
        (512, 0.42, 0.45, 0.55, 0.92, 1e-4, 7, "P2", "保留 lambda/r1 对照，隔离噪声来源"),
    ]
    plan = []
    for ns, rho, amin, amax, r1, lam, gpu, pri, gain in specs:
        plan.append(
            Experiment(
                name=format_exp_name(ns, rho, amin, amax, r1, lam),
                rho=rho,
                alpha_min=amin,
                alpha_max=amax,
                r1_ratio=r1,
                lambda_psi=lam,
                nsamples=ns,
                gpu=gpu,
                priority=pri,
                expected_gain=gain,
            )
        )
    return plan


def run_cmd(cmd: List[str], cwd: Optional[str] = None, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=capture,
        check=False,
    )


def get_gpu_free_mib(gpu: int) -> Optional[int]:
    result = run_cmd(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    if result.returncode != 0:
        return None
    text = result.stdout.strip().splitlines()
    if not text:
        return None
    try:
        return int(text[0].strip())
    except ValueError:
        return None


def wait_for_gpu(gpu: int, min_free_mib: int = 18000, poll_seconds: int = 30) -> None:
    while True:
        free_mib = get_gpu_free_mib(gpu)
        if free_mib is not None and free_mib >= min_free_mib:
            return
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{stamp}] GPU{gpu} 可用显存不足，当前 free={free_mib} MiB，继续等待...", flush=True)
        time.sleep(poll_seconds)


def parse_tag_metadata(name: str) -> Dict[str, Optional[float]]:
    data: Dict[str, Optional[float]] = {
        "nsamples": None,
        "rho": None,
        "alpha_min": None,
        "alpha_max": None,
        "r1_ratio": None,
        "lambda_psi": None,
    }
    m = re.search(r"ns(\d+)", name)
    if m:
        data["nsamples"] = int(m.group(1))
    m = re.search(r"rho([0-9.]+)", name)
    if m:
        token = m.group(1)
        if "." in token:
            data["rho"] = float(token)
        elif len(token) <= 2:
            data["rho"] = int(token) / 10.0
        else:
            data["rho"] = int(token) / 100.0
    m = re.search(r"_a(\d+)_([0-9]+)", name)
    if m:
        def decode_alpha(token: str) -> float:
            if len(token) <= 2:
                return int(token) / 10.0
            return int(token) / 100.0
        data["alpha_min"] = decode_alpha(m.group(1))
        data["alpha_max"] = decode_alpha(m.group(2))
    m = re.search(r"r1_([0-9.]+)", name)
    if m:
        data["r1_ratio"] = float(m.group(1))
    m = re.search(r"lam(1e-?\d+)", name)
    if m:
        token = m.group(1)
        if "e-" not in token:
            token = token.replace("e", "e-")
        data["lambda_psi"] = float(token)
    return data


def classify_status(text: str, ppl: Optional[float]) -> str:
    lower = text.lower()
    if ppl is not None:
        return "completed"
    if "out of memory" in lower or "cuda oom" in lower:
        return "oom"
    if "keyboardinterrupt" in lower:
        return "interrupted"
    if "traceback" in lower or "error" in lower or "failed" in lower:
        return "failed"
    if "running perplexity" in lower or "运行困惑度评估" in lower:
        return "incomplete"
    return "no_ppl"


def parse_experiment_log(path: str) -> Dict[str, object]:
    name = os.path.basename(path).rsplit(".", 1)[0]
    with open(path, "r", errors="ignore") as f:
        text = f.read()
    ppls = [float(x) for x in PPL_RE.findall(text)]
    ppl = ppls[-1] if ppls else None
    status = classify_status(text, ppl)
    meta = parse_tag_metadata(name)
    avg_beta = None
    avg_dh = None
    runtime_min = None
    samples_match = re.findall(r"(?:样本数|Samples):\s*([0-9]+)", text)
    if samples_match and meta["nsamples"] is None:
        meta["nsamples"] = int(samples_match[-1])
    rho_match = re.findall(r"rho[=:]\s*([0-9.]+)", text)
    if rho_match and meta["rho"] is None:
        meta["rho"] = float(rho_match[-1])
    alpha_match = re.findall(r"Alpha(?: 范围)?[:=]\s*\[([0-9.]+),\s*([0-9.]+)\]", text)
    if alpha_match and (meta["alpha_min"] is None or meta["alpha_max"] is None):
        meta["alpha_min"] = float(alpha_match[-1][0])
        meta["alpha_max"] = float(alpha_match[-1][1])
    r1_match = re.findall(r"r1_ratio[:=]\s*([0-9.]+)", text)
    if r1_match and meta["r1_ratio"] is None:
        meta["r1_ratio"] = float(r1_match[-1])
    lam_match = re.findall(r"lambda_psi[:=]\s*([0-9.eE-]+)", text)
    if lam_match and meta["lambda_psi"] is None:
        meta["lambda_psi"] = float(lam_match[-1])
    if meta["nsamples"] is None:
        meta["nsamples"] = 128
    if meta["lambda_psi"] is None:
        meta["lambda_psi"] = 1e-4
    m = AVG_BETA_RE.findall(text)
    if m:
        avg_beta = float(m[-1])
    m = AVG_DH_RE.findall(text)
    if m:
        avg_dh = float(m[-1])
    m = RUNTIME_RE.findall(text)
    if m:
        runtime_min = int(m[-1])
    return {
        "name": name,
        "path": path,
        "ppl": ppl,
        "status": status,
        "avg_beta": avg_beta,
        "avg_dh": avg_dh,
        "runtime_min": runtime_min,
        **meta,
    }


def collect_phase2_results() -> Dict[str, object]:
    log_paths = sorted(glob(os.path.join(PHASE2_LOG_DIR, "*.log")))
    experiment_logs = []
    auxiliary_logs = []
    for path in log_paths:
        name = os.path.basename(path)
        if any(key in name for key in ("nohup", "monitor")):
            auxiliary_logs.append(path)
        else:
            experiment_logs.append(path)

    results = [parse_experiment_log(path) for path in experiment_logs]
    completed = [r for r in results if r["ppl"] is not None]
    failed = [r for r in results if r["status"] in ("oom", "failed", "interrupted")]
    unfinished = [r for r in results if r["status"] in ("incomplete", "no_ppl")]
    top20 = sorted(completed, key=lambda x: x["ppl"])[:20]
    best = top20[0] if top20 else None
    return {
        "total_logs": len(log_paths),
        "experiment_logs": len(experiment_logs),
        "auxiliary_logs": len(auxiliary_logs),
        "extractable_ppl": len(completed),
        "failed_count": len(failed),
        "unfinished_count": len(unfinished),
        "best": best,
        "top20": top20,
        "results": results,
        "failed": failed,
        "unfinished": unfinished,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def load_round_state() -> Dict[str, object]:
    if os.path.exists(ROUND_STATE_FILE):
        with open(ROUND_STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "planned": [asdict(exp) for exp in build_round_plan()],
        "started": [],
        "completed": [],
        "failed": [],
    }


def save_round_state(state: Dict[str, object]) -> None:
    with open(ROUND_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def current_findings(results: Dict[str, object]) -> List[str]:
    lines = []
    best = results["best"]
    if best:
        lines.append(
            f"`{best['name']}` 仍是当前全局最优，PPL={best['ppl']:.6f}，对应 nsamples={best['nsamples']} / "
            f"rho={best['rho']} / alpha=[{best['alpha_min']:.2f},{best['alpha_max']:.2f}] / r1={best['r1_ratio']} / "
            f"lambda_psi={best['lambda_psi']}。"
        )
    lines.append("`nsamples=512` 明显优于 `256`，继续作为主搜索配置，`256` 仅保留少量复核点。")
    lines.append("`r1_ratio` 的稳定优选区间仍落在 `0.92-0.95`，低于 `0.90` 的点不再扩展。")
    lines.append("主有效盆地已从粗搜索收敛到 `rho=0.40-0.45`，后续优先做该区间 exploitation。")
    return lines


def next_round_suggestions(results: Dict[str, object], plan: List[Experiment]) -> List[str]:
    top_names = {item["name"] for item in results["top20"][:5]}
    pending = [exp.name for exp in plan if exp.name not in top_names]
    suggestions = [
        "优先比较 `rho=0.41/0.42/0.43` 在 `alpha=[0.46,0.54]`、`r1=0.95` 下的排序，确认最优点是否继续左移或右移。",
        "若本轮最佳仍来自 `ns512 + lambda_psi=1e-5`，下一轮只保留 1 个 `lambda_psi=1e-4` 对照即可。",
        "若 `r1=0.94` 与 `0.95` 差距小于 0.01 PPL，下一轮应改做多 seed 复核，而不是继续扩 r1 网格。",
    ]
    if pending:
        suggestions.insert(0, f"当前轮仍未完成的候选：`{', '.join(pending[:3])}`。")
    return suggestions


def write_summary(results: Dict[str, object], round_state: Dict[str, object]) -> None:
    plan = [Experiment(**item) for item in round_state.get("planned", [])]
    lines: List[str] = []
    lines.append("# FR-SVD Phase2 Result Summary")
    lines.append("")
    lines.append(f"- Generated at: {results['generated_at']}")
    lines.append(f"- PHASE2_LOG_DIR: `{PHASE2_LOG_DIR}`")
    lines.append(f"- RESULT_SUMMARY_FILE: `{RESULT_SUMMARY_FILE}`")
    lines.append("")
    lines.append("## 核验摘要")
    lines.append("")
    best = results["best"]
    if best:
        lines.append(
            f"- 总日志数: {results['total_logs']} (`实验日志 {results['experiment_logs']}` / `辅助日志 {results['auxiliary_logs']}`)"
        )
        lines.append(f"- 可提取 PPL 数: {results['extractable_ppl']}")
        lines.append(f"- 失败数: {results['failed_count']}")
        lines.append(f"- 当前最优实验: `{best['name']}`")
        lines.append(f"- 当前最优 PPL: `{best['ppl']:.6f}`")
    else:
        lines.append("- 未找到可提取 PPL 的实验日志。")
    lines.append("")
    lines.append("## Top20")
    lines.append("")
    lines.append("| Rank | Experiment | PPL | nsamples | rho | alpha | r1 | lambda_psi |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for idx, item in enumerate(results["top20"], start=1):
        lines.append(
            f"| {idx} | `{item['name']}` | {item['ppl']:.6f} | {item['nsamples']} | "
            f"{item['rho']} | [{item['alpha_min']:.2f},{item['alpha_max']:.2f}] | {item['r1_ratio']} | {item['lambda_psi']} |"
        )
    lines.append("")
    lines.append("## 本轮新增实验列表")
    lines.append("")
    lines.append("| Experiment | GPU | Priority | ETA(min) | Expected Gain |")
    lines.append("| --- | --- | --- | --- | --- |")
    for item in round_state.get("planned", []):
        lines.append(
            f"| `{item['name']}` | {item['gpu']} | {item['priority']} | {item['expected_minutes']} | {item['expected_gain']} |"
        )
    lines.append("")
    lines.append("## 失败与中断记录")
    lines.append("")
    if results["failed"]:
        for item in results["failed"]:
            lines.append(f"- `{item['name']}`: {item['status']}")
    else:
        lines.append("- 无。")
    lines.append("")
    lines.append("## 未完成日志")
    lines.append("")
    if results["unfinished"]:
        for item in results["unfinished"][:20]:
            lines.append(f"- `{item['name']}`: {item['status']}")
    else:
        lines.append("- 无。")
    lines.append("")
    lines.append("## 关键结论")
    lines.append("")
    for finding in current_findings(results):
        lines.append(f"- {finding}")
    lines.append("")
    lines.append("## 下一轮建议")
    lines.append("")
    for item in next_round_suggestions(results, plan):
        lines.append(f"- {item}")
    lines.append("")
    with open(RESULT_SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def verify_and_update_summary() -> Dict[str, object]:
    round_state = load_round_state()
    results = collect_phase2_results()
    write_summary(results, round_state)
    save_round_state(round_state)
    return results


def experiment_command(exp: Experiment) -> List[str]:
    return [
        PYTHON_BIN,
        SCRIPT_PATH,
        "--model", MODEL_PATH,
        "--ratio", str(exp.ratio),
        "--dataset", "wikitext2",
        "--nsamples", str(exp.nsamples),
        "--seed", str(exp.seed),
        "--DEV", "cuda:0",
        "--model_seq_len", str(exp.model_seq_len),
        "--eps", str(exp.eps),
        "--r1_ratio", str(exp.r1_ratio),
        "--rho", str(exp.rho),
        "--alpha_min", str(exp.alpha_min),
        "--alpha_max", str(exp.alpha_max),
        "--lambda_psi", str(exp.lambda_psi),
        "--fisher_mode", exp.fisher_mode,
        "--result_dir", os.path.join(PHASE2_LOG_DIR, exp.name),
    ]


def run_experiment(exp: Experiment) -> Tuple[str, Optional[float]]:
    log_path = os.path.join(PHASE2_LOG_DIR, f"{exp.name}.log")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(exp.gpu)

    for attempt in (1, 2):
        wait_for_gpu(exp.gpu)
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{stamp}] >>> GPU{exp.gpu} START {exp.name} attempt={attempt}", flush=True)
        with open(log_path, "w", encoding="utf-8") as log_file:
            proc = subprocess.run(
                experiment_command(exp),
                cwd=REPO_DIR,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        parsed = parse_experiment_log(log_path)
        ppl = parsed["ppl"]
        status = parsed["status"]
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{stamp}] <<< GPU{exp.gpu} END {exp.name} status={status} ppl={ppl}", flush=True)
        if ppl is not None and proc.returncode == 0:
            return "completed", ppl
        if status == "oom" and attempt == 1:
            print(f"[{stamp}] GPU{exp.gpu} OOM，等待后按单卡独占策略重试一次。", flush=True)
            time.sleep(30)
            continue
        return status if status != "no_ppl" else "failed", ppl
    return "failed", None


def execute_round() -> Dict[str, object]:
    os.makedirs(PHASE2_LOG_DIR, exist_ok=True)
    state = load_round_state()
    planned = [Experiment(**item) for item in state["planned"]]
    completed_names = {item["name"] for item in state.get("completed", [])}
    failed_names = {item["name"] for item in state.get("failed", [])}

    queues = {
        6: [exp for exp in planned if exp.gpu == 6 and exp.name not in completed_names and exp.name not in failed_names],
        7: [exp for exp in planned if exp.gpu == 7 and exp.name not in completed_names and exp.name not in failed_names],
    }

    lock = threading.Lock()

    def worker(gpu: int) -> None:
        while queues[gpu]:
            exp = queues[gpu].pop(0)
            with lock:
                state["started"].append(
                    {"name": exp.name, "gpu": gpu, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                )
                save_round_state(state)
            status, ppl = run_experiment(exp)
            record = {
                "name": exp.name,
                "gpu": gpu,
                "status": status,
                "ppl": ppl,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            with lock:
                if status == "completed":
                    state["completed"].append(record)
                else:
                    state["failed"].append(record)
                save_round_state(state)
                verify_and_update_summary()

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=False) for gpu in (6, 7)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return load_round_state()


def print_verification(results: Dict[str, object]) -> None:
    best = results["best"]
    best_name = best["name"] if best else "N/A"
    best_ppl = f"{best['ppl']:.6f}" if best else "N/A"
    print(
        f"核验摘要: 总日志数={results['total_logs']}, 可提取PPL数={results['extractable_ppl']}, "
        f"失败数={results['failed_count']}, 当前最优={best_name}, PPL={best_ppl}"
    )


def print_plan_table(plan: List[Experiment]) -> None:
    print("| 实验名 | 参数 | 归属 GPU | 预计时长 | 优先级 | 预期收益 |")
    print("| --- | --- | --- | --- | --- | --- |")
    for exp in plan:
        params = (
            f"ns={exp.nsamples}, rho={exp.rho}, alpha=[{exp.alpha_min:.2f},{exp.alpha_max:.2f}], "
            f"r1={exp.r1_ratio}, lam={exp.lambda_psi}"
        )
        print(
            f"| {exp.name} | {params} | GPU{exp.gpu} | {exp.expected_minutes} min | {exp.priority} | {exp.expected_gain} |"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="FR-SVD phase2 manager")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--run-round", action="store_true")
    args = parser.parse_args()

    os.makedirs(PHASE2_LOG_DIR, exist_ok=True)
    results = verify_and_update_summary()
    print_verification(results)
    print_plan_table(build_round_plan())

    if args.verify_only and not args.run_round:
        return 0
    if args.run_round:
        execute_round()
        results = verify_and_update_summary()
        print_verification(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
