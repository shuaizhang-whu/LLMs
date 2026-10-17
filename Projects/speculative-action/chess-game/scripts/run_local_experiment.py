#!/usr/bin/env python3
"""Chess 本地免费模型实验的安全统一启动器。

``--preflight-only`` 只连续检查 GPU 并输出分配结果。完整模式会先完成同样的
检查，再验证模型、独立启动 Actor/Speculator vLLM、验证动态 READY、运行
``run_chess_local.py``，最后只清理由本启动器创建的进程组。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable, Sequence

try:
    from .gpu_resources import (
        DEFAULT_SAMPLE_INTERVAL_SECONDS,
        GPUSelectionError,
        NvidiaSmiClient,
        NvidiaSmiError,
        SUPPORTED_GPU_COUNTS,
        SUPPORTED_SPECULATORS,
        collect_samples,
        evaluate_samples,
        parse_gpu_ids,
        select_gpus,
        write_manifest,
    )
    from .service_manager import (
        DEFAULT_CHESS_ENTRYPOINT,
        DEFAULT_VLLM_PYTHON,
        BASELINE_COMMIT,
        LOCAL_NO_PROXY,
        VLLM_RUNTIME_OVERRIDES,
        LauncherError,
        ManagedProcess,
        ServiceReadinessError,
        assert_ports_available,
        build_chess_command,
        build_run_directory,
        public_command,
        inspect_python_environment,
        service_plans_from_allocation,
        start_chess_process,
        start_service,
        terminate_all,
        verify_static_model,
        wait_until_ready,
        workspace_git_head,
    )
except ImportError:  # 支持 ``python scripts/run_local_experiment.py``。
    from gpu_resources import (  # type: ignore[no-redef]
        DEFAULT_SAMPLE_INTERVAL_SECONDS,
        GPUSelectionError,
        NvidiaSmiClient,
        NvidiaSmiError,
        SUPPORTED_GPU_COUNTS,
        SUPPORTED_SPECULATORS,
        collect_samples,
        evaluate_samples,
        parse_gpu_ids,
        select_gpus,
        write_manifest,
    )
    from service_manager import (  # type: ignore[no-redef]
        DEFAULT_CHESS_ENTRYPOINT,
        DEFAULT_VLLM_PYTHON,
        BASELINE_COMMIT,
        LOCAL_NO_PROXY,
        VLLM_RUNTIME_OVERRIDES,
        LauncherError,
        ManagedProcess,
        ServiceReadinessError,
        assert_ports_available,
        build_chess_command,
        build_run_directory,
        public_command,
        inspect_python_environment,
        service_plans_from_allocation,
        start_chess_process,
        start_service,
        terminate_all,
        verify_static_model,
        wait_until_ready,
        workspace_git_head,
    )


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "reproduction" / "local_runs"


def executable_path(path: Path) -> Path:
    """转为绝对路径但保留 venv 的 Python 符号链接入口。

    对 ``venv/bin/python`` 调用 ``Path.resolve()`` 会跳到创建该 venv 的基础
    Conda Python，进而丢失 venv 的 ``pyvenv.cfg`` 与依赖搜索路径。
    """

    return Path(os.path.abspath(os.fspath(path)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpu-count",
        type=int,
        choices=SUPPORTED_GPU_COUNTS,
        required=True,
        help="必须手动指定使用 2、3 或 4 张 GPU",
    )
    parser.add_argument(
        "--gpu-ids",
        help="可选：按 Actor、Speculator 分配顺序给出物理 GPU index，如 0,2,3",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="只检查并分配 GPU，不验证模型、不启动服务或实验",
    )
    parser.add_argument("--environment", default="chess", choices=("chess",))
    parser.add_argument("--experiment", default="smoke")
    parser.add_argument("--run-id", help="完整运行必填；用于隔离日志和产物")
    parser.add_argument(
        "--execution-mode",
        default="speculative",
        choices=("sequential", "speculative"),
    )
    parser.add_argument("--stop-after", type=int, default=10)
    parser.add_argument("--num-guesses", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--seed", type=int, default=248)
    parser.add_argument("--actor-port", type=int, default=8100)
    parser.add_argument("--speculator-port", type=int, default=8101)
    parser.add_argument(
        "--speculator-model",
        default="qwen3.5-4b",
        choices=SUPPORTED_SPECULATORS,
        help="3/4 卡拓扑中的 Speculator；2 卡 smoke 固定使用 4B",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="vLLM KV cache 上限；默认显式限制为 8192，绝不自动降低",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="vLLM 显存利用率，默认 0.90，绝不自动调整",
    )
    parser.add_argument(
        "--vllm-python",
        type=Path,
        default=DEFAULT_VLLM_PYTHON,
        help="vLLM 环境 Python；默认 agent-action-vllm 的绝对路径",
    )
    parser.add_argument(
        "--runner-python",
        type=Path,
        default=Path(sys.executable),
        help="Chess runner Python；默认当前已激活环境的 Python",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        help="兼容旧 preflight 的可选输出；完整运行固定写入隔离目录",
    )
    return parser


def _print_error(message: str, *, preflight: dict[str, object] | None = None) -> None:
    payload: dict[str, object] = {"status": "error", "error": message}
    if preflight is not None:
        payload["preflight"] = preflight
    print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)


def _runner_supports(entrypoint: Path, option: str) -> bool:
    """静态检查可选 CLI 字符串，不执行 runner。"""

    try:
        return option in entrypoint.read_text(encoding="utf-8")
    except OSError:
        return False


def _write_lifecycle_manifest(
    manifest: dict[str, object], manifest_path: Path | None
) -> None:
    if manifest_path is not None:
        write_manifest(manifest, manifest_path)


def _release_report(
    release_samples,
    selected_gpu_ids: Sequence[int],
) -> tuple[dict[str, object], bool]:
    """确认进程与显存已释放；保留但不以瞬时利用率判失败。

    ``utilization.gpu`` 是时间窗口统计，进程退出后的首帧仍可能显示 100%。它适合
    启动前防止抢卡，却不能证明退出后仍有资源所有权。释放条件仍严格拒绝 compute
    进程、显存超限、缺失采样和 GPU 身份变化。
    """

    inventory = evaluate_samples(release_samples)
    evaluations = {
        int(evaluation["index"]): evaluation
        for evaluation in inventory["gpu_evaluations"]
    }
    selected_released: dict[str, bool] = {}
    selected_preflight_idle: dict[str, bool] = {}
    residual_utilization: dict[str, int | None] = {}
    for gpu_id in selected_gpu_ids:
        evaluation = evaluations.get(gpu_id)
        if evaluation is None:
            selected_released[str(gpu_id)] = False
            selected_preflight_idle[str(gpu_id)] = False
            residual_utilization[str(gpu_id)] = None
            continue
        blocking_reasons = set(evaluation["reasons"]) - {
            "utilization_above_limit"
        }
        selected_released[str(gpu_id)] = not blocking_reasons
        selected_preflight_idle[str(gpu_id)] = bool(evaluation["is_idle"])
        residual_utilization[str(gpu_id)] = evaluation["max_utilization_percent"]

    confirmed = all(selected_released.values())
    return (
        {
            "confirmed_released": confirmed,
            # 兼容已有消费者；这里的 idle 明确定义为资源释放，而非启动前低利用率。
            "confirmed_idle": confirmed,
            "selected_gpu_released": selected_released,
            "selected_gpu_preflight_idle": selected_preflight_idle,
            "residual_max_utilization_percent": residual_utilization,
            "release_criteria": {
                "requires_no_compute_process": True,
                "requires_memory_within_idle_limit": True,
                "requires_low_utilization": False,
            },
            "inventory": inventory,
        },
        confirmed,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    client: NvidiaSmiClient | None = None,
    sleeper=None,
    clock=None,
    static_verifier: Callable = verify_static_model,
    port_checker: Callable = assert_ports_available,
    service_starter: Callable = start_service,
    readiness_waiter: Callable = wait_until_ready,
    chess_starter: Callable = start_chess_process,
    process_terminator: Callable = terminate_all,
    chess_entrypoint: Path | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        requested_gpu_ids = parse_gpu_ids(args.gpu_ids)
    except ValueError as exc:
        parser.error(str(exc))

    sample_kwargs = {"sample_interval_seconds": DEFAULT_SAMPLE_INTERVAL_SECONDS}
    if sleeper is not None:
        sample_kwargs["sleeper"] = sleeper
    if clock is not None:
        sample_kwargs["clock"] = clock
    gpu_client = client or NvidiaSmiClient()

    try:
        initial_samples = collect_samples(gpu_client, **sample_kwargs)
        manifest = select_gpus(
            initial_samples,
            gpu_count=args.gpu_count,
            requested_gpu_ids=requested_gpu_ids,
            speculator_model=args.speculator_model,
            environment=args.environment,
            experiment=args.experiment,
            sample_interval_seconds=DEFAULT_SAMPLE_INTERVAL_SECONDS,
        )
    except (NvidiaSmiError, GPUSelectionError) as exc:
        _print_error(
            str(exc),
            preflight=exc.report if isinstance(exc, GPUSelectionError) else None,
        )
        return 2

    if args.preflight_only:
        if args.manifest_output is not None:
            manifest["manifest_path"] = str(args.manifest_output.resolve())
            write_manifest(manifest, args.manifest_output)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0

    # 完整运行的参数检查全部发生在启动任何服务之前。
    if not args.run_id:
        _print_error("完整运行必须显式提供 --run-id")
        return 2
    if args.manifest_output is not None:
        _print_error("完整运行的 manifest 固定写入隔离目录，请勿传 --manifest-output")
        return 2
    if args.stop_after <= 0:
        _print_error("--stop-after 必须为正整数")
        return 2
    if args.seed < 0:
        _print_error("--seed 不能为负数")
        return 2
    if args.max_model_len <= 0:
        _print_error("--max-model-len 必须为正整数")
        return 2
    if not 0 < args.gpu_memory_utilization <= 1:
        _print_error("--gpu-memory-utilization 必须在 (0, 1] 内")
        return 2
    if manifest["allocation"]["smoke_only"] and args.stop_after > 10:
        _print_error("2 卡拓扑仅用于 smoke，--stop-after 不能超过 10")
        return 2

    entrypoint = (chess_entrypoint or DEFAULT_CHESS_ENTRYPOINT).resolve()
    vllm_python = executable_path(args.vllm_python)
    runner_python = executable_path(args.runner_python)
    try:
        run_directory = build_run_directory(
            args.output_root,
            manifest["selection"]["topology_key"],
            manifest["selection"]["gpu_ids"],
            args.run_id,
        )
    except ValueError as exc:
        _print_error(str(exc))
        return 2
    try:
        run_directory.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        _print_error(f"run-id 已存在，拒绝覆盖: {run_directory}")
        return 2
    except OSError as exc:
        _print_error(f"无法创建运行目录: {type(exc).__name__}: {exc}")
        return 2

    manifest_path = run_directory / "experiment_manifest.json"
    topology_path = run_directory / "gpu_topology.json"
    manifest.update(
        {
            "manifest_path": str(manifest_path),
            "run_directory": str(run_directory),
            "status": "PREPARING",
            "configuration": {
                "run_id": args.run_id,
                "execution_mode": args.execution_mode,
                "stop_after": args.stop_after,
                "num_guesses": args.num_guesses,
                "seed": args.seed,
                "actor_port": args.actor_port,
                "speculator_port": args.speculator_port,
                "max_model_len": args.max_model_len,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "vllm_python": str(vllm_python),
                "runner_python": str(runner_python),
                "chess_entrypoint": str(entrypoint),
                "vllm_runtime_overrides": dict(VLLM_RUNTIME_OVERRIDES),
                "local_http_no_proxy": LOCAL_NO_PROXY,
            },
        }
    )
    write_manifest(
        {
            "schema_version": manifest["schema_version"],
            "selection": manifest["selection"],
            "allocation": manifest["allocation"],
        },
        topology_path,
    )
    _write_lifecycle_manifest(manifest, manifest_path)

    managed_processes: list[ManagedProcess] = []
    exit_code = 0
    selected_gpu_ids = list(manifest["selection"]["gpu_ids"])
    cleanup_reports: list[dict[str, object]] = []

    try:
        if not entrypoint.is_file():
            raise ServiceReadinessError(f"Chess runner 不存在: {entrypoint}")
        if not vllm_python.is_file():
            raise ServiceReadinessError(f"vLLM Python 不存在: {vllm_python}")
        if not runner_python.is_file():
            raise ServiceReadinessError(f"runner Python 不存在: {runner_python}")

        manifest["software_environment"] = {
            "baseline_commit": BASELINE_COMMIT,
            "workspace_head": workspace_git_head(),
            "launcher_python": sys.version,
            "vllm": inspect_python_environment(str(vllm_python)),
            "runner": inspect_python_environment(str(runner_python)),
        }

        actor_plan, speculator_plan = service_plans_from_allocation(
            manifest["allocation"],
            actor_port=args.actor_port,
            speculator_port=args.speculator_port,
            run_directory=run_directory,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        plans = (actor_plan, speculator_plan)
        manifest["services"] = [plan.to_manifest() for plan in plans]

        port_checker((args.actor_port, args.speculator_port))
        manifest["port_preflight"] = "PASS"
        static_reports = [static_verifier(plan) for plan in plans]
        manifest["static_model_checks"] = static_reports

        # preflight 是快照而非锁；耗时的模型检查结束后，在启动前重新连续采样三次，
        # 且必须仍能按同一物理卡顺序完成显式分配。
        try:
            launch_samples = collect_samples(gpu_client, **sample_kwargs)
            launch_recheck = select_gpus(
                launch_samples,
                gpu_count=args.gpu_count,
                requested_gpu_ids=selected_gpu_ids,
                speculator_model=args.speculator_model,
                environment=args.environment,
                experiment=args.experiment,
                sample_interval_seconds=DEFAULT_SAMPLE_INTERVAL_SECONDS,
            )
        except (NvidiaSmiError, GPUSelectionError) as exc:
            raise LauncherError(
                f"服务启动前 GPU 复检失败: {exc}", exit_code=2
            ) from exc
        manifest["launch_gpu_recheck"] = {
            "status": "PASS",
            "selection": launch_recheck["selection"],
            "inventory": launch_recheck["inventory"],
        }
        manifest["status"] = "STARTING_SERVICES"
        _write_lifecycle_manifest(manifest, manifest_path)

        for plan in plans:
            managed = service_starter(
                plan,
                python_executable=str(vllm_python),
            )
            managed_processes.append(managed)
        manifest["service_processes"] = [
            {
                "role": managed.role,
                "pid": managed.pid,
                "command": public_command(managed.command),
                "log_path": str(managed.log_path),
            }
            for managed in managed_processes
        ]
        _write_lifecycle_manifest(manifest, manifest_path)

        dynamic_ready = [
            readiness_waiter(managed, plan)
            for managed, plan in zip(managed_processes, plans)
        ]
        manifest["dynamic_ready"] = dynamic_ready
        manifest["status"] = "RUNNING_CHESS"

        optional_manifest = None
        if _runner_supports(entrypoint, "--experiment-manifest"):
            # 主 manifest 会在 runner 退出及清理后继续更新。传给 runner 的必须是
            # 冻结快照，否则 runner 记录的 SHA-256 会在 launcher 收尾后失配。
            optional_manifest = run_directory / "runner_input_manifest.json"
            runner_input = dict(manifest)
            runner_input["status"] = "READY_TO_RUN"
            runner_input["snapshot_note"] = (
                "Immutable launcher snapshot referenced by the Chess runner; "
                "final lifecycle state is in experiment_manifest.json."
            )
            write_manifest(runner_input, optional_manifest)
            manifest["runner_input_manifest_path"] = str(optional_manifest)
        optional_topology = (
            topology_path if _runner_supports(entrypoint, "--topology-path") else None
        )
        chess_command = build_chess_command(
            entrypoint=entrypoint,
            actor_plan=actor_plan,
            speculator_plan=speculator_plan,
            execution_mode=args.execution_mode,
            num_guesses=args.num_guesses,
            seed=args.seed,
            run_id=args.run_id,
            stop_after=args.stop_after,
            output_directory=run_directory / "artifacts",
            gpu_ids=selected_gpu_ids,
            experiment_manifest=optional_manifest,
            topology_path=optional_topology,
            python_executable=str(runner_python),
        )
        manifest["chess_command"] = public_command(chess_command)
        _write_lifecycle_manifest(manifest, manifest_path)

        chess_process = chess_starter(
            chess_command,
            run_directory / "logs" / "chess.log",
        )
        managed_processes.append(chess_process)
        chess_returncode = chess_process.process.wait()
        manifest["chess_returncode"] = chess_returncode
        if chess_returncode != 0:
            manifest["status"] = "CHESS_FAILED"
            exit_code = (
                chess_returncode
                if isinstance(chess_returncode, int) and 1 <= chess_returncode <= 255
                else 1
            )
        else:
            manifest["status"] = "SUCCEEDED"
    except LauncherError as exc:
        exit_code = exc.exit_code
        manifest["status"] = "FAILED"
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
    except KeyboardInterrupt:
        exit_code = 130
        manifest["status"] = "INTERRUPTED"
        manifest["error"] = {"type": "KeyboardInterrupt", "message": "用户中断"}
    except Exception as exc:  # 防止未知启动错误绕过 finally 清理。
        exit_code = 4
        manifest["status"] = "FAILED"
        manifest["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
    finally:
        try:
            cleanup_reports = process_terminator(managed_processes)
        except Exception as exc:  # 清理器自身失败也必须落盘，不能继续操作其他进程。
            cleanup_reports = [
                {
                    "role": "launcher_cleanup",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            ]
        manifest["cleanup"] = cleanup_reports

        cleanup_failed = any(report.get("error") for report in cleanup_reports)
        try:
            release_samples = collect_samples(gpu_client, **sample_kwargs)
            release_report, released = _release_report(
                release_samples, selected_gpu_ids
            )
            manifest["gpu_release_check"] = release_report
        except Exception as exc:
            released = False
            manifest["gpu_release_check"] = {
                "confirmed_idle": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

        if (cleanup_failed or not released) and exit_code == 0:
            exit_code = 5
            manifest["status"] = "CLEANUP_UNCONFIRMED"
        _write_lifecycle_manifest(manifest, manifest_path)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
