"""动态 GPU 探测与分配。

本模块只执行只读的 ``nvidia-smi`` 查询，不会启动、停止或抢占任何进程。
GPU 选择是一次瞬时 preflight，不代表 GPU 已被锁定；真正启动模型前必须再次校验。
"""

from __future__ import annotations

import csv
import io
import json
import os
import pwd
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence


GPU_QUERY_FIELDS = (
    "index",
    "uuid",
    "name",
    "memory.used",
    "memory.total",
    "utilization.gpu",
)
COMPUTE_QUERY_FIELDS = (
    "gpu_uuid",
    "pid",
    "process_name",
    "used_gpu_memory",
)

SAMPLE_COUNT = 3
DEFAULT_SAMPLE_INTERVAL_SECONDS = 1.0
MAX_MEMORY_USED_MIB = 1024
MAX_UTILIZATION_PERCENT = 5
SUPPORTED_GPU_COUNTS = (2, 3, 4)
SUPPORTED_SPECULATORS = ("qwen3.5-4b", "qwen3.5-9b")

MODEL_NAMES = {
    "qwen3.6-27b-fp8": "Qwen3.6-27B-FP8",
    "qwen3.5-4b": "Qwen3.5-4B",
    "qwen3.5-9b": "Qwen3.5-9B",
}


class NvidiaSmiError(RuntimeError):
    """``nvidia-smi`` 不可用或返回了无法解析的内容。"""


class GPUSelectionError(RuntimeError):
    """当前 GPU 状态不能满足请求。

    ``report`` 包含已经采集到的 GPU、占用进程和不空闲原因，可供 CLI 在
    不泄露进程命令行参数的前提下输出诊断信息。
    """

    def __init__(self, message: str, report: dict[str, object]):
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class GPUInfo:
    index: int
    uuid: str
    name: str
    memory_used_mib: int
    memory_total_mib: int
    utilization_percent: int

    @property
    def memory_free_mib(self) -> int:
        return self.memory_total_mib - self.memory_used_mib


@dataclass(frozen=True)
class ComputeProcess:
    gpu_uuid: str
    pid: int
    process_name: str
    used_gpu_memory_mib: int | None
    owner: str | None


@dataclass(frozen=True)
class GPUSample:
    sampled_at: str
    gpus: tuple[GPUInfo, ...]
    compute_processes: tuple[ComputeProcess, ...]


CommandRunner = Callable[[Sequence[str]], str]
Sleeper = Callable[[float], None]
Clock = Callable[[], str]
OwnerLookup = Callable[[int], str | None]


def utc_now_iso() -> str:
    """返回带时区的 UTC ISO-8601 时间。"""

    return datetime.now(timezone.utc).isoformat()


def _default_command_runner(command: Sequence[str]) -> str:
    try:
        result = subprocess.run(
            list(command),
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise NvidiaSmiError("未找到 nvidia-smi") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        suffix = f": {detail}" if detail else ""
        raise NvidiaSmiError(f"nvidia-smi 查询失败{suffix}") from exc
    return result.stdout


def _default_owner_lookup(pid: int) -> str | None:
    """只通过 ``/proc`` 查询进程属主，不读取进程命令行。"""

    try:
        uid = os.stat(f"/proc/{pid}").st_uid
        return pwd.getpwuid(uid).pw_name
    except (FileNotFoundError, KeyError, PermissionError, ProcessLookupError):
        return None


def _parse_required_int(raw: str, field: str) -> int:
    value = raw.strip()
    try:
        return int(value)
    except ValueError as exc:
        raise NvidiaSmiError(f"无法解析 {field}: {value!r}") from exc


def _parse_optional_int(raw: str) -> int | None:
    value = raw.strip()
    if not value or value.lower() in {"n/a", "not supported", "[not supported]"}:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise NvidiaSmiError(f"无法解析显存值: {value!r}") from exc


def _csv_rows(output: str) -> Iterable[list[str]]:
    for row in csv.reader(io.StringIO(output), skipinitialspace=True):
        if not row or not any(cell.strip() for cell in row):
            continue
        if row[0].strip().lower().startswith("no running processes"):
            continue
        yield [cell.strip() for cell in row]


def parse_gpu_query(output: str) -> tuple[GPUInfo, ...]:
    """解析 ``--query-gpu`` 的 CSV 输出。"""

    gpus: list[GPUInfo] = []
    for row in _csv_rows(output):
        if len(row) != len(GPU_QUERY_FIELDS):
            raise NvidiaSmiError(
                f"GPU 查询列数错误：期望 {len(GPU_QUERY_FIELDS)}，实际 {len(row)}"
            )
        gpu = GPUInfo(
            index=_parse_required_int(row[0], "GPU index"),
            uuid=row[1],
            name=row[2],
            memory_used_mib=_parse_required_int(row[3], "memory.used"),
            memory_total_mib=_parse_required_int(row[4], "memory.total"),
            utilization_percent=_parse_required_int(row[5], "utilization.gpu"),
        )
        gpus.append(gpu)

    indices = [gpu.index for gpu in gpus]
    if len(indices) != len(set(indices)):
        raise NvidiaSmiError("GPU 查询返回了重复 index")
    return tuple(sorted(gpus, key=lambda gpu: gpu.index))


def parse_compute_query(
    output: str,
    owner_lookup: OwnerLookup = _default_owner_lookup,
) -> tuple[ComputeProcess, ...]:
    """解析 ``--query-compute-apps`` 输出，并补充进程属主。"""

    processes: list[ComputeProcess] = []
    for row in _csv_rows(output):
        # 极少数进程名可能含未转义逗号；首两列与末列固定，中间重新拼为名称。
        if len(row) < len(COMPUTE_QUERY_FIELDS):
            raise NvidiaSmiError(
                f"compute-apps 查询列数错误：至少应有 {len(COMPUTE_QUERY_FIELDS)} 列"
            )
        pid = _parse_required_int(row[1], "process pid")
        processes.append(
            ComputeProcess(
                gpu_uuid=row[0],
                pid=pid,
                process_name=",".join(row[2:-1]),
                used_gpu_memory_mib=_parse_optional_int(row[-1]),
                owner=owner_lookup(pid),
            )
        )
    return tuple(processes)


class NvidiaSmiClient:
    """可注入命令执行器的只读 ``nvidia-smi`` 客户端。"""

    def __init__(
        self,
        command_runner: CommandRunner = _default_command_runner,
        owner_lookup: OwnerLookup = _default_owner_lookup,
    ) -> None:
        self._command_runner = command_runner
        self._owner_lookup = owner_lookup

    def sample(self, sampled_at: str) -> GPUSample:
        gpu_output = self._command_runner(
            (
                "nvidia-smi",
                f"--query-gpu={','.join(GPU_QUERY_FIELDS)}",
                "--format=csv,noheader,nounits",
            )
        )
        process_output = self._command_runner(
            (
                "nvidia-smi",
                f"--query-compute-apps={','.join(COMPUTE_QUERY_FIELDS)}",
                "--format=csv,noheader,nounits",
            )
        )
        return GPUSample(
            sampled_at=sampled_at,
            gpus=parse_gpu_query(gpu_output),
            compute_processes=parse_compute_query(process_output, self._owner_lookup),
        )


def collect_samples(
    client: NvidiaSmiClient,
    *,
    sleeper: Sleeper = time.sleep,
    clock: Clock = utc_now_iso,
    sample_interval_seconds: float = DEFAULT_SAMPLE_INTERVAL_SECONDS,
) -> tuple[GPUSample, ...]:
    """连续采集三次 GPU 与 compute-app 状态。

    ``sleeper`` 与 ``clock`` 可在测试中注入，从而无需真实等待或读取时钟。
    """

    if sample_interval_seconds < 0:
        raise ValueError("sample_interval_seconds 不能为负数")

    samples: list[GPUSample] = []
    for sample_index in range(SAMPLE_COUNT):
        samples.append(client.sample(clock()))
        if sample_index < SAMPLE_COUNT - 1:
            sleeper(sample_interval_seconds)
    return tuple(samples)


def _process_dict(process: ComputeProcess) -> dict[str, object]:
    return asdict(process)


def _sample_dict(sample: GPUSample) -> dict[str, object]:
    return {
        "sampled_at": sample.sampled_at,
        "gpus": [
            {
                **asdict(gpu),
                "memory_free_mib": gpu.memory_free_mib,
            }
            for gpu in sample.gpus
        ],
        "compute_processes": [
            _process_dict(process) for process in sample.compute_processes
        ],
    }


def evaluate_samples(samples: Sequence[GPUSample]) -> dict[str, object]:
    """按三次采样共同满足阈值的规则判定空闲 GPU。"""

    if len(samples) != SAMPLE_COUNT:
        raise ValueError(f"必须提供恰好 {SAMPLE_COUNT} 次采样")

    all_indices = sorted({gpu.index for sample in samples for gpu in sample.gpus})
    evaluations: list[dict[str, object]] = []

    for index in all_indices:
        observations = []
        missing_sample_indices: list[int] = []
        for sample_index, sample in enumerate(samples):
            matches = [gpu for gpu in sample.gpus if gpu.index == index]
            if len(matches) != 1:
                missing_sample_indices.append(sample_index)
            else:
                observations.append(matches[0])

        reasons: list[str] = []
        if missing_sample_indices:
            reasons.append("missing_from_sample")

        uuids = {gpu.uuid for gpu in observations}
        names = {gpu.name for gpu in observations}
        if len(uuids) > 1 or len(names) > 1:
            reasons.append("gpu_identity_changed")

        max_memory_used = max(
            (gpu.memory_used_mib for gpu in observations), default=None
        )
        max_utilization = max(
            (gpu.utilization_percent for gpu in observations), default=None
        )
        if max_memory_used is not None and max_memory_used > MAX_MEMORY_USED_MIB:
            reasons.append("memory_used_above_limit")
        if max_utilization is not None and max_utilization > MAX_UTILIZATION_PERCENT:
            reasons.append("utilization_above_limit")

        processes: dict[tuple[object, ...], ComputeProcess] = {}
        matching_uuids = uuids
        for sample in samples:
            for process in sample.compute_processes:
                if process.gpu_uuid in matching_uuids:
                    key = (
                        process.gpu_uuid,
                        process.pid,
                        process.process_name,
                        process.used_gpu_memory_mib,
                        process.owner,
                    )
                    processes[key] = process
        if processes:
            reasons.append("compute_process_present")

        latest = observations[-1] if observations else None
        evaluations.append(
            {
                "index": index,
                "uuid": latest.uuid if latest else None,
                "name": latest.name if latest else None,
                "memory_total_mib": latest.memory_total_mib if latest else None,
                "memory_free_mib": latest.memory_free_mib if latest else None,
                "max_memory_used_mib": max_memory_used,
                "max_utilization_percent": max_utilization,
                "compute_processes": [
                    _process_dict(process)
                    for process in sorted(
                        processes.values(), key=lambda item: (item.pid, item.process_name)
                    )
                ],
                "missing_sample_indices": missing_sample_indices,
                "is_idle": not reasons,
                "reasons": reasons,
            }
        )

    idle_gpu_ids = [
        int(evaluation["index"])
        for evaluation in evaluations
        if evaluation["is_idle"]
    ]
    return {
        "samples": [_sample_dict(sample) for sample in samples],
        "gpu_evaluations": evaluations,
        "idle_gpu_ids": idle_gpu_ids,
        "idle_gpu_count": len(idle_gpu_ids),
    }


def parse_gpu_ids(raw_gpu_ids: str | None) -> tuple[int, ...] | None:
    """解析逗号分隔的物理 GPU index，并拒绝空项、负数和重复项。"""

    if raw_gpu_ids is None:
        return None
    parts = [part.strip() for part in raw_gpu_ids.split(",")]
    if not parts or any(not part for part in parts):
        raise ValueError("--gpu-ids 必须是非空的逗号分隔整数")
    try:
        gpu_ids = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ValueError("--gpu-ids 必须是逗号分隔整数") from exc
    if any(gpu_id < 0 for gpu_id in gpu_ids):
        raise ValueError("--gpu-ids 不能包含负数")
    if len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError("--gpu-ids 不能包含重复卡号")
    return gpu_ids


def build_allocation(
    selected_gpu_ids: Sequence[int],
    speculator_model: str = "qwen3.5-4b",
) -> dict[str, object]:
    """按已确认的 2/3/4 卡规则生成 Actor/Speculator 拓扑。"""

    gpu_count = len(selected_gpu_ids)
    if gpu_count not in SUPPORTED_GPU_COUNTS:
        raise ValueError(f"仅支持 GPU 数量 {SUPPORTED_GPU_COUNTS}")
    if speculator_model not in SUPPORTED_SPECULATORS:
        raise ValueError(f"不支持的 Speculator: {speculator_model}")
    if gpu_count == 2 and speculator_model != "qwen3.5-4b":
        raise ValueError("2 卡 smoke 的 Speculator 固定为 qwen3.5-4b")

    ids = list(selected_gpu_ids)
    if gpu_count == 2:
        actor_model = "qwen3.5-9b"
        actor_ids = ids[:1]
        speculator_key = "qwen3.5-4b"
        speculator_ids = ids[1:]
        smoke_only = True
    else:
        actor_model = "qwen3.6-27b-fp8"
        actor_ids = ids[:2]
        speculator_key = speculator_model
        speculator_ids = ids[2:]
        smoke_only = False

    return {
        "gpu_count": gpu_count,
        "actor": {
            "model_key": actor_model,
            "model_name": MODEL_NAMES[actor_model],
            "gpu_ids": actor_ids,
            "tensor_parallel_size": len(actor_ids),
        },
        "speculator": {
            "model_key": speculator_key,
            "model_name": MODEL_NAMES[speculator_key],
            "gpu_ids": speculator_ids,
            "tensor_parallel_size": len(speculator_ids),
        },
        "smoke_only": smoke_only,
    }


def _selection_report(
    *,
    request: dict[str, object],
    inventory: dict[str, object],
) -> dict[str, object]:
    return {
        "request": request,
        "thresholds": {
            "sample_count": SAMPLE_COUNT,
            "max_memory_used_mib": MAX_MEMORY_USED_MIB,
            "max_utilization_percent": MAX_UTILIZATION_PERCENT,
            "requires_no_compute_process": True,
        },
        "inventory": inventory,
    }


def select_gpus(
    samples: Sequence[GPUSample],
    *,
    gpu_count: int,
    requested_gpu_ids: Sequence[int] | None = None,
    speculator_model: str = "qwen3.5-4b",
    environment: str = "chess",
    experiment: str = "smoke",
    generated_at: str | None = None,
    sample_interval_seconds: float = DEFAULT_SAMPLE_INTERVAL_SECONDS,
) -> dict[str, object]:
    """选择空闲卡并生成不启动服务的 preflight manifest。"""

    if gpu_count not in SUPPORTED_GPU_COUNTS:
        raise ValueError(f"--gpu-count 仅支持 {SUPPORTED_GPU_COUNTS}")

    explicit_ids = list(requested_gpu_ids) if requested_gpu_ids is not None else None
    if explicit_ids is not None and len(explicit_ids) != len(set(explicit_ids)):
        raise ValueError("requested_gpu_ids 不能包含重复卡号")
    request = {
        "gpu_count": gpu_count,
        "gpu_ids": explicit_ids,
        "selection_mode": "explicit" if explicit_ids is not None else "automatic",
        "environment": environment,
        "experiment": experiment,
        "speculator_model": speculator_model,
    }
    inventory = evaluate_samples(samples)
    report = _selection_report(request=request, inventory=inventory)
    idle_ids = list(inventory["idle_gpu_ids"])

    if gpu_count == 2 and speculator_model != "qwen3.5-4b":
        raise GPUSelectionError(
            "2 卡 smoke 的 Speculator 固定为 qwen3.5-4b",
            report,
        )

    if len(idle_ids) < 2:
        raise GPUSelectionError(
            f"当前仅有 {len(idle_ids)} 张空闲 GPU，少于最低要求 2 张",
            report,
        )

    if explicit_ids is not None:
        if len(explicit_ids) != gpu_count:
            raise GPUSelectionError(
                f"--gpu-ids 数量 {len(explicit_ids)} 与 --gpu-count {gpu_count} 不一致",
                report,
            )
        unavailable_ids = [gpu_id for gpu_id in explicit_ids if gpu_id not in idle_ids]
        if unavailable_ids:
            raise GPUSelectionError(
                f"显式指定的 GPU 当前不空闲或不存在: {unavailable_ids}",
                report,
            )
        selected_ids = explicit_ids
    else:
        if len(idle_ids) < gpu_count:
            raise GPUSelectionError(
                f"请求 {gpu_count} 张 GPU，但当前只有 {len(idle_ids)} 张空闲 GPU",
                report,
            )
        # 使用稳定的物理 index 顺序，保证同一快照下分配可重复。
        selected_ids = sorted(idle_ids)[:gpu_count]

    allocation = build_allocation(selected_ids, speculator_model)
    evaluations_by_id = {
        evaluation["index"]: evaluation
        for evaluation in inventory["gpu_evaluations"]
    }
    selected_gpus = [evaluations_by_id[gpu_id] for gpu_id in selected_ids]
    topology_components = [
        f"gpu{gpu['index']}:{gpu['name']}" for gpu in selected_gpus
    ]

    return {
        "schema_version": "1.0",
        "kind": "chess_gpu_preflight",
        "generated_at": generated_at or utc_now_iso(),
        **report,
        "sampling": {
            "sample_interval_seconds": sample_interval_seconds,
        },
        "selection": {
            "status": "selected_not_reserved",
            "gpu_ids": selected_ids,
            "gpus": selected_gpus,
            "topology_key": f"{gpu_count}cards|" + "|".join(topology_components),
            "warning": "GPU selection is a snapshot, not a reservation; recheck before launch.",
        },
        "allocation": allocation,
        "execution": {
            "services_started": False,
            "experiment_started": False,
        },
    }


def write_manifest(manifest: dict[str, object], output_path: str | Path) -> Path:
    """原子写入 manifest；不会覆盖其他实验文件。"""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return path
