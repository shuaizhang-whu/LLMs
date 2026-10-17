"""本地 Chess 实验的模型服务生命周期管理。

安全约束：

* 每个 vLLM/Chess 子进程都以独立 session 启动；
* 只记录并终止本模块亲自启动的进程组；
* 不使用 ``pkill``、不扫描或终止其他用户进程；
* Actor 与 Speculator 只看见各自被分配的物理 GPU；
* 本模块不自动降低 tensor parallel、切换模型或修改端口。
"""

from __future__ import annotations

import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence


CHESS_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
MODEL_MANAGER_PATH = WORKSPACE_ROOT / "Tool" / "model_manager.py"
DEFAULT_CHESS_ENTRYPOINT = CHESS_ROOT / "run_chess_local.py"

DEFAULT_SERVICE_HOST = "127.0.0.1"
DEFAULT_SERVICE_TIMEOUT_SECONDS = 900.0
DEFAULT_HEALTH_POLL_SECONDS = 2.0
DEFAULT_TERMINATE_TIMEOUT_SECONDS = 30.0
DEFAULT_VLLM_PYTHON = Path(
    "/mnt/data/zhangshuai/anaconda3/envs/agent-action-vllm/bin/python"
)
BASELINE_COMMIT = "dc938b9ef7474caf07fe4ad16549c1fa8c7d268c"

SECRET_ENV_KEYS = {
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "ANTHROPIC_API_KEY",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACEHUB_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
    "HUGGINGFACE_TOKEN",
    "HF_API_TOKEN",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
}

# 当前服务器提供 CUDA runtime/driver，但没有 nvcc。vLLM 0.25.1 默认的
# FlashInfer sampler 会在首次 profile 时尝试 JIT 编译并因此失败；其源码明确
# 支持通过此变量回退到内置 PyTorch sampler。确定性 temperature=0 主实验的
# 贪心解码语义不变。
VLLM_RUNTIME_OVERRIDES = {
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
}
LOCAL_NO_PROXY = "127.0.0.1,localhost,::1"

# 健康检查只允许直连本机服务。使用显式空 ProxyHandler，避免服务器全局
# HTTP_PROXY 将 127.0.0.1 请求转发到外部代理。
_DIRECT_HTTP_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _is_secret_environment_key(key: str) -> bool:
    """保守剔除服务不需要的凭据类环境变量。"""

    normalized = key.upper()
    return (
        normalized in SECRET_ENV_KEYS
        or normalized.endswith("_API_KEY")
        or normalized.endswith("_TOKEN")
        or "SECRET" in normalized
        or "PASSWORD" in normalized
    )


@dataclass(frozen=True)
class RegisteredModel:
    alias: str
    served_model_name: str
    local_path: Path
    pinned_revision: str
    language_model_only: bool = False


MODEL_REGISTRY: dict[str, RegisteredModel] = {
    "qwen3.6-27b-fp8": RegisteredModel(
        alias="qwen3.6-27b-fp8",
        served_model_name="qwen3.6-27b-fp8",
        local_path=WORKSPACE_ROOT / "Models" / "Qwen3.6-27B-FP8",
        pinned_revision="e89b16ebf1988b3d6befa7de50abc2d76f26eb09",
        language_model_only=True,
    ),
    "qwen3.5-4b": RegisteredModel(
        alias="qwen3.5-4b",
        served_model_name="qwen3.5-4b",
        local_path=WORKSPACE_ROOT / "Models" / "Qwen3.5-4B",
        pinned_revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        language_model_only=True,
    ),
    "qwen3.5-9b": RegisteredModel(
        alias="qwen3.5-9b",
        served_model_name="qwen3.5-9b",
        local_path=WORKSPACE_ROOT / "Models" / "Qwen3.5-9B",
        pinned_revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        language_model_only=True,
    ),
}


class LauncherError(RuntimeError):
    """带稳定退出码、可安全写入 manifest 的启动器错误。"""

    def __init__(self, message: str, *, exit_code: int):
        super().__init__(message)
        self.exit_code = exit_code


class PortUnavailableError(LauncherError):
    def __init__(self, message: str):
        super().__init__(message, exit_code=2)


class StaticModelError(LauncherError):
    def __init__(self, message: str):
        super().__init__(message, exit_code=3)


class ServiceReadinessError(LauncherError):
    def __init__(self, message: str):
        super().__init__(message, exit_code=4)


class CleanupError(LauncherError):
    def __init__(self, message: str):
        super().__init__(message, exit_code=5)


@dataclass(frozen=True)
class ServicePlan:
    role: str
    model_alias: str
    served_model_name: str
    model_path: Path
    pinned_revision: str
    physical_gpu_ids: tuple[int, ...]
    tensor_parallel_size: int
    host: str
    port: int
    log_path: Path
    language_model_only: bool = False
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.90

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def to_manifest(self) -> dict[str, object]:
        data = asdict(self)
        data["model_path"] = str(self.model_path)
        data["log_path"] = str(self.log_path)
        data["physical_gpu_ids"] = list(self.physical_gpu_ids)
        data["base_url"] = self.base_url
        return data


class ProcessLike(Protocol):
    pid: int
    returncode: int | None

    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...


@dataclass
class ManagedProcess:
    role: str
    process: ProcessLike
    command: tuple[str, ...]
    log_path: Path
    log_handle: object | None = None

    @property
    def pid(self) -> int:
        return self.process.pid


ProcessFactory = Callable[..., ProcessLike]
CommandRunner = Callable[..., subprocess.CompletedProcess]
JsonRequester = Callable[[str, str, Mapping[str, object] | None, float], dict[str, object]]
GroupSignaler = Callable[[int, signal.Signals], None]
GroupChecker = Callable[[int], bool]


def process_group_exists(process_group_id: int) -> bool:
    """只探测指定 PGID 是否仍存在，不枚举或接触其他进程组。"""

    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # 存在但无权发送信号时仍应报告为未释放，不能误判为空闲。
        return True
    return True


def _wait_for_process_group_exit(
    process_group_id: int,
    *,
    group_checker: GroupChecker,
    timeout_seconds: float,
    monotonic: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> bool:
    """等待本启动器创建的进程组消失，超时返回 ``False``。"""

    deadline = monotonic() + max(timeout_seconds, 0.0)
    while group_checker(process_group_id):
        remaining = deadline - monotonic()
        if remaining <= 0:
            return False
        sleeper(min(0.1, remaining))
    return True


def _default_command_runner(command: Sequence[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(list(command), **kwargs)


def _default_json_requester(
    method: str,
    url: str,
    payload: Mapping[str, object] | None,
    timeout_seconds: float,
) -> dict[str, object]:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    with _DIRECT_HTTP_OPENER.open(request, timeout=timeout_seconds) as response:
        if response.status != 200:
            raise OSError(f"HTTP {response.status}")
        decoded = json.loads(response.read().decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("HTTP JSON 顶层不是对象")
    return decoded


def validate_run_id(run_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id):
        raise ValueError(
            "--run-id 必须以字母或数字开头，且仅含字母、数字、点、下划线、连字符"
        )
    return run_id


def topology_slug(topology_key: str, gpu_ids: Sequence[int]) -> str:
    """生成可读且稳定的拓扑目录名，避免把 GPU 型号原文直接用作路径。"""

    import hashlib

    digest = hashlib.sha256(topology_key.encode("utf-8")).hexdigest()[:12]
    ids = "-".join(str(gpu_id) for gpu_id in gpu_ids)
    return f"{len(gpu_ids)}gpu-ids-{ids}-{digest}"


def build_run_directory(
    output_root: Path,
    topology_key: str,
    gpu_ids: Sequence[int],
    run_id: str,
) -> Path:
    return output_root.expanduser().resolve() / topology_slug(
        topology_key, gpu_ids
    ) / validate_run_id(run_id)


def service_plans_from_allocation(
    allocation: Mapping[str, object],
    *,
    actor_port: int,
    speculator_port: int,
    run_directory: Path,
    host: str = DEFAULT_SERVICE_HOST,
    max_model_len: int = 8192,
    gpu_memory_utilization: float = 0.90,
) -> tuple[ServicePlan, ServicePlan]:
    if actor_port == speculator_port:
        raise PortUnavailableError("Actor 与 Speculator 端口必须不同")
    for port in (actor_port, speculator_port):
        if port < 1 or port > 65535:
            raise PortUnavailableError(f"非法 TCP 端口: {port}")
    if max_model_len <= 0:
        raise StaticModelError("max_model_len 必须为正整数")
    if not 0 < gpu_memory_utilization <= 1:
        raise StaticModelError("gpu_memory_utilization 必须在 (0, 1] 内")

    plans = []
    for role, port in (("actor", actor_port), ("speculator", speculator_port)):
        raw = allocation.get(role)
        if not isinstance(raw, Mapping):
            raise StaticModelError(f"allocation 缺少 {role}")
        alias = raw.get("model_key")
        if not isinstance(alias, str) or alias not in MODEL_REGISTRY:
            raise StaticModelError(f"allocation 中存在未注册模型: {alias!r}")
        registered = MODEL_REGISTRY[alias]
        gpu_ids = raw.get("gpu_ids")
        tensor_parallel_size = raw.get("tensor_parallel_size")
        if (
            not isinstance(gpu_ids, list)
            or not all(isinstance(item, int) for item in gpu_ids)
            or not isinstance(tensor_parallel_size, int)
            or tensor_parallel_size != len(gpu_ids)
            or tensor_parallel_size < 1
        ):
            raise StaticModelError(f"{role} GPU/TP allocation 非法")
        plans.append(
            ServicePlan(
                role=role,
                model_alias=alias,
                served_model_name=registered.served_model_name,
                model_path=registered.local_path,
                pinned_revision=registered.pinned_revision,
                physical_gpu_ids=tuple(gpu_ids),
                tensor_parallel_size=tensor_parallel_size,
                host=host,
                port=port,
                log_path=run_directory / "logs" / f"{role}_vllm.log",
                language_model_only=registered.language_model_only,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
            )
        )
    return plans[0], plans[1]


def verify_static_model(
    plan: ServicePlan,
    *,
    runner: CommandRunner = _default_command_runner,
    python_executable: str = sys.executable,
    model_manager_path: Path = MODEL_MANAGER_PATH,
) -> dict[str, object]:
    """通过统一模型管理器执行不联网的静态结构检查。"""

    if not model_manager_path.is_file():
        raise StaticModelError(f"模型管理器不存在: {model_manager_path}")
    completed = runner(
        [
            python_executable,
            str(model_manager_path),
            "--json",
            "status",
            "--model",
            plan.model_alias,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    stdout = completed.stdout or ""
    try:
        report = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise StaticModelError(
            f"{plan.role} 模型静态检查未返回合法 JSON（rc={completed.returncode}）"
        ) from exc
    if not isinstance(report, dict):
        raise StaticModelError(f"{plan.role} 模型静态检查报告格式错误")
    reported_path = report.get("local_dir")
    if Path(str(reported_path)).resolve() != plan.model_path.resolve():
        raise StaticModelError(
            f"{plan.role} 模型路径与白名单不一致: {reported_path!r}"
        )
    if completed.returncode != 0 or report.get("ok") is not True:
        issue_codes = [
            issue.get("code", "unknown")
            for issue in report.get("issues", [])
            if isinstance(issue, dict)
        ]
        raise StaticModelError(
            f"{plan.role} 模型静态不完整: {plan.model_alias}; issues={issue_codes}"
        )
    report["expected_pinned_revision"] = plan.pinned_revision
    return report


def assert_ports_available(
    ports: Sequence[int],
    *,
    host: str = DEFAULT_SERVICE_HOST,
    socket_factory: Callable[..., socket.socket] = socket.socket,
) -> None:
    if len(ports) != len(set(ports)):
        raise PortUnavailableError("服务端口不能重复")
    sockets: list[socket.socket] = []
    try:
        for port in ports:
            if port < 1 or port > 65535:
                raise PortUnavailableError(f"非法 TCP 端口: {port}")
            candidate = socket_factory(socket.AF_INET, socket.SOCK_STREAM)
            sockets.append(candidate)
            try:
                candidate.bind((host, port))
            except OSError as exc:
                raise PortUnavailableError(f"端口 {host}:{port} 已被占用") from exc
    finally:
        for candidate in sockets:
            candidate.close()


def build_vllm_command(
    plan: ServicePlan,
    *,
    python_executable: str = str(DEFAULT_VLLM_PYTHON),
) -> tuple[str, ...]:
    command = [
        python_executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(plan.model_path),
        "--served-model-name",
        plan.served_model_name,
        "--host",
        plan.host,
        "--port",
        str(plan.port),
        "--tensor-parallel-size",
        str(plan.tensor_parallel_size),
        "--max-model-len",
        str(plan.max_model_len),
        "--max-num-seqs",
        "8",
        "--gpu-memory-utilization",
        str(plan.gpu_memory_utilization),
        "--generation-config",
        "vllm",
        "--reasoning-parser",
        "qwen3",
        # vLLM 0.25.x 使用 BooleanOptionalAction 风格参数；旧名称
        # ``--disable-log-requests`` 会在 CUDA 初始化前被 argparse 拒绝。
        "--no-enable-log-requests",
    ]
    if plan.language_model_only:
        command.append("--language-model-only")
    return tuple(command)


def build_service_environment(
    plan: ServicePlan,
    parent_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    source = os.environ if parent_environment is None else parent_environment
    environment = {
        key: value for key, value in source.items() if not _is_secret_environment_key(key)
    }
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(
        str(gpu_id) for gpu_id in plan.physical_gpu_ids
    )
    environment.update(VLLM_RUNTIME_OVERRIDES)
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    environment["TOKENIZERS_PARALLELISM"] = "false"
    return environment


def start_service(
    plan: ServicePlan,
    *,
    process_factory: ProcessFactory = subprocess.Popen,
    python_executable: str = str(DEFAULT_VLLM_PYTHON),
    parent_environment: Mapping[str, str] | None = None,
) -> ManagedProcess:
    plan.log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = plan.log_path.open("a", encoding="utf-8", buffering=1)
    command = build_vllm_command(plan, python_executable=python_executable)
    try:
        process = process_factory(
            list(command),
            cwd=str(CHESS_ROOT),
            env=build_service_environment(plan, parent_environment),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    except Exception:
        log_handle.close()
        raise
    return ManagedProcess(plan.role, process, command, plan.log_path, log_handle)


def _models_from_response(response: Mapping[str, object]) -> list[str]:
    data = response.get("data")
    if not isinstance(data, list):
        return []
    return [
        item["id"]
        for item in data
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]


def wait_until_ready(
    managed: ManagedProcess,
    plan: ServicePlan,
    *,
    requester: JsonRequester = _default_json_requester,
    timeout_seconds: float = DEFAULT_SERVICE_TIMEOUT_SECONDS,
    poll_interval_seconds: float = DEFAULT_HEALTH_POLL_SECONDS,
    monotonic: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    """等待模型列表接口，并执行一次小生成；两者成功才返回 READY。"""

    started = monotonic()
    deadline = started + timeout_seconds
    last_error = "尚未收到响应"
    observed_models: list[str] = []

    while monotonic() <= deadline:
        returncode = managed.process.poll()
        if returncode is not None:
            raise ServiceReadinessError(
                f"{plan.role} vLLM 在 READY 前退出，rc={returncode}; log={plan.log_path}"
            )
        try:
            response = requester(
                "GET", f"{plan.base_url}/models", None, min(10.0, timeout_seconds)
            )
            observed_models = _models_from_response(response)
            if plan.served_model_name in observed_models:
                break
            last_error = f"模型列表不含 {plan.served_model_name!r}: {observed_models}"
        except (OSError, ValueError, urllib.error.URLError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        sleeper(poll_interval_seconds)
    else:
        raise ServiceReadinessError(
            f"等待 {plan.role} vLLM 超时: {last_error}; log={plan.log_path}"
        )

    generation_request: dict[str, object] = {
        "model": plan.served_model_name,
        "messages": [
            {
                "role": "user",
                "content": "Reply with exactly: READY",
            }
        ],
        "temperature": 0,
        "max_tokens": 16,
        "seed": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        generation = requester(
            "POST",
            f"{plan.base_url}/chat/completions",
            generation_request,
            min(60.0, timeout_seconds),
        )
    except (OSError, ValueError, urllib.error.URLError) as exc:
        raise ServiceReadinessError(
            f"{plan.role} READY 生成失败: {type(exc).__name__}: {exc}"
        ) from exc

    choices = generation.get("choices")
    usage = generation.get("usage")
    if not isinstance(choices, list) or not choices:
        raise ServiceReadinessError(f"{plan.role} READY 生成缺少 choices")
    completion_tokens = (
        usage.get("completion_tokens") if isinstance(usage, dict) else None
    )
    first_choice = choices[0] if isinstance(choices[0], dict) else {}
    message = first_choice.get("message") if isinstance(first_choice, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    reasoning = (
        message.get("reasoning_content") if isinstance(message, dict) else None
    )
    produced_text = any(
        isinstance(value, str) and bool(value.strip()) for value in (content, reasoning)
    )
    produced_tokens = isinstance(completion_tokens, int) and completion_tokens > 0
    if not produced_text and not produced_tokens:
        raise ServiceReadinessError(f"{plan.role} READY 生成没有产生 token")

    return {
        "status": "READY",
        "role": plan.role,
        "model_alias": plan.model_alias,
        "served_model_name": plan.served_model_name,
        "pinned_revision": plan.pinned_revision,
        "base_url": plan.base_url,
        "observed_models": observed_models,
        "ready_after_seconds": monotonic() - started,
        "generation": {
            "completion_tokens": completion_tokens,
            "total_tokens": usage.get("total_tokens") if isinstance(usage, dict) else None,
            "produced_text": produced_text,
        },
    }


def build_chess_command(
    *,
    entrypoint: Path,
    actor_plan: ServicePlan,
    speculator_plan: ServicePlan,
    execution_mode: str,
    num_guesses: int,
    seed: int,
    run_id: str,
    stop_after: int,
    output_directory: Path,
    gpu_ids: Sequence[int],
    experiment_manifest: Path | None = None,
    topology_path: Path | None = None,
    python_executable: str = sys.executable,
) -> tuple[str, ...]:
    """独立构造 Chess runner 命令，便于接口变化时单点调整。"""

    command = [
        python_executable,
        str(entrypoint),
        "--gpu-count",
        str(len(gpu_ids)),
        "--gpu-ids",
        ",".join(str(gpu_id) for gpu_id in gpu_ids),
        "--execution-mode",
        execution_mode,
        "--actor-base-url",
        actor_plan.base_url,
        "--actor-model",
        actor_plan.served_model_name,
        "--speculator-base-url",
        speculator_plan.base_url,
        "--speculator-model",
        speculator_plan.served_model_name,
        "--num-guesses",
        str(num_guesses),
        "--seed",
        str(seed),
        "--run-id",
        run_id,
        "--stop-after",
        str(stop_after),
        "--output-dir",
        str(output_directory),
    ]
    if experiment_manifest is not None:
        command.extend(("--experiment-manifest", str(experiment_manifest)))
    if topology_path is not None:
        command.extend(("--topology-path", str(topology_path)))
    return tuple(command)


def start_chess_process(
    command: Sequence[str],
    log_path: Path,
    *,
    process_factory: ProcessFactory = subprocess.Popen,
) -> ManagedProcess:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", encoding="utf-8", buffering=1)
    try:
        environment = {
            key: value
            for key, value in os.environ.items()
            if not _is_secret_environment_key(key)
        }
        # Chess runner 只通过 HTTP 使用模型，禁止它意外占用实验 GPU。
        environment["CUDA_VISIBLE_DEVICES"] = ""
        environment["NO_PROXY"] = LOCAL_NO_PROXY
        environment["no_proxy"] = LOCAL_NO_PROXY
        process = process_factory(
            list(command),
            cwd=str(CHESS_ROOT),
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    except Exception:
        log_handle.close()
        raise
    return ManagedProcess("chess", process, tuple(command), log_path, log_handle)


def run_chess_process(
    command: Sequence[str],
    log_path: Path,
    *,
    process_factory: ProcessFactory = subprocess.Popen,
) -> tuple[int, ManagedProcess]:
    """兼容性封装；安全编排器应先取得句柄，再显式 ``wait``。"""

    managed = start_chess_process(
        command, log_path, process_factory=process_factory
    )
    return managed.process.wait(), managed


def terminate_managed_process(
    managed: ManagedProcess,
    *,
    group_signaler: GroupSignaler = os.killpg,
    group_checker: GroupChecker = process_group_exists,
    terminate_timeout_seconds: float = DEFAULT_TERMINATE_TIMEOUT_SECONDS,
) -> dict[str, object]:
    """只终止传入句柄对应的、由启动器创建的新进程组。"""

    initial_returncode: int | None = None
    term_sent = False
    kill_sent = False
    final_returncode = initial_returncode
    group_alive_initial: bool | None = None
    group_alive_final: bool | None = None
    error: str | None = None
    try:
        initial_returncode = managed.process.poll()
        final_returncode = initial_returncode
        # 子进程均以 start_new_session=True 启动，因此 PGID 等于启动时 PID。
        # 即使父进程已经退出，vLLM worker 也可能仍留在这个进程组中；不能只看
        # 父进程 poll() 的返回值决定是否清理。
        group_alive_initial = (
            True if initial_returncode is None else group_checker(managed.pid)
        )
        if group_alive_initial:
            group_signaler(managed.pid, signal.SIGTERM)
            term_sent = True

            if initial_returncode is None:
                try:
                    final_returncode = managed.process.wait(
                        timeout=terminate_timeout_seconds
                    )
                except subprocess.TimeoutExpired:
                    group_signaler(managed.pid, signal.SIGKILL)
                    kill_sent = True
                    final_returncode = managed.process.wait(
                        timeout=terminate_timeout_seconds
                    )

            # 父进程退出并不代表同组 worker 已退出。TERM 后继续核对整个 PGID；
            # 超时只升级清理这个已记录的进程组，不扫描或抢占任何其他任务。
            if not _wait_for_process_group_exit(
                managed.pid,
                group_checker=group_checker,
                timeout_seconds=terminate_timeout_seconds,
            ):
                group_signaler(managed.pid, signal.SIGKILL)
                kill_sent = True
                if initial_returncode is None and final_returncode is None:
                    final_returncode = managed.process.wait(
                        timeout=terminate_timeout_seconds
                    )
                _wait_for_process_group_exit(
                    managed.pid,
                    group_checker=group_checker,
                    timeout_seconds=terminate_timeout_seconds,
                )
        group_alive_final = group_checker(managed.pid)
        if group_alive_final:
            error = "ProcessGroupStillAlive: managed process group was not released"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if managed.log_handle is not None:
            close = getattr(managed.log_handle, "close", None)
            if callable(close):
                close()
    return {
        "role": managed.role,
        "pid": managed.pid,
        "initial_returncode": initial_returncode,
        "term_sent": term_sent,
        "kill_sent": kill_sent,
        "final_returncode": final_returncode,
        "group_alive_initial": group_alive_initial,
        "group_alive_final": group_alive_final,
        "error": error,
    }


def terminate_all(
    managed_processes: Sequence[ManagedProcess],
    *,
    group_signaler: GroupSignaler = os.killpg,
    terminate_timeout_seconds: float = DEFAULT_TERMINATE_TIMEOUT_SECONDS,
) -> list[dict[str, object]]:
    # 与启动相反顺序清理：Chess -> Speculator -> Actor。
    return [
        terminate_managed_process(
            managed,
            group_signaler=group_signaler,
            terminate_timeout_seconds=terminate_timeout_seconds,
        )
        for managed in reversed(managed_processes)
    ]


def public_command(command: Sequence[str]) -> list[str]:
    """当前命令不含 Key；集中保留该接口供 manifest 安全输出。"""

    return list(command)


def inspect_python_environment(python_executable: str) -> dict[str, object]:
    """只读采集指定 Python 的关键依赖与编译 CUDA 版本。"""

    program = r'''
import importlib.metadata
import json
import sys

def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None

torch_cuda = None
try:
    import torch
    torch_cuda = torch.version.cuda
except Exception:
    pass

print(json.dumps({
    "python": sys.version,
    "packages": {
        name: version(name)
        for name in ("vllm", "torch", "transformers", "openai", "chess")
    },
    "torch_compile_cuda": torch_cuda,
}))
'''
    environment = {
        key: value
        for key, value in os.environ.items()
        if not _is_secret_environment_key(key)
    }
    environment["CUDA_VISIBLE_DEVICES"] = ""
    try:
        completed = subprocess.run(
            [python_executable, "-c", program],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env=environment,
        )
        payload = json.loads(completed.stdout) if completed.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        return {
            "python_executable": python_executable,
            "status": "unavailable",
            "error_type": type(exc).__name__,
        }
    if not isinstance(payload, dict):
        return {
            "python_executable": python_executable,
            "status": "unavailable",
            "returncode": completed.returncode,
        }
    return {
        "python_executable": python_executable,
        "status": "ok",
        **payload,
    }


def workspace_git_head() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=WORKSPACE_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() or None
