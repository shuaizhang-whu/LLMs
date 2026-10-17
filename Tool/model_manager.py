#!/usr/bin/env python3
"""Qwen 实验模型的下载续传与完整性校验工具。

这个工具有意与 ``download_model.py`` 分离：它只允许管理实验计划中明确
批准的三个公开模型，不读取隐式 Hugging Face 凭据，也不会删除任何文件。

``verify`` 完成的是静态完整性校验。模型只有在外层 GPU 启动器完成 vLLM
加载和一次真实生成后，才应被实验系统标记为 ``READY``。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable, Mapping, Protocol, Sequence


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_ENDPOINT = "https://huggingface.co"
DEFAULT_MIRROR_ENDPOINT = "https://hf-mirror.com"
COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")
TOKEN_ENV_KEYS = {
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACEHUB_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
    "HUGGINGFACE_TOKEN",
    "HF_API_TOKEN",
}


@dataclass(frozen=True)
class ModelSpec:
    """白名单模型及其唯一允许的本地目录。"""

    alias: str
    repo_id: str
    local_dir: Path
    pinned_revision: str | None = None
    requires_generation_config: bool = False


MODEL_SPECS: dict[str, ModelSpec] = {
    "qwen3.6-27b-fp8": ModelSpec(
        alias="qwen3.6-27b-fp8",
        repo_id="Qwen/Qwen3.6-27B-FP8",
        local_dir=WORKSPACE_ROOT / "Models" / "Qwen3.6-27B-FP8",
        pinned_revision="e89b16ebf1988b3d6befa7de50abc2d76f26eb09",
        requires_generation_config=True,
    ),
    "qwen3.5-4b": ModelSpec(
        alias="qwen3.5-4b",
        repo_id="Qwen/Qwen3.5-4B",
        local_dir=WORKSPACE_ROOT / "Models" / "Qwen3.5-4B",
        pinned_revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    ),
    "qwen3.5-9b": ModelSpec(
        alias="qwen3.5-9b",
        repo_id="Qwen/Qwen3.5-9B",
        local_dir=WORKSPACE_ROOT / "Models" / "Qwen3.5-9B",
        pinned_revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    ),
}


# 这些是 vLLM 加载 Qwen 模型所需的最低静态结构。远端 manifest 校验还会
# 对比仓库中的每一个文件，因此 processor 等模型专有文件也不会被漏掉。
REQUIRED_EXACT_FILES = (
    "config.json",
    "model.safetensors.index.json",
    "tokenizer_config.json",
)
REQUIRED_ANY_GROUPS = {
    "tokenizer_data": ("tokenizer.json", "tokenizer.model", "vocab.json"),
    "processor_config": (
        "preprocessor_config.json",
        "processor_config.json",
        "video_preprocessor_config.json",
    ),
}
MAX_SAFETENSORS_HEADER_BYTES = 100 * 1024 * 1024


class ModelManagerError(RuntimeError):
    """可安全呈现给用户的模型管理错误。"""


class ManifestError(ModelManagerError):
    """远端 manifest 无法安全解析。"""


@dataclass(frozen=True)
class RemoteFile:
    path: str
    size: int | None = None
    lfs_sha256: str | None = None


@dataclass(frozen=True)
class RemoteManifest:
    repo_id: str
    revision: str
    files: tuple[RemoteFile, ...]
    endpoint: str = OFFICIAL_ENDPOINT


class ManifestProvider(Protocol):
    def fetch(self, spec: ModelSpec, revision: str | None = None) -> RemoteManifest:
        """解析 revision，并返回固定 commit 对应的完整文件清单。"""


@dataclass(frozen=True)
class Issue:
    code: str
    path: str | None
    message: str
    expected: object | None = None
    actual: object | None = None


@dataclass
class VerificationReport:
    alias: str
    repo_id: str
    local_dir: str
    revision: str | None = None
    endpoint: str | None = None
    issues: list[Issue] = field(default_factory=list)
    cache_verify_returncode: int | None = None
    remote_file_count: int | None = None
    checked_lfs_sha256: int = 0
    checked_json: int = 0
    checked_safetensors: int = 0

    @property
    def ok(self) -> bool:
        return not self.issues and self.cache_verify_returncode in (None, 0)

    @property
    def state(self) -> str:
        if not self.ok:
            return "INCOMPLETE"
        if self.revision is None:
            return "LOCAL_STATIC_OK"
        return "STATIC_VERIFIED"

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["ok"] = self.ok
        result["state"] = self.state
        # 静态校验不能代替由动态 GPU 启动器执行的 vLLM 加载测试。
        result["ready"] = False
        result["ready_requires"] = "vLLM load and one successful generation"
        return result


@dataclass(frozen=True)
class ResumeResult:
    download_returncode: int
    report: VerificationReport | None = None

    @property
    def returncode(self) -> int:
        if self.download_returncode != 0:
            return self.download_returncode
        if self.report is not None and not self.report.ok:
            return 1
        return 0


CommandRunner = Callable[..., subprocess.CompletedProcess]


def _safe_remote_path(value: str) -> str:
    """拒绝会逃逸模型目录的远端路径。"""

    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ManifestError("远端 manifest 包含不安全路径")
    return path.as_posix()


def _extract_status_code(exc: BaseException) -> int | None:
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def _is_connectivity_failure(exc: BaseException) -> bool:
    """仅网络不可达、超时或服务端故障允许降级到镜像。"""

    status = _extract_status_code(exc)
    if status is not None:
        return status >= 500
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True
    name = type(exc).__name__.lower()
    return "connect" in name or "timeout" in name


class HuggingFaceManifestProvider:
    """通过公开 Hub API 读取文件 metadata，不使用本地隐式 token。"""

    def __init__(
        self,
        official_endpoint: str = OFFICIAL_ENDPOINT,
        mirror_endpoint: str | None = DEFAULT_MIRROR_ENDPOINT,
        api_factory: Callable[..., object] | None = None,
    ) -> None:
        self.official_endpoint = official_endpoint.rstrip("/")
        self.mirror_endpoint = mirror_endpoint.rstrip("/") if mirror_endpoint else None
        self._api_factory = api_factory

    def _factory(self) -> Callable[..., object]:
        if self._api_factory is not None:
            return self._api_factory
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:  # pragma: no cover - 依赖错误由 CLI 呈现
            raise ModelManagerError("缺少 huggingface_hub；请在下载环境中安装") from exc
        return HfApi

    def _fetch_endpoint(
        self, endpoint: str, spec: ModelSpec, revision: str | None
    ) -> RemoteManifest:
        api = self._factory()(endpoint=endpoint, token=False)
        try:
            info = api.model_info(
                spec.repo_id,
                revision=revision or "main",
                files_metadata=True,
                token=False,
            )
        except Exception:
            raise

        resolved = getattr(info, "sha", None)
        if not isinstance(resolved, str) or not COMMIT_RE.fullmatch(resolved):
            raise ManifestError("Hub 未返回合法的 40 位 commit revision")

        remote_files: list[RemoteFile] = []
        for sibling in getattr(info, "siblings", None) or ():
            filename = _safe_remote_path(str(getattr(sibling, "rfilename", "")))
            lfs = getattr(sibling, "lfs", None)
            sha256 = getattr(lfs, "sha256", None) if lfs is not None else None
            lfs_size = getattr(lfs, "size", None) if lfs is not None else None
            size = lfs_size if isinstance(lfs_size, int) else getattr(sibling, "size", None)
            if size is not None and (not isinstance(size, int) or size < 0):
                raise ManifestError("Hub manifest 包含非法文件大小")
            if sha256 is not None:
                sha256 = str(sha256).lower()
                if not re.fullmatch(r"[0-9a-f]{64}", sha256):
                    raise ManifestError("Hub manifest 包含非法 LFS SHA-256")
            remote_files.append(RemoteFile(filename, size, sha256))

        if not remote_files:
            raise ManifestError("Hub manifest 为空")
        if len({item.path for item in remote_files}) != len(remote_files):
            raise ManifestError("Hub manifest 包含重复路径")
        return RemoteManifest(
            repo_id=spec.repo_id,
            revision=resolved.lower(),
            files=tuple(sorted(remote_files, key=lambda item: item.path)),
            endpoint=endpoint,
        )

    def fetch(self, spec: ModelSpec, revision: str | None = None) -> RemoteManifest:
        if revision is not None and not revision.strip():
            raise ManifestError("revision 不能为空")
        try:
            return self._fetch_endpoint(self.official_endpoint, spec, revision)
        except Exception as exc:
            if isinstance(exc, ManifestError) or not _is_connectivity_failure(exc):
                status = _extract_status_code(exc)
                suffix = f"（HTTP {status}）" if status is not None else ""
                raise ManifestError(f"无法从官方 Hub 获取 manifest{suffix}") from None
            if self.mirror_endpoint is None:
                raise ManifestError("官方 Hub 当前不可达，且未启用镜像回退") from None

        try:
            return self._fetch_endpoint(self.mirror_endpoint, spec, revision)
        except Exception as exc:
            status = _extract_status_code(exc)
            suffix = f"（HTTP {status}）" if status is not None else ""
            raise ManifestError(f"官方 Hub 不可达且镜像 manifest 获取失败{suffix}") from None


def get_model_spec(alias: str) -> ModelSpec:
    try:
        return MODEL_SPECS[alias.lower()]
    except KeyError:
        allowed = ", ".join(sorted(MODEL_SPECS))
        raise ModelManagerError(f"未知模型 {alias!r}；仅允许：{allowed}") from None


def _iter_model_files(local_dir: Path) -> Iterable[Path]:
    if not local_dir.is_dir():
        return ()
    return (
        path
        for path in local_dir.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(local_dir).parts
    )


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _check_json(path: Path) -> None:
    with path.open("r", encoding="utf-8") as handle:
        json.load(handle)


def _check_safetensors_header(path: Path) -> None:
    """只读取 safetensors header，并验证 tensor offsets 没有越界。"""

    file_size = path.stat().st_size
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError("文件短于 safetensors 头部长度字段")
        (header_size,) = struct.unpack("<Q", prefix)
        if (
            header_size <= 1
            or header_size > MAX_SAFETENSORS_HEADER_BYTES
            or header_size > file_size - 8
        ):
            raise ValueError("safetensors header 长度非法")
        header_bytes = handle.read(header_size)
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("safetensors header 不是合法 JSON") from exc
    if not isinstance(header, dict):
        raise ValueError("safetensors header 顶层必须是对象")

    data_size = file_size - 8 - header_size
    tensor_count = 0
    for name, metadata in header.items():
        if name == "__metadata__":
            continue
        tensor_count += 1
        if not isinstance(metadata, dict):
            raise ValueError("tensor metadata 必须是对象")
        offsets = metadata.get("data_offsets")
        shape = metadata.get("shape")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(value, int) for value in offsets)
            or offsets[0] < 0
            or offsets[0] > offsets[1]
            or offsets[1] > data_size
        ):
            raise ValueError("tensor data_offsets 非法或越界")
        if not isinstance(shape, list) or any(
            not isinstance(value, int) or value < 0 for value in shape
        ):
            raise ValueError("tensor shape 非法")
        if not isinstance(metadata.get("dtype"), str):
            raise ValueError("tensor dtype 缺失")
    if tensor_count == 0:
        raise ValueError("safetensors header 不包含 tensor")


def _add_structure_issues(spec: ModelSpec, report: VerificationReport) -> None:
    local_dir = spec.local_dir
    if not local_dir.is_dir():
        report.issues.append(
            Issue("missing_model_directory", None, "本地模型目录不存在")
        )
        return

    for filename in REQUIRED_EXACT_FILES:
        path = local_dir / filename
        if not path.is_file():
            report.issues.append(
                Issue("missing_required_file", filename, "缺少必需文件")
            )
    # Qwen3.5-4B/9B 的已固定官方 commit 本身不提供此文件；27B-FP8
    # 官方仓库提供且 vLLM 会读取，因此只对后者强制要求。
    if spec.requires_generation_config:
        filename = "generation_config.json"
        if not (local_dir / filename).is_file():
            report.issues.append(
                Issue("missing_required_file", filename, "缺少必需文件")
            )
    for group, filenames in REQUIRED_ANY_GROUPS.items():
        if not any((local_dir / filename).is_file() for filename in filenames):
            report.issues.append(
                Issue(
                    "missing_required_group",
                    group,
                    "该类别至少需要一个文件",
                    expected=list(filenames),
                )
            )


def _add_index_issues(local_dir: Path, report: VerificationReport) -> None:
    index_path = local_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        return
    try:
        with index_path.open("r", encoding="utf-8") as handle:
            index = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return  # 通用 JSON 检查会给出单一、明确的错误。
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        report.issues.append(
            Issue("invalid_weight_map", index_path.name, "权重索引缺少非空 weight_map")
        )
        return
    invalid_shard_names = any(not isinstance(value, str) for value in weight_map.values())
    shards = {value for value in weight_map.values() if isinstance(value, str)}
    if invalid_shard_names:
        report.issues.append(
            Issue("invalid_shard_name", index_path.name, "weight_map 包含非字符串分片名")
        )
    for shard in sorted(shards):
        try:
            safe_shard = _safe_remote_path(shard)
        except ManifestError:
            report.issues.append(
                Issue("unsafe_shard_path", index_path.name, "权重索引包含不安全路径")
            )
            continue
        if not (local_dir / safe_shard).is_file():
            report.issues.append(
                Issue("missing_indexed_shard", safe_shard, "权重索引引用的分片不存在")
            )


def inspect_local_model(
    spec: ModelSpec, manifest: RemoteManifest | None = None
) -> VerificationReport:
    """检查本地结构；传入 manifest 时额外执行逐文件远端对比。"""

    report = VerificationReport(
        alias=spec.alias,
        repo_id=spec.repo_id,
        local_dir=str(spec.local_dir),
        revision=manifest.revision if manifest else None,
        endpoint=manifest.endpoint if manifest else None,
        remote_file_count=len(manifest.files) if manifest else None,
    )
    _add_structure_issues(spec, report)
    if not spec.local_dir.is_dir():
        return report

    if manifest is None:
        files = list(_iter_model_files(spec.local_dir))
    else:
        remote_by_path = {item.path: item for item in manifest.files}
        files = []
        for item in manifest.files:
            path = spec.local_dir / item.path
            if not path.is_file():
                report.issues.append(
                    Issue("missing_remote_file", item.path, "远端文件在本地缺失")
                )
                continue
            files.append(path)
            actual_size = path.stat().st_size
            if item.size is not None and actual_size != item.size:
                report.issues.append(
                    Issue(
                        "size_mismatch",
                        item.path,
                        "文件大小与远端 manifest 不一致",
                        expected=item.size,
                        actual=actual_size,
                    )
                )
            if item.lfs_sha256 is not None:
                actual_hash = _sha256(path)
                report.checked_lfs_sha256 += 1
                if actual_hash != item.lfs_sha256:
                    report.issues.append(
                        Issue(
                            "sha256_mismatch",
                            item.path,
                            "LFS SHA-256 与远端 manifest 不一致",
                            expected=item.lfs_sha256,
                            actual=actual_hash,
                        )
                    )

        for path in _iter_model_files(spec.local_dir):
            relative = path.relative_to(spec.local_dir).as_posix()
            if relative not in remote_by_path:
                report.issues.append(
                    Issue("extra_local_file", relative, "本地文件不在远端 manifest 中")
                )

    for path in files:
        relative = path.relative_to(spec.local_dir).as_posix()
        if path.suffix.lower() == ".json":
            report.checked_json += 1
            try:
                _check_json(path)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                report.issues.append(
                    Issue("invalid_json", relative, "JSON 文件无法解析")
                )
        if path.suffix.lower() == ".safetensors":
            report.checked_safetensors += 1
            try:
                _check_safetensors_header(path)
            except (OSError, ValueError):
                report.issues.append(
                    Issue(
                        "invalid_safetensors_header",
                        relative,
                        "safetensors header 无法解析或 tensor offset 越界",
                    )
                )

    _add_index_issues(spec.local_dir, report)
    return report


def _public_hf_environment(endpoint: str) -> dict[str, str]:
    """构造不携带任何 HF token 的子进程环境。"""

    env = {key: value for key, value in os.environ.items() if key not in TOKEN_ENV_KEYS}
    env["HF_ENDPOINT"] = endpoint
    env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    return env


def find_hf_executable(explicit: str | None = None) -> str:
    candidates = [explicit, shutil.which("hf"), str(Path(sys.executable).with_name("hf"))]
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return str(Path(candidate))
    raise ModelManagerError("找不到 hf CLI；请激活包含 huggingface_hub 的环境或传入 --hf-executable")


def _default_runner(
    command: Sequence[str], *, env: Mapping[str, str]
) -> subprocess.CompletedProcess:
    return subprocess.run(list(command), env=dict(env), check=False)


def verify_model(
    spec: ModelSpec,
    provider: ManifestProvider,
    *,
    revision: str | None = None,
    manifest: RemoteManifest | None = None,
    runner: CommandRunner = _default_runner,
    hf_executable: str | None = None,
) -> VerificationReport:
    manifest = manifest or provider.fetch(spec, revision or spec.pinned_revision)
    if manifest.repo_id != spec.repo_id:
        raise ManifestError("manifest repo_id 与白名单模型不一致")
    if not COMMIT_RE.fullmatch(manifest.revision):
        raise ManifestError("manifest revision 不是固定 commit")

    report = inspect_local_model(spec, manifest)
    executable = find_hf_executable(hf_executable)
    command = [
        executable,
        "cache",
        "verify",
        spec.repo_id,
        "--revision",
        manifest.revision,
        "--local-dir",
        str(spec.local_dir),
        "--fail-on-missing-files",
    ]
    # ``hf download --local-dir`` 会在模型目录内维护
    # ``.cache/huggingface/**``，而 ``hf cache verify
    # --fail-on-extra-files`` 会把它自己创建的这些元数据误报为额外模型文件。
    # 上面的逐文件 manifest 对比已经严格检查额外文件，并只排除该内部缓存，
    # 因此这里保留 missing/checksum 校验但不重复启用有冲突的 extra 检查。
    completed = runner(command, env=_public_hf_environment(manifest.endpoint))
    report.cache_verify_returncode = int(completed.returncode)
    if completed.returncode != 0:
        report.issues.append(
            Issue("hf_cache_verify_failed", None, "hf cache verify 返回非零退出码", actual=int(completed.returncode))
        )
    return report


def resume_model(
    spec: ModelSpec,
    provider: ManifestProvider,
    *,
    revision: str | None = None,
    runner: CommandRunner = _default_runner,
    hf_executable: str | None = None,
) -> ResumeResult:
    """固定 commit 后原地续传；下载失败码原样向上传播。"""

    manifest = provider.fetch(spec, revision or spec.pinned_revision)
    executable = find_hf_executable(hf_executable)
    spec.local_dir.mkdir(parents=True, exist_ok=True)
    command = [
        executable,
        "download",
        "--repo-type",
        "model",
        spec.repo_id,
        "--revision",
        manifest.revision,
        "--local-dir",
        str(spec.local_dir),
    ]
    completed = runner(command, env=_public_hf_environment(manifest.endpoint))
    download_returncode = int(completed.returncode)
    if download_returncode != 0:
        return ResumeResult(download_returncode=download_returncode)

    report = verify_model(
        spec,
        provider,
        manifest=manifest,
        runner=runner,
        hf_executable=executable,
    )
    return ResumeResult(download_returncode=0, report=report)


def _print_report(report: VerificationReport, as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return
    print(f"[{report.alias}] {report.state}")
    print(f"  repo: {report.repo_id}")
    print(f"  local: {report.local_dir}")
    if report.revision:
        print(f"  revision: {report.revision}")
    if report.remote_file_count is not None:
        print(f"  remote files: {report.remote_file_count}")
    print(
        "  checked: "
        f"json={report.checked_json}, safetensors={report.checked_safetensors}, "
        f"lfs_sha256={report.checked_lfs_sha256}"
    )
    if report.cache_verify_returncode is not None:
        print(f"  hf cache verify: rc={report.cache_verify_returncode}")
    if report.issues:
        print(f"  issues: {len(report.issues)}")
        for issue in report.issues:
            location = f" ({issue.path})" if issue.path else ""
            print(f"    - {issue.code}{location}: {issue.message}")
    else:
        print("  issues: none")
    print("  READY: no（仍需通过动态 GPU 启动器完成 vLLM 加载与生成）")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf-executable",
        help="hf CLI 路径；默认从当前环境自动发现",
    )
    parser.add_argument(
        "--mirror-endpoint",
        default=DEFAULT_MIRROR_ENDPOINT,
        help="仅在官方 Hub 网络不可达时使用；传空字符串可禁用",
    )
    parser.add_argument("--json", action="store_true", help="以 JSON 输出报告")
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status", help="执行不联网的本地快速检查")
    status.add_argument("--model", choices=sorted(MODEL_SPECS))

    verify = subparsers.add_parser("verify", help="按远端固定 commit 完整校验")
    verify.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    verify.add_argument("--revision", help="branch/tag/commit；最终总会解析成 commit")

    resume = subparsers.add_parser("resume", help="固定 commit 并原地断点续传后校验")
    resume.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    resume.add_argument("--revision", help="branch/tag/commit；最终总会解析成 commit")

    subparsers.add_parser("verify-all", help="依次完整校验三个白名单模型")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    provider: ManifestProvider | None = None,
    runner: CommandRunner = _default_runner,
) -> int:
    args = build_parser().parse_args(argv)
    provider = provider or HuggingFaceManifestProvider(
        mirror_endpoint=args.mirror_endpoint or None
    )

    try:
        if args.command == "status":
            specs = [get_model_spec(args.model)] if args.model else list(MODEL_SPECS.values())
            reports = [inspect_local_model(spec) for spec in specs]
            for report in reports:
                _print_report(report, args.json)
            return 0 if all(report.ok for report in reports) else 1

        if args.command == "verify":
            report = verify_model(
                get_model_spec(args.model),
                provider,
                revision=args.revision,
                runner=runner,
                hf_executable=args.hf_executable,
            )
            _print_report(report, args.json)
            return 0 if report.ok else 1

        if args.command == "resume":
            result = resume_model(
                get_model_spec(args.model),
                provider,
                revision=args.revision,
                runner=runner,
                hf_executable=args.hf_executable,
            )
            if result.report is not None:
                _print_report(result.report, args.json)
            elif not args.json:
                print(f"下载命令失败：rc={result.download_returncode}")
            else:
                print(json.dumps({"download_returncode": result.download_returncode}))
            return result.returncode

        reports = []
        for spec in MODEL_SPECS.values():
            try:
                report = verify_model(
                    spec,
                    provider,
                    runner=runner,
                    hf_executable=args.hf_executable,
                )
            except ModelManagerError as exc:
                report = inspect_local_model(spec)
                report.issues.append(Issue("verification_error", None, str(exc)))
            reports.append(report)
            _print_report(report, args.json)
        return 0 if all(report.ok for report in reports) else 1
    except ModelManagerError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
