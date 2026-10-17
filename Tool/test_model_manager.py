"""model_manager 的纯离线单元测试。"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_manager as mm  # noqa: E402


REVISION = "a" * 40


def write_json(path: Path, value: object | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({} if value is None else value), encoding="utf-8")


def write_safetensors(path: Path) -> None:
    header = json.dumps(
        {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
        separators=(",", ":"),
    ).encode("utf-8")
    header += b" " * ((8 - len(header) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0\0\0\0")


def make_model(root: Path, alias: str = "fixture") -> mm.ModelSpec:
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "config.json")
    write_json(root / "generation_config.json")
    write_json(root / "tokenizer_config.json")
    write_json(root / "tokenizer.json")
    write_json(root / "preprocessor_config.json")
    shard = "model-00001-of-00001.safetensors"
    write_safetensors(root / shard)
    write_json(root / "model.safetensors.index.json", {"weight_map": {"weight": shard}})
    return mm.ModelSpec(alias=alias, repo_id=f"Qwen/{alias}", local_dir=root)


def make_manifest(spec: mm.ModelSpec, *, endpoint: str = mm.OFFICIAL_ENDPOINT) -> mm.RemoteManifest:
    files = []
    for path in sorted(spec.local_dir.iterdir()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.suffix == ".safetensors" else None
        files.append(mm.RemoteFile(path.name, path.stat().st_size, digest))
    return mm.RemoteManifest(spec.repo_id, REVISION, tuple(files), endpoint)


class FakeProvider:
    def __init__(self, manifest: mm.RemoteManifest):
        self.manifest = manifest
        self.calls: list[tuple[mm.ModelSpec, str | None]] = []

    def fetch(self, spec: mm.ModelSpec, revision: str | None = None) -> mm.RemoteManifest:
        self.calls.append((spec, revision))
        return self.manifest


class RecordingRunner:
    def __init__(self, returncodes: list[int] | None = None):
        self.returncodes = list(returncodes or [0])
        self.calls: list[tuple[list[str], dict[str, str]]] = []

    def __call__(self, command, *, env):
        self.calls.append((list(command), dict(env)))
        rc = self.returncodes.pop(0) if self.returncodes else 0
        return subprocess.CompletedProcess(command, rc)


class LocalInspectionTests(unittest.TestCase):
    def test_complete_fixture_passes_local_static_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = make_model(Path(tmp) / "model")
            report = mm.inspect_local_model(spec)
            self.assertTrue(report.ok, report.issues)
            self.assertEqual(report.state, "LOCAL_STATIC_OK")
            self.assertFalse(report.to_dict()["ready"])
            self.assertEqual(report.checked_safetensors, 1)

    def test_generation_config_is_model_specific(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = make_model(Path(tmp) / "model")
            (base.local_dir / "generation_config.json").unlink()
            optional = mm.inspect_local_model(base)
            self.assertTrue(optional.ok, optional.issues)

            required = mm.ModelSpec(
                base.alias,
                base.repo_id,
                base.local_dir,
                requires_generation_config=True,
            )
            strict = mm.inspect_local_model(required)
            self.assertIn(
                "generation_config.json",
                {issue.path for issue in strict.issues if issue.code == "missing_required_file"},
            )

    def test_missing_files_and_indexed_shard_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "model"
            root.mkdir()
            write_json(root / "model.safetensors.index.json", {"weight_map": {"x": "missing.safetensors"}})
            spec = mm.ModelSpec("fixture", "Qwen/fixture", root)
            report = mm.inspect_local_model(spec)
            codes = {issue.code for issue in report.issues}
            self.assertIn("missing_required_file", codes)
            self.assertIn("missing_required_group", codes)
            self.assertIn("missing_indexed_shard", codes)

    def test_invalid_json_and_safetensors_header_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = make_model(Path(tmp) / "model")
            (spec.local_dir / "config.json").write_text("{broken", encoding="utf-8")
            (spec.local_dir / "model-00001-of-00001.safetensors").write_bytes(b"broken")
            report = mm.inspect_local_model(spec)
            codes = {issue.code for issue in report.issues}
            self.assertIn("invalid_json", codes)
            self.assertIn("invalid_safetensors_header", codes)

    def test_remote_size_hash_missing_and_extra_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = make_model(Path(tmp) / "model")
            manifest = make_manifest(spec)
            files = list(manifest.files)
            shard_index = next(i for i, item in enumerate(files) if item.path.endswith(".safetensors"))
            shard = files[shard_index]
            files[shard_index] = mm.RemoteFile(shard.path, shard.size + 1, "0" * 64)
            files.append(mm.RemoteFile("remote-only.txt", 1, None))
            (spec.local_dir / "extra.txt").write_text("extra", encoding="utf-8")
            changed = mm.RemoteManifest(spec.repo_id, REVISION, tuple(files))
            report = mm.inspect_local_model(spec, changed)
            codes = {issue.code for issue in report.issues}
            self.assertIn("size_mismatch", codes)
            self.assertIn("sha256_mismatch", codes)
            self.assertIn("missing_remote_file", codes)
            self.assertIn("extra_local_file", codes)

    def test_huggingface_local_cache_metadata_is_not_an_extra_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = make_model(Path(tmp) / "model")
            manifest = make_manifest(spec)
            cache_file = spec.local_dir / ".cache" / "huggingface" / "download" / "metadata"
            cache_file.parent.mkdir(parents=True)
            cache_file.write_text("ignored", encoding="utf-8")
            report = mm.inspect_local_model(spec, manifest)
            self.assertTrue(report.ok, report.issues)


class VerificationTests(unittest.TestCase):
    def test_default_revision_uses_model_pin(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = make_model(Path(tmp) / "model")
            spec = mm.ModelSpec(
                base.alias,
                base.repo_id,
                base.local_dir,
                pinned_revision=REVISION,
            )
            provider = FakeProvider(make_manifest(spec))
            report = mm.verify_model(
                spec,
                provider,
                runner=RecordingRunner(),
                hf_executable=sys.executable,
            )
            self.assertTrue(report.ok, report.issues)
            self.assertEqual(provider.calls[0][1], REVISION)

    def test_verify_uses_fixed_revision_strict_cache_check_and_no_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = make_model(Path(tmp) / "model")
            manifest = make_manifest(spec)
            provider = FakeProvider(manifest)
            runner = RecordingRunner()
            token_environment = {key: "test-secret" for key in mm.TOKEN_ENV_KEYS}
            with mock.patch.dict(os.environ, token_environment):
                report = mm.verify_model(
                    spec,
                    provider,
                    revision="main",
                    runner=runner,
                    hf_executable=sys.executable,
                )
            self.assertTrue(report.ok, report.issues)
            self.assertEqual(provider.calls[0][1], "main")
            command, env = runner.calls[0]
            self.assertEqual(command[1:3], ["cache", "verify"])
            self.assertIn(REVISION, command)
            self.assertIn("--fail-on-missing-files", command)
            self.assertNotIn("--fail-on-extra-files", command)
            self.assertTrue(mm.TOKEN_ENV_KEYS.isdisjoint(env))
            self.assertEqual(env["HF_HUB_DISABLE_IMPLICIT_TOKEN"], "1")

    def test_nonzero_hf_cache_verify_fails_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = make_model(Path(tmp) / "model")
            runner = RecordingRunner([9])
            report = mm.verify_model(
                spec,
                FakeProvider(make_manifest(spec)),
                runner=runner,
                hf_executable=sys.executable,
            )
            self.assertFalse(report.ok)
            self.assertEqual(report.cache_verify_returncode, 9)
            self.assertIn("hf_cache_verify_failed", {issue.code for issue in report.issues})

    def test_resume_propagates_download_failure_without_verifying(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = make_model(Path(tmp) / "model")
            provider = FakeProvider(make_manifest(spec))
            runner = RecordingRunner([23])
            result = mm.resume_model(
                spec,
                provider,
                revision="release-tag",
                runner=runner,
                hf_executable=sys.executable,
            )
            self.assertEqual(result.returncode, 23)
            self.assertIsNone(result.report)
            self.assertEqual(len(runner.calls), 1)
            command = runner.calls[0][0]
            self.assertEqual(command[1], "download")
            self.assertIn(REVISION, command)

    def test_successful_resume_downloads_then_verifies(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = make_model(Path(tmp) / "model")
            provider = FakeProvider(make_manifest(spec))
            runner = RecordingRunner([0, 0])
            result = mm.resume_model(
                spec,
                provider,
                runner=runner,
                hf_executable=sys.executable,
            )
            self.assertEqual(result.returncode, 0)
            self.assertTrue(result.report.ok)
            self.assertEqual([call[0][1] for call in runner.calls], ["download", "cache"])


class ManifestProviderTests(unittest.TestCase):
    @dataclass
    class Lfs:
        size: int
        sha256: str

    @dataclass
    class Sibling:
        rfilename: str
        size: int | None = None
        lfs: object | None = None

    @dataclass
    class Info:
        sha: str
        siblings: list[object]

    def test_official_connectivity_failure_falls_back_to_mirror(self):
        calls = []

        class Api:
            def __init__(self, endpoint, token):
                self.endpoint = endpoint
                calls.append((endpoint, token))

            def model_info(self, *args, **kwargs):
                if self.endpoint == mm.OFFICIAL_ENDPOINT:
                    raise TimeoutError()
                return ManifestProviderTests.Info(
                    REVISION,
                    [ManifestProviderTests.Sibling("config.json", 2)],
                )

        provider = mm.HuggingFaceManifestProvider(api_factory=Api)
        spec = mm.ModelSpec("fixture", "Qwen/fixture", Path("/unused"))
        manifest = provider.fetch(spec)
        self.assertEqual(manifest.endpoint, mm.DEFAULT_MIRROR_ENDPOINT)
        self.assertEqual(manifest.revision, REVISION)
        self.assertEqual(calls[0], (mm.OFFICIAL_ENDPOINT, False))

    def test_http_4xx_does_not_fall_back_to_mirror(self):
        calls = []

        class Response:
            status_code = 404

        class HttpError(Exception):
            response = Response()

        class Api:
            def __init__(self, endpoint, token):
                self.endpoint = endpoint
                calls.append(endpoint)

            def model_info(self, *args, **kwargs):
                raise HttpError()

        provider = mm.HuggingFaceManifestProvider(api_factory=Api)
        spec = mm.ModelSpec("fixture", "Qwen/fixture", Path("/unused"))
        with self.assertRaises(mm.ManifestError):
            provider.fetch(spec)
        self.assertEqual(calls, [mm.OFFICIAL_ENDPOINT])

    def test_lfs_metadata_and_revision_are_validated(self):
        class Api:
            def __init__(self, endpoint, token):
                pass

            def model_info(self, *args, **kwargs):
                return ManifestProviderTests.Info(
                    REVISION,
                    [
                        ManifestProviderTests.Sibling(
                            "model.safetensors",
                            lfs=ManifestProviderTests.Lfs(7, "b" * 64),
                        )
                    ],
                )

        provider = mm.HuggingFaceManifestProvider(api_factory=Api)
        spec = mm.ModelSpec("fixture", "Qwen/fixture", Path("/unused"))
        manifest = provider.fetch(spec, "main")
        self.assertEqual(manifest.files[0].size, 7)
        self.assertEqual(manifest.files[0].lfs_sha256, "b" * 64)

    def test_unsafe_remote_path_is_rejected(self):
        with self.assertRaises(mm.ManifestError):
            mm._safe_remote_path("../escape")


class CliTests(unittest.TestCase):
    def test_unknown_model_is_rejected_by_whitelist(self):
        with self.assertRaises(mm.ModelManagerError):
            mm.get_model_spec("unapproved")

    def test_verify_all_is_offline_when_provider_and_runner_are_injected(self):
        with tempfile.TemporaryDirectory() as tmp:
            specs = {}
            manifests = {}
            for name in ("one", "two", "three"):
                spec = make_model(Path(tmp) / name, alias=name)
                specs[name] = spec
                manifests[name] = make_manifest(spec)

            class Provider:
                def fetch(self, spec, revision=None):
                    return manifests[spec.alias]

            runner = RecordingRunner([0, 0, 0])
            stdout = io.StringIO()
            with mock.patch.object(mm, "MODEL_SPECS", specs), contextlib.redirect_stdout(stdout):
                rc = mm.main(
                    ["--hf-executable", sys.executable, "verify-all"],
                    provider=Provider(),
                    runner=runner,
                )
            self.assertEqual(rc, 0, stdout.getvalue())
            self.assertEqual(len(runner.calls), 3)


if __name__ == "__main__":
    unittest.main()
