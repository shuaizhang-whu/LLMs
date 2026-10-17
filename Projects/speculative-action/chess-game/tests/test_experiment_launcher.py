from __future__ import annotations

import contextlib
import io
import json
import signal
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.gpu_resources import GPUInfo, GPUSample, NvidiaSmiClient
from scripts.run_local_experiment import _release_report, executable_path, main
from scripts.service_manager import (
    ManagedProcess,
    PortUnavailableError,
    ServiceReadinessError,
    StaticModelError,
    assert_ports_available,
    _default_json_requester,
    build_chess_command,
    build_service_environment,
    build_vllm_command,
    service_plans_from_allocation,
    start_chess_process,
    start_service,
    terminate_managed_process,
    verify_static_model,
    wait_until_ready,
)


def allocation(gpu_count: int = 3) -> dict[str, object]:
    if gpu_count == 2:
        return {
            "actor": {
                "model_key": "qwen3.5-9b",
                "gpu_ids": [0],
                "tensor_parallel_size": 1,
            },
            "speculator": {
                "model_key": "qwen3.5-4b",
                "gpu_ids": [1],
                "tensor_parallel_size": 1,
            },
        }
    return {
        "actor": {
            "model_key": "qwen3.6-27b-fp8",
            "gpu_ids": [0, 1],
            "tensor_parallel_size": 2,
        },
        "speculator": {
            "model_key": "qwen3.5-4b",
            "gpu_ids": list(range(2, gpu_count)),
            "tensor_parallel_size": gpu_count - 2,
        },
    }


def idle_gpu_client(gpu_count: int = 4) -> NvidiaSmiClient:
    rows = "".join(
        f"{index}, GPU-{index}, NVIDIA RTX 3090, 100, 24576, 0\n"
        for index in range(gpu_count)
    )

    def runner(command):
        if "--query-gpu=" in command[1]:
            return rows
        return ""

    return NvidiaSmiClient(runner, owner_lookup=lambda pid: None)


class FakeProcess:
    def __init__(self, pid: int, returncode: int = 0, running: bool = True):
        self.pid = pid
        self.returncode = None if running else returncode
        self.wait_returncode = returncode
        self.wait_calls = []

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        self.returncode = self.wait_returncode
        return self.returncode


def managed(role: str, pid: int, returncode: int = 0) -> ManagedProcess:
    return ManagedProcess(
        role=role,
        process=FakeProcess(pid, returncode=returncode),
        command=(role, "command"),
        log_path=Path(f"/{role}.log"),
        log_handle=io.StringIO(),
    )


class PlanAndCommandTests(unittest.TestCase):
    def test_three_and_four_card_plans_preserve_strict_tp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            actor, speculator = service_plans_from_allocation(
                allocation(3),
                actor_port=8100,
                speculator_port=8101,
                run_directory=Path(directory),
                max_model_len=8192,
                gpu_memory_utilization=0.90,
            )
            self.assertEqual(actor.physical_gpu_ids, (0, 1))
            self.assertEqual(actor.tensor_parallel_size, 2)
            self.assertEqual(speculator.physical_gpu_ids, (2,))
            self.assertEqual(speculator.tensor_parallel_size, 1)
            self.assertEqual(actor.max_model_len, 8192)
            self.assertEqual(actor.gpu_memory_utilization, 0.90)

            _, speculator4 = service_plans_from_allocation(
                allocation(4),
                actor_port=8100,
                speculator_port=8101,
                run_directory=Path(directory),
            )
            self.assertEqual(speculator4.physical_gpu_ids, (2, 3))
            self.assertEqual(speculator4.tensor_parallel_size, 2)

    def test_vllm_command_explicitly_limits_context_and_memory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            actor, _ = service_plans_from_allocation(
                allocation(3),
                actor_port=8100,
                speculator_port=8101,
                run_directory=Path(directory),
            )
            command = build_vllm_command(actor, python_executable="/vllm/python")
            self.assertEqual(command[0], "/vllm/python")
            self.assertEqual(command[command.index("--tensor-parallel-size") + 1], "2")
            self.assertEqual(command[command.index("--max-model-len") + 1], "8192")
            self.assertEqual(
                command[command.index("--gpu-memory-utilization") + 1], "0.9"
            )
            self.assertIn("--language-model-only", command)
            self.assertEqual(command[command.index("--reasoning-parser") + 1], "qwen3")
            self.assertIn("--no-enable-log-requests", command)
            self.assertNotIn("--disable-log-requests", command)

    def test_runner_executable_keeps_virtualenv_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_python = root / "base-python"
            base_python.write_text("", encoding="utf-8")
            venv_python = root / "venv-python"
            venv_python.symlink_to(base_python)

            self.assertEqual(executable_path(venv_python), venv_python.absolute())
            self.assertNotEqual(executable_path(venv_python), venv_python.resolve())

    def test_environment_is_gpu_isolated_and_drops_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            actor, _ = service_plans_from_allocation(
                allocation(3),
                actor_port=8100,
                speculator_port=8101,
                run_directory=Path(directory),
            )
            environment = build_service_environment(
                actor,
                {
                    "PATH": "/bin",
                    "OPENAI_API_KEY": "secret",
                    "HF_TOKEN": "secret",
                    "UNLISTED_SERVICE_TOKEN": "secret",
                    "DATABASE_PASSWORD": "secret",
                },
            )
            self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "0,1")
            self.assertEqual(environment["HF_HUB_OFFLINE"], "1")
            self.assertEqual(environment["VLLM_USE_FLASHINFER_SAMPLER"], "0")
            self.assertNotIn("OPENAI_API_KEY", environment)
            self.assertNotIn("HF_TOKEN", environment)
            self.assertNotIn("UNLISTED_SERVICE_TOKEN", environment)
            self.assertNotIn("DATABASE_PASSWORD", environment)

    def test_chess_command_optional_manifest_flags_are_decoupled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actor, speculator = service_plans_from_allocation(
                allocation(3),
                actor_port=8100,
                speculator_port=8101,
                run_directory=root,
            )
            command = build_chess_command(
                entrypoint=root / "run_chess_local.py",
                actor_plan=actor,
                speculator_plan=speculator,
                execution_mode="speculative",
                num_guesses=3,
                seed=248,
                run_id="smoke",
                stop_after=6,
                output_directory=root / "artifacts",
                gpu_ids=(0, 1, 2),
                experiment_manifest=root / "manifest.json",
                topology_path=root / "topology.json",
                python_executable="/runner/python",
            )
            self.assertIn("--experiment-manifest", command)
            self.assertIn("--topology-path", command)
            self.assertEqual(command[command.index("--gpu-count") + 1], "3")
            self.assertEqual(command[command.index("--gpu-ids") + 1], "0,1,2")
            self.assertEqual(command[0], "/runner/python")


class StaticAndPortChecks(unittest.TestCase):
    def test_static_incomplete_model_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            actor, _ = service_plans_from_allocation(
                allocation(3),
                actor_port=8100,
                speculator_port=8101,
                run_directory=Path(directory),
            )

            def runner(command, **kwargs):
                report = {
                    "local_dir": str(actor.model_path),
                    "ok": False,
                    "issues": [{"code": "missing_indexed_shard"}],
                }
                return SimpleNamespace(
                    returncode=1,
                    stdout=json.dumps(report),
                    stderr="",
                )

            with self.assertRaises(StaticModelError) as context:
                verify_static_model(
                    actor,
                    runner=runner,
                    model_manager_path=Path(__file__),
                )
            self.assertIn("missing_indexed_shard", str(context.exception))

    def test_port_occupancy_is_rejected_and_sockets_are_closed(self) -> None:
        created = []

        class FakeSocket:
            def __init__(self):
                self.closed = False
                created.append(self)

            def bind(self, address):
                if address[1] == 8101:
                    raise OSError("busy")

            def close(self):
                self.closed = True

        with self.assertRaises(PortUnavailableError):
            assert_ports_available(
                (8100, 8101), socket_factory=lambda *args: FakeSocket()
            )
        self.assertTrue(all(item.closed for item in created))


class ServiceLifecycleTests(unittest.TestCase):
    def test_health_requester_uses_explicit_direct_opener(self) -> None:
        class FakeResponse:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'{"data": []}'

        class FakeOpener:
            def __init__(self):
                self.calls = []

            def open(self, request, timeout):
                self.calls.append((request.full_url, request.method, timeout))
                return FakeResponse()

        import scripts.service_manager as service_manager

        opener = FakeOpener()
        original = service_manager._DIRECT_HTTP_OPENER
        service_manager._DIRECT_HTTP_OPENER = opener
        try:
            response = _default_json_requester(
                "GET", "http://127.0.0.1:8100/v1/models", None, 3.0
            )
        finally:
            service_manager._DIRECT_HTTP_OPENER = original

        self.assertEqual(response, {"data": []})
        self.assertEqual(
            opener.calls,
            [("http://127.0.0.1:8100/v1/models", "GET", 3.0)],
        )
        self.assertFalse(
            any(getattr(handler, "proxies", {}) for handler in original.handlers)
        )

    def test_chess_process_bypasses_proxy_and_cannot_see_gpus(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            captured = {}

            def factory(command, **kwargs):
                captured.update(kwargs)
                return FakeProcess(444)

            handle = start_chess_process(
                ("runner-python", "run.py"),
                Path(directory) / "chess.log",
                process_factory=factory,
            )
            try:
                self.assertEqual(captured["env"]["CUDA_VISIBLE_DEVICES"], "")
                self.assertIn("127.0.0.1", captured["env"]["NO_PROXY"])
                self.assertEqual(
                    captured["env"]["NO_PROXY"], captured["env"]["no_proxy"]
                )
            finally:
                handle.log_handle.close()

    def test_start_service_uses_new_session_and_exact_python(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            actor, _ = service_plans_from_allocation(
                allocation(3),
                actor_port=8100,
                speculator_port=8101,
                run_directory=Path(directory),
            )
            captured = {}

            def factory(command, **kwargs):
                captured["command"] = command
                captured["kwargs"] = kwargs
                return FakeProcess(111)

            handle = start_service(
                actor,
                process_factory=factory,
                python_executable="/vllm/python",
                parent_environment={"PATH": "/bin"},
            )
            try:
                self.assertEqual(captured["command"][0], "/vllm/python")
                self.assertTrue(captured["kwargs"]["start_new_session"])
                self.assertEqual(
                    captured["kwargs"]["env"]["CUDA_VISIBLE_DEVICES"], "0,1"
                )
                self.assertEqual(handle.pid, 111)
            finally:
                handle.log_handle.close()

    def test_models_endpoint_plus_generation_marks_dynamic_ready(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            actor, _ = service_plans_from_allocation(
                allocation(3),
                actor_port=8100,
                speculator_port=8101,
                run_directory=Path(directory),
            )
            handle = managed("actor", 111)
            calls = []

            def requester(method, url, payload, timeout):
                calls.append((method, url, payload))
                if method == "GET":
                    return {"data": [{"id": actor.served_model_name}]}
                return {
                    "choices": [{"message": {"content": "READY"}}],
                    "usage": {"completion_tokens": 1, "total_tokens": 8},
                }

            ticks = iter((0.0, 0.1, 0.2))
            report = wait_until_ready(
                handle,
                actor,
                requester=requester,
                monotonic=lambda: next(ticks),
                sleeper=lambda _: None,
            )
            self.assertEqual(report["status"], "READY")
            self.assertEqual([call[0] for call in calls], ["GET", "POST"])
            self.assertEqual(
                calls[1][2]["chat_template_kwargs"], {"enable_thinking": False}
            )

    def test_termination_escalates_only_the_tracked_process_group(self) -> None:
        class SlowProcess(FakeProcess):
            def __init__(self):
                super().__init__(pid=222)
                self.wait_count = 0

            def wait(self, timeout=None):
                self.wait_count += 1
                if self.wait_count == 1:
                    raise subprocess.TimeoutExpired("fake", timeout)
                self.returncode = -signal.SIGKILL
                return self.returncode

        handle = ManagedProcess(
            "actor",
            SlowProcess(),
            ("command",),
            Path("actor.log"),
            io.StringIO(),
        )
        signals = []
        report = terminate_managed_process(
            handle,
            group_signaler=lambda pgid, sig: signals.append((pgid, sig)),
            terminate_timeout_seconds=0.01,
        )
        self.assertEqual(
            signals,
            [(222, signal.SIGTERM), (222, signal.SIGKILL)],
        )
        self.assertTrue(report["kill_sent"])
        self.assertIsNone(report["error"])

    def test_exited_parent_still_cleans_its_lingering_worker_group(self) -> None:
        handle = ManagedProcess(
            "actor",
            FakeProcess(pid=333, returncode=1, running=False),
            ("command",),
            Path("actor.log"),
            io.StringIO(),
        )
        signals = []
        group_states = iter((True, False, False))

        report = terminate_managed_process(
            handle,
            group_signaler=lambda pgid, sig: signals.append((pgid, sig)),
            group_checker=lambda pgid: next(group_states),
            terminate_timeout_seconds=0.01,
        )

        self.assertEqual(signals, [(333, signal.SIGTERM)])
        self.assertTrue(report["group_alive_initial"])
        self.assertFalse(report["group_alive_final"])
        self.assertIsNone(report["error"])


class FullLauncherTests(unittest.TestCase):
    def test_release_accepts_transient_utilization_after_process_exit(self) -> None:
        samples = tuple(
            GPUSample(
                sampled_at=f"sample-{index}",
                gpus=(
                    GPUInfo(
                        index=0,
                        uuid="GPU-0",
                        name="RTX 3090",
                        memory_used_mib=15,
                        memory_total_mib=24576,
                        utilization_percent=utilization,
                    ),
                ),
                compute_processes=(),
            )
            for index, utilization in enumerate((100, 0, 0))
        )

        report, released = _release_report(samples, (0,))

        self.assertTrue(released)
        self.assertTrue(report["confirmed_released"])
        self.assertTrue(report["selected_gpu_released"]["0"])
        self.assertFalse(report["selected_gpu_preflight_idle"]["0"])
        self.assertEqual(report["residual_max_utilization_percent"]["0"], 100)

    def _run(self, chess_returncode: int = 0, static_error: bool = False):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        entrypoint = root / "run_chess_local.py"
        entrypoint.write_text(
            "# --experiment-manifest\n# --topology-path\n", encoding="utf-8"
        )
        processes = []
        terminated_roles = []

        def service_starter(plan, **kwargs):
            handle = managed(plan.role, 100 + len(processes))
            processes.append(handle)
            return handle

        def readiness_waiter(handle, plan):
            return {"status": "READY", "role": plan.role}

        def static_verifier(plan):
            if static_error:
                raise StaticModelError("injected incomplete model")
            return {"ok": True, "alias": plan.model_alias}

        def chess_starter(command, log_path):
            handle = managed("chess", 200, chess_returncode)
            processes.append(handle)
            return handle

        def terminator(handles):
            terminated_roles.extend(handle.role for handle in reversed(handles))
            return [
                {"role": handle.role, "pid": handle.pid, "error": None}
                for handle in reversed(handles)
            ]

        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exit_code = main(
                [
                    "--gpu-count",
                    "3",
                    "--gpu-ids",
                    "0,1,2",
                    "--run-id",
                    "smoke-test",
                    "--output-root",
                    str(root / "runs"),
                    "--vllm-python",
                    "/usr/bin/python3",
                    "--runner-python",
                    "/usr/bin/python3",
                    "--stop-after",
                    "6",
                ],
                client=idle_gpu_client(),
                sleeper=lambda _: None,
                clock=lambda: "fixed",
                static_verifier=static_verifier,
                port_checker=lambda ports: None,
                service_starter=service_starter,
                readiness_waiter=readiness_waiter,
                chess_starter=chess_starter,
                process_terminator=terminator,
                chess_entrypoint=entrypoint,
            )
        payload = json.loads(stdout.getvalue())
        return temporary, exit_code, payload, terminated_roles, stderr.getvalue()

    def test_full_success_is_isolated_and_cleans_reverse_order(self) -> None:
        temporary, exit_code, payload, terminated_roles, stderr = self._run()
        try:
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["status"], "SUCCEEDED")
            self.assertEqual(terminated_roles, ["chess", "speculator", "actor"])
            self.assertTrue(payload["gpu_release_check"]["confirmed_idle"])
            self.assertIn("3gpu-ids-0-1-2-", payload["run_directory"])
            self.assertTrue(Path(payload["manifest_path"]).is_file())
            self.assertIn("--experiment-manifest", payload["chess_command"])
            self.assertIn("--topology-path", payload["chess_command"])
            self.assertTrue(Path(payload["runner_input_manifest_path"]).is_file())
            self.assertNotEqual(
                Path(payload["runner_input_manifest_path"]).read_bytes(),
                Path(payload["manifest_path"]).read_bytes(),
            )
            self.assertEqual(stderr, "")
        finally:
            temporary.cleanup()

    def test_chess_nonzero_return_code_is_propagated_and_cleaned(self) -> None:
        temporary, exit_code, payload, terminated_roles, _ = self._run(7)
        try:
            self.assertEqual(exit_code, 7)
            self.assertEqual(payload["status"], "CHESS_FAILED")
            self.assertEqual(payload["chess_returncode"], 7)
            self.assertEqual(terminated_roles, ["chess", "speculator", "actor"])
        finally:
            temporary.cleanup()

    def test_static_model_failure_starts_no_process_and_returns_three(self) -> None:
        temporary, exit_code, payload, terminated_roles, _ = self._run(
            static_error=True
        )
        try:
            self.assertEqual(exit_code, 3)
            self.assertEqual(payload["status"], "FAILED")
            self.assertEqual(payload["error"]["type"], "StaticModelError")
            self.assertEqual(terminated_roles, [])
            self.assertTrue(payload["gpu_release_check"]["confirmed_idle"])
        finally:
            temporary.cleanup()


if __name__ == "__main__":
    unittest.main()
