from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from scripts.gpu_resources import (
    GPUInfo,
    GPUSample,
    GPUSelectionError,
    ComputeProcess,
    NvidiaSmiClient,
    build_allocation,
    collect_samples,
    parse_gpu_ids,
    select_gpus,
)
from scripts.run_local_experiment import main


def gpu(
    index: int,
    *,
    memory_used: int = 100,
    utilization: int = 0,
    name: str = "NVIDIA RTX 3090",
) -> GPUInfo:
    return GPUInfo(
        index=index,
        uuid=f"GPU-{index}",
        name=name,
        memory_used_mib=memory_used,
        memory_total_mib=24576,
        utilization_percent=utilization,
    )


def samples(
    gpu_count: int,
    *,
    memory_by_sample: dict[tuple[int, int], int] | None = None,
    utilization_by_sample: dict[tuple[int, int], int] | None = None,
    processes_by_sample: dict[int, tuple[ComputeProcess, ...]] | None = None,
) -> tuple[GPUSample, ...]:
    memory_by_sample = memory_by_sample or {}
    utilization_by_sample = utilization_by_sample or {}
    processes_by_sample = processes_by_sample or {}
    result = []
    for sample_index in range(3):
        result.append(
            GPUSample(
                sampled_at=f"t{sample_index}",
                gpus=tuple(
                    gpu(
                        index,
                        memory_used=memory_by_sample.get((sample_index, index), 100),
                        utilization=utilization_by_sample.get(
                            (sample_index, index), 0
                        ),
                    )
                    for index in range(gpu_count)
                ),
                compute_processes=processes_by_sample.get(sample_index, ()),
            )
        )
    return tuple(result)


class GPUSelectionTests(unittest.TestCase):
    def test_zero_or_one_idle_gpu_is_rejected(self) -> None:
        for gpu_count in (0, 1):
            with self.subTest(gpu_count=gpu_count):
                with self.assertRaises(GPUSelectionError) as context:
                    select_gpus(samples(gpu_count), gpu_count=2)
                self.assertEqual(context.exception.report["inventory"]["idle_gpu_count"], gpu_count)

    def test_auto_selects_two_three_and_four_cards(self) -> None:
        for requested_count in (2, 3, 4):
            with self.subTest(requested_count=requested_count):
                manifest = select_gpus(
                    samples(4),
                    gpu_count=requested_count,
                    generated_at="fixed",
                )
                self.assertEqual(
                    manifest["selection"]["gpu_ids"], list(range(requested_count))
                )
                self.assertFalse(manifest["execution"]["services_started"])
                self.assertFalse(manifest["execution"]["experiment_started"])

    def test_transient_utilization_makes_gpu_busy(self) -> None:
        manifest = select_gpus(
            samples(3, utilization_by_sample={(1, 0): 6}),
            gpu_count=2,
        )
        self.assertEqual(manifest["selection"]["gpu_ids"], [1, 2])
        evaluation = manifest["inventory"]["gpu_evaluations"][0]
        self.assertIn("utilization_above_limit", evaluation["reasons"])

    def test_transient_memory_makes_gpu_busy(self) -> None:
        manifest = select_gpus(
            samples(3, memory_by_sample={(2, 1): 1025}),
            gpu_count=2,
        )
        self.assertEqual(manifest["selection"]["gpu_ids"], [0, 2])

    def test_memory_and_utilization_limits_are_inclusive(self) -> None:
        manifest = select_gpus(
            samples(
                2,
                memory_by_sample={(1, 0): 1024},
                utilization_by_sample={(2, 1): 5},
            ),
            gpu_count=2,
        )
        self.assertEqual(manifest["selection"]["gpu_ids"], [0, 1])

    def test_any_compute_process_makes_gpu_busy_and_records_owner(self) -> None:
        process = ComputeProcess("GPU-1", 4321, "python", 500, "alice")
        manifest = select_gpus(
            samples(3, processes_by_sample={0: (process,)}),
            gpu_count=2,
        )
        self.assertEqual(manifest["selection"]["gpu_ids"], [0, 2])
        evaluation = manifest["inventory"]["gpu_evaluations"][1]
        self.assertIn("compute_process_present", evaluation["reasons"])
        self.assertEqual(evaluation["compute_processes"][0]["owner"], "alice")

    def test_explicit_gpu_order_is_preserved(self) -> None:
        manifest = select_gpus(
            samples(4),
            gpu_count=3,
            requested_gpu_ids=(3, 1, 2),
            speculator_model="qwen3.5-9b",
        )
        self.assertEqual(manifest["selection"]["gpu_ids"], [3, 1, 2])
        self.assertEqual(manifest["allocation"]["actor"]["gpu_ids"], [3, 1])
        self.assertEqual(manifest["allocation"]["speculator"]["gpu_ids"], [2])

    def test_explicit_busy_gpu_is_rejected_even_if_others_are_idle(self) -> None:
        with self.assertRaises(GPUSelectionError) as context:
            select_gpus(
                samples(4, utilization_by_sample={(0, 0): 99}),
                gpu_count=2,
                requested_gpu_ids=(0, 1),
            )
        self.assertIn("[0]", str(context.exception))

    def test_explicit_count_must_match(self) -> None:
        with self.assertRaises(GPUSelectionError):
            select_gpus(samples(4), gpu_count=3, requested_gpu_ids=(0, 1))
        with self.assertRaises(ValueError):
            select_gpus(samples(4), gpu_count=2, requested_gpu_ids=(0, 0))

    def test_allocation_rules(self) -> None:
        allocation2 = build_allocation((0, 1), "qwen3.5-4b")
        self.assertEqual(allocation2["actor"]["model_key"], "qwen3.5-9b")
        self.assertEqual(allocation2["speculator"]["model_key"], "qwen3.5-4b")
        self.assertTrue(allocation2["smoke_only"])

        allocation3 = build_allocation((0, 1, 2), "qwen3.5-9b")
        self.assertEqual(allocation3["actor"]["tensor_parallel_size"], 2)
        self.assertEqual(allocation3["speculator"]["tensor_parallel_size"], 1)
        self.assertFalse(allocation3["smoke_only"])

        allocation4 = build_allocation((0, 1, 2, 3), "qwen3.5-9b")
        self.assertEqual(allocation4["actor"]["tensor_parallel_size"], 2)
        self.assertEqual(allocation4["speculator"]["tensor_parallel_size"], 2)

        with self.assertRaises(ValueError):
            build_allocation((0, 1), "qwen3.5-9b")

    def test_gpu_id_parser_rejects_duplicates(self) -> None:
        self.assertEqual(parse_gpu_ids("3, 1,2"), (3, 1, 2))
        with self.assertRaises(ValueError):
            parse_gpu_ids("1,1")


class SamplingTests(unittest.TestCase):
    def test_collects_three_samples_and_waits_only_between_them(self) -> None:
        gpu_outputs = iter(
            ["0, GPU-0, NVIDIA RTX 3090, 100, 24576, 0\n"] * 3
        )
        calls: list[tuple[str, ...]] = []

        def runner(command):
            calls.append(tuple(command))
            if "--query-gpu=" in command[1]:
                return next(gpu_outputs)
            return ""

        sleeps: list[float] = []
        timestamps = iter(("t0", "t1", "t2"))
        client = NvidiaSmiClient(runner, owner_lookup=lambda pid: "test")
        observed = collect_samples(
            client,
            sleeper=sleeps.append,
            clock=lambda: next(timestamps),
            sample_interval_seconds=0.25,
        )

        self.assertEqual(len(observed), 3)
        self.assertEqual([item.sampled_at for item in observed], ["t0", "t1", "t2"])
        self.assertEqual(sleeps, [0.25, 0.25])
        self.assertEqual(len(calls), 6)


class CLITests(unittest.TestCase):
    def _client(self, gpu_count: int) -> NvidiaSmiClient:
        rows = "".join(
            f"{index}, GPU-{index}, NVIDIA RTX 3090, 100, 24576, 0\n"
            for index in range(gpu_count)
        )

        def runner(command):
            if "--query-gpu=" in command[1]:
                return rows
            return ""

        return NvidiaSmiClient(runner, owner_lookup=lambda pid: None)

    def test_cli_returns_two_when_fewer_than_two_are_idle(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exit_code = main(
                ["--gpu-count", "2"],
                client=self._client(1),
                sleeper=lambda _: None,
                clock=lambda: "fixed",
            )
        self.assertEqual(exit_code, 2)
        payload = json.loads(stderr.getvalue())
        self.assertEqual(payload["preflight"]["inventory"]["idle_gpu_count"], 1)

    def test_cli_can_write_manifest_without_starting_any_service(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "manifest.json"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--gpu-count",
                        "3",
                        "--preflight-only",
                        "--gpu-ids",
                        "2,0,1",
                        "--speculator-model",
                        "qwen3.5-9b",
                        "--manifest-output",
                        str(output_path),
                    ],
                    client=self._client(4),
                    sleeper=lambda _: None,
                    clock=lambda: "fixed",
                )
            self.assertEqual(exit_code, 0)
            stored = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(stored["selection"]["gpu_ids"], [2, 0, 1])
            self.assertFalse(stored["execution"]["services_started"])
            self.assertFalse(stored["execution"]["experiment_started"])


if __name__ == "__main__":
    unittest.main()
