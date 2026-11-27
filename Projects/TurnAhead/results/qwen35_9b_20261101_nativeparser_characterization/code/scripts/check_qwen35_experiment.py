#!/usr/bin/env python3
"""CPU metadata checks, read-only server checks, and sealed result gate."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_endpoint(path):
    port = int(os.environ.get("SPORK_CHECK_PORT", "8027"))
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"http://127.0.0.1:{port}{path}", timeout=30) as response:
        return json.load(response)


def local_check():
    from transformers import AutoTokenizer
    from spork_core.benchmarks.tau2 import load_tools
    from spork_core.qwen35_native import NativeQwen35Adapter
    from spork_core import paths
    import tau2
    model = Path(os.environ.get("SPORK_CODEX_MODEL", ROOT.parents[1] / "Models/Qwen3.5-9B"))
    config = json.loads((model / "config.json").read_text())
    if config.get("model_type") != "qwen3_5":
        raise ValueError("This profile requires Qwen3.5 dense weights")
    index = json.loads((model / "model.safetensors.index.json").read_text())
    missing = [f for f in set(index["weight_map"].values()) if not (model / f).is_file()]
    if missing:
        raise ValueError(f"Missing weight shards: {missing}")
    if not Path(tau2.__file__).resolve().is_relative_to(ROOT) or not paths.TAU2_ROOT.resolve().is_relative_to(ROOT):
        raise ValueError("tau2 code must be imported from spork-codex")
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    adapter = NativeQwen35Adapter(tokenizer, load_tools("retail"))
    call = {"name": "get_order_details", "arguments": {"order_id": "#W0001234"}}
    text = adapter.speculative_tool_message(call)
    if adapter.parse_main_tool_call(text) != call:
        raise ValueError("Native tool round trip failed")
    return {"model": str(model), "tool_call_open_token_id": adapter.tool_open_token_id,
            "history_policy": adapter.history_policy, "weight_shards_present": len(set(index["weight_map"].values())),
            "full_weight_checksums_verified": False, "generation_config_present": (model / "generation_config.json").exists(),
            "generation_requests": 0}


def server_check():
    from spork_core.evaluation_artifacts import digest
    local = local_check()
    status = read_endpoint("/spork/status")
    config = status.get("server_config", {})
    if (config.get("project_root") != str(ROOT)
            or config.get("experiment_profile") != "qwen35_9b_successor_v1"
            or config.get("tool_call_open_token_id") != local["tool_call_open_token_id"]):
        raise ValueError("Server belongs to a different checkout/model profile")
    for rel in ("launch_vllm_spork_http.py", "spork_core/vllm029_d3.py"):
        if config.get("startup_source_sha256", {}).get(rel) != digest(ROOT / rel):
            raise ValueError(f"Server source changed: restart it ({rel})")
    expected_serving = {"visible_devices": os.environ.get("SPORK_Q35_DEVICES", "2,3"),
        "tensor_parallel_size": os.environ.get("SPORK_Q35_TP", "2"),
        "max_model_len": "40960", "max_num_seqs": "4",
        "max_num_batched_tokens": os.environ.get("SPORK_Q35_BATCH_TOKENS", "4096"),
        "block_size": os.environ.get("SPORK_Q35_BLOCK_SIZE", "640"),
        "hf_overrides": json.dumps({"head_dtype": os.environ.get("SPORK_Q35_HEAD_DTYPE", "model")}, separators=(",", ":"))}
    if config.get("startup_serving") != expected_serving:
        raise ValueError("Serving devices/TP/batch configuration differs from this profile")
    expected_gdn = {"decode_kernel": os.environ.get("VLLM_GDN_DECODE_KERNEL", "cuda"),
        "packed_recurrent_decode": os.environ.get("VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE", "1")}
    if config.get("gdn_execution") != expected_gdn:
        raise ValueError("GDN kernel configuration differs from this profile")
    models = read_endpoint("/v1/models")["data"]
    if not any(x["id"] == local["model"] and int(x.get("max_model_len", 0)) == 40960 for x in models):
        raise ValueError("Model identity/context window differs from the experiment")
    if (config.get("batch_invariant_requested") is not False or not config.get("prefix_cache_reset_available")
            or not status.get("has_drafter") or status.get("active_requests") != 0
            or status.get("pending_verifications") != 0):
        raise ValueError("A clean dedicated GDN-compatible D3 server (batch-invariant off) is required")
    return {**local, "server_profile": config["experiment_profile"], "server_state_checked": True}


def results_check(output, stage):
    from spork_core.evaluation_artifacts import digest, planned_records, write_json
    from scripts.run_successor_full_eval import analyze, require_matching_trajectories
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["model_family"] != "qwen35" or not manifest["require_identical_trajectories"]:
        raise ValueError("Wrong experiment family or disabled trajectory gate")
    for rel, expected in manifest["code_sha256"].items():
        if digest(ROOT / rel) != expected or digest(output / "code" / rel) != expected:
            raise ValueError(f"Frozen implementation changed: {rel}")
    for source, expected in manifest["input_sha256"].items():
        if digest(source) != expected:
            raise ValueError(f"Frozen input changed: {source}")
    for rel, expected in manifest.get("external_parser_source_sha256", {}).items():
        if (digest(Path(manifest["external_parser_root"]) / rel) != expected
                or digest(output / "external_code" / "vllm" / rel) != expected):
            raise ValueError(f"Frozen external parser changed: {rel}")
    for name, expected in manifest["model_config_sha256"].items():
        if digest(Path(manifest["model"]) / name) != expected:
            raise ValueError(f"Model metadata changed: {name}")
    for name in manifest["absent_optional_model_files"]:
        if (Path(manifest["model"]) / name).exists():
            raise ValueError(f"Previously absent model metadata appeared: {name}")
    records, unsealed = planned_records(output, manifest)
    if len(records) != len(manifest["schedule"]) or unsealed:
        raise ValueError(f"Run incomplete: {len(records)}/{len(manifest['schedule'])}")
    report = analyze(output, manifest)
    require_matching_trajectories(report, manifest)
    failures = [r for r in records if r["errors"]]
    if stage == "preflight" and failures:
        raise ValueError(f"Preflight has {len(failures)} failed records; inspect them before proceeding")
    if any(r["d3_counters"].get(k, 0) for r in records
           for k in ("accounting_errors", "dropped_injections")):
        raise ValueError("D3 accounting/cleanup error")
    raw = [r for r in records if r["config"] == "raw" and not r["errors"]]
    early = sum(p.get("child_launch_phase") == "before_parent_completion"
                for r in raw for p in r["prefetch"]["pairs"])
    adopted = sum(r["prefetch"].get("adopted", 0) for r in raw)
    verified = sum(r["d3_counters"].get("verified_injections", 0) for r in records)
    if stage == "preflight" and (not early or not adopted or not verified):
        raise ValueError(f"Insufficient mechanism coverage: early={early}, adopted={adopted}, D3={verified}")
    repeated = {}
    for r in records:
        if r["errors"]:
            continue
        signature = ([t["main_token_ids"] for t in r["turns"]],
                     [t.get("environment_after") for t in r["turns"]])
        key = (r["group_id"], r["config"])
        if key in repeated and repeated[key] != signature:
            raise ValueError(f"Same-seed repeated trajectory changed: {key}")
        repeated[key] = signature
    result = {"passed": True, "stage": stage, "recorded": len(records),
              "failed_records": len(failures), "raw_early_children": early,
              "raw_adopted_children": adopted, "d3_verified_injections": verified,
              "status": "compatibility evidence only; not a positive speedup assertion",
              "complete_interactive_tau2": False}
    write_json(output / "stage_check.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--local", action="store_true")
    group.add_argument("--server", action="store_true")
    group.add_argument("--results", type=Path)
    p.add_argument("--stage", choices=["preflight", "ablation"], default="preflight")
    args = p.parse_args()
    result = (local_check() if args.local else server_check() if args.server
              else results_check(args.results, args.stage))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
