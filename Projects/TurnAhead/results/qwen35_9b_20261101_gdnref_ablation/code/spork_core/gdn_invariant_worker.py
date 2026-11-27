"""Opt-in, worker-local Qwen3.5 shared-recurrence integration for vLLM 0.29.

Based on the arithmetic approach in vllm-project/vllm PR #60033, extended
locally to the native speculative-state API. This is an experimental backend,
not an assertion of upstream support. Shared site-packages are never edited.
"""
from __future__ import annotations

import functools
import hashlib
from importlib.metadata import version
import os
from pathlib import Path

PATCH_VERSION = "qwen35_shared_recurrence_spec_v1"
CORE_SHA256 = "6abb1076341f6652a37ec38f9a6656d8c1de5392e606fa3453c9406b1db0ad08"
VENDOR_SHA256 = {
    "model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py": "cb819264c1e791177d866f484ca0579a7d57141926705a1f75dd4a79de4d0125",
    "model_executor/layers/mamba/ops/causal_conv1d.py": "044d005cfe59fd0818ed421274e04f3e8dd679b8cdb64fca9f4422f2643484b2",
    "third_party/flash_linear_attention/ops/fused_recurrent.py": "fe6f1311014809040497aa0a623e7fa97f4b3457cdc7ee1f1f8aed21f326f755",
    "third_party/flash_linear_attention/ops/fused_gdn_prefill_post_conv.py": "590717b2107abc90d72548de06b5d315229eb2775c94614f3f545df11cc2b5df",
    "third_party/flash_linear_attention/ops/layernorm_guard.py": "0ca5e46aab408b80d673561b91cf7bff7987c71d798d3bdacac75b7e29416f49",
    "v1/attention/backends/gdn_attn.py": "c65552d9aad86472544033d44ad8a872221a83e6b60ba9918cc049ab0c580c7c",
    "v1/worker/gpu_worker.py": "b0f39124f31ead3d2d20f1c6ac8696393c0ed6fed585ba8d31a4f7673e390407",
}
_STATE = {"version": PATCH_VERSION, "installed": False, "hook_registered": False,
          "prefill_calls": 0, "ordinary_decode_calls": 0,
          "speculative_calls": 0, "mixed_calls": 0, "actual_tokens": 0, "norm_calls": 0}
_NATIVE = _NORM = _BACKEND = _ORIGINAL_CORE = None


def enabled():
    return os.environ.get("SPORK_Q35_REFERENCE_GDN", "0") == "1"


def verify_sources():
    import importlib.util
    if version("vllm") != "0.29.0":
        raise RuntimeError("Reference GDN is pinned to reviewed vLLM 0.29.0 sources")
    base = Path(importlib.util.find_spec("vllm").origin).parent
    for rel, expected in VENDOR_SHA256.items():
        if hashlib.sha256((base / rel).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Unreviewed vLLM dependency for reference GDN: {rel}")
    core = Path(__file__).with_name("gdn_invariant_core.py")
    if hashlib.sha256(core.read_bytes()).hexdigest() != CORE_SHA256:
        raise RuntimeError("Reference GDN core differs from the GPU-tested source")


def validate_configuration(config, capability):
    expected_env = {"VLLM_BATCH_INVARIANT": "1", "VLLM_USE_V2_MODEL_RUNNER": "0",
        "VLLM_GDN_DECODE_KERNEL": "triton", "VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE": "0",
        "VLLM_SSM_CONV_STATE_LAYOUT": "SD"}
    if not enabled() or any(os.environ.get(k) != v for k, v in expected_env.items()):
        raise RuntimeError("Reference GDN requires the explicit deterministic eager profile")
    model = config.model_config
    text = getattr(model.hf_config, "text_config", None)
    if (getattr(model.hf_config, "model_type", None) != "qwen3_5" or text is None
            or (text.hidden_size, text.num_hidden_layers, text.linear_num_key_heads,
                text.linear_num_value_heads, text.linear_key_head_dim, text.linear_value_head_dim)
            != (4096, 32, 16, 32, 128, 128)):
        raise RuntimeError("Reference GDN profile is scoped to the reviewed Qwen3.5-9B dimensions")
    mode = config.compilation_config.mode
    spec = config.speculative_config
    if (tuple(capability) != (8, 6) or config.parallel_config.tensor_parallel_size != 2
            or not model.enforce_eager or getattr(mode, "value", mode) != 0
            or spec is None or spec.method != "ngram" or spec.num_speculative_tokens != 20
            or config.cache_config.mamba_ssm_cache_dtype != "float32"
            or config.cache_config.mamba_cache_mode != "align"
            or not config.cache_config.enable_prefix_caching):
        raise RuntimeError("Reference GDN requires SM86, TP2, eager mode, FP32 state, aligned prefix cache and ngram K=20")


def _fixed_rows(_rows, _device):
    _STATE["norm_calls"] += 1
    return 1


def _patched_core(self, mixed_qkv, b, a, core_attn_out):
    from .gdn_invariant_core import forward_core
    context = _NATIVE.get_forward_context()
    metadata = context.attn_metadata
    metadata = metadata.get(self.prefix) if isinstance(metadata, dict) else None
    if metadata is None:
        # Preserve the native allocation/warmup path; no real request is present.
        return _ORIGINAL_CORE(self, mixed_qkv, b, a, core_attn_out)
    import torch
    if self.kv_cache[1].dtype != torch.float32:
        raise RuntimeError("Untested GDN state dtype")
    _STATE["prefill_calls"] += int(metadata.num_prefills > 0)
    _STATE["ordinary_decode_calls"] += int(metadata.num_decodes > 0)
    _STATE["speculative_calls"] += int(metadata.spec_sequence_masks is not None)
    _STATE["mixed_calls"] += int(metadata.num_prefills > 0 and metadata.spec_sequence_masks is not None)
    _STATE["actual_tokens"] += int(metadata.num_actual_tokens)
    return forward_core(self, metadata, mixed_qkv, b, a, core_attn_out)


def _active():
    return (_STATE["installed"] and _NATIVE is not None
            and _NATIVE.QwenGatedDeltaNetAttention._forward_core is _patched_core
            and _NORM.calc_rows_per_block is _fixed_rows
            and _BACKEND.supports_batch_invariance.__func__ is _supports_batch_invariance)


def _supports_batch_invariance(cls):
    # Support is declared only together with the scoped arithmetic replacement.
    return enabled() and _active()


def install(config):
    global _NATIVE, _NORM, _BACKEND, _ORIGINAL_CORE
    verify_sources()
    import torch
    validate_configuration(config, torch.cuda.get_device_capability())
    if config.model_config.dtype != torch.bfloat16 or config.model_config.head_dtype != torch.bfloat16:
        raise RuntimeError("Reference GDN expects BF16 weights and the standard model-dtype head")
    if _STATE["installed"]:
        if not _active():
            raise RuntimeError("Reference GDN patch was replaced after installation")
        return
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as native
    from vllm.third_party.flash_linear_attention.ops import layernorm_guard
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionBackend
    _NATIVE, _NORM, _BACKEND = native, layernorm_guard, GDNAttentionBackend
    _ORIGINAL_CORE = native.QwenGatedDeltaNetAttention._forward_core
    native.QwenGatedDeltaNetAttention._forward_core = _patched_core
    layernorm_guard.calc_rows_per_block = _fixed_rows
    GDNAttentionBackend.supports_batch_invariance = classmethod(_supports_batch_invariance)
    _STATE.update(installed=True, device=torch.cuda.current_device(),
        rank=torch.distributed.get_rank(),
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), Path(__file__).with_name("gdn_invariant_core.py")]},
        vendor_source_sha256=dict(VENDOR_SHA256))


def register_worker_hook():
    if not enabled() or _STATE["hook_registered"]:
        return
    verify_sources()
    from vllm.v1.worker.gpu_worker import Worker
    original = Worker.load_model

    @functools.wraps(original)
    def load_with_reference_core(self, *args, **kwargs):
        # init_device has selected the TP rank's CUDA device by this point.
        install(self.vllm_config)
        return original(self, *args, **kwargs)

    Worker.load_model = load_with_reference_core
    _STATE["hook_registered"] = True


def runtime_status():
    return {**_STATE, "enabled": enabled(), "active": _active()}


def verify_runtime_status(status, expected_hashes, workers=2):
    rows = status.get("gdn_reference_workers", [])
    if len(rows) != workers or {r.get("rank") for r in rows} != set(range(workers)):
        raise RuntimeError("Reference GDN status is missing a TP rank")
    for row in rows:
        if (row.get("version") != PATCH_VERSION or not row.get("active")
                or not row.get("installed") or not row.get("enabled")):
            raise RuntimeError("Reference GDN arithmetic is not active on every worker")
        for name in ("gdn_invariant_core.py", "gdn_invariant_worker.py"):
            if row.get("source_sha256", {}).get(name) != expected_hashes["spork_core/" + name]:
                raise RuntimeError("Worker reference-GDN source differs from this experiment")
