"""CPU guards for the opt-in backend; GPU arithmetic is tested separately."""
import copy
import os
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from spork_core.gdn_invariant_worker import (
    PATCH_VERSION, validate_configuration, verify_runtime_status, verify_sources)


class Tests(unittest.TestCase):
    def setUp(self):
        self.env = {"SPORK_Q35_REFERENCE_GDN": "1", "VLLM_BATCH_INVARIANT": "1",
            "VLLM_USE_V2_MODEL_RUNNER": "0", "VLLM_GDN_DECODE_KERNEL": "triton",
            "VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE": "0", "VLLM_SSM_CONV_STATE_LAYOUT": "SD"}
        text = NS(hidden_size=4096, num_hidden_layers=32, linear_num_key_heads=16,
            linear_num_value_heads=32, linear_key_head_dim=128, linear_value_head_dim=128)
        self.config = NS(model_config=NS(hf_config=NS(model_type="qwen3_5", text_config=text),
            enforce_eager=True), compilation_config=NS(mode=0),
            parallel_config=NS(tensor_parallel_size=2),
            speculative_config=NS(method="ngram", num_speculative_tokens=20),
            cache_config=NS(mamba_ssm_cache_dtype="float32", mamba_cache_mode="align",
                            enable_prefix_caching=True))

    def test_reviewed_dependency_bytes_and_profile(self):
        verify_sources()
        with patch.dict(os.environ, self.env):
            validate_configuration(self.config, (8, 6))

    def test_other_execution_routes_are_rejected(self):
        with patch.dict(os.environ, self.env):
            for key, value in [("VLLM_BATCH_INVARIANT", "0"),
                               ("VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE", "1"),
                               ("VLLM_GDN_DECODE_KERNEL", "cuda")]:
                with self.subTest(key=key), patch.dict(os.environ, {key: value}):
                    with self.assertRaises(RuntimeError):
                        validate_configuration(self.config, (8, 6))
            with self.assertRaises(RuntimeError):
                validate_configuration(self.config, (9, 0))
            for obj, field, value in [("parallel_config", "tensor_parallel_size", 1),
                    ("speculative_config", "num_speculative_tokens", 1),
                    ("model_config", "enforce_eager", False),
                    ("compilation_config", "mode", 3),
                    ("cache_config", "enable_prefix_caching", False)]:
                config = copy.deepcopy(self.config)
                setattr(getattr(config, obj), field, value)
                with self.subTest(field=field), self.assertRaises(RuntimeError):
                    validate_configuration(config, (8, 6))

    def test_runtime_identity_requires_every_rank_and_exact_sources(self):
        hashes = {"spork_core/gdn_invariant_core.py": "core",
                  "spork_core/gdn_invariant_worker.py": "worker",
                  "spork_core/gdn_metadata_reference.py": "metadata"}
        rows = [{"rank": rank, "version": PATCH_VERSION, "installed": True,
                 "enabled": True, "active": True,
                 "source_sha256": {"gdn_invariant_core.py": "core",
                                   "gdn_invariant_worker.py": "worker",
                                   "gdn_metadata_reference.py": "metadata"}} for rank in [0, 1]]
        verify_runtime_status({"gdn_reference_workers": rows}, hashes)
        with self.assertRaises(RuntimeError):
            verify_runtime_status({"gdn_reference_workers": rows[:1]}, hashes)
        for field, value in [("active", False), ("rank", 0), ("version", "other")]:
            changed = copy.deepcopy(rows)
            changed[1][field] = value
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                verify_runtime_status({"gdn_reference_workers": changed}, hashes)
        changed = copy.deepcopy(rows)
        changed[1]["source_sha256"]["gdn_invariant_core.py"] = "other"
        with self.assertRaises(RuntimeError):
            verify_runtime_status({"gdn_reference_workers": changed}, hashes)

    def test_missing_current_drafts_still_pass_previous_runner_acceptance(self):
        import torch
        from spork_core import gdn_invariant_worker as module
        owner = NS(num_accepted_tokens=NS(gpu=torch.tensor([5, 2])))
        received = []
        def build(builder, prefix, metadata, accepted, drafts, fast):
            received.append(accepted)
            return accepted
        with patch.object(module, "_MODEL_RUNNER", owner), patch.object(module, "_METADATA_BUILD", build):
            value = module._build_with_runner_state(None, 0, NS(num_reqs=1))
            self.assertEqual(value.tolist(), [5])
            explicit = torch.tensor([7])
            module._build_with_runner_state(None, 0, NS(num_reqs=1), explicit)
            self.assertIs(received[-1], explicit)


if __name__ == "__main__":
    unittest.main()
