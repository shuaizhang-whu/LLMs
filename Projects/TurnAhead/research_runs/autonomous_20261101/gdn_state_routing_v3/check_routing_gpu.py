"""Regression for a zero-new-draft step following a multi-token acceptance."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "gdn_core_test_v3/code"))
import check_gpu as core
from metadata_reference import build_preserving_accepted_state
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import MambaSpec


def build_native_or_candidate(fn, lengths, draft_counts, accepted, slots, prefilling=None):
    starts = [0]
    for length in lengths:
        starts.append(starts[-1] + length)
    prefilling = prefilling if prefilling is not None else [False] * len(lengths)
    common = NS(query_start_loc=core.ints(starts),
        query_start_loc_cpu=torch.tensor(starts, dtype=torch.int32),
        seq_lens=core.ints([length if prefill else 100 + length for length, prefill in zip(lengths, prefilling)]),
        is_prefilling=torch.tensor(prefilling, dtype=torch.bool),
        block_table_tensor=core.ints(slots), max_query_len=max(lengths),
        num_reqs=len(lengths), num_actual_tokens=sum(lengths))
    common.compute_num_computed_tokens = lambda: common.seq_lens - (common.query_start_loc[1:] - common.query_start_loc[:-1])
    spec = MambaSpec(block_size=640, shapes=((23, 1024), (4, 128, 128)),
        dtypes=(torch.bfloat16, torch.float32), mamba_cache_mode="align",
        num_speculative_blocks=20, mamba_type=MambaAttentionBackendEnum.GDN_ATTN)
    builder = NS(use_spec_decode=True, num_spec=20, kv_cache_spec=spec,
        vllm_config=NS(cache_config=NS(mamba_cache_mode="align")), use_full_cuda_graph=False, gdn_prefill_backend="triton")
    from types import MethodType
    builder._build_chunk_metadata = MethodType(GDNAttentionMetadataBuilder._build_chunk_metadata, builder)
    counts = torch.tensor(draft_counts, dtype=torch.int32)
    original = counts.clone()
    metadata = fn(builder, 0, common, core.ints(accepted), counts)
    assert torch.equal(counts, original), "Routing must not mutate actual scheduler draft counts"
    assert metadata.num_actual_tokens == sum(lengths)
    return metadata


class Checks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(731)
        self.slots = list(range(core.BASE, core.BASE + 21))

    def test_routing_preserves_previous_acceptance_even_when_new_drafts_are_zero(self):
        old = build_native_or_candidate(GDNAttentionMetadataBuilder.build,
            [1], [-1], [5], [self.slots])
        fixed = build_native_or_candidate(build_preserving_accepted_state,
            [1], [-1], [5], [self.slots])
        self.assertIsNone(old.num_accepted_tokens)
        self.assertIsNone(old.spec_sequence_masks)
        self.assertEqual(fixed.num_accepted_tokens.tolist(), [5])
        self.assertEqual(fixed.num_spec_decode_tokens, 1)
        self.assertEqual(fixed.spec_state_indices_tensor.tolist(), [self.slots])

    def test_reachable_state_transition_changes_output_with_old_route_only(self):
        for accepted in [2, 5, 20, 21]:
            with self.subTest(accepted=accepted):
                initial = core.layer()
                data = core.inputs(initial, 29)
                state = core.clone_layer(initial)
                core.run(state, core.meta([7], [core.BASE], [False]), tuple(x[:7] for x in data))
                core.run(state, core.meta(spec_lengths=[21], spec_slots=[self.slots], accepted=[1]),
                         tuple(x[7:28] for x in data))
                # Entire sequence is inside one 640-token block: native align
                # postprocess does not copy/rebase this accepted state.
                running_tokens = 7 + 21 - 20
                new_computed = running_tokens + accepted - 1
                self.assertLess((new_computed // 640) * 640, running_tokens)
                old_meta = build_native_or_candidate(GDNAttentionMetadataBuilder.build,
                    [1], [-1], [accepted], [self.slots])
                fixed_meta = build_native_or_candidate(build_preserving_accepted_state,
                    [1], [-1], [accepted], [self.slots])
                old = core.run(core.clone_layer(state), old_meta, tuple(x[28:29] for x in data))
                fixed_state = core.clone_layer(state)
                fixed = core.run(fixed_state, fixed_meta, tuple(x[28:29] for x in data))
                reference = core.clone_layer(initial)
                expected = core.run(reference, core.meta([8 + accepted], [core.BASE], [False]),
                    tuple(torch.cat([x[:7 + accepted], x[28:29]]) for x in data))[-1:]
                self.assertFalse(torch.equal(old, expected), "Regression must expose the stale-state path")
                self.assertTrue(torch.equal(fixed, expected))
                self.assertTrue(torch.equal(fixed_state.kv_cache[1][core.BASE], reference.kv_cache[1][core.BASE]))

    def test_zero_draft_target_is_independent_of_another_requests_drafts(self):
        initial = core.layer()
        prefix, speculative, target, other = (core.inputs(initial, n) for n in [7, 21, 1, 21])
        base = core.clone_layer(initial)
        core.run(base, core.meta([7], [core.BASE], [False]), prefix)
        core.run(base, core.meta(spec_lengths=[21], spec_slots=[self.slots], accepted=[1]), speculative)
        base.kv_cache[1][32].zero_()
        target_meta = build_native_or_candidate(build_preserving_accepted_state,
            [1], [-1], [5], [self.slots])
        mixed_meta = build_native_or_candidate(build_preserving_accepted_state,
            [1, 21], [-1, 20], [5, 1], [self.slots, list(range(32, 53))])
        solo = core.run(core.clone_layer(base), target_meta, target)
        mixed = core.run(core.clone_layer(base), mixed_meta,
            tuple(torch.cat([a, b]) for a, b in zip(target, other)))
        self.assertTrue(torch.equal(solo, mixed[:1]))


    def test_stale_positive_hints_do_not_turn_real_prefill_into_verification(self):
        for length in [1, 73]:
            with self.subTest(length=length):
                m = build_native_or_candidate(build_preserving_accepted_state,
                    [length], [20], [1], [self.slots], prefilling=[True])
                self.assertEqual(m.num_prefills, 1)
                self.assertEqual(m.num_prefill_tokens, length)
                self.assertIsNone(m.spec_sequence_masks)
                self.assertEqual(m.has_initial_state.tolist(), [False])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a fresh output")
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    args.output.write_text(json.dumps({"tests": result.testsRun, "failures": len(result.failures),
        "errors": len(result.errors), "passed": result.wasSuccessful(),
        "scope": "Native metadata routing plus native GPU state arithmetic; full-model verification still required"}, indent=2) + "\n")
    raise SystemExit(not result.wasSuccessful())
