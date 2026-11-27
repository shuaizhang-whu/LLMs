#!/usr/bin/env python3
"""GPU arithmetic/state checks only; does not claim full-model invariance."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from invariant_core import forward_core
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as native
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

DEVICE = torch.device("cuda")
WIDTH, DIM, POOL = 4, 128, 64
BASE = 1  # vLLM reserves block zero as NULL_BLOCK_ID.


def ints(values):
    return torch.tensor(values, dtype=torch.int32, device=DEVICE)


def layer(heads=2, spec=20):
    hv = 2 * heads
    conv_dim = (2 * heads + hv) * DIM
    shape = ((POOL, conv_dim, WIDTH - 1 + spec) if native.is_conv_state_dim_first()
             else (POOL, WIDTH - 1 + spec, conv_dim))
    return SimpleNamespace(tp_size=1, num_k_heads=heads, num_v_heads=hv,
        head_k_dim=DIM, head_v_dim=DIM, activation="silu",
        A_log=torch.randn(hv, device=DEVICE) * .2,
        dt_bias=torch.randn(hv, device=DEVICE) * .1,
        conv1d=SimpleNamespace(weight=torch.randn(conv_dim, 1, WIDTH,
            dtype=torch.bfloat16, device=DEVICE) * .1, bias=None),
        kv_cache=(torch.zeros(shape, dtype=torch.bfloat16, device=DEVICE),
                  torch.zeros(POOL, hv, DIM, DIM, dtype=torch.float32, device=DEVICE)))


def clone_layer(value):
    result = copy.copy(value)
    result.kv_cache = tuple(t.clone() for t in value.kv_cache)
    return result


def inputs(value, count):
    h = value.num_k_heads
    return (torch.randn(count, 4 * h * DIM, dtype=torch.bfloat16, device=DEVICE) * .2,
            torch.randn(count, 2 * h, dtype=torch.bfloat16, device=DEVICE) * .2,
            torch.randn(count, 2 * h, dtype=torch.bfloat16, device=DEVICE) * .2)


def meta(non_lengths=(), non_slots=(), initial=(), *, decode=False,
         spec_lengths=(), spec_slots=(), accepted=()):
    if any(length > 0 and slot <= 0 for length, slot in zip(non_lengths, non_slots)):
        raise ValueError("A real query cannot use vLLM's null cache block")
    if any(length > 0 and min(slots) <= 0 for length, slots in zip(spec_lengths, spec_slots)):
        raise ValueError("A real speculative query cannot use a null cache block")
    ns, ss = sum(non_lengths), sum(spec_lengths)
    spec_count = len(spec_lengths)
    m = GDNAttentionMetadata(
        num_prefills=0 if decode else len(non_lengths),
        num_prefill_tokens=0 if decode else ns,
        num_decodes=len(non_lengths) if decode else 0,
        num_decode_tokens=ns if decode else 0,
        num_spec_decodes=spec_count, num_spec_decode_tokens=ss,
        num_actual_tokens=ns + ss)
    if non_lengths:
        starts = [0]
        for length in non_lengths:
            starts.append(starts[-1] + length)
        m.non_spec_query_start_loc = ints(starts)
        m.non_spec_state_indices_tensor = ints(non_slots)
        if not decode:
            m.has_initial_state = torch.tensor(initial, dtype=torch.bool, device=DEVICE)
            m.nums_dict, m.batch_ptr, m.token_chunk_offset_ptr = compute_causal_conv1d_metadata(
                torch.tensor(starts, dtype=torch.int32), device=DEVICE)
    if spec_count:
        starts = [0]
        for length in spec_lengths:
            starts.append(starts[-1] + length)
        m.spec_query_start_loc = ints(starts)
        m.spec_state_indices_tensor = ints(spec_slots)
        m.spec_sequence_masks = torch.tensor(
            [True] * spec_count + [False] * len(non_lengths), device=DEVICE)
        m.spec_token_indx = torch.arange(ss, device=DEVICE, dtype=torch.int32)
        m.non_spec_token_indx = torch.arange(ss, ss + ns, device=DEVICE, dtype=torch.int32)
        m.num_accepted_tokens = ints(accepted)
    return m


def run(value, metadata, data):
    qkv, b, a = (x.clone() for x in data)
    output = torch.zeros(metadata.num_actual_tokens, value.num_v_heads, DIM,
                         dtype=torch.bfloat16, device=DEVICE)
    forward_core(value, metadata, qkv, b, a, output)
    torch.cuda.synchronize()
    if not torch.isfinite(output).all() or (output.numel() and not output.abs().max() > 0):
        raise AssertionError("Non-finite or vacuous all-zero output on nonzero random inputs")
    return output


class Checks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(731)

    def exact(self, left, right, label):
        if not torch.equal(left, right):
            diff = (left.float() - right.float()).abs()
            self.fail(f"{label}: unequal={int((left != right).sum())}, max_abs={float(diff.max())}")

    def test_prefill_partitions_and_state_reload(self):
        for heads in [2, 8]:  # 8/16 heads are the local TP2 shape of Qwen3.5-9B.
            with self.subTest(heads=heads):
                initial = layer(heads)
                data = inputs(initial, 257)
                whole = clone_layer(initial)
                expected = run(whole, meta([257], [BASE], [False]), data)
                for partitions in [[73, 56, 128], [1, 63, 64, 65, 64]]:
                    split = clone_layer(initial)
                    outputs, offset = [], 0
                    for length in partitions:
                        outputs.append(run(split, meta([length], [BASE], [offset > 0]),
                                           tuple(x[offset:offset + length] for x in data)))
                        split = clone_layer(split)  # Exercise persistent-state reload.
                        offset += length
                    self.exact(torch.cat(outputs), expected, "partition outputs")
                    self.exact(split.kv_cache[0], whole.kv_cache[0], "convolution state")
                    self.exact(split.kv_cache[1], whole.kv_cache[1], "recurrent state")

    def test_prefill_batch_composition(self):
        initial = layer()
        target, background = inputs(initial, 73), inputs(initial, 129)
        solo, mixed = clone_layer(initial), clone_layer(initial)
        expected = run(solo, meta([73], [BASE], [False]), target)
        observed = run(mixed, meta([73, 129], [BASE, 32], [False, False]),
                       tuple(torch.cat([a, b]) for a, b in zip(target, background)))
        self.exact(observed[:73], expected, "mixed prefill output")
        for left, right in zip(solo.kv_cache, mixed.kv_cache):
            self.exact(left[BASE], right[BASE], "mixed prefill target state")

    def test_matches_independent_float32_delta_rule_reference(self):
        value = layer()
        data = inputs(value, 17)
        observed = run(value, meta([17], [BASE], [False]), data)
        x, b, a = (t.float().cpu() for t in data)
        weights = value.conv1d.weight.float().cpu().squeeze(1)
        history = torch.zeros(WIDTH - 1, x.shape[1])
        state = torch.zeros(value.num_v_heads, DIM, DIM)
        outputs = []
        group = value.num_v_heads // value.num_k_heads
        for i in range(len(x)):
            window = torch.cat([history, x[i:i+1]])
            conv = torch.nn.functional.silu((window.T * weights).sum(-1))
            conv = conv.to(torch.bfloat16).float()
            history = window[1:]
            key_width = value.num_k_heads * DIM
            q, k, v = conv.split([key_width, key_width, value.num_v_heads * DIM])
            q, k = q.view(-1, DIM), k.view(-1, DIM)
            q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)
            k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
            q, k = q.repeat_interleave(group, 0), k.repeat_interleave(group, 0)
            v = v.view(-1, DIM)
            g = -value.A_log.cpu().exp() * torch.nn.functional.softplus(a[i] + value.dt_bias.cpu())
            beta = b[i].sigmoid()
            state = state * g.exp()[:, None, None]
            delta = (v - torch.einsum("hvk,hk->hv", state, k)) * beta[:, None]
            state = state + delta[:, :, None] * k[:, None, :]
            outputs.append(torch.einsum("hvk,hk->hv", state, q) * (DIM ** -.5))
        expected = torch.stack(outputs)
        self.assertGreater(float(expected.abs().max()), 1e-5)
        torch.testing.assert_close(observed.float().cpu(), expected, rtol=.03, atol=3e-6)
        torch.testing.assert_close(value.kv_cache[1][BASE].cpu(), state, rtol=.03, atol=3e-6)

    def test_ordinary_single_token_decode_matches_prefill(self):
        # Non-spec models have a width-1 convolution cache, not draft history.
        initial = layer(spec=0)
        data = inputs(initial, 17)
        whole, decoded = clone_layer(initial), clone_layer(initial)
        expected = run(whole, meta([17], [BASE], [False]), data)
        observed = [run(decoded, meta([1], [BASE], [], decode=True),
                        tuple(x[i:i+1] for x in data)) for i in range(17)]
        self.exact(torch.cat(observed), expected, "ordinary decode output")
        for a, b in zip(whole.kv_cache, decoded.kv_cache):
            self.exact(a, b, "ordinary decode state")

    def test_speculative_outputs_match_prefill_and_reject_suffix_state(self):
        initial = layer()
        data = inputs(initial, 29)  # Seven-token prefix, 21 verifier inputs, one continuation.
        for accepted in [1, 2, 5, 20, 21]:
            with self.subTest(accepted=accepted):
                spec = clone_layer(initial)
                run(spec, meta([7], [BASE], [False]), tuple(x[:7] for x in data))
                output = run(spec, meta(spec_lengths=[21], spec_slots=[list(range(BASE, BASE + 21))], accepted=[1]),
                             tuple(x[7:28] for x in data))
                full = clone_layer(initial)
                expected = run(full, meta([28], [BASE], [False]), tuple(x[:28] for x in data))
                self.exact(output, expected[7:], "verification outputs")
                self.exact(spec.kv_cache[1][BASE + 20], full.kv_cache[1][BASE], "all-accepted state")
                # The next verifier selects only the accepted prefix of prior state.
                actual_next = run(spec, meta(spec_lengths=[1], spec_slots=[list(range(BASE, BASE + 21))],
                    accepted=[accepted]), tuple(x[28:29] for x in data))
                reference = clone_layer(initial)
                canonical = tuple(torch.cat([x[:7+accepted], x[28:29]]) for x in data)
                reference_out = run(reference, meta([8+accepted], [BASE], [False]), canonical)
                self.exact(actual_next, reference_out[-1:], "rejected-suffix continuation")
                self.exact(spec.kv_cache[1][BASE], reference.kv_cache[1][BASE], "accepted-state recovery")

    def test_mixed_speculative_decode_and_prefill(self):
        initial = layer()
        prefix, target, background = inputs(initial, 7), inputs(initial, 21), inputs(initial, 73)
        solo, mixed = clone_layer(initial), clone_layer(initial)
        for value in [solo, mixed]:
            run(value, meta([7], [BASE], [False]), prefix)
        expected = run(solo, meta(spec_lengths=[21], spec_slots=[list(range(BASE, BASE + 21))], accepted=[1]), target)
        observed = run(mixed, meta([73], [32], [False],
            spec_lengths=[21], spec_slots=[list(range(BASE, BASE + 21))], accepted=[1]),
            tuple(torch.cat([a, b]) for a, b in zip(target, background)))
        self.exact(observed[:21], expected, "mixed verification output")
        self.exact(solo.kv_cache[1][BASE:BASE + 21], mixed.kv_cache[1][BASE:BASE + 21], "mixed verifier states")

    def test_empty_padding_rows_preserve_null_cache_block(self):
        initial = layer()
        initial.kv_cache[0][0].fill_(3.5)
        initial.kv_cache[1][0].fill_(7.25)
        data = inputs(initial, 73)
        solo, padded = clone_layer(initial), clone_layer(initial)
        expected = run(solo, meta([73], [BASE], [False]), data)
        observed = run(padded, meta([73, 0], [BASE, 0], [False, False]), data)
        self.exact(observed, expected, "padded prefill")
        for actual, old in zip(padded.kv_cache, initial.kv_cache):
            self.exact(actual[0], old[0], "null block after prefill")
        target = inputs(initial, 21)
        expected = run(solo, meta(spec_lengths=[21],
            spec_slots=[list(range(BASE, BASE + 21))], accepted=[1]), target)
        observed = run(padded, meta(spec_lengths=[21, 0],
            spec_slots=[list(range(BASE, BASE + 21)), [0] * 21], accepted=[1, 1]), target)
        self.exact(observed, expected, "padded verifier")
        for actual, old in zip(padded.kv_cache, initial.kv_cache):
            self.exact(actual[0], old[0], "null block after verifier")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("GPU diagnostic outputs are immutable")
    if torch.cuda.get_device_capability() != (8, 6):
        raise ValueError("This experimental validation is scoped to the local SM86 device")
    started = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    args.output.write_text(json.dumps({"tests": result.testsRun, "errors": len(result.errors),
        "failures": len(result.failures), "passed": result.wasSuccessful(),
        "seconds_including_compilation": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "scope": "GDN core arithmetic and cache addressing only; no model-level invariance or speed claim"}, indent=2) + "\n")
    raise SystemExit(not result.wasSuccessful())


if __name__ == "__main__":
    main()
