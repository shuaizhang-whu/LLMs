# SPDX-License-Identifier: Apache-2.0
# Shared-recurrence approach informed by vllm-project/vllm PR #60033.
# This local experiment additionally routes speculative verification through
# the same existing recurrent operator. It is NOT installed into the server.
"""Experimental GDN core for GPU state/partition checks, pinned to vLLM 0.29."""
import torch


def recurrent(layer, conv_output, a, b, state, indices, starts, *,
              has_initial_state=None, accepted=None):
    from vllm.third_party.flash_linear_attention.ops import (
        fused_post_conv_prep, fused_recurrent_gated_delta_rule)

    if has_initial_state is not None:
        if indices.ndim != 1:
            raise ValueError("Initial-state mask is for the non-speculative rows")
        valid = indices >= 0
        selected = indices[valid]
        old = state[selected]
        state[selected] = torch.where(has_initial_state[valid, None, None, None], old, 0)
    q, k, v, g, beta = fused_post_conv_prep(
        conv_output=conv_output, a=a, b=b, A_log=layer.A_log,
        dt_bias=layer.dt_bias, num_k_heads=layer.num_k_heads // layer.tp_size,
        head_k_dim=layer.head_k_dim, head_v_dim=layer.head_v_dim,
        apply_l2norm=False, output_g_exp=False)
    if indices.ndim == 1:
        # Repeated writes to one request-owned slot retain its final state.
        # Speculative rows instead retain every verifier position separately.
        indices = indices[:, None].expand(-1, max(1, conv_output.shape[0])).contiguous()
    output, _ = fused_recurrent_gated_delta_rule(
        q=q.unsqueeze(0), k=k.unsqueeze(0), v=v.unsqueeze(0),
        g=g.unsqueeze(0), beta=beta.unsqueeze(0), initial_state=state,
        inplace_final_state=True, cu_seqlens=starts,
        ssm_state_indices=indices, num_accepted_tokens=accepted,
        use_qk_l2norm_in_kernel=True)
    return output.squeeze(0)


def forward_core(layer, meta, mixed_qkv, b, a, core_attn_out):
    """Keep native convolution/cache addressing, share recurrence in all routes.

    This function operates only on supplied buffers and native metadata. The
    caller owns isolation and must validate the real model integration separately.
    """
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as native

    count = meta.num_actual_tokens
    mixed_qkv, a, b = mixed_qkv[:count], a[:count], b[:count]
    conv_state, state = layer.kv_cache[:2]
    if not native.is_conv_state_dim_first():
        conv_state = conv_state.transpose(-1, -2)
    weights = layer.conv1d.weight.view(layer.conv1d.weight.shape[0], -1)
    has_spec = meta.spec_sequence_masks is not None
    has_non_spec = meta.num_prefills > 0 or meta.num_decodes > 0

    if has_spec:
        if not has_non_spec:
            spec_qkv, spec_a, spec_b = mixed_qkv, a, b
        else:
            spec_qkv = mixed_qkv.index_select(0, meta.spec_token_indx)
            spec_a = a.index_select(0, meta.spec_token_indx)
            spec_b = b.index_select(0, meta.spec_token_indx)
        indices = meta.spec_state_indices_tensor
        starts = meta.spec_query_start_loc[:meta.num_spec_decodes + 1]
        conv_spec = native.causal_conv1d_update(
            spec_qkv, conv_state, weights, layer.conv1d.bias, layer.activation,
            conv_state_indices=indices[:, 0][:meta.num_spec_decodes],
            num_accepted_tokens=meta.num_accepted_tokens,
            query_start_loc=meta.spec_query_start_loc,
            max_query_len=indices.shape[-1], validate_data=False)
        spec_output = recurrent(layer, conv_spec, spec_a, spec_b, state,
            indices, starts, accepted=meta.num_accepted_tokens)
        if has_non_spec:
            core_attn_out.index_copy_(0, meta.spec_token_indx.long(), spec_output)
        else:
            core_attn_out[:count] = spec_output

    if has_non_spec:
        if has_spec:
            non_qkv = mixed_qkv.index_select(0, meta.non_spec_token_indx)
            non_a = a.index_select(0, meta.non_spec_token_indx)
            non_b = b.index_select(0, meta.non_spec_token_indx)
        else:
            non_qkv, non_a, non_b = mixed_qkv, a, b
        indices = meta.non_spec_state_indices_tensor
        starts = meta.non_spec_query_start_loc
        if meta.num_prefills:
            conv_non = native.causal_conv1d_fn(
                non_qkv.transpose(0, 1), weights, layer.conv1d.bias,
                activation=layer.activation, conv_states=conv_state,
                has_initial_state=meta.has_initial_state,
                cache_indices=indices, query_start_loc=starts,
                metadata=meta).transpose(0, 1)
        else:
            conv_non = native.causal_conv1d_update(
                non_qkv, conv_state, weights, layer.conv1d.bias, layer.activation,
                conv_state_indices=indices[:meta.num_actual_tokens],
                validate_data=True)
        non_output = recurrent(layer, conv_non, non_a, non_b, state, indices,
            starts, has_initial_state=meta.has_initial_state)
        if has_spec:
            core_attn_out.index_copy_(0, meta.non_spec_token_indx.long(), non_output)
        else:
            core_attn_out[:count] = non_output
