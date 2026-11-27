# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Derived from vLLM 0.29.0 GDNAttentionMetadataBuilder.build.
# Use authoritative per-step prefill/decode flags rather than draft-count hints,
# which can be negative or stale when no drafts are scheduled. All decoder rows
# retain the previous accepted-state offset; no new input tokens are created.
from __future__ import annotations
import torch
from vllm.v1.attention.backends.gdn_attn import (
    CommonAttentionMetadata, GDNAttentionMetadata, mamba_get_block_table_tensor,
    async_tensor_h2d, split_decodes_and_prefills, compute_causal_conv1d_metadata,
    NULL_BLOCK_ID,
)

def build_preserving_accepted_state(self, common_prefix_len: int, common_attn_metadata: CommonAttentionMetadata, num_accepted_tokens: torch.Tensor | None=None, num_decode_draft_tokens_cpu: torch.Tensor | None=None, fast_build: bool=False) -> GDNAttentionMetadata:
    m = common_attn_metadata
    query_start_loc = m.query_start_loc
    query_start_loc_cpu = m.query_start_loc_cpu
    (nums_dict, batch_ptr, token_chunk_offset_ptr) = (None, None, None)
    block_table_tensor = mamba_get_block_table_tensor(m.block_table_tensor, m.seq_lens, self.kv_cache_spec, self.vllm_config.cache_config.mamba_cache_mode)
    spec_sequence_masks_cpu: torch.Tensor | None = None
    if not self.use_spec_decode:
        spec_sequence_masks = None
        num_spec_decodes = 0
    else:
        if m.is_prefilling is None or m.is_prefilling.device.type != 'cpu':
            raise RuntimeError('Reference GDN requires authoritative CPU prefill/decode flags')
        spec_sequence_masks_cpu = ~m.is_prefilling & (query_start_loc_cpu[1:] - query_start_loc_cpu[:-1] > 0)
        num_spec_decodes = spec_sequence_masks_cpu.sum().item()
        if num_spec_decodes == 0:
            num_spec_decodes = 0
            spec_sequence_masks = None
            spec_sequence_masks_cpu = None
        else:
            spec_sequence_masks = async_tensor_h2d(spec_sequence_masks_cpu, device=query_start_loc.device)
    if spec_sequence_masks is None:
        (num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens) = split_decodes_and_prefills(m, decode_threshold=1, treat_short_extends_as_decodes=False)
        num_spec_decode_tokens = 0
        spec_token_indx = None
        non_spec_token_indx = None
        spec_state_indices_tensor = None
        non_spec_state_indices_tensor = block_table_tensor[:, 0]
        spec_query_start_loc = None
        non_spec_query_start_loc = query_start_loc
        non_spec_query_start_loc_cpu = query_start_loc_cpu
        num_accepted_tokens = None
    else:
        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        assert spec_sequence_masks_cpu is not None
        non_spec_sequence_masks_cpu = ~spec_sequence_masks_cpu
        query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        non_spec_query_lens_cpu = query_lens_cpu[non_spec_sequence_masks_cpu]
        num_decodes = (non_spec_query_lens_cpu == 1).sum().item()
        num_zero_len = (non_spec_query_lens_cpu == 0).sum().item()
        num_prefills = non_spec_query_lens_cpu.size(0) - num_decodes - num_zero_len
        num_decode_tokens = num_decodes
        num_prefill_tokens = non_spec_query_lens_cpu.sum().item() - num_decode_tokens
        num_spec_decode_tokens = query_lens_cpu.sum().item() - num_prefill_tokens - num_decode_tokens
        if num_decodes > 0 and num_spec_decodes > 0:
            num_prefills += num_decodes
            num_prefill_tokens += num_decode_tokens
            num_decodes = 0
            num_decode_tokens = 0
        if num_prefills == 0 and num_decodes == 0:
            spec_token_size = min(num_spec_decodes * (self.num_spec + 1), query_start_loc_cpu[-1].item())
            spec_token_indx = torch.arange(spec_token_size, dtype=torch.int32, device=query_start_loc.device)
            non_spec_token_indx = torch.empty(0, dtype=torch.int32, device=query_start_loc.device)
            spec_state_indices_tensor = block_table_tensor[spec_sequence_masks_cpu, :self.num_spec + 1]
            non_spec_state_indices_tensor = None
            spec_query_start_loc = query_start_loc[:num_spec_decodes + 1]
            non_spec_query_start_loc = None
            non_spec_query_start_loc_cpu = None
        else:
            spec_token_masks = torch.repeat_interleave(spec_sequence_masks, query_lens, output_size=query_start_loc_cpu[-1].item())
            index = torch.argsort(spec_token_masks, stable=True)
            num_non_spec_tokens = num_prefill_tokens + num_decode_tokens
            non_spec_token_indx = index[:num_non_spec_tokens]
            spec_token_indx = index[num_non_spec_tokens:]
            spec_state_indices_tensor = block_table_tensor[spec_sequence_masks_cpu, :self.num_spec + 1]
            non_spec_state_indices_tensor = block_table_tensor[non_spec_sequence_masks_cpu, 0]
            spec_query_start_loc = torch.zeros(num_spec_decodes + 1, dtype=torch.int32, device=query_start_loc.device)
            torch.cumsum(query_lens[spec_sequence_masks_cpu], dim=0, out=spec_query_start_loc[1:])
            non_spec_query_start_loc = torch.zeros(query_lens.size(0) - num_spec_decodes + 1, dtype=torch.int32, device=query_start_loc.device)
            torch.cumsum(query_lens[non_spec_sequence_masks_cpu], dim=0, out=non_spec_query_start_loc[1:])
            non_spec_query_start_loc_cpu = torch.zeros(query_lens_cpu.size(0) - num_spec_decodes + 1, dtype=torch.int32)
            torch.cumsum(query_lens_cpu[non_spec_sequence_masks_cpu], dim=0, out=non_spec_query_start_loc_cpu[1:])
        assert num_accepted_tokens is not None
        num_accepted_tokens = num_accepted_tokens[spec_sequence_masks_cpu]
    chunk_indices: torch.Tensor | None = None
    chunk_offsets: torch.Tensor | None = None
    prefill_query_start_loc: torch.Tensor | None = None
    prefill_state_indices: torch.Tensor | None = None
    prefill_has_initial_state: torch.Tensor | None = None
    if num_prefills > 0:
        if spec_sequence_masks is None and num_decodes > 0:
            assert non_spec_query_start_loc is not None
            assert non_spec_query_start_loc_cpu is not None
            assert non_spec_state_indices_tensor is not None
            prefill_query_start_loc = non_spec_query_start_loc[num_decodes:] - num_decode_tokens
            prefill_query_start_loc_cpu = non_spec_query_start_loc_cpu[num_decodes:] - num_decode_tokens
            prefill_state_indices = non_spec_state_indices_tensor[num_decodes:]
        else:
            prefill_query_start_loc = non_spec_query_start_loc
            prefill_query_start_loc_cpu = non_spec_query_start_loc_cpu
            prefill_state_indices = non_spec_state_indices_tensor
        (chunk_indices, chunk_offsets) = self._build_chunk_metadata(prefill_query_start_loc, prefill_query_start_loc_cpu, query_start_loc.device)
    if num_prefills > 0:
        context_lens_tensor = m.compute_num_computed_tokens()
        has_initial_state = context_lens_tensor > 0
        if spec_sequence_masks_cpu is not None:
            has_initial_state = has_initial_state[~spec_sequence_masks_cpu]
            assert non_spec_query_start_loc_cpu is not None
        (nums_dict, batch_ptr, token_chunk_offset_ptr) = compute_causal_conv1d_metadata(non_spec_query_start_loc_cpu, device=query_start_loc.device)
        if spec_sequence_masks is None and num_decodes > 0:
            prefill_has_initial_state = has_initial_state[num_decodes:]
        else:
            prefill_has_initial_state = has_initial_state
    else:
        has_initial_state = None
    assert not (num_decodes > 0 and num_spec_decodes > 0), f'num_decodes: {num_decodes}, num_spec_decodes: {num_spec_decodes}'
    batch_size = m.num_reqs
    if self.use_full_cuda_graph and num_prefills == 0 and (num_decodes == 0) and (num_spec_decodes <= self.decode_cudagraph_max_bs) and (num_spec_decode_tokens <= self.decode_cudagraph_max_bs):
        assert spec_sequence_masks is not None
        self.spec_state_indices_tensor[:num_spec_decodes].copy_(spec_state_indices_tensor, non_blocking=True)
        spec_state_indices_tensor = self.spec_state_indices_tensor[:batch_size]
        spec_state_indices_tensor[num_spec_decodes:].fill_(NULL_BLOCK_ID)
        self.spec_sequence_masks[:num_spec_decodes].copy_(spec_sequence_masks[:num_spec_decodes], non_blocking=True)
        spec_sequence_masks = self.spec_sequence_masks[:batch_size]
        spec_sequence_masks[num_spec_decodes:].fill_(False)
        assert non_spec_token_indx is not None and spec_token_indx is not None
        self.non_spec_token_indx[:non_spec_token_indx.size(0)].copy_(non_spec_token_indx, non_blocking=True)
        non_spec_token_indx = self.non_spec_token_indx[:non_spec_token_indx.size(0)]
        self.spec_token_indx[:spec_token_indx.size(0)].copy_(spec_token_indx, non_blocking=True)
        spec_token_indx = self.spec_token_indx[:spec_token_indx.size(0)]
        self.spec_query_start_loc[:num_spec_decodes + 1].copy_(spec_query_start_loc, non_blocking=True)
        spec_num_query_tokens = spec_query_start_loc[-1]
        spec_query_start_loc = self.spec_query_start_loc[:batch_size + 1]
        spec_query_start_loc[num_spec_decodes + 1:].fill_(spec_num_query_tokens)
        self.num_accepted_tokens[:num_spec_decodes].copy_(num_accepted_tokens, non_blocking=True)
        num_accepted_tokens = self.num_accepted_tokens[:batch_size]
        num_accepted_tokens[num_spec_decodes:].fill_(1)
    if self.use_full_cuda_graph and num_prefills == 0 and (num_spec_decodes == 0) and (num_decodes <= self.decode_cudagraph_max_bs):
        self.non_spec_state_indices_tensor[:num_decodes].copy_(non_spec_state_indices_tensor, non_blocking=True)
        non_spec_state_indices_tensor = self.non_spec_state_indices_tensor[:batch_size]
        non_spec_state_indices_tensor[num_decodes:].fill_(NULL_BLOCK_ID)
        self.non_spec_query_start_loc[:num_decodes + 1].copy_(non_spec_query_start_loc, non_blocking=True)
        non_spec_num_query_tokens = non_spec_query_start_loc[-1]
        non_spec_query_start_loc = self.non_spec_query_start_loc[:batch_size + 1]
        non_spec_query_start_loc[num_decodes + 1:].fill_(non_spec_num_query_tokens)
    attn_metadata = GDNAttentionMetadata(num_prefills=num_prefills, num_prefill_tokens=num_prefill_tokens, num_decodes=num_decodes, num_decode_tokens=num_decode_tokens, num_spec_decodes=num_spec_decodes, num_spec_decode_tokens=num_spec_decode_tokens, num_actual_tokens=m.num_actual_tokens, has_initial_state=has_initial_state, chunk_indices=chunk_indices, chunk_offsets=chunk_offsets, prefill_query_start_loc=prefill_query_start_loc, prefill_state_indices=prefill_state_indices, prefill_has_initial_state=prefill_has_initial_state, spec_query_start_loc=spec_query_start_loc, non_spec_query_start_loc=non_spec_query_start_loc, spec_state_indices_tensor=spec_state_indices_tensor, non_spec_state_indices_tensor=non_spec_state_indices_tensor, spec_sequence_masks=spec_sequence_masks, spec_token_indx=spec_token_indx, non_spec_token_indx=non_spec_token_indx, num_accepted_tokens=num_accepted_tokens, nums_dict=nums_dict, batch_ptr=batch_ptr, token_chunk_offset_ptr=token_chunk_offset_ptr)
    return attn_metadata
