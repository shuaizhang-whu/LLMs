"""Identify the native convolution's mixed-precision arithmetic, no model run."""
import json
from pathlib import Path
import sys

import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_gpu as helper

torch.manual_seed(731)
value = helper.layer()
data = helper.inputs(value, 17)
metadata = helper.meta([17], [helper.BASE], [False])
conv_state = value.kv_cache[0]
if not helper.native.is_conv_state_dim_first():
    conv_state = conv_state.transpose(-1, -2)
observed = helper.native.causal_conv1d_fn(
    data[0].clone().T, value.conv1d.weight.view(-1, helper.WIDTH), None,
    activation="silu", conv_states=conv_state, has_initial_state=metadata.has_initial_state,
    cache_indices=metadata.non_spec_state_indices_tensor,
    query_start_loc=metadata.non_spec_query_start_loc, metadata=metadata).T.float().cpu()
x = data[0].float().cpu()
w = value.conv1d.weight.float().cpu().squeeze(1)
history = torch.zeros(helper.WIDTH - 1, x.shape[1])
references = {"fp32_products": [], "bf16_products": []}
for i in range(len(x)):
    window = torch.cat([history, x[i:i+1]])
    full = window.T * w
    rounded = (window.T.to(torch.bfloat16) * w.to(torch.bfloat16)).float()
    for key, products in [("fp32_products", full), ("bf16_products", rounded)]:
        accumulator = torch.zeros(x.shape[1])
        for j in range(helper.WIDTH):
            accumulator += products[:, j]
        references[key].append(torch.nn.functional.silu(accumulator).to(torch.bfloat16).float())
    history = window[1:]
report = {key: {"unequal_values": int((torch.stack(rows) != observed).sum()),
                "max_abs": float((torch.stack(rows) - observed).abs().max()),
                "values": observed.numel()} for key, rows in references.items()}
print(json.dumps(report, indent=2))
