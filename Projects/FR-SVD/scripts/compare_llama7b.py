#!/data/zhangs/anaconda3/envs/compress/bin/python
"""LLaMA-7B, WT2, 20% removal: FR-SVD v1.6, AA-SVD Obj2 + refinement, GFWSVD."""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pyarrow.parquet as pq
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM

FR_ROOT = Path(__file__).resolve().parents[1]
PROJECTS = FR_ROOT.parent
ROOT = PROJECTS.parent
MODEL = ROOT / "Models/llama-7b"
WT2 = ROOT / "Datasets/wikitext/wikitext-2-raw-v1"
METHODS = ("fr_v16", "aa_svd", "gfwsvd")
GROUPS = (("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
          ("self_attn.o_proj",), ("mlp.gate_proj", "mlp.up_proj"), ("mlp.down_proj",))
PROJECTIONS = tuple(name for group in GROUPS for name in group)
KEEP, SEED, LENGTH, SAMPLES = 0.8, 3, 2048, 256
FR_OPTIONS = dict(fisher_mode="guide", enable_innovation2=True,
                  innovation2_mode="cealc_residual", r1_ratio=0.92,
                  eps=1e-4, lambda_phi=1e-4, lambda_psi=1e-4,
                  max_fisher_samples=32, alpha_min=0.4, alpha_max=0.6, rho=0.95)


def source_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(FR_ROOT))
fr = source_module("fr_svd_v16", FR_ROOT / "FR_SVD_v1.6.py")
aa_math = source_module("aa_math", PROJECTS / "AA-SVD/aa_svd/compression/utils.py")


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def prepare_tokens(output):
    cache = output / "wt2_tokens.pt"
    if cache.exists():
        return
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False, local_files_only=True)

    def encode(split):
        parquet = WT2 / f"{split}-00000-of-00001.parquet"
        texts = pq.read_table(parquet, columns=["text"])["text"].to_pylist()
        return tokenizer("\n\n".join(texts), return_tensors="pt", verbose=False).input_ids[0]

    train = encode("train")
    rng = random.Random(SEED)
    offsets = [rng.randrange(len(train) - LENGTH + 1) for _ in range(SAMPLES)]
    calib = torch.stack([train[i:i + LENGTH] for i in offsets])
    test = encode("test").unfold(0, LENGTH, LENGTH).contiguous()
    write_json(output / "protocol.json", {
        "model": str(MODEL), "dataset": str(WT2), "seed": SEED, "train_offsets": offsets,
        "calibration_shape": list(calib.shape), "test_shape": list(test.shape),
        "calibration_sha256": hashlib.sha256(calib.numpy().tobytes()).hexdigest(),
        "test_sha256": hashlib.sha256(test.numpy().tobytes()).hexdigest(),
        "rank_rule": "floor(0.8 * in * out / (in + out))", "projections": PROJECTIONS,
        "evaluation": "fp16 weights, fp32 CE, every complete 2048-token WT2 test window",
        "fr_v16": FR_OPTIONS,
        "aa_svd": {"objective": "obj2", "calibration_dtype": "bf16", "remapping": False,
                   "block_refinement": "25 epochs, AdamW lr 1e-4, wd .01, clip 1, batch 32, microbatch 4"},
        "gfwsvd": {"method": "arXiv v1, matched per-projection ranks",
                   "gradients": "dense-model next-token NLL, 32 effective batches of 8 sequences",
                   "fit": "LowRankArena CuPy svds(k=1), FP32 weight-gradient accumulation",
                   "factorization": "official main two-sided Cholesky formulation, GPU triangular solves"},
        "torch": str(torch.__version__), "python": sys.version, "script": str(Path(__file__).resolve()),
    })
    temporary = cache.with_suffix(".tmp")
    torch.save({"calib": calib, "test": test}, temporary)
    temporary.replace(cache)


def rank_for(weight):
    m, n = weight.shape
    return int(KEEP * m * n / (m + n))


def run_block(block, x):
    # All windows are full-length and unpadded. LlamaSdpaAttention applies its causal mask.
    positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
    return block(x, position_ids=positions, use_cache=False)[0]


@torch.no_grad()
def block_outputs(block, inputs):
    outputs = torch.empty_like(inputs, device="cpu")
    for start in range(0, len(inputs), 4):
        part = inputs[start:start + 4]
        outputs[start:start + len(part)].copy_(run_block(block, part.cuda()))
    return outputs


@torch.no_grad()
def install_factors(block, name, right_chol, left_chol=None):
    # AA Obj2 = whiten only on original inputs. GFWSVD = whiten on both sides.
    layer = block.get_submodule(name)
    rank = rank_for(layer.weight)
    weighted = layer.weight.float() @ right_chol
    if left_chol is not None:
        weighted = left_chol.T @ weighted
    u, s, vh = aa_math.safe_svd(weighted, full_matrices=False)
    sqrt_s = s[:rank].sqrt()
    left = u[:, :rank] * sqrt_s
    if left_chol is not None:
        left = torch.linalg.solve_triangular(left_chol.T, left, upper=True)
    right = torch.linalg.solve_triangular(right_chol.T, vh[:rank].T * sqrt_s, upper=True).T
    fr.replace_linear_with_factorized(block, name, left, right, layer.bias, "cuda", layer.weight.dtype)


def aa_refine(block, inputs, targets, epochs):
    # Official LlamaModelAdapter.finetune_layer objective, schedule and update budget.
    batches = math.ceil(len(inputs) / 32)
    total_steps = epochs * batches
    warmup = int(total_steps * 0.1)

    def multiplier(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / (total_steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    block.requires_grad_(True)
    optimizer = torch.optim.AdamW(block.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    for epoch in range(epochs):
        for start in range(0, len(inputs), 32):
            stop = min(start + 32, len(inputs))
            optimizer.zero_grad(set_to_none=True)
            batch_loss = torch.zeros((), device="cuda")
            for i in range(start, stop, 4):
                end = min(i + 4, stop)
                prediction = run_block(block, inputs[i:end].cuda())
                loss = F.mse_loss(prediction.float(), targets[i:end].cuda().float())
                loss = loss * ((end - i) / (stop - start))
                loss.backward()
                batch_loss += loss.detach()
            torch.nn.utils.clip_grad_norm_(block.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
        if epoch in (0, epochs - 1):
            print(f"  refinement epoch {epoch + 1}/{epochs}, last batch MSE={batch_loss.item():.6g}")
    optimizer.zero_grad(set_to_none=True)
    block.requires_grad_(False)


def run_aa(model, tokens, smoke):
    model.bfloat16().requires_grad_(False)
    reference = model.model.embed_tokens(tokens)
    student = reference
    refinement_seconds = 0.0
    for index, block in enumerate(model.model.layers):
        print(f"AA block {index + 1}/{len(model.model.layers)}")
        block.cuda()
        grams = {}

        def collect(name):
            def hook(module, args):
                x = args[0].flatten(0, 1).float()
                if name not in grams:
                    grams[name] = x.T @ x
                else:
                    grams[name].addmm_(x.T, x)
            return hook

        handles = [block.get_submodule(group[0]).register_forward_pre_hook(collect(group[0]))
                   for group in GROUPS]
        targets = block_outputs(block, reference)
        for handle in handles:
            handle.remove()
        for group in GROUPS:
            chol = aa_math.safe_cholesky(grams.pop(group[0]).div_(tokens.numel()))
            for name in group:
                install_factors(block, name, chol)
        torch.cuda.synchronize()
        started = time.monotonic()
        aa_refine(block, student, targets, 1 if smoke else 25)
        torch.cuda.synchronize()
        refinement_seconds += time.monotonic() - started
        student = block_outputs(block, student)
        reference = targets
        block.cpu()
    return {"refinement_seconds": refinement_seconds}


@torch.no_grad()
def fisher_cholesky(factor):
    # Fix the arbitrary global scale before applying the author's trace-scaled shrinkage.
    factor = (factor + factor.T) * 0.5
    factor /= factor.diagonal().mean()
    chol, info = torch.linalg.cholesky_ex(factor)
    if info.item() == 0:
        return chol, 0.0
    # Estimated PSD factors can be singular. Upstream increases alpha by decades.
    for alpha in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        regularized = (1.0 - alpha) * factor
        regularized.diagonal().add_(alpha)
        chol, info = torch.linalg.cholesky_ex(regularized)
        if info.item() == 0:
            return chol, alpha
    raise ArithmeticError("GFWSVD Fisher factor has no finite Cholesky decomposition")


def gfw_gradients(block, inputs, upstream):
    # Replay one dense block backwards, before installing any of its low-rank factors.
    # upstream contains the derivative of each sequence's mean NLL at this block's output.
    group_size = 8
    weights = [block.get_submodule(name).weight for name in PROJECTIONS]
    for weight in weights:
        weight.requires_grad_(True)
    grads = {name: torch.empty((len(inputs) // group_size, *w.shape), dtype=torch.float32)
             for name, w in zip(PROJECTIONS, weights)}
    for group, start in enumerate(range(0, len(inputs), group_size)):
        accum = [torch.zeros_like(w, dtype=torch.float32) for w in weights]
        for i in range(start, start + group_size, 2):
            x = inputs[i:i + 2].cuda().requires_grad_()
            output = run_block(block, x)
            dx, *dw = torch.autograd.grad(output, (x, *weights), upstream[i:i + 2].cuda())
            upstream[i:i + 2].copy_(dx)
            for total, grad in zip(accum, dw):
                total.add_(grad)
        for name, total in zip(PROJECTIONS, accum):
            grads[name][group].copy_(total.div_(group_size))
    block.requires_grad_(False)
    return grads


def run_gfw(model, tokens):
    import cupy as cp
    fit = source_module("krondecomp_cupy", PROJECTS / "GFW-SVD_LowRankArena/kronfwsvd/krondecomp_cupy.py")
    cp.random.seed(SEED)
    model.bfloat16().requires_grad_(False)
    layers = model.model.layers
    shrinkage = {}
    with tempfile.TemporaryDirectory(prefix="gfw-llama7b-", dir="/tmp") as directory:
        scratch = Path(directory)
        hidden = model.model.embed_tokens(tokens)
        for idx, block in enumerate(layers):
            print(f"GFWSVD dense forward {idx + 1}/{len(layers)}")
            torch.save(hidden, scratch / f"{idx}.pt")
            hidden = block_outputs(block.cuda(), hidden)
            block.cpu()
        upstream = torch.empty_like(hidden)
        model.model.norm.cuda()
        model.lm_head.cuda()
        for i in range(len(tokens)):
            x = hidden[i:i + 1].cuda().requires_grad_()
            logits = model.lm_head(model.model.norm(x[:, :-1])).float()
            loss = F.cross_entropy(logits[0], tokens[i, 1:].cuda())
            upstream[i:i + 1].copy_(torch.autograd.grad(loss, x)[0])
        del hidden, logits, x, loss
        model.model.norm.cpu()
        model.lm_head.cpu()
        for idx in reversed(range(len(layers))):
            print(f"GFWSVD dense backward and compression, block {idx + 1}/{len(layers)}")
            block = layers[idx].cuda()
            path = scratch / f"{idx}.pt"
            inputs = torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
            gradients = gfw_gradients(block, inputs, upstream)
            del inputs
            path.unlink()
            for name in PROJECTIONS:
                batch_grads = gradients.pop(name)
                # One common scale leaves the leading Kronecker directions unchanged.
                batch_grads.div_(torch.linalg.vector_norm(batch_grads[0]))
                torch.cuda.empty_cache()
                out_cp, in_cp = fit.get_kron_factors(batch_grads.numpy(), top_k=1,
                                                     layer_name=f"L{idx}.{name}", device_id=0)
                out_factor = torch.from_dlpack(out_cp).clone()
                in_factor = torch.from_dlpack(in_cp).clone()
                del out_cp, in_cp, batch_grads
                cp.get_default_memory_pool().free_all_blocks()
                out_chol, out_alpha = fisher_cholesky(out_factor)
                in_chol, in_alpha = fisher_cholesky(in_factor)
                del out_factor, in_factor
                install_factors(block, name, in_chol, out_chol)
                shrinkage[f"{idx}.{name}"] = [out_alpha, in_alpha]
                del out_chol, in_chol
            block.cpu()
    return {"output_input_shrinkage": shrinkage}


@torch.inference_mode()
def perplexity(model, windows):
    model.half().cuda().eval()
    total = torch.zeros((), device="cuda", dtype=torch.float64)
    for window in windows:
        ids = window.unsqueeze(0).cuda()
        logits = model(ids, use_cache=False).logits[0, :-1].float()
        total += F.cross_entropy(logits, ids[0, 1:], reduction="sum")
    predicted = len(windows) * (windows.shape[1] - 1)
    return math.exp(total.item() / predicted), predicted


def run_method(method, output, smoke):
    torch.manual_seed(SEED)
    torch.set_num_threads(8)
    tokens = torch.load(output / "wt2_tokens.pt", map_location="cpu", weights_only=True)
    calib, test = tokens["calib"], tokens["test"]
    if smoke:
        cfg = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
                          num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4)
        cfg._attn_implementation = "sdpa"
        model = LlamaForCausalLM(cfg).half()
        calib, test = calib[:16, :32] % 128, test[:2, :32] % 128
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16,
                                                     attn_implementation="sdpa", local_files_only=True)
    model.eval()
    model.config.use_cache = False
    model.seqlen = calib.shape[1]
    original = sum(p.numel() for p in model.parameters())
    shapes = {name: list(model.model.layers[0].get_submodule(name).weight.shape) for name in PROJECTIONS}
    ranks = {name: rank_for(model.model.layers[0].get_submodule(name).weight) for name in PROJECTIONS}
    print(f"{method}: {len(model.model.layers)} blocks, calib={tuple(calib.shape)}, ranks={ranks}")
    dest = output / ("smoke" if smoke else "formal") / method
    dest.mkdir(parents=True, exist_ok=True)
    gpu_log = dest / "gpu_process_mib.csv"
    torch.cuda.reset_peak_memory_stats()
    sampler = subprocess.Popen(["nvidia-smi", "-i", os.environ["CUDA_VISIBLE_DEVICES"],
                                "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits",
                                "-lms", "1000", "-f", str(gpu_log)])
    started = time.monotonic()
    try:
        if method == "fr_v16":
            loader = [{"input_ids": x[None], "attention_mask": torch.ones_like(x[None])} for x in calib]
            info = fr.fr_svd_compress_layerwise(str(MODEL), model, loader, KEEP, "cuda:0", **FR_OPTIONS)
        elif method == "aa_svd":
            info = run_aa(model, calib, smoke)
        else:
            info = run_gfw(model, calib)
        torch.cuda.synchronize()
        seconds = time.monotonic() - started
        peak = torch.cuda.max_memory_allocated() / 2**30
        reserved = torch.cuda.max_memory_reserved() / 2**30
    finally:
        sampler.terminate()
        sampler.wait()
    samples = [int(line.split(",")[1]) for line in gpu_log.read_text().splitlines()
               if line.split(",")[0].strip() == str(os.getpid())]
    # v1.6 records an infinite score when its final TSVD fallback is selected.
    torch.save(info, dest / "compression_details.pt")
    for block in model.model.layers:
        for name, rank in ranks.items():
            assert block.get_submodule(name).rank == rank, name
    dense_linear = len(model.model.layers) * sum(m * n for m, n in shapes.values())
    factors = len(model.model.layers) * sum(ranks[name] * (m + n) for name, (m, n) in shapes.items())
    installed = sum(p.numel() for p in model.parameters())
    assert installed == original - dense_linear + factors
    if not smoke:
        checkpoint = dest / "model.pt"
        temporary = checkpoint.with_suffix(".tmp")
        torch.save({"config": model.config.to_dict(), "ranks_per_block": ranks,
                    "factor_class": "FR_SVD_v1.6.FactorizedLinear",
                    "state_dict": model.cpu().half().state_dict()}, temporary)
        temporary.replace(checkpoint)
    ppl, count = perplexity(model, test)
    result = dict(method=method, wt2_ppl=ppl, test_predictions=count, compressed_matrices=len(ranks)*len(model.model.layers),
                  original_linear=dense_linear, factor_params=factors, total_params=installed,
                  compression_seconds=seconds, torch_peak_allocated_gib=peak, torch_peak_reserved_gib=reserved,
                  process_peak_mib_1s=max(samples) if samples else None)
    write_json(dest / "metrics.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--smoke", action="store_true", help="Run the same three paths on a tiny two-block LLaMA")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prepare_tokens(output)
    if args.method:
        run_method(args.method, output, args.smoke)
        return
    base = output / ("smoke" if args.smoke else "formal")
    results = []
    for method in METHODS:
        dest = base / method
        dest.mkdir(parents=True, exist_ok=True)
        if not (dest / "metrics.json").exists():
            cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--output", str(output), "--method", method]
            if args.smoke:
                cmd.append("--smoke")
            print(time.strftime("%F %T"), " ".join(cmd), flush=True)
            with (dest / "run.log").open("a", buffering=1) as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        results.append(json.loads((dest / "metrics.json").read_text()))
        with (base / "summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"{method} WT2 PPL = {results[-1]['wt2_ppl']:.6f}", flush=True)


if __name__ == "__main__":
    main()
