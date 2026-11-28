"""Run the khDv C1 diagnostic, uXrn W4 calibration comparison and khDv C2 model evaluations."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import time

import compare_llama7b as paired
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


BLOCKS = (8, 24)  # zero-based indices, fixed before observing any held-out losses
PROBE = "self_attn.q_proj"
VALIDATION_WINDOWS = 16
ALPACA = paired.ROOT / "Datasets/alpaca-cleaned/alpaca_data_cleaned.json"


def load_model(path):
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float16, attn_implementation="sdpa",
        low_cpu_mem_usage=True, local_files_only=True)
    model.config.use_cache = False
    model.seqlen = paired.LENGTH
    return model.eval()


def wt_stream(tokenizer, split):
    path = paired.WT2 / f"{split}-00000-of-00001.parquet"
    texts = pq.read_table(path, columns=["text"])["text"].to_pylist()
    return tokenizer("\n\n".join(texts), return_tensors="pt", verbose=False).input_ids[0]


def sample_windows(ids):
    rng = random.Random(paired.SEED)
    offsets = [rng.randrange(len(ids) - paired.LENGTH + 1) for _ in range(paired.SAMPLES)]
    return torch.stack([ids[i:i + paired.LENGTH] for i in offsets]), offsets


def digest(ids):
    return hashlib.sha256(ids.contiguous().numpy().tobytes()).hexdigest()


def measure_fr(path, keep, calibration, test, destination, provenance):
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "metrics.json").exists():
        return json.loads((destination / "metrics.json").read_text())
    model = load_model(path)
    loader = [{"input_ids": row[None], "attention_mask": torch.ones_like(row[None])}
              for row in calibration]
    paired.write_json(destination / "protocol.json", dict(
        model=str(path), version="FR-SVD v1.6", keep=keep, options=paired.FR_OPTIONS,
        calibration_shape=list(calibration.shape), calibration_sha256=digest(calibration),
        test_sha256=digest(test), seed=paired.SEED, **provenance))
    started = time.monotonic()
    stats = paired.fr.fr_svd_compress_layerwise(str(path), model, loader, keep, "cuda:0", **paired.FR_OPTIONS)
    torch.cuda.synchronize()
    seconds = time.monotonic() - started
    torch.save(stats, destination / "compression_details.pt")
    model.cpu().half()
    dense = factors = 0
    ranks = {}
    for block in model.model.layers:
        for name in paired.PROJECTIONS:
            layer = block.get_submodule(name)
            m, n = layer.out_features, layer.in_features
            assert layer.rank == int(keep * m * n / (m + n)), name
            dense += m * n
            factors += layer.rank * (m + n)
            ranks[name] = layer.rank
    temp = destination / "model.tmp"
    torch.save({"config": model.config.to_dict(), "ranks_per_block": ranks,
                "factor_class": "FR_SVD_v1.6.FactorizedLinear", "state_dict": model.state_dict()}, temp)
    temp.replace(destination / "model.pt")
    ppl, count = paired.perplexity(model, test)
    result = {"version": "FR-SVD v1.6", "model": path.name, "wt2_ppl": ppl, "test_predictions": count,
              "actual_linear_reduction": 1 - factors / dense, "compression_seconds": seconds}
    paired.write_json(destination / "metrics.json", result)
    print(json.dumps(result), flush=True)
    return result


def domain(root):
    destination = root / "followup" / "alpaca_to_wt2"
    tokenizer = AutoTokenizer.from_pretrained(paired.MODEL, use_fast=False, local_files_only=True)
    data = json.loads(ALPACA.read_text())
    records = ("\n".join(row[key] for key in ("instruction", "input", "output") if row[key])
               for row in data)
    ids = tokenizer("\n\n".join(records), return_tensors="pt", verbose=False).input_ids[0]
    calibration, offsets = sample_windows(ids)
    test = torch.load(root / "wt2_tokens.pt", weights_only=True)["test"]
    different_domain = measure_fr(paired.MODEL, paired.KEEP, calibration, test, destination,
                                  {"corpus": str(ALPACA), "train_offsets": offsets})
    reference = root / "formal/fr_v16/metrics.json"
    while not reference.exists():
        if (root / "formal.exit").exists():
            raise RuntimeError("Matched WT2 run ended before writing FR-SVD metrics")
        time.sleep(30)
    same_domain = json.loads(reference.read_text())
    paired.write_json(destination / "comparison.json", {
        "wt2_to_wt2_ppl": same_domain["wt2_ppl"], "alpaca_to_wt2_ppl": different_domain["wt2_ppl"],
        "ppl_change": different_domain["wt2_ppl"] - same_domain["wt2_ppl"]})


def high_compression(root, path):
    destination = root / "followup" / "c060" / path.name
    if (destination / "metrics.json").exists():
        return
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=not (path / "tokenizer.model").exists(),
                                               local_files_only=True)
    calibration, offsets = sample_windows(wt_stream(tokenizer, "train"))
    test = wt_stream(tokenizer, "test").unfold(0, paired.LENGTH, paired.LENGTH).contiguous()
    measure_fr(path, 0.4, calibration, test, destination,
               {"corpus": str(paired.WT2), "train_offsets": offsets})


def discarded_directions(block, reference, student):
    """Same coefficient, upstream inputs and rank. Score every discarded rank-one direction."""
    fr = paired.fr
    opts = paired.FR_OPTIONS
    teacher = copy.deepcopy(block).cuda()
    block.cuda().float()
    pos = torch.arange(student.shape[1])[None]
    fisher = fr.estimate_kronecker_fisher(
        block, teacher, student, reference, [None] * len(student), [pos] * len(student),
        str(paired.MODEL), "cuda:0", len(student), lambda_psi=opts["lambda_psi"],
        max_fisher_samples=opts["max_fisher_samples"], fisher_mode="guide")[PROBE]
    del teacher
    with torch.no_grad():
        weight = block.get_submodule(PROBE).weight
        h = torch.zeros((weight.shape[1], weight.shape[1]), device="cuda", dtype=torch.float32)
        delta = torch.zeros_like(h)
        for start in range(0, len(student), 4):
            x = block.input_layernorm(student[start:start + 4].cuda().float()).flatten(0, 1)
            ref = block.input_layernorm(reference[start:start + 4].cuda().float()).flatten(0, 1)
            h.addmm_(x.T, x)
            delta.addmm_((ref - x).T, x)
        _, _, h_sqrt, h_inv = fr.compute_H_decomposition(h, "cuda:0", opts["eps"])
        s, d = weight @ h_sqrt, weight @ delta @ h_inv
        rank = int(paired.rank_for(weight) * opts["r1_ratio"])
        beta, _ = fr.aces_beta_select(s, d, rank, alpha_min=opts["alpha_min"], alpha_max=opts["alpha_max"],
                                      rho=opts["rho"], beta_cap=opts["alpha_max"] / (1 + opts["alpha_max"]))
        guide = fr.build_fisher_transform(PROBE, fisher, h, "guide", "cuda:0",
                                          lambda_phi=opts["lambda_phi"], lambda_psi=opts["lambda_psi"])
        directions = {}
        for name, transform in (("activation", {"mode": "none"}), ("guide", guide)):
            sw, dw = fr.apply_fisher_to_target(s, d, transform)
            u, sigma, vh = torch.linalg.svd(sw + beta * dw, full_matrices=False)
            left, right = fr.factors_from_metric_svd(u[:, rank:], torch.ones_like(sigma[rank:]),
                                                     vh[rank:], h_inv, transform)
            directions[name] = (F.normalize(left, dim=0), F.normalize(right, dim=1))
    block.half()
    return directions, {"stage1_rank": rank, "discarded_directions": len(sigma) - rank, "fixed_beta": beta}


def validation_gradients(model, windows):
    # Replay the dense chain rule block by block. Only the active block and its
    # one-sequence autograd graph reside on GPU while other experiments run.
    model.bfloat16().requires_grad_(False)
    inputs = []
    with torch.no_grad():
        hidden = model.model.embed_tokens(windows)
        for i, block in enumerate(model.model.layers):
            if i >= min(BLOCKS):
                inputs.append(hidden)
            hidden = paired.block_outputs(block.cuda(), hidden)
            block.cpu()
    upstream = torch.empty_like(hidden)
    model.lm_head.cuda()
    model.model.norm.cuda()
    for j, ids in enumerate(windows):
        x = hidden[j:j + 1].cuda().requires_grad_()
        logits = model.lm_head(model.model.norm(x))[:, :-1].float()
        loss = F.cross_entropy(logits[0], ids[1:].cuda())
        upstream[j:j + 1].copy_(torch.autograd.grad(loss, x)[0])
    model.lm_head.cpu()
    model.model.norm.cpu()
    del hidden, x, logits, loss
    gradients = {i: torch.empty((len(windows), *model.model.layers[i].get_submodule(PROBE).weight.shape),
                                dtype=torch.bfloat16, device="cpu") for i in BLOCKS}
    for i in range(len(model.model.layers) - 1, min(BLOCKS) - 1, -1):
        block = model.model.layers[i].cuda()
        hidden = inputs.pop()
        weight = block.get_submodule(PROBE).weight.requires_grad_() if i in BLOCKS else None
        for j in range(len(windows)):
            x = hidden[j:j + 1].cuda().requires_grad_()
            output = paired.run_block(block, x)
            targets = (x, weight) if weight is not None else (x,)
            derivs = torch.autograd.grad(output, targets, upstream[j:j + 1].cuda())
            upstream[j:j + 1].copy_(derivs[0])
            if weight is not None:
                gradients[i][j].copy_(derivs[1])
        block.requires_grad_(False).cpu()
    return gradients


def diagnose(root):
    dest = root / "followup" / "discarded_directions"
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "metrics.json").exists():
        return
    fr_dir = root / "formal/fr_v16"
    cache = dest / "validation_gradients.pt"
    if not cache.exists():
        tokenizer = AutoTokenizer.from_pretrained(paired.MODEL, use_fast=False, local_files_only=True)
        validation = wt_stream(tokenizer, "validation").unfold(0, paired.LENGTH, paired.LENGTH)
        indices = random.Random(paired.SEED).sample(range(len(validation)), VALIDATION_WINDOWS)
        print("Collecting dense-model NLL gradients on 16 held-out WT2 windows", flush=True)
        model = load_model(paired.MODEL)
        gradients = validation_gradients(model, validation[indices])
        torch.save({"indices": indices, "gradients": gradients}, cache.with_suffix(".tmp"))
        cache.with_suffix(".tmp").replace(cache)
        del model, gradients
        torch.cuda.empty_cache()
    print("Dense NLL gradients ready. Waiting for the paired FR-SVD checkpoint.", flush=True)
    while not (fr_dir / "model.pt").exists():
        if (root / "formal.exit").exists():
            raise RuntimeError("Matched FR-SVD run ended before writing its checkpoint")
        time.sleep(15)
    factors = torch.load(str(fr_dir / "model.pt"), mmap=True, weights_only=True)["state_dict"]
    selection = torch.load(fr_dir / "compression_details.pt", weights_only=True)
    model = load_model(paired.MODEL).requires_grad_(False)
    calib = torch.load(root / "wt2_tokens.pt", weights_only=True)["calib"]
    reference = model.model.embed_tokens(calib)
    student = reference
    directions, results = {}, {}
    for i, block in enumerate(model.model.layers):
        if i in BLOCKS:
            print(f"Deriving discarded directions for block {i}", flush=True)
            directions[i], results[i] = discarded_directions(block, reference, student)
            results[i]["deployed_candidate"] = selection[i]["sublayers"][PROBE]["selected_candidate"]
        if i == max(BLOCKS):
            block.cpu()
            break
        with torch.no_grad():
            reference = paired.block_outputs(block.cuda(), reference)
            compressed = copy.deepcopy(block)
            for name in paired.PROJECTIONS:
                prefix = f"model.layers.{i}.{name}"
                paired.fr.replace_linear_with_factorized(
                    compressed, name, factors[f"{prefix}.u_proj.weight"], factors[f"{prefix}.v_proj.weight"],
                    None, "cuda", torch.float16)
            student = paired.block_outputs(compressed, student)
            del compressed
            block.cpu()
    del reference, student, factors
    del model
    validation = torch.load(str(cache), mmap=True, weights_only=True)
    indices = validation["indices"]
    scores = {i: {"guide": [], "activation": []} for i in BLOCKS}
    for i in BLOCKS:
        for saved in validation["gradients"][i]:
            grad = saved.to(device="cuda", dtype=torch.float32)
            for name, (left, right) in directions[i].items():
                projection = (left * (grad @ right.T)).sum(0)
                scores[i][name].append(projection.double().square().mean().item())
    for i in BLOCKS:
        means = {k: sum(v) / len(v) for k, v in scores[i].items()}
        results[i].update(mean_squared_directional_derivative=means, per_sequence=scores[i],
                          guide_to_activation_ratio=means["guide"] / means["activation"])
    paired.write_json(dest / "metrics.json", {"model": str(paired.MODEL), "projection": PROBE,
                      "held_out_split": "WT2 validation", "window_indices": indices, "blocks": results})
    print("Guide / activation mean discarded sensitivity:",
          {i: results[i]["guide_to_activation_ratio"] for i in BLOCKS}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="Directory of the completed three-way run")
    parser.add_argument("--experiment", required=True, choices=["directions", "domain", "high-compression"])
    parser.add_argument("--model", type=Path, help="Base model directory for high-compression")
    args = parser.parse_args()
    torch.manual_seed(paired.SEED)
    torch.set_num_threads(8)
    if args.experiment == "directions":
        diagnose(args.output)
    elif args.experiment == "domain":
        domain(args.output)
    else:
        if args.model is None:
            parser.error("--model is required for high-compression")
        high_compression(args.output, args.model.resolve())


if __name__ == "__main__":
    main()
