"""Measure sparsity/neuron-like activity statistics for frozen latents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from torchvision.transforms import v2 as transforms


def image_transform(img_size: int):
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=img_size),
    ])


@torch.no_grad()
def encode_latents(model, dataset, indices, *, batch_size, img_size, device):
    tfm = image_transform(img_size)
    xs = []
    model.eval().to(device)
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    for start in range(0, len(indices), batch_size):
        rows = [dataset[int(i)] for i in indices[start:start + batch_size]]
        pixels = torch.stack([row["pixels"] for row in rows], dim=0)
        b, t = pixels.shape[:2]
        pixels = tfm(pixels.reshape(b * t, *pixels.shape[2:])).reshape(b, t, 3, img_size, img_size)
        out = model.encode({"pixels": pixels.to(device)})
        xs.append(out["emb"].detach().float().cpu().reshape(b * t, -1))
    return torch.cat(xs, dim=0)


def vg_sparseness(x: torch.Tensor, dim: int, eps: float = 1e-12):
    n = x.shape[dim]
    if n <= 1:
        return x.new_zeros(x.shape[:dim] + x.shape[dim + 1:])
    mean = x.mean(dim=dim)
    mean2 = x.square().mean(dim=dim).clamp_min(eps)
    s = (1.0 - mean.square() / mean2) / (1.0 - 1.0 / n)
    return s.clamp(0.0, 1.0)


def hoyer_sparseness(x: torch.Tensor, dim: int, eps: float = 1e-12):
    n = x.shape[dim]
    if n <= 1:
        return x.new_zeros(x.shape[:dim] + x.shape[dim + 1:])
    l1 = x.abs().sum(dim=dim)
    l2 = x.square().sum(dim=dim).sqrt().clamp_min(eps)
    return ((n ** 0.5 - l1 / l2) / (n ** 0.5 - 1.0)).clamp(0.0, 1.0)


def gini(x: torch.Tensor, dim: int = -1, eps: float = 1e-12):
    x = x.abs().sort(dim=dim).values
    n = x.shape[dim]
    idx = torch.arange(1, n + 1, dtype=x.dtype, device=x.device)
    shape = [1] * x.ndim
    shape[dim] = n
    idx = idx.reshape(shape)
    total = x.sum(dim=dim).clamp_min(eps)
    return ((2.0 * (idx * x).sum(dim=dim)) / (n * total) - (n + 1.0) / n).clamp(0.0, 1.0)


def top_share(x: torch.Tensor, frac: float, dim: int = -1, eps: float = 1e-12):
    x = x.clamp_min(0.0)
    n = x.shape[dim]
    k = max(1, int(round(frac * n)))
    vals = x.topk(k, dim=dim).values
    return vals.sum(dim=dim) / x.sum(dim=dim).clamp_min(eps)


def summarize_activity(name: str, x: torch.Tensor):
    flat = x.flatten()
    pop_vg = vg_sparseness(x, dim=1)
    life_vg = vg_sparseness(x, dim=0)
    pop_hoyer = hoyer_sparseness(x, dim=1)
    life_hoyer = hoyer_sparseness(x, dim=0)
    pop_gini = gini(x, dim=1)
    return {
        "name": name,
        "shape": list(x.shape),
        "mean": float(flat.mean()),
        "std": float(flat.std(unbiased=False)),
        "min": float(flat.min()),
        "p50": float(torch.quantile(flat, 0.50)),
        "p90": float(torch.quantile(flat, 0.90)),
        "p95": float(torch.quantile(flat, 0.95)),
        "p99": float(torch.quantile(flat, 0.99)),
        "max": float(flat.max()),
        "population_vg_mean": float(pop_vg.mean()),
        "population_vg_std": float(pop_vg.std(unbiased=False)),
        "lifetime_vg_mean": float(life_vg.mean()),
        "lifetime_vg_std": float(life_vg.std(unbiased=False)),
        "population_hoyer_mean": float(pop_hoyer.mean()),
        "lifetime_hoyer_mean": float(life_hoyer.mean()),
        "population_gini_mean": float(pop_gini.mean()),
        "top_1pct_share_mean": float(top_share(x, 0.01, dim=1).mean()),
        "top_5pct_share_mean": float(top_share(x, 0.05, dim=1).mean()),
        "active_frac_gt_1e-6": float((x > 1e-6).float().mean()),
        "active_frac_gt_1e-3": float((x > 1e-3).float().mean()),
        "active_frac_gt_1e-2": float((x > 1e-2).float().mean()),
        "active_frac_gt_1e-1": float((x > 1e-1).float().mean()),
    }


def poisson_summaries(model, emb):
    lambda0 = float(getattr(model, "lambda0", 1.0))
    if hasattr(model, "log_rate_min") and hasattr(model, "log_rate_max"):
        r = emb.clamp(float(model.log_rate_min), float(model.log_rate_max))
    else:
        r = emb
    lam = lambda0 * torch.exp(r)
    prior_kl = lam * r - lam + lambda0
    excess = (lam - lambda0).clamp_min(0.0)
    deficit = (lambda0 - lam).clamp_min(0.0)
    out = {
        "log_rate": summarize_activity("log_rate_abs", r.abs()),
        "rate": summarize_activity("rate", lam),
        "prior_kl_energy": summarize_activity("prior_kl_energy", prior_kl),
        "excess_rate": summarize_activity("excess_rate_above_lambda0", excess),
        "deficit_rate": summarize_activity("deficit_rate_below_lambda0", deficit),
        "lambda0": lambda0,
        "frac_rate_gt_lambda0": float((lam > lambda0).float().mean()),
        "frac_rate_gt_2x_lambda0": float((lam > 2.0 * lambda0).float().mean()),
        "frac_rate_gt_4x_lambda0": float((lam > 4.0 * lambda0).float().mean()),
        "frac_rate_lt_half_lambda0": float((lam < 0.5 * lambda0).float().mean()),
        "frac_rate_within_5pct_lambda0": float(((lam - lambda0).abs() <= 0.05 * lambda0).float().mean()),
        "frac_prior_kl_gt_0p01": float((prior_kl > 0.01).float().mean()),
        "frac_prior_kl_gt_0p1": float((prior_kl > 0.1).float().mean()),
        "frac_prior_kl_gt_1": float((prior_kl > 1.0).float().mean()),
    }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--out-dir", default="outputs/latent_sparsity")
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    dataset = swm.data.load_dataset(
        "pusht_expert_train.h5",
        transform=None,
        cache_dir=None,
        num_steps=args.num_steps,
        frameskip=args.frameskip,
        keys_to_load=["pixels"],
        keys_to_cache=[],
    )
    rng = np.random.default_rng(args.seed)
    indices = np.sort(rng.choice(len(dataset), size=min(args.num_samples, len(dataset)), replace=False))
    model = swm.wm.utils.load_pretrained(args.policy)
    emb = encode_latents(model, dataset, indices, batch_size=args.batch_size, img_size=args.img_size, device=args.device)

    result = {
        "name": args.name,
        "policy": args.policy,
        "num_vectors": int(emb.shape[0]),
        "embedding_dim": int(emb.shape[1]),
        "num_samples": int(len(indices)),
        "num_steps": args.num_steps,
        "frameskip": args.frameskip,
        "embedding_abs": summarize_activity("embedding_abs", emb.abs()),
    }
    if hasattr(model, "lambda0") or "pois" in type(model).__name__.lower():
        result["poisson"] = poisson_summaries(model, emb)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.name}_sparsity.json"
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
