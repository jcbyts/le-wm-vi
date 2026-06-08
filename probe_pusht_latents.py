"""Linear probes from frozen world-model latents to PushT state/proprio.

This is a cheap complement to planning evals: it asks whether the pusher and
block state variables are linearly recoverable from the learned latent space.
"""

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
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=img_size),
        ]
    )


def load_rows(dataset, indices):
    rows = [dataset[int(i)] for i in indices]
    pixels = torch.stack([row["pixels"] for row in rows], dim=0)
    state = torch.stack([torch.as_tensor(row["state"]).float() for row in rows], dim=0)
    proprio = torch.stack([torch.as_tensor(row["proprio"]).float() for row in rows], dim=0)
    return pixels, {"state": state, "proprio": proprio}


@torch.no_grad()
def encode_dataset(model, dataset, indices, *, batch_size, img_size, device):
    tfm = image_transform(img_size)
    xs = []
    ys = {"state": [], "proprio": []}

    model.eval().to(device)
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        pixels, targets = load_rows(dataset, batch_idx)
        b, t = pixels.shape[:2]
        pixels = tfm(pixels.reshape(b * t, *pixels.shape[2:])).reshape(b, t, 3, img_size, img_size)
        out = model.encode({"pixels": pixels.to(device)})
        emb = out["emb"].detach().float().cpu()
        xs.append(emb.reshape(b * t, -1))
        for key, value in targets.items():
            ys[key].append(value.reshape(b * t, -1))

    x = torch.cat(xs, dim=0)
    y = {key: torch.cat(parts, dim=0) for key, parts in ys.items()}
    return x, y


def standardize_train_test(train, test, eps=1e-6):
    mean = train.mean(0, keepdim=True)
    std = train.std(0, keepdim=True).clamp_min(eps)
    return (train - mean) / std, (test - mean) / std, mean, std


def ridge_fit_predict(x_train, y_train, x_test, ridge):
    ones_train = torch.ones(x_train.shape[0], 1, dtype=x_train.dtype)
    ones_test = torch.ones(x_test.shape[0], 1, dtype=x_test.dtype)
    x_train = torch.cat([x_train, ones_train], dim=1)
    x_test = torch.cat([x_test, ones_test], dim=1)

    eye = torch.eye(x_train.shape[1], dtype=x_train.dtype)
    eye[-1, -1] = 0.0
    lhs = x_train.T @ x_train + ridge * eye
    rhs = x_train.T @ y_train
    weights = torch.linalg.solve(lhs, rhs)
    return x_test @ weights


def probe_one_representation(name, x_train, x_test, y_train, y_test, ridge):
    result = {}
    x_train_z, x_test_z, _, _ = standardize_train_test(x_train, x_test)

    for target, ytr in y_train.items():
        yte = y_test[target]
        ytr_z, _, y_mean, y_std = standardize_train_test(ytr, yte)
        pred_z = ridge_fit_predict(x_train_z, ytr_z, x_test_z, ridge)
        pred = pred_z * y_std + y_mean

        resid = yte - pred
        ss_res = resid.pow(2).sum(0)
        ss_tot = (yte - yte.mean(0, keepdim=True)).pow(2).sum(0).clamp_min(1e-12)
        r2 = 1.0 - ss_res / ss_tot
        rmse = resid.pow(2).mean(0).sqrt()
        result[target] = {
            "r2_mean": float(r2.mean()),
            "r2_per_dim": [float(v) for v in r2],
            "rmse_mean": float(rmse.mean()),
            "rmse_per_dim": [float(v) for v in rmse],
        }

    result["representation"] = name
    result["x_dim"] = int(x_train.shape[1])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, help="Checkpoint path relative to stable_worldmodel checkpoints")
    parser.add_argument("--name", required=True)
    parser.add_argument("--out-dir", default="outputs/pusht_latent_probes")
    parser.add_argument("--num-samples", type=int, default=2048)
    parser.add_argument("--test-frac", type=float, default=0.25)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    dataset = swm.data.load_dataset(
        "pusht_expert_train.h5",
        transform=None,
        cache_dir=None,
        num_steps=args.num_steps,
        frameskip=args.frameskip,
        keys_to_load=["pixels", "proprio", "state"],
        keys_to_cache=["proprio", "state"],
    )

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(dataset), size=min(args.num_samples, len(dataset)), replace=False)
    split = int(round(len(indices) * (1.0 - args.test_frac)))
    train_idx = np.sort(indices[:split])
    test_idx = np.sort(indices[split:])

    model = swm.wm.utils.load_pretrained(args.policy)
    x_train_r, y_train = encode_dataset(
        model, dataset, train_idx, batch_size=args.batch_size, img_size=args.img_size, device=args.device
    )
    x_test_r, y_test = encode_dataset(
        model, dataset, test_idx, batch_size=args.batch_size, img_size=args.img_size, device=args.device
    )

    reps = {
        "log_rate": (x_train_r, x_test_r),
        "rate": (torch.exp(x_train_r.clamp(-12.0, 5.0)), torch.exp(x_test_r.clamp(-12.0, 5.0))),
    }
    results = {
        "name": args.name,
        "policy": args.policy,
        "num_train_vectors": int(x_train_r.shape[0]),
        "num_test_vectors": int(x_test_r.shape[0]),
        "num_steps": args.num_steps,
        "frameskip": args.frameskip,
        "ridge": args.ridge,
        "representations": {},
    }
    for rep_name, (xtr, xte) in reps.items():
        results["representations"][rep_name] = probe_one_representation(
            rep_name, xtr, xte, y_train, y_test, args.ridge
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.name}_linear_probe.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
