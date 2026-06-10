from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from marmo.backimage_sequences import BackImagePaths, MarmoBackImageSequenceDataset, collate_marmo, normalize_level_specs
from marmo.faithful_train_utils import load_faithful_from_checkpoint, faithful_forward


def parse_args():
    p = argparse.ArgumentParser(description="Extract faithful Gaussian LeWM latents aligned to BackImage V1 responses")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--session", default=None)
    p.add_argument(
        "--sessions",
        default=None,
        help="Comma-separated sessions, or 'all-allen'. Defaults to checkpoint extra['sessions'] when available.",
    )
    p.add_argument("--data-root", default=None)
    p.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val", "all"])
    p.add_argument("--center-mode", default=None, choices=["dset", "gaze"])
    p.add_argument("--crop-sizes", default=None)
    p.add_argument("--pyramid-mode", default=None, choices=["raw", "gaussian", "hybrid_l0_gaussian", "laplacian"])
    p.add_argument("--blur-sigmas", default=None)
    p.add_argument("--laplacian-contrast", type=float, default=None)
    p.add_argument("--action-history", type=int, default=None)
    p.add_argument("--target-hz", type=int, default=None)
    p.add_argument("--split-mode", choices=["numpy", "torch"], default=None)
    p.add_argument("--robs-downsample-mode", choices=["sample", "sum"], default=None)
    p.add_argument("--covariate-downsample-mode", choices=["sample", "mean"], default=None)
    p.add_argument("--validity-downsample-mode", choices=["sample", "all"], default=None)
    p.add_argument("--pixel-normalization", choices=["unit", "visioncore"], default=None)
    p.add_argument("--dfs-mode", choices=["none", "valid_nlags", "visioncore"], default=None)
    p.add_argument("--dfs-valid-lags", type=int, default=None)
    p.add_argument("--dfs-missing-threshold", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--precompute-pixels", action="store_true")
    p.add_argument("--max-windows-per-split", type=int, default=8192)
    p.add_argument("--window-sample-mode", choices=["random", "first", "last"], default="first")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    return p.parse_args()


def parse_crop_sizes(text):
    return normalize_level_specs(text)


def to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def maybe_max(value: int) -> int | None:
    return None if int(value) <= 0 else int(value)


def concatenate_latent_arrays(arrays: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, values in arrays.items():
        if not values:
            continue
        if key in {"robs", "dfs"}:
            max_width = max(int(v.shape[-1]) if v.ndim > 1 else 1 for v in values)
            padded = []
            for value in values:
                arr = value[:, None] if value.ndim == 1 else value
                if int(arr.shape[-1]) == max_width:
                    padded.append(arr)
                    continue
                pad_width = [(0, 0)] * arr.ndim
                pad_width[-1] = (0, max_width - int(arr.shape[-1]))
                padded.append(np.pad(arr, pad_width, mode="constant", constant_values=0))
            out[key] = np.concatenate(padded, axis=0)
        else:
            out[key] = np.concatenate(values, axis=0)
    return out


def discover_sessions(session: str | None, sessions: str | None, data_root: str | Path, extra: dict) -> list[str]:
    if sessions is not None:
        spec = str(sessions).strip()
        if spec.lower() == "all-allen":
            root = Path(data_root) / "processed"
            found = [path.parents[1].name for path in sorted(root.glob("Allen_*/datasets/backimage.dset"))]
            if not found:
                raise FileNotFoundError(f"No Allen backimage.dset files found under {root}")
            return found
        out = [x.strip() for x in spec.split(",") if x.strip()]
        if not out:
            raise ValueError("--sessions did not contain any session names")
        return out
    if session is not None:
        return [session]
    extra_sessions = extra.get("sessions", None)
    if extra_sessions:
        return [str(x) for x in extra_sessions]
    return [str(extra.get("session", "Allen_2022-04-13"))]


def per_example_pred_loss(model, out: dict[str, torch.Tensor]) -> torch.Tensor:
    pred, target = model.loss_views(out["pred_hat"].detach(), out["target"].detach())
    if model.cfg.family == "gaussian":
        return (pred - target).pow(2).mean(dim=-1)
    raise ValueError("This extractor currently supports faithful Gaussian prediction loss")


def main():
    args = parse_args()
    model, cfg, extra = load_faithful_from_checkpoint(args.checkpoint, map_location="cpu")
    if cfg.family != "gaussian":
        raise ValueError("This extractor is currently intended for faithful Gaussian LeWM checkpoints")
    model.to(args.device).eval()
    data_root = args.data_root or extra.get("data_root", "/mnt/sata/YatesMarmoV1")
    sessions = discover_sessions(args.session, args.sessions, data_root, extra)
    crop_sizes = parse_crop_sizes(args.crop_sizes) if args.crop_sizes else normalize_level_specs(extra.get("crop_sizes", (51, 101, 201)))
    center_mode = args.center_mode or extra.get("center_mode", "dset")
    pyramid_mode = args.pyramid_mode or extra.get("pyramid_mode", "raw")
    blur_sigmas = args.blur_sigmas if args.blur_sigmas is not None else extra.get("blur_sigmas", None)
    laplacian_contrast = float(args.laplacian_contrast if args.laplacian_contrast is not None else extra.get("laplacian_contrast", 1.0))
    action_history = int(args.action_history if args.action_history is not None else extra.get("action_history", max(1, cfg.action_dim // 2)))
    target_hz = args.target_hz or int(extra.get("target_hz", 120))
    split_mode = args.split_mode or extra.get("split_mode", "numpy")
    robs_downsample_mode = args.robs_downsample_mode or extra.get("robs_downsample_mode", "sample")
    covariate_downsample_mode = args.covariate_downsample_mode or extra.get("covariate_downsample_mode", "sample")
    validity_downsample_mode = args.validity_downsample_mode or extra.get("validity_downsample_mode", "sample")
    pixel_normalization = args.pixel_normalization or extra.get("pixel_normalization", "unit")
    dfs_mode = args.dfs_mode or extra.get("dfs_mode", "none")
    dfs_valid_lags = int(args.dfs_valid_lags if args.dfs_valid_lags is not None else extra.get("dfs_valid_lags", 32))
    dfs_missing_threshold = float(
        args.dfs_missing_threshold if args.dfs_missing_threshold is not None else extra.get("dfs_missing_threshold", 45.0)
    )
    seq_len = cfg.history_size + cfg.num_preds
    arrays = {k: [] for k in [
        "eta", "code", "pred_hat", "target", "pred_loss", "robs", "dfs", "eyepos", "action", "t_bins",
        "trial_inds", "row_indices", "split", "session_id",
    ]}
    foveal_dim = int(getattr(cfg, "foveal_dim", 0) or 0)
    context_dim = int(getattr(cfg, "context_dim", 0) or 0)
    split_latent = foveal_dim > 0 and context_dim > 0
    if split_latent:
        for base in ["eta", "code", "pred_hat", "target"]:
            arrays[f"{base}_foveal"] = []
            arrays[f"{base}_context"] = []
    split_to_id = {"train": 0, "val": 1, "all": 2}
    session_unit_counts: list[int] = []
    for session_id, session in enumerate(sessions):
        unit_count = None
        for split in args.splits:
            ds = MarmoBackImageSequenceDataset(
                session_name=session,
                dset_path=BackImagePaths(session, Path(data_root)).dset_path,
                split=split,
                target_hz=target_hz,
                seq_len=seq_len,
                crop_sizes=crop_sizes,
                output_hw=cfg.img_hw,
                center_mode=center_mode,
                pyramid_mode=pyramid_mode,
                blur_sigmas=blur_sigmas,
                laplacian_contrast=laplacian_contrast,
                action_history=action_history,
                max_windows=maybe_max(args.max_windows_per_split),
                window_sample_mode=args.window_sample_mode,
                split_mode=split_mode,
                robs_downsample_mode=robs_downsample_mode,
                covariate_downsample_mode=covariate_downsample_mode,
                validity_downsample_mode=validity_downsample_mode,
                pixel_normalization=pixel_normalization,
                dfs_mode=dfs_mode,
                dfs_valid_lags=dfs_valid_lags,
                dfs_missing_threshold=dfs_missing_threshold,
            )
            if unit_count is None:
                unit_count = int(ds.cov["robs"].shape[1]) if ds.cov["robs"].ndim == 2 else 1
            if args.precompute_pixels:
                ds.precompute_pixels(verbose=True)
            loader = DataLoader(
                ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=collate_marmo,
                persistent_workers=args.num_workers > 0,
                pin_memory=args.device.startswith("cuda"),
            )
            for batch in loader:
                batch = to_device(batch, args.device)
                with torch.no_grad():
                    out = faithful_forward(model, batch)
                    eta = out["emb"].detach()
                    code = out["code"].detach()
                    pred = out["pred_hat"].detach()
                    target = out["target"].detach()
                    pred_loss = per_example_pred_loss(model, out).detach()
                    action_index = int(cfg.history_size) - 1
                for name, tensor in [
                    ("eta", eta[:, -1]),
                    ("code", code[:, -1]),
                    ("pred_hat", pred[:, -1]),
                    ("target", target[:, -1]),
                    ("pred_loss", pred_loss[:, -1]),
                    ("robs", batch["robs"][:, -1]),
                    ("dfs", batch["dfs"][:, -1]),
                    ("eyepos", batch["eyepos"][:, -1]),
                    ("action", batch["action"][:, action_index]),
                    ("t_bins", batch["t_bins"][:, -1]),
                    ("trial_inds", batch["trial_inds"][:, -1]),
                    ("row_indices", batch["row_indices"][:, -1]),
                ]:
                    arrays[name].append(tensor.detach().cpu().numpy())
                if split_latent:
                    for name, tensor in [
                        ("eta", eta[:, -1]),
                        ("code", code[:, -1]),
                        ("pred_hat", pred[:, -1]),
                        ("target", target[:, -1]),
                    ]:
                        arrays[f"{name}_foveal"].append(tensor[:, :foveal_dim].detach().cpu().numpy())
                        arrays[f"{name}_context"].append(tensor[:, foveal_dim:].detach().cpu().numpy())
                n_batch = int(eta.shape[0])
                arrays["split"].append(np.full((n_batch,), split_to_id[split], dtype=np.int64))
                arrays["session_id"].append(np.full((n_batch,), int(session_id), dtype=np.int64))
            print(f"extracted session={session} split={split} windows={len(ds)}")
        session_unit_counts.append(int(unit_count or 0))
    out_np = concatenate_latent_arrays(arrays)
    out_np["split_names"] = np.array(["train", "val", "all"])
    out_np["session_names"] = np.asarray(sessions)
    out_np["session_unit_counts"] = np.asarray(session_unit_counts, dtype=np.int64)
    out_np["session"] = np.array(["all_allen" if len(sessions) > 1 else sessions[0]])
    out_np["model_family"] = np.array([cfg.family])
    out_np["pyramid_mode"] = np.array([pyramid_mode])
    out_np["action_history"] = np.array([action_history])
    out_np["split_mode"] = np.array([split_mode])
    out_np["robs_downsample_mode"] = np.array([robs_downsample_mode])
    out_np["covariate_downsample_mode"] = np.array([covariate_downsample_mode])
    out_np["validity_downsample_mode"] = np.array([validity_downsample_mode])
    out_np["pixel_normalization"] = np.array([pixel_normalization])
    out_np["target_hz"] = np.array([target_hz])
    out_np["source_hz"] = np.array([240])
    out_np["downsample"] = np.array([240 // int(target_hz)])
    out_np["dfs_mode"] = np.array([dfs_mode])
    out_np["dfs_valid_lags"] = np.array([dfs_valid_lags])
    out_np["dfs_missing_threshold"] = np.array([dfs_missing_threshold])
    out_np["foveal_dim"] = np.array([foveal_dim])
    out_np["context_dim"] = np.array([context_dim])
    out_path = Path(args.out) if args.out else Path(args.checkpoint).with_name("latents.npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out_np)
    print(f"saved latents: {out_path}")
    for k, v in out_np.items():
        if hasattr(v, "shape"):
            print(f"  {k}: {v.shape}")


if __name__ == "__main__":
    main()
