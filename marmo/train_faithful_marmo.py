from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from marmo.backimage_sequences import BackImagePaths, MarmoBackImageSequenceDataset, collate_marmo, normalize_level_specs
from marmo.faithful_train_utils import (
    FaithfulMarmoConfig,
    build_faithful_model,
    checkpoint_payload,
    faithful_forward,
    prediction_loss,
)
from marmo.train_fond_marmo import grad_norm


def parse_crop_sizes(text: str):
    return normalize_level_specs(text)


def parse_args():
    p = argparse.ArgumentParser(description="Train faithful Gaussian LeWM on BackImage without pixel reconstruction")
    p.add_argument("--session", default="Allen_2022-04-13")
    p.add_argument(
        "--sessions",
        default=None,
        help="Comma-separated sessions, or 'all-allen' for all Allen backimage.dset sessions.",
    )
    p.add_argument("--data-root", default="/mnt/sata/YatesMarmoV1")
    p.add_argument("--family", choices=["gaussian"], default="gaussian")
    p.add_argument("--center-mode", choices=["dset", "gaze"], default="dset")
    p.add_argument("--crop-sizes", default="51,201,601,1201")
    p.add_argument("--pyramid-mode", choices=["raw", "gaussian", "hybrid_l0_gaussian", "laplacian"], default="raw")
    p.add_argument("--blur-sigmas", default=None)
    p.add_argument("--laplacian-contrast", type=float, default=1.0)
    p.add_argument("--output-hw", type=int, default=51)
    p.add_argument("--target-hz", type=int, default=120)
    p.add_argument("--split-mode", choices=["numpy", "torch"], default="torch")
    p.add_argument("--robs-downsample-mode", choices=["sample", "sum"], default="sum")
    p.add_argument("--covariate-downsample-mode", choices=["sample", "mean"], default="mean")
    p.add_argument("--validity-downsample-mode", choices=["sample", "all"], default="all")
    p.add_argument("--pixel-normalization", choices=["unit", "visioncore"], default="visioncore")
    p.add_argument("--dfs-mode", choices=["none", "valid_nlags", "visioncore"], default="visioncore")
    p.add_argument("--dfs-valid-lags", type=int, default=32)
    p.add_argument("--dfs-missing-threshold", type=float, default=45.0)
    p.add_argument("--history-size", type=int, default=3)
    p.add_argument("--num-preds", type=int, default=1)
    p.add_argument("--embed-dim", type=int, default=192)
    p.add_argument("--target-mode", choices=["full", "l0"], default="full")
    p.add_argument("--target-level", type=int, default=0)
    p.add_argument(
        "--masked-channel-value",
        type=float,
        default=None,
        help="Fill value for non-target channels in target_mode=l0. Defaults to 0.5 for unit pixels, 0.0 for visioncore pixels.",
    )
    p.add_argument("--foveal-dim", type=int, default=0)
    p.add_argument("--context-dim", type=int, default=0)
    p.add_argument("--foveal-level", type=int, default=0)
    p.add_argument("--encoder-width", type=int, default=64)
    p.add_argument("--encoder-kind", choices=["conv", "alexnet_v1", "voneblock", "vonealexnet"], default="conv")
    p.add_argument("--no-neural-pretrained", action="store_true")
    p.add_argument("--train-neural-frontend", action="store_true")
    p.add_argument("--neural-resize-hw", type=int, default=224)
    p.add_argument("--neural-feature-index", type=int, default=2)
    p.add_argument("--neural-pool-hw", type=int, default=1)
    p.add_argument("--neural-pixel-mode", choices=["unit", "visioncore", "auto"], default=None)
    p.add_argument("--vone-simple-channels", type=int, default=128)
    p.add_argument("--vone-complex-channels", type=int, default=128)
    p.add_argument("--vone-ksize", type=int, default=25)
    p.add_argument("--vone-stride", type=int, default=4)
    p.add_argument("--vone-visual-degrees", type=float, default=8.0)
    p.add_argument("--vone-sf-corr", type=float, default=0.75)
    p.add_argument("--vone-sf-max", type=float, default=9.0)
    p.add_argument("--vone-sf-min", type=float, default=0.0)
    p.add_argument("--vone-noise-mode", choices=["none", "gaussian", "neuronal"], default="none")
    p.add_argument("--predictor-depth", type=int, default=6)
    p.add_argument("--predictor-heads", type=int, default=8)
    p.add_argument("--predictor-mlp-dim", type=int, default=1024)
    p.add_argument("--predictor-dim-head", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--emb-dropout", type=float, default=0.0)
    p.add_argument("--sigreg-weight", type=float, default=0.09)
    p.add_argument("--sigreg-num-proj", type=int, default=1024)
    p.add_argument("--sigreg-knots", type=int, default=17)
    p.add_argument("--action-scale", type=float, default=1.0)
    p.add_argument("--action-history", type=int, default=1)
    p.add_argument("--action-smoothed-dim", type=int, default=8)
    p.add_argument("--action-mlp-scale", type=int, default=4)
    p.add_argument("--no-projectors", action="store_true")
    p.add_argument("--projector-hidden-dim", type=int, default=2048)
    p.add_argument("--projector-norm", choices=["batchnorm", "layernorm", "none"], default="layernorm")
    p.add_argument("--detach-target", action="store_true")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--precompute-pixels", action="store_true")
    p.add_argument("--max-train-windows", type=int, default=0)
    p.add_argument("--max-val-windows", type=int, default=0)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--min-lr-scale", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--val-interval", type=int, default=250)
    p.add_argument("--val-batches", type=int, default=8)
    p.add_argument("--save-interval", type=int, default=1000)
    p.add_argument("--saccade-threshold-deg-s", type=float, default=25.0)
    p.add_argument("--seed", type=int, default=1002)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--outdir", default="/home/tejas/le-wm-vi/outputs/marmo_faithful_gaussian")
    return p.parse_args()


def maybe_max(value: int) -> int | None:
    return None if int(value) <= 0 else int(value)


def discover_sessions(args) -> list[str]:
    if args.sessions is None:
        return [args.session]
    if args.sessions.strip().lower() == "all-allen":
        root = Path(args.data_root) / "processed"
        sessions = [path.parents[1].name for path in sorted(root.glob("Allen_*/datasets/backimage.dset"))]
        if not sessions:
            raise FileNotFoundError(f"No Allen backimage.dset files found under {root}")
        return sessions
    sessions = [x.strip() for x in args.sessions.split(",") if x.strip()]
    if not sessions:
        raise ValueError("--sessions did not contain any session names")
    return sessions


def per_session_cap(total: int, n_sessions: int) -> int | None:
    if int(total) <= 0:
        return None
    return int(math.ceil(int(total) / max(1, int(n_sessions))))


def to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def scalar(x) -> float:
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return float(x)


def lr_multiplier(step: int, max_steps: int, warmup_steps: int, min_scale: float) -> float:
    step = max(1, int(step))
    warmup_steps = max(0, int(warmup_steps))
    min_scale = float(min(max(min_scale, 0.0), 1.0))
    if warmup_steps > 0 and step <= warmup_steps:
        return step / float(warmup_steps)
    denom = max(1, int(max_steps) - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / float(denom)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_scale + (1.0 - min_scale) * cosine


def per_example_pred_loss(model, out: dict[str, torch.Tensor]) -> torch.Tensor:
    if "pred_loss_per_example" in out:
        return out["pred_loss_per_example"].detach()
    pred, target = model.loss_views(out["pred_hat"].detach(), out["target"].detach())
    if model.cfg.family == "gaussian":
        return (pred - target).pow(2).mean(dim=-1)
    raise ValueError("train_faithful_marmo currently runs Gaussian only")


def speed_split_metrics(
    model,
    batch: dict[str, torch.Tensor],
    out: dict[str, torch.Tensor],
    threshold_deg_s: float,
    target_hz: float,
) -> dict[str, float]:
    loss_bt = per_example_pred_loss(model, out)
    action = batch["action"][:, : model.cfg.history_size].detach()
    speed = torch.linalg.norm(action.float()[..., :2], dim=-1) * float(target_hz)
    fixation = speed < float(threshold_deg_s)
    saccade = ~fixation
    metrics = {
        "speed_mean_deg_s": float(speed.mean().cpu()),
        "speed_p95_deg_s": float(torch.quantile(speed.flatten(), 0.95).cpu()),
        "fixation_frac": float(fixation.float().mean().cpu()),
    }
    if fixation.any():
        metrics["fixation_pred_loss"] = float(loss_bt[fixation].mean().cpu())
    if saccade.any():
        metrics["saccade_pred_loss"] = float(loss_bt[saccade].mean().cpu())
    return metrics


def summarize(
    model,
    batch: dict[str, torch.Tensor],
    out,
    phase: str,
    step: int,
    threshold_deg_s: float,
    target_hz: float,
    grad: float | None = None,
):
    row = {
        "phase": phase,
        "step": int(step),
        "loss": scalar(out["loss"]),
        "pred_loss": scalar(out["pred_loss"]),
        "reg_loss": scalar(out["reg_loss"]),
        "no_action_loss": scalar(out["no_action_loss"]),
        "identity_loss": scalar(out["identity_loss"]),
        "action_gain": scalar(out["action_gain"]),
        "identity_gain": scalar(out["identity_gain"]),
        "emb_std": scalar(out["emb_std"]),
        "code_mean": scalar(out["code_mean"]),
        "code_std": scalar(out["code_std"]),
        "grad_norm": float("nan") if grad is None else float(grad),
    }
    for key, value in out.items():
        if key.startswith(("code_collapse_", "emb_collapse_")):
            row[key] = scalar(value)
    row.update(speed_split_metrics(model, batch, out, threshold_deg_s, target_hz))
    return row


def validate(model, loader, device, max_batches: int, threshold_deg_s: float, target_hz: float):
    rows = []
    first = None
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            batch = to_device(batch, device)
            out = faithful_forward(model, batch)
            rows.append(summarize(model, batch, out, "val", 0, threshold_deg_s, target_hz))
            if first is None:
                first = {"batch": batch, "out": out}
    model.train()
    if not rows:
        return {}, first
    keys = sorted(set().union(*(r.keys() for r in rows)) - {"phase"})
    mean = {"phase": "val"}
    for key in keys:
        vals = [r[key] for r in rows if key in r and isinstance(r[key], (int, float)) and np.isfinite(r[key])]
        if vals:
            mean[key] = float(np.mean(vals))
    return mean, first


def write_metrics(path: Path, rows: list[dict]):
    if not rows:
        return
    keys = sorted(set().union(*(row.keys() for row in rows)))
    preferred = [
        "phase",
        "step",
        "loss",
        "pred_loss",
        "reg_loss",
        "fixation_pred_loss",
        "saccade_pred_loss",
        "action_gain",
        "identity_gain",
        "code_collapse_eff_rank_frac",
        "code_collapse_batch_var_median",
    ]
    keys = [k for k in preferred if k in keys] + [k for k in keys if k not in preferred]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(path: Path, rows: list[dict]):
    train = [r for r in rows if r.get("phase") == "train"]
    val = [r for r in rows if r.get("phase") == "val"]
    if not train:
        return
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    keys = [
        "loss",
        "pred_loss",
        "reg_loss",
        "fixation_pred_loss",
        "saccade_pred_loss",
        "code_collapse_eff_rank_frac",
    ]
    for ax, key in zip(axes.ravel(), keys):
        ax.plot([r["step"] for r in train], [r.get(key, np.nan) for r in train], color="0.15", lw=1, label="train")
        if val:
            ax.plot([r["step"] for r in val], [r.get(key, np.nan) for r in val], "o-", color="C3", ms=3, label="val")
        ax.set_title(key)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    sessions = discover_sessions(args)
    crop_sizes = parse_crop_sizes(args.crop_sizes)
    seq_len = args.history_size + args.num_preds
    train_cap = per_session_cap(args.max_train_windows, len(sessions))
    val_cap = per_session_cap(args.max_val_windows, len(sessions))

    def make_ds(session_name: str, split: str, cap: int | None):
        return MarmoBackImageSequenceDataset(
            session_name=session_name,
            dset_path=BackImagePaths(session_name, Path(args.data_root)).dset_path,
            split=split,
            seed=args.seed,
            target_hz=args.target_hz,
            seq_len=seq_len,
            crop_sizes=crop_sizes,
            output_hw=args.output_hw,
            center_mode=args.center_mode,
            pyramid_mode=args.pyramid_mode,
            blur_sigmas=args.blur_sigmas,
            laplacian_contrast=args.laplacian_contrast,
            action_history=args.action_history,
            max_windows=cap,
            stride=args.stride,
            split_mode=args.split_mode,
            robs_downsample_mode=args.robs_downsample_mode,
            covariate_downsample_mode=args.covariate_downsample_mode,
            validity_downsample_mode=args.validity_downsample_mode,
            pixel_normalization=args.pixel_normalization,
            dfs_mode=args.dfs_mode,
            dfs_valid_lags=args.dfs_valid_lags,
            dfs_missing_threshold=args.dfs_missing_threshold,
        )

    train_parts = []
    val_parts = []
    t_build = time.time()
    for i, session in enumerate(sessions, start=1):
        print(f"[dataset {i}/{len(sessions)}] building {session} train...", flush=True)
        part_train = make_ds(session, "train", train_cap)
        print(f"[dataset {i}/{len(sessions)}] {session} train windows={len(part_train)}", flush=True)
        print(f"[dataset {i}/{len(sessions)}] building {session} val...", flush=True)
        part_val = make_ds(session, "val", val_cap)
        print(f"[dataset {i}/{len(sessions)}] {session} val windows={len(part_val)}", flush=True)
        train_parts.append(part_train)
        val_parts.append(part_val)
    print(f"dataset construction finished in {time.time() - t_build:.1f}s", flush=True)
    train_ds = train_parts[0] if len(train_parts) == 1 else ConcatDataset(train_parts)
    val_ds = val_parts[0] if len(val_parts) == 1 else ConcatDataset(val_parts)
    if args.precompute_pixels:
        for ds in train_parts:
            ds.precompute_pixels(verbose=True)
        for ds in val_parts:
            ds.precompute_pixels(verbose=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_marmo,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        pin_memory=args.device.startswith("cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_marmo,
        persistent_workers=args.num_workers > 0,
        pin_memory=args.device.startswith("cuda"),
    )
    masked_channel_value = (
        float(args.masked_channel_value)
        if args.masked_channel_value is not None
        else (0.0 if args.pixel_normalization == "visioncore" else 0.5)
    )
    cfg = FaithfulMarmoConfig(
        family=args.family,
        embed_dim=args.embed_dim,
        img_ch=len(crop_sizes),
        img_hw=args.output_hw,
        action_dim=train_parts[0].action_dim,
        action_scale=args.action_scale,
        history_size=args.history_size,
        num_preds=args.num_preds,
        target_mode=args.target_mode,
        target_level=args.target_level,
        masked_channel_value=masked_channel_value,
        foveal_dim=args.foveal_dim,
        context_dim=args.context_dim,
        foveal_level=args.foveal_level,
        encoder_width=args.encoder_width,
        encoder_kind=args.encoder_kind,
        neural_pretrained=not args.no_neural_pretrained,
        neural_freeze_frontend=not args.train_neural_frontend,
        neural_resize_hw=args.neural_resize_hw,
        neural_feature_index=args.neural_feature_index,
        neural_pool_hw=args.neural_pool_hw,
        neural_pixel_mode=args.neural_pixel_mode or args.pixel_normalization,
        vone_simple_channels=args.vone_simple_channels,
        vone_complex_channels=args.vone_complex_channels,
        vone_ksize=args.vone_ksize,
        vone_stride=args.vone_stride,
        vone_visual_degrees=args.vone_visual_degrees,
        vone_sf_corr=args.vone_sf_corr,
        vone_sf_max=args.vone_sf_max,
        vone_sf_min=args.vone_sf_min,
        vone_noise_mode=args.vone_noise_mode,
        predictor_depth=args.predictor_depth,
        predictor_heads=args.predictor_heads,
        predictor_mlp_dim=args.predictor_mlp_dim,
        predictor_dim_head=args.predictor_dim_head,
        dropout=args.dropout,
        emb_dropout=args.emb_dropout,
        sigreg_weight=args.sigreg_weight,
        sigreg_num_proj=args.sigreg_num_proj,
        sigreg_knots=args.sigreg_knots,
        action_smoothed_dim=args.action_smoothed_dim,
        action_mlp_scale=args.action_mlp_scale,
        use_projectors=not args.no_projectors,
        projector_hidden_dim=args.projector_hidden_dim,
        projector_norm=args.projector_norm,
        detach_target=True if args.detach_target else None,
    )
    model = build_faithful_model(cfg).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lr_lambda=lambda s: lr_multiplier(
            s + 1,
            max_steps=args.max_steps,
            warmup_steps=args.warmup_steps,
            min_scale=args.min_lr_scale,
        ),
    )
    run_session_name = args.session if len(sessions) == 1 else "all_allen"
    outdir = Path(args.outdir) / f"{run_session_name}_faithful_{args.family}_{args.center_mode}"
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "run_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"sessions={len(sessions)} train windows={len(train_ds)} val windows={len(val_ds)} device={args.device}")
    for ds in train_parts:
        if ds.missing_image_trials:
            print(f"{ds.session_name}: skipping {len(ds.missing_image_trials)} trials with missing source images")
    print(cfg)
    metric_rows = []
    best_val = float("inf")

    def save_ckpt(name, step, extra=None):
        payload_extra = {
            "session": run_session_name,
            "sessions": sessions,
            "data_root": args.data_root,
            "crop_sizes": crop_sizes,
            "center_mode": args.center_mode,
            "pyramid_mode": args.pyramid_mode,
            "blur_sigmas": args.blur_sigmas,
            "laplacian_contrast": args.laplacian_contrast,
            "target_hz": args.target_hz,
            "split_mode": args.split_mode,
            "robs_downsample_mode": args.robs_downsample_mode,
            "covariate_downsample_mode": args.covariate_downsample_mode,
            "validity_downsample_mode": args.validity_downsample_mode,
            "pixel_normalization": args.pixel_normalization,
            "masked_channel_value": masked_channel_value,
            "dfs_mode": args.dfs_mode,
            "dfs_valid_lags": args.dfs_valid_lags,
            "dfs_missing_threshold": args.dfs_missing_threshold,
            "action_history": args.action_history,
            "encoder_kind": args.encoder_kind,
            "neural_pretrained": not args.no_neural_pretrained,
            "neural_freeze_frontend": not args.train_neural_frontend,
            "neural_resize_hw": args.neural_resize_hw,
            "neural_feature_index": args.neural_feature_index,
            "neural_pool_hw": args.neural_pool_hw,
            "neural_pixel_mode": args.neural_pixel_mode or args.pixel_normalization,
            "vone_simple_channels": args.vone_simple_channels,
            "vone_complex_channels": args.vone_complex_channels,
            "vone_ksize": args.vone_ksize,
            "vone_stride": args.vone_stride,
            "vone_visual_degrees": args.vone_visual_degrees,
            "vone_sf_corr": args.vone_sf_corr,
            "vone_sf_max": args.vone_sf_max,
            "vone_sf_min": args.vone_sf_min,
            "vone_noise_mode": args.vone_noise_mode,
            "train_windows": len(train_ds),
            "val_windows": len(val_ds),
            "step": step,
            "stride": args.stride,
            "saccade_threshold_deg_s": args.saccade_threshold_deg_s,
        }
        if extra:
            payload_extra.update(extra)
        path = outdir / name
        torch.save(checkpoint_payload(model, cfg, payload_extra), path)
        return path

    step = 0
    model.train()
    while step < args.max_steps:
        for batch in train_loader:
            step += 1
            batch = to_device(batch, args.device)
            opt.zero_grad(set_to_none=True)
            out = faithful_forward(model, batch)
            if not torch.isfinite(out["loss_for_backward"]):
                raise FloatingPointError(f"non-finite loss at step {step}")
            out["loss_for_backward"].backward()
            gnorm = grad_norm(model)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            scheduler.step()
            row = summarize(model, batch, out, "train", step, args.saccade_threshold_deg_s, args.target_hz, gnorm)
            row["lr"] = float(scheduler.get_last_lr()[0])
            metric_rows.append(row)
            if step == 1 or step % args.log_interval == 0:
                print(
                    f"step {step:05d} loss={row['loss']:.5f} pred={row['pred_loss']:.5f} "
                    f"reg={row['reg_loss']:.5f} fix={row.get('fixation_pred_loss', np.nan):.5f} "
                    f"sac={row.get('saccade_pred_loss', np.nan):.5f} "
                    f"rank={row.get('code_collapse_eff_rank_frac', np.nan):.3f} "
                    f"action_gain={row['action_gain']:.5f}"
                )
            if args.val_interval > 0 and step % args.val_interval == 0:
                val_row, _preview = validate(
                    model,
                    val_loader,
                    args.device,
                    args.val_batches,
                    args.saccade_threshold_deg_s,
                    args.target_hz,
                )
                if val_row:
                    val_row["step"] = step
                    metric_rows.append(val_row)
                    print(
                        f"val  {step:05d} loss={val_row.get('loss', np.nan):.5f} pred={val_row.get('pred_loss', np.nan):.5f} "
                        f"reg={val_row.get('reg_loss', np.nan):.5f} fix={val_row.get('fixation_pred_loss', np.nan):.5f} "
                        f"sac={val_row.get('saccade_pred_loss', np.nan):.5f} "
                        f"rank={val_row.get('code_collapse_eff_rank_frac', np.nan):.3f} "
                        f"action_gain={val_row.get('action_gain', np.nan):.5f}"
                    )
                    if val_row.get("loss", float("inf")) < best_val:
                        best_val = val_row["loss"]
                        save_ckpt("best.pt", step, {"best_val_loss": best_val})
                write_metrics(outdir / "metrics.csv", metric_rows)
                plot_curves(outdir / "training_curves.png", metric_rows)
            if args.save_interval > 0 and step % args.save_interval == 0:
                print(f"saved checkpoint: {save_ckpt(f'step{step:06d}.pt', step, {'best_val_loss': best_val})}")
            if step >= args.max_steps:
                break

    val_row, _preview = validate(
        model,
        val_loader,
        args.device,
        max(1, args.val_batches),
        args.saccade_threshold_deg_s,
        args.target_hz,
    )
    if val_row:
        val_row["step"] = step
        metric_rows.append(val_row)
        print("validation preview:")
        for key in [
            "loss",
            "pred_loss",
            "reg_loss",
            "fixation_pred_loss",
            "saccade_pred_loss",
            "code_collapse_eff_rank_frac",
            "code_collapse_batch_var_median",
            "action_gain",
        ]:
            if key in val_row:
                print(f"  {key}: {val_row[key]:.6f}")
    write_metrics(outdir / "metrics.csv", metric_rows)
    plot_curves(outdir / "training_curves.png", metric_rows)
    ckpt = save_ckpt("last.pt", step, {"best_val_loss": best_val})
    print(f"saved checkpoint: {ckpt}")
    print(f"saved metrics: {outdir / 'metrics.csv'}")
    print(f"saved training curves: {outdir / 'training_curves.png'}")


if __name__ == "__main__":
    main()
