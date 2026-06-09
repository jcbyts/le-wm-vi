from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from diagnostics import collapse_report
from marmo.amortized_train_utils import (
    AmortizedMarmoConfig,
    build_amortized_model,
    checkpoint_payload,
    compute_amortized_loss,
)
from marmo.backimage_sequences import MarmoBackImageSequenceDataset, collate_marmo
from marmo.train_fond_marmo import grad_norm, save_recon_panel


def parse_crop_sizes(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split(",") if x)


def parse_args():
    p = argparse.ArgumentParser(description="Train amortized encoder world model on marmoset BackImage")
    p.add_argument("--session", default="Allen_2022-04-13")
    p.add_argument("--family", choices=["poisson", "gaussian"], default="poisson")
    p.add_argument("--pred-loss", choices=["exact_kl", "quadratic_fisher"], default="exact_kl")
    p.add_argument("--center-mode", choices=["dset", "gaze"], default="dset")
    p.add_argument("--crop-sizes", default="51,101,201")
    p.add_argument("--output-hw", type=int, default=64)
    p.add_argument("--target-hz", type=int, default=120)
    p.add_argument("--history-size", type=int, default=3)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--encoder-width", type=int, default=64)
    p.add_argument("--predictor-depth", type=int, default=2)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--recon-weight", type=float, default=1.0)
    p.add_argument("--sigreg-weight", type=float, default=0.09)
    p.add_argument("--sigreg-target-rate", type=float, default=1.0)
    p.add_argument("--sigreg-num-proj", type=int, default=1024)
    p.add_argument("--sigreg-knots", type=int, default=17)
    p.add_argument("--poisson-log-lo", type=float, default=-12.0)
    p.add_argument("--poisson-log-hi", type=float, default=5.0)
    p.add_argument("--fixed-unit-variance", action="store_true")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-train-windows", type=int, default=0)
    p.add_argument("--max-val-windows", type=int, default=0)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--log-interval", type=int, default=25)
    p.add_argument("--val-interval", type=int, default=100)
    p.add_argument("--val-batches", type=int, default=4)
    p.add_argument("--save-interval", type=int, default=500)
    p.add_argument("--seed", type=int, default=1002)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--outdir", default="/home/tejas/le-wm-vi/outputs/marmo_amortized")
    return p.parse_args()


def maybe_max(value: int) -> int | None:
    return None if int(value) <= 0 else int(value)


def to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def scalar(x) -> float:
    return float(x.detach().cpu().item()) if torch.is_tensor(x) else float(x)


def summarize(model, out, phase: str, step: int, grad: float | None = None):
    eta = out["eta"].detach()
    code = out["code"].detach()
    row = {
        "phase": phase,
        "step": int(step),
        "loss": scalar(out["loss"]),
        "recon_loss": scalar(out["recon_loss"]),
        "pred_loss": scalar(out["pred_loss"]),
        "sigreg_loss": scalar(out["sigreg_loss"]),
        "kl_exact": scalar(out["kl_exact"]),
        "fisher_quad": scalar(out["fisher_quad"]),
        "exact_quad_ratio": scalar(out["exact_quad_ratio"]),
        "eta_std": scalar(out["eta_std"]),
        "code_mean": float(code.mean().cpu()),
        "code_std": float(code.std().cpu()),
        "grad_norm": float("nan") if grad is None else float(grad),
    }
    for key, value in collapse_report(eta).items():
        row[f"collapse_{key}"] = float(value)
    for key, value in model.head.param_stats(eta).items():
        row[key] = float(value)
    return row


def validate(model, loader, cfg, device, max_batches: int):
    rows = []
    first = None
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            batch = to_device(batch, device)
            out = compute_amortized_loss(model, batch, cfg)
            rows.append(summarize(model, out, "val", 0))
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
    preferred = ["phase", "step", "loss", "recon_loss", "pred_loss", "sigreg_loss", "kl_exact", "fisher_quad"]
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
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    for ax, key in zip(axes.ravel(), ["loss", "recon_loss", "pred_loss", "collapse_eff_rank_frac"]):
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
    crop_sizes = parse_crop_sizes(args.crop_sizes)
    seq_len = args.history_size + 1
    train_ds = MarmoBackImageSequenceDataset(
        session_name=args.session,
        split="train",
        seed=args.seed,
        target_hz=args.target_hz,
        seq_len=seq_len,
        crop_sizes=crop_sizes,
        output_hw=args.output_hw,
        center_mode=args.center_mode,
        max_windows=maybe_max(args.max_train_windows),
        stride=args.stride,
    )
    val_ds = MarmoBackImageSequenceDataset(
        session_name=args.session,
        split="val",
        seed=args.seed,
        target_hz=args.target_hz,
        seq_len=seq_len,
        crop_sizes=crop_sizes,
        output_hw=args.output_hw,
        center_mode=args.center_mode,
        max_windows=maybe_max(args.max_val_windows),
        stride=args.stride,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_marmo, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_marmo)

    cfg = AmortizedMarmoConfig(
        family=args.family,
        pred_loss=args.pred_loss,
        embed_dim=args.embed_dim,
        img_ch=len(crop_sizes),
        img_hw=args.output_hw,
        action_dim=train_ds.action_dim,
        history_size=args.history_size,
        predictor_depth=args.predictor_depth,
        encoder_width=args.encoder_width,
        beta=args.beta,
        recon_weight=args.recon_weight,
        sigreg_weight=args.sigreg_weight,
        sigreg_target_rate=args.sigreg_target_rate,
        sigreg_num_proj=args.sigreg_num_proj,
        sigreg_knots=args.sigreg_knots,
        poisson_log_lo=args.poisson_log_lo,
        poisson_log_hi=args.poisson_log_hi,
        fixed_unit_variance=args.fixed_unit_variance,
    )
    model = build_amortized_model(cfg).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    outdir = Path(args.outdir) / f"{args.session}_amortized_{args.family}_{args.pred_loss}_{args.center_mode}"
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "run_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"train windows={len(train_ds)} val windows={len(val_ds)} device={args.device}")
    if train_ds.missing_image_trials:
        print(f"skipping {len(train_ds.missing_image_trials)} trials with missing source images: {sorted(set(train_ds.missing_image_trials.values()))}")
    print(cfg)
    metric_rows = []
    best_val = float("inf")

    def save_ckpt(name, step, extra=None):
        payload_extra = {
            "session": args.session,
            "crop_sizes": crop_sizes,
            "center_mode": args.center_mode,
            "target_hz": args.target_hz,
            "train_windows": len(train_ds),
            "val_windows": len(val_ds),
            "step": step,
            "stride": args.stride,
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
            out = compute_amortized_loss(model, batch, cfg)
            if not torch.isfinite(out["loss_for_backward"]):
                raise FloatingPointError(f"non-finite loss at step {step}")
            out["loss_for_backward"].backward()
            gnorm = grad_norm(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            row = summarize(model, out, "train", step, gnorm)
            metric_rows.append(row)
            if step == 1 or step % args.log_interval == 0:
                print(
                    f"step {step:04d} loss={row['loss']:.5f} recon={row['recon_loss']:.5f} "
                    f"pred={row['pred_loss']:.5f} sigreg={row['sigreg_loss']:.5f} "
                    f"rank={row['collapse_eff_rank_frac']:.3f} code_std={row['code_std']:.3f}"
                )
            if args.val_interval > 0 and step % args.val_interval == 0:
                val_row, preview = validate(model, val_loader, cfg, args.device, args.val_batches)
                if val_row:
                    val_row["step"] = step
                    metric_rows.append(val_row)
                    print(
                        f"val  {step:04d} loss={val_row.get('loss', np.nan):.5f} recon={val_row.get('recon_loss', np.nan):.5f} "
                        f"pred={val_row.get('pred_loss', np.nan):.5f} sigreg={val_row.get('sigreg_loss', np.nan):.5f} "
                        f"rank={val_row.get('collapse_eff_rank_frac', np.nan):.3f}"
                    )
                    if preview is not None:
                        save_recon_panel(outdir / "reconstruction_panel.png", preview["batch"], preview["out"]["recon"])
                    if val_row.get("loss", float("inf")) < best_val:
                        best_val = val_row["loss"]
                        save_ckpt("best.pt", step, {"best_val_loss": best_val})
                write_metrics(outdir / "metrics.csv", metric_rows)
                plot_curves(outdir / "training_curves.png", metric_rows)
            if args.save_interval > 0 and step % args.save_interval == 0:
                print(f"saved checkpoint: {save_ckpt(f'step{step:06d}.pt', step, {'best_val_loss': best_val})}")
            if step >= args.max_steps:
                break

    val_row, preview = validate(model, val_loader, cfg, args.device, max(1, args.val_batches))
    if val_row:
        val_row["step"] = step
        metric_rows.append(val_row)
        print("validation preview:")
        for key in ["loss", "recon_loss", "pred_loss", "sigreg_loss", "collapse_eff_rank_frac", "collapse_batch_var_median"]:
            if key in val_row:
                print(f"  {key}: {val_row[key]:.6f}")
        if preview is not None:
            save_recon_panel(outdir / "reconstruction_panel.png", preview["batch"], preview["out"]["recon"])
    write_metrics(outdir / "metrics.csv", metric_rows)
    plot_curves(outdir / "training_curves.png", metric_rows)
    ckpt = save_ckpt("last.pt", step, {"best_val_loss": best_val})
    print(f"saved checkpoint: {ckpt}")
    print(f"saved metrics: {outdir / 'metrics.csv'}")
    print(f"saved reconstruction panel: {outdir / 'reconstruction_panel.png'}")


if __name__ == "__main__":
    main()
