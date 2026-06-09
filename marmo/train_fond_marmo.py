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
from marmo.backimage_sequences import MarmoBackImageSequenceDataset, collate_marmo
from marmo.fond_train_utils import (
    FondMarmoConfig,
    build_fond_model,
    checkpoint_payload,
    compute_fond_loss,
)


def parse_crop_sizes(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split(",") if x)


def parse_args():
    p = argparse.ArgumentParser(description="Train a tiny/real FOND world model on marmoset BackImage")
    p.add_argument("--session", default="Allen_2022-04-13")
    p.add_argument("--family", choices=["poisson", "gaussian"], default="poisson")
    p.add_argument("--pred-loss", choices=["exact_kl", "quadratic_fisher"], default="exact_kl")
    p.add_argument("--center-mode", choices=["dset", "gaze"], default="dset")
    p.add_argument("--crop-sizes", default="51,101,201")
    p.add_argument("--output-hw", type=int, default=64)
    p.add_argument("--target-hz", type=int, default=120)
    p.add_argument("--history-size", type=int, default=3)
    p.add_argument("--num-preds", type=int, default=1)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--predictor-depth", type=int, default=2)
    p.add_argument("--k-inner", type=int, default=2)
    p.add_argument("--infer-backprop", action="store_true")
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--infer-lr", type=float, default=None)
    p.add_argument("--infer-grad-clip", type=float, default=1.0)
    p.add_argument("--recon-weight", type=float, default=1.0)
    p.add_argument("--sigreg-weight", type=float, default=0.0)
    p.add_argument("--sigreg-target-rate", type=float, default=1.0)
    p.add_argument("--sigreg-num-proj", type=int, default=1024)
    p.add_argument("--sigreg-knots", type=int, default=17)
    p.add_argument("--poisson-log-lo", type=float, default=-12.0)
    p.add_argument("--poisson-log-hi", type=float, default=5.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-train-windows", type=int, default=0, help="0 means use all valid windows")
    p.add_argument("--max-val-windows", type=int, default=0, help="0 means use all valid windows")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--val-interval", type=int, default=100)
    p.add_argument("--val-batches", type=int, default=4)
    p.add_argument("--save-interval", type=int, default=500)
    p.add_argument("--recon-panel-interval", type=int, default=100)
    p.add_argument("--seed", type=int, default=1002)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--outdir", default="/home/tejas/le-wm-vi/outputs/marmo_fond")
    return p.parse_args()


def to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def _maybe_max_windows(value: int) -> int | None:
    return None if value is None or int(value) <= 0 else int(value)


def _scalar(value) -> float:
    if value is None:
        return float("nan")
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def save_recon_panel(path: Path, batch, recon, n=4):
    n = min(n, batch["pixels"].shape[0])
    n_ch = int(batch["pixels"].shape[2])
    fig, axes = plt.subplots(n, n_ch * 3, figsize=(2.0 * n_ch * 3, 1.8 * n))
    if n == 1:
        axes = axes[None, :]
    x = batch["pixels"][:n, -1].detach().cpu()
    y = recon[:n, -1].detach().cpu().clamp(0, 1)
    for i in range(n):
        for c in range(n_ch):
            diff = (x[i, c] - y[i, c]).abs()
            panels = [(x[i, c], f"target L{c}", "gray", 0, 1), (y[i, c], f"recon L{c}", "gray", 0, 1), (diff, f"|diff| L{c}", "magma", 0, float(diff.max().clamp_min(1e-6)))]
            for j, (arr, title, cmap, vmin, vmax) in enumerate(panels):
                ax = axes[i, c * 3 + j]
                ax.imshow(arr.numpy(), cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
                if i == 0:
                    ax.set_title(title, fontsize=8)
                ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_metric_plot(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    steps = np.asarray([r["step"] for r in rows if r.get("phase") == "train"], dtype=np.float64)
    train_rows = [r for r in rows if r.get("phase") == "train"]
    val_rows = [r for r in rows if r.get("phase") == "val"]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=False)
    for ax, key, title in [
        (axes[0, 0], "loss", "loss"),
        (axes[0, 1], "recon_loss", "reconstruction"),
        (axes[1, 0], "pred_loss", "prediction divergence"),
        (axes[1, 1], "sigreg_loss", "SIGReg"),
    ]:
        if train_rows:
            ax.plot(steps, [r.get(key, np.nan) for r in train_rows], color="0.15", lw=1, label="train")
        if val_rows:
            ax.plot([r["step"] for r in val_rows], [r.get(key, np.nan) for r in val_rows], "o-", color="C3", ms=3, label="val")
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


@torch.no_grad()
def grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        total += float(param.grad.detach().pow(2).sum().cpu())
    return total ** 0.5


def summarize_out(model, out, phase: str, step: int, grad: float | None = None) -> dict[str, float | str]:
    eta = out["eta"].detach()
    code = model.head.to_code(eta.reshape(-1, eta.shape[-1])).detach()
    stats = model.head.param_stats(eta.detach())
    row: dict[str, float | str] = {
        "phase": phase,
        "step": int(step),
        "loss": _scalar(out.get("loss")),
        "recon_loss": _scalar(out.get("recon_loss")),
        "pred_loss": _scalar(out.get("pred_loss")),
        "sigreg_loss": _scalar(out.get("sigreg_loss")),
        "kl_exact": _scalar(out.get("kl_exact")),
        "fisher_quad": _scalar(out.get("fisher_quad")),
        "exact_quad_ratio": _scalar(out.get("exact_quad_ratio")),
        "recon_gain": _scalar(out.get("recon_gain")),
        "F_gain": _scalar(out.get("F_gain")),
        "infer_kl_final": _scalar(out.get("infer_kl_final")),
        "correction_norm": _scalar(out.get("correction_norm")),
        "eta_std": _scalar(out.get("eta_std")),
        "code_mean": float(code.mean().cpu()),
        "code_std": float(code.std().cpu()),
        "grad_norm": float("nan") if grad is None else float(grad),
    }
    collapse = collapse_report(eta.detach())
    for key, value in collapse.items():
        row[f"collapse_{key}"] = float(value)
    for key, value in stats.items():
        row[key] = float(value)
    return row


def validate(model, val_loader, cfg, device, max_batches: int) -> tuple[dict[str, float], dict | None]:
    rows = []
    first = None
    model.eval()
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        batch = to_device(batch, device)
        out = compute_fond_loss(model, batch, cfg)
        row = summarize_out(model, out, "val", 0)
        try:
            action_report = model.action_prior_report(batch, out["eta"].detach(), cfg.history_size, cfg.beta)
            for key, value in action_report.items():
                row[f"action_{key}"] = float(value)
        except Exception as exc:  # diagnostics should not kill training
            row["action_report_failed"] = 1.0
            row["action_report_error_hash"] = float(abs(hash(str(exc))) % 1000000)
        rows.append(row)
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


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    crop_sizes = parse_crop_sizes(args.crop_sizes)
    seq_len = args.history_size + args.num_preds

    train_ds = MarmoBackImageSequenceDataset(
        session_name=args.session,
        split="train",
        seed=args.seed,
        target_hz=args.target_hz,
        seq_len=seq_len,
        crop_sizes=crop_sizes,
        output_hw=args.output_hw,
        center_mode=args.center_mode,
        max_windows=_maybe_max_windows(args.max_train_windows),
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
        max_windows=_maybe_max_windows(args.max_val_windows),
        stride=args.stride,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_marmo,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_marmo,
        drop_last=False,
    )

    cfg = FondMarmoConfig(
        family=args.family,
        pred_loss=args.pred_loss,
        embed_dim=args.embed_dim,
        img_ch=len(crop_sizes),
        img_hw=args.output_hw,
        action_dim=train_ds.action_dim,
        history_size=args.history_size,
        predictor_depth=args.predictor_depth,
        k_inner=args.k_inner,
        infer_backprop=args.infer_backprop,
        beta=args.beta,
        recon_weight=args.recon_weight,
        infer_lr=args.infer_lr if args.infer_lr is not None else (0.3 if args.family == "gaussian" else 0.1),
        infer_grad_clip=args.infer_grad_clip,
        sigreg_weight=args.sigreg_weight,
        sigreg_target_rate=args.sigreg_target_rate,
        sigreg_num_proj=args.sigreg_num_proj,
        sigreg_knots=args.sigreg_knots,
        poisson_log_lo=args.poisson_log_lo,
        poisson_log_hi=args.poisson_log_hi,
    )
    if cfg.sigreg_weight > 0 and not cfg.infer_backprop:
        raise ValueError(
            "SIGReg on FOND posteriors requires --infer-backprop; otherwise the "
            "regularizer cannot influence the inferred latents."
        )
    model = build_fond_model(cfg).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    outdir = Path(args.outdir) / f"{args.session}_{args.family}_{args.pred_loss}_{args.center_mode}"
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "run_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"train windows={len(train_ds)} val windows={len(val_ds)} device={args.device}")
    if train_ds.missing_image_trials:
        missing = sorted(set(train_ds.missing_image_trials.values()))
        print(f"skipping {len(train_ds.missing_image_trials)} trials with missing source images: {missing}")
    print(cfg)
    model.train()
    step = 0
    last_out = None
    metric_rows: list[dict[str, float | str]] = []
    metric_path = outdir / "metrics.csv"
    best_val = float("inf")

    def write_metrics():
        if not metric_rows:
            return
        keys = sorted(set().union(*(row.keys() for row in metric_rows)))
        preferred = ["phase", "step", "loss", "recon_loss", "pred_loss", "sigreg_loss", "kl_exact", "fisher_quad", "exact_quad_ratio"]
        keys = [k for k in preferred if k in keys] + [k for k in keys if k not in preferred]
        with open(metric_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(metric_rows)
        save_metric_plot(outdir / "training_curves.png", metric_rows)

    def save_checkpoint(name: str, extra_update: dict | None = None):
        extra = {
            "session": args.session,
            "crop_sizes": crop_sizes,
            "center_mode": args.center_mode,
            "target_hz": args.target_hz,
            "train_windows": len(train_ds),
            "val_windows": len(val_ds),
            "stride": args.stride,
            "missing_image_trials": train_ds.missing_image_trials,
            "step": step,
        }
        if extra_update:
            extra.update(extra_update)
        path = outdir / name
        torch.save(checkpoint_payload(model, cfg, extra), path)
        return path

    while step < args.max_steps:
        for batch in train_loader:
            step += 1
            batch = to_device(batch, args.device)
            opt.zero_grad(set_to_none=True)
            out = compute_fond_loss(model, batch, cfg)
            if not torch.isfinite(out["loss_for_backward"]):
                raise FloatingPointError(f"non-finite loss at step {step}: {out['loss_for_backward'].item()}")
            out["loss_for_backward"].backward()
            gnorm = grad_norm(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            last_out = out
            train_row = summarize_out(model, out, "train", step, gnorm)
            metric_rows.append(train_row)
            if step == 1 or step % max(1, args.log_interval) == 0:
                print(
                    f"step {step:04d} "
                    f"loss={out['loss'].item():.5f} "
                    f"recon={out['recon_loss'].item():.5f} "
                    f"pred={out['pred_loss'].item():.5f} "
                    f"sigreg={out['sigreg_loss'].item():.5f} "
                    f"eta_std={out['eta_std'].item():.5f} "
                    f"rank={train_row.get('collapse_eff_rank_frac', float('nan')):.3f} "
                    f"grad={gnorm:.3f}"
                )
            if args.val_interval > 0 and step % args.val_interval == 0:
                val_row, preview = validate(model, val_loader, cfg, args.device, args.val_batches)
                if val_row:
                    val_row["step"] = step
                    metric_rows.append(val_row)
                    val_loss = float(val_row.get("loss", float("inf")))
                    print(
                        f"val  {step:04d} "
                        f"loss={val_row.get('loss', float('nan')):.5f} "
                        f"recon={val_row.get('recon_loss', float('nan')):.5f} "
                        f"pred={val_row.get('pred_loss', float('nan')):.5f} "
                        f"sigreg={val_row.get('sigreg_loss', float('nan')):.5f} "
                        f"rank={val_row.get('collapse_eff_rank_frac', float('nan')):.3f} "
                        f"act_gain={val_row.get('action_action_gain_R', float('nan')):.4f}"
                    )
                    if preview is not None and (
                        args.recon_panel_interval > 0 and step % args.recon_panel_interval == 0
                    ):
                        save_recon_panel(outdir / f"reconstruction_panel_step{step:06d}.png", preview["batch"], preview["out"]["recon"])
                        save_recon_panel(outdir / "reconstruction_panel.png", preview["batch"], preview["out"]["recon"])
                    if val_loss < best_val:
                        best_val = val_loss
                        save_checkpoint("best.pt", {"best_val_loss": best_val})
                write_metrics()
            if args.save_interval > 0 and step % args.save_interval == 0:
                ckpt_path = save_checkpoint(f"step{step:06d}.pt", {"best_val_loss": best_val})
                print(f"saved checkpoint: {ckpt_path}")
            if step >= args.max_steps:
                break

    val_row, preview = validate(model, val_loader, cfg, args.device, max(1, args.val_batches))
    if val_row:
        val_row["step"] = step
        metric_rows.append(val_row)
        print("validation preview:")
        for k in ["loss", "recon_loss", "pred_loss", "sigreg_loss", "recon_gain", "F_gain", "collapse_eff_rank_frac", "collapse_batch_var_median", "action_action_gain_R", "action_action_gain_vs_noop"]:
            if k in val_row:
                print(f"  {k}: {val_row[k]:.6f}")
        if preview is not None:
            save_recon_panel(outdir / "reconstruction_panel.png", preview["batch"], preview["out"]["recon"])
    write_metrics()

    ckpt_path = save_checkpoint("last.pt", {"best_val_loss": best_val})
    print(f"saved checkpoint: {ckpt_path}")
    print(f"saved reconstruction panel: {outdir / 'reconstruction_panel.png'}")
    print(f"saved metrics: {metric_path}")


if __name__ == "__main__":
    main()
