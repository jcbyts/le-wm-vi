from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from marmo.backimage_sequences import (
    MarmoBackImageSequenceDataset,
    collate_marmo,
    level_spec_label,
    normalize_level_specs,
)
from marmo.faithful_train_utils import faithful_forward, load_faithful_from_checkpoint
from marmo.saliency_utils import compute_backimage_saliency


def parse_args():
    p = argparse.ArgumentParser(description="Debug L6/full-screen saliency on faithful BackImage LeWM checkpoints")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=["train", "val", "all"], default="val")
    p.add_argument("--session", default=None)
    p.add_argument("--center-mode", choices=["dset", "gaze"], default=None)
    p.add_argument("--crop-sizes", default=None)
    p.add_argument("--target-hz", type=int, default=None)
    p.add_argument("--max-windows", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--precompute-pixels", action="store_true")
    p.add_argument("--saliency-mode", choices=["pred_output", "pred_loss"], default="pred_output")
    p.add_argument(
        "--saliency-methods",
        default="grad_x_input,grad,integrated_gradients",
        help="Comma-separated list from grad_x_input,grad,integrated_gradients",
    )
    p.add_argument("--ig-steps", type=int, default=12)
    p.add_argument("--ig-baseline", choices=["gray", "zero", "channel_mean"], default="gray")
    p.add_argument("--saccade-threshold-deg-s", type=float, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--outdir", default=None)
    return p.parse_args()


def to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def finite_mean(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def finite_quantile(values, q):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.quantile(arr, q))


def _letterbox_bool(mask: np.ndarray, output_hw: int) -> np.ndarray:
    h, w = mask.shape[:2]
    scale = min(output_hw / max(1, h), output_hw / max(1, w))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    resized = np.asarray(
        Image.fromarray(mask.astype(np.uint8) * 255).resize((new_w, new_h), resample=Image.NEAREST)
    ) > 0
    canvas = np.zeros((output_hw, output_hw), dtype=bool)
    i0 = (output_hw - new_h) // 2
    j0 = (output_hw - new_w) // 2
    canvas[i0 : i0 + new_h, j0 : j0 + new_w] = resized
    return canvas


def screen_content_mask(sampler, output_hw: int) -> np.ndarray:
    h, w = sampler.screen_shape_ij
    return _letterbox_bool(np.ones((h, w), dtype=bool), output_hw)


def image_area_mask(sampler, trial_id: int, output_hw: int) -> np.ndarray:
    rect = np.asarray(sampler.exp["S"]["screenRect"], dtype=np.int64)
    screen_x0, screen_y0, screen_x1, screen_y1 = rect.tolist()
    height, width = int(screen_y1 - screen_y0), int(screen_x1 - screen_x0)
    mask = np.zeros((height, width), dtype=bool)
    left, top, right, bottom = np.asarray(sampler.trial(int(trial_id)).dest_rect, dtype=np.int64)
    r0 = max(0, int(top - screen_y0))
    r1 = min(height, int(bottom - screen_y0))
    c0 = max(0, int(left - screen_x0))
    c1 = min(width, int(right - screen_x0))
    if r1 > r0 and c1 > c0:
        mask[r0:r1, c0:c1] = True
    return _letterbox_bool(mask, output_hw)


def build_screen_masks_for_trials(ds: MarmoBackImageSequenceDataset) -> dict[int, dict[str, np.ndarray]]:
    output_hw = int(ds.output_hw)
    screen_mask = screen_content_mask(ds.sampler, output_hw)
    trials = np.unique(ds.cov["trial_inds"].detach().cpu().numpy().astype(np.int64))
    out = {}
    for trial_id in trials:
        image_mask = image_area_mask(ds.sampler, int(trial_id), output_hw)
        monitor_bg = screen_mask & ~image_mask
        letterbox = ~screen_mask
        out[int(trial_id)] = {
            "image": image_mask,
            "monitor_bg": monitor_bg,
            "letterbox": letterbox,
            "gray_all": ~image_mask,
        }
    return out


def mask_tensor_for_batch(
    batch: dict[str, torch.Tensor],
    mask_cache: dict[int, dict[str, np.ndarray]],
    key: str,
    history_size: int,
    device: torch.device | str,
) -> torch.Tensor:
    trial_ids = batch["trial_inds"][:, :history_size].detach().cpu().numpy().astype(np.int64)
    masks = []
    for row in trial_ids:
        masks.append(np.stack([mask_cache[int(t)][key] for t in row], axis=0))
    return torch.from_numpy(np.stack(masks, axis=0)).to(device=device, dtype=torch.bool)


def model_pred_loss_with_context_pixels(model, batch: dict[str, torch.Tensor], context_pixels: torch.Tensor | None = None):
    cfg = model.cfg
    orig = model.encode(batch)
    target = orig["emb"][:, cfg.num_preds : cfg.num_preds + cfg.history_size].detach()
    if context_pixels is None:
        emb = orig["emb"]
        act_emb = orig["act_emb"]
    else:
        mod_batch = dict(batch)
        mod_batch["pixels"] = context_pixels
        mod = model.encode(mod_batch)
        emb = mod["emb"]
        act_emb = mod["act_emb"]
    pred = model.predict(emb[:, : cfg.history_size], act_emb[:, : cfg.history_size])
    return (pred - target).pow(2).mean(dim=-1)


def summarize_loss_rows(rows: list[dict], threshold: float) -> dict:
    out = {}
    for label, subset in [
        ("all", rows),
        ("fixation", [r for r in rows if r["speed_deg_s"] < threshold]),
        ("saccade", [r for r in rows if r["speed_deg_s"] >= threshold]),
    ]:
        if not subset:
            continue
        modes = sorted({r["mode"] for r in subset})
        out[label] = {"n": int(len(subset) // max(1, len(modes)))}
        for mode in modes:
            vals = [r["loss"] for r in subset if r["mode"] == mode]
            out[label][mode] = {
                "mean_loss": finite_mean(vals),
                "p50_loss": finite_quantile(vals, 0.5),
                "p90_loss": finite_quantile(vals, 0.9),
            }
    return out


def run_context_ablation(
    model,
    loader,
    ds: MarmoBackImageSequenceDataset,
    crop_sizes,
    screen_idx: int,
    mask_cache,
    threshold: float,
    target_hz: int,
    device,
):
    cfg = model.cfg
    largest_numeric_idx = None
    for idx, spec in enumerate(crop_sizes):
        if not isinstance(spec, str):
            largest_numeric_idx = idx

    rows = []
    gen = torch.Generator(device="cpu").manual_seed(1002)
    model.eval()
    with torch.no_grad():
        for batch_cpu in loader:
            batch = to_device(batch_cpu, device)
            source_action = batch["action"][:, cfg.history_size - 1]
            speed = torch.linalg.norm(source_action.float(), dim=-1) * float(target_hz)
            pixels = batch["pixels"]
            masks = {
                key: mask_tensor_for_batch(batch_cpu, mask_cache, key, cfg.history_size, device)
                for key in ["image", "monitor_bg", "letterbox", "gray_all"]
            }

            variants = {"baseline": pixels}

            p = pixels.clone()
            p[:, : cfg.history_size, screen_idx] = 0.5
            variants["screen_to_gray"] = p

            p = pixels.clone()
            p[:, : cfg.history_size, screen_idx] = 0.0
            variants["screen_to_zero"] = p

            p = pixels.clone()
            screen = p[:, : cfg.history_size, screen_idx]
            screen[masks["image"]] = 0.5
            variants["screen_image_to_gray"] = p

            p = pixels.clone()
            screen = p[:, : cfg.history_size, screen_idx]
            screen[masks["gray_all"]] = 0.0
            variants["screen_gray_to_zero"] = p

            p = pixels.clone()
            screen = p[:, : cfg.history_size, screen_idx]
            screen[masks["monitor_bg"]] = 0.0
            variants["screen_monitor_bg_to_zero"] = p

            p = pixels.clone()
            screen = p[:, : cfg.history_size, screen_idx]
            screen[masks["letterbox"]] = 0.0
            variants["screen_letterbox_to_zero"] = p

            if pixels.size(0) > 1:
                p = pixels.clone()
                perm = torch.randperm(pixels.size(0), generator=gen).to(device)
                p[:, : cfg.history_size, screen_idx] = pixels[perm, : cfg.history_size, screen_idx]
                variants["screen_batch_shuffle"] = p

            if largest_numeric_idx is not None:
                p = pixels.clone()
                p[:, : cfg.history_size, largest_numeric_idx] = 0.5
                variants[f"L{largest_numeric_idx}_{level_spec_label(crop_sizes[largest_numeric_idx])}_to_gray"] = p

            for mode, variant_pixels in variants.items():
                loss_bt = model_pred_loss_with_context_pixels(
                    model,
                    batch,
                    None if mode == "baseline" else variant_pixels,
                )
                loss = loss_bt[:, -1].detach().cpu().numpy()
                speed_np = speed.detach().cpu().numpy()
                for i in range(loss.shape[0]):
                    rows.append(
                        {
                            "mode": mode,
                            "speed_deg_s": float(speed_np[i]),
                            "loss": float(loss[i]),
                        }
                    )
    return rows, summarize_loss_rows(rows, threshold)


def summarize_saliency_rows(rows: list[dict], threshold: float) -> dict:
    out = {}
    for method in sorted({r["method"] for r in rows}):
        method_rows = [r for r in rows if r["method"] == method]
        out[method] = {}
        for label, subset in [
            ("all", method_rows),
            ("fixation", [r for r in method_rows if r["speed_deg_s"] < threshold]),
            ("saccade", [r for r in method_rows if r["speed_deg_s"] >= threshold]),
        ]:
            if not subset:
                continue
            out[method][label] = {
                "n": len(subset),
                "screen_channel_pct_mean": finite_mean([r["screen_channel_pct"] for r in subset]),
                "screen_channel_pct_p50": finite_quantile([r["screen_channel_pct"] for r in subset], 0.5),
                "screen_channel_pct_p90": finite_quantile([r["screen_channel_pct"] for r in subset], 0.9),
                "image_mass_frac_mean": finite_mean([r["image_mass_frac"] for r in subset]),
                "gray_mass_frac_mean": finite_mean([r["gray_mass_frac"] for r in subset]),
                "monitor_bg_mass_frac_mean": finite_mean([r["monitor_bg_mass_frac"] for r in subset]),
                "letterbox_mass_frac_mean": finite_mean([r["letterbox_mass_frac"] for r in subset]),
                "image_area_frac_mean": finite_mean([r["image_area_frac"] for r in subset]),
                "gray_area_frac_mean": finite_mean([r["gray_area_frac"] for r in subset]),
                "gray_vs_image_density_mean": finite_mean([r["gray_vs_image_density"] for r in subset]),
            }
    return out


def run_saliency(
    model,
    loader,
    crop_sizes,
    screen_idx: int,
    mask_cache,
    methods: list[str],
    saliency_mode: str,
    ig_steps: int,
    ig_baseline: str,
    threshold: float,
    target_hz: int,
    device,
):
    cfg = model.cfg
    rows = []
    for method in methods:
        for batch_cpu in loader:
            batch = to_device(batch_cpu, device)
            with torch.enable_grad():
                result = compute_backimage_saliency(
                    model,
                    batch,
                    mode=saliency_mode,
                    method=method,
                    pred_index=-1,
                    baseline=ig_baseline,
                    ig_steps=ig_steps,
                    source_reduce="current",
                )
            source_idx = int(result.source_index)
            heat = result.heatmaps[:, screen_idx].detach().cpu().numpy()
            pct = result.channel_pct[:, screen_idx].detach().cpu().numpy()
            source_action = batch["action"][:, source_idx]
            speed = torch.linalg.norm(source_action.float(), dim=-1) * float(target_hz)
            speed_np = speed.detach().cpu().numpy()
            trial_ids = batch_cpu["trial_inds"][:, source_idx].detach().cpu().numpy().astype(np.int64)

            for i in range(heat.shape[0]):
                masks = mask_cache[int(trial_ids[i])]
                total = float(heat[i].sum())
                denom = max(total, 1e-12)
                image_mask = masks["image"]
                gray_mask = masks["gray_all"]
                monitor_mask = masks["monitor_bg"]
                letterbox_mask = masks["letterbox"]
                image_mass = float(heat[i][image_mask].sum())
                gray_mass = float(heat[i][gray_mask].sum())
                monitor_mass = float(heat[i][monitor_mask].sum())
                letterbox_mass = float(heat[i][letterbox_mask].sum())
                image_area = float(image_mask.mean())
                gray_area = float(gray_mask.mean())
                image_density = image_mass / max(float(image_mask.sum()), 1.0)
                gray_density = gray_mass / max(float(gray_mask.sum()), 1.0)
                rows.append(
                    {
                        "method": method,
                        "speed_deg_s": float(speed_np[i]),
                        "screen_channel_pct": float(pct[i]),
                        "image_mass_frac": image_mass / denom,
                        "gray_mass_frac": gray_mass / denom,
                        "monitor_bg_mass_frac": monitor_mass / denom,
                        "letterbox_mass_frac": letterbox_mass / denom,
                        "image_area_frac": image_area,
                        "gray_area_frac": gray_area,
                        "gray_vs_image_density": gray_density / max(image_density, 1e-12),
                    }
                )
    return rows, summarize_saliency_rows(rows, threshold)


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(summary: dict, outdir: Path):
    methods = sorted(summary)
    labels = ["fixation", "saccade"]
    if not methods:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    x = np.arange(len(methods))
    width = 0.35
    for li, label in enumerate(labels):
        vals = [summary[m].get(label, {}).get("screen_channel_pct_mean", np.nan) for m in methods]
        axes[0].bar(x + (li - 0.5) * width, vals, width, label=label)
    axes[0].set_title("L6 channel pct")
    axes[0].set_ylabel("% total attribution")
    axes[0].set_xticks(x, methods, rotation=20, ha="right")
    axes[0].legend(frameon=False)

    for li, label in enumerate(labels):
        vals = [summary[m].get(label, {}).get("gray_mass_frac_mean", np.nan) for m in methods]
        axes[1].bar(x + (li - 0.5) * width, vals, width, label=label)
    axes[1].set_title("L6 gray mass")
    axes[1].set_ylabel("fraction of L6 heatmap")
    axes[1].set_xticks(x, methods, rotation=20, ha="right")

    for li, label in enumerate(labels):
        vals = [summary[m].get(label, {}).get("gray_vs_image_density_mean", np.nan) for m in methods]
        axes[2].bar(x + (li - 0.5) * width, vals, width, label=label)
    axes[2].axhline(1.0, color="0.2", lw=1)
    axes[2].set_title("gray/image density")
    axes[2].set_ylabel("ratio")
    axes[2].set_xticks(x, methods, rotation=20, ha="right")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "screen_saliency_gray_diagnostic.png", dpi=170)
    plt.close(fig)


def plot_ablation(summary: dict, outdir: Path):
    labels = [k for k in ["fixation", "saccade"] if k in summary]
    if not labels:
        return
    modes = sorted({mode for label in labels for mode in summary[label] if mode != "n"})
    modes = ["baseline"] + [m for m in modes if m != "baseline"]
    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(modes)), 4))
    x = np.arange(len(modes))
    width = 0.35
    for li, label in enumerate(labels):
        vals = [summary[label].get(mode, {}).get("mean_loss", np.nan) for mode in modes]
        ax.bar(x + (li - 0.5) * width, vals, width, label=label)
    ax.set_xticks(x, modes, rotation=35, ha="right")
    ax.set_ylabel("final-step prediction loss")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "screen_context_ablation.png", dpi=170)
    plt.close(fig)


def main():
    args = parse_args()
    ckpt = Path(args.checkpoint)
    model, cfg, extra = load_faithful_from_checkpoint(str(ckpt), map_location="cpu")
    if cfg.family != "gaussian":
        raise ValueError("This diagnostic currently targets faithful Gaussian LeWM checkpoints")
    model.to(args.device).eval()
    for param in model.parameters():
        param.requires_grad_(False)

    session = args.session or extra.get("session", "Allen_2022-04-13")
    crop_sizes = normalize_level_specs(args.crop_sizes) if args.crop_sizes else normalize_level_specs(extra.get("crop_sizes", (51, 101, 201)))
    center_mode = args.center_mode or extra.get("center_mode", "dset")
    target_hz = int(args.target_hz if args.target_hz is not None else extra.get("target_hz", 120))
    pixel_normalization = extra.get("pixel_normalization", "unit")
    threshold = float(args.saccade_threshold_deg_s if args.saccade_threshold_deg_s is not None else extra.get("saccade_threshold_deg_s", 25.0))
    screen_indices = [i for i, spec in enumerate(crop_sizes) if isinstance(spec, str)]
    if not screen_indices:
        raise ValueError(f"No screen/fullscreen level found in crop sizes: {crop_sizes}")
    screen_idx = screen_indices[-1]

    ds = MarmoBackImageSequenceDataset(
        session_name=session,
        split=args.split,
        target_hz=target_hz,
        seq_len=cfg.history_size + cfg.num_preds,
        crop_sizes=crop_sizes,
        output_hw=cfg.img_hw,
        center_mode=center_mode,
        pixel_normalization=pixel_normalization,
        max_windows=args.max_windows,
    )
    if args.precompute_pixels:
        ds.precompute_pixels(verbose=True)
    mask_cache = build_screen_masks_for_trials(ds)

    loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_marmo,
    )
    methods = [m.strip() for m in args.saliency_methods.split(",") if m.strip()]
    outdir = Path(args.outdir) if args.outdir else ckpt.with_name("screen_saliency_debug")
    outdir.mkdir(parents=True, exist_ok=True)

    sal_loader = DataLoader(ds, **loader_kwargs)
    saliency_rows, saliency_summary = run_saliency(
        model,
        sal_loader,
        crop_sizes,
        screen_idx,
        mask_cache,
        methods,
        args.saliency_mode,
        args.ig_steps,
        args.ig_baseline,
        threshold,
        target_hz,
        args.device,
    )
    write_csv(outdir / "screen_saliency_rows.csv", saliency_rows)
    with (outdir / "screen_saliency_summary.json").open("w") as f:
        json.dump(saliency_summary, f, indent=2)
    plot_summary(saliency_summary, outdir)

    abl_loader = DataLoader(ds, **loader_kwargs)
    ablation_rows, ablation_summary = run_context_ablation(
        model,
        abl_loader,
        ds,
        crop_sizes,
        screen_idx,
        mask_cache,
        threshold,
        target_hz,
        args.device,
    )
    write_csv(outdir / "screen_context_ablation_rows.csv", ablation_rows)
    with (outdir / "screen_context_ablation_summary.json").open("w") as f:
        json.dump(ablation_summary, f, indent=2)
    plot_ablation(ablation_summary, outdir)

    manifest = {
        "checkpoint": str(ckpt),
        "session": session,
        "split": args.split,
        "center_mode": center_mode,
        "crop_sizes": [level_spec_label(s) for s in crop_sizes],
        "screen_level": f"L{screen_idx} {level_spec_label(crop_sizes[screen_idx])}",
        "target_hz": target_hz,
        "saccade_threshold_deg_s": threshold,
        "max_windows": args.max_windows,
        "saliency_mode": args.saliency_mode,
        "saliency_methods": methods,
        "ig_baseline": args.ig_baseline,
        "ig_steps": args.ig_steps,
    }
    with (outdir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps({"manifest": manifest, "saliency": saliency_summary, "ablation": ablation_summary}, indent=2))
    print(f"saved screen saliency diagnostic: {outdir}")


if __name__ == "__main__":
    main()
