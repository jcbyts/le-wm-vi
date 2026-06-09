from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from marmo.backimage_sequences import (
    MarmoBackImageSequenceDataset,
    collate_marmo,
    level_spec_label,
    normalize_level_specs,
)
from marmo.faithful_train_utils import load_faithful_from_checkpoint
from marmo.saliency_utils import compute_backimage_saliency


def parse_crop_sizes(text):
    return normalize_level_specs(text)


def parse_args():
    p = argparse.ArgumentParser(description="Summarize faithful Gaussian saliency by fixation/saccade bins")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=["train", "val", "all"], default="val")
    p.add_argument("--session", default=None)
    p.add_argument("--center-mode", choices=["dset", "gaze"], default=None)
    p.add_argument("--crop-sizes", default=None)
    p.add_argument("--target-hz", type=int, default=None)
    p.add_argument("--max-windows", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--precompute-pixels", action="store_true")
    p.add_argument("--saliency-mode", choices=["pred_output", "pred_loss"], default="pred_output")
    p.add_argument("--saliency-method", choices=["grad", "grad_x_input", "integrated_gradients"], default="grad_x_input")
    p.add_argument("--ig-steps", type=int, default=16)
    p.add_argument("--ig-baseline", choices=["gray", "zero", "channel_mean"], default="gray")
    p.add_argument("--saccade-threshold-deg-s", type=float, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--outdir", default=None)
    return p.parse_args()


def to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def main():
    args = parse_args()
    ckpt = Path(args.checkpoint)
    model, cfg, extra = load_faithful_from_checkpoint(str(ckpt), map_location="cpu")
    if cfg.family != "gaussian":
        raise ValueError("This summary currently targets faithful Gaussian LeWM checkpoints")
    model.to(args.device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    session = args.session or extra.get("session", "Allen_2022-04-13")
    crop_sizes = parse_crop_sizes(args.crop_sizes) if args.crop_sizes else normalize_level_specs(extra.get("crop_sizes", (51, 101, 201)))
    center_mode = args.center_mode or extra.get("center_mode", "dset")
    target_hz = int(args.target_hz if args.target_hz is not None else extra.get("target_hz", 120))
    pixel_normalization = extra.get("pixel_normalization", "unit")
    threshold = float(args.saccade_threshold_deg_s if args.saccade_threshold_deg_s is not None else extra.get("saccade_threshold_deg_s", 25.0))
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
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_marmo)
    rows = []
    for batch in loader:
        batch = to_device(batch, args.device)
        with torch.enable_grad():
            result = compute_backimage_saliency(
                model,
                batch,
                mode=args.saliency_mode,
                method=args.saliency_method,
                pred_index=-1,
                baseline=args.ig_baseline,
                ig_steps=args.ig_steps,
                source_reduce="current",
            )
        source_action = batch["action"][:, cfg.history_size - 1]
        speed = torch.linalg.norm(source_action.float(), dim=-1) * float(target_hz)
        pct = result.channel_pct.detach().cpu().numpy()
        speed_np = speed.detach().cpu().numpy()
        score_np = result.score.detach().cpu().numpy()
        for i in range(pct.shape[0]):
            row = {
                "label": "saccade" if speed_np[i] >= threshold else "fixation",
                "speed_deg_s": float(speed_np[i]),
                "score": float(score_np[i]),
            }
            for level, _spec in enumerate(crop_sizes):
                row[f"L{level}_pct"] = float(pct[i, level])
            rows.append(
                row
            )
    outdir = Path(args.outdir) if args.outdir else ckpt.with_name("saliency_summary")
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "channel_pct_by_window.csv"
    pct_fields = [f"L{i}_pct" for i in range(len(crop_sizes))]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "speed_deg_s", "score", *pct_fields])
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for label in ["fixation", "saccade"]:
        subset = [r for r in rows if r["label"] == label]
        if not subset:
            continue
        arr = np.asarray([[r[field] for field in pct_fields] for r in subset], dtype=np.float32)
        summary[label] = {
            "n": int(len(subset)),
            "levels": [level_spec_label(spec) for spec in crop_sizes],
            "mean_pct": arr.mean(axis=0).tolist(),
            "sem_pct": (arr.std(axis=0) / np.sqrt(len(subset))).tolist(),
            "mean_speed_deg_s": float(np.mean([r["speed_deg_s"] for r in subset])),
            "saliency_method": args.saliency_method,
            "ig_baseline": args.ig_baseline if args.saliency_method == "integrated_gradients" else None,
        }
    with (outdir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    labels = [k for k in ["fixation", "saccade"] if k in summary]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6, 4))
    width = min(0.8 / max(1, len(crop_sizes)), 0.24)
    colors = plt.cm.tab10(np.linspace(0, 1, max(3, len(crop_sizes))))[: len(crop_sizes)]
    offset_center = (len(crop_sizes) - 1) / 2.0
    for i, spec in enumerate(crop_sizes):
        level = f"L{i} {level_spec_label(spec)}"
        vals = [summary[label]["mean_pct"][i] for label in labels]
        err = [summary[label]["sem_pct"][i] for label in labels]
        ax.bar(x + (i - offset_center) * width, vals, width, yerr=err, label=level, color=colors[i])
    ax.set_xticks(x, [f"{label}\nn={summary[label]['n']}" for label in labels])
    ax.set_ylabel("mean saliency %")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "channel_pct_fixation_saccade.png", dpi=170)
    plt.close(fig)
    print(f"saved saliency summary: {outdir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
