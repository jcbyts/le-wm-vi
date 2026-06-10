from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Summarize faithful Gaussian variant-search outputs")
    p.add_argument("--root", default="/home/tejas/le-wm-vi/outputs/gaussian_variant_search")
    p.add_argument("--out", default=None)
    return p.parse_args()


def load_summary(path: Path) -> dict:
    with path.open() as f:
        row = json.load(f)
    args = row.get("args", {})
    flat = {
        "tag": row.get("tag", path.parents[1].name),
        "outdir": row.get("outdir", str(path.parent)),
        "checkpoint": row.get("checkpoint"),
        "latents": row.get("latents"),
        "readout_dir": row.get("readout_dir"),
    }
    for key in [
        "pyramid_mode",
        "crop_sizes",
        "target_mode",
        "foveal_dim",
        "context_dim",
        "action_history",
        "action_smoothed_dim",
        "action_mlp_scale",
        "max_steps",
        "max_train_windows",
        "max_val_windows",
        "seed",
    ]:
        flat[key] = args.get(key)
    for prefix, data in [("readout", row.get("best_readout", {})), ("wm", row.get("last_val", {}))]:
        for key, value in data.items():
            if isinstance(value, (int, float, str, bool)) or value is None:
                flat[f"{prefix}_{key}"] = value
    return flat


def main():
    args = parse_args()
    root = Path(args.root)
    rows = [load_summary(p) for p in sorted(root.glob("*/**/variant_summary.json"))]
    if not rows:
        raise SystemExit(f"No variant_summary.json files found under {root}")
    df = pd.DataFrame(rows)
    if "readout_val_mean_bps" in df:
        df = df.sort_values("readout_val_mean_bps", ascending=False, na_position="last")
    out = Path(args.out) if args.out else root / "variant_leaderboard.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    cols = [
        "tag",
        "pyramid_mode",
        "crop_sizes",
        "target_mode",
        "foveal_dim",
        "context_dim",
        "action_history",
        "readout_feature_key",
        "readout_lag_set",
        "readout_weight_decay",
        "readout_val_mean_bps",
        "readout_val_median_bps",
        "readout_val_median_corr",
        "wm_pred_loss",
        "wm_saccade_pred_loss",
        "wm_action_gain",
    ]
    cols = [c for c in cols if c in df.columns]
    print(f"saved {out}")
    print(df[cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
