from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Diagnose whether saved LeWM latents are overly constant during fixation")
    p.add_argument("--latents", nargs="+", required=True, help="One or more extracted latent .npz files")
    p.add_argument("--labels", default=None, help="Comma-separated labels matching --latents")
    p.add_argument("--feature-keys", default="code,pred_hat,target,eta")
    p.add_argument("--outdir", default=None)
    p.add_argument("--speed-source", choices=["eyepos", "action"], default="eyepos")
    p.add_argument("--speed-threshold-deg-s", type=float, default=25.0)
    p.add_argument("--row-step", type=int, default=0)
    p.add_argument("--min-bout-pairs", type=int, default=3)
    p.add_argument("--near-thresholds", default="0.05,0.10,0.20")
    return p.parse_args()


def parse_list(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def split_names(data: dict[str, np.ndarray]) -> dict[int, str]:
    names = data.get("split_names")
    if names is None:
        return {0: "train", 1: "val", 2: "all"}
    out = {}
    for i, name in enumerate(np.asarray(names).reshape(-1)):
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        out[int(i)] = str(name)
    return out


def corrcoef_safe(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5:
        return float("nan")
    xs = x[ok]
    ys = y[ok]
    if np.std(xs) <= 1e-12 or np.std(ys) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def pair_indices(data: dict[str, np.ndarray], *, row_step: int) -> tuple[np.ndarray, np.ndarray]:
    rows = data["row_indices"].astype(np.int64)
    trials = data["trial_inds"].astype(np.int64)
    splits = data["split"].astype(np.int64)
    if "session_id" in data:
        sessions = data["session_id"].astype(np.int64)
    else:
        sessions = np.zeros_like(splits)
    order = np.lexsort((rows, trials, splits, sessions))
    prev = order[:-1]
    cur = order[1:]
    ok = (
        (sessions[prev] == sessions[cur])
        & (splits[prev] == splits[cur])
        & (trials[prev] == trials[cur])
        & ((rows[cur] - rows[prev]) == int(row_step))
    )
    return prev[ok], cur[ok]


def pair_speed_deg_s(
    data: dict[str, np.ndarray],
    prev: np.ndarray,
    cur: np.ndarray,
    *,
    source: str,
    target_hz: float,
) -> np.ndarray:
    if source == "action" and "action" in data:
        action = data["action"].astype(np.float32)
        if action.ndim != 2 or action.shape[1] < 2:
            raise RuntimeError("action feature is missing x/y displacement columns")
        disp = action[cur, :2]
    else:
        eyepos = data["eyepos"].astype(np.float32)
        disp = eyepos[cur] - eyepos[prev]
    return np.linalg.norm(disp, axis=1) * float(target_hz)


def zscore_feature(feature: np.ndarray, split: np.ndarray) -> np.ndarray:
    x = feature.astype(np.float32)
    if x.ndim == 1:
        x = x[:, None]
    train = split == 0
    if not train.any():
        train = np.ones(len(x), dtype=bool)
    mean = np.nanmean(x[train], axis=0, keepdims=True)
    std = np.nanstd(x[train], axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (x - mean) / std


def summarize_mask(
    *,
    label: str,
    feature_key: str,
    split_name: str,
    mask_name: str,
    mask: np.ndarray,
    dz: np.ndarray,
    step_rms: np.ndarray,
    step_l1: np.ndarray,
    active_frac: np.ndarray,
    cosine: np.ndarray,
    speed: np.ndarray,
    thresholds: list[float],
) -> dict[str, object]:
    row: dict[str, object] = {
        "label": label,
        "feature_key": feature_key,
        "split": split_name,
        "state": mask_name,
        "n_pairs": int(mask.sum()),
    }
    if not mask.any():
        return row
    x = step_rms[mask]
    row.update(
        {
            "speed_mean_deg_s": float(np.nanmean(speed[mask])),
            "speed_median_deg_s": float(np.nanmedian(speed[mask])),
            "step_rms_mean_z": float(np.nanmean(x)),
            "step_rms_median_z": float(np.nanmedian(x)),
            "step_rms_p90_z": float(np.nanquantile(x, 0.9)),
            "step_l1_mean_z": float(np.nanmean(step_l1[mask])),
            "active_dim_frac_mean": float(np.nanmean(active_frac[mask])),
            "cosine_mean": float(np.nanmean(cosine[mask])),
            "cosine_median": float(np.nanmedian(cosine[mask])),
            "speed_step_rms_corr": corrcoef_safe(speed[mask], x),
        }
    )
    for threshold in thresholds:
        row[f"frac_step_rms_lt_{threshold:g}"] = float(np.mean(x < float(threshold)))
    dim_abs_step = np.nanmean(np.abs(dz[mask]), axis=0)
    dim_rms_step = np.sqrt(np.nanmean(dz[mask] * dz[mask], axis=0))
    row.update(
        {
            "dim_abs_step_mean_z": float(np.nanmean(dim_abs_step)),
            "dim_abs_step_median_z": float(np.nanmedian(dim_abs_step)),
            "dim_abs_step_p90_z": float(np.nanquantile(dim_abs_step, 0.9)),
            "dim_rms_step_mean_z": float(np.nanmean(dim_rms_step)),
            "dim_rms_step_median_z": float(np.nanmedian(dim_rms_step)),
            "dim_rms_step_p90_z": float(np.nanquantile(dim_rms_step, 0.9)),
        }
    )
    for threshold in thresholds:
        row[f"frac_dims_abs_step_lt_{threshold:g}"] = float(np.mean(dim_abs_step < float(threshold)))
    return row


def fixation_bout_rows(pair_prev: np.ndarray, pair_cur: np.ndarray, fixation: np.ndarray, min_pairs: int) -> list[np.ndarray]:
    bouts: list[np.ndarray] = []
    starts = np.flatnonzero(fixation)
    if len(starts) == 0:
        return bouts
    run: list[int] = []
    last_pair_index = None
    for pair_index in starts:
        if last_pair_index is None or pair_index == last_pair_index + 1:
            run.append(int(pair_index))
        else:
            if len(run) >= int(min_pairs):
                bouts.append(np.concatenate([[pair_prev[run[0]]], pair_cur[run]]).astype(np.int64))
            run = [int(pair_index)]
        last_pair_index = int(pair_index)
    if len(run) >= int(min_pairs):
        bouts.append(np.concatenate([[pair_prev[run[0]]], pair_cur[run]]).astype(np.int64))
    return bouts


def summarize_bouts(
    *,
    label: str,
    feature_key: str,
    split_name: str,
    z: np.ndarray,
    pair_prev: np.ndarray,
    pair_cur: np.ndarray,
    split_pair_mask: np.ndarray,
    fixation: np.ndarray,
    target_hz: float,
    min_pairs: int,
) -> dict[str, object]:
    local_prev = pair_prev[split_pair_mask]
    local_cur = pair_cur[split_pair_mask]
    local_fix = fixation[split_pair_mask]
    bouts = fixation_bout_rows(local_prev, local_cur, local_fix, min_pairs=min_pairs)
    row: dict[str, object] = {
        "label": label,
        "feature_key": feature_key,
        "split": split_name,
        "state": "fixation_bouts",
        "n_bouts": int(len(bouts)),
        "min_bout_pairs": int(min_pairs),
    }
    if not bouts:
        return row
    durations_ms = []
    within_std = []
    within_step = []
    for rows in bouts:
        zz = z[rows]
        durations_ms.append(1000.0 * max(0, len(rows) - 1) / float(target_hz))
        within_std.append(float(np.nanmean(np.nanstd(zz, axis=0))))
        if len(rows) > 1:
            dz = np.diff(zz, axis=0)
            within_step.append(float(np.nanmean(np.sqrt(np.nanmean(dz * dz, axis=1)))))
    row.update(
        {
            "duration_ms_median": float(np.nanmedian(durations_ms)),
            "duration_ms_p90": float(np.nanquantile(durations_ms, 0.9)),
            "within_bout_dim_std_mean_z": float(np.nanmean(within_std)),
            "within_bout_dim_std_median_z": float(np.nanmedian(within_std)),
            "within_bout_step_rms_mean_z": float(np.nanmean(within_step)) if within_step else float("nan"),
            "within_bout_step_rms_median_z": float(np.nanmedian(within_step)) if within_step else float("nan"),
        }
    )
    return row


def diagnose_one(
    latents_path: Path,
    *,
    label: str,
    feature_keys: list[str],
    speed_source: str,
    speed_threshold_deg_s: float,
    row_step: int,
    min_bout_pairs: int,
    near_thresholds: list[float],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    data = dict(np.load(latents_path, allow_pickle=True))
    step = int(row_step) if int(row_step) > 0 else int(np.asarray(data.get("downsample", np.array([2]))).reshape(-1)[0])
    target_hz = float(np.asarray(data.get("target_hz", np.array([120]))).reshape(-1)[0])
    prev, cur = pair_indices(data, row_step=step)
    speed = pair_speed_deg_s(data, prev, cur, source=speed_source, target_hz=target_hz)
    fixation = speed < float(speed_threshold_deg_s)
    split_pair = data["split"].astype(np.int64)[cur]
    splits = split_names(data)
    rows: list[dict[str, object]] = []
    bout_rows: list[dict[str, object]] = []

    if "action" in data and "eyepos" in data:
        action_speed = pair_speed_deg_s(data, prev, cur, source="action", target_hz=target_hz)
        eye_speed = pair_speed_deg_s(data, prev, cur, source="eyepos", target_hz=target_hz)
        rows.append(
            {
                "label": label,
                "feature_key": "__speed_check__",
                "split": "all",
                "state": "all",
                "n_pairs": int(len(speed)),
                "row_step": int(step),
                "target_hz": float(target_hz),
                "speed_source": speed_source,
                "action_eye_speed_corr": corrcoef_safe(action_speed, eye_speed),
                "action_eye_speed_absdiff_mean": float(np.nanmean(np.abs(action_speed - eye_speed))),
            }
        )

    for feature_key in feature_keys:
        if feature_key not in data:
            continue
        z = zscore_feature(data[feature_key], data["split"].astype(np.int64))
        dz = z[cur] - z[prev]
        step_rms = np.sqrt(np.nanmean(dz * dz, axis=1))
        step_l1 = np.nanmean(np.abs(dz), axis=1)
        active_frac = np.nanmean(np.abs(dz) > 0.05, axis=1)
        zn_prev = z[prev]
        zn_cur = z[cur]
        denom = np.linalg.norm(zn_prev, axis=1) * np.linalg.norm(zn_cur, axis=1)
        cosine = np.sum(zn_prev * zn_cur, axis=1) / np.maximum(denom, 1e-8)

        for split_id, name in splits.items():
            split_mask = split_pair == int(split_id)
            if not split_mask.any():
                continue
            fix_mask = split_mask & fixation
            sac_mask = split_mask & ~fixation
            rows.append(
                summarize_mask(
                    label=label,
                    feature_key=feature_key,
                    split_name=name,
                    mask_name="fixation",
                    mask=fix_mask,
                    dz=dz,
                    step_rms=step_rms,
                    step_l1=step_l1,
                    active_frac=active_frac,
                    cosine=cosine,
                    speed=speed,
                    thresholds=near_thresholds,
                )
            )
            rows.append(
                summarize_mask(
                    label=label,
                    feature_key=feature_key,
                    split_name=name,
                    mask_name="saccade",
                    mask=sac_mask,
                    dz=dz,
                    step_rms=step_rms,
                    step_l1=step_l1,
                    active_frac=active_frac,
                    cosine=cosine,
                    speed=speed,
                    thresholds=near_thresholds,
                )
            )
            if fix_mask.any() and sac_mask.any():
                fix_mean = float(np.nanmean(step_rms[fix_mask]))
                sac_mean = float(np.nanmean(step_rms[sac_mask]))
                fix_dim_abs = np.nanmean(np.abs(dz[fix_mask]), axis=0)
                sac_dim_abs = np.nanmean(np.abs(dz[sac_mask]), axis=0)
                dim_ratio = fix_dim_abs / np.maximum(sac_dim_abs, 1e-8)
                rows.append(
                    {
                        "label": label,
                        "feature_key": feature_key,
                        "split": name,
                        "state": "fixation_vs_saccade",
                        "n_pairs": int(split_mask.sum()),
                        "fixation_step_rms_mean_z": fix_mean,
                        "saccade_step_rms_mean_z": sac_mean,
                        "fix_sac_step_rms_ratio": fix_mean / max(sac_mean, 1e-8),
                        "fix_sac_constancy_index": 1.0 - (fix_mean / max(sac_mean, 1e-8)),
                        "dimwise_fix_sac_abs_ratio_median": float(np.nanmedian(dim_ratio)),
                        "dimwise_fix_sac_abs_ratio_p90": float(np.nanquantile(dim_ratio, 0.9)),
                        "fixation_pair_frac": float(np.mean(fixation[split_mask])),
                    }
                )
            bout_rows.append(
                summarize_bouts(
                    label=label,
                    feature_key=feature_key,
                    split_name=name,
                    z=z,
                    pair_prev=prev,
                    pair_cur=cur,
                    split_pair_mask=split_mask,
                    fixation=fixation,
                    target_hz=target_hz,
                    min_pairs=min_bout_pairs,
                )
            )
    return rows, bout_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    keys = sorted(set().union(*(row.keys() for row in rows)))
    preferred = ["label", "feature_key", "split", "state", "n_pairs", "n_bouts"]
    keys = [k for k in preferred if k in keys] + [k for k in keys if k not in preferred]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(path: Path, rows: list[dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    usable = [
        row
        for row in rows
        if row.get("state") in {"fixation", "saccade"}
        and row.get("split") == "val"
        and "step_rms_mean_z" in row
        and not str(row.get("feature_key", "")).startswith("__")
    ]
    if not usable:
        return
    labels = sorted({str(row["label"]) for row in usable})
    features = sorted({str(row["feature_key"]) for row in usable})
    fig, axes = plt.subplots(len(features), 1, figsize=(max(7, 2.2 * len(labels)), 2.8 * len(features)), squeeze=False)
    for ax, feature in zip(axes.ravel(), features):
        x = np.arange(len(labels), dtype=np.float32)
        fix = []
        sac = []
        for label in labels:
            rf = next((r for r in usable if r["label"] == label and r["feature_key"] == feature and r["state"] == "fixation"), None)
            rs = next((r for r in usable if r["label"] == label and r["feature_key"] == feature and r["state"] == "saccade"), None)
            fix.append(float(rf.get("step_rms_mean_z", np.nan)) if rf else np.nan)
            sac.append(float(rs.get("step_rms_mean_z", np.nan)) if rs else np.nan)
        width = 0.36
        ax.bar(x - width / 2, fix, width=width, label="fixation", color="0.25")
        ax.bar(x + width / 2, sac, width=width, label="saccade", color="C3")
        ax.set_title(feature)
        ax.set_ylabel("mean latent step RMS (z)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    latents = [Path(x) for x in args.latents]
    if args.labels is None:
        labels = [path.parent.parent.name if path.parent.name.startswith("readout") else path.parent.name for path in latents]
    else:
        labels = parse_list(args.labels)
        if len(labels) != len(latents):
            raise ValueError("--labels must match the number of --latents paths")
    outdir = Path(args.outdir) if args.outdir else latents[0].with_suffix("").with_name("latent_constancy")
    outdir.mkdir(parents=True, exist_ok=True)
    feature_keys = parse_list(args.feature_keys)
    thresholds = parse_float_list(args.near_thresholds)

    rows: list[dict[str, object]] = []
    bout_rows: list[dict[str, object]] = []
    manifest = {
        "latents": [str(path) for path in latents],
        "labels": labels,
        "feature_keys": feature_keys,
        "speed_source": args.speed_source,
        "speed_threshold_deg_s": float(args.speed_threshold_deg_s),
        "row_step": int(args.row_step),
        "min_bout_pairs": int(args.min_bout_pairs),
        "near_thresholds": thresholds,
    }
    for path, label in zip(latents, labels, strict=False):
        these_rows, these_bouts = diagnose_one(
            path,
            label=label,
            feature_keys=feature_keys,
            speed_source=args.speed_source,
            speed_threshold_deg_s=args.speed_threshold_deg_s,
            row_step=args.row_step,
            min_bout_pairs=args.min_bout_pairs,
            near_thresholds=thresholds,
        )
        rows.extend(these_rows)
        bout_rows.extend(these_bouts)
    write_csv(outdir / "latent_constancy_pairs.csv", rows)
    write_csv(outdir / "latent_constancy_bouts.csv", bout_rows)
    plot_summary(outdir / "latent_constancy_val_step_rms.png", rows)
    with (outdir / "latent_constancy_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"saved {outdir / 'latent_constancy_pairs.csv'}")
    print(f"saved {outdir / 'latent_constancy_bouts.csv'}")
    print(f"saved {outdir / 'latent_constancy_manifest.json'}")
    if (outdir / "latent_constancy_val_step_rms.png").exists():
        print(f"saved {outdir / 'latent_constancy_val_step_rms.png'}")


if __name__ == "__main__":
    main()
