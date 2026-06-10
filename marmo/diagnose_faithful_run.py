from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from marmo.backimage_sequences import (
    BackImagePaths,
    BackImageSampler,
    load_backimage_dataset,
    normalize_level_specs,
)
from marmo.faithful_train_utils import faithful_forward, load_faithful_from_checkpoint
from marmo.make_backimage_latent_movie import (
    build_clip,
    downsample_factor,
    require_covariates,
    sample_covariate_array,
    to_model_batch,
)
from marmo.make_faithful_saliency_movie import select_sequence_rows
from marmo.saliency_utils import compute_backimage_saliency
from marmo.train_latent_spike_readout import (
    ReadoutConfig,
    build_visioncore_behavior,
    calc_poisson_bits_per_spike,
    expand_dfs,
    safe_corr,
    train_one_config,
)
from marmo.v1_saliency_utils import (
    compute_v1_observed_saliency_for_clip,
    load_v1_readout_saliency_state,
)


def parse_args():
    p = argparse.ArgumentParser(description="Diagnose faithful BackImage Gaussian WM artifacts and V1 readouts")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--session", default="Allen_2022-04-13")
    p.add_argument("--latents", default=None)
    p.add_argument("--readout-model", default=None)
    p.add_argument("--readout-predictions", default=None)
    p.add_argument("--outdir", default=None)
    p.add_argument("--candidate-crop-sets", default="51,1201;51,101,201,401;51,101,201,401,641")
    p.add_argument("--row-sample-step", type=int, default=50)
    p.add_argument("--integrity-samples", type=int, default=256)
    p.add_argument("--saliency-frames", type=int, default=32)
    p.add_argument("--v1-saliency-frames", type=int, default=16)
    p.add_argument("--readout-ablation-epochs", type=int, default=35)
    p.add_argument("--readout-ablation-patience", type=int, default=8)
    p.add_argument("--readout-ablation-batch-size", type=int, default=4096)
    p.add_argument("--confound-max-rows", type=int, default=50000)
    p.add_argument("--seed", type=int, default=1002)
    p.add_argument("--saccade-threshold-deg-s", type=float, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--skip-saliency", action="store_true")
    p.add_argument("--skip-v1-saliency", action="store_true")
    p.add_argument("--skip-readout-ablation", action="store_true")
    return p.parse_args()


def tensor_np(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def cov_np(cov: dict, key: str, rows: np.ndarray | None = None) -> np.ndarray:
    x = cov[key]
    if rows is not None:
        x = x[rows]
    return tensor_np(x)


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_crop_sets(text: str):
    return [normalize_level_specs(x.strip()) for x in text.split(";") if x.strip()]


def level_labels(crop_set) -> list[str]:
    out = []
    for spec in crop_set:
        out.append("screen" if isinstance(spec, str) else str(int(spec)))
    return out


def split_map_for_trials(trial_ids: np.ndarray, *, seed: int, split_mode: str, train_frac: float = 0.8) -> dict[int, int]:
    unique = np.array(sorted(int(t) for t in np.unique(trial_ids)), dtype=np.int64)
    if str(split_mode).lower() == "torch":
        generator = torch.Generator().manual_seed(int(seed))
        perm = torch.randperm(len(unique), generator=generator).cpu().numpy()
        ordered = unique[perm]
    else:
        rng = np.random.default_rng(int(seed))
        ordered = unique.copy()
        rng.shuffle(ordered)
    n_train = int(math.floor(len(ordered) * float(train_frac)))
    train = set(int(x) for x in ordered[:n_train])
    return {int(t): (0 if int(t) in train else 1) for t in unique}


def screen_roi(sampler: BackImageSampler) -> np.ndarray:
    return sampler.screen_roi().astype(np.float64)


def dest_roi(sampler: BackImageSampler, trial_id: int) -> np.ndarray:
    left, top, right, bottom = np.asarray(sampler.trial(int(trial_id)).dest_rect, dtype=np.float64).tolist()
    return np.asarray([[top, bottom], [left, right]], dtype=np.float64)


def area_fraction(roi: np.ndarray, bounds: np.ndarray) -> float:
    roi = np.asarray(roi, dtype=np.float64)
    bounds = np.asarray(bounds, dtype=np.float64)
    h = max(0.0, float(roi[0, 1] - roi[0, 0]))
    w = max(0.0, float(roi[1, 1] - roi[1, 0]))
    if h <= 0.0 or w <= 0.0:
        return 0.0
    i0 = max(float(roi[0, 0]), float(bounds[0, 0]))
    i1 = min(float(roi[0, 1]), float(bounds[0, 1]))
    j0 = max(float(roi[1, 0]), float(bounds[1, 0]))
    j1 = min(float(roi[1, 1]), float(bounds[1, 1]))
    return max(0.0, i1 - i0) * max(0.0, j1 - j0) / (h * w)


def resized_mask(roi: np.ndarray, bounds: np.ndarray, output_hw: int) -> np.ndarray:
    roi = np.asarray(roi, dtype=np.float64)
    bounds = np.asarray(bounds, dtype=np.float64)
    hw = int(output_hw)
    ii = roi[0, 0] + (np.arange(hw) + 0.5) * (roi[0, 1] - roi[0, 0]) / max(hw, 1)
    jj = roi[1, 0] + (np.arange(hw) + 0.5) * (roi[1, 1] - roi[1, 0]) / max(hw, 1)
    mi = (ii >= bounds[0, 0]) & (ii < bounds[0, 1])
    mj = (jj >= bounds[1, 0]) & (jj < bounds[1, 1])
    return mi[:, None] & mj[None, :]


def finite_anchor_rows(cov: dict, *, downsample: int, validity_mode: str) -> np.ndarray:
    trial_inds = cov_np(cov, "trial_inds").astype(np.int64)
    dpi_valid = cov_np(cov, "dpi_valid") > 0
    rows = np.arange(0, len(trial_inds) - max(1, downsample), max(1, downsample), dtype=np.int64)
    if str(validity_mode).lower() == "all" and downsample > 1:
        offsets = np.arange(downsample, dtype=np.int64)
        bins = rows[:, None] + offsets[None, :]
        same_trial = trial_inds[bins] == trial_inds[rows][:, None]
        valid = np.all(dpi_valid[bins], axis=1) & np.all(same_trial, axis=1)
    else:
        valid = dpi_valid[rows]
    return rows[valid]


def summarize_values(values: np.ndarray, prefix: str) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_mean": None,
            f"{prefix}_median": None,
            f"{prefix}_p10": None,
            f"{prefix}_p90": None,
            f"{prefix}_min": None,
            f"{prefix}_max": None,
        }
    return {
        f"{prefix}_n": int(values.size),
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p10": float(np.percentile(values, 10)),
        f"{prefix}_p90": float(np.percentile(values, 90)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
    }


def integrity_checks(
    *,
    cov: dict,
    sampler: BackImageSampler,
    session: str,
    extra: dict,
    cfg,
    latents_path: Path | None,
    outdir: Path,
    n_samples: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    trial_inds = cov_np(cov, "trial_inds").astype(np.int64)
    valid = cov_np(cov, "dpi_valid") > 0
    rows = np.flatnonzero(valid)
    if rows.size > n_samples:
        rows = np.sort(rng.choice(rows, size=n_samples, replace=False))
    crop_errors = []
    shape_errors = 0
    for row in rows:
        tid = int(trial_inds[row])
        if not sampler.has_image(tid):
            continue
        roi = cov_np(cov, "roi", np.asarray([row]))[0].astype(np.int64)
        stim = cov_np(cov, "stim", np.asarray([row]))[0]
        crop = sampler.crop_roi(tid, roi)
        if crop.shape != stim.shape:
            shape_errors += 1
            continue
        crop_errors.append(float(np.max(np.abs(crop.astype(np.int16) - stim.astype(np.int16)))))

    report = {
        "session": session,
        "sampled_rows": int(len(rows)),
        "l0_shape_errors": int(shape_errors),
        "l0_max_abs_diff_max": None if not crop_errors else float(np.max(crop_errors)),
        "l0_max_abs_diff_mean": None if not crop_errors else float(np.mean(crop_errors)),
        "checkpoint_crop_sizes": level_labels(extra.get("crop_sizes", ())),
        "target_hz": int(extra.get("target_hz", 120)),
        "robs_downsample_mode": str(extra.get("robs_downsample_mode", "sample")),
        "covariate_downsample_mode": str(extra.get("covariate_downsample_mode", "sample")),
    }

    if latents_path is not None and latents_path.exists():
        lat = np.load(latents_path, allow_pickle=True)
        lat_rows = lat["row_indices"].astype(np.int64)
        lat_trials = lat["trial_inds"].astype(np.int64)
        take = np.arange(len(lat_rows))
        if take.size > n_samples:
            take = np.sort(rng.choice(take, size=n_samples, replace=False))
        rows_take = lat_rows[take]
        trial_match = trial_inds[rows_take] == lat_trials[take]
        down = downsample_factor(int(extra.get("target_hz", 120)))
        eye_key = require_covariates(cov)
        checks = {"trial_match_frac": float(np.mean(trial_match)) if trial_match.size else None}
        for key, mode in [
            ("robs", str(extra.get("robs_downsample_mode", "sample"))),
            (eye_key, str(extra.get("covariate_downsample_mode", "sample"))),
            ("t_bins", str(extra.get("covariate_downsample_mode", "sample"))),
        ]:
            got = sample_covariate_array(cov, key, rows_take, downsample=down, mode=mode).astype(np.float32)
            saved_key = "eyepos" if key == eye_key else key
            if saved_key in lat:
                saved = lat[saved_key][take].astype(np.float32)
                diff = np.abs(got - saved)
                checks[f"{saved_key}_max_abs_diff"] = float(np.nanmax(diff))
                checks[f"{saved_key}_mean_abs_diff"] = float(np.nanmean(diff))
        report["latent_table_checks"] = checks
    write_json(outdir / "baseline_integrity.json", report)
    return report


def padding_and_coverage(
    *,
    cov: dict,
    sampler: BackImageSampler,
    crop_sets,
    extra: dict,
    outdir: Path,
    row_sample_step: int,
    seed: int,
    threshold_deg_s: float,
) -> dict:
    target_hz = int(extra.get("target_hz", 120))
    down = downsample_factor(target_hz)
    rows = finite_anchor_rows(
        cov,
        downsample=down,
        validity_mode=str(extra.get("validity_downsample_mode", "sample")),
    )
    rows = rows[:: max(1, int(row_sample_step))]
    trial_inds = cov_np(cov, "trial_inds").astype(np.int64)
    split_map = split_map_for_trials(
        trial_inds,
        seed=int(extra.get("seed", seed)),
        split_mode=str(extra.get("split_mode", "numpy")),
    )
    screen = screen_roi(sampler)
    eye_key = require_covariates(cov)
    eyepos = sample_covariate_array(
        cov,
        eye_key,
        rows,
        downsample=down,
        mode=str(extra.get("covariate_downsample_mode", "sample")),
    ).astype(np.float32)
    next_rows = rows + down
    ok_next = next_rows < len(trial_inds)
    ok_next &= trial_inds[next_rows] == trial_inds[rows]
    next_eye = np.full_like(eyepos, np.nan)
    next_eye[ok_next] = sample_covariate_array(
        cov,
        eye_key,
        next_rows[ok_next],
        downsample=down,
        mode=str(extra.get("covariate_downsample_mode", "sample")),
    ).astype(np.float32)
    speed = np.linalg.norm(next_eye - eyepos, axis=1) * float(target_hz)
    state = np.where(speed >= float(threshold_deg_s), "saccade", "fixation")
    state[~np.isfinite(speed)] = "unknown"

    detail_rows = []
    summary: dict[str, object] = {"n_rows": int(len(rows)), "target_hz": target_hz}
    for crop_set in crop_sets:
        labels = level_labels(crop_set)
        set_name = ",".join(labels)
        level_screen = [[] for _ in labels]
        level_display = [[] for _ in labels]
        coverage = []
        coverage_state = {"fixation": [], "saccade": []}
        for idx, row in enumerate(rows):
            tid = int(trial_inds[row])
            if not sampler.has_image(tid):
                continue
            rois = sampler.pyramid_rois_for_row(
                cov,
                int(row),
                crop_sizes=crop_set,
                center_mode=str(extra.get("center_mode", "dset")),
            )
            dst = dest_roi(sampler, tid)
            for li, roi in enumerate(rois):
                level_screen[li].append(area_fraction(roi, screen))
                level_display[li].append(area_fraction(roi, dst))
                detail_rows.append(
                    {
                        "crop_set": set_name,
                        "row": int(row),
                        "trial": tid,
                        "split": int(split_map.get(tid, -1)),
                        "state": str(state[idx]),
                        "level": int(li),
                        "level_label": labels[li],
                        "screen_frac": level_screen[li][-1],
                        "display_frac": level_display[li][-1],
                    }
                )
            if ok_next[idx]:
                next_roi = cov_np(cov, "roi", np.asarray([next_rows[idx]]))[0].astype(np.float64)
                next_center = next_roi.mean(axis=1)
                largest = rois[-1].astype(np.float64)
                inside = (
                    largest[0, 0] <= next_center[0] < largest[0, 1]
                    and largest[1, 0] <= next_center[1] < largest[1, 1]
                )
                coverage.append(float(inside))
                if state[idx] in coverage_state:
                    coverage_state[str(state[idx])].append(float(inside))
        set_summary = {}
        for li, label in enumerate(labels):
            set_summary[f"L{li}_{label}_screen"] = summarize_values(np.asarray(level_screen[li]), "frac")
            set_summary[f"L{li}_{label}_display"] = summarize_values(np.asarray(level_display[li]), "frac")
        set_summary["next_l0_center_coverage_all"] = float(np.mean(coverage)) if coverage else None
        for key, vals in coverage_state.items():
            set_summary[f"next_l0_center_coverage_{key}"] = float(np.mean(vals)) if vals else None
        summary[set_name] = set_summary
    write_csv(outdir / "padding_rows.csv", detail_rows)
    write_json(outdir / "padding_summary.json", summary)
    return summary


def effective_rank(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or min(x.shape) < 2:
        return float("nan")
    x = x - np.nanmean(x, axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(np.nan_to_num(x), full_matrices=False)
    p = s / np.maximum(s.sum(), 1e-12)
    ent = -np.sum(p * np.log(np.maximum(p, 1e-12)))
    return float(np.exp(ent))


def latent_dynamics_and_confound(
    *,
    cov: dict,
    sampler: BackImageSampler,
    cfg,
    extra: dict,
    latents_path: Path | None,
    outdir: Path,
    max_rows: int,
    seed: int,
    threshold_deg_s: float,
) -> dict:
    if latents_path is None or not latents_path.exists():
        return {}
    rng = np.random.default_rng(seed)
    lat = np.load(latents_path, allow_pickle=True)
    n = len(lat["row_indices"])
    take = np.arange(n)
    if n > max_rows:
        take = np.sort(rng.choice(take, size=int(max_rows), replace=False))
    rows = lat["row_indices"][take].astype(np.int64)
    trials = lat["trial_inds"][take].astype(np.int64)
    action = lat["action"][take].astype(np.float32) if "action" in lat else np.zeros((len(take), 2), dtype=np.float32)
    target_hz = int(extra.get("target_hz", 120))
    speed = np.linalg.norm(action[..., :2], axis=1) * float(target_hz)
    fixation = speed < float(threshold_deg_s)
    saccade = speed >= float(threshold_deg_s)

    report: dict[str, object] = {
        "latents": str(latents_path),
        "sampled_rows": int(len(take)),
        "fixation_frac": float(np.mean(fixation)) if len(fixation) else None,
    }
    fdim = int(getattr(cfg, "foveal_dim", 0) or 0)
    for key in ["eta", "code", "pred_hat", "target"]:
        if key not in lat:
            continue
        arr = lat[key][take].astype(np.float32)
        groups = {"all": np.ones(len(arr), dtype=bool), "fixation": fixation, "saccade": saccade}
        for name, mask in groups.items():
            if int(mask.sum()) < 10:
                continue
            sub = arr[mask]
            report[f"{key}_{name}_std_mean"] = float(np.nanmean(np.nanstd(sub, axis=0)))
            report[f"{key}_{name}_eff_rank"] = effective_rank(sub)
            report[f"{key}_{name}_eff_rank_frac"] = report[f"{key}_{name}_eff_rank"] / float(sub.shape[1])
            if fdim > 0 and arr.shape[1] > fdim:
                report[f"{key}_{name}_foveal_std_mean"] = float(np.nanmean(np.nanstd(sub[:, :fdim], axis=0)))
                report[f"{key}_{name}_context_std_mean"] = float(np.nanmean(np.nanstd(sub[:, fdim:], axis=0)))
                report[f"{key}_{name}_foveal_eff_rank"] = effective_rank(sub[:, :fdim])
                report[f"{key}_{name}_context_eff_rank"] = effective_rank(sub[:, fdim:])

        # Consecutive-row deltas within the same trial and split.
        order = np.lexsort((rows, trials))
        sorted_arr = arr[order]
        sorted_rows = rows[order]
        sorted_trials = trials[order]
        delta_ok = (sorted_trials[1:] == sorted_trials[:-1]) & (
            sorted_rows[1:] - sorted_rows[:-1] == downsample_factor(target_hz)
        )
        if delta_ok.any():
            deltas = np.linalg.norm(sorted_arr[1:][delta_ok] - sorted_arr[:-1][delta_ok], axis=1)
            report[f"{key}_consecutive_delta_mean"] = float(np.mean(deltas))
            report[f"{key}_consecutive_delta_p10"] = float(np.percentile(deltas, 10))
            report[f"{key}_consecutive_delta_p90"] = float(np.percentile(deltas, 90))

    # Regress latent dimensions against screen-geometry covariates.
    good = np.zeros(len(rows), dtype=bool)
    geom = []
    screen = screen_roi(sampler)
    run_crop = normalize_level_specs(extra.get("crop_sizes", (51, 1201)))
    largest_level = len(run_crop) - 1
    for i, (row, tid) in enumerate(zip(rows, trials, strict=False)):
        if int(row) < 0 or int(row) >= len(cov["trial_inds"]):
            continue
        if not sampler.has_image(int(tid)):
            continue
        roi = cov_np(cov, "roi", np.asarray([row]))[0].astype(np.float64)
        center = roi.mean(axis=1)
        rois = sampler.pyramid_rois_for_row(
            cov,
            int(row),
            crop_sizes=run_crop,
            center_mode=str(extra.get("center_mode", "dset")),
        )
        largest = rois[largest_level]
        scr_frac = area_fraction(largest, screen)
        dst_frac = area_fraction(largest, dest_roi(sampler, int(tid)))
        edge = np.asarray(
            [
                center[0] - screen[0, 0],
                screen[0, 1] - center[0],
                center[1] - screen[1, 0],
                screen[1, 1] - center[1],
            ],
            dtype=np.float64,
        )
        geom.append([center[0], center[1], *edge.tolist(), scr_frac, dst_frac, speed[i]])
        good[i] = True
    if good.sum() >= 100 and "pred_hat" in lat:
        x = np.asarray(geom, dtype=np.float32)
        y = lat["pred_hat"][take][good].astype(np.float32)
        x = x[np.isfinite(x).all(axis=1)]
        y = y[: len(x)]
        if len(x) >= 100:
            perm = rng.permutation(len(x))
            split = max(10, int(0.8 * len(x)))
            tr = perm[:split]
            va = perm[split:]
            xm = x[tr].mean(axis=0, keepdims=True)
            xs = x[tr].std(axis=0, keepdims=True)
            xs[xs < 1e-6] = 1.0
            ym = y[tr].mean(axis=0, keepdims=True)
            xtr = np.c_[np.ones((len(tr), 1), dtype=np.float32), (x[tr] - xm) / xs]
            xva = np.c_[np.ones((len(va), 1), dtype=np.float32), (x[va] - xm) / xs]
            coef, *_ = np.linalg.lstsq(xtr, y[tr] - ym, rcond=None)
            pred = xva @ coef + ym
            ss_res = ((y[va] - pred) ** 2).sum(axis=0)
            ss_tot = ((y[va] - y[va].mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
            r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
            report["pred_hat_geometry_r2_mean"] = float(np.nanmean(r2))
            report["pred_hat_geometry_r2_median"] = float(np.nanmedian(r2))
            report["pred_hat_geometry_r2_p90"] = float(np.nanpercentile(r2, 90))
            if fdim > 0 and y.shape[1] > fdim:
                report["pred_hat_foveal_geometry_r2_mean"] = float(np.nanmean(r2[:fdim]))
                report["pred_hat_context_geometry_r2_mean"] = float(np.nanmean(r2[fdim:]))

    write_json(outdir / "latent_dynamics_confound.json", report)
    return report


def build_lagged_subset_features(
    data: dict[str, np.ndarray],
    *,
    latent_key: str,
    dim_slice: slice | None,
    lag_set: tuple[int, ...],
    split_id: int,
    include_eye: bool,
    include_action: bool,
    behavior_mode: str,
    behavior_features: np.ndarray | None,
    unit_mask: np.ndarray,
    row_step: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split = data["split"].astype(np.int64)
    rows = data["row_indices"].astype(np.int64)
    trials = data["trial_inds"].astype(np.int64)
    robs = data["robs"].astype(np.float32)[:, unit_mask]
    dfs = data["dfs"].astype(np.float32)
    if dfs.ndim == 2 and dfs.shape[1] == unit_mask.shape[0]:
        dfs = dfs[:, unit_mask]
    if dfs.ndim == 2 and dfs.shape[1] not in {1, robs.shape[1]}:
        raise RuntimeError(f"dfs width {dfs.shape[1]} does not match robs width {robs.shape[1]}")
    eyepos = data["eyepos"].astype(np.float32)
    action = data["action"].astype(np.float32)
    use_latent = latent_key.lower() not in {"none", "null", "covariates", "behavior"}
    if use_latent and latent_key not in data:
        raise RuntimeError(f"latent key {latent_key!r} is missing from {sorted(data.keys())}")
    latent = data[latent_key].astype(np.float32) if use_latent else None
    if latent is not None and dim_slice is not None:
        latent = latent[:, dim_slice]
    index = {
        (int(s), int(t), int(r)): i
        for i, (s, t, r) in enumerate(zip(split, trials, rows, strict=False))
        if int(s) == int(split_id)
    }
    feats = []
    ys = []
    masks = []
    for i in np.flatnonzero(split == int(split_id)):
        pieces = []
        ok = True
        if latent is not None:
            for lag in lag_set:
                j = index.get((int(split[i]), int(trials[i]), int(rows[i]) - int(lag) * int(row_step)))
                if j is None:
                    ok = False
                    break
                pieces.append(latent[j])
        if not ok:
            continue
        if behavior_mode in {"raw", "raw+visioncore"} and include_eye:
            pieces.append(eyepos[i])
        if behavior_mode in {"raw", "raw+visioncore"} and include_action:
            pieces.append(action[i])
            pieces.append(np.asarray([np.linalg.norm(action[i])], dtype=np.float32))
        if behavior_mode in {"visioncore", "raw+visioncore"}:
            if behavior_features is None:
                raise RuntimeError("Readout requests VisionCore behavior features but they were not built")
            pieces.append(behavior_features[i].astype(np.float32))
        if not pieces:
            pieces.append(np.ones(1, dtype=np.float32))
        feats.append(np.concatenate(pieces).astype(np.float32))
        ys.append(robs[i].astype(np.float32))
        masks.append(dfs[i].astype(np.float32))
    return np.stack(feats, axis=0), np.stack(ys, axis=0), np.stack(masks, axis=0)


def readout_ablation(
    *,
    cfg,
    latents_path: Path | None,
    readout_model_path: Path | None,
    outdir: Path,
    epochs: int,
    patience: int,
    batch_size: int,
    device: str,
    seed: int,
) -> list[dict]:
    if latents_path is None or not latents_path.exists():
        return []
    data = dict(np.load(latents_path, allow_pickle=True))
    robs = data["robs"].astype(np.float32)
    dfs_exp = expand_dfs(data["dfs"].astype(np.float32), robs.shape[1])
    train = data["split"].astype(np.int64) == 0
    train_spikes = (robs[train] * dfs_exp[train]).sum(axis=0)
    unit_mask = train_spikes >= 5.0
    base_cfg = None
    if readout_model_path is not None and readout_model_path.exists():
        payload = torch.load(readout_model_path, map_location="cpu", weights_only=False)
        raw_cfg = dict(payload["config"])
        raw_cfg["lag_set"] = tuple(int(x) for x in raw_cfg["lag_set"])
        raw_cfg.setdefault("behavior_mode", "raw")
        base_cfg = ReadoutConfig(**raw_cfg)
        unit_mask = np.asarray(payload["unit_mask"], dtype=bool)
    if base_cfg is None:
        base_cfg = ReadoutConfig(
            arch="mlp",
            lag_set=(3, 4, 5, 6),
            feature_key="pred_hat",
            behavior_mode="raw",
            include_eye=True,
            include_action=True,
            hidden_dim=128,
            depth=2,
            dropout=0.1,
            weight_decay=3e-4,
            lr=1e-3,
        )
    row_step = int(np.asarray(data.get("downsample", np.array([2]))).reshape(-1)[0])
    if readout_model_path is not None and readout_model_path.exists():
        try:
            row_step = int(payload.get("row_step", row_step))
        except UnboundLocalError:
            pass
    behavior_features = None
    if base_cfg.behavior_mode in {"visioncore", "raw+visioncore"}:
        behavior_features = build_visioncore_behavior(data, row_step=row_step)
    fdim = int(getattr(cfg, "foveal_dim", 0) or 0)
    total_dim = int(data["pred_hat"].shape[1])
    variants = []
    for latent_key in ("code", "pred_hat"):
        if latent_key not in data:
            continue
        variants.extend(
            [
                (f"{latent_key}_full_plus_behavior", latent_key, slice(None), True, True, base_cfg.behavior_mode),
                (f"{latent_key}_full_no_behavior", latent_key, slice(None), False, False, "none"),
            ]
        )
        if fdim > 0 and total_dim > fdim:
            variants.extend(
                [
                    (f"{latent_key}_foveal_plus_behavior", latent_key, slice(0, fdim), True, True, base_cfg.behavior_mode),
                    (f"{latent_key}_context_plus_behavior", latent_key, slice(fdim, total_dim), True, True, base_cfg.behavior_mode),
                    (f"{latent_key}_foveal_no_behavior", latent_key, slice(0, fdim), False, False, "none"),
                    (f"{latent_key}_context_no_behavior", latent_key, slice(fdim, total_dim), False, False, "none"),
                ]
            )
    variants.append(("behavior_only", "none", None, True, True, base_cfg.behavior_mode))
    rows = []
    for vi, (name, latent_key, dim_slice, include_eye, include_action, behavior_mode) in enumerate(variants):
        cfg_i = ReadoutConfig(
            arch=base_cfg.arch,
            lag_set=base_cfg.lag_set,
            feature_key=name,
            behavior_mode=behavior_mode,
            include_eye=include_eye,
            include_action=include_action,
            hidden_dim=base_cfg.hidden_dim,
            depth=base_cfg.depth,
            dropout=base_cfg.dropout,
            weight_decay=base_cfg.weight_decay,
            lr=base_cfg.lr,
        )
        x_train, y_train, dfs_train = build_lagged_subset_features(
            data,
            latent_key=latent_key,
            dim_slice=dim_slice,
            lag_set=base_cfg.lag_set,
            split_id=0,
            include_eye=include_eye,
            include_action=include_action,
            behavior_mode=behavior_mode,
            behavior_features=behavior_features,
            unit_mask=unit_mask,
            row_step=row_step,
        )
        x_val, y_val, dfs_val = build_lagged_subset_features(
            data,
            latent_key=latent_key,
            dim_slice=dim_slice,
            lag_set=base_cfg.lag_set,
            split_id=1,
            include_eye=include_eye,
            include_action=include_action,
            behavior_mode=behavior_mode,
            behavior_features=behavior_features,
            unit_mask=unit_mask,
            row_step=row_step,
        )
        model, metrics = train_one_config(
            cfg_i,
            x_train,
            y_train,
            dfs_train,
            x_val,
            y_val,
            dfs_val,
            epochs=int(epochs),
            patience=int(patience),
            batch_size=int(batch_size),
            device=device,
            seed=int(seed + vi),
        )
        row = {
            "variant": name,
            "latent_key": latent_key,
            "input_dim": int(x_train.shape[1]),
            "n_train": int(x_train.shape[0]),
            "n_val": int(x_val.shape[0]),
            "best_epoch": int(metrics["best_epoch"]),
            "train_mean_bps": float(metrics["train"]["mean_bps"]),
            "train_median_bps": float(metrics["train"]["median_bps"]),
            "val_mean_bps": float(metrics["val"]["mean_bps"]),
            "val_median_bps": float(metrics["val"]["median_bps"]),
            "val_median_corr": float(metrics["val"]["median_corr"]),
            "val_loss": float(metrics["val"]["loss"]),
            "behavior_mode": behavior_mode,
            "row_step": int(row_step),
        }
        rows.append(row)
        write_csv(outdir / "readout_ablation.csv", rows)
    write_json(outdir / "readout_ablation_summary.json", {"rows": rows, "base_config": asdict(base_cfg)})
    return rows


def saliency_artifact_diagnostic(
    *,
    cov: dict,
    sampler: BackImageSampler,
    model,
    cfg,
    extra: dict,
    outdir: Path,
    session: str,
    v1_predictions: Path | None,
    n_wm: int,
    n_v1: int,
    threshold_deg_s: float,
    device: str,
) -> dict:
    if n_wm <= 0 and n_v1 <= 0:
        return {}
    target_hz = int(extra.get("target_hz", 120))
    down = downsample_factor(target_hz)
    crop_sizes = normalize_level_specs(extra.get("crop_sizes", (51, 1201)))
    trial_id, rows = select_sequence_rows(cov, sampler, None, None, max(240, n_wm + int(cfg.history_size) + 8), int(cfg.history_size), down)
    eye_key = require_covariates(cov)
    clip = build_clip(
        cov,
        sampler,
        int(trial_id),
        rows,
        eye_key,
        crop_sizes,
        int(cfg.img_hw),
        str(extra.get("center_mode", "dset")),
        pyramid_mode=str(extra.get("pyramid_mode", "raw")),
        blur_sigmas=extra.get("blur_sigmas", None),
        laplacian_contrast=float(extra.get("laplacian_contrast", 1.0)),
        action_history=int(extra.get("action_history", max(1, int(cfg.action_dim) // 2))),
        downsample=down,
        robs_downsample_mode=str(extra.get("robs_downsample_mode", "sample")),
        covariate_downsample_mode=str(extra.get("covariate_downsample_mode", "sample")),
        pixel_normalization=str(extra.get("pixel_normalization", "unit")),
    )
    dev = torch.device(device)
    model.to(dev).eval()
    screen = screen_roi(sampler)
    rows_out = []
    n_windows = len(clip.rows) - int(cfg.history_size)
    chosen = np.unique(np.linspace(0, n_windows - 1, min(n_wm, n_windows), dtype=np.int64))
    for start in chosen:
        sub = clip.__class__(
            trial_id=clip.trial_id,
            rows=clip.rows[start : start + int(cfg.history_size) + 1],
            pixels=clip.pixels[start : start + int(cfg.history_size) + 1],
            action=clip.action[start : start + int(cfg.history_size) + 1],
            eyepos=clip.eyepos[start : start + int(cfg.history_size) + 1],
            robs=clip.robs[start : start + int(cfg.history_size) + 1],
            dfs=clip.dfs[start : start + int(cfg.history_size) + 1],
            t_bins=clip.t_bins[start : start + int(cfg.history_size) + 1],
            roi=clip.roi[start : start + int(cfg.history_size) + 1],
            dpi_valid=clip.dpi_valid[start : start + int(cfg.history_size) + 1],
        )
        batch = to_model_batch(sub, dev)
        result = compute_backimage_saliency(
            model,
            batch,
            mode="pred_output",
            method="grad_x_input",
            pred_index=-1,
            source_reduce="current",
        )
        source_row = int(sub.rows[int(cfg.history_size) - 1])
        rois = sampler.pyramid_rois_for_row(cov, source_row, crop_sizes=crop_sizes, center_mode=str(extra.get("center_mode", "dset")))
        heat = result.heatmaps[0].detach().cpu().numpy()
        speed = float(np.linalg.norm(sub.action[int(cfg.history_size) - 1, :2]) * float(target_hz))
        for li, roi in enumerate(rois):
            mask = resized_mask(roi, screen, int(cfg.img_hw))
            mass = float(heat[li].sum())
            on = float(heat[li][mask].sum())
            rows_out.append(
                {
                    "kind": "wm",
                    "trial": int(trial_id),
                    "row": source_row,
                    "state": "saccade" if speed >= threshold_deg_s else "fixation",
                    "level": int(li),
                    "level_label": level_labels(crop_sizes)[li],
                    "channel_pct": float(result.channel_pct[0, li].detach().cpu()),
                    "screen_area_frac": float(mask.mean()),
                    "screen_saliency_frac": on / max(mass, 1e-12),
                    "offscreen_saliency_frac": 1.0 - on / max(mass, 1e-12),
                }
            )

    if n_v1 > 0 and v1_predictions is not None and Path(v1_predictions).exists():
        state = load_v1_readout_saliency_state(
            readout_path=None,
            latents_path=None,
            predictions_path=v1_predictions,
            device=dev,
        )
        max_lag = max(int(x) for x in state.config.lag_set)
        lo = int(cfg.history_size) + max_lag
        hi = len(clip.rows) - 1
        chosen_v1 = np.unique(np.linspace(lo, hi, min(n_v1, max(1, hi - lo + 1)), dtype=np.int64))
        for target_index in chosen_v1:
            result = compute_v1_observed_saliency_for_clip(
                model,
                state,
                clip,
                target_index=int(target_index),
                to_model_batch=to_model_batch,
                method="grad_x_input",
                source_reduce="current",
                score_mode="poisson_loss",
                readout_lags=None,
                row_step=down,
            )
            if not bool(result["valid"][0].detach().cpu()):
                continue
            heat = result["heatmaps"].detach().cpu().numpy()
            speed = float(np.linalg.norm(clip.action[int(target_index) - 1, :2]) * float(target_hz))
            # V1 heat aggregates lags; average masks over the lagged source rows.
            rois_by_level = []
            for li in range(len(crop_sizes)):
                masks = []
                for lag in state.config.lag_set:
                    source_i = int(target_index) - int(lag) - 1
                    source_row = int(clip.rows[source_i])
                    rois = sampler.pyramid_rois_for_row(cov, source_row, crop_sizes=crop_sizes, center_mode=str(extra.get("center_mode", "dset")))
                    masks.append(resized_mask(rois[li], screen, int(cfg.img_hw)).astype(np.float32))
                rois_by_level.append(np.mean(np.stack(masks, axis=0), axis=0))
            for li, valid_prob in enumerate(rois_by_level):
                mass = float(heat[li].sum())
                on = float((heat[li] * valid_prob).sum())
                rows_out.append(
                    {
                        "kind": "v1",
                        "trial": int(trial_id),
                        "row": int(clip.rows[int(target_index)]),
                        "state": "saccade" if speed >= threshold_deg_s else "fixation",
                        "level": int(li),
                        "level_label": level_labels(crop_sizes)[li],
                        "channel_pct": float(result["channel_pct"][li].detach().cpu()),
                        "screen_area_frac": float(valid_prob.mean()),
                        "screen_saliency_frac": on / max(mass, 1e-12),
                        "offscreen_saliency_frac": 1.0 - on / max(mass, 1e-12),
                    }
                )

    write_csv(outdir / "saliency_artifact_rows.csv", rows_out)
    summary = {}
    for kind in sorted({r["kind"] for r in rows_out}):
        kind_rows = [r for r in rows_out if r["kind"] == kind]
        for level in sorted({r["level"] for r in kind_rows}):
            vals = np.asarray([r["offscreen_saliency_frac"] for r in kind_rows if r["level"] == level], dtype=np.float64)
            pct = np.asarray([r["channel_pct"] for r in kind_rows if r["level"] == level], dtype=np.float64)
            summary[f"{kind}_L{level}_offscreen_saliency"] = summarize_values(vals, "frac")
            summary[f"{kind}_L{level}_channel_pct"] = summarize_values(pct, "pct")
    write_json(outdir / "saliency_artifact_summary.json", summary)
    return summary


def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    ckpt = Path(args.checkpoint)
    model, cfg, extra = load_faithful_from_checkpoint(str(ckpt), map_location="cpu")
    outdir = Path(args.outdir) if args.outdir else ckpt.parent / "diagnostics"
    outdir.mkdir(parents=True, exist_ok=True)
    dset = load_backimage_dataset(BackImagePaths(args.session).dset_path)
    cov = dset.covariates
    sampler = BackImageSampler(args.session, dset)
    threshold = float(args.saccade_threshold_deg_s if args.saccade_threshold_deg_s is not None else extra.get("saccade_threshold_deg_s", 25.0))
    latents = Path(args.latents) if args.latents else None
    readout_model = Path(args.readout_model) if args.readout_model else None
    readout_predictions = Path(args.readout_predictions) if args.readout_predictions else None

    report = {
        "checkpoint": str(ckpt),
        "session": args.session,
        "outdir": str(outdir),
        "integrity": integrity_checks(
            cov=cov,
            sampler=sampler,
            session=args.session,
            extra=extra,
            cfg=cfg,
            latents_path=latents,
            outdir=outdir,
            n_samples=int(args.integrity_samples),
            seed=int(args.seed),
        ),
        "padding": padding_and_coverage(
            cov=cov,
            sampler=sampler,
            crop_sets=parse_crop_sets(args.candidate_crop_sets),
            extra=extra,
            outdir=outdir,
            row_sample_step=int(args.row_sample_step),
            seed=int(args.seed),
            threshold_deg_s=threshold,
        ),
        "latent": latent_dynamics_and_confound(
            cov=cov,
            sampler=sampler,
            cfg=cfg,
            extra=extra,
            latents_path=latents,
            outdir=outdir,
            max_rows=int(args.confound_max_rows),
            seed=int(args.seed),
            threshold_deg_s=threshold,
        ),
    }
    if not args.skip_readout_ablation:
        report["readout_ablation"] = readout_ablation(
            cfg=cfg,
            latents_path=latents,
            readout_model_path=readout_model,
            outdir=outdir,
            epochs=int(args.readout_ablation_epochs),
            patience=int(args.readout_ablation_patience),
            batch_size=int(args.readout_ablation_batch_size),
            device=args.device,
            seed=int(args.seed),
        )
    if not args.skip_saliency:
        report["saliency"] = saliency_artifact_diagnostic(
            cov=cov,
            sampler=sampler,
            model=model,
            cfg=cfg,
            extra=extra,
            outdir=outdir,
            session=args.session,
            v1_predictions=None if args.skip_v1_saliency else readout_predictions,
            n_wm=int(args.saliency_frames),
            n_v1=0 if args.skip_v1_saliency else int(args.v1_saliency_frames),
            threshold_deg_s=threshold,
            device=args.device,
        )
    write_json(outdir / "diagnostic_report.json", report)
    print(json.dumps(report, indent=2)[:12000])
    print(f"saved diagnostics: {outdir}", flush=True)


if __name__ == "__main__":
    main()
