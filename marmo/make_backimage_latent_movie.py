#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from marmo.backimage_sequences import (
    BackImagePaths,
    BackImageSampler,
    load_backimage_dataset,
    normalize_level_specs,
    normalize_pixel_values,
)
from marmo.fond_train_utils import load_model_from_checkpoint
from marmo.amortized_train_utils import load_amortized_from_checkpoint


SOURCE_HZ = 240


@dataclass(frozen=True)
class Clip:
    trial_id: int
    rows: np.ndarray
    pixels: np.ndarray
    action: np.ndarray
    eyepos: np.ndarray
    robs: np.ndarray
    dfs: np.ndarray
    t_bins: np.ndarray
    roi: np.ndarray
    dpi_valid: np.ndarray


def parse_args():
    p = argparse.ArgumentParser(description="Render a BackImage/FOND latent movie")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--session", default="Allen_2022-04-13")
    p.add_argument("--trial-id", type=int, default=None)
    p.add_argument("--start-row", type=int, default=None)
    p.add_argument("--duration-s", type=float, default=8.0)
    p.add_argument("--target-hz", type=int, default=None)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--center-mode", choices=["dset", "gaze"], default=None)
    p.add_argument("--crop-sizes", default=None)
    p.add_argument("--output-hw", type=int, default=None)
    p.add_argument("--pixel-normalization", choices=["unit", "visioncore"], default=None)
    p.add_argument("--robs-downsample-mode", choices=["sample", "sum"], default=None)
    p.add_argument("--covariate-downsample-mode", choices=["sample", "mean"], default=None)
    p.add_argument("--latent-kind", choices=["code", "eta"], default="code")
    p.add_argument("--latent-panel", choices=["rate_raster", "sample_raster", "zscore_heatmap"], default="rate_raster")
    p.add_argument("--latent-raster-quantile", type=float, default=0.9)
    p.add_argument("--latent-sample-scale", type=float, default=0.1)
    p.add_argument("--latent-sample-seed", type=int, default=0)
    p.add_argument("--max-render-frames", type=int, default=360)
    return p.parse_args()


def parse_crop_sizes(text: str | Sequence[int]):
    return normalize_level_specs(text)


def tensor_to_numpy(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def cov_array(cov: dict, key: str, rows: np.ndarray | None = None) -> np.ndarray:
    x = cov[key]
    if rows is not None:
        x = x[rows]
    return tensor_to_numpy(x)


def sample_covariate_array(
    cov: dict,
    key: str,
    rows: np.ndarray,
    *,
    downsample: int = 1,
    mode: str = "sample",
) -> np.ndarray:
    """Sample covariates the same way BackImageSequenceDataset does.

    Pixels stay anchored to ``rows``. Non-image covariates may represent the
    whole source-rate bin behind each target-rate row: spikes are usually
    summed, while eye/dfs/time covariates are usually averaged.
    """

    rows = np.asarray(rows, dtype=np.int64)
    mode = str(mode).lower()
    if int(downsample) <= 1 or mode == "sample":
        return cov_array(cov, key, rows)
    if mode not in {"sum", "mean"}:
        raise ValueError(f"Unsupported downsample mode {mode!r}")
    offsets = np.arange(int(downsample), dtype=np.int64)
    bin_rows = rows[:, None] + offsets[None, :]
    values = cov[key]
    if int(bin_rows.max()) >= len(values):
        raise IndexError(f"downsampled {key!r} rows exceed covariate length")
    if torch.is_tensor(values):
        index = torch.from_numpy(bin_rows).to(values.device)
        gathered = tensor_to_numpy(values[index])
    else:
        gathered = np.asarray(values)[bin_rows]
    if mode == "sum":
        return gathered.sum(axis=1)
    return gathered.astype(np.float32).mean(axis=1)


def require_covariates(cov: dict) -> str:
    required = ["trial_inds", "roi", "t_bins", "robs", "dpi_valid"]
    missing = [k for k in required if k not in cov]
    if missing:
        raise KeyError(f"backimage.dset is missing required covariates: {missing}")
    if "eyepos" in cov:
        return "eyepos"
    if "dpi_pix" in cov:
        warnings.warn("covariate 'eyepos' is missing; using 'dpi_pix' for action/eye traces")
        return "dpi_pix"
    raise KeyError("backimage.dset is missing both 'eyepos' and 'dpi_pix'")


def contiguous_runs(rows: np.ndarray) -> list[np.ndarray]:
    rows = np.asarray(rows, dtype=np.int64)
    if rows.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(rows) != 1) + 1
    return [r for r in np.split(rows, breaks) if r.size]


def first_true_run(mask: np.ndarray, length: int) -> int | None:
    count = 0
    for i, ok in enumerate(mask):
        count = count + 1 if bool(ok) else 0
        if count >= length:
            return i - length + 1
    return None


def downsample_factor(target_hz: int) -> int:
    if target_hz <= 0:
        raise ValueError("--target-hz must be positive")
    if SOURCE_HZ % target_hz != 0:
        raise ValueError(f"--target-hz must divide source rate {SOURCE_HZ}; got {target_hz}")
    return SOURCE_HZ // target_hz


def clip_frame_count(duration_s: float, target_hz: int) -> int:
    if duration_s <= 0:
        raise ValueError("--duration-s must be positive")
    return max(1, int(round(duration_s * target_hz)))


def select_clip_rows(
    cov: dict,
    sampler: BackImageSampler,
    trial_id: int | None,
    start_row: int | None,
    n_frames: int,
    downsample: int,
) -> tuple[int, np.ndarray]:
    trial_inds = cov_array(cov, "trial_inds").astype(np.int64)
    dpi_valid = cov_array(cov, "dpi_valid") > 0

    if start_row is not None:
        if start_row < 0 or start_row >= len(trial_inds):
            raise IndexError(f"--start-row {start_row} is outside backimage.dset length {len(trial_inds)}")
        found_trial = int(trial_inds[start_row])
        if trial_id is not None and found_trial != int(trial_id):
            raise ValueError(f"--start-row {start_row} belongs to trial {found_trial}, not --trial-id {trial_id}")
        if not sampler.has_image(found_trial):
            raise FileNotFoundError(f"trial {found_trial} source image is missing: {sampler.image_path(found_trial)}")
        rows = start_row + downsample * np.arange(n_frames, dtype=np.int64)
        if rows[-1] >= len(trial_inds):
            raise RuntimeError(f"clip from row {start_row} needs row {rows[-1]}, past dset length {len(trial_inds)}")
        if not np.all(trial_inds[rows] == found_trial):
            raise RuntimeError(f"clip from row {start_row} crosses a trial boundary before {n_frames} frames")
        bad = rows[~dpi_valid[rows]]
        if bad.size:
            raise RuntimeError(f"clip from row {start_row} includes {bad.size} invalid dpi rows; first bad row {bad[0]}")
        return found_trial, rows

    candidate_trials = [int(trial_id)] if trial_id is not None else sorted(int(t) for t in np.unique(trial_inds))
    skipped_missing = 0
    for tid in candidate_trials:
        if not sampler.has_image(tid):
            if trial_id is not None:
                raise FileNotFoundError(f"trial {tid} source image is missing: {sampler.image_path(tid)}")
            skipped_missing += 1
            continue
        trial_rows = np.flatnonzero(trial_inds == tid)
        for run in contiguous_runs(trial_rows):
            for offset in range(min(downsample, len(run))):
                sampled = run[offset::downsample]
                if len(sampled) < n_frames:
                    continue
                start = first_true_run(dpi_valid[sampled], n_frames)
                if start is not None:
                    return tid, sampled[start : start + n_frames]

    msg = f"No valid {n_frames}-frame clip found"
    if trial_id is not None:
        msg += f" for trial {trial_id}"
    if skipped_missing:
        msg += f"; skipped {skipped_missing} trials with missing source images"
    raise RuntimeError(msg)


def normalize_dfs_robs(robs: np.ndarray, dfs: np.ndarray) -> np.ndarray:
    robs = np.asarray(robs, dtype=np.float32)
    dfs = np.asarray(dfs, dtype=np.float32)
    if robs.ndim == 1:
        robs = robs[:, None]
    if dfs.ndim == 1:
        dfs = dfs[:, None]
    try:
        obs = dfs * robs
    except ValueError:
        if dfs.shape[0] == robs.shape[0] and dfs.shape[1] == 1:
            obs = robs * dfs
        else:
            raise
    return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)


def build_clip(
    cov: dict,
    sampler: BackImageSampler,
    trial_id: int,
    rows: np.ndarray,
    eye_key: str,
    crop_sizes: Sequence[int],
    output_hw: int,
    center_mode: str,
    pyramid_mode: str = "raw",
    blur_sigmas=None,
    laplacian_contrast: float = 1.0,
    action_history: int = 1,
    downsample: int = 1,
    robs_downsample_mode: str = "sample",
    covariate_downsample_mode: str = "sample",
    pixel_normalization: str = "unit",
) -> Clip:
    pixels = []
    l0_rois = []
    for row in rows:
        pyramid, rois = sampler.pyramid_and_rois_for_row(
            cov,
            int(row),
            crop_sizes=crop_sizes,
            output_hw=output_hw,
            center_mode=center_mode,
            pyramid_mode=pyramid_mode,
            blur_sigmas=blur_sigmas,
            laplacian_contrast=laplacian_contrast,
        )
        pixels.append(pyramid)
        l0_rois.append(rois[0])
    raw_pixels = np.stack(pixels, axis=0)
    pixels_np = normalize_pixel_values(raw_pixels, pixel_normalization)
    eyepos = sample_covariate_array(
        cov,
        eye_key,
        rows,
        downsample=downsample,
        mode=covariate_downsample_mode,
    ).astype(np.float32)
    base_action = np.zeros_like(eyepos, dtype=np.float32)
    if len(base_action) > 1:
        base_action[:-1] = eyepos[1:] - eyepos[:-1]
    if int(action_history) <= 1:
        action = base_action
    else:
        parts = []
        for lag in range(int(action_history)):
            shifted = np.zeros_like(base_action, dtype=np.float32)
            if lag == 0:
                shifted = base_action
            else:
                shifted[lag:] = base_action[:-lag]
            parts.append(shifted)
        action = np.concatenate(parts, axis=-1)
    action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
    robs = sample_covariate_array(
        cov,
        "robs",
        rows,
        downsample=downsample,
        mode=robs_downsample_mode,
    ).astype(np.float32)
    if "dfs" in cov:
        dfs = sample_covariate_array(
            cov,
            "dfs",
            rows,
            downsample=downsample,
            mode=covariate_downsample_mode,
        ).astype(np.float32)
    else:
        dfs = np.ones((len(rows), 1), dtype=np.float32)
    t_bins = sample_covariate_array(
        cov,
        "t_bins",
        rows,
        downsample=downsample,
        mode=covariate_downsample_mode,
    ).astype(np.float64)
    return Clip(
        trial_id=trial_id,
        rows=rows.astype(np.int64),
        pixels=pixels_np,
        action=action,
        eyepos=eyepos,
        robs=robs,
        dfs=dfs,
        t_bins=t_bins,
        roi=np.stack(l0_rois, axis=0).astype(np.float64),
        dpi_valid=(cov_array(cov, "dpi_valid", rows) > 0),
    )


def to_model_batch(clip: Clip, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "pixels": torch.from_numpy(clip.pixels).unsqueeze(0).to(device),
        "action": torch.from_numpy(clip.action).unsqueeze(0).to(device),
        "eyepos": torch.from_numpy(clip.eyepos).unsqueeze(0).to(device),
        "robs": torch.from_numpy(clip.robs).unsqueeze(0).to(device),
        "dfs": torch.from_numpy(clip.dfs).unsqueeze(0).to(device),
        "t_bins": torch.from_numpy(clip.t_bins.astype(np.float32)).unsqueeze(0).to(device),
    }


def load_any_model(checkpoint: Path):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    kind = ckpt.get("model_kind", "fond")
    if kind == "amortized":
        model, cfg, extra = load_amortized_from_checkpoint(str(checkpoint), map_location="cpu")
        return model, cfg, extra, "amortized"
    model, cfg, extra = load_model_from_checkpoint(str(checkpoint), map_location="cpu")
    return model, cfg, extra, "fond"


def run_model_sequence(model, cfg, clip: Clip, device: torch.device, model_kind: str):
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    batch = to_model_batch(clip, device)
    if model_kind == "amortized":
        with torch.no_grad():
            encoded = model.encode(batch)
            eta = encoded["emb"].detach()
            code = model.deterministic_code(eta).detach()
            torch.manual_seed(0)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(0)
            recon = model.decode(eta).detach().clamp(0.0, 1.0)
        return eta.cpu().numpy()[0], code.cpu().numpy()[0], recon.cpu().numpy()[0]

    if hasattr(model, "_set_runtime_inference"):
        model._set_runtime_inference(infer_backprop=False)
    with torch.enable_grad():
        info = model.filter_sequence(
            batch,
            history_size=cfg.history_size,
            beta=cfg.beta,
            infer_objective=cfg.infer_objective,
            return_diag=False,
        )
    eta = info["emb"].detach()
    code = model.head.to_code(eta.reshape(-1, eta.shape[-1])).reshape(*eta.shape[:-1], -1).detach()
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)
    with torch.no_grad():
        recon = model.decode(eta).detach().clamp(0.0, 1.0)
    return eta.cpu().numpy()[0], code.cpu().numpy()[0], recon.cpu().numpy()[0]


def normalize_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image).astype(np.float32)
    if arr.size == 0:
        return arr
    if np.nanmax(arr) > 1.5:
        arr = arr / 255.0
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(arr, 0.0, 1.0) ** 0.8


def trial_src_pos_ij(sampler: BackImageSampler, trial_id: int) -> np.ndarray:
    trial = sampler.trial(trial_id)
    return np.flipud(np.asarray(trial.dest_rect[:2], dtype=np.float64))


def point_inside_fraction(points_ij: np.ndarray, image_shape: tuple[int, int]) -> float:
    points = np.asarray(points_ij, dtype=np.float64)
    finite = np.isfinite(points).all(axis=1)
    if not finite.any():
        return -1.0
    h, w = image_shape
    inside = (
        (points[:, 0] >= 0)
        & (points[:, 0] < h)
        & (points[:, 1] >= 0)
        & (points[:, 1] < w)
        & finite
    )
    return float(inside.sum() / finite.sum())


def choose_roi_image_coords(roi: np.ndarray, src_pos_ij: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    raw = np.asarray(roi, dtype=np.float64)
    return raw - src_pos_ij[:, None]


def choose_gaze_image_coords(
    cov: dict,
    rows: np.ndarray,
    clip: Clip,
    roi_img: np.ndarray,
    src_pos_ij: np.ndarray,
    image_shape: tuple[int, int],
    sampler: BackImageSampler | None = None,
) -> tuple[np.ndarray, str]:
    roi_centers = roi_img.mean(axis=2)
    best_points = roi_centers
    best_name = "roi center"
    best_score = -math.inf
    candidates: list[tuple[str, np.ndarray]] = []
    if sampler is not None:
        try:
            gaze_screen = np.stack([sampler.gaze_center_ij(cov, int(row)) for row in rows], axis=0)
            candidates.append(("eyepos->image", gaze_screen - src_pos_ij[None, :]))
        except KeyError:
            pass
    if "dpi_pix" in cov:
        raw = cov_array(cov, "dpi_pix", rows).astype(np.float64)
        if raw.ndim == 2 and raw.shape[1] == 2:
            candidates.append(("dpi_pix->image", raw - src_pos_ij[None, :]))

    for name, points in candidates:
        inside = point_inside_fraction(points, image_shape)
        dist = np.linalg.norm(points - roi_centers, axis=1)
        dist = dist[np.isfinite(dist)]
        med_dist = float(np.median(dist)) if dist.size else max(image_shape)
        score = inside - med_dist / max(image_shape)
        if score > best_score:
            best_score = score
            best_points = points
            best_name = name
    return best_points, best_name


def relative_time(t_bins: np.ndarray, target_hz: int) -> np.ndarray:
    t = np.asarray(t_bins, dtype=np.float64)
    if t.size == 0 or not np.isfinite(t).all():
        return np.arange(len(t), dtype=np.float64) / float(target_hz)
    t = t - t[0]
    if len(t) > 1 and t[-1] > 0:
        return t
    return np.arange(len(t), dtype=np.float64) / float(target_hz)


def heatmap_extent(t_rel: np.ndarray, n_rows: int, target_hz: int) -> list[float]:
    if len(t_rel) <= 1:
        t0, t1 = 0.0, 1.0 / float(target_hz)
    else:
        t0, t1 = float(t_rel[0]), float(t_rel[-1])
    return [t0, t1, 0, max(1, int(n_rows))]


def sort_latents(latents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(latents, dtype=np.float32)
    if z.ndim == 1:
        z = z[:, None]
    var = np.nanvar(z, axis=0)
    order = np.argsort(var)[::-1]
    return np.nan_to_num(z[:, order], nan=0.0, posinf=0.0, neginf=0.0), order


def zscore_latents(latents: np.ndarray) -> np.ndarray:
    z = np.asarray(latents, dtype=np.float32)
    mean = np.nanmean(z, axis=0, keepdims=True)
    std = np.nanstd(z, axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    z = (z - mean) / std
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def latent_events_from_rates(
    latent: np.ndarray,
    panel: str,
    quantile: float,
    sample_scale: float,
    sample_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(latent, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if panel == "sample_raster":
        rng = np.random.default_rng(sample_seed)
        lam = np.clip(arr, 0.0, None) * max(0.0, float(sample_scale))
        counts = rng.poisson(lam).astype(np.float32)
        return counts > 0, counts
    q = float(np.clip(quantile, 0.0, 1.0))
    thresh = np.nanquantile(arr, q, axis=0, keepdims=True)
    events = arr >= thresh
    events &= np.isfinite(arr)
    return events, arr


def render_frame_ids(n_frames: int, duration_s: float, fps: int, max_render_frames: int | None) -> np.ndarray:
    if fps <= 0:
        raise ValueError("--fps must be positive")
    ideal = min(n_frames, max(1, int(round(duration_s * fps))))
    if max_render_frames is not None:
        if max_render_frames <= 0:
            raise ValueError("--max-render-frames must be positive")
        ideal = min(ideal, int(max_render_frames))
    return np.unique(np.linspace(0, n_frames - 1, ideal, dtype=np.int64))


def robust_vmax(values: np.ndarray, default: float = 1.0) -> float:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return default
    positive = finite[finite > 0]
    base = positive if positive.size else finite
    vmax = float(np.percentile(base, 99))
    return vmax if vmax > 0 else default


def make_display_payload(
    cov: dict,
    sampler: BackImageSampler,
    clip: Clip,
    eta: np.ndarray,
    code: np.ndarray,
    recon: np.ndarray,
    latent_kind: str,
    target_hz: int,
    latent_panel: str,
    latent_raster_quantile: float,
    latent_sample_scale: float,
    latent_sample_seed: int,
):
    full_image = normalize_image(sampler.screen_canvas(clip.trial_id))
    image_shape = full_image.shape[:2]
    screen_origin = sampler.screen_roi()[:, 0].astype(np.float64)
    roi_img = np.asarray(clip.roi, dtype=np.float64) - screen_origin[:, None]
    gaze_img = np.stack([sampler.gaze_center_ij(cov, int(row)) for row in clip.rows], axis=0) - screen_origin[None, :]
    gaze_name = "eyepos->screen"
    latent = code if latent_kind == "code" else eta
    latent_sorted, latent_order = sort_latents(latent)
    latent_z = zscore_latents(latent_sorted)
    latent_events, latent_event_values = latent_events_from_rates(
        latent_sorted,
        latent_panel,
        latent_raster_quantile,
        latent_sample_scale,
        latent_sample_seed,
    )
    obs = normalize_dfs_robs(clip.robs, clip.dfs)
    t_rel = relative_time(clip.t_bins, target_hz)
    return {
        "full_image": full_image,
        "roi_img": roi_img,
        "gaze_img": gaze_img,
        "gaze_name": gaze_name,
        "target_l0": clip.pixels[:, 0],
        "recon_l0": recon[:, 0],
        "obs": obs,
        "latent_z": latent_z,
        "latent_raw": latent_sorted,
        "latent_events": latent_events,
        "latent_event_values": latent_event_values,
        "latent_order": latent_order,
        "latent_panel": latent_panel,
        "latent_raster_quantile": latent_raster_quantile,
        "latent_sample_scale": latent_sample_scale,
        "t_rel": t_rel,
    }


def init_figure(payload: dict, clip: Clip, latent_kind: str, target_hz: int):
    full_image = payload["full_image"]
    obs = payload["obs"]
    latent_z = payload["latent_z"]
    t_rel = payload["t_rel"]
    fig = plt.figure(figsize=(15.0, 7.2), dpi=140)
    gs = GridSpec(1, 2, width_ratios=[1.3, 1.0], wspace=0.12, figure=fig)

    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(full_image, cmap="gray", origin="upper", vmin=0, vmax=1)
    ax_img.set_axis_off()
    ax_img.set_title(f"trial {clip.trial_id}  rows {clip.rows[0]}-{clip.rows[-1]}", fontsize=11)
    rect = Rectangle((0, 0), 1, 1, ec="red", fc="none", lw=2)
    ax_img.add_patch(rect)
    crop_center, = ax_img.plot([], [], "x", color="red", ms=7, mew=1.5)
    gaze_path, = ax_img.plot([], [], color="cyan", lw=1.4, alpha=0.9, solid_capstyle="round")
    gaze_dot, = ax_img.plot([], [], "o", color="cyan", ms=5)

    ax_target = inset_axes(ax_img, width="28%", height="28%", loc="lower left", borderpad=0.8)
    im_target = ax_target.imshow(payload["target_l0"][0], cmap="gray", origin="upper", vmin=0, vmax=1)
    ax_target.set_title("target L0", fontsize=8, color="red", pad=2)
    ax_target.set_xticks([])
    ax_target.set_yticks([])
    for spine in ax_target.spines.values():
        spine.set_color("red")
        spine.set_linewidth(1.6)

    ax_recon = inset_axes(ax_img, width="28%", height="28%", loc="lower right", borderpad=0.8)
    im_recon = ax_recon.imshow(payload["recon_l0"][0], cmap="gray", origin="upper", vmin=0, vmax=1)
    ax_recon.set_title("recon L0", fontsize=8, color="cyan", pad=2)
    ax_recon.set_xticks([])
    ax_recon.set_yticks([])
    for spine in ax_recon.spines.values():
        spine.set_color("cyan")
        spine.set_linewidth(1.6)

    gs_r = gs[0, 1].subgridspec(3, 1, height_ratios=[1.05, 1.05, 0.65], hspace=0.13)
    ax_obs = fig.add_subplot(gs_r[0, 0])
    ax_lat = fig.add_subplot(gs_r[1, 0], sharex=ax_obs)
    ax_eye = fig.add_subplot(gs_r[2, 0], sharex=ax_obs)

    obs_t, obs_unit = np.nonzero(obs > 0)
    if obs_t.size:
        sizes = 3.0 + 2.0 * np.clip(obs[obs_t, obs_unit], 0, 3)
        ax_obs.scatter(t_rel[obs_t], obs_unit, s=sizes, c="black", marker="|", linewidths=0.7)
    ax_obs.set_xlim(float(t_rel[0]), float(t_rel[-1]))
    ax_obs.set_ylim(-0.5, max(0.5, obs.shape[1] - 0.5))

    if payload["latent_panel"] == "zscore_heatmap":
        lat_extent = heatmap_extent(t_rel, latent_z.shape[1], target_hz)
        lat_lim = max(1.0, min(3.5, float(np.percentile(np.abs(latent_z), 99)))) if latent_z.size else 1.0
        ax_lat.imshow(
            latent_z.T,
            aspect="auto",
            interpolation="none",
            cmap="gray",
            origin="lower",
            vmin=-lat_lim,
            vmax=lat_lim,
            extent=lat_extent,
        )
        lat_title = f"{latent_kind} latents, z-scored heatmap"
    else:
        ev_t, ev_dim = np.nonzero(payload["latent_events"])
        if ev_t.size:
            values = payload["latent_event_values"][ev_t, ev_dim]
            sizes = 3.0 + 1.2 * np.clip(values, 0, 3)
            ax_lat.scatter(t_rel[ev_t], ev_dim, s=sizes, c="black", marker="|", linewidths=0.7)
        ax_lat.set_ylim(-0.5, max(0.5, payload["latent_raw"].shape[1] - 0.5))
        if payload["latent_panel"] == "sample_raster":
            lat_title = f"sampled Poisson latent events ({latent_kind}, scale={payload['latent_sample_scale']:g})"
        else:
            lat_title = f"Poisson latent high-rate events ({latent_kind}, q>={payload['latent_raster_quantile']:.2f})"

    ax_obs.set_title("observed V1 spikes", fontsize=10)
    ax_lat.set_title(lat_title, fontsize=10)
    ax_obs.set_ylabel("unit")
    ax_lat.set_ylabel("latent")
    ax_obs.tick_params(labelbottom=False)
    ax_lat.tick_params(labelbottom=False)

    eyepos = np.asarray(clip.eyepos, dtype=np.float64)
    ax_eye.plot(t_rel, eyepos[:, 1], color="black", lw=0.9, label="eye x")
    ax_eye.plot(t_rel, eyepos[:, 0], color="0.45", lw=0.9, label="eye y")
    finite_eye = eyepos[np.isfinite(eyepos).all(axis=1)]
    if finite_eye.size:
        lo, hi = np.percentile(finite_eye, [1, 99])
        pad = max(1.0, 0.08 * (hi - lo))
        ax_eye.set_ylim(lo - pad, hi + pad)
    ax_eye.set_ylabel("px")
    ax_eye.set_xlabel("time (s)")
    ax_eye.legend(frameon=False, fontsize=7, ncol=2, loc="upper right")

    vlines = [
        ax_obs.axvline(t_rel[0], color="red", ls="--", lw=1.4),
        ax_lat.axvline(t_rel[0], color="red", ls="--", lw=1.4),
        ax_eye.axvline(t_rel[0], color="red", ls="--", lw=1.4),
    ]
    for ax in (ax_obs, ax_lat, ax_eye):
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    artists = {
        "rect": rect,
        "crop_center": crop_center,
        "gaze_path": gaze_path,
        "gaze_dot": gaze_dot,
        "im_target": im_target,
        "im_recon": im_recon,
        "vlines": vlines,
    }
    return fig, artists


def update_artists(artists: dict, payload: dict, k: int, target_hz: int):
    roi = payload["roi_img"][k]
    i0, i1 = roi[0]
    j0, j1 = roi[1]
    artists["rect"].set_xy((j0, i0))
    artists["rect"].set_width(j1 - j0)
    artists["rect"].set_height(i1 - i0)
    crop_center = roi.mean(axis=1)
    artists["crop_center"].set_data([crop_center[1]], [crop_center[0]])

    tail = max(5, int(round(0.5 * target_hz)))
    start = max(0, k - tail + 1)
    gaze = payload["gaze_img"][start : k + 1]
    finite = np.isfinite(gaze).all(axis=1)
    if finite.any():
        artists["gaze_path"].set_data(gaze[finite, 1], gaze[finite, 0])
        current = payload["gaze_img"][k]
        if np.isfinite(current).all():
            artists["gaze_dot"].set_data([current[1]], [current[0]])
        else:
            artists["gaze_dot"].set_data([], [])
    else:
        artists["gaze_path"].set_data([], [])
        artists["gaze_dot"].set_data([], [])

    artists["im_target"].set_array(payload["target_l0"][k])
    artists["im_recon"].set_array(payload["recon_l0"][k])
    t_now = float(payload["t_rel"][k])
    for vline in artists["vlines"]:
        vline.set_xdata([t_now, t_now])


def render_movie(out_path: Path, payload: dict, clip: Clip, latent_kind: str, target_hz: int, fps: int, frame_ids: np.ndarray):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, artists = init_figure(payload, clip, latent_kind, target_hz)
    update_artists(artists, payload, int(frame_ids[0]), target_hz)

    try:
        if FFMpegWriter.isAvailable():
            writer = FFMpegWriter(
                fps=fps,
                codec="libx264",
                bitrate=16000,
                extra_args=["-pix_fmt", "yuv420p"],
            )
            with writer.saving(fig, str(out_path), dpi=fig.dpi):
                for k in frame_ids:
                    update_artists(artists, payload, int(k), target_hz)
                    writer.grab_frame()
        else:
            import imageio.v2 as imageio

            with imageio.get_writer(str(out_path), fps=fps, codec="libx264", macro_block_size=16) as writer:
                for k in frame_ids:
                    update_artists(artists, payload, int(k), target_hz)
                    fig.canvas.draw()
                    frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
                    writer.append_data(frame)
    finally:
        plt.close(fig)


def save_contact_sheet(path: Path, payload: dict, clip: Clip, frame_ids: np.ndarray):
    n = min(8, len(frame_ids))
    if n <= 0:
        return
    take = frame_ids[np.linspace(0, len(frame_ids) - 1, n, dtype=np.int64)]
    fig, axes = plt.subplots(2, n, figsize=(1.7 * n, 3.4), squeeze=False)
    for col, k in enumerate(take):
        k = int(k)
        axes[0, col].imshow(payload["target_l0"][k], cmap="gray", vmin=0, vmax=1)
        axes[1, col].imshow(payload["recon_l0"][k], cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"{payload['t_rel'][k]:.2f}s\nrow {clip.rows[k]}", fontsize=8)
        for ax in axes[:, col]:
            ax.set_xticks([])
            ax.set_yticks([])
    axes[0, 0].set_ylabel("target", fontsize=9)
    axes[1, 0].set_ylabel("recon", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def default_out_path(checkpoint: Path, session: str, trial_id: int, start_row: int, latent_kind: str) -> Path:
    name = f"{checkpoint.stem}_{session}_trial{trial_id}_row{start_row}_{latent_kind}_latent_movie.mp4"
    return checkpoint.with_name(name)


def resolve_out_path(out_arg: str | None, checkpoint: Path, session: str, trial_id: int, start_row: int, latent_kind: str) -> Path:
    if out_arg is None:
        return default_out_path(checkpoint, session, trial_id, start_row, latent_kind)
    out = Path(out_arg)
    if out.suffix.lower() != ".mp4":
        out = out.with_suffix(".mp4")
    return out


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device requests CUDA, but torch.cuda.is_available() is false")

    model, cfg, extra, model_kind = load_any_model(checkpoint)
    target_hz = int(args.target_hz if args.target_hz is not None else extra.get("target_hz", 120))
    center_mode = args.center_mode or extra.get("center_mode", "dset")
    robs_downsample_mode = args.robs_downsample_mode or extra.get("robs_downsample_mode", "sample")
    covariate_downsample_mode = args.covariate_downsample_mode or extra.get("covariate_downsample_mode", "sample")
    pixel_normalization = args.pixel_normalization or extra.get("pixel_normalization", "unit")
    crop_sizes = parse_crop_sizes(args.crop_sizes) if args.crop_sizes else parse_crop_sizes(extra.get("crop_sizes", (51, 101, 201)))
    output_hw = int(args.output_hw if args.output_hw is not None else cfg.img_hw)
    if len(crop_sizes) != int(cfg.img_ch):
        raise ValueError(f"crop sizes define {len(crop_sizes)} channels, but checkpoint model expects img_ch={cfg.img_ch}")
    if output_hw != int(cfg.img_hw):
        raise ValueError(f"output_hw={output_hw} does not match checkpoint img_hw={cfg.img_hw}")

    downsample = downsample_factor(target_hz)
    n_clip_frames = clip_frame_count(args.duration_s, target_hz)
    dset_path = BackImagePaths(args.session).dset_path
    dset = load_backimage_dataset(dset_path)
    cov = dset.covariates
    eye_key = require_covariates(cov)
    if center_mode == "gaze" and "dpi_pix" not in cov:
        raise KeyError("--center-mode gaze requires covariate 'dpi_pix' for BackImageSampler")
    sampler = BackImageSampler(args.session, dset)
    trial_id, rows = select_clip_rows(cov, sampler, args.trial_id, args.start_row, n_clip_frames, downsample)

    print(f"checkpoint: {checkpoint}")
    print(f"model kind: {model_kind}")
    print(f"dset: {dset_path}")
    print(
        f"clip: trial={trial_id} start_row={rows[0]} frames={len(rows)} "
        f"duration={args.duration_s:g}s target_hz={target_hz} downsample={downsample}"
    )
    print(f"pixels: crop_sizes={crop_sizes} output_hw={output_hw} center_mode={center_mode}")
    print(
        "covariates: "
        f"robs_downsample_mode={robs_downsample_mode} "
        f"covariate_downsample_mode={covariate_downsample_mode} "
        f"pixel_normalization={pixel_normalization}"
    )

    clip = build_clip(
        cov,
        sampler,
        trial_id,
        rows,
        eye_key,
        crop_sizes,
        output_hw,
        center_mode,
        downsample=downsample,
        robs_downsample_mode=robs_downsample_mode,
        covariate_downsample_mode=covariate_downsample_mode,
        pixel_normalization=pixel_normalization,
    )
    eta, code, recon = run_model_sequence(model, cfg, clip, device, model_kind)
    payload = make_display_payload(
        cov,
        sampler,
        clip,
        eta,
        code,
        recon,
        args.latent_kind,
        target_hz,
        args.latent_panel,
        args.latent_raster_quantile,
        args.latent_sample_scale,
        args.latent_sample_seed,
    )
    frame_ids = render_frame_ids(len(rows), args.duration_s, args.fps, args.max_render_frames)
    out_path = resolve_out_path(args.out, checkpoint, args.session, trial_id, int(rows[0]), args.latent_kind)
    contact_path = out_path.with_name(f"{out_path.stem}_contact.png")

    print(f"render: {len(frame_ids)} frames at {args.fps} fps -> {out_path}")
    print(f"gaze overlay source: {payload['gaze_name']}")
    crop_centers = payload["roi_img"].mean(axis=2)
    gaze = payload["gaze_img"]
    finite = np.isfinite(crop_centers).all(axis=1) & np.isfinite(gaze).all(axis=1)
    if finite.any():
        median_gaze_offset = np.median(gaze[finite] - crop_centers[finite], axis=0)
        print(
            "median gaze minus crop center: "
            f"di={median_gaze_offset[0]:.2f}px dj={median_gaze_offset[1]:.2f}px"
        )
    render_movie(out_path, payload, clip, args.latent_kind, target_hz, args.fps, frame_ids)
    save_contact_sheet(contact_path, payload, clip, frame_ids)
    print(f"saved movie: {out_path}")
    print(f"saved contact sheet: {contact_path}")


if __name__ == "__main__":
    main()
