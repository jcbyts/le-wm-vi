from __future__ import annotations

import argparse
import sys
from pathlib import Path

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

from marmo.backimage_sequences import (
    BackImagePaths,
    BackImageSampler,
    level_spec_label,
    load_backimage_dataset,
    normalize_level_specs,
)
from marmo.faithful_train_utils import faithful_forward, load_faithful_from_checkpoint
from marmo.make_backimage_latent_movie import (
    build_clip,
    choose_gaze_image_coords,
    choose_roi_image_coords,
    clip_frame_count,
    downsample_factor,
    normalize_dfs_robs,
    normalize_image,
    relative_time,
    render_frame_ids,
    require_covariates,
    select_clip_rows,
    sort_latents,
    to_model_batch,
    trial_src_pos_ij,
)
from marmo.saliency_utils import compute_backimage_saliency
from marmo.v1_saliency_utils import (
    compute_v1_observed_saliency_for_clip,
    load_v1_readout_saliency_state,
)


def parse_args():
    p = argparse.ArgumentParser(description="Render faithful Gaussian BackImage LeWM saliency movie")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--session", default=None)
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
    p.add_argument("--pyramid-mode", choices=["raw", "gaussian", "hybrid_l0_gaussian", "laplacian"], default=None)
    p.add_argument("--blur-sigmas", default=None)
    p.add_argument("--laplacian-contrast", type=float, default=None)
    p.add_argument("--action-history", type=int, default=None)
    p.add_argument("--robs-downsample-mode", choices=["sample", "sum"], default=None)
    p.add_argument("--covariate-downsample-mode", choices=["sample", "mean"], default=None)
    p.add_argument("--pixel-normalization", choices=["unit", "visioncore"], default=None)
    p.add_argument("--saliency-mode", choices=["pred_output", "pred_loss"], default="pred_output")
    p.add_argument("--saliency-method", choices=["grad", "grad_x_input", "integrated_gradients"], default="grad_x_input")
    p.add_argument("--ig-steps", type=int, default=16)
    p.add_argument("--ig-baseline", choices=["gray", "zero", "channel_mean"], default="gray")
    p.add_argument("--saliency-source", choices=["current", "context_sum"], default="current")
    p.add_argument("--readout-predictions", default=None, help="Optional readout_predictions_full.npz for a V1 prediction panel")
    p.add_argument("--v1-saliency-readout", default=None, help="Optional best_model.pt readout for observed V1 saliency")
    p.add_argument("--v1-saliency-latents", default=None, help="Optional latents_full.npz used by the V1 readout")
    p.add_argument("--v1-saliency-method", choices=["grad", "grad_x_input"], default="grad_x_input")
    p.add_argument("--v1-saliency-score", choices=["poisson_loss", "spike_lograte", "poisson_loglik", "rate_sum"], default="poisson_loss")
    p.add_argument("--v1-saliency-lags", default="all", help="Readout lags to attribute, e.g. all or 2,3,4,5")
    p.add_argument("--v1-saliency-row-step", type=int, default=0, help="Raw dataset rows per latent step; 0 reads it from the readout")
    p.add_argument("--v1-saliency-source", choices=["current", "context_sum"], default="context_sum")
    p.add_argument("--saccade-threshold-deg-s", type=float, default=None)
    p.add_argument("--max-render-frames", type=int, default=360)
    return p.parse_args()


def parse_crop_sizes(text: str | tuple[int, ...]):
    return normalize_level_specs(text)


def parse_v1_lags(text: str):
    text = str(text).strip().lower()
    if text in {"", "all", "auto"}:
        return None
    return tuple(int(x.strip()) for x in text.split(",") if x.strip())


def select_sequence_rows(cov, sampler, trial_id, start_row, n_display_frames, history_size, downsample):
    n_clip_frames = n_display_frames + history_size
    tid, rows = select_clip_rows(cov, sampler, trial_id, start_row, n_clip_frames, downsample)
    return tid, rows


def run_faithful_sequence(
    model,
    clip,
    device,
    saliency_mode,
    saliency_method,
    saliency_source,
    ig_steps,
    ig_baseline,
    *,
    v1_state=None,
    v1_method="grad_x_input",
    v1_score_mode="spike_lograte",
    v1_lags=None,
    v1_row_step=2,
    v1_source="context_sum",
):
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    hist = int(model.cfg.history_size)
    n_windows = len(clip.rows) - hist
    if n_windows <= 0:
        raise ValueError("clip is too short for model history")
    latents = []
    pred_hat = []
    pred_loss = []
    sal = []
    pct = []
    v1_sal = []
    v1_pct = []
    v1_score = []
    v1_valid = []
    v1_rows = []
    for start in range(n_windows):
        sub_clip = clip.__class__(
            trial_id=clip.trial_id,
            rows=clip.rows[start : start + hist + 1],
            pixels=clip.pixels[start : start + hist + 1],
            action=clip.action[start : start + hist + 1],
            eyepos=clip.eyepos[start : start + hist + 1],
            robs=clip.robs[start : start + hist + 1],
            dfs=clip.dfs[start : start + hist + 1],
            t_bins=clip.t_bins[start : start + hist + 1],
            roi=clip.roi[start : start + hist + 1],
            dpi_valid=clip.dpi_valid[start : start + hist + 1],
        )
        batch = to_model_batch(sub_clip, device)
        with torch.no_grad():
            out = faithful_forward(model, batch)
            emb = out["emb"][0, -1].detach().cpu().numpy()
            pred = out["pred_hat"][0, -1].detach().cpu().numpy()
            pl = (out["pred_hat"][0, -1] - out["target"][0, -1]).pow(2).mean().detach().cpu().item()
        with torch.enable_grad():
            result = compute_backimage_saliency(
                model,
                batch,
                mode=saliency_mode,
                method=saliency_method,
                pred_index=-1,
                baseline=ig_baseline,
                ig_steps=ig_steps,
                source_reduce=saliency_source,
            )
        if v1_state is not None:
            with torch.enable_grad():
                v1_result = compute_v1_observed_saliency_for_clip(
                    model,
                    v1_state,
                    clip,
                    target_index=int(start + hist),
                    to_model_batch=to_model_batch,
                    method=v1_method,
                    source_reduce=v1_source,
                    score_mode=v1_score_mode,
                    readout_lags=v1_lags,
                    row_step=int(v1_row_step),
                )
            v1_sal.append(v1_result["heatmaps"].detach().cpu().numpy())
            v1_pct.append(v1_result["channel_pct"].detach().cpu().numpy())
            v1_score.append(float(v1_result["score"][0].detach().cpu()))
            v1_valid.append(bool(v1_result["valid"][0].detach().cpu()))
            v1_rows.append(tuple(int(x) for x in v1_result["score_rows"]))
        latents.append(emb)
        pred_hat.append(pred)
        pred_loss.append(pl)
        sal.append(result.heatmaps[0].detach().cpu().numpy())
        pct.append(result.channel_pct[0].detach().cpu().numpy())
    out = {
        "latent": np.stack(latents, axis=0),
        "pred_hat": np.stack(pred_hat, axis=0),
        "pred_loss": np.asarray(pred_loss, dtype=np.float32),
        "saliency": np.stack(sal, axis=0),
        "channel_pct": np.stack(pct, axis=0),
    }
    if v1_state is not None:
        out["v1_saliency"] = np.stack(v1_sal, axis=0)
        out["v1_channel_pct"] = np.stack(v1_pct, axis=0)
        out["v1_score"] = np.asarray(v1_score, dtype=np.float32)
        out["v1_valid"] = np.asarray(v1_valid, dtype=bool)
        out["v1_score_rows"] = np.asarray(v1_rows, dtype=object)
    return out


def load_readout_clip(path: str | Path | None, trial_id: int, rows: np.ndarray, n_obs_units: int):
    if path is None:
        return None
    data = np.load(Path(path), allow_pickle=True)
    pred_rows = data["row_indices"].astype(np.int64)
    pred_trials = data["trial_inds"].astype(np.int64)
    rate = data["rate"].astype(np.float32)
    unit_mask = data["unit_mask"].astype(bool)
    lookup = {(int(t), int(r)): i for i, (t, r) in enumerate(zip(pred_trials, pred_rows, strict=False))}
    aligned = np.full((len(rows), rate.shape[1]), np.nan, dtype=np.float32)
    for i, row in enumerate(rows.astype(np.int64)):
        j = lookup.get((int(trial_id), int(row)))
        if j is not None:
            aligned[i] = rate[j]
    if unit_mask.shape[0] != int(n_obs_units):
        raise ValueError(
            f"readout unit_mask has {unit_mask.shape[0]} units, but observed spikes have {n_obs_units}"
        )
    return {
        "rate": aligned,
        "unit_mask": unit_mask,
        "path": str(path),
    }


def prepare_payload(
    cov,
    sampler,
    clip,
    model_out,
    target_hz,
    history_size,
    saliency_mode,
    saliency_method,
    saliency_source,
    threshold_deg_s,
    crop_sizes,
    ig_baseline,
    readout_predictions=None,
    v1_saliency_source=None,
    v1_saliency_score=None,
):
    source_slice = slice(history_size - 1, -1)
    target_slice = slice(history_size, None)
    full_image = normalize_image(sampler.screen_canvas(clip.trial_id))
    image_shape = full_image.shape[:2]
    screen_origin = sampler.screen_roi()[:, 0].astype(np.float64)
    roi_img = np.asarray(clip.roi[source_slice], dtype=np.float64) - screen_origin[:, None]
    gaze_img = np.stack(
        [sampler.gaze_center_ij(cov, int(row)) for row in clip.rows[source_slice]],
        axis=0,
    ) - screen_origin[None, :]
    gaze_name = "eyepos->screen"
    obs = normalize_dfs_robs(clip.robs[target_slice], clip.dfs[target_slice])
    readout = load_readout_clip(
        readout_predictions,
        clip.trial_id,
        clip.rows[target_slice],
        obs.shape[1],
    )
    if readout is not None:
        obs = obs[:, readout["unit_mask"]]
    latent_sorted, latent_order = sort_latents(model_out["latent"])
    latent_z = latent_sorted
    source_pyramid = clip.pixels[source_slice]
    action = clip.action[source_slice]
    speed = np.linalg.norm(action, axis=1) * float(target_hz)
    t_rel = relative_time(clip.t_bins[target_slice], target_hz)
    return {
        "full_image": full_image,
        "roi_img": roi_img,
        "gaze_img": gaze_img,
        "gaze_name": gaze_name,
        "obs": obs,
        "readout_rate": None if readout is None else readout["rate"],
        "readout_path": None if readout is None else readout["path"],
        "source_pyramid": source_pyramid,
        "saliency": model_out["saliency"],
        "channel_pct": model_out["channel_pct"],
        "v1_saliency": model_out.get("v1_saliency"),
        "v1_channel_pct": model_out.get("v1_channel_pct"),
        "v1_score": model_out.get("v1_score"),
        "v1_valid": model_out.get("v1_valid"),
        "v1_score_rows": model_out.get("v1_score_rows"),
        "v1_saliency_source": v1_saliency_source,
        "v1_saliency_score_mode": v1_saliency_score,
        "pred_loss": model_out["pred_loss"],
        "latent": latent_z,
        "latent_order": latent_order,
        "speed": speed,
        "t_rel": t_rel,
        "saliency_mode": saliency_mode,
        "saliency_method": saliency_method,
        "ig_baseline": ig_baseline,
        "saliency_source": saliency_source,
        "saccade_threshold_deg_s": threshold_deg_s,
        "level_labels": [f"L{i} {level_spec_label(spec)}" for i, spec in enumerate(crop_sizes)],
}


def draw_bar(ax, pct, v1_pct=None):
    ax.clear()
    n_levels = len(pct)
    colors = plt.cm.tab10(np.linspace(0, 1, max(3, n_levels)))[:n_levels]
    labels = [f"L{i}" for i in range(n_levels)]
    y = np.arange(n_levels)
    if v1_pct is None:
        ax.barh(y, pct, color=colors)
    else:
        ax.barh(y - 0.18, pct, height=0.34, color=colors, alpha=0.45, label="WM")
        ax.barh(y + 0.18, v1_pct, height=0.34, color=colors, alpha=0.95, label="V1")
        ax.legend(loc="lower right", fontsize=7, frameon=False)
    ax.set_xlim(0, 100)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("% saliency")
    ax.grid(axis="x", alpha=0.25)
    if v1_pct is None:
        for i, val in enumerate(pct):
            ax.text(min(98, val + 2), i, f"{val:.0f}", va="center", fontsize=7)
    else:
        for i, val in enumerate(v1_pct):
            ax.text(min(98, val + 2), i + 0.18, f"{val:.0f}", va="center", fontsize=7)


def init_figure(payload, clip, target_hz):
    n_levels = int(payload["source_pyramid"].shape[1])
    has_v1_saliency = payload.get("v1_saliency") is not None
    fig = plt.figure(figsize=(18 if has_v1_saliency else 16, max(9.0, 2.15 * n_levels)), dpi=130)
    n_cols = 5 if has_v1_saliency else 4
    width_ratios = [1.35, 0.48, 0.48, 0.48, 1.2] if has_v1_saliency else [1.45, 0.55, 0.55, 1.2]
    gs = GridSpec(
        n_levels,
        n_cols,
        figure=fig,
        width_ratios=width_ratios,
        height_ratios=[1.0] * n_levels,
        wspace=0.22,
        hspace=0.26,
    )

    ax_img = fig.add_subplot(gs[:, 0])
    ax_img.imshow(payload["full_image"], cmap="gray", origin="upper", vmin=0, vmax=1)
    ax_img.set_axis_off()
    ax_img.set_title(f"trial {clip.trial_id}  rows {clip.rows[0]}-{clip.rows[-1]}", fontsize=10)
    rect = Rectangle((0, 0), 1, 1, ec="red", fc="none", lw=2)
    ax_img.add_patch(rect)
    crop_center, = ax_img.plot([], [], "x", color="red", ms=7, mew=1.5)
    gaze_path, = ax_img.plot([], [], color="cyan", lw=1.2, alpha=0.9)
    gaze_dot, = ax_img.plot([], [], "o", color="cyan", ms=4)

    ax_in = [fig.add_subplot(gs[i, 1]) for i in range(n_levels)]
    ax_sal = [fig.add_subplot(gs[i, 2]) for i in range(n_levels)]
    ax_v1 = [fig.add_subplot(gs[i, 3]) for i in range(n_levels)] if has_v1_saliency else []
    im_in = []
    im_sal = []
    im_v1 = []
    sal_vmax = max(1e-8, float(np.percentile(payload["saliency"], 99)))
    if has_v1_saliency:
        valid = payload["v1_valid"] if payload.get("v1_valid") is not None else np.ones(len(payload["v1_saliency"]), dtype=bool)
        v1_values = payload["v1_saliency"][valid] if np.any(valid) else payload["v1_saliency"]
        v1_vmax = max(1e-8, float(np.percentile(v1_values, 99)))
    level_labels = payload["level_labels"]
    for i in range(n_levels):
        im_in.append(ax_in[i].imshow(payload["source_pyramid"][0, i], cmap="gray", origin="upper", vmin=0, vmax=1))
        ax_in[i].set_title(f"{level_labels[i]} input", fontsize=8)
        ax_in[i].set_xticks([])
        ax_in[i].set_yticks([])
        im_sal.append(ax_sal[i].imshow(payload["saliency"][0, i], cmap="magma", origin="upper", vmin=0, vmax=sal_vmax))
        ax_sal[i].set_title(f"{level_labels[i]} saliency", fontsize=8)
        ax_sal[i].set_xticks([])
        ax_sal[i].set_yticks([])
        if has_v1_saliency:
            im_v1.append(ax_v1[i].imshow(payload["v1_saliency"][0, i], cmap="magma", origin="upper", vmin=0, vmax=v1_vmax))
            ax_v1[i].set_title(f"{level_labels[i]} V1 obs", fontsize=8)
            ax_v1[i].set_xticks([])
            ax_v1[i].set_yticks([])

    has_readout = payload.get("readout_rate") is not None
    right_col = 4 if has_v1_saliency else 3
    if has_readout:
        gs_r = gs[:, right_col].subgridspec(5, 1, height_ratios=[1.0, 1.0, 1.0, 0.75, 0.7], hspace=0.22)
    else:
        gs_r = gs[:, right_col].subgridspec(4, 1, height_ratios=[1.0, 1.0, 0.8, 0.75], hspace=0.22)
    ax_obs = fig.add_subplot(gs_r[0, 0])
    if has_readout:
        ax_pred = fig.add_subplot(gs_r[1, 0], sharex=ax_obs)
        ax_lat = fig.add_subplot(gs_r[2, 0], sharex=ax_obs)
        ax_eye = fig.add_subplot(gs_r[3, 0], sharex=ax_obs)
        ax_bar = fig.add_subplot(gs_r[4, 0])
    else:
        ax_pred = None
        ax_lat = fig.add_subplot(gs_r[1, 0], sharex=ax_obs)
        ax_eye = fig.add_subplot(gs_r[2, 0], sharex=ax_obs)
        ax_bar = fig.add_subplot(gs_r[3, 0])

    obs_t, obs_unit = np.nonzero(payload["obs"] > 0)
    if obs_t.size:
        sizes = 3.0 + 2.0 * np.clip(payload["obs"][obs_t, obs_unit], 0, 3)
        ax_obs.scatter(payload["t_rel"][obs_t], obs_unit, s=sizes, c="black", marker="|", linewidths=0.7)
    ax_obs.set_xlim(float(payload["t_rel"][0]), float(payload["t_rel"][-1]))
    ax_obs.set_ylim(-0.5, max(0.5, payload["obs"].shape[1] - 0.5))
    ax_obs.set_title("observed V1 spikes", fontsize=9)
    ax_obs.set_ylabel("unit")
    ax_obs.tick_params(labelbottom=False)

    if has_readout and ax_pred is not None:
        pred = np.log1p(np.nan_to_num(payload["readout_rate"], nan=0.0, posinf=0.0, neginf=0.0))
        pred_vmax = max(1e-6, float(np.nanpercentile(pred, 99)))
        ax_pred.imshow(
            pred.T,
            aspect="auto",
            interpolation="none",
            cmap="gray_r",
            origin="lower",
            vmin=0.0,
            vmax=pred_vmax,
            extent=[float(payload["t_rel"][0]), float(payload["t_rel"][-1]), 0, pred.shape[1]],
        )
        ax_pred.set_title("readout predicted V1 rate", fontsize=9)
        ax_pred.set_ylabel("unit")
        ax_pred.tick_params(labelbottom=False)

    lat = payload["latent"]
    lat_z = (lat - np.nanmean(lat, axis=0, keepdims=True)) / np.nanstd(lat, axis=0, keepdims=True).clip(1e-6)
    lat_lim = max(1.0, min(3.5, float(np.percentile(np.abs(lat_z), 99)))) if lat_z.size else 1.0
    ax_lat.imshow(
        np.nan_to_num(lat_z).T,
        aspect="auto",
        interpolation="none",
        cmap="gray",
        origin="lower",
        vmin=-lat_lim,
        vmax=lat_lim,
        extent=[float(payload["t_rel"][0]), float(payload["t_rel"][-1]), 0, lat_z.shape[1]],
    )
    ax_lat.set_title("Gaussian latent activity (z-scored)", fontsize=9)
    ax_lat.set_ylabel("latent")
    ax_lat.tick_params(labelbottom=False)

    ax_eye.plot(payload["t_rel"], payload["speed"], color="black", lw=0.9, label="eye speed")
    threshold = payload["saccade_threshold_deg_s"]
    if threshold is not None:
        ax_eye.axhline(threshold, color="red", lw=0.8, ls="--")
    ax_eye.set_ylabel("deg/s")
    ax_eye.set_xlabel("time (s)")
    ax_eye.spines[["top", "right"]].set_visible(False)
    draw_bar(ax_bar, payload["channel_pct"][0], payload["v1_channel_pct"][0] if has_v1_saliency else None)

    vlines = [
        ax_obs.axvline(payload["t_rel"][0], color="red", ls="--", lw=1.2),
        ax_lat.axvline(payload["t_rel"][0], color="red", ls="--", lw=1.2),
        ax_eye.axvline(payload["t_rel"][0], color="red", ls="--", lw=1.2),
    ]
    if has_readout and ax_pred is not None:
        vlines.insert(1, ax_pred.axvline(payload["t_rel"][0], color="red", ls="--", lw=1.2))
    for ax in (ax_obs, ax_pred, ax_lat):
        if ax is None:
            continue
        ax.spines[["top", "right"]].set_visible(False)
    title = fig.suptitle("", fontsize=10)
    artists = {
        "rect": rect,
        "crop_center": crop_center,
        "gaze_path": gaze_path,
        "gaze_dot": gaze_dot,
        "im_in": im_in,
        "im_sal": im_sal,
        "im_v1": im_v1,
        "ax_bar": ax_bar,
        "vlines": vlines,
        "title": title,
    }
    return fig, artists


def update_artists(artists, payload, k, target_hz):
    roi = payload["roi_img"][k]
    i0, i1 = roi[0]
    j0, j1 = roi[1]
    artists["rect"].set_xy((j0, i0))
    artists["rect"].set_width(j1 - j0)
    artists["rect"].set_height(i1 - i0)
    center = roi.mean(axis=1)
    artists["crop_center"].set_data([center[1]], [center[0]])

    tail = max(5, int(round(0.5 * target_hz)))
    start = max(0, k - tail + 1)
    gaze = payload["gaze_img"][start : k + 1]
    finite = np.isfinite(gaze).all(axis=1)
    if finite.any():
        artists["gaze_path"].set_data(gaze[finite, 1], gaze[finite, 0])
        cur = payload["gaze_img"][k]
        if np.isfinite(cur).all():
            artists["gaze_dot"].set_data([cur[1]], [cur[0]])
    else:
        artists["gaze_path"].set_data([], [])
        artists["gaze_dot"].set_data([], [])

    for level in range(payload["source_pyramid"].shape[1]):
        artists["im_in"][level].set_array(payload["source_pyramid"][k, level])
        artists["im_sal"][level].set_array(payload["saliency"][k, level])
        if payload.get("v1_saliency") is not None:
            artists["im_v1"][level].set_array(payload["v1_saliency"][k, level])
    draw_bar(
        artists["ax_bar"],
        payload["channel_pct"][k],
        payload["v1_channel_pct"][k] if payload.get("v1_channel_pct") is not None else None,
    )
    t_now = float(payload["t_rel"][k])
    for vline in artists["vlines"]:
        vline.set_xdata([t_now, t_now])
    speed = float(payload["speed"][k])
    pred_loss = float(payload["pred_loss"][k])
    threshold = payload["saccade_threshold_deg_s"]
    state = "saccade" if threshold is not None and speed >= threshold else "fixation"
    v1_text = ""
    if payload.get("v1_score") is not None:
        valid = bool(payload["v1_valid"][k])
        v1_score = float(payload["v1_score"][k])
        v1_text = f" | V1score={v1_score:.3f}" if valid else " | V1score=NA"
        if valid and payload.get("v1_saliency_score_mode"):
            v1_text += f" ({payload['v1_saliency_score_mode']}, {payload.get('v1_saliency_source')})"
    artists["title"].set_text(
        f"{payload['saliency_method']} {payload['saliency_mode']} saliency, "
        f"{payload['saliency_source']} source | baseline={payload['ig_baseline']} | "
        f"{state} speed={speed:.1f}deg/s pred_loss={pred_loss:.4f}{v1_text}"
    )


def render_movie(out_path, payload, clip, target_hz, fps, frame_ids):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, artists = init_figure(payload, clip, target_hz)
    update_artists(artists, payload, int(frame_ids[0]), target_hz)
    try:
        if FFMpegWriter.isAvailable():
            writer = FFMpegWriter(
                fps=fps,
                codec="libx264",
                bitrate=18000,
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


def save_frame(path, payload, clip, target_hz, frame_id=0):
    fig, artists = init_figure(payload, clip, target_hz)
    update_artists(artists, payload, int(frame_id), target_hz)
    fig.savefig(path, dpi=fig.dpi)
    plt.close(fig)


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    model, cfg, extra = load_faithful_from_checkpoint(str(checkpoint), map_location="cpu")
    if cfg.family != "gaussian":
        raise ValueError("This movie currently targets faithful Gaussian LeWM checkpoints")
    session = args.session or extra.get("session", "Allen_2022-04-13")
    target_hz = int(args.target_hz if args.target_hz is not None else extra.get("target_hz", 120))
    center_mode = args.center_mode or extra.get("center_mode", "dset")
    pyramid_mode = args.pyramid_mode or extra.get("pyramid_mode", "raw")
    blur_sigmas = args.blur_sigmas if args.blur_sigmas is not None else extra.get("blur_sigmas", None)
    laplacian_contrast = float(args.laplacian_contrast if args.laplacian_contrast is not None else extra.get("laplacian_contrast", 1.0))
    action_history = int(args.action_history if args.action_history is not None else extra.get("action_history", max(1, cfg.action_dim // 2)))
    robs_downsample_mode = args.robs_downsample_mode or extra.get("robs_downsample_mode", "sample")
    covariate_downsample_mode = args.covariate_downsample_mode or extra.get("covariate_downsample_mode", "sample")
    pixel_normalization = args.pixel_normalization or extra.get("pixel_normalization", "unit")
    crop_sizes = parse_crop_sizes(args.crop_sizes) if args.crop_sizes else normalize_level_specs(extra.get("crop_sizes", (51, 101, 201)))
    if len(crop_sizes) != int(cfg.img_ch):
        raise ValueError(f"crop sizes define {len(crop_sizes)} channels, but checkpoint model expects img_ch={cfg.img_ch}")
    output_hw = int(args.output_hw if args.output_hw is not None else cfg.img_hw)
    threshold = args.saccade_threshold_deg_s
    if threshold is None:
        threshold = float(extra.get("saccade_threshold_deg_s", 25.0))

    downsample = downsample_factor(target_hz)
    n_display_frames = clip_frame_count(args.duration_s, target_hz)
    dset = load_backimage_dataset(BackImagePaths(session).dset_path)
    cov = dset.covariates
    eye_key = require_covariates(cov)
    sampler = BackImageSampler(session, dset)
    trial_id, rows = select_sequence_rows(
        cov,
        sampler,
        args.trial_id,
        args.start_row,
        n_display_frames,
        cfg.history_size,
        downsample,
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
        pyramid_mode=pyramid_mode,
        blur_sigmas=blur_sigmas,
        laplacian_contrast=laplacian_contrast,
        action_history=action_history,
        downsample=downsample,
        robs_downsample_mode=robs_downsample_mode,
        covariate_downsample_mode=covariate_downsample_mode,
        pixel_normalization=pixel_normalization,
    )
    v1_state = load_v1_readout_saliency_state(
        readout_path=args.v1_saliency_readout,
        latents_path=args.v1_saliency_latents,
        predictions_path=args.readout_predictions,
        device=torch.device(args.device),
    )
    model_out = run_faithful_sequence(
        model,
        clip,
        torch.device(args.device),
        args.saliency_mode,
        args.saliency_method,
        args.saliency_source,
        args.ig_steps,
        args.ig_baseline,
        v1_state=v1_state,
        v1_method=args.v1_saliency_method,
        v1_score_mode=args.v1_saliency_score,
        v1_lags=parse_v1_lags(args.v1_saliency_lags),
        v1_row_step=args.v1_saliency_row_step,
        v1_source=args.v1_saliency_source,
    )
    payload = prepare_payload(
        cov,
        sampler,
        clip,
        model_out,
        target_hz,
        cfg.history_size,
        args.saliency_mode,
        args.saliency_method,
        args.saliency_source,
        threshold,
        crop_sizes,
        args.ig_baseline,
        args.readout_predictions,
        v1_saliency_source=args.v1_saliency_source if v1_state is not None else None,
        v1_saliency_score=args.v1_saliency_score if v1_state is not None else None,
    )
    frame_ids = render_frame_ids(len(payload["t_rel"]), args.duration_s, args.fps, args.max_render_frames)
    frame0 = int(frame_ids[0])
    if payload.get("v1_valid") is not None and payload["v1_valid"].any():
        first_valid = int(np.flatnonzero(payload["v1_valid"])[0])
        frame_ids = frame_ids[frame_ids >= first_valid]
        if frame_ids.size == 0:
            frame_ids = np.asarray([first_valid], dtype=np.int64)
        frame0 = int(frame_ids[0])
    out_path = Path(args.out) if args.out else checkpoint.with_name(f"{checkpoint.stem}_{session}_faithful_saliency.mp4")
    if out_path.suffix.lower() != ".mp4":
        out_path = out_path.with_suffix(".mp4")
    frame_path = out_path.with_name(f"{out_path.stem}_frame0.png")
    print(f"checkpoint: {checkpoint}")
    print(f"clip: trial={trial_id} rows={rows[0]}-{rows[-1]} display_frames={len(payload['t_rel'])}")
    print(f"covariates: robs_downsample_mode={robs_downsample_mode} covariate_downsample_mode={covariate_downsample_mode}")
    if v1_state is not None:
        print(f"v1 readout: behavior_mode={v1_state.config.behavior_mode} row_step={v1_state.row_step}")
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
    print(f"saliency mean channel pct: {payload['channel_pct'].mean(axis=0).round(2).tolist()}")
    if payload.get("v1_channel_pct") is not None:
        valid = payload["v1_valid"]
        if valid.any():
            print(f"V1 observed saliency valid frames: {int(valid.sum())}/{len(valid)}")
            print(f"V1 observed saliency mean channel pct: {payload['v1_channel_pct'][valid].mean(axis=0).round(2).tolist()}")
        else:
            print("V1 observed saliency valid frames: 0")
    if payload.get("readout_rate") is not None:
        finite_pred = np.isfinite(payload["readout_rate"]).any(axis=1)
        print(f"readout prediction rows available: {int(finite_pred.sum())}/{len(finite_pred)}")
    render_movie(out_path, payload, clip, target_hz, args.fps, frame_ids)
    save_frame(frame_path, payload, clip, target_hz, frame_id=frame0)
    print(f"saved movie: {out_path}")
    print(f"saved frame: {frame_path}")


if __name__ == "__main__":
    main()
