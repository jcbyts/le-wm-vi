from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import torch

from marmo.faithful_train_utils import FaithfulBackImageLeWM, faithful_forward
from marmo.saliency_utils import SaliencyMethod, SourceReduce, _finish_result, _source_index, _with_pixels
from marmo.train_latent_spike_readout import ReadoutConfig, build_visioncore_behavior, make_model


V1ScoreMode = Literal["poisson_loss", "spike_lograte", "poisson_loglik", "rate_sum"]


@dataclass
class V1ReadoutSaliencyState:
    readout: torch.nn.Module
    config: ReadoutConfig
    data: dict[str, np.ndarray]
    feature_mean: torch.Tensor
    feature_std: torch.Tensor
    unit_mask: np.ndarray
    row_lookup: dict[tuple[int, int], int]
    split_row_lookup: dict[tuple[int, int, int], int]
    behavior_features: np.ndarray | None
    row_step: int
    device: torch.device


def _load_readout_config(payload: dict) -> ReadoutConfig:
    cfg = dict(payload["config"])
    cfg["lag_set"] = tuple(int(x) for x in cfg["lag_set"])
    cfg.setdefault("behavior_mode", "raw")
    return ReadoutConfig(**cfg)


def _infer_latents_from_predictions(predictions_path: str | Path | None) -> Path | None:
    if predictions_path is None:
        return None
    data = np.load(Path(predictions_path), allow_pickle=True)
    if "latents" not in data:
        return None
    value = data["latents"]
    if np.asarray(value).size == 0:
        return None
    return Path(str(np.asarray(value).reshape(-1)[0]))


def load_v1_readout_saliency_state(
    *,
    readout_path: str | Path | None,
    latents_path: str | Path | None = None,
    predictions_path: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> V1ReadoutSaliencyState | None:
    """Load a trained latent-to-V1 readout for differentiable saliency.

    If ``readout_path`` is omitted, this tries ``best_model.pt`` next to the
    predictions file. If ``latents_path`` is omitted, this tries the ``latents``
    field stored in ``readout_predictions_full.npz``.
    """

    if readout_path is None and predictions_path is None:
        return None
    pred_path = Path(predictions_path) if predictions_path is not None else None
    if readout_path is None:
        assert pred_path is not None
        readout_path = pred_path.with_name("best_model.pt")
    if latents_path is None:
        latents_path = _infer_latents_from_predictions(pred_path)
    if latents_path is None:
        raise ValueError("--v1-saliency-latents is required when it cannot be inferred from predictions")

    dev = torch.device(device)
    payload = torch.load(Path(readout_path), map_location="cpu", weights_only=False)
    cfg = _load_readout_config(payload)
    unit_mask = np.asarray(payload["unit_mask"], dtype=bool)
    feature_mean = torch.as_tensor(np.asarray(payload["feature_mean"], dtype=np.float32), device=dev)
    feature_std = torch.as_tensor(np.asarray(payload["feature_std"], dtype=np.float32), device=dev).clamp_min(1e-6)
    readout = make_model(
        cfg,
        input_dim=int(feature_mean.numel()),
        n_units=int(unit_mask.sum()),
        bias=torch.zeros(int(unit_mask.sum()), dtype=torch.float32),
    ).to(dev)
    readout.load_state_dict(payload["state_dict"])
    readout.eval()
    for param in readout.parameters():
        param.requires_grad_(False)

    data = dict(np.load(Path(latents_path), allow_pickle=True))
    row_step = int(payload.get("row_step", int(np.asarray(data.get("downsample", np.array([2]))).reshape(-1)[0])))
    behavior_features = None
    if cfg.behavior_mode in {"visioncore", "raw+visioncore"}:
        behavior_features = build_visioncore_behavior(data, row_step=row_step)
    trials = data["trial_inds"].astype(np.int64)
    rows = data["row_indices"].astype(np.int64)
    splits = data["split"].astype(np.int64)
    row_lookup = {(int(t), int(r)): i for i, (t, r) in enumerate(zip(trials, rows, strict=False))}
    split_row_lookup = {
        (int(s), int(t), int(r)): i
        for i, (s, t, r) in enumerate(zip(splits, trials, rows, strict=False))
    }
    return V1ReadoutSaliencyState(
        readout=readout,
        config=cfg,
        data=data,
        feature_mean=feature_mean,
        feature_std=feature_std,
        unit_mask=unit_mask,
        row_lookup=row_lookup,
        split_row_lookup=split_row_lookup,
        behavior_features=behavior_features,
        row_step=row_step,
        device=dev,
    )


def _dfs_for_index(data: dict[str, np.ndarray], index: int, unit_mask: np.ndarray) -> torch.Tensor:
    dfs = np.asarray(data["dfs"], dtype=np.float32)
    if dfs.ndim == 1:
        value = np.asarray([dfs[index]], dtype=np.float32)
    elif dfs.shape[1] == 1:
        value = np.asarray([dfs[index, 0]], dtype=np.float32)
    else:
        value = dfs[index, unit_mask].astype(np.float32)
    if value.size == 1:
        value = np.full(int(unit_mask.sum()), float(value[0]), dtype=np.float32)
    return torch.from_numpy(value.astype(np.float32))


def _constant_feature(data: dict[str, np.ndarray], key: str, index: int, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(data[key][index], dtype=torch.float32, device=device)


def _build_feature_with_current_pred(
    *,
    state: V1ReadoutSaliencyState,
    current_features: dict[str, torch.Tensor],
    trial_id: int,
    neural_index: int,
    neural_row: int,
    current_pred_row: int,
    current_lag: int,
    row_step: int,
) -> torch.Tensor | None:
    cfg = state.config
    data = state.data
    split_id = int(data["split"][neural_index])
    pieces: list[torch.Tensor] = []
    use_latent_feature = cfg.feature_key.lower() not in {"none", "null", "covariates", "behavior"}
    latent_keys = [x.strip() for x in cfg.feature_key.split("+") if x.strip()] if use_latent_feature else []
    if use_latent_feature:
        for lag in cfg.lag_set:
            feature_row = int(neural_row) - int(lag) * int(row_step)
            feature_index = state.split_row_lookup.get((split_id, int(trial_id), feature_row))
            if feature_index is None:
                return None
            for key in latent_keys:
                if key in current_features and int(lag) == int(current_lag) and feature_row == int(current_pred_row):
                    pieces.append(current_features[key].reshape(-1))
                else:
                    pieces.append(_constant_feature(data, key, feature_index, state.device))
    if cfg.behavior_mode in {"raw", "raw+visioncore"} and cfg.include_eye:
        pieces.append(_constant_feature(data, "eyepos", neural_index, state.device))
    if cfg.behavior_mode in {"raw", "raw+visioncore"} and cfg.include_action:
        action = _constant_feature(data, "action", neural_index, state.device)
        pieces.append(action)
        pieces.append(torch.linalg.norm(action).reshape(1))
    if cfg.behavior_mode in {"visioncore", "raw+visioncore"}:
        if state.behavior_features is None:
            raise RuntimeError("Readout requests VisionCore behavior features but they were not built")
        pieces.append(torch.as_tensor(state.behavior_features[int(neural_index)], dtype=torch.float32, device=state.device))
    if not pieces:
        pieces.append(torch.ones(1, dtype=torch.float32, device=state.device))
    x = torch.cat(pieces).reshape(1, -1)
    if x.shape[1] != state.feature_mean.numel():
        return None
    return (x - state.feature_mean.reshape(1, -1)) / state.feature_std.reshape(1, -1)


def _differentiable_feature_dict(
    model: FaithfulBackImageLeWM,
    emb: torch.Tensor,
    pred_hat: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Features that can be substituted into a trained readout with gradients."""
    code = model.code_from_emb(emb)
    features = {
        "emb": emb,
        "eta": emb,
        "code": code,
        "pred_hat": pred_hat,
    }
    fdim = int(getattr(model.cfg, "foveal_dim", 0) or 0)
    if fdim > 0 and emb.shape[-1] > fdim:
        features.update(
            {
                "emb_foveal": emb[..., :fdim],
                "eta_foveal": emb[..., :fdim],
                "code_foveal": code[..., :fdim],
                "pred_hat_foveal": pred_hat[..., :fdim],
                "emb_context": emb[..., fdim:],
                "eta_context": emb[..., fdim:],
                "code_context": code[..., fdim:],
                "pred_hat_context": pred_hat[..., fdim:],
            }
        )
    return features


def compute_v1_observed_saliency(
    model: FaithfulBackImageLeWM,
    state: V1ReadoutSaliencyState,
    batch: dict[str, torch.Tensor],
    *,
    trial_id: int,
    current_pred_row: int,
    method: SaliencyMethod = "grad_x_input",
    source_reduce: SourceReduce = "current",
    score_mode: V1ScoreMode = "spike_lograte",
    readout_lags: tuple[int, ...] | None = None,
    pred_index: int = -1,
    row_step: int | None = None,
):
    """Attribute observed V1 activity to the current predicted latent.

    For each requested readout lag ``L``, the current WM ``pred_hat`` row ``T``
    is inserted into the trained readout feature vector for neural row
    ``T + L * row_step``. The observed spikes at that neural row define the
    scalar score, and gradients are propagated back to ``batch["pixels"]``.
    """

    if method not in {"grad", "grad_x_input"}:
        raise ValueError("V1 observed saliency currently supports grad and grad_x_input")
    model.eval()
    row_step = int(state.row_step if row_step is None or int(row_step) <= 0 else row_step)
    pixels0 = batch["pixels"].detach().clone()
    pixels = pixels0.requires_grad_(True)
    score_terms: list[torch.Tensor] = []
    score_rows: list[int] = []
    with_pixels = _with_pixels(batch, pixels)
    out = faithful_forward(model, with_pixels)
    pred = out["pred_hat"][:, pred_index]
    if pred.shape[0] != 1:
        raise ValueError("V1 observed saliency currently expects batch size 1")
    emb_current = out["emb"][:, pred_index]
    current_features = {key: value[0] for key, value in _differentiable_feature_dict(model, emb_current, pred).items()}
    lags = tuple(int(x) for x in (readout_lags if readout_lags is not None else state.config.lag_set))
    data = state.data
    robs = data["robs"].astype(np.float32)
    for lag in lags:
        neural_row = int(current_pred_row) + int(lag) * int(row_step)
        neural_index = state.row_lookup.get((int(trial_id), neural_row))
        if neural_index is None:
            continue
        x = _build_feature_with_current_pred(
            state=state,
            current_features=current_features,
            trial_id=int(trial_id),
            neural_index=int(neural_index),
            neural_row=neural_row,
            current_pred_row=int(current_pred_row),
            current_lag=int(lag),
            row_step=int(row_step),
        )
        if x is None:
            continue
        log_rate = state.readout(x).reshape(-1)
        y_np = robs[int(neural_index), state.unit_mask].astype(np.float32)
        y = torch.as_tensor(y_np, dtype=torch.float32, device=state.device)
        dfs = _dfs_for_index(data, int(neural_index), state.unit_mask).to(state.device)
        if score_mode == "spike_lograte":
            weights = y * dfs
            if float(weights.sum().detach().cpu()) <= 0.0:
                continue
            term = (weights * log_rate).sum() / weights.sum().clamp_min(1.0)
        elif score_mode == "poisson_loglik":
            term = (dfs * (y * log_rate - log_rate.clamp(-20, 8).exp())).sum() / dfs.sum().clamp_min(1.0)
        elif score_mode == "rate_sum":
            term = log_rate.clamp(-20, 8).exp().mean()
        else:
            raise ValueError("score_mode must be spike_lograte, poisson_loglik, or rate_sum")
        score_terms.append(term)
        score_rows.append(neural_row)

    if not score_terms:
        history_size = int(model.cfg.history_size)
        src_idx = _source_index(history_size, pred_index)
        zeros = torch.zeros_like(pixels[:, src_idx])
        mass = zeros.flatten(-2).sum(dim=-1)
        return {
            "heatmaps": zeros.detach(),
            "channel_pct": torch.zeros_like(mass),
            "score": torch.zeros(1, device=state.device),
            "valid": torch.zeros(1, dtype=torch.bool, device=state.device),
            "score_rows": (),
            "source_index": src_idx,
        }

    score = torch.stack(score_terms).mean().reshape(1)
    grad = torch.autograd.grad(score.sum(), pixels, create_graph=False, retain_graph=False)[0]
    attr = grad if method == "grad" else grad * pixels
    result = _finish_result(attr, out, score, int(model.cfg.history_size), pred_index, source_reduce)
    return {
        "heatmaps": result.heatmaps,
        "channel_pct": result.channel_pct,
        "score": result.score,
        "valid": torch.ones(1, dtype=torch.bool, device=state.device),
        "score_rows": tuple(score_rows),
        "source_index": result.source_index,
    }


def _clip_window(clip, start: int, stop: int):
    return clip.__class__(
        trial_id=clip.trial_id,
        rows=clip.rows[start:stop],
        pixels=clip.pixels[start:stop],
        action=clip.action[start:stop],
        eyepos=clip.eyepos[start:stop],
        robs=clip.robs[start:stop],
        dfs=clip.dfs[start:stop],
        t_bins=clip.t_bins[start:stop],
        roi=clip.roi[start:stop],
        dpi_valid=clip.dpi_valid[start:stop],
    )


def compute_v1_observed_saliency_for_clip(
    model: FaithfulBackImageLeWM,
    state: V1ReadoutSaliencyState,
    clip,
    *,
    target_index: int,
    to_model_batch: Callable,
    method: SaliencyMethod = "grad_x_input",
    source_reduce: SourceReduce = "context_sum",
    score_mode: V1ScoreMode = "poisson_loss",
    readout_lags: tuple[int, ...] | None = None,
    row_step: int | None = None,
):
    """Attribute actual V1 readout loss at one displayed target row.

    ``target_index`` indexes ``clip.rows`` at the observed neural frame. For
    each readout lag, this function recomputes the lagged WM prediction
    differentiably from its own history window and inserts those predictions
    into the trained readout feature vector.
    """

    if method not in {"grad", "grad_x_input"}:
        raise ValueError("V1 observed saliency currently supports grad and grad_x_input")
    model.eval()
    row_step = int(state.row_step if row_step is None or int(row_step) <= 0 else row_step)
    hist = int(model.cfg.history_size)
    cfg = state.config
    data = state.data
    lags = tuple(int(x) for x in (readout_lags if readout_lags is not None else cfg.lag_set))
    target_row = int(clip.rows[int(target_index)])
    neural_index = state.row_lookup.get((int(clip.trial_id), target_row))
    n_levels = int(clip.pixels.shape[1])
    height = int(clip.pixels.shape[2])
    width = int(clip.pixels.shape[3])
    if neural_index is None:
        zeros = torch.zeros((n_levels, height, width), dtype=torch.float32, device=state.device)
        return {
            "heatmaps": zeros,
            "channel_pct": torch.zeros(n_levels, dtype=torch.float32, device=state.device),
            "score": torch.zeros(1, dtype=torch.float32, device=state.device),
            "valid": torch.zeros(1, dtype=torch.bool, device=state.device),
            "score_rows": (),
            "source_index": hist - 1,
        }
    if state.unit_mask.shape[0] != int(data["robs"].shape[1]):
        raise ValueError("readout unit_mask length does not match robs unit count")

    split_id = int(data["split"][int(neural_index)])
    use_latent_feature = cfg.feature_key.lower() not in {"none", "null", "covariates", "behavior"}
    latent_keys = [x.strip() for x in cfg.feature_key.split("+") if x.strip()] if use_latent_feature else []

    pred_by_lag: dict[int, torch.Tensor] = {}
    feature_by_lag: dict[int, dict[str, torch.Tensor]] = {}
    pixels_by_lag: dict[int, torch.Tensor] = {}
    out_by_lag = {}
    for lag in lags:
        feature_index = int(target_index) - int(lag)
        start = feature_index - hist
        stop = feature_index + 1
        if start < 0 or stop > len(clip.rows):
            continue
        feature_row = int(clip.rows[feature_index])
        expected_row = target_row - int(lag) * int(row_step)
        if feature_row != expected_row:
            continue
        if state.split_row_lookup.get((split_id, int(clip.trial_id), feature_row)) is None:
            continue
        sub_clip = _clip_window(clip, start, stop)
        batch = to_model_batch(sub_clip, state.device)
        pixels = batch["pixels"].detach().clone().requires_grad_(True)
        batch = _with_pixels(batch, pixels)
        out = faithful_forward(model, batch)
        pred_by_lag[int(lag)] = out["pred_hat"][0, -1]
        emb = out["emb"][:, -1]
        pred = out["pred_hat"][:, -1]
        feature_by_lag[int(lag)] = {
            key: value[0]
            for key, value in _differentiable_feature_dict(model, emb, pred).items()
        }
        pixels_by_lag[int(lag)] = pixels
        out_by_lag[int(lag)] = out

    pieces: list[torch.Tensor] = []
    if use_latent_feature:
        for lag in lags:
            feature_row = target_row - int(lag) * int(row_step)
            feature_index = state.split_row_lookup.get((split_id, int(clip.trial_id), feature_row))
            if feature_index is None:
                return _invalid_v1_result(n_levels, height, width, hist, state.device)
            for key in latent_keys:
                if key in {"pred_hat", "code", "emb", "eta", "pred_hat_foveal", "code_foveal", "emb_foveal", "eta_foveal", "pred_hat_context", "code_context", "emb_context", "eta_context"}:
                    features = feature_by_lag.get(int(lag))
                    if features is None or key not in features:
                        return _invalid_v1_result(n_levels, height, width, hist, state.device)
                    pieces.append(features[key].reshape(-1))
                else:
                    pieces.append(_constant_feature(data, key, int(feature_index), state.device))
    if cfg.behavior_mode in {"raw", "raw+visioncore"} and cfg.include_eye:
        pieces.append(_constant_feature(data, "eyepos", int(neural_index), state.device))
    if cfg.behavior_mode in {"raw", "raw+visioncore"} and cfg.include_action:
        action = _constant_feature(data, "action", int(neural_index), state.device)
        pieces.append(action)
        pieces.append(torch.linalg.norm(action).reshape(1))
    if cfg.behavior_mode in {"visioncore", "raw+visioncore"}:
        if state.behavior_features is None:
            raise RuntimeError("Readout requests VisionCore behavior features but they were not built")
        pieces.append(torch.as_tensor(state.behavior_features[int(neural_index)], dtype=torch.float32, device=state.device))
    if not pieces:
        pieces.append(torch.ones(1, dtype=torch.float32, device=state.device))
    x = torch.cat(pieces).reshape(1, -1)
    if x.shape[1] != state.feature_mean.numel():
        raise ValueError(f"online V1 feature dim {x.shape[1]} != readout dim {state.feature_mean.numel()}")
    x = (x - state.feature_mean.reshape(1, -1)) / state.feature_std.reshape(1, -1)
    log_rate = state.readout(x).reshape(-1)
    y_np = data["robs"].astype(np.float32)[int(neural_index), state.unit_mask]
    y = torch.as_tensor(y_np, dtype=torch.float32, device=state.device)
    dfs = _dfs_for_index(data, int(neural_index), state.unit_mask).to(state.device)
    if float(dfs.sum().detach().cpu()) <= 0.0:
        return _invalid_v1_result(n_levels, height, width, hist, state.device)

    rate = log_rate.clamp(-20, 8).exp()
    if score_mode == "poisson_loss":
        score = (dfs * (rate - y * log_rate)).sum() / dfs.sum().clamp_min(1.0)
    elif score_mode == "spike_lograte":
        weights = y * dfs
        if float(weights.sum().detach().cpu()) <= 0.0:
            return _invalid_v1_result(n_levels, height, width, hist, state.device)
        score = -((weights * log_rate).sum() / weights.sum().clamp_min(1.0))
    elif score_mode == "poisson_loglik":
        score = -((dfs * (y * log_rate - rate)).sum() / dfs.sum().clamp_min(1.0))
    elif score_mode == "rate_sum":
        score = rate.mean()
    else:
        raise ValueError("unknown V1 score mode")
    if not bool(torch.isfinite(score).detach().cpu()):
        return _invalid_v1_result(n_levels, height, width, hist, state.device)
    pixel_tensors = [pixels_by_lag[int(lag)] for lag in lags if int(lag) in pixels_by_lag]
    grads = torch.autograd.grad(score, pixel_tensors, create_graph=False, retain_graph=False, allow_unused=True)
    heat = torch.zeros((n_levels, height, width), dtype=torch.float32, device=state.device)
    src_idx = hist - 1
    for lag, pixels, grad in zip([lag for lag in lags if int(lag) in pixels_by_lag], pixel_tensors, grads, strict=False):
        if grad is None:
            continue
        attr = grad if method == "grad" else grad * pixels
        if source_reduce == "current":
            signed = attr[:, src_idx]
        elif source_reduce == "context_sum":
            signed = attr[:, :hist].sum(dim=1)
        else:
            raise ValueError("source_reduce must be current or context_sum")
        heat = heat + signed[0].abs()
    if not torch.isfinite(heat).all():
        return _invalid_v1_result(n_levels, height, width, hist, state.device)
    mass = heat.flatten(-2).sum(dim=-1)
    if float(mass.sum().detach().cpu()) <= 0.0:
        pct = torch.zeros(n_levels, dtype=torch.float32, device=state.device)
    else:
        pct = 100.0 * mass / mass.sum().clamp_min(1e-12)
    return {
        "heatmaps": heat.detach(),
        "channel_pct": pct.detach(),
        "score": score.detach().reshape(1),
        "valid": torch.ones(1, dtype=torch.bool, device=state.device),
        "score_rows": (target_row,),
        "source_index": src_idx,
    }


def _invalid_v1_result(n_levels: int, height: int, width: int, hist: int, device: torch.device):
    zeros = torch.zeros((n_levels, height, width), dtype=torch.float32, device=device)
    return {
        "heatmaps": zeros,
        "channel_pct": torch.zeros(n_levels, dtype=torch.float32, device=device),
        "score": torch.zeros(1, dtype=torch.float32, device=device),
        "valid": torch.zeros(1, dtype=torch.bool, device=device),
        "score_rows": (),
        "source_index": hist - 1,
    }
