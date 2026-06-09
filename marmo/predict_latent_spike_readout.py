from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from marmo.train_latent_spike_readout import (
    ReadoutConfig,
    build_visioncore_behavior,
    expand_dfs,
    filter_rows,
    make_model,
    predict_log_rate,
    row_session_ids,
)


def parse_args():
    p = argparse.ArgumentParser(description="Apply a trained latent-to-V1 Poisson readout to extracted BackImage latents")
    p.add_argument("--latents", required=True)
    p.add_argument("--readout", required=True, help="Path to best_model.pt from train_latent_spike_readout.py")
    p.add_argument("--out", default=None)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--row-step", type=int, default=0, help="Raw dataset rows per latent step; 0 reads it from the readout/latent file")
    p.add_argument("--session-id", type=int, default=None, help="Override/readout session_id for multi-session latent files")
    p.add_argument("--allow-mixed-sessions", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_config(payload: dict) -> ReadoutConfig:
    cfg = dict(payload["config"])
    cfg["lag_set"] = tuple(int(x) for x in cfg["lag_set"])
    cfg.setdefault("behavior_mode", "raw")
    return ReadoutConfig(**cfg)


def build_features_for_indices(
    data: dict[str, np.ndarray],
    cfg: ReadoutConfig,
    *,
    sample_indices: np.ndarray,
    row_step: int,
    behavior_features: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    split = data["split"].astype(np.int64)
    rows = data["row_indices"].astype(np.int64)
    trials = data["trial_inds"].astype(np.int64)
    sessions = row_session_ids(data)
    eyepos = data["eyepos"].astype(np.float32)
    action = data["action"].astype(np.float32)
    feature_key = cfg.feature_key
    use_latent_feature = feature_key.lower() not in {"none", "null", "covariates", "behavior"}
    latent_keys = [k.strip() for k in feature_key.split("+") if k.strip()] if use_latent_feature else []
    features = [data[k].astype(np.float32) for k in latent_keys]
    index = {
        (int(sess), int(s), int(t), int(r)): i
        for i, (sess, s, t, r) in enumerate(zip(sessions, split, trials, rows, strict=False))
    }
    x = []
    kept = []
    for i in sample_indices.astype(np.int64):
        pieces = []
        ok = True
        if use_latent_feature:
            for lag in cfg.lag_set:
                j = index.get(
                    (
                        int(sessions[i]),
                        int(split[i]),
                        int(trials[i]),
                        int(rows[i]) - int(lag) * int(row_step),
                    )
                )
                if j is None:
                    ok = False
                    break
                for feature in features:
                    pieces.append(feature[j])
        if not ok:
            continue
        if cfg.behavior_mode in {"raw", "raw+visioncore"} and cfg.include_eye:
            pieces.append(eyepos[i])
        if cfg.behavior_mode in {"raw", "raw+visioncore"} and cfg.include_action:
            pieces.append(action[i])
            pieces.append(np.asarray([np.linalg.norm(action[i])], dtype=np.float32))
        if cfg.behavior_mode in {"visioncore", "raw+visioncore"}:
            if behavior_features is None:
                raise RuntimeError("Readout requests VisionCore behavior features but they were not built")
            pieces.append(behavior_features[i].astype(np.float32))
        if not pieces:
            pieces.append(np.asarray([1.0], dtype=np.float32))
        x.append(np.concatenate(pieces).astype(np.float32))
        kept.append(int(i))
    if not x:
        raise RuntimeError("No rows had the lagged features required by the readout")
    return np.stack(x, axis=0), np.asarray(kept, dtype=np.int64)


def main():
    args = parse_args()
    lat_path = Path(args.latents)
    readout_path = Path(args.readout)
    out_path = Path(args.out) if args.out else readout_path.with_name(readout_path.stem + "_predictions_full.npz")

    data = dict(np.load(lat_path, allow_pickle=True))
    payload = torch.load(readout_path, map_location="cpu", weights_only=False)
    session_ids = row_session_ids(data)
    requested_session_id = args.session_id
    if requested_session_id is None:
        requested_session_id = payload.get("session_id", None)
    unique_sessions = np.unique(session_ids)
    if requested_session_id is not None:
        requested_session_id = int(requested_session_id)
        if requested_session_id not in set(int(x) for x in unique_sessions):
            raise RuntimeError(f"Requested session_id={requested_session_id}, available={unique_sessions.tolist()}")
        data = filter_rows(data, session_ids == requested_session_id)
        session_unit_counts = data.get("session_unit_counts")
        if session_unit_counts is not None:
            unit_count = int(np.asarray(session_unit_counts).reshape(-1)[requested_session_id])
            for key in ["robs", "dfs"]:
                if key in data and np.asarray(data[key]).ndim == 2 and data[key].shape[1] > unit_count:
                    data[key] = data[key][:, :unit_count]
        print(f"filtered predictions to session_id={requested_session_id}", flush=True)
    elif len(unique_sessions) > 1 and not args.allow_mixed_sessions:
        raise RuntimeError(
            "Latent file contains multiple session_id values. Provide --session-id or use a readout trained with --session-id."
        )
    cfg = load_config(payload)
    row_step = int(args.row_step)
    if row_step <= 0:
        if "row_step" in payload:
            row_step = int(payload["row_step"])
        else:
            row_step = int(np.asarray(data.get("downsample", np.array([2]))).reshape(-1)[0])
    behavior_features = None
    if cfg.behavior_mode in {"visioncore", "raw+visioncore"}:
        behavior_features = build_visioncore_behavior(data, row_step=row_step)
    unit_mask = np.asarray(payload["unit_mask"], dtype=bool)
    if "feature_mean" not in payload or "feature_std" not in payload:
        raise RuntimeError(
            f"{readout_path} does not contain feature_mean/feature_std. "
            "Rerun train_latent_spike_readout.py after the normalization-save patch."
        )
    feature_mean = np.asarray(payload["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(payload["feature_std"], dtype=np.float32)

    n_rows = int(data["robs"].shape[0])
    sample_indices = np.arange(n_rows, dtype=np.int64)
    x, kept = build_features_for_indices(
        data,
        cfg,
        sample_indices=sample_indices,
        row_step=row_step,
        behavior_features=behavior_features,
    )
    if x.shape[1] != feature_mean.shape[0]:
        raise RuntimeError(f"Feature dim mismatch: built {x.shape[1]}, readout expects {feature_mean.shape[0]}")
    x = (x - feature_mean[None, :]) / np.maximum(feature_std[None, :], 1e-6)

    model = make_model(
        cfg,
        input_dim=x.shape[1],
        n_units=int(unit_mask.sum()),
        bias=torch.zeros(int(unit_mask.sum()), dtype=torch.float32),
    ).to(args.device)
    model.load_state_dict(payload["state_dict"])
    x_t = torch.from_numpy(x.astype(np.float32))
    with torch.no_grad():
        log_rate_kept = predict_log_rate(model, x_t, args.batch_size, args.device).cpu().numpy().astype(np.float32)
    rate_kept = np.exp(np.clip(log_rate_kept, -20.0, 8.0)).astype(np.float32)

    log_rate = np.full((n_rows, int(unit_mask.sum())), np.nan, dtype=np.float32)
    rate = np.full_like(log_rate, np.nan)
    log_rate[kept] = log_rate_kept
    rate[kept] = rate_kept

    dfs_exp = expand_dfs(data["dfs"].astype(np.float32), data["robs"].shape[1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        rate=rate,
        log_rate=log_rate,
        valid_indices=kept,
        unit_mask=unit_mask,
        robs=data["robs"].astype(np.float32)[:, unit_mask],
        dfs=dfs_exp[:, unit_mask].astype(np.float32),
        row_indices=data["row_indices"].astype(np.int64),
        trial_inds=data["trial_inds"].astype(np.int64),
        t_bins=data["t_bins"].astype(np.float32),
        split=data["split"].astype(np.int64),
        session_id=row_session_ids(data),
        session_names=data.get("session_names", np.asarray([])),
        session_unit_counts=data.get("session_unit_counts", np.asarray([])),
        eyepos=data["eyepos"].astype(np.float32),
        action=data["action"].astype(np.float32),
        config_json=np.array([json.dumps(payload["config"])]),
        latents=str(lat_path),
        readout=str(readout_path),
        row_step=np.array([row_step], dtype=np.int64),
    )
    print(f"saved readout predictions: {out_path}")
    print(f"valid rows: {len(kept)}/{n_rows} units: {int(unit_mask.sum())}/{len(unit_mask)} row_step={row_step}")


if __name__ == "__main__":
    main()
