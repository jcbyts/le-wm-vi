from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

VISIONCORE_ROOT = Path("/home/tejas/VisionCore")
if str(VISIONCORE_ROOT) not in sys.path:
    sys.path.insert(0, str(VISIONCORE_ROOT))

try:
    from models.losses.poisson import calc_poisson_bits_per_spike
except Exception:
    # VisionCore's package import can require optional model dependencies. Keep
    # the metric identical to VisionCore/models/losses/poisson.py for standalone
    # readout searches.
    def calc_poisson_bits_per_spike(r_pred, r_obs, dfs=None):
        r_pred = torch.as_tensor(r_pred)
        r_obs = torch.as_tensor(r_obs)
        if dfs is None:
            dfs = torch.ones_like(r_obs)
        dfs = torch.as_tensor(dfs, device=r_obs.device, dtype=r_obs.dtype)
        if r_pred.ndim == 1:
            r_pred = r_pred.unsqueeze(1)
        if r_obs.ndim == 1:
            r_obs = r_obs.unsqueeze(1)
        if dfs.ndim == 1:
            dfs = dfs.unsqueeze(1)
        if r_pred.shape != r_obs.shape:
            raise AssertionError("r_pred and r_obs must have the same shape")
        if len(r_pred.shape) != len(dfs.shape):
            raise AssertionError("r_pred and dfs must have the same number of dimensions")
        with torch.no_grad():
            t = dfs.sum(dim=0).clamp(1)
            n = (dfs * r_obs).sum(dim=0).clamp(1)
            r_bar = n / t
            ll_pred = r_obs * torch.log(r_pred + 1e-8) - r_pred
            ll_null = r_obs * torch.log(r_bar + 1e-8) - r_bar
            iss = (ll_pred - ll_null) * dfs
            iss = iss.sum(dim=0) / n / math.log(2)
        return iss


@dataclass(frozen=True)
class ReadoutConfig:
    arch: str
    lag_set: tuple[int, ...]
    feature_key: str
    behavior_mode: str
    include_eye: bool
    include_action: bool
    hidden_dim: int
    depth: int
    dropout: float
    weight_decay: float
    lr: float


class PoissonLinear(nn.Module):
    def __init__(self, input_dim: int, n_units: int, bias_init: torch.Tensor):
        super().__init__()
        self.linear = nn.Linear(input_dim, n_units)
        nn.init.zeros_(self.linear.weight)
        self.linear.bias.data.copy_(bias_init)

    def forward(self, x):
        return self.linear(x)


class PoissonMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_units: int,
        bias_init: torch.Tensor,
        hidden_dim: int = 256,
        depth: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers: list[nn.Module] = [nn.LayerNorm(input_dim)]
        dim = input_dim
        for _ in range(max(1, int(depth))):
            layers.extend(
                [
                    nn.Linear(dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                ]
            )
            dim = hidden_dim
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(dim, n_units)
        nn.init.zeros_(self.out.weight)
        self.out.bias.data.copy_(bias_init)

    def forward(self, x):
        return self.out(self.net(x))


def parse_args():
    p = argparse.ArgumentParser(description="Train Poisson V1 readouts from BackImage world-model latents")
    p.add_argument("--latents", required=True)
    p.add_argument("--outdir", default=None)
    p.add_argument(
        "--feature-keys",
        default="code",
        help=(
            "Comma-separated feature keys: code,eta,pred_hat,target,none. "
            "Use + for concatenated latent keys, e.g. code+pred_hat. "
            "Use none for eye/action/null controls."
        ),
    )
    p.add_argument(
        "--lag-sets",
        default="0;0:2;0:4;0:8;2:8;0:12",
        help="Semicolon-separated lag sets. Use comma lists or inclusive ranges, e.g. 0,2,4;0:8",
    )
    p.add_argument("--archs", default="linear,mlp")
    p.add_argument("--include-eye", action="store_true")
    p.add_argument("--include-action", action="store_true")
    p.add_argument(
        "--behavior-mode",
        choices=["raw", "visioncore", "raw+visioncore", "none"],
        default="visioncore",
        help="Behavior covariates appended to readout features. visioncore matches the BackImage digital-twin eye-velocity basis plus eyepos.",
    )
    p.add_argument("--mlp-hidden-dims", default="256,512")
    p.add_argument("--mlp-depths", default="1,2")
    p.add_argument("--dropouts", default="0.0,0.1")
    p.add_argument("--linear-weight-decays", default="1e-6,1e-5,1e-4,1e-3,1e-2")
    p.add_argument("--mlp-weight-decays", default="1e-5,1e-4,1e-3")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument(
        "--row-step",
        type=int,
        default=0,
        help="Raw dset row step per latent bin. Defaults to latents['downsample'] when available, else 2.",
    )
    p.add_argument("--min-train-spikes", type=float, default=5.0)
    p.add_argument(
        "--session-id",
        type=int,
        default=None,
        help="For multi-session latent files, train/evaluate only this session_id. Required unless --allow-mixed-sessions is set.",
    )
    p.add_argument(
        "--allow-mixed-sessions",
        action="store_true",
        help="Allow one flat readout across multiple session_id values. Usually wrong for V1 units; use only for explicit controls.",
    )
    p.add_argument("--max-configs", type=int, default=0)
    p.add_argument("--seed", type=int, default=1002)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_lag_set(spec: str) -> tuple[int, ...]:
    spec = spec.strip()
    if ":" in spec:
        a, b = [int(x) for x in spec.split(":", 1)]
        if b < a:
            raise ValueError(f"Invalid lag range {spec}")
        return tuple(range(a, b + 1))
    return tuple(int(x.strip()) for x in spec.split(",") if x.strip())


def masked_poisson_nll_lograte(log_rate: torch.Tensor, robs: torch.Tensor, dfs: torch.Tensor) -> torch.Tensor:
    loss = F.poisson_nll_loss(log_rate, robs, log_input=True, full=False, reduction="none")
    if dfs.ndim == 1:
        dfs = dfs[:, None]
    loss = loss * dfs
    if dfs.shape[1] == 1:
        div = dfs.sum().clamp_min(1.0) * robs.shape[1]
    else:
        div = dfs.sum().clamp_min(1.0)
    return loss.sum() / div


def safe_corr(pred: np.ndarray, obs: np.ndarray, dfs: np.ndarray | None = None) -> np.ndarray:
    if dfs is None:
        dfs = np.ones((obs.shape[0], 1), dtype=np.float32)
    if dfs.ndim == 1:
        dfs = dfs[:, None]
    out = np.full(obs.shape[1], np.nan, dtype=np.float64)
    for i in range(obs.shape[1]):
        mask = dfs[:, 0] > 0 if dfs.shape[1] == 1 else dfs[:, i] > 0
        x = pred[mask, i]
        y = obs[mask, i]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 5:
            continue
        if np.std(x[ok]) <= 1e-12 or np.std(y[ok]) <= 1e-12:
            continue
        out[i] = float(np.corrcoef(x[ok], y[ok])[0, 1])
    return out


def expand_dfs(dfs: np.ndarray, n_units: int) -> np.ndarray:
    if dfs.ndim == 1:
        dfs = dfs[:, None]
    if dfs.shape[1] == 1:
        return np.broadcast_to(dfs, (dfs.shape[0], n_units)).copy()
    return dfs


def row_session_ids(data: dict[str, np.ndarray]) -> np.ndarray:
    n = int(len(data["split"]))
    if "session_id" not in data:
        return np.zeros(n, dtype=np.int64)
    session_id = np.asarray(data["session_id"]).astype(np.int64)
    if session_id.shape[0] != n:
        raise RuntimeError(f"session_id length {session_id.shape[0]} does not match split length {n}")
    return session_id


def filter_rows(data: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    n = int(len(data["split"]))
    out: dict[str, np.ndarray] = {}
    for key, value in data.items():
        arr = np.asarray(value)
        if arr.shape[:1] == (n,):
            out[key] = arr[mask]
        else:
            out[key] = arr
    return out


def zscore_train_val(x_train: np.ndarray, x_val: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(x_train, axis=0, keepdims=True)
    std = np.nanstd(x_train, axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (x_train - mean) / std, (x_val - mean) / std, mean.squeeze(0), std.squeeze(0)


def build_visioncore_behavior(data: dict[str, np.ndarray], row_step: int) -> np.ndarray:
    """Recreate the BackImage digital-twin behavior transform for latent rows.

    VisionCore uses eye velocity -> maxnorm -> symlog -> acausal temporal basis
    -> split-ReLU, concatenated with raw eye position.
    """
    try:
        from models.modules.frontend import TemporalBasis
    except Exception:
        from DataYatesV1.models.modules.frontend import TemporalBasis

    eyepos = data["eyepos"].astype(np.float32)
    rows = data["row_indices"].astype(np.int64)
    trials = data["trial_inds"].astype(np.int64)
    sessions = row_session_ids(data)
    vel = np.zeros_like(eyepos, dtype=np.float32)
    for session_id in np.unique(sessions):
        session_mask = sessions == int(session_id)
        for trial in np.unique(trials[session_mask]):
            idx = np.flatnonzero(session_mask & (trials == int(trial)))
            order = idx[np.argsort(rows[idx], kind="stable")]
            if len(order) <= 1:
                continue
            prev = order[:-1]
            cur = order[1:]
            consecutive = (rows[cur] - rows[prev]) == int(row_step)
            vel[cur[consecutive]] = eyepos[cur[consecutive]] - eyepos[prev[consecutive]]

    denom = np.nanmax(np.abs(vel))
    if not np.isfinite(denom) or denom < 1e-8:
        denom = 1.0
    vel = vel / float(denom)
    vel = np.sign(vel) * np.log1p(np.abs(vel))

    basis = TemporalBasis(
        num_delta_funcs=0,
        num_cosine_funcs=10,
        history_bins=50,
        causal=False,
        log_spacing=False,
        peak_range_ms=(30, 200),
        normalize=True,
        sampling_rate=120,
    )
    out = np.zeros((eyepos.shape[0], 42), dtype=np.float32)
    out[:, :2] = eyepos
    with torch.no_grad():
        for session_id in np.unique(sessions):
            session_mask = sessions == int(session_id)
            for trial in np.unique(trials[session_mask]):
                idx = np.flatnonzero(session_mask & (trials == int(trial)))
                order = idx[np.argsort(rows[idx], kind="stable")]
                x = torch.from_numpy(vel[order].T[None]).float()
                y = basis(x).squeeze(0).T
                y = torch.cat([torch.relu(y), torch.relu(-y)], dim=1)
                out[order, 2:] = y.cpu().numpy().astype(np.float32)
    return out


def build_lagged_features(
    data: dict[str, np.ndarray],
    *,
    feature_key: str,
    lag_set: tuple[int, ...],
    split_id: int,
    include_eye: bool,
    include_action: bool,
    behavior_mode: str,
    behavior_features: np.ndarray | None,
    unit_mask: np.ndarray,
    row_step: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    split = data["split"].astype(np.int64)
    rows = data["row_indices"].astype(np.int64)
    trials = data["trial_inds"].astype(np.int64)
    sessions = row_session_ids(data)
    use_latent_feature = feature_key.lower() not in {"none", "null", "covariates", "behavior"}
    latent_keys = [k.strip() for k in feature_key.split("+") if k.strip()] if use_latent_feature else []
    missing = [k for k in latent_keys if k not in data]
    if missing:
        raise RuntimeError(f"Missing latent feature keys for feature_key={feature_key!r}: {missing}")
    features = [data[k].astype(np.float32) for k in latent_keys]
    robs = data["robs"].astype(np.float32)[:, unit_mask]
    dfs = data["dfs"].astype(np.float32)
    if dfs.ndim == 2 and dfs.shape[1] == unit_mask.shape[0]:
        dfs = dfs[:, unit_mask]
    if dfs.ndim == 2 and dfs.shape[1] not in {1, robs.shape[1]}:
        raise RuntimeError(f"dfs width {dfs.shape[1]} does not match robs width {robs.shape[1]}")
    eyepos = data["eyepos"].astype(np.float32)
    action = data["action"].astype(np.float32)

    index = {
        (int(sess), int(s), int(t), int(r)): i
        for i, (sess, s, t, r) in enumerate(zip(sessions, split, trials, rows, strict=False))
        if int(s) == int(split_id)
    }
    feats = []
    ys = []
    masks = []
    keep_rows = []
    candidates = np.flatnonzero(split == int(split_id))
    for i in candidates:
        pieces = []
        ok = True
        if use_latent_feature:
            for lag in lag_set:
                j = index.get(
                    (
                        int(sessions[i]),
                        int(split[i]),
                        int(trials[i]),
                        int(rows[i]) - int(lag) * row_step,
                    )
                )
                if j is None:
                    ok = False
                    break
                for feature in features:
                    pieces.append(feature[j])
        if not ok:
            continue
        if behavior_mode in {"raw", "raw+visioncore"} and include_eye:
            pieces.append(eyepos[i])
        if behavior_mode in {"raw", "raw+visioncore"} and include_action:
            pieces.append(action[i])
            pieces.append(np.asarray([np.linalg.norm(action[i])], dtype=np.float32))
        if behavior_mode in {"visioncore", "raw+visioncore"}:
            if behavior_features is None:
                raise RuntimeError("behavior_mode requests visioncore features, but behavior_features is None")
            pieces.append(behavior_features[i].astype(np.float32))
        if not pieces:
            pieces.append(np.asarray([1.0], dtype=np.float32))
        feats.append(np.concatenate(pieces).astype(np.float32))
        ys.append(robs[i].astype(np.float32))
        masks.append(dfs[i].astype(np.float32))
        keep_rows.append(rows[i])
    if not feats:
        raise RuntimeError(f"No valid lagged rows for split={split_id}, lag_set={lag_set}")
    return (
        np.stack(feats, axis=0),
        np.stack(ys, axis=0),
        np.stack(masks, axis=0),
        np.asarray(keep_rows, dtype=np.int64),
    )


def init_bias(y: np.ndarray, dfs: np.ndarray) -> torch.Tensor:
    if dfs.ndim == 1:
        dfs = dfs[:, None]
    weighted = y * dfs
    denom = dfs.sum(axis=0)
    if dfs.shape[1] == 1:
        denom = np.full(y.shape[1], float(denom[0]), dtype=np.float32)
    rate = weighted.sum(axis=0) / np.maximum(denom, 1.0)
    return torch.from_numpy(np.log(np.clip(rate, 1e-6, None)).astype(np.float32))


def make_model(cfg: ReadoutConfig, input_dim: int, n_units: int, bias: torch.Tensor) -> nn.Module:
    if cfg.arch == "linear":
        return PoissonLinear(input_dim, n_units, bias)
    if cfg.arch == "mlp":
        return PoissonMLP(
            input_dim,
            n_units,
            bias,
            hidden_dim=cfg.hidden_dim,
            depth=cfg.depth,
            dropout=cfg.dropout,
        )
    raise ValueError(f"Unknown arch: {cfg.arch}")


@torch.no_grad()
def predict_log_rate(model: nn.Module, x: torch.Tensor, batch_size: int, device: str) -> torch.Tensor:
    model.eval()
    outs = []
    for start in range(0, x.shape[0], batch_size):
        outs.append(model(x[start : start + batch_size].to(device)).cpu())
    return torch.cat(outs, dim=0)


def evaluate_model(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    dfs: torch.Tensor,
    batch_size: int,
    device: str,
) -> dict[str, object]:
    log_rate = predict_log_rate(model, x, batch_size, device)
    loss = float(masked_poisson_nll_lograte(log_rate, y, dfs).cpu())
    rate = log_rate.clamp(-20, 8).exp()
    bps = calc_poisson_bits_per_spike(rate, y, dfs).detach().cpu().numpy()
    corr = safe_corr(rate.numpy(), y.numpy(), dfs.numpy())
    return {
        "loss": loss,
        "bps": bps,
        "corr": corr,
        "mean_bps": float(np.nanmean(bps)),
        "median_bps": float(np.nanmedian(bps)),
        "p90_bps": float(np.nanquantile(bps, 0.9)),
        "mean_corr": float(np.nanmean(corr)),
        "median_corr": float(np.nanmedian(corr)),
        "p90_corr": float(np.nanquantile(corr, 0.9)),
    }


def train_one_config(
    cfg: ReadoutConfig,
    x_train_np: np.ndarray,
    y_train_np: np.ndarray,
    dfs_train_np: np.ndarray,
    x_val_np: np.ndarray,
    y_val_np: np.ndarray,
    dfs_val_np: np.ndarray,
    *,
    epochs: int,
    patience: int,
    batch_size: int,
    device: str,
    seed: int,
) -> tuple[nn.Module, dict[str, object]]:
    torch.manual_seed(seed)
    x_train_np, x_val_np, feature_mean, feature_std = zscore_train_val(x_train_np, x_val_np)
    x_train = torch.from_numpy(x_train_np.astype(np.float32))
    y_train = torch.from_numpy(y_train_np.astype(np.float32))
    dfs_train = torch.from_numpy(dfs_train_np.astype(np.float32))
    x_val = torch.from_numpy(x_val_np.astype(np.float32))
    y_val = torch.from_numpy(y_val_np.astype(np.float32))
    dfs_val = torch.from_numpy(dfs_val_np.astype(np.float32))

    model = make_model(cfg, x_train.shape[1], y_train.shape[1], init_bias(y_train_np, dfs_train_np)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    ds = TensorDataset(x_train, y_train, dfs_train)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    bad = 0
    for epoch in range(1, int(epochs) + 1):
        model.train()
        for xb, yb, mb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            mb = mb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = masked_poisson_nll_lograte(model(xb), yb, mb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        with torch.no_grad():
            val_log = predict_log_rate(model, x_val, batch_size, device)
            val_loss = float(masked_poisson_nll_lograte(val_log, y_val, dfs_val).cpu())
        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    train_eval = evaluate_model(model, x_train, y_train, dfs_train, batch_size, device)
    val_eval = evaluate_model(model, x_val, y_val, dfs_val, batch_size, device)
    metrics = {
        "best_epoch": best_epoch,
        "train": train_eval,
        "val": val_eval,
        "n_train": int(x_train.shape[0]),
        "n_val": int(x_val.shape[0]),
        "input_dim": int(x_train.shape[1]),
        "feature_mean": feature_mean.astype(np.float32),
        "feature_std": feature_std.astype(np.float32),
    }
    return model, metrics


def config_grid(args) -> list[ReadoutConfig]:
    feature_keys = [x.strip() for x in args.feature_keys.split(",") if x.strip()]
    lag_sets = [parse_lag_set(x) for x in args.lag_sets.split(";") if x.strip()]
    archs = [x.strip() for x in args.archs.split(",") if x.strip()]
    mlp_hidden = parse_int_list(args.mlp_hidden_dims)
    mlp_depths = parse_int_list(args.mlp_depths)
    dropouts = parse_float_list(args.dropouts)
    lin_wd = parse_float_list(args.linear_weight_decays)
    mlp_wd = parse_float_list(args.mlp_weight_decays)
    out = []
    for feature_key in feature_keys:
        for lag_set in lag_sets:
            for arch in archs:
                if arch == "linear":
                    for wd in lin_wd:
                        out.append(
                            ReadoutConfig(
                                arch=arch,
                                lag_set=lag_set,
                                feature_key=feature_key,
                                behavior_mode=args.behavior_mode,
                                include_eye=args.include_eye,
                                include_action=args.include_action,
                                hidden_dim=0,
                                depth=0,
                                dropout=0.0,
                                weight_decay=wd,
                                lr=args.lr,
                            )
                        )
                elif arch == "mlp":
                    for h in mlp_hidden:
                        for d in mlp_depths:
                            for drop in dropouts:
                                for wd in mlp_wd:
                                    out.append(
                                        ReadoutConfig(
                                            arch=arch,
                                            lag_set=lag_set,
                                            feature_key=feature_key,
                                            behavior_mode=args.behavior_mode,
                                            include_eye=args.include_eye,
                                            include_action=args.include_action,
                                            hidden_dim=h,
                                            depth=d,
                                            dropout=drop,
                                            weight_decay=wd,
                                            lr=args.lr,
                                        )
                                    )
                else:
                    raise ValueError(f"Unknown arch {arch}")
    if args.max_configs and len(out) > args.max_configs:
        rng = np.random.default_rng(args.seed)
        take = rng.choice(len(out), size=int(args.max_configs), replace=False)
        take.sort()
        out = [out[i] for i in take]
    return out


def flatten_metrics(cfg: ReadoutConfig, metrics: dict[str, object], config_id: int) -> dict[str, object]:
    row = {
        "config_id": int(config_id),
        **asdict(cfg),
        "lag_set": ",".join(str(x) for x in cfg.lag_set),
        "n_lags": len(cfg.lag_set),
        "best_epoch": metrics["best_epoch"],
        "n_train": metrics["n_train"],
        "n_val": metrics["n_val"],
        "input_dim": metrics["input_dim"],
    }
    for split in ["train", "val"]:
        m = metrics[split]
        for key in ["loss", "mean_bps", "median_bps", "p90_bps", "mean_corr", "median_corr", "p90_corr"]:
            row[f"{split}_{key}"] = m[key]
    return row


def write_rows(path: Path, rows: list[dict[str, object]]):
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    lat_path = Path(args.latents)
    data = dict(np.load(lat_path, allow_pickle=True))
    session_ids = row_session_ids(data)
    unique_sessions = np.unique(session_ids)
    selected_session_id = None
    selected_session_label = None
    if args.session_id is not None:
        if int(args.session_id) not in set(int(x) for x in unique_sessions):
            raise RuntimeError(f"Requested session_id={args.session_id}, available={unique_sessions.tolist()}")
        selected_session_id = int(args.session_id)
        data = filter_rows(data, session_ids == int(args.session_id))
        session_ids = row_session_ids(data)
        session_unit_counts = data.get("session_unit_counts")
        if session_unit_counts is not None:
            unit_count = int(np.asarray(session_unit_counts).reshape(-1)[int(args.session_id)])
            for key in ["robs", "dfs"]:
                if key in data and np.asarray(data[key]).ndim == 2 and data[key].shape[1] > unit_count:
                    data[key] = data[key][:, :unit_count]
        session_names = data.get("session_names")
        session_label = str(args.session_id)
        if session_names is not None and int(args.session_id) < len(session_names):
            name = session_names[int(args.session_id)]
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            session_label = str(name)
        selected_session_label = session_label
        print(f"filtered to session_id={args.session_id} ({session_label}) rows={len(data['split'])}", flush=True)
    elif len(unique_sessions) > 1 and not args.allow_mixed_sessions:
        raise RuntimeError(
            "Latent file contains multiple session_id values. "
            "Train per-session readouts with --session-id, or pass --allow-mixed-sessions for an explicit flat control."
        )
    row_step = int(args.row_step)
    if row_step <= 0:
        row_step = int(np.asarray(data.get("downsample", np.array([2]))).reshape(-1)[0])
    behavior_features = None
    if args.behavior_mode in {"visioncore", "raw+visioncore"}:
        print("building VisionCore-style behavior basis...", flush=True)
        behavior_features = build_visioncore_behavior(data, row_step=row_step)
    outdir = Path(args.outdir) if args.outdir else lat_path.with_suffix("").with_name(lat_path.stem + "_spike_readout")
    outdir.mkdir(parents=True, exist_ok=True)

    robs = data["robs"].astype(np.float32)
    dfs_all = data["dfs"].astype(np.float32)
    dfs_exp = expand_dfs(dfs_all, robs.shape[1])
    train = data["split"].astype(np.int64) == 0
    train_spikes = (robs[train] * dfs_exp[train]).sum(axis=0)
    unit_mask = train_spikes >= float(args.min_train_spikes)
    if not unit_mask.any():
        raise RuntimeError("No units passed min-train-spikes threshold")
    print(f"units kept: {int(unit_mask.sum())}/{len(unit_mask)} min_train_spikes={args.min_train_spikes}", flush=True)
    print(f"row_step={row_step}", flush=True)

    configs = config_grid(args)
    rows = []
    best = None
    best_payload = None
    for ci, cfg in enumerate(configs):
        print(f"[{ci + 1}/{len(configs)}] {cfg}", flush=True)
        try:
            x_train, y_train, dfs_train, train_rows = build_lagged_features(
                data,
                feature_key=cfg.feature_key,
                lag_set=cfg.lag_set,
                split_id=0,
                include_eye=cfg.include_eye,
                include_action=cfg.include_action,
                behavior_mode=cfg.behavior_mode,
                behavior_features=behavior_features,
                unit_mask=unit_mask,
                row_step=row_step,
            )
            x_val, y_val, dfs_val, val_rows = build_lagged_features(
                data,
                feature_key=cfg.feature_key,
                lag_set=cfg.lag_set,
                split_id=1,
                include_eye=cfg.include_eye,
                include_action=cfg.include_action,
                behavior_mode=cfg.behavior_mode,
                behavior_features=behavior_features,
                unit_mask=unit_mask,
                row_step=row_step,
            )
        except RuntimeError as exc:
            print(f"  skipped: {exc}", flush=True)
            continue
        model, metrics = train_one_config(
            cfg,
            x_train,
            y_train,
            dfs_train,
            x_val,
            y_val,
            dfs_val,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            device=args.device,
            seed=args.seed + ci,
        )
        row = flatten_metrics(cfg, metrics, ci)
        rows.append(row)
        print(
            "  val "
            f"loss={row['val_loss']:.5f} mean_bps={row['val_mean_bps']:.4f} "
            f"median_bps={row['val_median_bps']:.4f} median_r={row['val_median_corr']:.4f} "
            f"n_train={row['n_train']} n_val={row['n_val']}",
            flush=True,
        )
        score = (row["val_mean_bps"], -row["val_loss"])
        if best is None or score > best:
            best = score
            best_payload = {
                "config_id": ci,
                "config": asdict(cfg),
                "metrics": {
                    k: v
                    for k, v in metrics.items()
                    if k not in {"train", "val"}
                },
                "train_eval": {k: v for k, v in metrics["train"].items() if k not in {"bps", "corr"}},
                "val_eval": {k: v for k, v in metrics["val"].items() if k not in {"bps", "corr"}},
                "unit_mask": unit_mask,
                "feature_mean": metrics["feature_mean"],
                "feature_std": metrics["feature_std"],
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "row_step": row_step,
                "session_id": selected_session_id,
                "session_label": selected_session_label,
            }
            np.save(outdir / "best_val_bps.npy", metrics["val"]["bps"])
            np.save(outdir / "best_val_corr.npy", metrics["val"]["corr"])
        write_rows(outdir / "results.csv", rows)

    if best_payload is not None:
        torch.save(best_payload, outdir / "best_model.pt")
        with (outdir / "best_summary.json").open("w") as f:
            json.dump(
                {
                    "latents": str(lat_path),
                    "best_config_id": best_payload["config_id"],
                    "best_config": best_payload["config"],
                    "best_train_eval": best_payload["train_eval"],
                    "best_val_eval": best_payload["val_eval"],
                    "n_units": int(unit_mask.sum()),
                    "all_units": int(len(unit_mask)),
                    "row_step": int(row_step),
                    "session_id": selected_session_id,
                    "session_label": selected_session_label,
                },
                f,
                indent=2,
            )
    print(f"saved readout search: {outdir}", flush=True)


if __name__ == "__main__":
    main()
