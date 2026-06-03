import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import tempfile
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf, open_dict
from torchvision.utils import save_image

import planning
from utils import get_column_normalizer, get_img_preprocessor


def _register_resolvers():
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)


def _load_cfg(run_name):
    _register_resolvers()
    cfg = OmegaConf.load(f"/home/jake/.stable_worldmodel/checkpoints/{run_name}/config.yaml")
    return cfg


def _dataset(cfg, *, num_steps, batch_size=16, split="val"):
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_cfg["num_steps"] = num_steps
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    ds = swm.data.load_dataset(dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg)
    transforms = [get_img_preprocessor("pixels", "pixels", img_size=cfg.img_size, normalize=False)]
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            transforms.append(get_column_normalizer(ds, col, col))
    ds.transform = spt.data.transforms.Compose(*transforms)
    gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(ds, [cfg.train_split, 1 - cfg.train_split], generator=gen)
    subset = val_set if split == "val" else train_set
    return torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=2)


def _load_model(cfg, run_name, epoch, device):
    model = hydra.utils.instantiate(cfg.model)
    path = f"/home/jake/.stable_worldmodel/checkpoints/{run_name}/weights_epoch_{epoch}.pt"
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval().requires_grad_(False)
    model.infer_backprop = False
    return model


def _filmstrip(frames):
    frames = frames.detach().float().cpu().clamp(0, 1)
    if frames.ndim == 4:
        frames = frames.unsqueeze(0)
    rows = [torch.cat([frame for frame in seq], dim=2) for seq in frames]
    return torch.cat(rows, dim=1)


def _wandb_image(tensor, caption=None):
    import wandb
    img = _filmstrip(tensor).permute(1, 2, 0).numpy()
    return wandb.Image(img, caption=caption)


def _log_media(run_name, cfg, model, device, out_dir, step, use_wandb=True):
    wandb = None
    if use_wandb:
        import wandb as _wandb
        wandb = _wandb
    loader = _dataset(cfg, num_steps=cfg.history_size + 8, batch_size=16)
    batch = next(iter(loader))
    batch = {k: (torch.nan_to_num(v.to(device), 0.0) if torch.is_tensor(v) else v) for k, v in batch.items()}
    beta = float(cfg.loss.beta)
    info = model.filter_sequence(batch, cfg.history_size, beta=beta, infer_objective="free_energy", return_diag=False)
    eta = info["emb"]
    target = batch["pixels"].float().clamp(0, 1)
    recon = model.decode(eta).detach().clamp(0, 1)
    B, T = eta.shape[:2]
    n, t = min(B, 4), min(T, 8)

    flat_eta = eta.detach().reshape(B * T, -1)
    mu = flat_eta.mean(0, keepdim=True)
    std = flat_eta.std(0, keepdim=True).clamp_min(1e-4)
    agg = mu + std * torch.randn(n * t, flat_eta.size(-1), device=device)
    agg_img = model.decode(agg.view(n, t, -1)).detach().clamp(0, 1)
    prior = model.prior_param.expand(n * t, -1)
    prior_img = model.decode(prior.view(n, t, -1)).detach().clamp(0, 1)

    HS = cfg.history_size
    emb = eta[:, :HS].detach().clone()
    preds = []
    act_emb = info["act_emb"]
    for idx in range(HS, T):
        pred = model.predict(emb[:, -HS:], act_emb[:, idx - HS:idx])[:, -1:]
        preds.append(pred)
        emb = torch.cat([emb, pred], dim=1)
    pred_eta = torch.cat(preds, dim=1)
    pred_img = model.decode(pred_eta).detach().clamp(0, 1)
    true_img = target[:, HS:T]
    mse = {}
    for h in (1, 4, 8):
        i = h - 1
        if i < pred_img.size(1):
            mse[h] = F.mse_loss(pred_img[:, i], true_img[:, i]).item()

    out_dir.mkdir(parents=True, exist_ok=True)
    images = {
        "recon_input": target[:n, :t],
        "recon_posterior": recon[:n, :t],
        "hall_aggpost": agg_img,
        "hall_prior": prior_img,
        "rollout_true": true_img[:n, :t],
        "rollout_pred": pred_img[:n, :t],
    }
    for name, tensor in images.items():
        save_image(_filmstrip(tensor), out_dir / f"{run_name}_{name}.png")

    metrics = {
        "eta_absmean": eta.abs().mean().item(),
        "eta_std": eta.std().item(),
        **{f"rollout_mse_h{h}": value for h, value in mse.items()},
    }
    if use_wandb:
        payload = {
            "posthoc/recon_input": _wandb_image(images["recon_input"]),
            "posthoc/recon_posterior": _wandb_image(images["recon_posterior"]),
            "posthoc/hall_aggpost": _wandb_image(images["hall_aggpost"]),
            "posthoc/hall_prior": _wandb_image(images["hall_prior"], "prior samples (may look unstructured even for a good code)"),
            "posthoc/rollout_true": _wandb_image(images["rollout_true"]),
            "posthoc/rollout_pred": _wandb_image(images["rollout_pred"]),
            "posthoc/eta_absmean": metrics["eta_absmean"],
            "posthoc/eta_std": metrics["eta_std"],
        }
        for h, value in mse.items():
            payload[f"posthoc/rollout_mse_h{h}"] = value
        wandb.log(payload, step=step)
    return metrics
    print(f"[{run_name}] logged media and rollout MSE: {mse}")


def _planning_eval(run_name, cfg, model, device, out_dir, step, num_eval, use_wandb=True):
    wandb = None
    if use_wandb:
        import wandb as _wandb
        wandb = _wandb
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    planning_dataset = swm.data.load_dataset(
        dataset_name,
        transform=None,
        cache_dir=cache_dir,
        keys_to_cache=["action", "proprio", "state"],
    )
    kwargs = dict(
        env_name="swm/PushT-v1",
        num_eval=num_eval,
        eval_budget=50,
        goal_offset_steps=25,
        plan_config={"horizon": 5, "receding_horizon": 5, "action_block": 5},
        cem_kwargs={"batch_size": 1, "num_samples": 300, "var_scale": 1.0, "n_steps": 30, "topk": 30, "device": str(device), "seed": 42},
        callables=[
            {"method": "_set_state", "args": {"state": {"value": "state"}}},
            {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_state"}}},
        ],
        process=planning.build_process(planning_dataset, ["action", "proprio", "state"]),
        transform={
            "pixels": planning.planning_img_transform(cfg.img_size, normalize_img=False),
            "goal": planning.planning_img_transform(cfg.img_size, normalize_img=False),
        },
    )
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        metrics, videos = planning.run_planning_eval(model, planning_dataset, video_dir=tmp, **kwargs)
        payload = {"posthoc/success_rate": float(metrics["success_rate"])}
        for k, v in metrics.items():
            if isinstance(v, (int, float, np.floating)):
                payload[f"posthoc/{k}"] = float(v)
        if use_wandb:
            payload["posthoc/rollout_videos"] = [wandb.Video(v, format="mp4") for v in videos[: min(4, len(videos))]]
            wandb.log(payload, step=step)
        target = out_dir / f"{run_name}_planning_videos"
        target.mkdir(parents=True, exist_ok=True)
        for v in videos:
            Path(v).rename(target / Path(v).name)
        print(f"[{run_name}] planning metrics: {metrics}")
        return {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in metrics.items()}


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--epoch", type=int, default=2)
    parser.add_argument("--step", type=int, default=15000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-eval", type=int, default=4)
    parser.add_argument("--skip-planning", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    import json
    wandb = None
    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb as _wandb
        wandb = _wandb
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path("logs") / "posthoc_fond_media_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    for run_name in args.run:
        cfg = _load_cfg(run_name)
        model = _load_model(cfg, run_name, args.epoch, device)
        if use_wandb:
            wandb.init(project="lewm", id=run_name, name=run_name, resume="allow")
        try:
            metrics = {"media": _log_media(run_name, cfg, model, device, out_dir, args.step, use_wandb=use_wandb)}
            if not args.skip_planning:
                metrics["planning"] = _planning_eval(run_name, cfg, model, device, out_dir, args.step, args.num_eval, use_wandb=use_wandb)
            with (out_dir / f"{run_name}_posthoc_metrics.json").open("w") as f:
                json.dump(_jsonable(metrics), f, indent=2, sort_keys=True)
        finally:
            if use_wandb:
                wandb.finish()


if __name__ == "__main__":
    main()
