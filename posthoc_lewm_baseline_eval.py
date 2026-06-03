import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
import tempfile
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from einops import rearrange
from omegaconf import OmegaConf, open_dict

import planning
from utils import get_column_normalizer, get_img_preprocessor


def _register_resolvers():
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)


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


def _load_cfg(run_name):
    _register_resolvers()
    return OmegaConf.load(f"/home/jake/.stable_worldmodel/checkpoints/{run_name}/config.yaml")


def _dataset(cfg, *, num_steps, batch_size=16, normalize=True):
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_cfg["num_steps"] = num_steps
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    ds = swm.data.load_dataset(dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg)
    transforms = [get_img_preprocessor("pixels", "pixels", img_size=cfg.img_size, normalize=normalize)]
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            transforms.append(get_column_normalizer(ds, col, col))
    ds.transform = spt.data.transforms.Compose(*transforms)
    gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(ds, [cfg.train_split, 1 - cfg.train_split], generator=gen)
    return torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)


def _load_model(cfg, run_name, epoch, device):
    model = hydra.utils.instantiate(cfg.model)
    path = f"/home/jake/.stable_worldmodel/checkpoints/{run_name}/weights_epoch_{epoch}.pt"
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval().requires_grad_(False)
    model.interpolate_pos_encoding = True
    return model


@torch.no_grad()
def _latent_rollout_metrics(run_name, cfg, model, device, out_dir):
    loader = _dataset(cfg, num_steps=cfg.history_size + 8, batch_size=16, normalize=True)
    batch = next(iter(loader))
    batch = {k: (torch.nan_to_num(v.to(device), 0.0) if torch.is_tensor(v) else v) for k, v in batch.items()}
    out = model.encode(batch)
    true_emb = out["emb"]
    act_emb = out["act_emb"]
    HS = cfg.history_size
    emb = true_emb[:, :HS].clone()
    errs = []
    for t in range(HS, true_emb.size(1)):
        pred = model.predict(emb[:, -HS:], act_emb[:, t - HS:t])[:, -1:]
        errs.append((pred[:, 0] - true_emb[:, t]).pow(2).mean(dim=-1))
        emb = torch.cat([emb, pred], dim=1)
    per_h = torch.stack(errs, dim=1).mean(dim=0).detach().cpu()
    metrics = {"latent_rollout_mse": float(per_h.mean())}
    for h in (1, 4, 8):
        if h - 1 < per_h.numel():
            metrics[f"latent_rollout_mse_h{h}"] = float(per_h[h - 1])
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{run_name}_latent_rollout_mse.json").open("w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    print(f"[{run_name}] latent rollout metrics: {metrics}")
    return metrics


def _planning_eval(run_name, cfg, model, device, out_dir, num_eval):
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
        env_name=cfg.monitor.env_name,
        num_eval=num_eval,
        eval_budget=cfg.monitor.eval_budget,
        goal_offset_steps=cfg.monitor.goal_offset_steps,
        plan_config=OmegaConf.to_container(cfg.monitor.plan_config, resolve=True),
        cem_kwargs=OmegaConf.to_container(cfg.monitor.cem, resolve=True),
        callables=OmegaConf.to_container(cfg.monitor.callables, resolve=True),
        process=planning.build_process(planning_dataset, cfg.monitor.keys_to_cache),
        transform={
            "pixels": planning.planning_img_transform(cfg.img_size, normalize_img=True),
            "goal": planning.planning_img_transform(cfg.img_size, normalize_img=True),
        },
    )
    kwargs["cem_kwargs"]["device"] = str(device)
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        metrics, videos = planning.run_planning_eval(model, planning_dataset, video_dir=tmp, **kwargs)
        target = out_dir / f"{run_name}_planning_videos"
        target.mkdir(parents=True, exist_ok=True)
        for v in videos:
            Path(v).rename(target / Path(v).name)
        clean = _jsonable(metrics)
        with (out_dir / f"{run_name}_planning_metrics.json").open("w") as f:
            json.dump(clean, f, indent=2, sort_keys=True)
        print(f"[{run_name}] planning metrics: {metrics}")
        return clean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="lewm_baseline_20260602_212037")
    parser.add_argument("--epoch", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-eval", type=int, default=4)
    parser.add_argument("--skip-planning", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path("logs") / "posthoc_lewm_baseline_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = _load_cfg(args.run)
    model = _load_model(cfg, args.run, args.epoch, device)
    metrics = {"latent_rollout": _latent_rollout_metrics(args.run, cfg, model, device, out_dir)}
    if not args.skip_planning:
        metrics["planning"] = _planning_eval(args.run, cfg, model, device, out_dir, args.num_eval)
    with (out_dir / f"{args.run}_posthoc_metrics.json").open("w") as f:
        json.dump(_jsonable(metrics), f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
