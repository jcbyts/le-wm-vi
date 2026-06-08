import os

os.environ.setdefault("MUJOCO_GL", "egl")  # headless rendering for in-loop eval

from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

import planning
from model import vijepa_forward
from module import SIGReg
from monitor import BehaviorEvalCallback, FondVizCallback
from utils import get_column_normalizer, get_img_preprocessor, SaveCkptCallback


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]  

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


def poiswm_forward(self, batch, stage, cfg):
    """LeWM architecture with exact Poisson KL and Capacity-SIGReg."""
    output = self.model.compute_loss(batch, cfg)

    log_keys = {
        "loss",
        "pred_loss",
        "anchor_loss",
        "reg_loss",
        "beta",
        "A_over_mu",
        "mean_rate_mean",
        "mean_rate_std",
        "rate_min_seen",
        "rate_max_seen",
        "effective_rank",
        "poisson_mi_mean",
        "poisson_mi_min",
        "poisson_mi_max",
        "capacity_target",
        "tau",
        "effective_rank_u",
        "u_mean",
        "u_std",
        "rate_mean",
        "rate_std",
        "fisher_weight_mean",
        "fisher_weight_max",
        "alpha",
        "target_rate",
        "lambda0",
        "log_rate_min",
        "log_rate_max",
        "effective_rank_r",
        "r_mean",
        "r_std",
        "r_at_min_frac",
        "r_at_max_frac",
        "prior_kl_mean",
        "prior_kl_std",
        "pred_rate_mean",
        "pred_rate_max",
        "target_rate_mean",
        "target_rate_std",
        "target_prior_kl_mean",
        "compact_loss",
        "compact_loss_weight",
        "effective_rank_z",
        "z_mean",
        "z_std",
    }
    losses_dict = {
        f"{stage}/{k}": v.detach()
        for k, v in output.items()
        if k in log_keys and torch.is_tensor(v)
    }
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    # Most encoder-only runs use the ImageNet-normalized ViT path. FOND can opt
    # into either raw [0,1] decoder targets or normalized image-space targets.
    normalize_img = cfg.get(
        "normalize_img", cfg.get("forward_type", "lejepa") != "vijepa"
    )
    transforms = [get_img_preprocessor(source='pixels', target='pixels',
                                       img_size=cfg.img_size, normalize=normalize_img)]
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)
    init_weights = cfg.get("init_weights")
    if init_weights:
        pretrained = swm.wm.utils.load_pretrained(init_weights)
        world_model.load_state_dict(pretrained.state_dict())

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    forward_type = cfg.get("forward_type", "lejepa")
    if forward_type == "vijepa":
        # variants 3-6: variational FOND-JEPA, reconstruction anchor, no SIGReg
        world_model = spt.Module(
            model=world_model,
            forward=partial(vijepa_forward, cfg=cfg),
            optim=optimizers,
        )
    elif forward_type == "poiswm":
        world_model = spt.Module(
            model=world_model,
            forward=partial(poiswm_forward, cfg=cfg),
            optim=optimizers,
        )
    else:
        # variants 1-2: LeWM MSE-JEPA + SIGReg (variant 1 = sigreg weight 0)
        world_model = spt.Module(
            model=world_model,
            sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
            forward=partial(lejepa_forward, cfg=cfg),
            optim=optimizers,
        )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name,
        cfg=cfg.model,
        epoch_interval=1,
        epoch_offset=cfg.get("ckpt_epoch_offset", 0),
    )

    #############################
    ##     behavior monitor    ##
    #############################

    monitor_callbacks = []
    if cfg.get("forward_type", "lejepa") == "vijepa" and cfg.loss.get("log_viz", True):
        monitor_callbacks.append(FondVizCallback(
            history_size=cfg.history_size,
            val_loader=val,
            num_frames=cfg.loss.get("viz_num_frames", 8),
        ))

    if cfg.get("monitor") and cfg.monitor.enabled:
        monitor_keys = list(cfg.monitor.keys_to_cache)

        # Raw (untransformed) dataset for planning state/goal extraction.
        planning_dataset = swm.data.load_dataset(
            dataset_name, transform=None, cache_dir=cache_dir,
            keys_to_cache=monitor_keys,
        )
        planning_kwargs = dict(
            env_name=cfg.monitor.env_name,
            num_eval=cfg.monitor.num_eval,
            eval_budget=cfg.monitor.eval_budget,
            goal_offset_steps=cfg.monitor.goal_offset_steps,
            plan_config=OmegaConf.to_container(cfg.monitor.plan_config, resolve=True),
            cem_kwargs=OmegaConf.to_container(cfg.monitor.cem, resolve=True),
            callables=OmegaConf.to_container(cfg.monitor.callables, resolve=True),
            process=planning.build_process(planning_dataset, monitor_keys),
            transform={
                "pixels": planning.planning_img_transform(cfg.img_size, normalize_img=normalize_img),
                "goal": planning.planning_img_transform(cfg.img_size, normalize_img=normalize_img),
            },
        )

        # Long-horizon loader for the open-loop latent-rollout metric.
        latent_cfg = dict(dataset_cfg)
        latent_cfg["num_steps"] = cfg.history_size + cfg.monitor.roll_horizon
        latent_ds = swm.data.load_dataset(
            dataset_name, transform=transform, cache_dir=cache_dir, **latent_cfg
        )
        n_latent = min(
            len(latent_ds), cfg.monitor.num_latent_batches * cfg.loader.batch_size
        )
        latent_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(latent_ds, list(range(n_latent))),
            batch_size=cfg.loader.batch_size, shuffle=False, num_workers=2,
        )

        monitor_callbacks.append(BehaviorEvalCallback(
            every_n_epochs=cfg.monitor.every_n_epochs,
            history_size=cfg.history_size,
            latent_loader=latent_loader,
            num_latent_batches=cfg.monitor.num_latent_batches,
            planning_dataset=planning_dataset,
            planning_kwargs=planning_kwargs,
            num_videos=cfg.monitor.num_videos,
        ))

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback] + monitor_callbacks,
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()
    return


if __name__ == "__main__":
    run()
