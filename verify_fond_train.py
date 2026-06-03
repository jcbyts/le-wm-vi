"""Real-data verification of the variational FOND-JEPA forward (spec §8 stage 4).

Builds the dataset + FONDJEPA model exactly as train.py does (PushT, 64px [0,1]
pixels, frameskip actions), then runs real batches through vijepa_forward for all
four new variants, asserting every loss component is finite and printing the §5.2
predictive-vs-noop diagnostic. No training; this just proves the forward path is
correct on real data before any long run.

  conda run -n lewm python verify_fond_train.py
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")
import types
from functools import partial

import hydra
import torch
import stable_pretraining as spt
import stable_worldmodel as swm
from omegaconf import OmegaConf, open_dict

from model import vijepa_forward
from utils import get_column_normalizer, get_img_preprocessor


def build_loader_and_dims(cfg):
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    dataset = swm.data.load_dataset(dataset_name, transform=None, cache_dir=None, **dataset_cfg)
    transforms = [get_img_preprocessor(source="pixels", target="pixels",
                                       img_size=cfg.img_size, normalize=False)]
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            transforms.append(get_column_normalizer(dataset, col, col))
        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")
    dataset.transform = spt.data.transforms.Compose(*transforms)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)
    return loader


def run_variant(cfg, model_name, pred_loss, batch):
    with hydra.initialize(version_base=None, config_path="config/train"):
        mcfg = hydra.compose(config_name="fond", overrides=[f"model={model_name}"])
    # carry over the action input_dim resolved against the dataset
    mcfg.model.action_encoder.input_dim = cfg.model.action_encoder.input_dim
    model = hydra.utils.instantiate(mcfg.model)
    P = model.param_dim

    m = types.SimpleNamespace(model=model, log_dict=lambda *a, **k: None)
    fcfg = types.SimpleNamespace(
        history_size=cfg.history_size, num_preds=cfg.num_preds,
        loss=types.SimpleNamespace(get=lambda k, d=None:
            {"kl_weight": 1.0, "recon_weight": 1.0, "pred_loss": pred_loss}.get(k, d)),
    )
    b = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    out = vijepa_forward(m, b, "val", fcfg)
    for k in ("pred_loss", "recon_loss", "loss", "kl_exact", "fisher_quad", "pred_noop"):
        assert torch.isfinite(out[k]), f"{model_name}/{pred_loss}: {k} not finite"
    gap = out["pred_noop"].item() - out["pred_loss"].item()
    print(f"[{model_name:13s}/{pred_loss:16s}] P={P:4d}  "
          f"pred={out['pred_loss'].item():8.3f}  recon={out['recon_loss'].item():.4f}  "
          f"kl={out['kl_exact'].item():8.3f}  fq={out['fisher_quad'].item():8.3f}  "
          f"noop={out['pred_noop'].item():7.3f}  loss={out['loss'].item():8.3f}")
    return out


def main():
    with hydra.initialize(version_base=None, config_path="config/train"):
        cfg = hydra.compose(config_name="fond")
    loader = build_loader_and_dims(cfg)
    batch = next(iter(loader))
    px = batch["pixels"]
    print(f"batch: pixels {tuple(px.shape)} range[{px.min():.3f},{px.max():.3f}]  "
          f"action {tuple(batch['action'].shape)}")
    assert px.min() >= -1e-4 and px.max() <= 1.0 + 1e-4, "pixels not in [0,1]"

    for model_name in ["fond_poisson", "fond_gaussian"]:
        for pred_loss in ["exact_kl", "quadratic_fisher"]:
            run_variant(cfg, model_name, pred_loss, batch)
    print("\nALL VARIANTS: forward finite on real PushT data  OK")


if __name__ == "__main__":
    main()
