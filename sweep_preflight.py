"""Tiny K/recon_weight sweep to find the stable window (NOT a performance run).

For families {gaussian, poisson} x loss {exact_kl} x K {1,2,4,8,16} x
recon_weight {1e-4,3e-4,1e-3,3e-3,1e-2,3e-2}: train a fresh model for a few
hundred steps on a TINY PushT subset, then log the full diagnostic row on a
held-out batch. The window we want passes ALL of:
  - collapse gate (rank/var/temporal),
  - correction_norm nonzero,
  - recon_gain > 0,
  - D_pred < D_noop (noop_ratio < 1),
  - exact/quad ratio not pathological,
  - rates/logvars not saturated at the clamps.

Env knobs: SWEEP_STEPS (default 250), SWEEP_QUICK=1 (tiny grid + 20 steps).
Writes fond_sweep_results.csv and prints a table.

  conda run -n lewm python sweep_preflight.py
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")
import csv
import types
import itertools

import hydra
import torch
import stable_pretraining as spt
import stable_worldmodel as swm
from omegaconf import OmegaConf, open_dict

from model import FONDJEPA, ConvDecoder, vijepa_forward
from module import ARPredictor, Embedder
from latent import make_head, variant_name
from diagnostics import collapse_report
from utils import get_column_normalizer, get_img_preprocessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STEPS = int(os.environ.get("SWEEP_STEPS", "250"))
QUICK = os.environ.get("SWEEP_QUICK", "") != ""
EMBED_DIM, HISTORY, NUM_PREDS, IMG = 192, 3, 1, 64
N_TINY, BATCH = (128, 16) if QUICK else (512, 32)
K_GRID = [4] if QUICK else [1, 2, 4, 8, 16]
RW_GRID = [1e-3, 1e-2] if QUICK else [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
FAMILIES = ["poisson", "gaussian"]
if QUICK:
    STEPS = 20


def tiny_loader():
    with hydra.initialize(version_base=None, config_path="config/train"):
        cfg = hydra.compose(config_name="fond")
    dcfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    name = dcfg.pop("name")
    ds = swm.data.load_dataset(name, transform=None, cache_dir=None, **dcfg)
    tfs = [get_img_preprocessor("pixels", "pixels", img_size=IMG, normalize=False)]
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            tfs.append(get_column_normalizer(ds, col, col))
    ds.transform = spt.data.transforms.Compose(*tfs)
    act_in = cfg.data.dataset.frameskip * ds.get_dim("action")
    sub = torch.utils.data.Subset(ds, list(range(min(N_TINY, len(ds)))))
    loader = torch.utils.data.DataLoader(sub, batch_size=BATCH, shuffle=True,
                                         drop_last=True, num_workers=2)
    return loader, act_in


def build_model(family, K, act_in):
    head = make_head(family)
    P = EMBED_DIM * head.param_mult
    predictor = ARPredictor(num_frames=HISTORY, input_dim=P, hidden_dim=EMBED_DIM,
                            output_dim=P, depth=6, heads=16, mlp_dim=2048,
                            dim_head=64, dropout=0.1, emb_dropout=0.0)
    action_encoder = Embedder(input_dim=act_in, emb_dim=P)
    decoder = ConvDecoder(EMBED_DIM, img_ch=3, img_hw=IMG, grid=8)
    return FONDJEPA(decoder=decoder, predictor=predictor, action_encoder=action_encoder,
                    latent_dim=EMBED_DIM, head=head, k_inner=K, tau=0.2, infer_lr=1.0,
                    img_ch=3, img_hw=IMG).to(DEVICE)


def cfg_obj(recon_weight, log_diag):
    d = {"kl_weight": 1.0, "recon_weight": recon_weight, "pred_loss": "exact_kl",
         "log_diag": log_diag}
    return types.SimpleNamespace(history_size=HISTORY, num_preds=NUM_PREDS,
                                 loss=types.SimpleNamespace(get=lambda k, dd=None: d.get(k, dd)))


def to_dev(batch):
    return {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in batch.items()}


def run_config(family, K, rw, loader, act_in, eval_batch):
    torch.manual_seed(0)
    model = build_model(family, K, act_in)
    wrap = types.SimpleNamespace(model=model, log_dict=lambda *a, **k: None)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)
    fcfg_train = cfg_obj(rw, log_diag=False)

    data = itertools.cycle(loader)
    model.train()
    nonfinite = False
    for _ in range(STEPS):
        batch = to_dev(next(data))
        out = vijepa_forward(wrap, batch, "train", fcfg_train)
        if not torch.isfinite(out["loss"]):
            nonfinite = True
            break
        opt.zero_grad()
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()
    with torch.no_grad():
        out = vijepa_forward(wrap, to_dev(eval_batch), "val", cfg_obj(rw, log_diag=True))
        emb = out["emb"]
        rep = collapse_report(emb)
        stats = model.head.param_stats(emb)
    g = lambda k: float(out[k].item()) if k in out and torch.is_tensor(out[k]) else float("nan")
    row = {
        "variant": variant_name(family, "exact_kl"),
        "K": K, "recon_w": rw, "nonfinite": int(nonfinite),
        "rank_frac": round(rep["eff_rank_frac"], 3),
        "var_med": f"{rep['batch_var_median']:.2e}",
        "temp_var": f"{rep['temporal_var_mean']:.2e}",
        "collapse_pass": int(rep["passed"]),
        "corr_norm": round(g("correction_norm"), 4),
        "recon_gain": round(g("recon_gain"), 5),
        "D_pred": round(g("pred_loss"), 3),
        "D_noop": round(g("pred_noop"), 3),
        "noop_ratio": round(g("noop_ratio"), 3),
        "kl_exact": round(g("kl_exact"), 3),
        "fisher_q": round(g("fisher_quad"), 3),
        "exact/quad": round(g("exact_quad_ratio"), 3),
        "sat_frac": round(stats.get("sat_frac", 0.0), 3),
        "rate_or_lv_max": round(stats.get("rate_max", stats.get("logvar_max", float("nan"))), 2),
    }
    return row


def main():
    print(f"device={DEVICE} steps={STEPS} quick={QUICK} batch={BATCH} N_tiny={N_TINY}")
    loader, act_in = tiny_loader()
    eval_batch = next(iter(loader))      # fixed held-out-ish batch for all configs
    rows = []
    combos = list(itertools.product(FAMILIES, K_GRID, RW_GRID))
    for i, (family, K, rw) in enumerate(combos):
        try:
            row = run_config(family, K, rw, loader, act_in, eval_batch)
        except Exception as e:
            row = {"variant": variant_name(family, "exact_kl"), "K": K, "recon_w": rw,
                   "nonfinite": 1, "error": str(e)[:60]}
        rows.append(row)
        print(f"[{i+1}/{len(combos)}] {row}")

    cols = ["variant", "K", "recon_w", "nonfinite", "rank_frac", "var_med", "temp_var",
            "collapse_pass", "corr_norm", "recon_gain", "D_pred", "D_noop", "noop_ratio",
            "kl_exact", "fisher_q", "exact/quad", "sat_frac", "rate_or_lv_max"]
    with open("fond_sweep_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols + ["error"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("\nwrote fond_sweep_results.csv")


if __name__ == "__main__":
    main()
