"""Reduced online-filtering (scheme A) pilot — find the stable window.

Stability grid: families {poisson, gaussian} x loss {exact_kl} x
infer_objective {free_energy} x K {4,8} x beta {0.1,1.0,10.0} x infer_lr {0.05,0.1,0.3}. Trains each fresh on a
TINY PushT subset then logs the diagnostic row on a held-out batch.

Hard diagnostic (spec / user): a row with rate_max >= exp(5) is flagged SATURATED.

Env: SWEEP_STEPS (default 150), SWEEP_QUICK=1 (tiny grid + 20 steps).
Writes fond_sweep_online.csv.
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")
import csv
import math
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
STEPS = int(os.environ.get("SWEEP_STEPS", "150"))
QUICK = os.environ.get("SWEEP_QUICK", "") != ""
EMBED_DIM, HISTORY, NUM_PREDS, IMG = 192, 3, 1, 64
N_TINY, BATCH = (128, 16) if QUICK else (512, 32)
K_GRID = [4] if QUICK else [4, 8]
BETA_GRID = [10.0] if QUICK else [0.1, 1.0, 10.0]
LR_GRID = [0.1] if QUICK else [0.05, 0.1, 0.3]
FAMILIES = ["poisson", "gaussian"]
OBJECTIVES = ["free_energy"]
INFER_GRAD_CLIP = 1.0
INFER_MOMENTUM = 0.5
SAT_RATE = math.exp(5.0)
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


def build_model(family, K, infer_lr, act_in):
    head = make_head(family)
    P = EMBED_DIM * head.param_mult
    predictor = ARPredictor(num_frames=HISTORY, input_dim=P, hidden_dim=EMBED_DIM,
                            output_dim=P, depth=6, heads=16, mlp_dim=2048,
                            dim_head=64, dropout=0.1, emb_dropout=0.0)
    action_encoder = Embedder(input_dim=act_in, emb_dim=P)
    decoder = ConvDecoder(EMBED_DIM, img_ch=3, img_hw=IMG, grid=8)
    return FONDJEPA(decoder=decoder, predictor=predictor, action_encoder=action_encoder,
                    latent_dim=EMBED_DIM, head=head, k_inner=K, tau=0.2, infer_lr=infer_lr,
                    infer_grad_clip=INFER_GRAD_CLIP, infer_momentum=INFER_MOMENTUM,
                    infer_init="predictive_prior", img_ch=3, img_hw=IMG).to(DEVICE)


def cfg_obj(beta, infer_objective, log_diag):
    d = {"beta": beta, "pred_loss": "exact_kl",
         "target_scheme": "online_filtering", "infer_objective": infer_objective,
         "log_diag": log_diag}
    loss_obj = types.SimpleNamespace(**d)
    loss_obj.get = lambda k, dd=None: d.get(k, dd)
    return types.SimpleNamespace(history_size=HISTORY, num_preds=NUM_PREDS, loss=loss_obj)


def to_dev(batch):
    return {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in batch.items()}


def run_config(family, obj, K, beta, infer_lr, loader, act_in, eval_batch):
    torch.manual_seed(0)
    model = build_model(family, K, infer_lr, act_in)
    wrap = types.SimpleNamespace(model=model, log_dict=lambda *a, **k: None)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)
    fcfg = cfg_obj(beta, obj, log_diag=False)
    data = itertools.cycle(loader)
    model.train()
    nonfinite = False
    for _ in range(STEPS):
        out = vijepa_forward(wrap, to_dev(next(data)), "train", fcfg)
        if not torch.isfinite(out["loss"]):
            nonfinite = True
            break
        opt.zero_grad(); out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

    model.eval()
    with torch.no_grad():
        eb = to_dev(eval_batch)
        out = vijepa_forward(wrap, eb, "val", cfg_obj(beta, obj, log_diag=True))
        emb = out["emb"]; rep = collapse_report(emb); stats = model.head.param_stats(emb)
        ap = model.action_prior_report(eb, emb, HISTORY, beta)
    g = lambda k: float(out[k].item()) if k in out and torch.is_tensor(out[k]) else float("nan")
    rd = lambda x, n=4: round(float(x), n)
    rate_max = stats.get("rate_max", float("nan"))
    hi_sat, lo_sat = stats.get("hi_sat_frac", 0.0), stats.get("lo_sat_frac", 0.0)
    saturated = ((family == "poisson" and rate_max >= SAT_RATE - 1.0)
                 or hi_sat > 0.02 or lo_sat > 0.02)
    corr = ap["correction_norm"]
    # divergence guard: mu/log-rate can explode without hitting the clamp the
    # SATURATED flag checks (e.g. Gaussian mu is unbounded). Treat as failure.
    diverged = (not (corr == corr)) or corr > 1e3 or abs(ap["R_post"]) > 1e6
    saturated = saturated or diverged
    clean = (rep["passed"] and not saturated and not nonfinite
             and ap["R_post"] < ap["R_prior"]                 # observation correction helps
             and ap["F_post"] < ap["F_prior"]                 # free energy improves
             and 1e-3 < corr < 20.0                           # nonzero but not exploding
             and ap["action_gain_R"] > 0                      # true action beats shuffled
             and ap["action_gain_vs_noop"] > 0)               # true action beats no-op
    return {
        "scheme": "online", "variant": variant_name(family, "exact_kl"),
        "infer_obj": obj, "K": K, "beta": beta,
        "infer_lr": infer_lr, "infer_grad_clip": INFER_GRAD_CLIP,
        "infer_momentum": INFER_MOMENTUM, "detach_metric": 1,
        "nonfinite": int(nonfinite),
        "rank_frac": round(rep["eff_rank_frac"], 3),
        "var_med": f"{rep['batch_var_median']:.2e}", "temp_var": f"{rep['temporal_var_mean']:.2e}",
        "collapse_pass": int(rep["passed"]),
        "corr_norm": rd(corr), "innov_kl": rd(ap["innovation_kl"], 3),
        "R_prior": rd(ap["R_prior"], 4), "R_post": rd(ap["R_post"], 4),
        "recon_gain": rd(ap["R_prior"] - ap["R_post"], 5),
        "F_prior": rd(ap["F_prior"], 4), "F_post": rd(ap["F_post"], 4),
        "F_gain": rd(ap["F_prior"] - ap["F_post"], 5),
        "R_pr_true": rd(ap["R_prior_true"], 4), "R_pr_shuf": rd(ap["R_prior_shuffle"], 4),
        "R_pr_noop": rd(ap["R_prior_noop"], 4),
        "act_gain_R": rd(ap["action_gain_R"], 5), "act_gain_F": rd(ap["action_gain_F"], 5),
        "act_gain_noop": rd(ap["action_gain_vs_noop"], 5),
        "D_pred": round(g("D_pred_shift"), 3), "D_noop": round(g("pred_noop"), 3),
        "noop_ratio": round(g("noop_ratio"), 3),
        "kl_exact": round(g("kl_exact"), 3), "fisher_q": round(g("fisher_quad"), 3),
        "exact/quad": round(g("exact_quad_ratio"), 3),
        "rate_mean": round(stats.get("rate_mean", float("nan")), 3),
        "rate_p95": round(stats.get("rate_p95", float("nan")), 2),
        "rate_p99": round(stats.get("rate_p99", float("nan")), 2),
        "rate_max": round(rate_max, 2),
        "hi_sat": round(hi_sat, 3), "lo_sat": round(lo_sat, 3),
        "SATURATED": int(saturated), "CLEAN": int(clean),
    }


def main():
    print(f"device={DEVICE} steps={STEPS} quick={QUICK} batch={BATCH} N_tiny={N_TINY}")
    loader, act_in = tiny_loader()
    eval_batch = next(iter(loader))
    combos = []
    for family, K, infer_lr in itertools.product(FAMILIES, K_GRID, LR_GRID):
        for beta in BETA_GRID:
            combos.append((family, "free_energy", K, beta, infer_lr))
    print(f"{len(combos)} configs")
    rows = []
    for i, (family, obj, K, beta, infer_lr) in enumerate(combos):
        try:
            row = run_config(family, obj, K, beta, infer_lr, loader, act_in, eval_batch)
        except Exception as e:
            row = {"scheme": "online", "variant": variant_name(family, "exact_kl"),
                   "infer_obj": obj, "K": K, "beta": beta,
                   "infer_lr": infer_lr, "infer_grad_clip": INFER_GRAD_CLIP,
                   "infer_momentum": INFER_MOMENTUM, "detach_metric": 1,
                   "nonfinite": 1, "error": str(e)[:80]}
        rows.append(row)
        flag = (" *** SATURATED ***" if row.get("SATURATED") else "") + (" <CLEAN>" if row.get("CLEAN") else "")
        print(f"[{i+1}/{len(combos)}] {row}{flag}")

    cols = ["scheme", "variant", "infer_obj", "K", "beta", "nonfinite",
            "rank_frac", "var_med", "temp_var", "collapse_pass", "corr_norm", "innov_kl",
            "R_prior", "R_post", "recon_gain", "F_prior", "F_post", "F_gain",
            "R_pr_true", "R_pr_shuf", "R_pr_noop", "act_gain_R", "act_gain_F", "act_gain_noop",
            "D_pred", "D_noop", "noop_ratio", "kl_exact", "fisher_q", "exact/quad",
            "rate_mean", "rate_p95", "rate_p99", "rate_max", "hi_sat", "lo_sat",
            "SATURATED", "CLEAN"]
    with open("fond_sweep_online.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols + ["error"]); w.writeheader()
        for r in rows:
            w.writerow(r)
    print("\nwrote fond_sweep_online.csv")


if __name__ == "__main__":
    main()
