from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from model import ConvDecoder, FONDJEPA
from module import ARPredictor, Embedder, SIGReg, exponential_sigreg
from latent import make_head


@dataclass
class FondMarmoConfig:
    family: str = "poisson"
    pred_loss: str = "exact_kl"
    embed_dim: int = 64
    img_ch: int = 3
    img_hw: int = 64
    action_dim: int = 2
    history_size: int = 3
    predictor_depth: int = 2
    predictor_heads: int = 4
    predictor_mlp_dim: int = 256
    decoder_grid: int = 8
    k_inner: int = 2
    tau: float = 0.2
    infer_lr: float = 0.1
    infer_grad_clip: float | None = 1.0
    infer_momentum: float = 0.0
    infer_backprop: bool = False
    beta: float = 1.0
    infer_objective: str = "free_energy"
    recon_weight: float = 1.0
    sigreg_weight: float = 0.0
    sigreg_target_rate: float = 1.0
    sigreg_knots: int = 17
    sigreg_num_proj: int = 1024
    poisson_log_lo: float = -12.0
    poisson_log_hi: float = 5.0

    @property
    def param_dim(self) -> int:
        if self.family == "gaussian":
            return 2 * self.embed_dim
        return self.embed_dim


def build_fond_model(cfg: FondMarmoConfig) -> FONDJEPA:
    decoder = ConvDecoder(
        latent_dim=cfg.embed_dim,
        img_ch=cfg.img_ch,
        img_hw=cfg.img_hw,
        grid=cfg.decoder_grid,
    )
    predictor = ARPredictor(
        num_frames=cfg.history_size,
        depth=cfg.predictor_depth,
        heads=cfg.predictor_heads,
        mlp_dim=cfg.predictor_mlp_dim,
        input_dim=cfg.param_dim,
        hidden_dim=cfg.embed_dim,
        output_dim=cfg.param_dim,
        dim_head=max(16, cfg.embed_dim // cfg.predictor_heads),
        dropout=0.0,
        emb_dropout=0.0,
    )
    action_encoder = Embedder(
        input_dim=cfg.action_dim,
        smoothed_dim=max(8, cfg.action_dim),
        emb_dim=cfg.param_dim,
    )
    head = make_head(
        cfg.family,
        tau=cfg.tau,
        poisson_log_lo=cfg.poisson_log_lo,
        poisson_log_hi=cfg.poisson_log_hi,
    )
    model = FONDJEPA(
        decoder=decoder,
        predictor=predictor,
        action_encoder=action_encoder,
        latent_dim=cfg.embed_dim,
        head=head,
        k_inner=cfg.k_inner,
        tau=cfg.tau,
        infer_lr=cfg.infer_lr,
        infer_grad_clip=cfg.infer_grad_clip,
        infer_momentum=cfg.infer_momentum,
        infer_backprop=cfg.infer_backprop,
        img_ch=cfg.img_ch,
        img_hw=cfg.img_hw,
    )
    if cfg.sigreg_weight > 0 and cfg.family == "gaussian":
        model.sigreg = SIGReg(knots=cfg.sigreg_knots, num_proj=cfg.sigreg_num_proj)
    return model


def compute_sigreg_loss(model: FONDJEPA, eta: torch.Tensor, cfg: FondMarmoConfig) -> torch.Tensor:
    if cfg.sigreg_weight <= 0:
        return eta.new_zeros(())
    if cfg.family == "poisson":
        return exponential_sigreg(
            eta.reshape(-1, eta.shape[-1]),
            target_rate=cfg.sigreg_target_rate,
        )
    if cfg.family == "gaussian":
        code = model.head.to_code(eta).transpose(0, 1)
        if not hasattr(model, "sigreg"):
            model.sigreg = SIGReg(knots=cfg.sigreg_knots, num_proj=cfg.sigreg_num_proj).to(eta.device)
        return model.sigreg(code)
    return eta.new_zeros(())


def compute_fond_loss(model: FONDJEPA, batch: dict[str, torch.Tensor], cfg: FondMarmoConfig):
    batch = dict(batch)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    model._set_runtime_inference(infer_backprop=cfg.infer_backprop, k_bptt=cfg.k_inner)
    info = model.filter_sequence(
        batch,
        history_size=cfg.history_size,
        beta=cfg.beta,
        infer_objective=cfg.infer_objective,
        return_diag=True,
    )
    eta = info["emb"]
    ehat = info["pred_hat"]
    head = model.head
    pred_loss = head.pred_term(eta.detach(), ehat, cfg.pred_loss, detach_metric=True)
    recon = model.decode(eta)
    recon_loss = F.mse_loss(recon, batch["pixels"].float())
    sigreg_loss = compute_sigreg_loss(model, eta, cfg)
    loss = cfg.recon_weight * recon_loss + cfg.beta * pred_loss + cfg.sigreg_weight * sigreg_loss
    with torch.no_grad():
        out = {
            "loss": loss.detach(),
            "pred_loss": pred_loss.detach(),
            "recon_loss": recon_loss.detach(),
            "sigreg_loss": sigreg_loss.detach(),
            "kl_exact": head.kl_exact(eta.detach(), ehat.detach()).detach(),
            "fisher_quad": head.fisher_quad(eta.detach(), ehat.detach()).detach(),
            "correction_norm": (eta.detach() - ehat.detach()).norm(dim=-1).mean(),
            "eta_std": eta.detach().std(),
        }
        out["exact_quad_ratio"] = out["kl_exact"] / (out["fisher_quad"].clamp_min(1e-8))
        if "recon_init" in info and "recon_final" in info:
            out["recon_gain"] = (info["recon_init"] - info["recon_final"]).detach()
        if "F_init" in info and "F_final" in info:
            out["F_gain"] = (info["F_init"] - info["F_final"]).detach()
        if "infer_kl_final" in info:
            out["infer_kl_final"] = info["infer_kl_final"].detach()
    out["loss_for_backward"] = loss
    out["recon"] = recon
    out["eta"] = eta
    out["pred_hat"] = ehat
    return out


def checkpoint_payload(model: FONDJEPA, cfg: FondMarmoConfig, extra: dict | None = None) -> dict:
    payload = {
        "model_state": model.state_dict(),
        "model_config": asdict(cfg),
    }
    if extra:
        payload["extra"] = extra
    return payload


def load_model_from_checkpoint(path: str, map_location="cpu") -> tuple[FONDJEPA, FondMarmoConfig, dict]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    cfg = FondMarmoConfig(**ckpt["model_config"])
    model = build_fond_model(cfg)
    model.load_state_dict(ckpt["model_state"])
    return model, cfg, ckpt.get("extra", {})
