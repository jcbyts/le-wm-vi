from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from einops import rearrange
from torch import nn
import torch.nn.functional as F

from latent import make_head
from model import ConvDecoder
from module import ARPredictor, Embedder, SIGReg, exponential_sigreg


class ConvEncoder(nn.Module):
    def __init__(self, img_ch: int, param_dim: int, width: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(img_ch, width, 5, stride=2, padding=2),
            nn.GroupNorm(8, width),
            nn.SiLU(),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1),
            nn.GroupNorm(8, width * 2),
            nn.SiLU(),
            nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1),
            nn.GroupNorm(8, width * 4),
            nn.SiLU(),
            nn.Conv2d(width * 4, width * 4, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width * 4, param_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


@dataclass
class AmortizedMarmoConfig:
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
    encoder_width: int = 64
    decoder_grid: int = 8
    tau: float = 0.2
    beta: float = 1.0
    recon_weight: float = 1.0
    sigreg_weight: float = 0.09
    sigreg_target_rate: float = 1.0
    sigreg_knots: int = 17
    sigreg_num_proj: int = 1024
    poisson_log_lo: float = -12.0
    poisson_log_hi: float = 5.0
    fixed_unit_variance: bool = False

    @property
    def param_dim(self) -> int:
        return 2 * self.embed_dim if self.family == "gaussian" else self.embed_dim


class AmortizedWorldModel(nn.Module):
    def __init__(self, cfg: AmortizedMarmoConfig):
        super().__init__()
        self.cfg = cfg
        self.head = make_head(
            cfg.family,
            tau=cfg.tau,
            fixed_unit_variance=cfg.fixed_unit_variance,
            poisson_log_lo=cfg.poisson_log_lo,
            poisson_log_hi=cfg.poisson_log_hi,
        )
        self.encoder = ConvEncoder(cfg.img_ch, cfg.param_dim, width=cfg.encoder_width)
        self.decoder = ConvDecoder(cfg.embed_dim, img_ch=cfg.img_ch, img_hw=cfg.img_hw, grid=cfg.decoder_grid)
        self.predictor = ARPredictor(
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
        self.action_encoder = Embedder(
            input_dim=cfg.action_dim,
            smoothed_dim=max(8, cfg.action_dim),
            emb_dim=cfg.param_dim,
        )
        self.sigreg = SIGReg(knots=cfg.sigreg_knots, num_proj=cfg.sigreg_num_proj)

    def encode(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        pixels = batch["pixels"].float()
        b = pixels.size(0)
        flat = rearrange(pixels, "b t c h w -> (b t) c h w")
        eta = self.head.clamp_param(self.encoder(flat))
        batch = dict(batch)
        batch["emb"] = rearrange(eta, "(b t) d -> b t d", b=b)
        batch["act_emb"] = self.action_encoder(torch.nan_to_num(batch["action"], 0.0))
        return batch

    def predict(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        return self.predictor(emb, act_emb)

    def decode(self, emb: torch.Tensor) -> torch.Tensor:
        b = emb.size(0)
        flat = rearrange(emb, "b t d -> (b t) d")
        code = self.head.sample(flat)
        recon = self.decoder(code)
        return rearrange(recon, "(b t) c h w -> b t c h w", b=b)

    def deterministic_code(self, emb: torch.Tensor) -> torch.Tensor:
        flat = rearrange(emb, "b t d -> (b t) d")
        code = self.head.to_code(flat)
        return rearrange(code, "(b t) d -> b t d", b=emb.size(0))


def build_amortized_model(cfg: AmortizedMarmoConfig) -> AmortizedWorldModel:
    return AmortizedWorldModel(cfg)


def sigreg_loss(model: AmortizedWorldModel, eta: torch.Tensor, cfg: AmortizedMarmoConfig) -> torch.Tensor:
    if cfg.sigreg_weight <= 0:
        return eta.new_zeros(())
    if cfg.family == "poisson":
        return exponential_sigreg(eta.reshape(-1, eta.shape[-1]), target_rate=cfg.sigreg_target_rate)
    code = model.deterministic_code(eta).transpose(0, 1)
    return model.sigreg(code)


def compute_amortized_loss(model: AmortizedWorldModel, batch: dict[str, torch.Tensor], cfg: AmortizedMarmoConfig):
    batch = model.encode(batch)
    eta = batch["emb"]
    act_emb = batch["act_emb"]
    ctx = eta[:, : cfg.history_size]
    ctx_act = act_emb[:, : cfg.history_size]
    pred = model.predict(ctx, ctx_act)
    target = eta[:, 1 : cfg.history_size + 1].detach()
    pred_loss = model.head.pred_term(target, pred, cfg.pred_loss, detach_metric=True)
    recon = model.decode(eta)
    recon_loss = F.mse_loss(recon, batch["pixels"].float())
    reg_loss = sigreg_loss(model, eta, cfg)
    loss = cfg.recon_weight * recon_loss + cfg.beta * pred_loss + cfg.sigreg_weight * reg_loss
    with torch.no_grad():
        code = model.deterministic_code(eta)
        out = {
            "loss": loss.detach(),
            "pred_loss": pred_loss.detach(),
            "recon_loss": recon_loss.detach(),
            "sigreg_loss": reg_loss.detach(),
            "kl_exact": model.head.kl_exact(target.detach(), pred.detach()).detach(),
            "fisher_quad": model.head.fisher_quad(target.detach(), pred.detach()).detach(),
            "exact_quad_ratio": model.head.kl_exact(target.detach(), pred.detach()).detach()
            / model.head.fisher_quad(target.detach(), pred.detach()).clamp_min(1e-8).detach(),
            "eta_std": eta.detach().std(),
            "code_std": code.detach().std(),
        }
    out["loss_for_backward"] = loss
    out["eta"] = eta
    out["pred_hat"] = pred
    out["code"] = model.deterministic_code(eta)
    out["recon"] = recon
    return out


def checkpoint_payload(model: AmortizedWorldModel, cfg: AmortizedMarmoConfig, extra: dict | None = None) -> dict:
    payload = {
        "model_kind": "amortized",
        "model_state": model.state_dict(),
        "model_config": asdict(cfg),
    }
    if extra:
        payload["extra"] = extra
    return payload


def load_amortized_from_checkpoint(path: str, map_location="cpu") -> tuple[AmortizedWorldModel, AmortizedMarmoConfig, dict]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    cfg = AmortizedMarmoConfig(**ckpt["model_config"])
    model = build_amortized_model(cfg)
    model.load_state_dict(ckpt["model_state"])
    return model, cfg, ckpt.get("extra", {})
