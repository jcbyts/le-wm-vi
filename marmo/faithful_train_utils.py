from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import torch
from einops import rearrange
from torch import nn
import torch.nn.functional as F

from diagnostics import collapse_report
from jepa import PoisWM
from marmo.neuro_encoders import AlexNetV1Encoder, VOneAlexNetEncoder, VOneBlockEncoder
from module import ARPredictor, Embedder, MLP, SIGReg, exponential_sigreg


class ConvEncoder(nn.Module):
    """Small BackImage encoder with the same output contract as Jake's JEPA encoder."""

    def __init__(self, img_ch: int, embed_dim: int, width: int = 64):
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
            nn.GroupNorm(8, width * 4),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width * 4, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


@dataclass
class FaithfulMarmoConfig:
    family: str = "gaussian"
    embed_dim: int = 192
    img_ch: int = 3
    img_hw: int = 64
    action_dim: int = 2
    action_scale: float = 1.0
    history_size: int = 3
    num_preds: int = 1
    encoder_width: int = 64
    encoder_kind: str = "conv"
    neural_pretrained: bool = True
    neural_freeze_frontend: bool = True
    neural_resize_hw: int = 224
    neural_feature_index: int = 2
    neural_pool_hw: int = 1
    neural_pixel_mode: str = "visioncore"
    vone_simple_channels: int = 128
    vone_complex_channels: int = 128
    vone_ksize: int = 25
    vone_stride: int = 4
    vone_visual_degrees: float = 8.0
    vone_sf_corr: float = 0.75
    vone_sf_max: float = 9.0
    vone_sf_min: float = 0.0
    vone_noise_mode: str = "none"
    predictor_depth: int = 6
    predictor_heads: int = 8
    predictor_mlp_dim: int = 1024
    predictor_dim_head: int = 64
    dropout: float = 0.1
    emb_dropout: float = 0.0
    sigreg_weight: float = 0.09
    sigreg_knots: int = 17
    sigreg_num_proj: int = 1024
    poisson_target_rate: float = 1.0
    poisson_log_lo: float = -20.0
    poisson_log_hi: float = 5.0
    use_projectors: bool = True
    projector_hidden_dim: int = 2048
    projector_norm: str = "batchnorm"
    detach_target: Optional[bool] = None
    target_mode: str = "full"
    target_level: int = 0
    masked_channel_value: float = 0.5
    foveal_dim: int = 0
    context_dim: int = 0
    foveal_level: int = 0
    action_smoothed_dim: int = 8
    action_mlp_scale: int = 4


class FaithfulBackImageLeWM(nn.Module):
    """BackImage LeWM path with Jake's latent losses and no image decoder."""

    def __init__(self, cfg: FaithfulMarmoConfig):
        super().__init__()
        if cfg.family not in {"gaussian", "poisson"}:
            raise ValueError("family must be 'gaussian' or 'poisson'")
        if cfg.target_mode not in {"full", "l0"}:
            raise ValueError("target_mode must be 'full' or 'l0'")
        if cfg.encoder_kind not in {"conv", "alexnet_v1", "voneblock", "vonealexnet"}:
            raise ValueError("encoder_kind must be 'conv', 'alexnet_v1', 'voneblock', or 'vonealexnet'")
        self.cfg = cfg
        self.split_latent = int(cfg.foveal_dim) > 0 or int(cfg.context_dim) > 0

        def make_encoder(img_ch: int, embed_dim: int) -> nn.Module:
            if cfg.encoder_kind == "conv":
                return ConvEncoder(img_ch, embed_dim, width=cfg.encoder_width)
            if cfg.encoder_kind == "alexnet_v1":
                return AlexNetV1Encoder(
                    img_ch,
                    embed_dim,
                    input_hw=cfg.img_hw,
                    pretrained=bool(cfg.neural_pretrained),
                    freeze_frontend=bool(cfg.neural_freeze_frontend),
                    resize_hw=int(cfg.neural_resize_hw),
                    feature_index=int(cfg.neural_feature_index),
                    pool_hw=int(cfg.neural_pool_hw),
                    pixel_mode=str(cfg.neural_pixel_mode),
                )
            if cfg.encoder_kind == "voneblock":
                return VOneBlockEncoder(
                    img_ch,
                    embed_dim,
                    input_hw=cfg.img_hw,
                    freeze_frontend=bool(cfg.neural_freeze_frontend),
                    resize_hw=int(cfg.neural_resize_hw),
                    pool_hw=int(cfg.neural_pool_hw),
                    pixel_mode=str(cfg.neural_pixel_mode),
                    simple_channels=int(cfg.vone_simple_channels),
                    complex_channels=int(cfg.vone_complex_channels),
                    ksize=int(cfg.vone_ksize),
                    stride=int(cfg.vone_stride),
                    visual_degrees=float(cfg.vone_visual_degrees),
                    sf_corr=float(cfg.vone_sf_corr),
                    sf_max=float(cfg.vone_sf_max),
                    sf_min=float(cfg.vone_sf_min),
                    noise_mode=str(cfg.vone_noise_mode),
                )
            return VOneAlexNetEncoder(
                img_ch,
                embed_dim,
                input_hw=cfg.img_hw,
                pretrained=bool(cfg.neural_pretrained),
                freeze_frontend=bool(cfg.neural_freeze_frontend),
                resize_hw=int(cfg.neural_resize_hw),
                feature_index=int(cfg.neural_feature_index),
                pool_hw=int(cfg.neural_pool_hw),
                pixel_mode=str(cfg.neural_pixel_mode),
                simple_channels=int(cfg.vone_simple_channels),
                complex_channels=int(cfg.vone_complex_channels),
                ksize=int(cfg.vone_ksize),
                stride=int(cfg.vone_stride),
                visual_degrees=float(cfg.vone_visual_degrees),
                sf_corr=float(cfg.vone_sf_corr),
                sf_max=float(cfg.vone_sf_max),
                sf_min=float(cfg.vone_sf_min),
                noise_mode=str(cfg.vone_noise_mode),
            )

        if self.split_latent:
            if int(cfg.foveal_dim) <= 0 or int(cfg.context_dim) <= 0:
                raise ValueError("split latent mode requires positive foveal_dim and context_dim")
            if int(cfg.foveal_dim) + int(cfg.context_dim) != int(cfg.embed_dim):
                raise ValueError("foveal_dim + context_dim must equal embed_dim")
            if int(cfg.img_ch) < 2:
                raise ValueError("split latent mode requires at least two image channels")
            if int(cfg.foveal_level) != 0:
                raise ValueError("Only foveal_level=0 is currently supported")
            self.encoder = None
            self.foveal_encoder = make_encoder(1, cfg.foveal_dim)
            self.context_encoder = ConvEncoder(cfg.img_ch - 1, cfg.context_dim, width=cfg.encoder_width)
        else:
            self.encoder = make_encoder(cfg.img_ch, cfg.embed_dim)
        if cfg.use_projectors:
            if cfg.projector_norm == "batchnorm":
                norm_fn = nn.BatchNorm1d
            elif cfg.projector_norm == "layernorm":
                norm_fn = nn.LayerNorm
            elif cfg.projector_norm == "none":
                norm_fn = None
            else:
                raise ValueError("projector_norm must be batchnorm, layernorm, or none")
            if self.split_latent:
                self.foveal_projector = MLP(
                    input_dim=cfg.foveal_dim,
                    hidden_dim=cfg.projector_hidden_dim,
                    output_dim=cfg.foveal_dim,
                    norm_fn=norm_fn,
                )
                self.context_projector = MLP(
                    input_dim=cfg.context_dim,
                    hidden_dim=cfg.projector_hidden_dim,
                    output_dim=cfg.context_dim,
                    norm_fn=norm_fn,
                )
                self.foveal_pred_proj = MLP(
                    input_dim=cfg.foveal_dim,
                    hidden_dim=cfg.projector_hidden_dim,
                    output_dim=cfg.foveal_dim,
                    norm_fn=norm_fn,
                )
                self.context_pred_proj = MLP(
                    input_dim=cfg.context_dim,
                    hidden_dim=cfg.projector_hidden_dim,
                    output_dim=cfg.context_dim,
                    norm_fn=norm_fn,
                )
                self.projector = None
                self.pred_proj = None
            else:
                self.projector = MLP(
                    input_dim=cfg.embed_dim,
                    hidden_dim=cfg.projector_hidden_dim,
                    output_dim=cfg.embed_dim,
                    norm_fn=norm_fn,
                )
                self.pred_proj = MLP(
                    input_dim=cfg.embed_dim,
                    hidden_dim=cfg.projector_hidden_dim,
                    output_dim=cfg.embed_dim,
                    norm_fn=norm_fn,
                )
        else:
            if self.split_latent:
                self.foveal_projector = nn.Identity()
                self.context_projector = nn.Identity()
                self.foveal_pred_proj = nn.Identity()
                self.context_pred_proj = nn.Identity()
                self.projector = None
                self.pred_proj = None
            else:
                self.projector = nn.Identity()
                self.pred_proj = nn.Identity()
        self.predictor = ARPredictor(
            num_frames=cfg.history_size,
            depth=cfg.predictor_depth,
            heads=cfg.predictor_heads,
            mlp_dim=cfg.predictor_mlp_dim,
            input_dim=cfg.embed_dim,
            hidden_dim=cfg.embed_dim,
            output_dim=cfg.embed_dim,
            dim_head=cfg.predictor_dim_head,
            dropout=cfg.dropout,
            emb_dropout=cfg.emb_dropout,
        )
        self.action_encoder = Embedder(
            input_dim=cfg.action_dim,
            smoothed_dim=max(1, int(cfg.action_smoothed_dim)),
            emb_dim=cfg.embed_dim,
            mlp_scale=max(1, int(cfg.action_mlp_scale)),
        )
        self.sigreg = SIGReg(knots=cfg.sigreg_knots, num_proj=cfg.sigreg_num_proj)
        self.target_rate = float(cfg.poisson_target_rate)

    @property
    def foveal_slice(self) -> slice:
        return slice(0, int(self.cfg.foveal_dim))

    def encode_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        pixels = pixels.float()
        b = pixels.size(0)
        if self.split_latent:
            foveal = pixels[:, :, :1]
            context = pixels[:, :, 1:]
            flat_fov = rearrange(foveal, "b t c h w -> (b t) c h w")
            flat_ctx = rearrange(context, "b t c h w -> (b t) c h w")
            fov = self.foveal_projector(self.foveal_encoder(flat_fov))
            ctx = self.context_projector(self.context_encoder(flat_ctx))
            emb = torch.cat([fov, ctx], dim=-1)
        else:
            flat = rearrange(pixels, "b t c h w -> (b t) c h w")
            emb = self.projector(self.encoder(flat))
        return rearrange(emb, "(b t) d -> b t d", b=b)

    def encode(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        pixels = batch["pixels"].float()
        emb = self.encode_pixels(pixels)
        out = dict(batch)
        out["emb"] = emb
        action = torch.nan_to_num(batch["action"].float(), 0.0)
        if float(self.cfg.action_scale) != 1.0:
            action = action / float(self.cfg.action_scale)
        out["act_emb"] = self.action_encoder(action)
        return out

    def predict(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        pred = self.predictor(emb, act_emb)
        b = emb.size(0)
        if self.split_latent:
            fov = pred[..., : self.cfg.foveal_dim]
            ctx = pred[..., self.cfg.foveal_dim :]
            flat_fov = rearrange(fov, "b t d -> (b t) d")
            flat_ctx = rearrange(ctx, "b t d -> (b t) d")
            fov = self.foveal_pred_proj(flat_fov)
            ctx = self.context_pred_proj(flat_ctx)
            flat = torch.cat([fov, ctx], dim=-1)
        else:
            flat = rearrange(pred, "b t d -> (b t) d")
            flat = self.pred_proj(flat)
        return rearrange(flat, "(b t) d -> b t d", b=b)

    def target_from_batch(self, batch: dict[str, torch.Tensor], emb: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        if cfg.target_mode == "full":
            return emb[:, cfg.num_preds : cfg.num_preds + cfg.history_size]
        if cfg.target_mode == "l0":
            pixels = batch["pixels"][:, cfg.num_preds : cfg.num_preds + cfg.history_size].clone()
            target_level = int(cfg.target_level)
            if target_level < 0 or target_level >= pixels.shape[2]:
                raise ValueError(f"target_level={target_level} is outside img_ch={pixels.shape[2]}")
            mask = torch.ones(pixels.shape[2], dtype=torch.bool, device=pixels.device)
            mask[target_level] = False
            pixels[:, :, mask] = float(cfg.masked_channel_value)
            return self.encode_pixels(pixels)
        raise AssertionError(f"Unhandled target_mode {cfg.target_mode}")

    def loss_views(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.target_mode == "l0" and self.split_latent:
            return pred[..., self.foveal_slice], target[..., self.foveal_slice]
        return pred, target

    def code_from_emb(self, emb: torch.Tensor) -> torch.Tensor:
        if self.cfg.family == "poisson":
            return torch.exp(emb.clamp(self.cfg.poisson_log_lo, self.cfg.poisson_log_hi))
        return emb

    def poisson_kl(self, target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        return poisson_kl(
            target,
            pred,
            log_lo=self.cfg.poisson_log_lo,
            log_hi=self.cfg.poisson_log_hi,
        )


def poisson_kl(target: torch.Tensor, pred: torch.Tensor, log_lo: float = -20.0, log_hi: float = 5.0) -> torch.Tensor:
    target_c = target.clamp(log_lo, log_hi)
    pred_c = pred.clamp(log_lo, log_hi)
    lam_target = torch.exp(target_c)
    lam_pred = torch.exp(pred_c)
    kl = lam_target * (target_c - pred_c) - lam_target + lam_pred
    return kl.sum(dim=-1).mean()


def prediction_loss(
    model: FaithfulBackImageLeWM,
    target: torch.Tensor,
    pred: torch.Tensor,
    family: str | None = None,
) -> torch.Tensor:
    family = family or model.cfg.family
    if family == "poisson":
        return model.poisson_kl(target, pred)
    return (pred - target).pow(2).mean()


def prediction_loss_per_example(
    model: FaithfulBackImageLeWM,
    target: torch.Tensor,
    pred: torch.Tensor,
    family: str | None = None,
) -> torch.Tensor:
    family = family or model.cfg.family
    if family == "poisson":
        target_c = target.clamp(model.cfg.poisson_log_lo, model.cfg.poisson_log_hi)
        pred_c = pred.clamp(model.cfg.poisson_log_lo, model.cfg.poisson_log_hi)
        lam_target = torch.exp(target_c)
        lam_pred = torch.exp(pred_c)
        return (lam_target * (target_c - pred_c) - lam_target + lam_pred).sum(dim=-1)
    return (pred - target).pow(2).mean(dim=-1)


def faithful_forward(model: FaithfulBackImageLeWM, batch: dict[str, torch.Tensor]):
    cfg = model.cfg
    batch = model.encode(batch)
    emb = batch["emb"]
    act_emb = batch["act_emb"]
    ctx = emb[:, : cfg.history_size]
    ctx_act = act_emb[:, : cfg.history_size]
    pred = model.predict(ctx, ctx_act)
    target = model.target_from_batch(batch, emb)
    pred_for_loss, target_for_loss = model.loss_views(pred, target)
    detach_target = cfg.detach_target if cfg.detach_target is not None else cfg.family == "poisson"
    pred_target = target_for_loss.detach() if detach_target else target_for_loss
    pred_loss = prediction_loss(model, pred_target, pred_for_loss)

    if cfg.family == "poisson":
        reg_loss = exponential_sigreg(
            ctx.reshape(-1, ctx.shape[-1]),
            target_rate=cfg.poisson_target_rate,
        )
    else:
        reg_loss = model.sigreg(emb.transpose(0, 1))
    loss = pred_loss + cfg.sigreg_weight * reg_loss

    with torch.no_grad():
        code = model.code_from_emb(emb)
        zero_act = torch.zeros_like(ctx_act)
        pred_no_action = model.predict(ctx, zero_act)
        pred_identity = ctx
        target_det = target_for_loss.detach()
        no_action_view, _ = model.loss_views(pred_no_action, target)
        identity_view, _ = model.loss_views(pred_identity, target)
        no_action_loss = prediction_loss(model, target_det, no_action_view)
        identity_loss = prediction_loss(model, target_det, identity_view)
        out = {
            "loss": loss.detach(),
            "pred_loss": pred_loss.detach(),
            "pred_loss_per_example": prediction_loss_per_example(
                model,
                target_det,
                pred_for_loss.detach(),
            ).detach(),
            "reg_loss": reg_loss.detach(),
            "no_action_loss": no_action_loss.detach(),
            "identity_loss": identity_loss.detach(),
            "action_gain": (no_action_loss - pred_loss.detach()).detach(),
            "identity_gain": (identity_loss - pred_loss.detach()).detach(),
            "emb_std": emb.detach().std(),
            "code_mean": code.detach().mean(),
            "code_std": code.detach().std(),
        }
        for key, value in collapse_report(code.detach()).items():
            out[f"code_collapse_{key}"] = torch.tensor(float(value), device=emb.device)
        for key, value in collapse_report(emb.detach()).items():
            out[f"emb_collapse_{key}"] = torch.tensor(float(value), device=emb.device)
        if cfg.family == "poisson":
            lograte = emb.detach().clamp(cfg.poisson_log_lo, cfg.poisson_log_hi)
            rate = lograte.exp()
            flat = rate.flatten().float()
            out.update(
                {
                    "lograte_mean": lograte.mean(),
                    "rate_mean": rate.mean(),
                    "rate_p95": torch.quantile(flat, 0.95),
                    "rate_p99": torch.quantile(flat, 0.99),
                    "rate_max": rate.max(),
                    "sat_frac": (
                        (lograte <= cfg.poisson_log_lo + 1e-3).float().mean()
                        + (lograte >= cfg.poisson_log_hi - 1e-3).float().mean()
                    ),
                }
            )
    out["loss_for_backward"] = loss
    out["emb"] = emb
    out["code"] = code
    out["pred_hat"] = pred
    out["target"] = target
    return out


def checkpoint_payload(model: FaithfulBackImageLeWM, cfg: FaithfulMarmoConfig, extra: dict | None = None) -> dict:
    payload = {
        "model_kind": "faithful_lewm",
        "model_state": model.state_dict(),
        "model_config": asdict(cfg),
    }
    if extra:
        payload["extra"] = extra
    return payload


def build_faithful_model(cfg: FaithfulMarmoConfig) -> FaithfulBackImageLeWM:
    return FaithfulBackImageLeWM(cfg)


def load_faithful_from_checkpoint(path: str, map_location="cpu") -> tuple[FaithfulBackImageLeWM, FaithfulMarmoConfig, dict]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model_config = dict(ckpt["model_config"])
    model_config.setdefault("projector_norm", "layernorm")
    model_config.setdefault("target_mode", "full")
    model_config.setdefault("target_level", 0)
    model_config.setdefault("masked_channel_value", 0.5)
    model_config.setdefault("foveal_dim", 0)
    model_config.setdefault("context_dim", 0)
    model_config.setdefault("foveal_level", 0)
    model_config.setdefault("encoder_kind", "conv")
    model_config.setdefault("neural_pretrained", True)
    model_config.setdefault("neural_freeze_frontend", True)
    model_config.setdefault("neural_resize_hw", 224)
    model_config.setdefault("neural_feature_index", 2)
    model_config.setdefault("neural_pool_hw", 1)
    model_config.setdefault("neural_pixel_mode", "visioncore")
    model_config.setdefault("vone_simple_channels", 128)
    model_config.setdefault("vone_complex_channels", 128)
    model_config.setdefault("vone_ksize", 25)
    model_config.setdefault("vone_stride", 4)
    model_config.setdefault("vone_visual_degrees", 8.0)
    model_config.setdefault("vone_sf_corr", 0.75)
    model_config.setdefault("vone_sf_max", 9.0)
    model_config.setdefault("vone_sf_min", 0.0)
    model_config.setdefault("vone_noise_mode", "none")
    model_config.setdefault("action_smoothed_dim", 8)
    model_config.setdefault("action_mlp_scale", 4)
    cfg = FaithfulMarmoConfig(**model_config)
    model = build_faithful_model(cfg)
    model.load_state_dict(ckpt["model_state"])
    return model, cfg, ckpt.get("extra", {})


def assert_matches_jake_losses():
    """Tiny import-time-free check that our Poisson KL matches jepa.PoisWM."""
    dummy = PoisWM(target_rate=1.0, encoder=nn.Identity(), predictor=nn.Identity(), action_encoder=nn.Identity())
    target = torch.randn(3, 4, 5)
    pred = torch.randn(3, 4, 5)
    ours = poisson_kl(target, pred, log_lo=-20.0, log_hi=5.0)
    theirs = dummy._exact_poisson_kl(target, pred)
    if not torch.allclose(ours, theirs):
        raise AssertionError("faithful Poisson KL diverged from jepa.PoisWM")
