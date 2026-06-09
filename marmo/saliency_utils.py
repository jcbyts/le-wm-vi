from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from marmo.faithful_train_utils import FaithfulBackImageLeWM, faithful_forward


SaliencyMode = Literal["pred_output", "pred_loss"]
SaliencyMethod = Literal["grad", "grad_x_input", "integrated_gradients"]
SourceReduce = Literal["current", "context_sum"]


@dataclass
class SaliencyResult:
    heatmaps: torch.Tensor
    signed_attr: torch.Tensor
    channel_pct: torch.Tensor
    pred: torch.Tensor
    target: torch.Tensor
    score: torch.Tensor
    source_index: int


def _with_pixels(batch: dict[str, torch.Tensor], pixels: torch.Tensor) -> dict[str, torch.Tensor]:
    out = dict(batch)
    out["pixels"] = pixels
    return out


def _score_from_batch(
    model: FaithfulBackImageLeWM,
    batch: dict[str, torch.Tensor],
    mode: SaliencyMode,
    pred_index: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    out = faithful_forward(model, batch)
    pred = out["pred_hat"][:, pred_index]
    target = out["target"][:, pred_index].detach()
    pred_view, target_view = model.loss_views(pred, target)
    if mode == "pred_output":
        score = 0.5 * pred_view.pow(2).sum(dim=-1)
    elif mode == "pred_loss":
        score = 0.5 * (pred_view - target_view).pow(2).sum(dim=-1)
    else:
        raise ValueError("mode must be 'pred_output' or 'pred_loss'")
    return score, out


def _baseline_like(model: FaithfulBackImageLeWM, pixels: torch.Tensor, baseline: str | torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(baseline):
        return baseline.to(device=pixels.device, dtype=pixels.dtype).expand_as(pixels)
    if baseline == "zero":
        return torch.zeros_like(pixels)
    if baseline == "gray":
        return torch.full_like(pixels, float(getattr(model.cfg, "masked_channel_value", 0.5)))
    if baseline == "channel_mean":
        mean = pixels.mean(dim=(0, 1, 3, 4), keepdim=True)
        return mean.expand_as(pixels)
    raise ValueError("baseline must be 'zero', 'gray', 'channel_mean', or a tensor")


def _source_index(history_size: int, pred_index: int) -> int:
    idx = int(pred_index)
    if idx < 0:
        idx = history_size + idx
    if idx < 0 or idx >= history_size:
        raise IndexError(f"pred_index {pred_index} is outside history_size={history_size}")
    return idx


def _reduce_source(attr: torch.Tensor, history_size: int, source_reduce: SourceReduce, source_index: int) -> torch.Tensor:
    if source_reduce == "current":
        return attr[:, source_index]
    if source_reduce == "context_sum":
        return attr[:, :history_size].sum(dim=1)
    raise ValueError("source_reduce must be 'current' or 'context_sum'")


def _finish_result(
    attr: torch.Tensor,
    out: dict[str, torch.Tensor],
    score: torch.Tensor,
    history_size: int,
    pred_index: int,
    source_reduce: SourceReduce,
) -> SaliencyResult:
    src_idx = _source_index(history_size, pred_index)
    signed = _reduce_source(attr, history_size, source_reduce, src_idx)
    heatmaps = signed.abs()
    mass = heatmaps.flatten(-2).sum(dim=-1)
    channel_pct = 100.0 * mass / mass.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return SaliencyResult(
        heatmaps=heatmaps.detach(),
        signed_attr=signed.detach(),
        channel_pct=channel_pct.detach(),
        pred=out["pred_hat"][:, pred_index].detach(),
        target=out["target"][:, pred_index].detach(),
        score=score.detach(),
        source_index=src_idx,
    )


def compute_backimage_saliency(
    model: FaithfulBackImageLeWM,
    batch: dict[str, torch.Tensor],
    *,
    mode: SaliencyMode = "pred_output",
    method: SaliencyMethod = "grad_x_input",
    pred_index: int = -1,
    baseline: str | torch.Tensor = "channel_mean",
    ig_steps: int = 16,
    source_reduce: SourceReduce = "current",
) -> SaliencyResult:
    """Attribute faithful LeWM one-step prediction to BackImage pyramid channels.

    ``batch["pixels"]`` must be ``(B, history_size + num_preds, L, H, W)``.
    The default attributes the final context frame to the final predicted step.
    """
    model.eval()
    history_size = int(model.cfg.history_size)
    pixels0 = batch["pixels"].detach().clone()

    if method in {"grad", "grad_x_input"}:
        pixels = pixels0.requires_grad_(True)
        score, out = _score_from_batch(model, _with_pixels(batch, pixels), mode, pred_index)
        grad = torch.autograd.grad(score.sum(), pixels, create_graph=False, retain_graph=False)[0]
        attr = grad if method == "grad" else grad * pixels
        return _finish_result(attr, out, score, history_size, pred_index, source_reduce)

    if method != "integrated_gradients":
        raise ValueError("method must be 'grad', 'grad_x_input', or 'integrated_gradients'")
    if ig_steps <= 0:
        raise ValueError("ig_steps must be positive for integrated gradients")

    base = _baseline_like(model, pixels0, baseline)
    total_grad = torch.zeros_like(pixels0)
    last_score = None
    last_out = None
    for i in range(1, int(ig_steps) + 1):
        alpha = float(i) / float(ig_steps)
        pixels = (base + alpha * (pixels0 - base)).detach().requires_grad_(True)
        score, out = _score_from_batch(model, _with_pixels(batch, pixels), mode, pred_index)
        grad = torch.autograd.grad(score.sum(), pixels, create_graph=False, retain_graph=False)[0]
        total_grad = total_grad + grad.detach()
        last_score = score
        last_out = out
    attr = (pixels0 - base) * (total_grad / float(ig_steps))
    return _finish_result(attr, last_out, last_score, history_size, pred_index, source_reduce)
