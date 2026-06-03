"""Training-time behavioral monitoring.

Logs two signals to wandb on a configurable epoch cadence, on top of the
standard losses:

1. **Open-loop latent-rollout error** — feed only the initial ``history_size``
   true embeddings, then autoregressively predict embeddings from the model's
   own predictions (using ground-truth actions) and measure MSE-vs-horizon
   against the true encoded embeddings. Shows multi-step dynamics quality with
   no decoder and no environment.
2. **Planning success + rollout videos** — run the same MPC planning eval as
   ``eval.py`` on a fixed set of episodes and log ``success_rate`` plus a few
   rendered rollout mp4s. This is the "is it working like the paper" view.

The callback is defensive: a failure in either signal is logged and swallowed
so it can never crash the training run.
"""

import logging
import tempfile

import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback

import planning

log = logging.getLogger(__name__)


@torch.no_grad()
def latent_rollout_error(model, batch, history_size, device):
    """Per-step open-loop latent prediction MSE.

    Returns a ``(B, T - history_size)`` tensor of per-step MSE, or ``None`` if
    the sequence is too short to roll out. Uses ground-truth actions and the
    model's own predicted embeddings as context (open loop in state).
    """
    info = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    info["action"] = torch.nan_to_num(info["action"], 0.0)

    out = model.encode(info)
    true_emb = out["emb"]  # (B, T, D)
    act_emb = out["act_emb"]  # (B, T, A_emb)

    _, T, _ = true_emb.shape
    HS = history_size
    if T <= HS:
        return None

    emb = true_emb[:, :HS].clone()  # open-loop buffer, seeded with true history
    errs = []
    for t in range(HS, T):
        emb_ctx = emb[:, -HS:]  # predicted/true embeddings at [t-HS, t-1]
        act_ctx = act_emb[:, t - HS : t]  # actions at [t-HS, t-1]
        pred = model.predict(emb_ctx, act_ctx)[:, -1:]  # predict emb at t -> (B,1,D)
        errs.append((pred[:, 0] - true_emb[:, t]).pow(2).mean(dim=-1))  # (B,)
        emb = torch.cat([emb, pred], dim=1)  # feed prediction back

    return torch.stack(errs, dim=1)  # (B, T-HS)


class BehaviorEvalCallback(Callback):
    """Periodically log latent-rollout error and planning success+videos."""

    def __init__(
        self,
        *,
        every_n_epochs,
        history_size,
        latent_loader,
        num_latent_batches,
        planning_dataset,
        planning_kwargs,
        num_videos=4,
        enabled=True,
    ):
        super().__init__()
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.history_size = history_size
        self.latent_loader = latent_loader
        self.num_latent_batches = num_latent_batches
        self.planning_dataset = planning_dataset
        self.planning_kwargs = planning_kwargs
        self.num_videos = num_videos
        self.enabled = enabled

        # Fix the eval episodes once so the success curve is comparable.
        self._episodes_idx = None
        self._start_steps = None
        if enabled and planning_dataset is not None:
            self._episodes_idx, self._start_steps = planning.sample_eval_points(
                planning_dataset,
                planning_kwargs["num_eval"],
                planning_kwargs["goal_offset_steps"],
                planning_kwargs["cem_kwargs"].get("seed", 42),
            )

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.enabled or trainer.sanity_checking:
            return
        if not trainer.is_global_zero:
            return
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return

        model = pl_module.model
        was_training = model.training
        wandb_run = self._get_wandb_run(trainer)
        payload = {}

        try:
            payload.update(self._latent_metrics(model, pl_module.device))
        except Exception as exc:  # monitoring must never crash training
            log.warning(f"[monitor] latent-rollout metric failed: {exc!r}")

        try:
            payload.update(self._planning_metrics(model, wandb_run))
        except Exception as exc:
            log.warning(f"[monitor] planning eval failed: {exc!r}")
        finally:
            # Restore the model to its pre-eval training state.
            model.train(was_training)
            model.requires_grad_(True)

        if payload and wandb_run is not None:
            wandb_run.log(payload, step=trainer.global_step)
        elif payload:
            log.info(f"[monitor] (no wandb) {payload}")

    # -- helpers ----------------------------------------------------------

    def _get_wandb_run(self, trainer):
        logger = getattr(trainer, "logger", None)
        exp = getattr(logger, "experiment", None)
        # WandbLogger.experiment is the wandb Run (has `.log`); guard anything else.
        if exp is not None and hasattr(exp, "log"):
            return exp
        return None

    def _latent_metrics(self, model, device):
        model.eval()
        per_step_sum = None
        n = 0
        for i, batch in enumerate(self.latent_loader):
            if i >= self.num_latent_batches:
                break
            mse = latent_rollout_error(model, batch, self.history_size, device)
            if mse is None:
                continue
            step_mean = mse.mean(dim=0)  # (T-HS,)
            per_step_sum = step_mean if per_step_sum is None else per_step_sum + step_mean
            n += 1
        if per_step_sum is None or n == 0:
            return {}
        per_step = (per_step_sum / n).cpu()
        out = {"monitor/latent_rollout_mse": float(per_step.mean())}
        for k, v in enumerate(per_step.tolist(), start=1):
            out[f"monitor/latent_rollout_mse_step_{k}"] = v
        return out

    def _planning_metrics(self, model, wandb_run):
        if self.planning_dataset is None:
            return {}
        with tempfile.TemporaryDirectory() as tmp:
            metrics, videos = planning.run_planning_eval(
                model,
                self.planning_dataset,
                video_dir=tmp,
                episodes_idx=self._episodes_idx,
                start_steps=self._start_steps,
                **self.planning_kwargs,
            )
            out = {"monitor/success_rate": float(metrics["success_rate"])}
            if wandb_run is not None and videos:
                import wandb

                out["monitor/rollout_videos"] = [
                    wandb.Video(p, format="mp4") for p in videos[: self.num_videos]
                ]
            return out


class FondVizCallback(Callback):
    """Per-epoch FOND reconstruction and decoded-rollout visualization."""

    def __init__(self, *, history_size, val_loader, num_frames=8, enabled=True):
        super().__init__()
        self.history_size = history_size
        self.val_loader = val_loader
        self.num_frames = int(num_frames)
        self.enabled = enabled
        self._batch = None

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.enabled or trainer.sanity_checking or not trainer.is_global_zero:
            return
        wandb_run = self._get_wandb_run(trainer)
        if wandb_run is None:
            return
        model = pl_module.model
        if not hasattr(model, "filter_sequence"):
            return
        was_training = model.training
        old_backprop = getattr(model, "infer_backprop", False)
        model.eval()
        model.infer_backprop = False
        try:
            batch = self._fixed_batch(pl_module.device)
            payload = self._payload(model, batch)
            if payload:
                wandb_run.log(payload, step=trainer.global_step)
        except Exception as exc:
            log.warning(f"[fond-viz] validation visualization failed: {exc!r}")
        finally:
            model.infer_backprop = old_backprop
            model.train(was_training)

    def _get_wandb_run(self, trainer):
        logger = getattr(trainer, "logger", None)
        exp = getattr(logger, "experiment", None)
        return exp if exp is not None and hasattr(exp, "log") else None

    def _fixed_batch(self, device):
        if self._batch is None:
            self._batch = next(iter(self.val_loader))
        return {
            k: (torch.nan_to_num(v.to(device), 0.0) if torch.is_tensor(v) else v)
            for k, v in self._batch.items()
        }

    def _payload(self, model, batch):
        import wandb

        info = model.filter_sequence(
            batch,
            self.history_size,
            beta=float(getattr(model, "beta", 1.0)),
            infer_objective="free_energy",
            return_diag=False,
        )
        eta = info["emb"]
        target = batch["pixels"].float().clamp(0.0, 1.0)
        recon = model.decode(eta).detach().clamp(0.0, 1.0)
        B, T = eta.shape[:2]
        n = min(B, 4)
        t = min(T, self.num_frames)

        flat_eta = eta.detach().reshape(B * T, -1)
        mu = flat_eta.mean(dim=0, keepdim=True)
        std = flat_eta.std(dim=0, keepdim=True).clamp_min(1e-4)
        agg = mu + std * torch.randn(n * t, flat_eta.size(-1), device=flat_eta.device)
        agg_img = model.decode(agg.view(n, t, -1)).detach().clamp(0.0, 1.0)

        prior = model.prior_param.expand(n * t, -1)
        prior_img = model.decode(prior.view(n, t, -1)).detach().clamp(0.0, 1.0)

        rollout_true, rollout_pred, mse = self._decoded_rollout(model, batch, eta, info["act_emb"])
        payload = {
            "val/recon_input": wandb.Image(self._filmstrip(target[:n, :t])),
            "val/recon_posterior": wandb.Image(self._filmstrip(recon[:n, :t])),
            "val/hall_aggpost": wandb.Image(self._filmstrip(agg_img)),
            "val/hall_prior": wandb.Image(
                self._filmstrip(prior_img),
                caption="prior samples (may look unstructured even for a good code)",
            ),
            "val/eta_absmean": eta.detach().abs().mean().item(),
            "val/eta_std": eta.detach().std().item(),
        }
        if rollout_true is not None and rollout_pred is not None:
            payload["val/rollout_true"] = wandb.Image(self._filmstrip(rollout_true[:n]))
            payload["val/rollout_pred"] = wandb.Image(self._filmstrip(rollout_pred[:n]))
            for h in (1, 4, 8):
                if h in mse:
                    payload[f"val/rollout_mse_h{h}"] = mse[h]
        return payload

    def _decoded_rollout(self, model, batch, eta, act_emb):
        HS = self.history_size
        B, T = eta.shape[:2]
        if T <= HS:
            return None, None, {}
        emb = eta[:, :HS].detach().clone()
        preds = []
        for t in range(HS, T):
            pred = model.predict(emb[:, -HS:], act_emb[:, t - HS:t])[:, -1:]
            preds.append(pred)
            emb = torch.cat([emb, pred], dim=1)
        pred_eta = torch.cat(preds, dim=1)
        pred_img = model.decode(pred_eta).detach().clamp(0.0, 1.0)
        true_img = batch["pixels"][:, HS:T].float().clamp(0.0, 1.0)
        mse = {}
        for h in (1, 4, 8):
            idx = h - 1
            if idx < pred_img.size(1):
                mse[h] = F.mse_loss(pred_img[:, idx], true_img[:, idx]).item()
        return true_img, pred_img, mse

    def _filmstrip(self, frames):
        frames = frames.detach().float().cpu().clamp(0.0, 1.0)
        if frames.ndim == 4:
            frames = frames.unsqueeze(0)
        rows = [torch.cat([frame for frame in seq], dim=2) for seq in frames]
        return torch.cat(rows, dim=1).permute(1, 2, 0).numpy()
