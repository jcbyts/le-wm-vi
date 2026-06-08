"""JEPA Implementation"""

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from module import (
    SIGReg,
    effective_rank,
    poisson_kl_rates,
    poisson_kl_log_rates,
    sample_capacity_rates,
    scalar_poisson_mi,
    solve_poisson_capacity_prior,
    two_sample_sigreg,
)

def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v

class JEPA(nn.Module):

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()

    def encode(self, info):
        """Encode observations and actions into embeddings.
        info: dict with pixels and action keys
        """

        pixels = info['pixels'].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...") # flatten for encoding
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]  # cls token
        emb = self.projector(pixels_emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict(self, emb, act_emb):
        """Predict next state embedding
        emb: (B, T, D)
        act_emb: (B, T, A_emb)
        """
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    ####################
    ## Inference only ##
    ####################

    def rollout(self, info, action_sequence, history_size: int = 3):
        """Rollout the model given an initial info dict and action sequence.
        pixels: (B, S, T, C, H, W)
        action_sequence: (B, S, T, action_dim)
         - S is the number of action plan samples
         - T is the time horizon
        """

        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        # copy and encode initial info dict
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        _init = {k: detach_clone(v) for k, v in _init.items()}

        # flatten batch and sample dimensions for rollout
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        # rollout predictor autoregressively for n_steps
        HS = history_size
        for t in range(n_steps):
            act_emb = self.action_encoder(act)
            emb_trunc = emb[:, -HS:]  # (BS, HS, D)
            act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
            emb = torch.cat([emb, pred_emb], dim=1)  # (BS, T+1, D)

            next_act = act_future[:, t : t + 1, :]  # (BS, 1, action_dim)
            act = torch.cat([act, next_act], dim=1)  # (BS, T+1, action_dim)

        # predict the last state
        act_emb = self.action_encoder(act)  # (BS, T, A_emb)
        emb_trunc = emb[:, -HS:]  # (BS, HS, D)
        act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
        pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
        emb = torch.cat([emb, pred_emb], dim=1)

        # unflatten batch and sample dimensions
        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout

        return info

    def criterion(self, info_dict: dict):
        """Compute the cost between predicted embeddings and goal embeddings."""
        pred_emb = info_dict["predicted_emb"]  # (B,S, T-1, dim)
        goal_emb = info_dict["goal_emb"]  # (B, S, T, dim)

        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)

        # return last-step cost per action candidate
        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))  # (B, S)

        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """ Compute the cost of action candidates given an info dict with goal and initial state."""

        assert "goal" in info_dict, "goal not in info_dict"

        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]

        for k in info_dict:
            if k.startswith("goal_"):
                goal[k[len("goal_") :]] = goal.pop(k)

        goal.pop("action")
        goal = self.encode(goal)

        info_dict["goal_emb"] = goal["emb"]
        info_dict = self.rollout(info_dict, action_candidates)

        cost = self.criterion(info_dict)
        
        return cost


class PoisWM(JEPA):
    """Bounded-rate PoisWM with Capacity-SIGReg anchoring."""

    def __init__(
        self,
        target_rate=1.0,
        A_over_mu=8.0,
        capacity_grid_size=512,
        sigreg_knots=17,
        sigreg_num_proj=1024,
        goal_cost="sym_kl",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.target_rate = float(target_rate)
        self.A_over_mu = float(A_over_mu)
        self.rate_min = 1e-4 * self.target_rate
        self.rate_max = self.A_over_mu * self.target_rate
        self.capacity_grid_size = int(capacity_grid_size)
        self.sigreg_knots = int(sigreg_knots)
        self.sigreg_num_proj = int(sigreg_num_proj)
        self.goal_cost = str(goal_cost).lower()

        rates, pi, capacity, nmax = solve_poisson_capacity_prior(
            mu=self.target_rate,
            A=self.rate_max,
            K=self.capacity_grid_size,
        )
        self.register_buffer("capacity_rates", torch.as_tensor(rates, dtype=torch.float32))
        self.register_buffer("capacity_pi", torch.as_tensor(pi, dtype=torch.float32))
        self.capacity_target = float(capacity)
        self.capacity_nmax = int(nmax)

    def _bounded_rates(self, raw_u):
        return self.rate_min + (self.rate_max - self.rate_min) * torch.sigmoid(raw_u)

    def encode(self, info):
        output = super().encode(info)
        output["raw_emb"] = output["emb"]
        output["emb"] = self._bounded_rates(output["emb"])
        return output

    def predict(self, emb, act_emb):
        pred = super().predict(emb, act_emb)
        return self._bounded_rates(pred)

    def _exact_poisson_kl(self, lam_tgt, lam_pred):
        """Exact D_KL(Poisson(target rate) || Poisson(predicted rate))."""
        return poisson_kl_rates(lam_tgt, lam_pred).mean()

    def criterion(self, info_dict: dict):
        """Terminal planning cost over bounded Poisson rates."""
        pred_emb = info_dict["predicted_emb"]  # (B,S,T,dim)
        goal_emb = info_dict["goal_emb"]  # (B,S,T,dim)
        lam_goal = goal_emb[..., -1:, :].expand_as(pred_emb)[..., -1, :].detach()
        lam_pred = pred_emb[..., -1, :]

        if self.goal_cost in {"kl", "poisson_kl"}:
            return poisson_kl_rates(lam_goal, lam_pred).sum(dim=-1)
        if self.goal_cost in {"rate_mse", "euclidean_rate"}:
            return (lam_goal - lam_pred).pow(2).sum(dim=-1)
        if self.goal_cost == "log_rate_mse":
            return (
                torch.log(lam_goal.clamp_min(1e-8))
                - torch.log(lam_pred.clamp_min(1e-8))
            ).pow(2).sum(dim=-1)

        return 0.5 * (
            poisson_kl_rates(lam_goal, lam_pred).sum(dim=-1)
            + poisson_kl_rates(lam_pred, lam_goal).sum(dim=-1)
        )

    def _capacity_sigreg(self, lam_anchor):
        target_lam = sample_capacity_rates(
            self.capacity_rates,
            self.capacity_pi,
            shape=lam_anchor.shape,
            device=lam_anchor.device,
            dtype=lam_anchor.dtype,
        )
        s = 2.0 * torch.sqrt(lam_anchor.clamp_min(0.0))
        s_target = 2.0 * torch.sqrt(target_lam.clamp_min(0.0))
        return two_sample_sigreg(
            s,
            s_target,
            knots=self.sigreg_knots,
            num_proj=self.sigreg_num_proj,
        )

    def _diagnostics(self, lam_anchor, beta):
        with torch.no_grad():
            I_per_dim = scalar_poisson_mi(lam_anchor.detach(), nmax=self.capacity_nmax)
            log_lam = torch.log(lam_anchor.detach().clamp_min(1e-8))
            mean_per_dim = lam_anchor.detach().mean(dim=0)
            return {
                "beta": lam_anchor.new_tensor(float(beta)),
                "A_over_mu": lam_anchor.new_tensor(self.A_over_mu),
                "capacity_target": lam_anchor.new_tensor(self.capacity_target),
                "mean_rate_mean": mean_per_dim.mean(),
                "mean_rate_std": mean_per_dim.std(unbiased=False),
                "rate_min_seen": lam_anchor.detach().min(),
                "rate_max_seen": lam_anchor.detach().max(),
                "effective_rank": effective_rank(log_lam),
                "poisson_mi_mean": I_per_dim.mean(),
                "poisson_mi_min": I_per_dim.min(),
                "poisson_mi_max": I_per_dim.max(),
            }

    def compute_loss(self, batch, cfg):
        """Compute exact Poisson transition KL plus Capacity-SIGReg."""
        if "action" in batch:
            batch["action"] = torch.nan_to_num(batch["action"], 0.0)

        output = self.encode(batch)
        emb = output["emb"]
        act_emb = output.get("act_emb")

        ctx_len = cfg.history_size
        n_preds = cfg.num_preds
        beta = cfg.loss.get("beta", 1.0)

        ctx_emb = emb[:, :ctx_len]
        ctx_act = act_emb[:, :ctx_len] if act_emb is not None else None
        tgt_emb = emb[:, n_preds:].detach()
        pred_emb = self.predict(ctx_emb, ctx_act)

        pred_loss = self._exact_poisson_kl(tgt_emb, pred_emb)
        lam_anchor = emb.reshape(-1, emb.shape[-1])
        anchor_loss = self._capacity_sigreg(lam_anchor)

        output["pred_loss"] = pred_loss
        output["anchor_loss"] = anchor_loss
        output["reg_loss"] = anchor_loss
        output["loss"] = pred_loss + float(beta) * anchor_loss
        output.update(self._diagnostics(lam_anchor, beta))
        return output

class LogRateFisherPoisWM(JEPA):
    """Log-rate Fisher PoisWM with Gaussian SIGReg on residual log-rates."""

    def __init__(
        self,
        tau=1.0,
        sigreg_knots=17,
        sigreg_num_proj=1024,
        goal_cost="fisher",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.tau = float(tau)
        self.goal_cost = str(goal_cost).lower()
        self.sigreg = SIGReg(knots=sigreg_knots, num_proj=sigreg_num_proj)

    def _log_rate(self, u):
        tau = float(self.tau)
        return tau * u - 0.5 * tau * tau

    def _rates(self, u):
        return torch.exp(self._log_rate(u))

    def criterion(self, info_dict: dict):
        """Terminal planning cost over residual log-rate coordinates."""
        pred_u = info_dict["predicted_emb"][..., -1, :]
        goal_u = info_dict["goal_emb"][..., -1:, :].expand_as(
            info_dict["predicted_emb"]
        )[..., -1, :].detach()

        delta_u = goal_u - pred_u
        if self.goal_cost in {"u_mse", "mse"}:
            return delta_u.pow(2).sum(dim=-1)

        delta_log_rate = float(self.tau) * delta_u
        if self.goal_cost in {"log_rate_mse", "log_mse"}:
            return delta_log_rate.pow(2).sum(dim=-1)

        lam_pred = self._rates(pred_u)
        return 0.5 * (lam_pred * delta_log_rate.pow(2)).sum(dim=-1)

    def _diagnostics(self, u_anchor, lam_anchor, lam_pred, beta):
        with torch.no_grad():
            return {
                "beta": u_anchor.new_tensor(float(beta)),
                "tau": u_anchor.new_tensor(float(self.tau)),
                "effective_rank_u": effective_rank(u_anchor.detach()),
                "u_mean": u_anchor.detach().mean(),
                "u_std": u_anchor.detach().std(unbiased=False),
                "rate_mean": lam_anchor.detach().mean(),
                "rate_std": lam_anchor.detach().std(unbiased=False),
                "rate_min_seen": lam_anchor.detach().min(),
                "rate_max_seen": lam_anchor.detach().max(),
                "fisher_weight_mean": lam_pred.detach().mean(),
                "fisher_weight_max": lam_pred.detach().max(),
            }

    def compute_loss(self, batch, cfg):
        """Compute local Fisher Poisson transition KL plus Gaussian SIGReg."""
        if "action" in batch:
            batch["action"] = torch.nan_to_num(batch["action"], 0.0)

        output = self.encode(batch)
        u = output["emb"]
        act_emb = output.get("act_emb")

        ctx_len = cfg.history_size
        n_preds = cfg.num_preds
        beta = cfg.loss.get("beta", 0.09)

        ctx_u = u[:, :ctx_len]
        ctx_act = act_emb[:, :ctx_len] if act_emb is not None else None
        tgt_u = u[:, n_preds:]
        pred_u = self.predict(ctx_u, ctx_act)

        eps = float(self.tau) * (tgt_u - pred_u)
        lam_pred = self._rates(pred_u)
        pred_loss = 0.5 * (lam_pred * eps.pow(2)).mean()

        u_anchor = u.reshape(-1, u.shape[-1])
        anchor_loss = self.sigreg(u_anchor.unsqueeze(0))
        lam_anchor = self._rates(u_anchor)

        output["pred_loss"] = pred_loss
        output["anchor_loss"] = anchor_loss
        output["reg_loss"] = anchor_loss
        output["loss"] = pred_loss + float(beta) * anchor_loss
        output.update(self._diagnostics(u_anchor, lam_anchor, lam_pred, beta))
        return output

class MetabolicSigRegPoisWM(JEPA):
    """PoisWM with exact transition KL and a metabolic prior-KL SIGReg anchor."""

    def __init__(
        self,
        lambda0=1.0,
        alpha=1.0,
        log_rate_min=-12.0,
        log_rate_max=5.0,
        target_grid_min=-12.0,
        target_grid_max=5.0,
        target_grid_size=65536,
        sigreg_knots=17,
        sigreg_num_proj=1024,
        goal_cost="poisson_kl",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lambda0 = float(lambda0)
        self.alpha = float(alpha)
        self.log_rate_min = float(log_rate_min)
        self.log_rate_max = float(log_rate_max)
        self.target_grid_min = max(float(target_grid_min), self.log_rate_min)
        self.target_grid_max = min(float(target_grid_max), self.log_rate_max)
        self.target_grid_size = int(target_grid_size)
        if self.log_rate_min >= self.log_rate_max:
            raise ValueError("log_rate_min must be less than log_rate_max")
        if self.target_grid_min >= self.target_grid_max:
            raise ValueError("target grid support is empty after log-rate clamping")
        self.sigreg_knots = int(sigreg_knots)
        self.sigreg_num_proj = int(sigreg_num_proj)
        self.goal_cost = str(goal_cost).lower()

        r_grid = torch.linspace(
            self.target_grid_min,
            self.target_grid_max,
            self.target_grid_size,
            dtype=torch.float32,
        )
        lam_grid = self.lambda0 * torch.exp(r_grid)
        k0_grid = lam_grid * r_grid - lam_grid + self.lambda0
        # r = log(lambda / lambda0), so d lambda = lambda0 * exp(r) dr.
        # The constant log(lambda0) cancels in the normalized discrete density.
        logp = r_grid - self.alpha * k0_grid
        probs = torch.softmax(logp, dim=0)
        rate_mean = (probs * lam_grid).sum()

        self.register_buffer("target_r_grid", r_grid)
        self.register_buffer("target_r_probs", probs)
        self.register_buffer("target_rate_mean", rate_mean)
        self.register_buffer(
            "target_rate_std",
            torch.sqrt((probs * (lam_grid - rate_mean).square()).sum()),
        )
        self.register_buffer("target_prior_kl_mean", (probs * k0_grid).sum())

    def _clamp_log_rate(self, r):
        return r.clamp(self.log_rate_min, self.log_rate_max)

    def _rates(self, r):
        return self.lambda0 * torch.exp(self._clamp_log_rate(r))

    def _prior_kl_from_r(self, r):
        r = self._clamp_log_rate(r)
        lam = self._rates(r)
        return lam * r - lam + self.lambda0

    def encode(self, info):
        output = super().encode(info)
        output["raw_emb"] = output["emb"]
        output["emb"] = self._clamp_log_rate(output["emb"])
        return output

    def predict(self, emb, act_emb):
        return self._clamp_log_rate(super().predict(emb, act_emb))

    def _sample_target_r(self, shape, device, dtype):
        probs = self.target_r_probs.to(device=device)
        idx = torch.multinomial(
            probs,
            num_samples=math.prod(shape),
            replacement=True,
        )
        return self.target_r_grid.to(device=device, dtype=dtype)[idx].reshape(shape)

    def criterion(self, info_dict: dict):
        """Terminal planning cost over Poisson residual log-rate coordinates."""
        pred_r = info_dict["predicted_emb"][..., -1, :]
        goal_r = info_dict["goal_emb"][..., -1:, :].expand_as(
            info_dict["predicted_emb"]
        )[..., -1, :].detach()

        if self.goal_cost in {"r_mse", "log_rate_mse", "mse"}:
            return (goal_r - pred_r).pow(2).sum(dim=-1)

        lam_goal = self._rates(goal_r)
        lam_pred = self._rates(pred_r)
        if self.goal_cost in {"rate_mse", "euclidean_rate"}:
            return (lam_goal - lam_pred).pow(2).sum(dim=-1)
        if self.goal_cost in {"sym_kl", "symmetric_kl"}:
            return 0.5 * (
                poisson_kl_log_rates(goal_r, pred_r, self.lambda0).sum(dim=-1)
                + poisson_kl_log_rates(pred_r, goal_r, self.lambda0).sum(dim=-1)
            )

        return poisson_kl_log_rates(goal_r, pred_r, self.lambda0).sum(dim=-1)

    def _metabolic_sigreg(self, r_anchor):
        r_target = self._sample_target_r(
            r_anchor.shape,
            device=r_anchor.device,
            dtype=r_anchor.dtype,
        )
        return two_sample_sigreg(
            r_anchor,
            r_target,
            knots=self.sigreg_knots,
            num_proj=self.sigreg_num_proj,
        )

    def _diagnostics(self, r_anchor, lam_anchor, lam_pred, beta):
        with torch.no_grad():
            prior_kl = self._prior_kl_from_r(r_anchor.detach())
            return {
                "beta": r_anchor.new_tensor(float(beta)),
                "alpha": r_anchor.new_tensor(self.alpha),
                "lambda0": r_anchor.new_tensor(self.lambda0),
                "log_rate_min": r_anchor.new_tensor(self.log_rate_min),
                "log_rate_max": r_anchor.new_tensor(self.log_rate_max),
                "effective_rank_r": effective_rank(r_anchor.detach()),
                "r_mean": r_anchor.detach().mean(),
                "r_std": r_anchor.detach().std(unbiased=False),
                "r_at_min_frac": (
                    r_anchor.detach() <= self.log_rate_min + 1e-6
                ).float().mean(),
                "r_at_max_frac": (
                    r_anchor.detach() >= self.log_rate_max - 1e-6
                ).float().mean(),
                "prior_kl_mean": prior_kl.mean(),
                "prior_kl_std": prior_kl.std(unbiased=False),
                "rate_mean": lam_anchor.detach().mean(),
                "rate_std": lam_anchor.detach().std(unbiased=False),
                "rate_min_seen": lam_anchor.detach().min(),
                "rate_max_seen": lam_anchor.detach().max(),
                "pred_rate_mean": lam_pred.detach().mean(),
                "pred_rate_max": lam_pred.detach().max(),
                "target_rate_mean": self.target_rate_mean.to(r_anchor.device),
                "target_rate_std": self.target_rate_std.to(r_anchor.device),
                "target_prior_kl_mean": self.target_prior_kl_mean.to(r_anchor.device),
            }

    def compute_loss(self, batch, cfg):
        """Compute exact Poisson transition KL plus metabolic SIGReg."""
        if "action" in batch:
            batch["action"] = torch.nan_to_num(batch["action"], 0.0)

        output = self.encode(batch)
        r = output["emb"]
        act_emb = output.get("act_emb")

        ctx_len = cfg.history_size
        n_preds = cfg.num_preds
        beta = cfg.loss.get("beta", 0.09)

        ctx_r = r[:, :ctx_len]
        ctx_act = act_emb[:, :ctx_len] if act_emb is not None else None
        tgt_r = r[:, n_preds:].detach()
        pred_r = self.predict(ctx_r, ctx_act)

        lam_pred = self._rates(pred_r)
        pred_loss = poisson_kl_log_rates(tgt_r, pred_r, self.lambda0).sum(dim=-1).mean()

        r_anchor = r.reshape(-1, r.shape[-1])
        lam_anchor = self._rates(r_anchor)
        anchor_loss = self._metabolic_sigreg(r_anchor)

        output["pred_loss"] = pred_loss
        output["anchor_loss"] = anchor_loss
        output["reg_loss"] = anchor_loss
        output["loss"] = pred_loss + float(beta) * anchor_loss
        if not torch.isfinite(output["loss"]):
            raise FloatingPointError("non-finite MetabolicSigRegPoisWM loss")
        output.update(self._diagnostics(r_anchor, lam_anchor, lam_pred, beta))
        return output



class ConvRSSMPoissonWM(nn.Module):
    """Conv sparse-Poisson perceptual code with compact recurrent dynamics.

    The high-dimensional code is a spatial residual log-rate map trained with
    exact Poisson KL. The exposed ``emb`` is a compact state used by monitoring
    and planning, so CEM does not have to optimize directly in raw Poisson
    firing-rate geometry.
    """

    def __init__(
        self,
        action_encoder,
        img_size=224,
        in_channels=3,
        rate_channels=64,
        state_dim=128,
        hidden_channels=(32, 64, 128),
        lambda0=1.0,
        log_rate_min=-8.0,
        log_rate_max=5.0,
        alpha=3.0,
        target_grid_min=-8.0,
        target_grid_max=5.0,
        target_grid_size=65536,
        sigreg_knots=17,
        sigreg_num_proj=128,
        goal_cost="compact_mse",
        planner_poisson_weight=0.0,
        compact_loss_weight=0.1,
    ):
        super().__init__()
        self.action_encoder = action_encoder
        self.img_size = int(img_size)
        self.rate_channels = int(rate_channels)
        self.state_dim = int(state_dim)
        self.lambda0 = float(lambda0)
        self.log_rate_min = float(log_rate_min)
        self.log_rate_max = float(log_rate_max)
        self.alpha = float(alpha)
        self.target_grid_min = max(float(target_grid_min), self.log_rate_min)
        self.target_grid_max = min(float(target_grid_max), self.log_rate_max)
        self.target_grid_size = int(target_grid_size)
        self.sigreg_knots = int(sigreg_knots)
        self.sigreg_num_proj = int(sigreg_num_proj)
        self.goal_cost = str(goal_cost).lower()
        self.planner_poisson_weight = float(planner_poisson_weight)
        self.compact_loss_weight = float(compact_loss_weight)
        if self.log_rate_min >= self.log_rate_max:
            raise ValueError("log_rate_min must be less than log_rate_max")

        channels = [in_channels, *hidden_channels]
        blocks = []
        strides = [4, 2, 2]
        for idx, (cin, cout) in enumerate(zip(channels[:-1], channels[1:])):
            blocks += [
                nn.Conv2d(
                    cin,
                    cout,
                    kernel_size=5 if idx == 0 else 3,
                    stride=strides[idx],
                    padding=2 if idx == 0 else 1,
                ),
                nn.GroupNorm(num_groups=min(8, cout), num_channels=cout),
                nn.Softplus(beta=1.0),
            ]
        blocks += [nn.Conv2d(channels[-1], self.rate_channels, kernel_size=3, stride=2, padding=1)]
        self.encoder = nn.Sequential(*blocks)

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, self.img_size, self.img_size)
            rate_shape = self.encoder(dummy).shape[1:]
        self.rate_shape = tuple(int(x) for x in rate_shape)
        self.rate_dim = int(math.prod(self.rate_shape))

        self.to_state = nn.Sequential(
            nn.LayerNorm(self.rate_dim),
            nn.Linear(self.rate_dim, self.state_dim),
            nn.SiLU(),
            nn.LayerNorm(self.state_dim),
        )
        self.state_to_rate = nn.Sequential(
            nn.Linear(self.state_dim, 4 * self.state_dim),
            nn.SiLU(),
            nn.Linear(4 * self.state_dim, self.rate_dim),
        )

        action_dim = int(getattr(action_encoder, "emb_dim", self.state_dim))
        self.transition = nn.GRU(
            input_size=self.state_dim + action_dim,
            hidden_size=self.state_dim,
            batch_first=True,
        )
        self.sigreg_proj = nn.Linear(self.rate_dim, min(self.rate_dim, 512), bias=False)

        r_grid = torch.linspace(
            self.target_grid_min,
            self.target_grid_max,
            self.target_grid_size,
            dtype=torch.float32,
        )
        lam_grid = self.lambda0 * torch.exp(r_grid)
        k0_grid = lam_grid * r_grid - lam_grid + self.lambda0
        logp = r_grid - self.alpha * k0_grid
        probs = torch.softmax(logp, dim=0)
        self.register_buffer("target_r_grid", r_grid)
        self.register_buffer("target_r_probs", probs)
        self.register_buffer("target_rate_mean", (probs * lam_grid).sum())
        self.register_buffer("target_prior_kl_mean", (probs * k0_grid).sum())

    def _clamp_log_rate(self, r):
        return r.clamp(self.log_rate_min, self.log_rate_max)

    def _rates(self, r):
        return self.lambda0 * torch.exp(self._clamp_log_rate(r))

    def _prior_kl_from_r(self, r):
        r = self._clamp_log_rate(r)
        lam = self._rates(r)
        return lam * r - lam + self.lambda0

    def _sample_target_r(self, shape, device, dtype):
        probs = self.target_r_probs.to(device=device)
        idx = torch.multinomial(probs, num_samples=math.prod(shape), replacement=True)
        return self.target_r_grid.to(device=device, dtype=dtype)[idx].reshape(shape)

    def encode(self, info):
        pixels = info["pixels"].float()
        b = pixels.size(0)
        flat_pixels = rearrange(pixels, "b t c h w -> (b t) c h w")
        r_map = self._clamp_log_rate(self.encoder(flat_pixels))
        r_flat = r_map.flatten(1)
        state = self.to_state(r_flat)
        info["r_emb"] = rearrange(r_flat, "(b t) d -> b t d", b=b)
        info["rate_emb"] = self._rates(info["r_emb"])
        info["emb"] = rearrange(state, "(b t) d -> b t d", b=b)
        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])
        return info

    def decode_state(self, state):
        shape = state.shape
        flat = state.reshape(-1, shape[-1])
        r = self._clamp_log_rate(self.state_to_rate(flat))
        return r.reshape(*shape[:-1], self.rate_dim)

    def predict(self, emb, act_emb):
        x = torch.cat([emb, act_emb], dim=-1)
        out, _ = self.transition(x)
        return out

    def _anchor_loss(self, r_anchor):
        target = self._sample_target_r(
            r_anchor.shape,
            device=r_anchor.device,
            dtype=r_anchor.dtype,
        )
        r_proj = self.sigreg_proj(r_anchor)
        target_proj = self.sigreg_proj(target)
        return two_sample_sigreg(
            r_proj,
            target_proj,
            knots=self.sigreg_knots,
            num_proj=self.sigreg_num_proj,
        )

    def _diagnostics(self, z_anchor, r_anchor, r_pred, beta, compact_loss):
        with torch.no_grad():
            lam_anchor = self._rates(r_anchor.detach())
            prior_kl = self._prior_kl_from_r(r_anchor.detach())
            lam_pred = self._rates(r_pred.detach())
            return {
                "beta": z_anchor.new_tensor(float(beta)),
                "alpha": z_anchor.new_tensor(self.alpha),
                "lambda0": z_anchor.new_tensor(self.lambda0),
                "compact_loss_weight": z_anchor.new_tensor(self.compact_loss_weight),
                "compact_loss": compact_loss.detach(),
                "effective_rank_z": effective_rank(z_anchor.detach()),
                "z_mean": z_anchor.detach().mean(),
                "z_std": z_anchor.detach().std(unbiased=False),
                "r_mean": r_anchor.detach().mean(),
                "r_std": r_anchor.detach().std(unbiased=False),
                "r_at_min_frac": (r_anchor.detach() <= self.log_rate_min + 1e-6).float().mean(),
                "r_at_max_frac": (r_anchor.detach() >= self.log_rate_max - 1e-6).float().mean(),
                "rate_mean": lam_anchor.mean(),
                "rate_std": lam_anchor.std(unbiased=False),
                "rate_max_seen": lam_anchor.max(),
                "pred_rate_mean": lam_pred.mean(),
                "pred_rate_max": lam_pred.max(),
                "prior_kl_mean": prior_kl.mean(),
                "prior_kl_std": prior_kl.std(unbiased=False),
                "target_rate_mean": self.target_rate_mean.to(z_anchor.device),
                "target_prior_kl_mean": self.target_prior_kl_mean.to(z_anchor.device),
            }

    def compute_loss(self, batch, cfg):
        if "action" in batch:
            batch["action"] = torch.nan_to_num(batch["action"], 0.0)
        output = self.encode(batch)
        z = output["emb"]
        r = output["r_emb"]
        act_emb = output["act_emb"]
        beta = cfg.loss.get("beta", 0.1)

        ctx_z = z[:, :-1]
        ctx_act = act_emb[:, :-1]
        tgt_z = z[:, 1:].detach()
        tgt_r = r[:, 1:].detach()
        pred_z = self.predict(ctx_z, ctx_act)
        pred_r = self.decode_state(pred_z)

        pred_loss = poisson_kl_log_rates(tgt_r, pred_r, self.lambda0).mean()
        compact_loss = (pred_z - tgt_z).pow(2).mean()
        r_anchor = r.reshape(-1, r.shape[-1])
        z_anchor = z.reshape(-1, z.shape[-1])
        anchor_loss = self._anchor_loss(r_anchor)

        output["pred_loss"] = pred_loss
        output["compact_loss"] = compact_loss
        output["anchor_loss"] = anchor_loss
        output["reg_loss"] = anchor_loss
        output["loss"] = pred_loss + self.compact_loss_weight * compact_loss + float(beta) * anchor_loss
        if not torch.isfinite(output["loss"]):
            raise FloatingPointError("non-finite ConvRSSMPoissonWM loss")
        output.update(self._diagnostics(z_anchor, r_anchor, pred_r, beta, compact_loss))
        return output

    def rollout(self, info, action_sequence, history_size: int = 3):
        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        init = self.encode(init)
        emb = init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        HS = history_size
        for t in range(n_steps):
            act_emb = self.action_encoder(act)
            pred = self.predict(emb[:, -HS:], act_emb[:, -HS:])[:, -1:]
            emb = torch.cat([emb, pred], dim=1)
            act = torch.cat([act, act_future[:, t : t + 1]], dim=1)

        act_emb = self.action_encoder(act)
        pred = self.predict(emb[:, -HS:], act_emb[:, -HS:])[:, -1:]
        emb = torch.cat([emb, pred], dim=1)
        info["predicted_emb"] = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        return info

    def criterion(self, info_dict):
        pred_z = info_dict["predicted_emb"][..., -1, :]
        goal_z = info_dict["goal_emb"][..., -1:, :].expand_as(
            info_dict["predicted_emb"]
        )[..., -1, :].detach()
        compact_cost = (pred_z - goal_z).pow(2).sum(dim=-1)
        if self.goal_cost in {"compact_mse", "state_mse", "mse"}:
            return compact_cost

        pred_r = self.decode_state(pred_z)
        goal_r = info_dict["goal_r_emb"][..., -1, :].detach()
        poisson_cost = poisson_kl_log_rates(goal_r, pred_r, self.lambda0).mean(dim=-1)
        if self.goal_cost in {"poisson_kl", "kl"}:
            return poisson_cost
        if self.goal_cost == "hybrid":
            return compact_cost + self.planner_poisson_weight * poisson_cost
        return compact_cost

    def get_cost(self, info_dict, action_candidates):
        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]
        for k in list(goal.keys()):
            if k.startswith("goal_"):
                goal[k[len("goal_") :]] = goal.pop(k)
        goal.pop("action", None)
        goal = self.encode(goal)
        info_dict["goal_emb"] = goal["emb"]
        info_dict["goal_r_emb"] = goal["r_emb"]
        info_dict = self.rollout(info_dict, action_candidates)
        return self.criterion(info_dict)
