"""FOND-JEPA: variational-inference reframing of LeWorldModel, generalized over
the latent family via LatentHead (variants 3-6 of the Fisher-JEPA spec).

Relationship to LeWM (jepa.JEPA):
  - predictor (ARPredictor), action_encoder (Embedder): UNCHANGED architecturally.
  - amortized ViT encoder      -> iterative inference through a weak DECODER
  - target = slice of encoder  -> stop-grad inference posterior (natural param)
  - pred_loss = MSE            -> head.pred_term (exact KL or quadratic Fisher)
  - SIGReg anti-collapse       -> reconstruction anchor (the VI data term)

`emb` is the per-step NATURAL PARAMETER of the family (dim P = D * head.param_mult):
  poisson  -> log-rate u           (P = D)
  gaussian -> concat(mu, logvar)   (P = 2D)
The predictor operates on emb (P-dim), so the planner/eval path is shared.

Gradient routing (preserved from the validated reference, spec §2.3):
  - inference steps can be differentiable (BPTT through the inner loop) or
    first-order / detached for ablations. The predictive target remains
    stop-grad, while reconstruction can train the decoder through inference.
  - inference steps the natural parameter eta (spec §4 Alg.1), which is what makes
    it family-agnostic (the reference's Poisson code stepped the code z instead,
    which does not generalize to Gaussian's 2D parameter).

Drop-in for jepa.JEPA: exposes encode / predict / decode with the same `emb`
(B,T,P) contract, so rollout / criterion / get_cost work as-is (terminal cost is
adapted per family in a later stage — spec §5.5).
"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from latent import LatentHead, make_head


# ----------------------------------------------------------------------------
# Observation model (decoder). Deliberately light (spec §6): its job is to give
# inference a reconstruction gradient and make a collapsed code reconstructively
# costly. Capacity lives in the predictor, not here. Outputs [0,1] (Sigmoid) to
# match the PushT recon target (spec corrections C2).
# ----------------------------------------------------------------------------

class ConvDecoder(nn.Module):
    """Latent code (B, D) -> image (B, C, hw, hw). D reshaped to a grid x grid
    spatial map (lat_ch = D / grid^2) then nearest-upsampled to hw."""

    def __init__(self, latent_dim, img_ch=3, img_hw=64, grid=8):
        super().__init__()
        assert latent_dim % (grid * grid) == 0, "latent_dim must divide grid^2"
        self.grid = grid
        self.lat_ch = latent_dim // (grid * grid)
        ch = max(self.lat_ch, 32)
        ups = [nn.Conv2d(self.lat_ch, ch, 3, padding=1), nn.SiLU()]
        hw = grid
        while hw < img_hw:
            ups += [nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(ch, ch, 3, padding=1), nn.SiLU()]
            hw *= 2
        assert hw == img_hw, f"grid {grid} cannot upsample to img_hw {img_hw} by 2x"
        ups += [nn.Conv2d(ch, img_ch, 3, padding=1), nn.Sigmoid()]
        self.net = nn.Sequential(*ups)

    def forward(self, code):
        b = code.size(0)
        x = code.view(b, self.lat_ch, self.grid, self.grid)
        return self.net(x)


# ----------------------------------------------------------------------------
# FOND-JEPA
# ----------------------------------------------------------------------------

class FONDJEPA(nn.Module):
    """Inference-through-decoder JEPA, generalized over LatentHead.

    `head` selects the family (gaussian | poisson). `latent_dim` is D; the
    natural-parameter dim is P = D * head.param_mult and is what the predictor /
    action_encoder must be sized for (set in config)."""

    def __init__(
        self,
        decoder,                 # ConvDecoder (observation model)
        predictor,               # ARPredictor (input_dim=output_dim=P) — UNCHANGED arch
        action_encoder,          # Embedder (emb_dim=P) — UNCHANGED arch
        latent_dim,              # D
        head=None,               # LatentHead instance, or a family string
        projector=None,          # kept for API parity; Identity by default
        pred_proj=None,
        k_inner=4,               # inference steps per frame
        tau=0.2,                 # fixed EATcubic temperature (Poisson)
        infer_lr=1.0,            # inference step size on the natural parameter
        infer_grad_clip=None,     # optional per-example norm clip for inner gradients
        infer_momentum=0.0,       # optional heavy-ball momentum for inner inference
        infer_backprop=True,      # BPTT through the inner inference loop
        k_bptt=None,              # truncate BPTT to the final k steps when needed
        full_fisher=False,       # gaussian: full 2nd-order quad vs mu-only (see latent.variant_name)
        fixed_unit_variance=False,  # gaussian floor: hold logvar == 0 (G = I)
        infer_init="static_prior",  # static_prior (scheme B) | predictive_prior (scheme A)
        img_ch=3,
        img_hw=64,
    ):
        super().__init__()
        if head is None or isinstance(head, str):
            head = make_head(
                head or "poisson", tau=tau, full_fisher=full_fisher,
                fixed_unit_variance=fixed_unit_variance,
            )
        self.head = head
        infer_init = {"prior": "static_prior", "predictive": "predictive_prior"}.get(infer_init, infer_init)
        self.infer_init = infer_init
        assert infer_init in ("static_prior", "predictive_prior"), infer_init
        self.decoder = decoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        self.latent_dim = latent_dim
        self.param_dim = latent_dim * head.param_mult
        self.k_inner = k_inner
        self.tau = tau
        self.infer_lr = self._family_value(infer_lr)
        self.infer_grad_clip = self._family_value(infer_grad_clip, default=None)
        self.infer_momentum = float(self._family_value(infer_momentum))
        assert self.infer_momentum >= 0.0, "infer_momentum must be nonnegative"
        self.infer_backprop = bool(infer_backprop)
        self.k_bptt = k_inner if k_bptt is None else int(k_bptt)
        self.img_ch, self.img_hw = img_ch, img_hw
        self.beta = 1.0
        # learnable prior natural-parameter (inference initialization / belief prior)
        self.prior_param = nn.Parameter(torch.zeros(1, self.param_dim))

    def _family_value(self, value, default=0.0):
        """Allow Hydra configs to pass either a scalar or {family: value}."""
        if value is None:
            return default
        if isinstance(value, dict):
            return value.get(self.head.family, value.get("default", default))
        if hasattr(value, "get") and not isinstance(value, (str, bytes)):
            found = value.get(self.head.family, None)
            return value.get("default", default) if found is None else found
        return value

    def _inner_step(self, param, grad, velocity, preconditioner=None, detach_grad=True):
        if detach_grad:
            grad = grad.detach()

        if preconditioner is not None:
            if detach_grad:
                preconditioner = preconditioner.detach()
            grad = grad / (preconditioner + 1e-6)

        if self.infer_grad_clip is not None:
            max_norm = float(self.infer_grad_clip)
            flat = grad.flatten(1)
            norm = flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
            scale = (max_norm / norm).clamp(max=1.0).view(-1, *([1] * (grad.ndim - 1)))
            grad = grad * scale
        step = -float(self.infer_lr) * grad
        if self.infer_momentum > 0.0:
            velocity = self.infer_momentum * velocity + step
            step = velocity
        param = self.head.clamp_param(param + step)
        return param, velocity

    def _set_runtime_inference(self, *, infer_backprop=None, k_bptt=None):
        """Update runtime loss-controlled inference knobs from the train config."""
        if infer_backprop is not None:
            self.infer_backprop = bool(infer_backprop)
        if k_bptt is not None:
            self.k_bptt = int(k_bptt)

    # ---- inference: replaces the amortized encoder -------------------------

    def _recon_energy(self, param, x_img, reduce_sum=False, deterministic=False):
        """0.5 * ||x - decode(code(eta))||^2. code = reparameterized sample (for
        the inference gradient) or the posterior mean (deterministic=True, for
        recon diagnostics — removes resampling noise so recon_gain is clean).
        reduce_sum -> scalar; else per-example (N,)."""
        z = self.head.to_code(param) if deterministic else self.head.sample(param)
        y = self.decoder(z)
        e = 0.5 * (x_img - y).pow(2)
        return e.sum() if reduce_sum else e.flatten(1).sum(1)

    def _infer_loop(self, param, x_img, energy_fn, fisher_fn=None):
        """Run inner inference, optionally differentiating the final k_bptt steps."""
        velocity = torch.zeros_like(param) if self.infer_momentum > 0.0 else None
        if not self.infer_backprop:
            for _ in range(self.k_inner):
                with torch.inference_mode(False), torch.enable_grad():
                    p = param.detach().clone().requires_grad_(True)
                    g = torch.autograd.grad(energy_fn(p), p)[0]
                    P = fisher_fn(p) if fisher_fn is not None else None
                param, velocity = self._inner_step(
                    param, g, velocity, preconditioner=P, detach_grad=True
                )
            return param

        k_bptt = max(0, min(int(self.k_bptt), int(self.k_inner)))
        n_detached = self.k_inner - k_bptt
        for _ in range(n_detached):
            with torch.inference_mode(False), torch.enable_grad():
                p = param.detach().clone().requires_grad_(True)
                g = torch.autograd.grad(energy_fn(p), p)[0]
                P = fisher_fn(p) if fisher_fn is not None else None
            param, velocity = self._inner_step(
                param, g, velocity, preconditioner=P, detach_grad=True
            )

        if n_detached > 0:
            with torch.inference_mode(False):
                param = param.detach().clone().requires_grad_(True)
            velocity = velocity.detach() if velocity is not None else None
        for _ in range(k_bptt):
            with torch.inference_mode(False), torch.enable_grad():
                if not param.requires_grad:
                    param = param.detach().clone().requires_grad_(True)
                g = torch.autograd.grad(energy_fn(param), param, create_graph=True)[0]
                P = fisher_fn(param) if fisher_fn is not None else None
                param, velocity = self._inner_step(
                    param, g, velocity, preconditioner=P, detach_grad=False
                )
        return param

    def _infer_one_frame(self, x_img, init_param, return_recon=False):
        """Iterative inference for one frame, stepping the natural parameter eta
        to DESCEND reconstruction error (param <- param - lr * dR/deta). First-
        order / detached inner steps when ``infer_backprop`` is false, or a
        differentiable K-step unroll when it is true.
        x_img: (N, C, H, W). init_param: (N, P). Returns posterior param (N, P);
        if return_recon, also (R0, RK) per-example recon at eta^(0) and eta^(K)."""
        init_param = self.head.clamp_param(init_param)
        r0 = self._recon_energy(init_param, x_img, deterministic=True).detach() if return_recon else None
        param = init_param
        def fisher_fn(p):
            p_kl = torch.zeros_like(p)
            decoder_damping = torch.ones_like(p)
            return p_kl + decoder_damping

        param = self._infer_loop(
            param,
            x_img,
            lambda p: self._recon_energy(p, x_img, reduce_sum=True),
            fisher_fn=fisher_fn,
        )
        if return_recon:
            rK = self._recon_energy(param, x_img, deterministic=True).detach()
            return param, r0, rK
        return param

    def encode(self, info, init_params=None, return_diag=False):
        """FOND inference over a sequence. Sets info['emb'] (B,T,P) = posterior
        natural parameter, and info['act_emb']. `init_params` (B,T,P) optionally
        overrides the per-frame inference init; defaults to the learned prior
        (scheme B). return_diag also stashes recon_init/recon_final and the init
        param (for correction_norm / recon_gain)."""
        pixels = info["pixels"].float()
        B, T = pixels.shape[:2]
        flat = rearrange(pixels, "b t ... -> (b t) ...")
        if init_params is not None:
            init = rearrange(init_params, "b t d -> (b t) d")
        elif self.infer_init == "static_prior":
            init = self.prior_param.expand(B * T, -1)
        else:
            raise NotImplementedError(
                "encode() is the parallel static-prior path (scheme B). For "
                "predictive_prior (scheme A) use filter_sequence() — the sequential "
                "online-filtering loop. See IMPLEMENTED_OBJECTIVE.md")
        init = self.head.clamp_param(init)
        out = self._infer_one_frame(flat, init, return_recon=return_diag)
        if return_diag:
            post, r0, rK = out
            info["recon_init"] = r0.mean()
            info["recon_final"] = rK.mean()
            info["infer_init_param"] = rearrange(init, "(b t) d -> b t d", b=B)
        else:
            post = out
        post = self.projector(post)
        info["emb"] = rearrange(post, "(b t) d -> b t d", b=B)
        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])
        return info

    # ---- scheme A: online variational filtering ----------------------------

    def _infer_online(self, x_img, prior, beta, infer_objective, return_diag=False):
        """One frame of online VI. Inference is initialized AND priored at the
        predictive prior. The inner energy is

            free_energy : F(eta) = R(eta; x) + beta * KL(q_eta || q_prior)
            recon_only  : F(eta) = R(eta; x)                       (no prior pull)

        descended by inner inference steps. With
        K=0 the posterior == prior (zero_step). Returns posterior (N,P); if
        return_diag also per-example (R0, RK, F0, FK, KL_K) at eta^(0)=prior and
        eta^(K)."""
        use_kl = (infer_objective == "free_energy" and beta > 0)
        prior = self.head.clamp_param(prior)
        param = prior
        diag = None
        if return_diag:
            r0 = self._recon_energy(prior, x_img, deterministic=True).detach()   # (N,)
            f0 = r0.clone()   # KL(prior||prior) = 0 at eta^(0)
        def energy_fn(p):
            energy = self._recon_energy(p, x_img, reduce_sum=True)
            if use_kl:
                energy = energy + beta * self.head.kl_energy(p, prior)
            return energy

        def fisher_fn(p):
            p_kl = beta * self.head.fisher_metric(p) if use_kl else torch.zeros_like(p)
            decoder_damping = torch.ones_like(p)
            return p_kl + decoder_damping

        param = self._infer_loop(param, x_img, energy_fn, fisher_fn=fisher_fn)
        if return_diag:
            with torch.no_grad():
                rK = self._recon_energy(param, x_img, deterministic=True)        # (N,)
                klK = self.head.kl_perexample(param, prior)                       # (N,)
                fK = rK + (beta * klK if use_kl else 0.0)
            diag = {"R0": r0, "RK": rK, "F0": f0, "FK": fK, "KL_K": klK}
            return param, diag
        return param

    def filter_sequence(self, batch, history_size, beta=1.0,
                        infer_objective="free_energy", return_diag=True):
        """Sequential online-VI filter (scheme A). For each frame t:
          eta_hat_t = static prior (t=0) or predictor(eta_{<t window}, a_{<t})
          eta_t     = infer(x_t, init=prior=eta_hat_t, objective)
          eta_prev  = stopgrad(eta_t)
        The predictor-context buffer is detached so past inference is not
        backpropagated through. For ``infer_backprop=False`` the current prior is
        detached too, reproducing the old first-order ablation.
        Returns info with emb (posteriors, B,T,P) and pred_hat (predictions, B,T,P)."""
        pixels = batch["pixels"].float()
        B, T = pixels.shape[:2]
        act_emb = self.action_encoder(batch["action"])         # (B,T,P)
        HS = history_size
        etas, ehats = [], []
        buffer = []                                            # detached posteriors
        d_acc = {"R0": [], "RK": [], "F0": [], "FK": [], "KL_K": []} if return_diag else None
        for t in range(T):
            x_t = pixels[:, t]
            if t == 0:
                ehat = self.prior_param.expand(B, -1)          # static learned prior (has grad)
            else:
                h = min(t, HS)
                hist = torch.stack(buffer[-h:], dim=1)         # (B,h,P) detached
                act_h = act_emb[:, t - h:t]                    # (B,h,P) actions a_{t-h..t-1}
                ehat = self.predict(hist, act_h)[:, -1]        # (B,P) prediction of frame t
            prior = ehat if self.infer_backprop else ehat.detach()
            out = self._infer_online(x_t, prior, beta, infer_objective, return_diag=return_diag)
            if return_diag:
                eta, d = out
                for k in d_acc:
                    d_acc[k].append(d[k])
            else:
                eta = out
            etas.append(eta)
            ehats.append(ehat)
            buffer.append(eta.detach())
        info = {"emb": torch.stack(etas, dim=1), "pred_hat": torch.stack(ehats, dim=1),
                "act_emb": act_emb}
        if return_diag:
            info["recon_init"] = torch.stack(d_acc["R0"]).mean()
            info["recon_final"] = torch.stack(d_acc["RK"]).mean()
            info["F_init"] = torch.stack(d_acc["F0"]).mean()
            info["F_final"] = torch.stack(d_acc["FK"]).mean()
            info["infer_kl_final"] = torch.stack(d_acc["KL_K"]).mean()
        return info

    @torch.no_grad()
    def action_prior_report(self, batch, eta, history_size, beta):
        """Innovation / action-conditioning diagnostics for online filtering.

        D_pred in scheme A is an INNOVATION size (eta_t is inferred from eta_hat_t),
        not an independent-target error. So a small D_pred is ambiguous. This scores
        the PREDICTIVE PRIOR itself against the actual next observation, and asks
        whether the TRUE action gives a better prior than a shuffled action or a
        no-op (predict-no-change). Scored frames t>=1 (t=0 has no predictor prior).

        R_t(eta) = 0.5||x_t - decode(mean(eta))||^2 (deterministic). F_prior == R_prior
        because KL(q_eta_hat || q_eta_hat) = 0. eta is the (detached) posteriors from
        filter_sequence; the posterior history feeding the predictor is the same one
        used during filtering."""
        head = self.head
        pixels = batch["pixels"].float()
        B, T, P = eta.shape
        HS = history_size
        act_true = self.action_encoder(batch["action"])
        act_shuf = self.action_encoder(torch.roll(batch["action"], 1, dims=0))   # break action<->obs link

        def Rscore(param, x):                              # per-example -> mean scalar
            return self._recon_energy(param, x, deterministic=True).mean()

        acc = {k: [] for k in ("R_prior_true", "R_prior_shuffle", "R_prior_noop",
                               "R_post", "innovation_kl", "correction_norm")}
        for t in range(1, T):
            h = min(t, HS)
            hist = eta[:, t - h:t]
            x_t = pixels[:, t]
            ehat_true = self.predict(hist, act_true[:, t - h:t])[:, -1]
            ehat_shuf = self.predict(hist, act_shuf[:, t - h:t])[:, -1]
            ehat_noop = eta[:, t - 1]                       # predict-no-change prior
            eta_t = eta[:, t]
            acc["R_prior_true"].append(Rscore(ehat_true, x_t))
            acc["R_prior_shuffle"].append(Rscore(ehat_shuf, x_t))
            acc["R_prior_noop"].append(Rscore(ehat_noop, x_t))
            acc["R_post"].append(Rscore(eta_t, x_t))
            acc["innovation_kl"].append(head.kl_perexample(eta_t, ehat_true).mean())
            acc["correction_norm"].append((eta_t - ehat_true).norm(dim=-1).mean())
        m = {k: torch.stack(v).mean().item() for k, v in acc.items()}
        # free energies: prior FE == its recon (self-KL=0); posterior adds innovation
        m["F_prior_true"] = m["R_prior_true"]
        m["F_prior_shuffle"] = m["R_prior_shuffle"]
        m["F_prior"] = m["R_prior_true"]
        m["F_post"] = m["R_post"] + beta * m["innovation_kl"]
        m["R_prior"] = m["R_prior_true"]
        # action gains (positive => true action helps explain the real next obs).
        # F gains equal R gains for the prior (self-KL=0) but logged for completeness.
        m["action_gain_R"] = m["R_prior_shuffle"] - m["R_prior_true"]
        m["action_gain_F"] = m["F_prior_shuffle"] - m["F_prior_true"]
        m["action_gain_vs_noop"] = m["R_prior_noop"] - m["R_prior_true"]
        return m

    def decode(self, emb):
        """Observation model on a param sequence (B,T,P) -> images (B,T,C,hw,hw)."""
        B = emb.size(0)
        flat = rearrange(emb, "b t d -> (b t) d")
        z = self.head.sample(flat)
        img = self.decoder(z)
        return rearrange(img, "(b t) c h w -> b t c h w", b=B)

    # ---- predictor: UNCHANGED from jepa.JEPA -------------------------------

    def predict(self, emb, act_emb):
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        return rearrange(preds, "(b t) d -> b t d", b=emb.size(0))


    # ---- planning / evaluation --------------------------------------------

    def rollout(self, info, action_sequence, history_size: int = 3):
        """Autoregressive latent rollout for planning, matching JEPA.rollout.

        FOND encodes the initial observation history by inner inference, then uses
        the shared predictor/action_encoder path for candidate action rollouts.
        """
        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        init = self.encode(init)
        emb = info["emb"] = init["emb"].unsqueeze(1).expand(B, S, -1, -1)

        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        HS = history_size
        for t in range(n_steps):
            act_emb = self.action_encoder(act)
            pred_emb = self.predict(emb[:, -HS:], act_emb[:, -HS:])[:, -1:]
            emb = torch.cat([emb, pred_emb], dim=1)
            act = torch.cat([act, act_future[:, t:t + 1]], dim=1)

        act_emb = self.action_encoder(act)
        pred_emb = self.predict(emb[:, -HS:], act_emb[:, -HS:])[:, -1:]
        emb = torch.cat([emb, pred_emb], dim=1)

        info["predicted_emb"] = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        return info

    def criterion(self, info_dict: dict):
        """Terminal planning cost in the latent family natural-parameter space."""
        pred = info_dict["predicted_emb"][..., -1, :]  # (B,S,P)
        goal = info_dict["goal_emb"][..., -1:, :].expand_as(info_dict["predicted_emb"])[..., -1, :]
        flat_goal = rearrange(goal, "b s p -> (b s) p")
        flat_pred = rearrange(pred, "b s p -> (b s) p")
        if hasattr(self.head, "kl_perexample"):
            cost = self.head.kl_perexample(flat_goal, flat_pred)
        else:
            cost = (flat_goal - flat_pred).pow(2).sum(-1)
        return rearrange(cost, "(b s) -> b s", b=pred.size(0), s=pred.size(1))

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """Cost of action candidates for stable_worldmodel planning eval."""
        assert "goal" in info_dict, "goal not in info_dict"

        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        decoder_requires_grad = [p.requires_grad for p in self.decoder.parameters()]
        self.decoder.requires_grad_(True)
        try:
            goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
            goal["pixels"] = goal["goal"]
            for k in list(info_dict.keys()):
                if k.startswith("goal_"):
                    goal[k[len("goal_"):]] = goal.pop(k)
            goal.pop("action")
            goal = self.encode(goal)

            info_dict["goal_emb"] = goal["emb"]
            info_dict = self.rollout(info_dict, action_candidates)
            return self.criterion(info_dict)
        finally:
            for param, requires_grad in zip(self.decoder.parameters(), decoder_requires_grad):
                param.requires_grad_(requires_grad)


# ----------------------------------------------------------------------------
# Training forward (variants 3-6) — the ONE new probabilistic forward.
# Replaces lejepa_forward; structurally identical except (1) target = stop-grad
# inference posterior, (2) pred_loss = head.pred_term (KL or Fisher quad),
# (3) reconstruction anchor instead of SIGReg. The predictor call is byte-for-
# byte the LeWM call. Family/loss are pure config switches.
# ----------------------------------------------------------------------------

def _recon_target(pixels, img_hw):
    """Low-res [0,1] PushT target for the reconstruction anchor (corrections C2).
    Expects `pixels` already in [0,1]; downsamples to img_hw if needed.
    NOTE: the real training transform must supply [0,1] frames (NOT the ViT's
    ImageNet-normalized tensor) — wired in the training-integration stage."""
    B, T = pixels.shape[:2]
    flat = rearrange(pixels.float(), "b t c h w -> (b t) c h w")
    if flat.shape[-1] != img_hw:
        flat = F.interpolate(flat, size=(img_hw, img_hw), mode="bilinear", align_corners=False)
    return rearrange(flat, "(b t) c h w -> b t c h w", b=B)


def vijepa_forward(self, batch, stage, cfg):
    """Variational JEPA forward. Branches on cfg.loss.target_scheme:
       static_vi_target  (scheme B) : parallel static-prior inference target
       online_filtering  (scheme A) : sequential online-VI filter (main experiment)
    cfg.loss carries: pred_loss ('exact_kl'|'quadratic_fisher'), beta, and for
    scheme A: infer_objective ('recon_only'|'free_energy')."""
    if cfg.loss.get("target_scheme", "static_vi_target") == "online_filtering":
        return filter_forward(self, batch, stage, cfg)

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    beta = float(cfg.loss.beta)
    self.model.beta = beta
    loss_form = cfg.loss.get("pred_loss", "exact_kl")
    log_diag = cfg.loss.get("log_diag", True)
    head = self.model.head
    self.model._set_runtime_inference(
        infer_backprop=cfg.loss.get("infer_backprop", True),
        k_bptt=cfg.loss.get("k_bptt", self.model.k_inner),
    )

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    # inference posterior (natural parameter) for the whole sequence
    output = self.model.encode(batch, return_diag=log_diag)
    emb = output["emb"]                 # (B, T, P)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    pred_emb = self.model.predict(ctx_emb, ctx_act)        # UNCHANGED predictor call

    # target = stop-grad posterior, shifted by one (pred_emb[:,t] predicts emb[:,t+1])
    tgt_emb = emb[:, n_preds:].detach()

    # (1) predictive term: D_pred(posterior || prediction) per (family, loss)
    output["pred_loss"] = head.pred_term(tgt_emb, pred_emb, loss_form, detach_metric=True)

    # (2) reconstruction anchor (anti-collapse data term)
    recon = self.model.decode(emb)
    target = _recon_target(batch["pixels"], self.model.img_hw)
    assert target.min() >= -1e-3 and target.max() <= 1 + 1e-3, "recon target not in [0,1]"
    output["recon_loss"] = F.mse_loss(recon, target)

    output["loss"] = output["recon_loss"] + beta * output["pred_loss"]

    # ---- pre-flight diagnostics (no grad) ----------------------------------
    if log_diag:
        with torch.no_grad():
            # §5.3 approximation validity: BOTH exact KL and the quad, every step
            output["kl_exact"] = head.kl_exact(tgt_emb, pred_emb)
            output["fisher_quad"] = head.fisher_quad(tgt_emb, pred_emb)
            output["exact_quad_ratio"] = output["kl_exact"] / (output["fisher_quad"] + 1e-8)
            # §5.2 predictive-degeneracy: predict-no-change baseline
            noop_prior = emb[:, : emb.size(1) - n_preds].detach()
            output["pred_noop"] = head.pred_term(tgt_emb, noop_prior, loss_form, detach_metric=True)
            output["noop_gap"] = output["pred_noop"] - output["pred_loss"]
            output["noop_ratio"] = output["pred_loss"] / (output["pred_noop"] + 1e-8)
            # correction_nontriviality: how far inference moved from its init
            # (scheme B: init = static prior). ~0 => posterior stuck at the prior.
            init_param = output["infer_init_param"]
            output["correction_norm"] = (emb - init_param).norm(dim=-1).mean()
            # recon_gain: did the K inference steps improve reconstruction?
            # POSITIVE => the observation model actually corrects the latent.
            output["recon_gain"] = output["recon_init"] - output["recon_final"]
            # saturation: rates / logvars pinned at the clamp bounds?
            output["sat_frac"] = torch.tensor(
                head.param_stats(emb)["sat_frac"], device=emb.device)

    log_keys = ("loss", "kl_exact", "fisher_quad", "exact_quad_ratio", "pred_noop",
                "noop_gap", "noop_ratio", "correction_norm", "recon_gain", "sat_frac")
    losses = {f"{stage}/{k}": v.detach() for k, v in output.items()
              if torch.is_tensor(v) and v.ndim == 0 and ("loss" in k or k in log_keys)}
    self.log_dict(losses, on_step=True, sync_dist=True)
    return output


def filter_forward(self, batch, stage, cfg):
    """Online-VI filtering forward (scheme A) — the main experiment.

    Posteriors are produced by sequential filtering (filter_sequence): each frame's
    inference is initialized AND priored at the predictor's prediction. The
    predictive loss trains the predictor toward the stop-grad posterior; the recon
    anchor trains the decoder. Diagnostics use eta_hat_t as the reference point
    (here eta^(0) == eta_hat_t by construction)."""
    beta = float(cfg.loss.beta)
    self.model.beta = beta
    loss_form = cfg.loss.get("pred_loss", "exact_kl")
    infer_objective = cfg.loss.get("infer_objective", "free_energy")
    log_diag = cfg.loss.get("log_diag", True)
    HS = cfg.history_size
    head = self.model.head
    self.model._set_runtime_inference(
        infer_backprop=cfg.loss.get("infer_backprop", True),
        k_bptt=cfg.loss.get("k_bptt", self.model.k_inner),
    )

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    info = self.model.filter_sequence(batch, HS, beta, infer_objective, return_diag=log_diag)
    eta = info["emb"]                # (B,T,P) posteriors (detached by construction)
    ehat = info["pred_hat"]          # (B,T,P) predictions (predictor + prior_param grad)
    output = dict(info)

    # (1) predictive term over all frames: D_pred(stopgrad(eta_t), eta_hat_t)
    output["pred_loss"] = head.pred_term(eta.detach(), ehat, loss_form, detach_metric=True)

    # (2) reconstruction anchor uses graphed eta when infer_backprop=True.
    recon = self.model.decode(eta)
    target = _recon_target(batch["pixels"], self.model.img_hw)
    assert target.min() >= -1e-3 and target.max() <= 1 + 1e-3, "recon target not in [0,1]"
    output["recon_loss"] = F.mse_loss(recon, target)

    output["loss"] = output["recon_loss"] + beta * output["pred_loss"]

    if log_diag:
        with torch.no_grad():
            eps = 1e-8
            # approximation validity (over all frames)
            output["kl_exact"] = head.kl_exact(eta, ehat)
            output["fisher_quad"] = head.fisher_quad(eta, ehat)
            output["exact_quad_ratio"] = output["kl_exact"] / (output["fisher_quad"] + eps)
            # no-op baseline on the t>=1 slice (eta_t vs eta_{t-1}); compare on the
            # same slice so the ratio is apples-to-apples.
            tgt_s, hat_s, prev_s = eta[:, 1:], ehat[:, 1:], eta[:, :-1]
            output["D_pred_shift"] = head.pred_term(tgt_s, hat_s, loss_form, detach_metric=True)
            output["pred_noop"] = head.pred_term(tgt_s, prev_s, loss_form, detach_metric=True)
            output["noop_ratio"] = output["D_pred_shift"] / (output["pred_noop"] + eps)
            output["noop_gap"] = output["pred_noop"] - output["D_pred_shift"]
            # correction / recon / free-energy gains (vs the predictive prior)
            output["correction_norm"] = (eta - ehat).norm(dim=-1).mean()
            output["recon_gain"] = output["recon_init"] - output["recon_final"]
            output["F_gain"] = output["F_init"] - output["F_final"]
            output["sat_frac"] = torch.tensor(head.param_stats(eta)["sat_frac"], device=eta.device)
            output["eta_absmean"] = eta.abs().mean()
            output["eta_std"] = eta.std()

    log_keys = ("loss", "kl_exact", "fisher_quad", "exact_quad_ratio", "pred_noop",
                "D_pred_shift", "noop_gap", "noop_ratio", "correction_norm",
                "recon_gain", "F_gain", "sat_frac", "eta_absmean", "eta_std")
    losses = {f"{stage}/{k}": v.detach() for k, v in output.items()
              if torch.is_tensor(v) and v.ndim == 0 and ("loss" in k or k in log_keys)}
    self.log_dict(losses, on_step=True, sync_dist=True)
    return output
