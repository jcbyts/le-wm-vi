"""LatentHead: the one new structural piece for Fisher-JEPA.

A LatentHead makes the latent *family* a swappable component so the predictor,
decoder, training loop, planner, and eval are shared byte-for-byte across
variants. Each head operates on a per-step parameter vector of last-dim P:

    deterministic : P = D,   param = z
    gaussian      : P = 2D,  param = concat(mu, logvar)
    poisson       : P = D,   param = u (log-rate)

Every head provides (all operate on the LAST dim; batch/time dims pass through):

    sample(param)            -> z          reparameterized sample for the decoder
    pred_term(post, prior)   -> scalar     D_pred per the (family, loss) switch
    kl_exact(post, prior)    -> scalar     exact KL  (logged even in quad variants)
    fisher_quad(post, prior) -> scalar     quadratic form (logged even in exact)

Conventions fixed by the spec (do NOT vary):
  - KL direction: D_KL(posterior || prior), posterior is the FIRST argument.
  - The quadratic metric is evaluated at the REFERENCE (prior) parameters.
  - In inference losses, the weight in the quadratic is NOT stop-gradiented (the
    metric is part of the model); the posterior target detach is the caller's
    job, not the head's. Predictive losses may request detach_metric=True because
    their reference argument is the trainable prediction, not a fixed prior.

All reductions: sum over the parameter (feature) dim, mean over batch/time. The
heads are pure functions of (post, prior) tensors — detaching the target is done
by the training forward, exactly as in lejepa_forward.
"""

import torch
import torch.nn.functional as F

LOG_LO, LOG_HI = -12.0, 5.0       # log-rate clamp (Poisson)
LV_LO, LV_HI = -10.0, 5.0         # logvar clamp (Gaussian)


# ----------------------------------------------------------------------------
# Poisson machinery (ported verbatim from validated iP-VAE code)
# ----------------------------------------------------------------------------

def poisson_kl(log_post, log_prior, detach_metric=False):
    """Exact Poisson KL( Pois(e^log_post) || Pois(e^log_prior) ), elementwise."""
    log_post = log_post.clamp(LOG_LO, LOG_HI)
    log_prior = log_prior.clamp(LOG_LO, LOG_HI)
    log_prior_metric = log_prior.detach() if detach_metric else log_prior
    lam_post, lam_prior = torch.exp(log_post), torch.exp(log_prior_metric)
    return lam_post * (log_post - log_prior) - (lam_post - lam_prior)


def poisson_fisher_quad(log_post, log_prior, detach_metric=False):
    """Local (Fisher) reading: 1/2 * e^prior * (post - prior)^2, elementwise.
    Fisher of independent Poisson in log-rate coords is diag(e^u), evaluated at
    the reference (prior). Numerically robust at small delta (no cancellation)."""
    log_prior_ref = log_prior.detach() if detach_metric else log_prior
    log_prior_c = log_prior_ref.clamp(LOG_LO, LOG_HI)
    delta = log_post - log_prior
    return 0.5 * torch.exp(log_prior_c) * delta.pow(2)


def smoothstep(u):
    """EATcubic soft indicator: compact support, mean-unbiased for tau<=1."""
    w = ((u + 1.0) * 0.5).clamp(0.0, 1.0)
    return w * w * (3.0 - 2.0 * w)


class PoissonEAT:
    """Poisson reparameterized sample via exponential arrival times (cubic)."""
    MAX_EVENTS = 32

    def __init__(self, log_rate, tau=0.2):
        self.log_rate = log_rate
        self.rate = torch.exp(log_rate.clamp(LOG_LO, LOG_HI))
        self.tau = tau

    def _n(self):
        return min(int(max(self.rate.max().item(), 1.0) * 5) + 1, self.MAX_EVENTS)

    def rsample(self, hard=False):
        rate = self.rate.clamp_min(1e-6)
        inter = torch.distributions.Exponential(rate).rsample((self._n(),))
        times = torch.cumsum(inter, dim=0)
        if hard or self.tau == 0:
            cnt = (times < 1.0).float().sum(0)
            return rate - rate.detach() + cnt.detach()      # straight-through
        return smoothstep((1.0 - times) / self.tau).sum(0)


# ----------------------------------------------------------------------------
# Gaussian machinery
# ----------------------------------------------------------------------------

def _split_gaussian(param):
    """(..., 2D) -> (mu (...,D), logvar (...,D))."""
    mu, logvar = param.chunk(2, dim=-1)
    return mu, logvar


def gaussian_kl(post, prior, fixed_unit_variance=False, detach_metric=False):
    """Exact KL( N(mu,Sigma) || N(mu_hat,Sigma_hat) ), diagonal, elementwise.
    post/prior are (..., 2D) = concat(mu, logvar)."""
    mu, lv = _split_gaussian(post)
    mu_h, lv_h = _split_gaussian(prior)
    if fixed_unit_variance:
        lv = torch.zeros_like(lv)
        lv_h = torch.zeros_like(lv_h)
    lv = lv.clamp(LV_LO, LV_HI)
    lv_h = lv_h.clamp(LV_LO, LV_HI)
    lv_h_metric = lv_h.detach() if detach_metric else lv_h
    return 0.5 * (lv_h - lv + (torch.exp(lv) + (mu - mu_h).pow(2)) * torch.exp(-lv_h_metric) - 1.0)


def gaussian_fisher_quad(post, prior, include_var=True, detach_metric=False,
                         fixed_unit_variance=False):
    """Local (Fisher) quadratic form, elementwise.

    Fisher of a diagonal Gaussian in (mu, logvar) coords is diag(1/sigma^2, 1/2),
    so the genuine 2nd-order expansion of gaussian_kl at the reference (prior) is

        0.5 * (mu - mu_hat)^2 / sigma_hat^2   +   0.25 * (logvar - logvar_hat)^2
        \_______ mu (precision) block _______/    \____ logvar block ____/

    include_var=True  -> the FULL Fisher quadratic; matches gaussian_kl as
                         delta->0 in ALL directions (needed for the exact-vs-quad
                         comparison, spec corrections C6 / spec §2.2,§5.3).
    include_var=False -> the spec §2.1 literal "precision-weighted MSE" (mu only);
                         does NOT track the exact KL when the variance moves.

    The weights are NOT detached by default (the metric is part of the model).
    Predictive losses pass detach_metric=True so the trained prior/prediction only
    receives the residual gradient, not a curvature-shrinking side channel."""
    mu, lv = _split_gaussian(post)
    mu_h, lv_h = _split_gaussian(prior)
    if fixed_unit_variance:
        lv = torch.zeros_like(lv)
        lv_h = torch.zeros_like(lv_h)
    lv_h_ref = lv_h.detach() if detach_metric else lv_h
    lv_h_c = lv_h_ref.clamp(LV_LO, LV_HI)
    q = 0.5 * (mu - mu_h).pow(2) * torch.exp(-lv_h_c)
    if include_var:
        q = q + 0.25 * (lv.clamp(LV_LO, LV_HI) - lv_h).pow(2)
    return q


# ----------------------------------------------------------------------------
# LatentHead families
# ----------------------------------------------------------------------------

class LatentHead:
    """Base interface. `param_mult` = P/D for predictor/decoder dim bookkeeping."""
    family = None
    param_mult = 1

    def __init__(self, tau=0.2):
        self.tau = tau

    def sample(self, param):
        raise NotImplementedError

    def clamp_param(self, param):
        """Clamp the natural parameter to its safe range (spec §6). Applied
        after each inference step. Default: no clamp."""
        return param

    def param_stats(self, param, eps=1e-3):
        """Family-specific saturation / magnitude stats for the pre-flight
        diagnostics (detect rates/logvars pinned at the clamp bounds)."""
        return {}

    def kl_perexample(self, post, prior):
        """KL(q_post || q_prior) summed over features, per example. For 2D
        (N, P) inputs used inside online inference. Used in the free-energy
        objective and the descent diagnostics."""
        raise NotImplementedError

    def kl_energy(self, post, prior):
        """Scalar KL summed over all elements (batch+features) — the term added
        to the inference energy so its autograd gradient is not divided by N."""
        raise NotImplementedError

    def to_code(self, param):
        """Map a param vector to the D-dim code the decoder expects (no sampling).
        Used where a deterministic code is needed (e.g. the mean)."""
        raise NotImplementedError

    def pred_term(self, post, prior, loss, detach_metric=False):
        """D_pred per (family, loss). `loss` in {mse, exact_kl, quadratic_fisher}.
        Sum over feature dim, mean over batch/time."""
        raise NotImplementedError

    def kl_exact(self, post, prior):
        raise NotImplementedError

    def fisher_quad(self, post, prior):
        raise NotImplementedError

    def fisher_metric(self, param):
        """Analytical diagonal Fisher information I(eta)."""
        raise NotImplementedError


class DeterministicHead(LatentHead):
    family = "deterministic"
    param_mult = 1

    def sample(self, param):
        return param

    def to_code(self, param):
        return param

    def pred_term(self, post, prior, loss, detach_metric=False):
        assert loss == "mse", f"deterministic head supports loss=mse, got {loss}"
        return (post - prior).pow(2).sum(-1).mean()

    # No probabilistic divergence; return the mse so logging APIs stay total.
    def kl_exact(self, post, prior):
        return (post - prior).pow(2).sum(-1).mean()

    def fisher_quad(self, post, prior):
        return (post - prior).pow(2).sum(-1).mean()

    def fisher_metric(self, param):
        return torch.ones_like(param)


class GaussianHead(LatentHead):
    family = "gaussian"
    param_mult = 2

    def __init__(self, tau=0.2, full_fisher=False, fixed_unit_variance=False):
        super().__init__(tau=tau)
        # DECISION (user): variant 4 uses the spec §2.1 literal mu-only
        # precision-weighted MSE -> full_fisher=False is the default. Caveat
        # (corrections C6): variant 4 then differs from variant 3 by the local
        # approximation AND a dropped variance-curvature penalty; report it.
        # full_fisher=True is kept as the faithful-2nd-order ablation.
        self.full_fisher = full_fisher
        self.fixed_unit_variance = fixed_unit_variance

    def _with_unit_variance(self, param):
        if not self.fixed_unit_variance:
            return param
        mu, lv = _split_gaussian(param)
        return torch.cat([mu, torch.zeros_like(lv)], dim=-1)

    def sample(self, param):
        param = self._with_unit_variance(param)
        mu, lv = _split_gaussian(param)
        lv = lv.clamp(LV_LO, LV_HI)
        eps = torch.randn_like(mu)
        return mu + torch.exp(0.5 * lv) * eps

    def clamp_param(self, param):
        mu, lv = _split_gaussian(param)
        mu = mu.clamp(-10.0, 10.0)
        if self.fixed_unit_variance:
            return torch.cat([mu, torch.zeros_like(lv)], dim=-1)
        return torch.cat([mu, lv.clamp(LV_LO, LV_HI)], dim=-1)

    @torch.no_grad()
    def param_stats(self, param, eps=1e-3):
        mu, lv = _split_gaussian(param)
        hi = (lv >= LV_HI - eps).float().mean().item()
        lo = (lv <= LV_LO + eps).float().mean().item()
        return {"logvar_mean": lv.mean().item(), "logvar_min": lv.min().item(),
                "logvar_max": lv.max().item(), "sat_frac": hi + lo,
                "hi_sat_frac": hi, "lo_sat_frac": lo, "mu_absmean": mu.abs().mean().item()}

    def to_code(self, param):
        param = self._with_unit_variance(param)
        mu, _ = _split_gaussian(param)
        return mu

    def pred_term(self, post, prior, loss, detach_metric=False):
        post = self._with_unit_variance(post)
        prior = self._with_unit_variance(prior)
        if loss == "exact_kl":
            return gaussian_kl(
                post, prior, self.fixed_unit_variance, detach_metric=detach_metric
            ).sum(-1).mean()
        if loss == "quadratic_fisher":
            return gaussian_fisher_quad(
                post, prior, self.full_fisher, detach_metric, self.fixed_unit_variance
            ).sum(-1).mean()
        raise ValueError(f"gaussian head: unknown loss {loss}")

    def kl_exact(self, post, prior):
        return gaussian_kl(
            self._with_unit_variance(post), self._with_unit_variance(prior),
            self.fixed_unit_variance,
        ).sum(-1).mean()

    def kl_perexample(self, post, prior):
        return gaussian_kl(
            self._with_unit_variance(post), self._with_unit_variance(prior),
            self.fixed_unit_variance,
        ).sum(-1)

    def kl_energy(self, post, prior):
        return gaussian_kl(
            self._with_unit_variance(post), self._with_unit_variance(prior),
            self.fixed_unit_variance,
        ).sum()

    def fisher_quad(self, post, prior):
        return gaussian_fisher_quad(
            self._with_unit_variance(post), self._with_unit_variance(prior),
            self.full_fisher, fixed_unit_variance=self.fixed_unit_variance,
        ).sum(-1).mean()

    def fisher_metric(self, param):
        mu, lv = _split_gaussian(param)
        if self.fixed_unit_variance:
            return torch.ones_like(param)
        precision = torch.exp(-lv.clamp(LV_LO, LV_HI))
        return torch.cat([precision, torch.full_like(lv, 0.5)], dim=-1)


class PoissonHead(LatentHead):
    family = "poisson"
    param_mult = 1

    def sample(self, param):
        return PoissonEAT(param, tau=self.tau).rsample(hard=False)

    def clamp_param(self, param):
        return param.clamp(LOG_LO, LOG_HI)

    @torch.no_grad()
    def param_stats(self, param, eps=1e-3):
        lr = param.clamp(LOG_LO, LOG_HI)
        rate = lr.exp()
        flat = rate.flatten().float()
        hi = (lr >= LOG_HI - eps).float().mean().item()
        lo = (lr <= LOG_LO + eps).float().mean().item()
        return {"lograte_mean": lr.mean().item(), "rate_mean": rate.mean().item(),
                "rate_p95": torch.quantile(flat, 0.95).item(),
                "rate_p99": torch.quantile(flat, 0.99).item(),
                "rate_max": rate.max().item(), "sat_frac": hi + lo,
                "hi_sat_frac": hi, "lo_sat_frac": lo}

    def to_code(self, param):
        # deterministic code = the rate itself (clamped), differentiable in u
        return torch.exp(param.clamp(LOG_LO, LOG_HI))

    def pred_term(self, post, prior, loss, detach_metric=False):
        if loss == "exact_kl":
            return poisson_kl(post, prior, detach_metric=detach_metric).sum(-1).mean()
        if loss == "quadratic_fisher":
            return poisson_fisher_quad(post, prior, detach_metric=detach_metric).sum(-1).mean()
        raise ValueError(f"poisson head: unknown loss {loss}")

    def kl_exact(self, post, prior):
        return poisson_kl(post, prior).sum(-1).mean()

    def kl_perexample(self, post, prior):
        return poisson_kl(post, prior).sum(-1)

    def kl_energy(self, post, prior):
        return poisson_kl(post, prior).sum()

    def fisher_quad(self, post, prior):
        return poisson_fisher_quad(post, prior).sum(-1).mean()

    def fisher_metric(self, param):
        return torch.exp(param.clamp(LOG_LO, LOG_HI))


_HEADS = {
    "deterministic": DeterministicHead,
    "gaussian": GaussianHead,
    "poisson": PoissonHead,
}


def make_head(family, tau=0.2, full_fisher=False, fixed_unit_variance=False):
    if family not in _HEADS:
        raise ValueError(f"unknown latent family {family!r}; choose from {list(_HEADS)}")
    if family == "gaussian":
        return GaussianHead(
            tau=tau, full_fisher=full_fisher,
            fixed_unit_variance=fixed_unit_variance,
        )
    return _HEADS[family](tau=tau)


def variant_name(family, pred_loss, full_fisher=False):
    """Canonical report names (per user request). The mu-only Gaussian quadratic
    is a precision-weighted JEPA ablation, NOT the complete 2nd-order KL — it must
    NOT be called 'full Fisher'. Only full_fisher=True earns 'full_fisher_quad'."""
    if family == "gaussian":
        if pred_loss == "exact_kl":
            return "gaussian_exact_kl"
        return "gaussian_full_fisher_quad" if full_fisher else "gaussian_precision_mse"
    if family == "poisson":
        return "poisson_exact_kl" if pred_loss == "exact_kl" else "poisson_fisher_quad"
    return f"{family}_{pred_loss}"
