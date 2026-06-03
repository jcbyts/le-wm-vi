"""Scheme-A (online_filtering) correctness tests — run BEFORE the pilot sweep.

  predictive_init_test   : eta_init_t == predictor(eta_{<t}, a_{<t}) for t>0, and
                           != the static prior (except by coincidence).
  action_dependence_test : shuffling actions changes eta_hat_t (and eta_t).
  zero_step_test         : K=0 => eta_t == eta_hat_t, recon_gain=correction_norm=D_pred=0.
  free_energy_descent_test: F(eta_K) < F(eta_0) for most examples (free_energy).
  beta_behavior_test     : high beta shrinks correction_norm; low beta grows it.
  saturation_test        : param_stats flags log-rate pinned at exp(5) (hard diagnostic).

Stub predictor/action_encoder (deterministic, no dropout) so the init recompute
matches exactly. No dataset / GPU required.
"""

import torch
from torch import nn
from model import FONDJEPA, ConvDecoder, filter_forward
from latent import make_head
import types


class StubPredictor(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.net = nn.Linear(p, p); self.cond = nn.Linear(p, p)
    def forward(self, x, c):
        return self.net(x) + self.cond(c)


class StubActionEncoder(nn.Module):
    def __init__(self, a, p):
        super().__init__()
        self.lin = nn.Linear(a, p)
    def forward(self, x):
        return self.lin(x.float())


def build(family, k_inner, D=192, img_hw=32, infer_lr=0.3):
    head = make_head(family)
    P = D * head.param_mult
    m = FONDJEPA(
        decoder=ConvDecoder(D, img_ch=3, img_hw=img_hw, grid=8),
        predictor=StubPredictor(P), action_encoder=StubActionEncoder(4, P),
        latent_dim=D, head=head, k_inner=k_inner, tau=0.2, infer_lr=infer_lr,
        infer_backprop=False, infer_init="predictive_prior", img_ch=3, img_hw=img_hw,
    )
    m.eval()
    return m


def _batch(B=8, T=4, hw=32):
    return {"pixels": torch.rand(B, T, 3, hw, hw), "action": torch.randn(B, T, 4)}


def test_predictive_init():
    for family in ["poisson", "gaussian"]:
        m = build(family, k_inner=3)
        b = _batch()
        info = m.filter_sequence(b, history_size=3, beta=1.0,
                                 infer_objective="free_energy", return_diag=False)
        eta, ehat = info["emb"], info["pred_hat"]
        act_emb = m.action_encoder(b["action"])
        prior = m.prior_param.expand(eta.size(0), -1)
        # t=0 prediction is the static prior
        assert torch.allclose(ehat[:, 0], prior, atol=1e-6), f"{family}: ehat_0 != static prior"
        for t in range(1, eta.size(1)):
            h = min(t, 3)
            recomputed = m.predict(eta[:, t - h:t], act_emb[:, t - h:t])[:, -1]
            assert torch.allclose(ehat[:, t], recomputed, atol=1e-5), f"{family}: ehat_{t} != predict(eta_<t)"
            assert not torch.allclose(ehat[:, t], prior, atol=1e-3), f"{family}: ehat_{t} == static prior (suspicious)"
    print("[predictive_init] eta_hat_t == predictor(eta_<t) for t>0, != static prior  OK")


def test_action_dependence():
    for family in ["poisson", "gaussian"]:
        m = build(family, k_inner=3)
        b = _batch()
        i1 = m.filter_sequence(b, 3, 1.0, "free_energy", return_diag=False)
        b2 = {"pixels": b["pixels"], "action": b["action"].flip(0)}   # shuffle actions across batch
        i2 = m.filter_sequence(b2, 3, 1.0, "free_energy", return_diag=False)
        dhat = (i1["pred_hat"][:, 1:] - i2["pred_hat"][:, 1:]).abs().mean().item()
        deta = (i1["emb"][:, 1:] - i2["emb"][:, 1:]).abs().mean().item()
        print(f"[{family}] action shuffle: d(eta_hat)={dhat:.4f}  d(eta_post)={deta:.4f}")
        assert dhat > 1e-4, f"{family}: eta_hat not action-dependent"
        assert deta > 1e-5, f"{family}: posterior not action/prior-dependent"


def test_zero_step():
    for family in ["poisson", "gaussian"]:
        m = build(family, k_inner=0)
        b = _batch()
        info = m.filter_sequence(b, 3, 1.0, "free_energy", return_diag=True)
        eta, ehat = info["emb"], info["pred_hat"]
        assert torch.allclose(eta, ehat, atol=1e-6), f"{family}: K=0 eta != eta_hat"
        dpred = m.head.pred_term(eta.detach(), ehat, "exact_kl").item()
        recon_gain = (info["recon_init"] - info["recon_final"]).item()
        corr = (eta - ehat).norm(dim=-1).mean().item()
        assert abs(dpred) < 1e-6 and abs(recon_gain) < 1e-6 and corr < 1e-6, \
            f"{family}: K=0 not trivial (dpred={dpred}, gain={recon_gain}, corr={corr})"
    print("[zero_step]   K=0 => eta==eta_hat, D_pred=recon_gain=correction_norm=0  OK")


def test_free_energy_descent():
    for family in ["poisson", "gaussian"]:
        m = build(family, k_inner=8, infer_lr=0.3)
        N = 128
        x = torch.rand(N, 3, 32, 32)
        prior = m.prior_param.expand(N, -1).detach()
        _, d = m._infer_online(x, prior, beta=1.0, infer_objective="free_energy", return_diag=True)
        frac = (d["FK"] < d["F0"]).float().mean().item()
        print(f"[{family}] FE descent: frac F_down={frac:.2f}  R0={d['R0'].mean():.3f} "
              f"RK={d['RK'].mean():.3f}  KL_K={d['KL_K'].mean():.3f}  "
              f"F_gain={(d['F0']-d['FK']).mean():.4f}")
        assert frac > 0.5, f"{family}: free energy not descending for most examples"


def test_beta_behavior():
    """The KL-to-prior term must resist moving away from the prior: free_energy
    correction < recon_only correction, and growing beta (in the stable regime)
    shrinks the correction monotonically. (At extreme beta a fixed step size
    overshoots — the high-curvature failure mode in spec §6 — so we test moderate
    beta at a small, stable infer_lr.)"""
    for family in ["poisson", "gaussian"]:
        m = build(family, k_inner=8, infer_lr=0.1)
        N = 128
        x = torch.rand(N, 3, 32, 32)
        prior = m.prior_param.expand(N, -1).detach()
        def corr(beta, obj):
            p, _ = m._infer_online(x, prior, beta=beta, infer_objective=obj, return_diag=True)
            return (p - prior).norm(dim=-1).mean().item()
        c_recon = corr(0.0, "recon_only")
        cs = {b: corr(b, "free_energy") for b in [0.1, 1.0, 3.0]}
        print(f"[{family}] correction: recon_only={c_recon:.4f}  "
              f"FE(beta) " + " ".join(f"{b}->{cs[b]:.4f}" for b in [0.1, 1.0, 3.0]))
        assert cs[3.0] < c_recon, f"{family}: free_energy did not resist (KL-to-prior inactive?)"
        assert cs[3.0] < cs[0.1], f"{family}: higher beta did not shrink correction"


def test_inner_step_preconditions_poisson_gradient():
    m = build("poisson", k_inner=1, D=64, img_hw=32, infer_lr=1.0)
    param = torch.full((1, 64), 2.0)
    grad = torch.exp(param) * torch.full_like(param, 3.0)
    fisher = m.head.fisher_metric(param)
    out, _ = m._inner_step(param, grad, None, preconditioner=fisher, detach_grad=True)
    expected = m.head.clamp_param(param - grad / (fisher + 1e-6))
    assert torch.allclose(out, expected)
    assert torch.allclose(out, torch.full_like(param, -1.0), atol=1e-5)
    print("[preconditioner] Poisson Euclidean gradient divided by Fisher metric  OK")



def test_saturation_diagnostic():
    """param_stats must flag a log-rate pinned at exp(5) (LOG_HI). Hard diagnostic."""
    head = make_head("poisson")
    sat = torch.full((16, 192), 5.0)            # all at LOG_HI
    stats = head.param_stats(sat)
    assert stats["sat_frac"] > 0.99, stats
    assert abs(stats["rate_max"] - torch.tensor(5.0).exp().item()) < 1e-2, stats
    healthy = torch.zeros(16, 192)
    assert head.param_stats(healthy)["sat_frac"] < 1e-6
    print(f"[saturation]  flags log-rate@exp(5): sat_frac={stats['sat_frac']:.2f} "
          f"rate_max={stats['rate_max']:.2f}  OK")


if __name__ == "__main__":
    test_inner_step_preconditions_poisson_gradient()
    test_predictive_init()
    test_action_dependence()
    test_zero_step()
    test_free_energy_descent()
    test_beta_behavior()
    test_saturation_diagnostic()
    print("\nALL ONLINE-FILTERING (SCHEME A) TESTS PASSED")
