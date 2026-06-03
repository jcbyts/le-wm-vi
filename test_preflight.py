"""Pre-flight correctness tests for variants 3-6 — run BEFORE any longer run.

These catch the failure modes that would silently waste compute:
  eta_init_test          : eta^(0) equals the declared inference init (scheme B:
                           the static prior; NOT eta_hat_t). Verified with K=0.
  inner_update_sign_test : K inference steps REDUCE the recon anchor R(eta^K) <
                           R(eta^0) for most examples (catches an ascent sign bug).
  correction_nontriviality: ||eta_t - eta^(0)|| > 0 (posterior not stuck at prior).
  recon_gain_test        : R(eta^0) - R(eta^K) > 0 (observation model improves latent).

Uses random [0,1] images + a stub predictor — no dataset, no GPU required. The
sign/gain claims hold on arbitrary fixed targets because the inner loop is
gradient descent on R for that target.
"""

import torch
from torch import nn
from model import FONDJEPA, ConvDecoder
from latent import make_head


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


def build(family, k_inner, D=192, img_hw=32, infer_lr=1.0):
    head = make_head(family)
    P = D * head.param_mult
    return FONDJEPA(
        decoder=ConvDecoder(D, img_ch=3, img_hw=img_hw, grid=8),
        predictor=StubPredictor(P), action_encoder=StubActionEncoder(4, P),
        latent_dim=D, head=head, k_inner=k_inner, tau=0.2, infer_lr=infer_lr,
        img_ch=3, img_hw=img_hw,
    )


def test_eta_init():
    """With K=0, encode must return exactly the declared init (scheme B: prior)."""
    for family in ["poisson", "gaussian"]:
        m = build(family, k_inner=0)
        B, T = 3, 4
        info = m.encode({"pixels": torch.rand(B, T, 3, 32, 32)})
        emb = info["emb"]
        prior = m.prior_param.expand(B * T, -1).reshape(B, T, -1)
        assert torch.allclose(emb, prior, atol=1e-6), f"{family}: eta^0 != static prior"
        assert m.infer_init == "static_prior"
    print("[eta_init]   K=0 => eta^(0) == static learned prior (scheme B)  OK")


def _infer_recon(family, k_inner, infer_lr=1.0):
    torch.manual_seed(0)
    m = build(family, k_inner=k_inner, infer_lr=infer_lr)
    N = 64
    x = torch.rand(N, 3, 32, 32)
    init = m.prior_param.expand(N, -1)
    post, r0, rK = m._infer_one_frame(x, init, return_recon=True)
    return post, init, r0, rK


def test_inner_update_sign_and_gain():
    """K steps reduce recon for most examples AND in mean (recon_gain > 0)."""
    for family in ["poisson", "gaussian"]:
        # small step to avoid overshoot noise dominating the sign check
        post, init, r0, rK = _infer_recon(family, k_inner=4, infer_lr=1.0)
        frac_down = (rK < r0).float().mean().item()
        gain = (r0 - rK).mean().item()
        corr = (post - init).norm(dim=-1).mean().item()
        print(f"[{family}] K=4 lr=1.0: frac_recon_down={frac_down:.2f}  "
              f"recon_gain(mean)={gain:.4f}  correction_norm={corr:.4f}")
        assert frac_down > 0.5, f"{family}: inference did not reduce recon for most examples"
        assert gain > 0, f"{family}: recon_gain not positive (sign bug?)"
        assert corr > 1e-6, f"{family}: correction_norm ~0 (posterior stuck at prior)"


def test_more_K_helps_monotone_ish():
    """recon_gain should grow (not shrink) with more inference steps."""
    for family in ["poisson", "gaussian"]:
        gains = []
        for K in [1, 2, 4, 8]:
            _, _, r0, rK = _infer_recon(family, k_inner=K, infer_lr=1.0)
            gains.append((r0 - rK).mean().item())
        print(f"[{family}] recon_gain by K{[1,2,4,8]}: {[round(g,3) for g in gains]}")
        assert gains[-1] >= gains[0], f"{family}: more steps did not help"


if __name__ == "__main__":
    test_eta_init()
    test_inner_update_sign_and_gain()
    test_more_K_helps_monotone_ish()
    print("\nALL PRE-FLIGHT TESTS PASSED")
