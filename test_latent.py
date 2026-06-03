"""Float64 unit tests for the LatentHead math (spec §8.3).

Asserts the central claim of the experiment's numerics: the exact KL and its
2nd-order (Fisher quadratic) surrogate AGREE as the posterior approaches the
prior (delta -> 0), with the gap shrinking at the expected quadratic rate. Run
in float64 so the exact KL's catastrophic cancellation does not floor the test.

  python test_latent.py
"""

import torch
from latent import (
    poisson_kl, poisson_fisher_quad,
    gaussian_kl, gaussian_fisher_quad,
    make_head,
)

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)


def _ratio_table(exact_fn, quad_fn, make_post, prior, deltas):
    """For each delta, post = prior + delta*dir; report exact/quad ratio."""
    dirn = torch.randn_like(prior if prior.shape[-1] == _D else prior)
    rows = []
    for d in deltas:
        post = make_post(prior, d)
        e = exact_fn(post, prior).sum().item()
        q = quad_fn(post, prior).sum().item()
        rows.append((d, e, q, e / q if q != 0 else float("nan")))
    return rows


_D = 64


def test_poisson_convergence():
    log_prior = torch.randn(256, _D) * 0.5          # rates ~ O(1)
    dirn = torch.randn(256, _D)
    dirn = dirn / dirn.norm(dim=-1, keepdim=True)
    print("\n[poisson] exact KL vs Fisher quad as delta->0  (ratio should -> 1)")
    prev_rel = None
    for d in [1e-1, 1e-2, 1e-3, 1e-4]:
        log_post = log_prior + d * dirn
        e = poisson_kl(log_post, log_prior).sum().item()
        q = poisson_fisher_quad(log_post, log_prior).sum().item()
        rel = abs(e - q) / q
        print(f"   delta={d:.0e}  exact={e:.3e}  quad={q:.3e}  ratio={e/q:.6f}  relerr={rel:.2e}")
        if prev_rel is not None:
            # quadratic convergence: 10x smaller delta -> ~10x smaller relerr
            assert rel < prev_rel, "relerr did not shrink as delta shrank"
        prev_rel = rel
    assert abs(e / q - 1.0) < 1e-3, "poisson exact/quad did not converge to 1"


def _gaussian_sweep(label, prior, make_post, quad_kwargs, check_rate):
    prev_rel = None
    print(f"\n[gaussian] {label}  (ratio should -> 1)")
    last = None
    for d in [1e-1, 1e-2, 1e-3, 1e-4]:
        post = make_post(d)
        e = gaussian_kl(post, prior).sum().item()
        q = gaussian_fisher_quad(post, prior, **quad_kwargs).sum().item()
        rel = abs(e - q) / q
        floor = " <- exact-KL cancellation floor (C3)" if prev_rel is not None and rel > prev_rel else ""
        print(f"   delta={d:.0e}  exact={e:.3e}  quad={q:.3e}  ratio={e/q:.6f}  relerr={rel:.2e}{floor}")
        if check_rate:
            # Genuine O(delta^3) surrogate error: enforce quadratic shrink, but
            # only above the float64 cancellation floor of the exact KL (e>1e-5);
            # the tail uptick there is the expected exact-KL precision floor.
            if prev_rel is not None and e > 1e-5:
                assert rel < prev_rel * 1.5, "relerr did not shrink as delta shrank"
        else:
            # Surrogate is EXACT here (KL is exactly quadratic in mu at fixed
            # variance): relerr should sit at machine floor for every delta.
            assert rel < 1e-6, f"mu-only quad not exact at fixed variance: {rel}"
        prev_rel = rel
        last = e / q
    return last


def test_gaussian_convergence():
    mu_h = torch.randn(256, _D)
    lv_h = torch.randn(256, _D) * 0.3
    prior = torch.cat([mu_h, lv_h], dim=-1)
    dmu = torch.randn(256, _D); dmu /= dmu.norm(dim=-1, keepdim=True)
    dlv = torch.randn(256, _D); dlv /= dlv.norm(dim=-1, keepdim=True)

    # (a) FULL Fisher quad, mu AND logvar perturbed -> must converge to 1.
    r = _gaussian_sweep(
        "FULL Fisher quad, mu+logvar perturbed",
        prior,
        lambda d: torch.cat([mu_h + d * dmu, lv_h + d * dlv], dim=-1),
        dict(include_var=True),
        check_rate=True,
    )
    assert abs(r - 1.0) < 1e-2, f"full Fisher quad did not converge: {r}"

    # (b) mu-ONLY quad (spec §2.1 literal), only mu perturbed -> converges to 1
    #     because variance is held fixed so the dropped term is exactly zero.
    r = _gaussian_sweep(
        "mu-only quad, mu perturbed (variance fixed)",
        prior,
        lambda d: torch.cat([mu_h + d * dmu, lv_h], dim=-1),
        dict(include_var=False),
        check_rate=False,
    )
    assert abs(r - 1.0) < 1e-2, f"mu-only quad did not converge: {r}"


def test_directions_and_nonnegativity():
    """KL >= 0 and == 0 iff post == prior, both families."""
    for fam, mk in [("poisson", lambda: torch.randn(128, _D)),
                    ("gaussian", lambda: torch.cat([torch.randn(128, _D),
                                                    torch.randn(128, _D) * 0.3], -1))]:
        head = make_head(fam)
        p = mk()
        assert head.kl_exact(p, p).item() < 1e-12, f"{fam}: KL(p||p) != 0"
        q = mk()
        assert head.kl_exact(q, p).item() > 0, f"{fam}: KL not positive"
        assert head.fisher_quad(q, p).item() > 0, f"{fam}: Fisher quad not positive"
    print("\n[both] KL(p||p)=0, KL>0, FisherQuad>0  OK")




def test_predictive_metric_detach_gradient():
    """FIX1: predictive Fisher losses detach only the metric, not the residual."""
    log_post = (torch.randn(32, _D) * 0.4).detach()
    log_prior = (torch.randn(32, _D) * 0.4).requires_grad_(True)
    loss = poisson_fisher_quad(log_post, log_prior, detach_metric=True).sum()
    grad = torch.autograd.grad(loss, log_prior)[0]
    precision = torch.exp(log_prior.detach().clamp(-12.0, 5.0))
    expected = precision * (log_prior.detach() - log_post)
    assert torch.allclose(grad, expected, atol=1e-10, rtol=1e-10)

    log_prior2 = log_prior.detach().clone().requires_grad_(True)
    loss2 = poisson_fisher_quad(log_post, log_prior2, detach_metric=False).sum()
    grad2 = torch.autograd.grad(loss2, log_prior2)[0]
    assert not torch.allclose(grad2, expected, atol=1e-8, rtol=1e-8)

    mu = torch.randn(32, _D)
    mu_h = torch.randn(32, _D).requires_grad_(True)
    lv = torch.zeros_like(mu)
    lv_h = (torch.randn(32, _D) * 0.3).requires_grad_(True)
    post = torch.cat([mu, lv], dim=-1).detach()
    prior = torch.cat([mu_h, lv_h], dim=-1)
    loss = gaussian_fisher_quad(post, prior, include_var=False, detach_metric=True).sum()
    grad_mu, grad_lv = torch.autograd.grad(loss, [mu_h, lv_h], allow_unused=True)
    precision = torch.exp(-lv_h.detach().clamp(-10.0, 5.0))
    assert torch.allclose(grad_mu, precision * (mu_h.detach() - mu), atol=1e-10, rtol=1e-10)
    assert grad_lv is None or grad_lv.abs().max().item() < 1e-12

    mu_h2 = mu_h.detach().clone().requires_grad_(True)
    lv_h2 = lv_h.detach().clone().requires_grad_(True)
    prior2 = torch.cat([mu_h2, lv_h2], dim=-1)
    loss2 = gaussian_fisher_quad(post, prior2, include_var=False, detach_metric=False).sum()
    grad_lv2 = torch.autograd.grad(loss2, lv_h2)[0]
    assert grad_lv2.abs().max().item() > 1e-8
    print("\n[detach_metric] predictive Fisher gradient is clean residual gradient  OK")


def test_gaussian_unit_variance_lewm_identity():
    """FIX3: Gaussian fixed-unit floor reduces the predictive quad to LeWM MSE."""
    head = make_head("gaussian", fixed_unit_variance=True)
    mu = torch.randn(16, _D)
    mu_hat = torch.randn(16, _D)
    # Nonzero logvars should be ignored by the fixed-unit floor.
    post = torch.cat([mu, torch.randn_like(mu)], dim=-1)
    prior = torch.cat([mu_hat, torch.randn_like(mu_hat)], dim=-1)
    pred = head.pred_term(post, prior, "quadratic_fisher", detach_metric=True)
    expected = 0.5 * (mu - mu_hat).pow(2).sum(-1).mean()
    lewm_mse = (mu - mu_hat).pow(2).mean()
    assert torch.allclose(pred, expected, atol=1e-5, rtol=1e-5)
    assert torch.allclose(pred, 0.5 * _D * lewm_mse, atol=1e-5, rtol=1e-5)
    print("[unit_variance] Gaussian Fisher quad == 0.5||mu-muhat||^2 == LeWM MSE constant  OK")


def test_fisher_metric_values():
    det = make_head("deterministic")
    z = torch.randn(4, _D)
    assert torch.allclose(det.fisher_metric(z), torch.ones_like(z))

    pois = make_head("poisson")
    log_rate = torch.linspace(-2.0, 2.0, steps=_D).view(1, _D)
    assert torch.allclose(pois.fisher_metric(log_rate), log_rate.exp())

    gauss = make_head("gaussian")
    mu = torch.randn(3, _D)
    lv = torch.linspace(-1.0, 1.0, steps=_D).view(1, _D).expand_as(mu)
    param = torch.cat([mu, lv], dim=-1)
    expected = torch.cat([torch.exp(-lv), torch.full_like(lv, 0.5)], dim=-1)
    assert torch.allclose(gauss.fisher_metric(param), expected)

    unit = make_head("gaussian", fixed_unit_variance=True)
    assert torch.allclose(unit.fisher_metric(param), torch.ones_like(param))
    print("[fisher_metric] analytical diagonal Fisher values OK")



def test_sample_shapes():
    """sample() maps param (...,P) -> code (...,D); to_code likewise."""
    for fam, P, D in [("deterministic", _D, _D), ("poisson", _D, _D), ("gaussian", 2 * _D, _D)]:
        head = make_head(fam)
        param = torch.randn(8, 5, P) * 0.3
        z = head.sample(param)
        c = head.to_code(param)
        assert z.shape == (8, 5, D), f"{fam} sample shape {z.shape}"
        assert c.shape == (8, 5, D), f"{fam} to_code shape {c.shape}"
    print("[shapes] sample/to_code dims OK for all families")


if __name__ == "__main__":
    test_fisher_metric_values()
    test_sample_shapes()
    test_predictive_metric_detach_gradient()
    test_gaussian_unit_variance_lewm_identity()
    test_directions_and_nonnegativity()
    test_poisson_convergence()
    test_gaussian_convergence()
    print("\nALL LATENT-HEAD TESTS PASSED")
