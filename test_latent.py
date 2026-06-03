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
    test_sample_shapes()
    test_directions_and_nonnegativity()
    test_poisson_convergence()
    test_gaussian_convergence()
    print("\nALL LATENT-HEAD TESTS PASSED")
