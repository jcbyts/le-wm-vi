"""Collapse-gate diagnostics (spec §5.1) — the primary go/no-go for every variant.

These operate on the committed latent `emb` (B, T, P) — the posterior natural
parameter for variants 3-6, or the encoder embedding for variants 1-2 — so they
are family-agnostic and shared across all six variants. A variant is "working"
only if it passes the gate; do not report planning numbers for a collapsed model.

Wire these into the eval loop BEFORE any long run (§8). Variant 1 (sigreg=0) is
the positive control: it MUST fail the gate, else the metrics are broken.
"""

import torch


@torch.no_grad()
def effective_rank(x):
    """Participation ratio of the covariance eigenvalues of x (N, D):
        PR = (sum λ)^2 / sum λ^2   in [1, D].
    A point-collapsed code -> PR≈1; isotropic full-rank -> PR≈D."""
    x = x.float()
    x = x - x.mean(0, keepdim=True)
    n = max(x.size(0) - 1, 1)
    cov = (x.t() @ x) / n
    ev = torch.linalg.eigvalsh(cov).clamp_min(0)
    s1 = ev.sum()
    s2 = ev.pow(2).sum()
    if s2 <= 0:
        return torch.tensor(1.0)
    return s1 * s1 / s2


@torch.no_grad()
def collapse_report(emb, rank_frac_floor=0.10, var_floor=1e-4):
    """Compute the three §5.1 numbers on a batch of committed latents (B, T, P).

    Returns a dict:
      eff_rank          : participation ratio across all (B*T) latent vectors
      eff_rank_frac     : eff_rank / P
      batch_var_median  : median over dims of per-dim variance across the batch
      temporal_var_mean : mean over (B,dims) of within-clip variance across T
      passed            : gate verdict (all three healthy)
    PASS (§5.1): eff_rank_frac > rank_frac_floor, batch_var_median > var_floor,
    temporal_var_mean > 0 (the code moves with the video)."""
    emb = emb.float()
    B, T, P = emb.shape
    flat = emb.reshape(B * T, P)

    er = effective_rank(flat)
    er_frac = (er / P).item()

    batch_var = flat.var(dim=0, unbiased=False)               # (P,)
    batch_var_median = batch_var.median().item()

    temporal_var = emb.var(dim=1, unbiased=False)             # (B, P) within-clip
    temporal_var_mean = temporal_var.mean().item()

    passed = (er_frac > rank_frac_floor
              and batch_var_median > var_floor
              and temporal_var_mean > 0.0)
    return {
        "eff_rank": er.item(),
        "eff_rank_frac": er_frac,
        "batch_var_median": batch_var_median,
        "temporal_var_mean": temporal_var_mean,
        "passed": bool(passed),
    }


if __name__ == "__main__":
    # Self-test: the gate must FAIL on a collapsed code and PASS on a healthy one.
    torch.manual_seed(0)
    B, T, P = 16, 8, 192

    # N = B*T must exceed P for a clean full-rank signal (else the empirical
    # covariance is rank-limited by sample count — Marchenko-Pastur, not collapse).
    healthy = torch.randn(64, T, P)
    rep_h = collapse_report(healthy)
    assert rep_h["passed"], rep_h
    assert rep_h["eff_rank_frac"] > 0.5, rep_h

    # low-rank collapse: healthy scale, but energy in only 2 of P directions.
    # Effective rank must catch this even though variance is fine.
    basis = torch.randn(2, P)
    coeff = torch.randn(64 * T, 2)
    lowrank = (coeff @ basis).reshape(64, T, P)
    rep_lr = collapse_report(lowrank)
    assert not rep_lr["passed"], rep_lr
    assert rep_lr["eff_rank_frac"] < 0.10, rep_lr

    # point collapse: near-constant code -> caught by the variance floor.
    point = torch.randn(1, 1, P).expand(64, T, P) + 1e-6 * torch.randn(64, T, P)
    rep_pt = collapse_report(point)
    assert not rep_pt["passed"], rep_pt
    assert rep_pt["batch_var_median"] < 1e-4, rep_pt

    # temporal collapse: varies across batch but frozen across time.
    frozen = torch.randn(64, 1, P).expand(64, T, P).contiguous()
    rep_f = collapse_report(frozen)
    assert rep_f["temporal_var_mean"] < 1e-8, rep_f

    print(f"[healthy]   rank_frac={rep_h['eff_rank_frac']:.3f}  var_med={rep_h['batch_var_median']:.3f}  "
          f"temp_var={rep_h['temporal_var_mean']:.3f}  passed={rep_h['passed']}")
    print(f"[low-rank]  rank_frac={rep_lr['eff_rank_frac']:.3f}  -> caught by effective rank, passed={rep_lr['passed']}")
    print(f"[point]     var_med={rep_pt['batch_var_median']:.2e}  -> caught by variance floor, passed={rep_pt['passed']}")
    print(f"[frozen]    temp_var={rep_f['temporal_var_mean']:.2e}  -> caught by temporal variance")
    print("\nCOLLAPSE-GATE SELF-TEST PASSED")
