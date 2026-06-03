# Online-filtering (scheme A) pilot — results

48 configs, 150 steps each, tiny PushT subset, `infer_lr=1.0`. Full per-config
data in `fond_sweep_online.csv`. **Do NOT pick a window from this — there is none
that is clean; the run diagnoses a hyperparameter problem to fix first.**

## Headline

| metric | count |
|---|---|
| CLEAN (all stable-window criteria) | **0 / 48** |
| SATURATED (rate/logvar at clamp) | 45 / 48 |
| diverged (corr_norm > 1e3, μ-explosion the clamp flag missed) | 2 / 48 |
| nonfinite | 0 / 48 |

Per-criterion pass counts (independently):
collapse_pass 17/48 · R_post<R_prior 34/48 · F_post<F_prior 17/48 ·
correction∈(1e-3,20) **8/48** · act_gain_R>0 27/48 · act_gain_vs_noop>0 15/48 ·
not-saturated 3/48.

## The dominant failure is inner-inference instability, not the scheme

Grouped by family / objective / K (aggregated over β_infer, recon_weight):

```
family   obj    K   sat/n  pass/n  corr_med   agNoop_max  agR>0
poisson  recon  1    2/2    1/2      14.74      -370.6     0/2
poisson  recon  4    2/2    0/2      23.37      -591.6     2/2
poisson  recon  8    2/2    0/2      26.74      -882.0     0/2
poisson  free   1    6/6    3/6      14.71      -366.5     0/6
poisson  free   4    6/6    6/6     111.45      +514.0     2/6
poisson  free   8    6/6    6/6     131.68      +623.2     3/6
gaussian recon  1    2/2    0/2      31.74      -195.9     2/2
gaussian recon  4    2/2    0/2      39.47      -164.5     1/2
gaussian recon  8    2/2    0/2      46.59      -191.6     2/2
gaussian free   1    6/6    0/6      31.74      -195.9     6/6
gaussian free   4    6/6    1/6      42.30       +46.4     5/6
gaussian free   8    3/6    0/6      53.68      +939.9     4/6
```

Saturation decomposition (which clamp):
- **Poisson**: 12/24 hit the HIGH rate clamp (e^5≈148), 12/24 hit the LOW clamp
  (rate→0). Log-rates **bifurcate to a bound** — the high-curvature instability
  the spec §6 predicted (Hessian ~ diag(rate)); `infer_lr=1.0` is too large.
- **Gaussian**: 19/24 hit the HIGH logvar clamp (σ²→148, variance **exploding**),
  only 3 hit the low clamp. Plus 2 configs diverged outright (μ → ~9e5).

**Proof it's the step size, not the scheme**: the scheme-A unit tests, which use
`infer_lr=0.3`, are stable (free-energy descends, correction bounded). The sweep
hardcodes `infer_lr=1.0`. Lowering the inner step size is the first fix.

## Signal that the structure is correct (when it doesn't saturate)

- **free_energy beats recon_only for prior usefulness.** Every `recon_only` row
  has `act_gain_vs_noop < 0` (the predictive prior is worse than predict-no-change),
  because without the KL-to-prior term inference pulls the posterior far from the
  prediction and the prior is never anchored. Only `free_energy` (K≥4) produces a
  prior that **beats no-op** (poisson K8 +623, K4 +514). This validates
  `free_energy` as the main inner objective and `recon_only` as the weak-correction
  ablation.
- **K matters.** K=1 free_energy never beats no-op (correction too weak); K≥4
  does. Consistent with "more inference steps ⇒ a more informative target."
- **collapse gate** passes in poisson free K≥4 (6/6) — the latent stays alive;
  the problem there is purely rate saturation, not collapse.

So: the world-model prior *is* learning action-conditioned structure (it beats
no-op) exactly in the regime (free_energy, K≥4) we'd hope — but the inner
inference saturates the natural parameters, disqualifying every config.

## Recommended next step (NOT probes/planning)

A focused **inner-stability sweep**, before anything else:
- `infer_lr ∈ {0.05, 0.1, 0.3}` (the prime suspect), optionally with inner-loop
  gradient clipping; consider a **lower infer_lr for Poisson** (high curvature).
- Restrict to the promising regime: `free_energy`, `K ∈ {4, 8}`, both families,
  `β_infer ∈ {0.1, 1.0}`, `recon_weight ∈ {1e-3, 1e-2}`.
- Keep all the action-prior diagnostics + the new divergence guard.
- Goal: find `infer_lr` where rates/logvars stay off the clamps AND
  `act_gain_vs_noop > 0` survives. Only then consider short probe runs.

Code change applied: `sweep_online.py` now flags μ-explosion / corr_norm>1e3 as a
hard failure (the 2 diverged Gaussian configs were not caught by the clamp-only
SATURATED flag).
