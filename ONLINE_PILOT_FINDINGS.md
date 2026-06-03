# Online-filtering (scheme A) pilot — findings

48 configs, tiny PushT, **150 training steps each**, infer_lr=1.0. Full data:
`fond_sweep_online.csv`. **Result: 0/48 CLEAN. Do not pick a window; do not
proceed to probes/planning.** The diagnostics localize four failure modes.

## Headline: 4 blockers (none yet a refutation of the method)

1. **Predictor is action-AGNOSTIC at 150 steps (the dominant blocker).**
   `|act_gain_R|` (true- vs shuffled-action prior, scored against the real next
   frame): median **0.015**, max 2.78, against `R_prior ~1133`. The action *does*
   change the prediction (verified separately by action_dependence_test, Δη̂≈0.39),
   but the changed prediction does **not** reconstruct the actual next frame any
   better — the predictor uses actions but hasn't yet learned the *correct*
   action→state mapping. This is undertraining, not an ignored input. Until this
   turns positive, no config can be CLEAN by construction.

2. **Inference DIVERGES at K≥4 with infer_lr=1.0 (high-curvature overshoot).**
   `recon_gain<0` (posterior reconstructs *worse* than the prior) in 0/16 at K=1,
   **6/16 at K=4, 8/16 at K=8**. `corr_norm`: K=1 ∈ [11,35]; K=4 up to 121; K=8 up
   to **8.5e5** (one Gaussian config blew to F_gain≈−3e13). Fixed step size is too
   large for the local curvature — exactly the §6 Poisson/high-rate concern.

3. **Gaussian collapses + logvar saturates.** collapse-gate pass: gaussian
   free_energy **1/18**, recon_only **0/6** (vs poisson free_energy 15/18). logvar
   hits the high clamp in 19/24 Gaussian configs (variance blow-up). Gaussian needs
   a lower inference lr and/or logvar regularization here.

4. **recon_only collapses without the KL-to-prior.** collapse pass: poisson
   recon_only 1/6, gaussian 0/6 — the reconstruction anchor *alone* collapses the
   latent (weak-anchor failure). free_energy (KL-to-prior active) keeps Poisson
   rank up (15/18), which is the one encouraging structural signal.

## Saturation (hard diagnostic — 45/48 flagged)
- Poisson: hi_sat>2% in 12/24 (rates → exp(5)), lo_sat>2% in 12/24 (rates → 0).
- Gaussian: hi_sat>2% in 19/24 (logvar → +5), lo_sat>2% in 3/24.

## Interpretation
The *structure* shows one positive sign (free_energy prevents Poisson collapse
where recon_only doesn't), but the pilot cannot assess the world model because
(a) the predictor is undertrained (no action alignment) and (b) inference is
numerically unstable at K≥4. Both are fixable knobs, not method failures. The
`act_gain_vs_noop>0` seen in some Poisson K≥4 rows is an ARTIFACT of diverged
inference (the no-op reference is a diverged previous posterior), not a learned
predictor — `act_gain_R≈0` confirms no real action structure yet.

## Recommended next step (needs a go-ahead — it's more compute)
Before any larger run, a SMALL second pilot with two changes:
1. **infer_lr 1.0 → ~0.2**, plus **inference gradient clipping**, so K=4/8 stop
   diverging (K=1 is already stable). Pair lower lr with higher K.
2. **Train much longer** (≥1500–3000 steps) on a few promising configs — the
   action-alignment signal (`act_gain_R>0`) cannot appear at 150 steps.
Reduced grid candidates: poisson free_energy {K=4,8}×{β=1}×{rw=1e-2}, gaussian
free_energy {K=4}×{β=1}×{rw=1e-2} (+ a logvar-reg variant). Keep the same table.
