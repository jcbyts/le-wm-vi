# Fisher–JEPA Spec — Repo-Alignment Corrections

These override the matching sections of the original spec where it conflicts with
the actual `le-wm-vi` repo. Everything else in the original spec stands.

## C1. Transition model is the JEPA predictor; inference is independent (overrides §4)

The predictor (`module.ARPredictor`) is byte-identical to LeWM and is trained
exactly as in `lejepa_forward`: it regresses toward a **stop-gradient target
embedding**. For variants 3–6 the only changes vs `lejepa_forward` are:
  (a) the target embedding is produced by **decoder-inference** (`_infer_one_frame`)
      instead of the ViT, and
  (b) the predictive divergence is KL/Fisher instead of MSE.

Inference is computed **independently per frame, initialized from a static learned
prior** (the reference `FONDJEPA.encode` / `_infer_one_frame`), fully in parallel.
The predictor does NOT feed back into inference initialization. The original spec
§4 "initialize inference at the predictive prior f(η_{t-1},a_t)" (sequential
filtering) is **deferred to a later ablation**, not part of Stage 1. Rationale:
this keeps the predictor's training signal structurally identical to JEPA's
(regress to a fixed independent target) — the cleanest control.

## C2. Reconstruction target is PushT pixels in [0,1] at low resolution (overrides §1 data / §7 decoder)

We train on **PushT only**. Note that the existing pipeline
(`utils.get_img_preprocessor`) ImageNet-normalizes frames for the ViT — that is a
normalization constant choice for variants 1–2, NOT a different dataset, and it
must stay untouched for the validated baseline.

For the variants 3–6 reconstruction anchor:
- Target = PushT frame resized to **64×64, scaled to [0,1]** (a separate transform
  from the ViT path). Do not feed the ImageNet-normalized tensor to the decoder.
- `ConvDecoder` reconstructs at 64×64; keep the `Sigmoid()` head (matches [0,1]).
- Latent dim D = `embed_dim` = 192 (192 % (8·8) == 0 → 3 latent channels). The
  `emb` consumed by the predictor/planner stays (B,T,192) for poisson/deterministic
  and (B,T,384) param / 192-dim sample for gaussian.
- Keep the decoder deliberately low-capacity (§6 anti-domination). 64px is chosen
  to keep it cheap and to honor that control; revisit only if recon is too weak.

Variants 1–2 keep the existing 224px ImageNet-normalized ViT path unchanged.

## C3. Precision policy (refines §6)

The catastrophic-cancellation problem is **specific to the exact KL** (variants 3
& 5) and does **not** affect the Fisher quadratic (variants 4 & 6): the quadratic
forms `½Σ e^û(u−û)²` and `½Σ(μ−μ̂)²/σ̂²` build the small difference δ directly and
square it — no subtraction of two large nearly-equal quantities.

- Exact-KL variants (3, 5): compute the latent loss in **fp32**.
- Quadratic variants (4, 6): fp32 for uniformity (bf16 would also be safe).
- δ→0 unit tests (§8.3): **float64** (only place exact-vs-quad needs it).
- Bulk network may stay bf16 per the trainer config; cast the latent-loss math.

## C4. Two forward functions is the honest minimum (refines §0)

Variants 1–2 use the existing `lejepa_forward` (variant 1 = sigreg weight 0, a
config flag only). Variants 3–6 share **exactly one** new probabilistic forward,
parameterized by `LatentHead` (family) × `pred_loss` (exact_kl|quadratic_fisher).
The §0 "no second forward function" rule applies *within* the 3–6 group: those
four must be byte-identical except the head and the loss-form switch.

## C6. Gaussian Fisher quadratic — the spec's mu-only form is NOT the 2nd-order KL (overrides §2.1)

The spec §2.1 Gaussian quadratic `0.5·Σ(mu−mu_hat)²/sigma_hat²` is Fisher in the
**mu-coordinates only**. But the exact Gaussian KL also has a variance-mismatch
term that is **also O(delta²)** when logvar differs between posterior and prior.
So the mu-only form is NOT the full 2nd-order expansion of the exact KL — it drops
the curvature in the variance direction. This contradicts §2.2 ("the quadratic IS
the 2nd-order expansion") and §5.3 ("variant 4 vs 3 isolates ONLY the local
approximation"): with the mu-only form, variant 4 differs from variant 3 by the
local approximation AND a dropped variance penalty.

The genuine 2nd-order (full-Fisher) Gaussian quadratic is
```
0.5·Σ (mu − mu_hat)²/sigma_hat²  +  0.25·Σ (logvar − logvar_hat)²
```
(the 0.25·delta_lv² is the logvar Fisher block; verified to match the exact KL as
delta→0 in all directions — test_latent.py).

DECISION (user, confirmed): variant 4 uses the **spec §2.1 literal mu-only**
precision-weighted MSE (`GaussianHead(full_fisher=False)`, the default). CAVEAT TO
REPORT: variant 4 then differs from variant 3 by the local approximation AND a
dropped variance-curvature penalty, so the §5.3 exact-vs-quad ratio will NOT
converge to 1 when the posterior variance moves from the prior — that gap is the
omitted logvar term, not a bug. The full-Fisher form (`full_fisher=True`) is
retained as a faithful-2nd-order ablation. Poisson is a 1-parameter family and has
NO such ambiguity — its quadratic is the genuine 2nd-order expansion (test confirms
quadratic convergence, ratio→1).

## C5. Non-controlled differences to report (per §9)

- Variants 3–6 replace the ViT encoder with inference-through-decoder (intended,
  per original §4 — report it, do not hide it).
- `projector`/`pred_proj` are BatchNorm-MLPs in the JEPA baseline but Identity in
  FONDJEPA — note as an uncontrolled difference; consider matching if it matters.
- Gaussian predictor head is 2× wide (P=2D). Report D and param count both.
