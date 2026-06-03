# IMPLEMENTED OBJECTIVE — variants 3–6 (FOND-JEPA)

This states the **exact equations actually implemented** in `model.py`
(`FONDJEPA`, `vijepa_forward`) and `latent.py`, so there is no ambiguity about
what the upcoming compute is testing. Where the implementation differs from the
original spec's idealized algorithm, that is called out explicitly.

Notation per transition `t` in a training window: previous posterior `η_{t-1}`,
action `a_t`, observation `x_t`. `η` is the family's natural parameter
(Poisson: log-rate `u`, dim `P=D`; Gaussian: `concat(μ, logσ²)`, dim `P=2D`).

> **TWO schemes.** `loss.target_scheme` selects which forward runs:
> **SCHEME A — online variational filtering** (`online_filtering`, the MAIN
> experiment; equations in the "SCHEME A" section below), and **SCHEME B —
> static-prior independent target** (`static_vi_target`, an ablation). Sections
> (a)/(c) below (predictive prior, loss form) are shared; section (b) below
> describes scheme B's inference — scheme A replaces it with the filtering loop.

---

## a. Predictive prior (the transition / world model)

```
η̂_t = f_θ(η_{ctx}, a_{ctx})        # ARPredictor, byte-identical to LeWM
```

`f_θ` is the **unchanged** `ARPredictor`: it consumes the context window of
posteriors and actions and outputs next-step predictions. `η̂_t` is the
predictor's prediction of the posterior at `t`. This is the only place the
"predictive prior" enters — it is a **target in the loss**, NOT an input to
inference (see (b)).

## b. Posterior target inference

```
η^(0) = π_φ                                   # STATIC learned prior (a single
                                              #   nn.Parameter, broadcast to all
                                              #   frames). NOT η̂_t.
for k in 0..K-1:                              # K = k_inner
    η^(k+1) = clamp( η^(k) − ρ · ∂R/∂η |_{η^(k)} )   # first-order, detached
η_t = η^(K)
```

where the reconstruction energy driving inference is

```
R(η; x) = ½ · || x − decode(sample(η)) ||²
```

`ρ = infer_lr`. The inner steps are **detached** (`g.detach()`), so `η_t` is
differentiable only w.r.t. `π_φ`, never the decoder (spec §2.3 routing). The
step is gradient **descent** on `R` (verified by `inner_update_sign_test`).

### Scheme classification — READ THIS

> **The implemented scheme is (B): "independent variational target JEPA."**
>
> `η_t` is inferred from `x_t` starting at a **static learned prior `π_φ`**, with
> **no dependence on the predictive prior `η̂_t`** and no dependence on `η_{t-1}`.
> The predictor is trained, separately, to regress its prediction `η̂_t` toward
> the (stop-grad) independently-inferred target `η_t` — exactly the structure of
> the LeWM JEPA loss, with KL/Fisher replacing MSE and inference-through-decoder
> replacing the ViT encoder.
>
> This is **NOT** scheme (A). Scheme (A) is now implemented separately (below)
> and is the MAIN experiment; this static-prior path is retained as an ablation.

---

## SCHEME A — online variational filtering (MAIN experiment)

Selected by `loss.target_scheme=online_filtering` (default in `fond.yaml`),
implemented by `FONDJEPA.filter_sequence` / `_infer_online` (`filter_forward`).
This is the spec's online VI. The sequence is processed **sequentially**:

```
eta_prev = (none)                                  # buffer of past posteriors
for t in 0..T-1:
    if t == 0:  eta_hat_t = pi_phi                 # static learned prior
    else:       eta_hat_t = predictor(eta_{t-HS..t-1}, a_{t-HS..t-1})[-1]   # windowed, == LeWM usage
    prior_t = eta_hat_t.detach()                   # detached inside inference
    eta_t = argmin_eta  R(eta; x_t)  +  beta_infer * KL(q_eta || q_{prior_t})   # K detached steps
            initialized at eta^(0) = prior_t       # predictive_prior init
    pred_loss_t = D_pred( sg[eta_t] , eta_hat_t )  # eta_hat_t NOT detached -> trains predictor
    eta_prev <- sg[eta_t]                           # detached posterior feeds next prediction
```

Inner objective (`infer_objective`):
- `free_energy` (MAIN): `F(eta) = R(eta;x) + beta_infer * KL(q_eta || q_prior)`.
- `recon_only`: `F(eta) = R(eta;x)`. With K=1 this is BONG-style weak correction;
  with K>1 call it **"prior-initialized observation correction"**, NOT full online VI.

Gradient routing: inference steps are detached (decoder gets no gradient from the
predictive term); the predictor prior is detached inside inference (no gradient
through target construction); the predictor receives gradient only via
`pred_loss_t` (target `sg[eta_t]` detached, prediction `eta_hat_t` not). The
decoder is trained by the reconstruction anchor `MSE(decode(sample(eta)), x)`.

Reference points (here `eta^(0) == eta_hat_t` by construction):
- `correction_norm = || eta_t − eta_hat_t ||` (observation correction of the prediction)
- `recon_gain = R(eta_hat_t) − R(eta_t)`,  `F_gain = F(eta^(0)) − F(eta^(K))`
- `D_noop = D_pred(eta_t, eta_{t-1})` (predict-no-change), `noop_ratio = D_pred/D_noop`.

Config: `loss.target_scheme` ∈ {online_filtering, static_vi_target},
`loss.infer_objective` ∈ {free_energy, recon_only}, `loss.beta_infer` (KL-to-prior
weight). `model.infer_init` only affects the static encode() path.

---

## SCHEME B — static-prior independent target (ablation)

Config switch (explicit, per request): `model.infer_init` ∈ {`prior`,
`predictive`}, default **`prior`** (scheme B). `eta_init_test` verifies
`η^(0)` matches the declared scheme.

### Consequence for the diagnostics (important)

Because `η^(0) = π_φ` (a static prior), NOT `η̂_t`, the natural reference point
for "did inference do anything" is the **inference init `π_φ`**, not `η̂_t`:

- `correction_norm = || η_t − η^(0) || = || η_t − π_φ ||`
  (how far observation-correction moved the latent from its static init;
  ≈0 ⇒ posterior stuck at the prior, predictive loss good for a trivial reason).
- `recon_gain = R(η^(0)) − R(η^(K))`
  (did the K inference steps improve reconstruction; should be **> 0**).

The distance `D_pred(η_t, η̂_t)` from posterior to the *predictor's* prediction
is exactly the **predictive term** below — it is not "the correction." Reports
label `correction_norm`/`recon_gain` against the inference init, and the
posterior-vs-prediction distance as `D_pred`. (If we later switch to scheme (A),
`η^(0)=η̂_t` and the two reference points coincide.)

## c. Training loss

```
L = recon_w · R̄(η_t)  +  β · D_pred( sg[η_t] , η̂_t )
```

- `R̄(η_t) = MSE( decode(sample(η_t)), x_t )` over the window (the recon anchor;
  target = PushT pixels in [0,1] at `img_hw`, corrections C2). `recon_w = recon_weight`.
- `sg[·]` = stop-gradient (the predictor regresses toward a fixed target; spec §2.3).
- `β = kl_weight`.
- `D_pred` per (family, loss_form):

| family | loss_form | `D_pred(post, prior)` implemented |
|---|---|---|
| poisson  | `exact_kl`         | `Σ poisson_kl(u_post, u_prior)` = `Σ [λ_p(u_p−u_q) − (λ_p−λ_q)]` |
| poisson  | `quadratic_fisher` | `Σ ½ e^{u_q} (u_p − u_q)²` |
| gaussian | `exact_kl`         | `Σ ½[lv_q − lv_p + (e^{lv_p}+(μ_p−μ_q)²)e^{−lv_q} − 1]` |
| gaussian | `quadratic_fisher` (full_fisher=False, default) | `Σ ½ (μ_p−μ_q)² e^{−lv_q}` — **μ-only precision-weighted MSE** |
| gaussian | `quadratic_fisher` (full_fisher=True)            | above `+ Σ ¼ (lv_p−lv_q)²` — full 2nd-order KL |

KL direction is `D_KL(posterior ‖ prior)` = `D_KL(q_{η_t} ‖ q_{η̂_t})`
(posterior first), fixed for both exact and quadratic.

---

## Variant naming for reports (per request)

| family / loss | report name | note |
|---|---|---|
| poisson exact_kl | `poisson_exact_kl` | |
| poisson quadratic_fisher | `poisson_fisher_quad` | 1-param family: quad IS the 2nd-order KL |
| gaussian exact_kl | `gaussian_exact_kl` | |
| gaussian quad, `full_fisher=False` | `gaussian_precision_mse` | μ-only; a precision-weighted JEPA **ablation**, NOT the complete 2nd-order KL — do not call it "full Fisher" |
| gaussian quad, `full_fisher=True` | `gaussian_full_fisher_quad` | the paper's clean "quadratic Fisher" claim; matches exact KL as δ→0 in all directions |

Variant 4 (spec-literal) = `gaussian_precision_mse`. The paper's quadratic-Fisher
claim corresponds to `gaussian_full_fisher_quad` (`full_fisher=True`).

## Held fixed across variants (control surface)
Predictor (`ARPredictor`), action encoder (`Embedder`), data (PushT), optimizer/
schedule, history/horizon, latent dim D. Variants 1–2 use the ViT encoder +
MSE + SIGReg (`lejepa_forward`); variants 3–6 use inference-through-decoder +
`D_pred` + recon anchor (`vijepa_forward`). The encoder difference is intended,
not a confound (original spec §4).
