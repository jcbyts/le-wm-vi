# PoisWM Metabolic SIGReg

This note records the current PoisWM metabolic-SIGReg implementation and the
PushT results from the June 2026 sweep. It is intended to make the negative
planning result reproducible without relying on chat history.

## Objective

`MetabolicSigRegPoisWM` represents each latent dimension as a residual log-rate

```text
r = log(lambda / lambda0),   lambda = lambda0 * exp(r)
```

with `r` clamped to `[-12, 5]` for numerical stability. The transition term is
the exact asymmetric Poisson KL,

```text
K_dyn = KL(Pois(lambda_target) || Pois(lambda_pred)).
```

For log-rate latents this is computed directly as

```text
lambda_target * (r_target - r_pred) - lambda_target + lambda_pred,
```

which is algebraically identical to the rate-space KL but avoids rate ratios and
logs. Planning uses the same exact Poisson KL in log-rate form by default
(`goal_cost=poisson_kl`).

The regularizer is SIGReg to the maximum-entropy rate distribution induced by a
Poisson prior-KL budget. With baseline `lambda0`, the per-unit metabolic/prior
KL is

```text
K0(lambda) = KL(Pois(lambda) || Pois(lambda0))
           = lambda * log(lambda / lambda0) - lambda + lambda0.
```

In residual log-rate coordinates, the one-dimensional target density is

```text
rho*(r) proportional to exp(r) * exp(-alpha * K0(lambda0 * exp(r))).
```

The `exp(r)` factor is the change-of-variables Jacobian from rates to log-rates.
The total loss is

```text
loss = K_dyn + beta * SIGReg(r, rho*)
```

and current metabolic sweeps use `beta=1.0`.

## Implementation Pointers

- `jepa.MetabolicSigRegPoisWM`: metabolic Poisson world model.
- `module.poisson_kl_log_rates`: exact KL in residual log-rate coordinates.
- `config/train/model/poiswm_metabolic_sigreg.yaml`: default training config.
- `probe_pusht_latents.py`: frozen ridge probes from latents to PushT state and
  proprio.
- `measure_latent_sparsity.py`: Vinje/Gallant-style sparsity diagnostics for
  embeddings, rates, and metabolic/prior-KL activity.

`PoisWM` now refers to the bounded-rate capacity-SIGReg variant. The older
log-rate Fisher approximation is kept as `LogRateFisherPoisWM` with config
`poiswm_lograte_fisher_sigreg.yaml`.

## PushT Planning Results

The clamped metabolic sweep used ViT-tiny, `embed_dim=192`, batch size 128,
`lambda0=1`, and alphas `1.5`, `3.0`, `4.0`. Full PushT evals below used the
same 50 sampled starts as the LeWM comparison (`eval.num_eval=50`,
`eval.eval_budget=50`).

| model | checkpoint | success |
|---|---:|---:|
| PoisWM metabolic `alpha=1.5` | epoch 3 | 10.0% |
| PoisWM metabolic `alpha=3.0` | epoch 4 | 10.0% |
| PoisWM metabolic `alpha=4.0` | epoch 3 | 12.0% |
| LeWM baseline | epoch 3 | 90.0% |

Interpretation: under the matched ViT-tiny/192 PushT setup, this PoisWM family
is much worse for planning than LeWM. The eval stack and CEM planner are not the
bottleneck, because LeWM succeeds on the same starts.

## Linear Decodability

Frozen ridge probes used the same random seed and 512 PushT snippets. For the
Poisson models, the best linear space was residual log-rate `r`; raw rates were
worse. The LeWM row uses the model embedding directly.

| model | checkpoint | state R2 mean | proprio R2 mean |
|---|---:|---:|---:|
| PoisWM metabolic `alpha=1.5` | epoch 2 | 0.518 | 0.319 |
| PoisWM metabolic `alpha=3.0` | epoch 3 | 0.517 | 0.346 |
| PoisWM metabolic `alpha=4.0` | epoch 2 | 0.506 | 0.305 |
| LeWM baseline | epoch 3 | 0.664 | 0.501 |

LeWM makes the main PushT coordinates much more linearly available by epoch 3.
This matches the planning gap and suggests the Poisson code is not providing a
control-friendly geometry for CEM, even when it is non-collapsed.

## Sparsity / Neuron-Like Activity

The Poisson rates are positive, so literal zeros are not expected. The more
biologically meaningful sparse quantity here is the per-unit prior-KL energy
`K0(lambda)`, which is zero at baseline `lambda=lambda0` and grows with
stimulus-specific metabolic/information cost.

Vinje/Gallant-style sparseness is computed as

```text
S = (1 - mean(x)^2 / mean(x^2)) / (1 - 1/n)
```

where `0` is dense/equal activity and `1` is sparse.

| model | rate VG pop | prior-KL VG pop | top 5% prior-KL share | rate > 2x lambda0 | rate > 4x lambda0 |
|---|---:|---:|---:|---:|---:|
| PoisWM metabolic `alpha=1.5` e3 | 0.652 | 0.911 | 0.606 | 0.271 | 0.104 |
| PoisWM metabolic `alpha=3.0` e4 | 0.463 | 0.869 | 0.510 | 0.206 | 0.046 |
| PoisWM metabolic `alpha=4.0` e3 | 0.271 | 0.821 | 0.435 | 0.106 | 0.008 |
| LeWM baseline e3 abs embedding | n/a | n/a | 0.151 | n/a | n/a |

Conclusion: the metabolic PoisWM does produce a sparse, nonnegative,
neuron-like code in the sense of heavy-tailed firing rates and sparse
information-bearing deviations from baseline. The failure on PushT is therefore
not a failure to produce sparse neural-style activity; it is a failure to produce
planning-effective geometry in this simple low-dimensional control task.

## Caveats

- ViT-tiny/192 is matched to the LeWM comparison, but Poisson latents may need a
  different architecture, head, normalization, or planner metric.
- PushT is a very low-dimensional, smooth control problem embedded in 192 dims;
  Gaussian Euclidean latents are especially well suited to this structure.
- Exact Poisson KL is probabilistically clean, but CEM only uses it to rank
  sampled action sequences. A control metric can fail even when the model is a
  coherent Poisson world model.

The narrow conclusion is safe: this particular metabolic PoisWM setup is a bad
PushT planning model relative to LeWM. The broader conclusion that Poisson world
models are generally worse is not supported by these experiments alone.
