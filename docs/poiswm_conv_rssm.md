# Conv RSSM Poisson World Model

This experiment tests the hypothesis that Poisson latents should live in a
high-dimensional nonnegative perceptual code, while control and planning should
use a compact recurrent state.

## Objective

The model uses a convolutional encoder to produce residual log-rates
`r = log(lambda / lambda0)`. These rates define the high-dimensional sparse
Poisson code. A compact deterministic state `z` is learned from that code and is
the state optimized by the planner.

Training uses:

```text
L = KL(Pois(lambda_t) || Pois(lambda_hat_t))
  + compact_loss_weight * ||z_t - z_hat_t||^2
  + beta * MetabolicSIGReg(r_t)
```

The transition loss is the exact Poisson KL in log-rate coordinates. The SIGReg
anchor still targets the maximum-entropy log-rate distribution induced by the
Poisson prior-KL budget, including the log-rate Jacobian. The high-dimensional
anchor is sketched with a fixed random projection buffer; it is not trainable,
so the model cannot satisfy the anchor by changing the measurement map.

## Architecture

- Images are encoded by a small nonnegative convolutional tower with Softplus
  activations.
- The high-dimensional rate code is a spatial map, currently 64 x 7 x 7 for
  224 px PushT frames.
- A compact state is produced by a learned projection from flattened log-rates.
- A GRU predicts compact states from previous compact states and action
  embeddings.
- A learned state-to-rate decoder maps predicted compact states back to the
  high-dimensional Poisson code, where the Poisson KL is applied.

By default, CEM planning optimizes compact-state MSE. The model can also score
candidate plans by decoded Poisson KL or a hybrid compact/Poisson cost.

## Initial Validation

The smoke test instantiated the model, ran a real PushT batch, and completed a
one-epoch two-batch training pass with W&B enabled. Initial diagnostics showed:

- finite exact Poisson KL
- no log-rate clamp saturation
- noncollapsed compact state effective rank
- stable rates near the metabolic target distribution
- successful checkpoint writing

The overnight sweep should primarily test whether this architecture fixes the
planning instability seen with ViT-based Poisson latents.
