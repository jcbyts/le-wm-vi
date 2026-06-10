# Handoff: Marmoset BackImage LeWM / V1 Latent Evaluation

This bundle is source-code only. It intentionally excludes checkpoints, latents,
movies, PNGs, NPZ outputs, and the `outputs/` tree.

The code was developed inside `/home/tejas/le-wm-vi`, originally cloned from
`https://github.com/jcbyts/le-wm-vi`. As of 2026-06-09, Tejas's local checkout
was still on the old local `main` and was 3 commits behind Jake's `origin/main`.
Jake's latest `origin/main` had these extra commits:

- `1e59dd8 Add conv recurrent Poisson world model`
- `79594bd Add metabolic PoisWM SIGReg experiments`
- `e6dba74 Checkpoint current experiment state`

The local marmoset work consists of:

- new package: `marmo/`
- one optional patch: `patches/latent_poisson_clamp.patch`

The `latent.py` patch parameterizes the Poisson log-rate clamp and clamps
post/prior before Poisson KL/Fisher calculations. This is separate from the
Gaussian work and should be reviewed carefully because Jake is actively changing
the Poisson path.

## How To Apply This Bundle

Recommended integration:

```bash
cd /path/to/le-wm-vi
git fetch origin
git switch -c tejas/marmo-backimage origin/main

# From the unzipped bundle:
cp -a /path/to/bundle/marmo ./marmo

# Optional Poisson clamp patch:
git apply /path/to/bundle/patches/latent_poisson_clamp.patch
```

Then inspect:

```bash
git status --short
git diff --stat
git diff --check
```

Expected source footprint:

- `marmo/` is about 1.7 MB.
- No `outputs/`, checkpoints, images, videos, or pycache files should be staged.

## Scientific Goal

We are adapting Jake's LeWM/FOND/JEPA-style world-model code to marmoset
free-viewing data from `backimage.dset`.

Dataset:

- A head-fixed marmoset freely views static natural images.
- Eye position changes over time through fixations, drift/tremor, and saccades.
- The action input to the world model is the actual eye movement displacement.
- The primary data file is like:
  `/mnt/sata/YatesMarmoV1/processed/Allen_2022-04-13/datasets/backimage.dset`
- The broader Allen set includes multiple sessions under
  `/mnt/sata/YatesMarmoV1/processed/Allen_*/datasets/backimage.dset`.

World-model target:

- Train a Gaussian LeWM-style model on visual input and eye movement actions.
- Use an action-conditioned transition model to predict the next latent.
- Apply SIGReg to the latent/code.
- Do not optimize for pretty pixel reconstructions in the main path.
- Evaluate whether the learned/predicted latents help predict V1 spikes.

Main biological question:

- Do saccade-conditioned next-glimpse latents explain V1 activity?
- Later, compare Gaussian and Poisson latent priors once the Poisson path is
  ready.

Current practical focus:

- Gaussian case only.
- Best current V1 readout feature is `pred_hat + behavior`, not raw `code`.

## Important Data Interpretation

Do not assume `backimage.dset["stim"]` is simply a gaze-centered crop.

The digital-twin BackImage pipeline uses an RF-shifted ROI. For our V1-oriented
world model, the correct mode is:

```text
center_mode = dset
```

In this mode, level 0 is reconstructed to match the saved `backimage.dset`
stimulus crop. The implementation follows this logic:

```text
L0 ROI = int(dpi_pix) + metadata["roi_src"]
```

So the crop is offset relative to gaze by the V1 receptive-field ROI. This is
intentional. The cyan gaze dot will not necessarily sit in the center of the red
RF crop if the crop is showing the V1 RF-relative input rather than a pure
gaze-centered input.

Use `center_mode=gaze` only for behavior/gaze-centered controls.

## Validation Against backimage.dset

The first thing to run after integration is the dset crop validation:

```bash
cd /path/to/le-wm-vi
export PYTHONPATH=/path/to/le-wm-vi:/home/tejas/DataYatesV1:/home/tejas/VisionCore:$PYTHONPATH

python -m marmo.validate_backimage \
  --session Allen_2022-04-13 \
  --n 512
```

The validation reconstructs crops from the original displayed image and checks
that L0 matches the existing `backimage.dset["stim"]` path. This is the main
guard against using the wrong coordinate system.

Core implementation files:

- `marmo/backimage_sequences.py`
  - Loads `backimage.dset`.
  - Reconstructs screen/image crops from DataYatesV1.
  - Handles `center_mode=dset` vs `center_mode=gaze`.
  - Builds pyramid levels and screen controls.
  - Produces train/val sequence windows at 120 Hz.
- `marmo/validate_backimage.py`
  - Pixel-level validation of regenerated crops against `backimage.dset`.

## Sampling And Timebase

We targeted 120 Hz because the VisionCore digital-twin model is trained at
120 Hz for spike prediction.

Important conventions:

- Raw dset rows are downsampled with `downsample=2` when source is 240 Hz.
- `target_hz=120`.
- Spike counts (`robs`) should be summed across the raw rows in a bin.
- Eye position/action covariates can be sampled or averaged; the current main
  path uses mean covariates.
- Validity/downsample mode should usually require all raw rows in a bin to be
  valid: `validity_downsample_mode=all`.
- `dfs_mode=visioncore` follows the VisionCore-style validity mask and uses a
  missing threshold.

Readout lags:

- At 120 Hz, one bin is about 8.33 ms.
- Lag set `2,3,4` means about 16.7, 25.0, and 33.3 ms of lagged latent context.
- This range performed best for `pred_hat + behavior`.

## Visual Input / Pyramid Work

We explored multiple input schemes:

- raw crops at increasing RF-centered sizes
- Gaussian blurred pyramid
- Laplacian pyramid
- hybrid sharp L0 plus blurred/context levels
- L0-only RF crop
- optional full screen channel

Current key conclusion:

- The best V1 readout result came from L0-only RF crop with AlexNetV1 features.
- The full screen channel can improve world-model prediction but is less
  biologically clean because it is allocentric, while the ventral stream is
  retinotopic.
- Screen-channel saliency can show hot gray/background regions under
  `grad_x_input`; this is largely a baseline/attribution artifact. Integrated
  gradients with a gray baseline greatly reduces this artifact.

For biologically cleaner V1 comparisons, prefer retinotopic RF-centered inputs.

## Faithful Gaussian LeWM Path

This is the main Gaussian path we currently trust most.

It is "faithful" because it avoids pixel reconstruction and instead follows the
LeWM latent prediction setup:

```text
image crop(s) x_t
  -> encoder
  -> code / emb z_t
  -> action-conditioned transition model
  -> predicted next latent pred_hat

target branch:
image crop(s) x_{t+1}
  -> same encoder
  -> target latent

loss:
prediction loss(pred_hat, target) + SIGReg(code)
```

There is no decoder/reconstruction objective in the main Gaussian comparison.
The earlier reconstruction panels were useful only for debugging whether images
were wired in, not as the scientific endpoint.

Main files:

- `marmo/train_faithful_marmo.py`
  - CLI for Gaussian LeWM training on BackImage sequences.
- `marmo/faithful_train_utils.py`
  - Model components and forward/loss utilities.
- `marmo/extract_faithful_latents.py`
  - Extracts `code`, `eta`, `pred_hat`, `target`, spikes, actions, eyepos,
    row/trial/split metadata into NPZ.
- `marmo/run_faithful_variant_pipeline.py`
  - Orchestrates training, latent extraction, and readout search.

Useful baseline command shape:

```bash
python -m marmo.train_faithful_marmo \
  --session Allen_2022-04-13 \
  --center-mode dset \
  --crop-sizes 51 \
  --target-hz 120 \
  --encoder-kind alexnet_v1 \
  --neural-feature-index 2 \
  --neural-pool-hw 4 \
  --action-history 3 \
  --embed-dim 192 \
  --predictor-depth 6 \
  --predictor-heads 16 \
  --predictor-mlp-dim 2048 \
  --projector-hidden-dim 2048 \
  --projector-norm batchnorm \
  --sigreg-weight 0.3 \
  --lr 5e-5 \
  --weight-decay 1e-3 \
  --warmup-steps 500 \
  --max-steps 4000 \
  --batch-size 128 \
  --num-workers 4 \
  --precompute-pixels \
  --val-interval 1000 \
  --val-batches 16 \
  --save-interval 2000 \
  --seed 1002 \
  --outdir outputs/neural_encoder_search/alexnetv1_fov51_pool4_ah3_s1002_4000
```

That run name corresponds to:

- AlexNetV1 frontend
- RF L0 crop size 51
- pool 4
- action history 3
- seed 1002
- 4000 training steps

## Neural / BrainScore-Inspired Encoders

Jake suggested trying the "neural inspired" model from BrainScore. The code adds
these frontends:

- `conv`: small trainable convolutional encoder
- `alexnet_v1`: pretrained AlexNet early V1 layer proxy
- `voneblock`: VOneBlock frontend
- `vonealexnet`: VOneAlexNet early feature taps

File:

- `marmo/neuro_encoders.py`

Important interpretation:

- For `alexnet_v1`, the frontend is frozen by default.
- The AlexNet feature map is pooled and projected into the LeWM `embed_dim`.
- The LeWM transition model still predicts in the 192-dimensional projected
  latent space.
- So the latent is not the raw huge AlexNet feature map; it is a projected code
  derived from a frozen neural-inspired frontend.

This helped V1 prediction. The best result so far is from the AlexNetV1 L0-only
run.

## V1 Spike Readout

The readout asks whether the world-model latents predict V1 spikes on validation
data.

Main file:

- `marmo/train_latent_spike_readout.py`

It trains Poisson readouts from extracted latent NPZ files. It supports:

- `feature_key=code`
- `feature_key=pred_hat`
- `feature_key=target`
- `feature_key=none` or dummy controls
- `behavior_mode=visioncore`
- `behavior_mode=none`
- raw eye/action controls
- multiple lag sets
- linear and MLP readouts

The BPS metric mirrors VisionCore's `calc_poisson_bits_per_spike`.

Behavior covariates:

- `behavior_mode=visioncore` reconstructs the BackImage digital-twin style
  behavior basis: eye position plus temporal basis over eye velocity.
- This is a strong baseline and must be included in ablations.

Session handling:

- For multi-session latent files, V1 units are session-specific.
- Do not use one mixed readout across sessions unless doing a deliberate
  control.
- Use `--session-id` for per-session readout.

## Matched Readout Ablation Result

Best current matched ablation on:

```text
outputs/neural_encoder_search/alexnetv1_fov51_pool4_ah3_s1002_4000/
  Allen_2022-04-13_faithful_gaussian_dset
```

Those outputs are not included in this zip, but the key table is:

| Ablation | Val mean BPS | Delta vs behavior | Val median BPS | Val mean corr |
|---|---:|---:|---:|---:|
| behavior-only | 0.3041 | +0.0000 | 0.2355 | 0.1346 |
| code/emb-only | 0.1561 | -0.1480 | 0.1278 | 0.0913 |
| pred_hat-only | 0.1883 | -0.1158 | 0.1420 | 0.1037 |
| code/emb + behavior | 0.3802 | +0.0761 | 0.3281 | 0.1470 |
| pred_hat + behavior | 0.3928 | +0.0887 | 0.3398 | 0.1485 |

Interpretation:

- Behavior alone is strong.
- Raw `code/emb` alone is weak compared with behavior.
- `pred_hat` alone is better than `code/emb` alone.
- Latents add real predictive signal beyond behavior.
- `pred_hat + behavior` is the current best readout.
- The transition-predicted latent is more useful for V1 prediction than the raw
  current code in this setup.

## Exact Readout Commands Used For The Split

Latent-only broad grid:

```bash
python -m marmo.train_latent_spike_readout \
  --latents outputs/neural_encoder_search/alexnetv1_fov51_pool4_ah3_s1002_4000/Allen_2022-04-13_faithful_gaussian_dset/latents_stage.npz \
  --outdir outputs/neural_encoder_search/alexnetv1_fov51_pool4_ah3_s1002_4000/Allen_2022-04-13_faithful_gaussian_dset/readout_matched_latent_only_capacity \
  --feature-keys code,pred_hat \
  --lag-sets '3,4;3,4,5;2,3,4' \
  --archs mlp \
  --behavior-mode none \
  --mlp-hidden-dims '128,256' \
  --mlp-depths '2,3' \
  --dropouts '0.1,0.3,0.5' \
  --mlp-weight-decays '1e-5,3e-5,1e-4,3e-4,1e-3' \
  --epochs 80 \
  --patience 12 \
  --batch-size 1024 \
  --seed 1002 \
  --device cuda
```

Behavior-only was rerun with a zero-width dummy latent to force the same
latent-valid lag windows. That prevents an unfair sample-count mismatch.

Plus-behavior broad grid:

```bash
python -m marmo.train_latent_spike_readout \
  --latents outputs/neural_encoder_search/alexnetv1_fov51_pool4_ah3_s1002_4000/Allen_2022-04-13_faithful_gaussian_dset/latents_stage.npz \
  --outdir outputs/neural_encoder_search/alexnetv1_fov51_pool4_ah3_s1002_4000/Allen_2022-04-13_faithful_gaussian_dset/readout_capacity_search \
  --feature-keys code,pred_hat \
  --lag-sets '3,4;3,4,5;2,3,4' \
  --archs mlp \
  --behavior-mode visioncore \
  --include-eye \
  --include-action \
  --mlp-hidden-dims '128,256' \
  --mlp-depths '2,3' \
  --dropouts '0.1,0.3,0.5' \
  --mlp-weight-decays '1e-5,3e-5,1e-4,3e-4,1e-3' \
  --epochs 80 \
  --patience 12 \
  --batch-size 1024 \
  --seed 1002 \
  --device cuda
```

## Visualization / Movies

The requested movie format is implemented in:

- `marmo/make_faithful_saliency_movie.py`
- `marmo/make_backimage_latent_movie.py`
- `marmo/saliency_utils.py`
- `marmo/v1_saliency_utils.py`

The intended video panels:

- left: full static displayed image / screen canvas
- RF crop box and gaze dot overlay
- target/reconstruction or current/target crop diagnostics when requested
- right: observed V1 spike raster
- right: latent raster or latent-derived predicted activity
- right: V1 readout predicted rates
- right: eye-position traces
- saliency heatmaps and channel percentage readouts

V1 actual-data saliency:

- Computes gradients of the trained V1 readout's masked Poisson loss/rate with
  respect to the world-model input pixels through the latent extractor.
- Useful to ask which image level or image location affects neural prediction.

Important saliency caveat:

- `grad_x_input` can make gray background look hot because gray pixels are
  nonzero relative to black.
- Prefer integrated gradients with `--ig-baseline gray` for screen-channel
  interpretation.

## Diagnostics

Useful files:

- `marmo/diagnose_latent_constancy.py`
  - Measures whether latent activations are too constant during fixations.
  - Produces fixation/saccade/bout statistics.
- `marmo/debug_screen_saliency.py`
  - Tests whether screen-channel saliency is content, gray background, monitor
    background, or letterbox artifact.
- `marmo/diagnose_faithful_run.py`
  - General integrity checks for crops, padding, saliency artifacts, and readout
    ablations.
- `marmo/summarize_faithful_variant_search.py`
  - Summarizes variant searches.
- `marmo/summarize_faithful_saliency.py`
  - Summarizes saliency outputs.

Key diagnostic conclusion:

- Earlier "hot gray" saliency in the full screen channel was mostly a
  `grad_x_input` baseline artifact.
- Integrated gradients with gray baseline reduced gray/background attribution
  substantially.
- Separately, the allocentric screen channel can still help prediction, which
  means it may be a shortcut/control rather than a clean V1-retinotopic input.

## Multi-Session / All Allen Notes

The code can train a single global world model on all Allen sessions:

```bash
python -m marmo.train_faithful_marmo \
  --sessions all-allen \
  ...
```

Important:

- The world model can be global across sessions because it sees images/actions.
- V1 readouts should be per-session because units differ across sessions.
- Extracted latent files include session metadata when using multi-session
  paths.
- `train_latent_spike_readout.py` has `--session-id` for per-session readouts.

## Poisson Work

We paused active Poisson training because Jake said the Poisson version still
needs sparsity-related fixes.

The bundle still includes older Poisson/FOND/amortized support:

- `marmo/train_fond_marmo.py`
- `marmo/fond_train_utils.py`
- `marmo/train_amortized_marmo.py`
- `marmo/amortized_train_utils.py`
- `marmo/extract_amortized_latents.py`

However, current empirical conclusions should be treated as Gaussian-only.

The optional `latent.py` patch:

- adds `PoissonHead(log_lo, log_hi)`
- clamps Poisson sample/to_code/KL/Fisher paths with instance-level bounds
- extends `make_head(..., poisson_log_lo=..., poisson_log_hi=...)`

Because Jake's latest repo now has more Poisson work, review this patch manually
before applying.

## Files In `marmo/`

High-level grouping:

- Data/crops:
  - `backimage_sequences.py`
  - `validate_backimage.py`
- Gaussian faithful LeWM:
  - `train_faithful_marmo.py`
  - `faithful_train_utils.py`
  - `extract_faithful_latents.py`
  - `run_faithful_variant_pipeline.py`
- Neural-inspired encoders:
  - `neuro_encoders.py`
- V1 readout:
  - `train_latent_spike_readout.py`
  - `predict_latent_spike_readout.py`
- Movies/saliency:
  - `make_faithful_saliency_movie.py`
  - `make_backimage_latent_movie.py`
  - `saliency_utils.py`
  - `v1_saliency_utils.py`
- Diagnostics:
  - `diagnose_faithful_run.py`
  - `diagnose_latent_constancy.py`
  - `debug_screen_saliency.py`
  - `summarize_faithful_saliency.py`
  - `summarize_faithful_variant_search.py`
- Older/alternate model paths:
  - `train_fond_marmo.py`
  - `fond_train_utils.py`
  - `train_amortized_marmo.py`
  - `amortized_train_utils.py`
  - `extract_latents.py`
  - `extract_amortized_latents.py`
  - `visualize_latents.py`

## Suggested Next Steps For Jake / LLM

1. Apply `marmo/` onto latest `origin/main`.
2. Decide whether to apply or skip `patches/latent_poisson_clamp.patch`.
3. Run `python -m marmo.validate_backimage --session Allen_2022-04-13 --n 512`.
4. Smoke train Gaussian faithful path on a small subset.
5. Reproduce the AlexNetV1 L0-only run.
6. Extract latents.
7. Train V1 readout ablations:
   - behavior-only
   - code/emb-only
   - pred_hat-only
   - code/emb + behavior
   - pred_hat + behavior
8. Use `pred_hat + behavior` as the current best neural readout baseline.
9. If revisiting pyramids, compare V1 readout BPS rather than reconstruction
   appearance.
10. Once Poisson sparsity is fixed, repeat the same readout protocol for
    Gaussian vs Poisson with the exact same dataset splits and readout search.

## Things To Be Careful About

- Do not commit `outputs/`.
- Do not commit checkpoints (`*.pt`), latent files (`*.npz`), movies (`*.mp4`),
  or frame images.
- Validate coordinates before training. Wrong crop center can silently produce
  plausible-looking but scientifically wrong inputs.
- Use `center_mode=dset` for V1/RF experiments.
- Use `center_mode=gaze` only for gaze-centered controls.
- Treat full-screen input as a control/diagnostic, not the clean retinotopic
  model.
- Keep behavior-only baselines in all V1 readout comparisons.
- For multi-session readouts, keep session-specific readouts unless explicitly
  testing a mixed-session control.
