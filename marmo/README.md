# Marmoset BackImage World-Model Path

Run these scripts from the `VisionCore` uv environment so the local torch/DataYates
dependencies are available:

```bash
cd /home/tejas/VisionCore
export PYTHONPATH=/home/tejas/le-wm-vi:/home/tejas/DataYatesV1:/home/tejas/VisionCore:$PYTHONPATH
```

Validate that regenerated level-0 crops match `backimage.dset`. This checks the
saved dataset ROI path, which is intentionally RF-offset relative to gaze:

```bash
uv run python /home/tejas/le-wm-vi/marmo/validate_backimage.py \
  --session Allen_2022-04-13 --n 512
```

Tiny smoke train, Poisson FOND:

```bash
uv run python /home/tejas/le-wm-vi/marmo/train_fond_marmo.py \
  --session Allen_2022-04-13 \
  --family poisson \
  --max-train-windows 256 \
  --max-val-windows 64 \
  --max-steps 20 \
  --batch-size 8
```

Gaussian comparison smoke:

```bash
uv run python /home/tejas/le-wm-vi/marmo/train_fond_marmo.py \
  --session Allen_2022-04-13 \
  --family gaussian \
  --max-train-windows 256 \
  --max-val-windows 64 \
  --max-steps 20 \
  --batch-size 8
```

Extract latents and make V1/latent figures:

```bash
CKPT=/home/tejas/le-wm-vi/outputs/marmo_fond/Allen_2022-04-13_poisson_exact_kl_dset/last.pt
uv run python /home/tejas/le-wm-vi/marmo/extract_latents.py --checkpoint "$CKPT"
uv run python /home/tejas/le-wm-vi/marmo/visualize_latents.py \
  --latents /home/tejas/le-wm-vi/outputs/marmo_fond/Allen_2022-04-13_poisson_exact_kl_dset/latents.npz
```

The foveated input is a grayscale pyramid resized to 64x64. Use
`--center-mode dset` for V1/RF runs: L0 is the exact RF-shifted ROI used by
`backimage.dset["stim"]`. The dset ROI is `int(dpi_pix) + metadata["roi_src"]`,
so it intentionally carries the V1/RF offset relative to gaze. Use
`--center-mode gaze` only for behavioral gaze-centered experiments.

## Faithful Gaussian LeWM Path

This is the clean Gaussian baseline: no decoder/reconstruction loss, just
Jake-style continuous latents, action-conditioned next-latent prediction, and
SIGReg. For V1/RF interpretation, the main biologically cleaner run should use
retinotopic RF-centered pyramid channels only: `51,101,201,401,801,1201`.
The older `screen` channel is useful as a diagnostic/control input, but it is an
allocentric full-display canvas rather than a retinotopic retinal channel.

```bash
python -m marmo.train_faithful_marmo \
  --center-mode dset \
  --crop-sizes 51,101,201,401,801,1201 \
  --max-steps 4000 \
  --batch-size 128 \
  --num-workers 4 \
  --precompute-pixels \
  --embed-dim 192 \
  --encoder-width 64 \
  --predictor-depth 6 \
  --predictor-heads 16 \
  --predictor-mlp-dim 2048 \
  --projector-hidden-dim 2048 \
  --projector-norm batchnorm \
  --sigreg-weight 0.3 \
  --lr 5e-5 \
  --weight-decay 1e-3 \
  --warmup-steps 500 \
  --val-interval 1000 \
  --val-batches 16 \
  --log-interval 100 \
  --save-interval 2000 \
  --outdir /home/tejas/le-wm-vi/outputs/marmo_faithful_gaussian_rf_retino_sig03
```

Current retinotopic checkpoint:

```bash
CKPT=/home/tejas/le-wm-vi/outputs/marmo_faithful_gaussian_rf_retino_sig03/Allen_2022-04-13_faithful_gaussian_dset/last.pt
```

Final validation preview from that run: `pred_loss=0.0450`,
`fixation_pred_loss=0.0217`, `saccade_pred_loss=0.2075`,
`code_collapse_eff_rank_frac=0.138`, `code_collapse_batch_var_median=0.924`,
and `action_gain=0.071`.

RF-wide screen-control checkpoint:

```bash
SCREEN_CKPT=/home/tejas/le-wm-vi/outputs/marmo_faithful_gaussian_rf_wide_cached_sig03/Allen_2022-04-13_faithful_gaussian_dset/last.pt
```

Final validation preview from the screen-control run: `pred_loss=0.0386`,
`fixation_pred_loss=0.0179`, `saccade_pred_loss=0.1919`,
`code_collapse_eff_rank_frac=0.139`, `code_collapse_batch_var_median=0.897`,
and `action_gain=0.073`. This run is slightly better, but it uses the
allocentric full-screen context channel.

Screen-channel diagnostic: the `screen` model can make gray/letterbox pixels
look hot under `grad_x_input`, because gray is nonzero (`127/255`) and the
method implicitly uses a black baseline. Use this diagnostic to separate image
content, monitor background, and letterbox attribution:

```bash
python -m marmo.debug_screen_saliency \
  --checkpoint "$SCREEN_CKPT" \
  --max-windows 1024 \
  --batch-size 16 \
  --num-workers 2 \
  --precompute-pixels \
  --saliency-mode pred_loss \
  --saliency-methods grad_x_input,grad,integrated_gradients \
  --ig-baseline gray \
  --outdir /home/tejas/le-wm-vi/outputs/marmo_faithful_gaussian_rf_wide_cached_sig03/Allen_2022-04-13_faithful_gaussian_dset/screen_saliency_debug_pred_loss_1024
```

In the 1024-window screen diagnostic, `grad_x_input` assigned about 52-55% of
L6 heatmap mass to gray/background pixels, roughly their area share. With
gray-baseline integrated gradients, gray/background mass dropped to about 7-11%,
so the hot gray bars are mostly a baseline/attribution artifact. Separately,
ablating the screen image content hurts prediction, so the screen channel is a
real allocentric context shortcut even though gray saliency itself is not good
evidence of useful gray content.

Extract latents and render the saliency movie. If the model has a `screen`
channel, prefer integrated gradients with a gray baseline for visual
interpretation:

```bash
python -m marmo.extract_faithful_latents \
  --checkpoint "$CKPT" \
  --max-windows-per-split 16384 \
  --batch-size 256 \
  --num-workers 4 \
  --precompute-pixels \
  --out /home/tejas/le-wm-vi/outputs/marmo_faithful_gaussian_rf_retino_sig03/Allen_2022-04-13_faithful_gaussian_dset/latents_16384.npz

python -m marmo.visualize_latents \
  --latents /home/tejas/le-wm-vi/outputs/marmo_faithful_gaussian_rf_retino_sig03/Allen_2022-04-13_faithful_gaussian_dset/latents_16384.npz \
  --outdir /home/tejas/le-wm-vi/outputs/marmo_faithful_gaussian_rf_retino_sig03/Allen_2022-04-13_faithful_gaussian_dset/latents_16384 \
  --target-hz 120

python -m marmo.make_faithful_saliency_movie \
  --checkpoint "$CKPT" \
  --trial-id 662 \
  --start-row 405662 \
  --duration-s 4 \
  --fps 30 \
  --max-render-frames 120 \
  --saliency-mode pred_loss \
  --saliency-method integrated_gradients \
  --ig-baseline gray \
  --saliency-source current \
  --out /home/tejas/le-wm-vi/outputs/marmo_faithful_gaussian_rf_retino_sig03/Allen_2022-04-13_faithful_gaussian_dset/faithful_gaussian_rf_retino_saliency_movie.mp4
```

The RF-wide movies report `gaze overlay source: eyepos->screen` and median
gaze-minus-RF-center of about `di=2 px, dj=23 px`, matching the RF offset in the
dset ROI.

## Amortized Encoder Path

FOND gives useful decoder/inference diagnostics, but SIGReg acts only indirectly
on inferred posteriors. The amortized path adds a trainable convolutional encoder
so the model follows `image -> encoder -> latent -> action-conditioned predictor
-> decoder`, and SIGReg directly shapes the encoder output.

The first healthy Allen 2022-04-13 Poisson run used:

```bash
uv run python /home/tejas/le-wm-vi/marmo/train_amortized_marmo.py \
  --session Allen_2022-04-13 \
  --family poisson \
  --pred-loss exact_kl \
  --center-mode dset \
  --crop-sizes 51,101,201 \
  --target-hz 120 \
  --embed-dim 64 \
  --encoder-width 32 \
  --beta 0.01 \
  --recon-weight 1.0 \
  --sigreg-weight 1.0 \
  --poisson-log-hi 2.0 \
  --batch-size 32 \
  --num-workers 2 \
  --max-train-windows 0 \
  --max-val-windows 0 \
  --max-steps 1000 \
  --outdir /home/tejas/le-wm-vi/outputs/marmo_amortized_all_4_13_poisson_loghi2_sigreg1_recon1
```

This run passed the collapse gate on shuffled validation windows with no Poisson
saturation: `collapse_eff_rank_frac ~= 0.113`, `rate_mean ~= 1.01`,
`rate_p99 ~= 2.02`, and `sat_frac = 0`.

Extract/plot latents:

```bash
CKPT=/home/tejas/le-wm-vi/outputs/marmo_amortized_all_4_13_poisson_loghi2_sigreg1_recon1/Allen_2022-04-13_amortized_poisson_exact_kl_dset/last.pt
LAT=/home/tejas/le-wm-vi/outputs/marmo_amortized_all_4_13_poisson_loghi2_sigreg1_recon1/Allen_2022-04-13_amortized_poisson_exact_kl_dset/latents_8192.npz

uv run python /home/tejas/le-wm-vi/marmo/extract_amortized_latents.py \
  --checkpoint "$CKPT" --session Allen_2022-04-13 \
  --splits train val --max-windows-per-split 8192 --out "$LAT"

uv run python /home/tejas/le-wm-vi/marmo/visualize_latents.py \
  --latents "$LAT" \
  --outdir /home/tejas/le-wm-vi/outputs/marmo_amortized_all_4_13_poisson_loghi2_sigreg1_recon1/Allen_2022-04-13_amortized_poisson_exact_kl_dset/latents_8192
```

Render the BackImage/V1/latent movie:

```bash
uv run python /home/tejas/le-wm-vi/marmo/make_backimage_latent_movie.py \
  --checkpoint "$CKPT" \
  --session Allen_2022-04-13 \
  --duration-s 8 \
  --fps 30 \
  --max-render-frames 240 \
  --latent-kind code \
  --out /home/tejas/le-wm-vi/outputs/marmo_amortized_all_4_13_poisson_loghi2_sigreg1_recon1/Allen_2022-04-13_amortized_poisson_exact_kl_dset/backimage_latent_movie.mp4
```

Gaussian amortized pilots currently collapse to about one effective direction
(`collapse_eff_rank_frac ~= 0.009`) even with stronger SIGReg and fixed-unit
variance. Treat Gaussian as the next debugging target rather than a valid
comparison checkpoint yet.
