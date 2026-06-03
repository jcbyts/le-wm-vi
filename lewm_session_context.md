# LeWorldModel (le-wm-vi) — Session Context / Handoff

Context for resuming work on this repo in a future session. Repo is **jcbyts/le-wm-vi**
(a fork; upstream is `lucas-maes/le-wm`). Machine: Linux, 4× RTX 6000 Ada (48 GB),
driver `555.42.06` (CUDA 12.5 max). User: Jake — prefers **conda** envs and **wandb** monitoring.

---

## TL;DR of where things stand
- Environment is set up and working; a model was trained and **hits 90% PushT planning success at epoch 5**.
- Behavioral monitoring (success rate + rollout videos to wandb) was built and committed.
- Commit `a69b02a` is pushed to `main`.
- Training was stopped at epoch 5 (90% was plenty; losses plateaued). All GPUs free.

---

## Environment setup (conda)
```bash
conda create -n lewm python=3.10
conda activate lewm
pip install "stable-worldmodel[train,env]" wandb "huggingface_hub[cli]"
pip install hdf5plugin            # REQUIRED, see gotcha #2
# CUDA fix (gotcha #1): generic install pulls torch cu130 which the driver can't run
pip install --force-reinstall torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install "fsspec[http]<=2026.2.0" "pillow<12.0"   # re-pin deps torch reinstall bumped
# verify: `pip check` is clean; torch 2.6.0+cu124; torch.cuda.is_available() == True
```

## Gotchas discovered (these are the non-obvious bits)
1. **CUDA/driver mismatch.** Default install grabs `torch 2.12+cu130`; the driver only
   supports CUDA 12.5 → GPU init fails ("NVIDIA driver too old, found 12050"). Fix = pin
   `torch==2.6.0+cu124` (the highest cu wheels this driver supports).
2. **HDF5 reader silently unregistered.** `stable_worldmodel` only registers the `hdf5`
   dataset format if **`hdf5plugin`** is installed (h5py alone is NOT enough). Without it you
   get `AttributeError: ... has no attribute 'HDF5Dataset'`. (Upstream issue #76.)
3. **Data format mismatch (.h5 vs .lance).** The committed training config was switched to
   `.lance` (PR #63), but the **published HF dataset is still `.h5.zst`** and there is no
   lance version on HF. We train directly from `.h5` instead (one-line config change). The
   HDF5 reader works fine and is what `eval.py` uses too.
4. **Cache dir** is `~/.stable_worldmodel` (NOT `~/.stable-wm` as the README says).
5. **No lockfile upstream** (issue #48); deps are fragile. The pins above are a known-good set.
6. **Epoch size:** one epoch = ~13,933 steps (~1.78M windowed sub-trajectories) ≈ **44 min**
   on one GPU at ~5.3 it/s. So `max_epochs: 100` ≈ **3 days** (upstream issue #52 flags it as
   likely too high). Most learning happens in epochs 0–1; 90% success reached by epoch 5.

## Data
- HF dataset repo: `quentinll/lewm-pusht` ships `pusht_expert_train.h5.zst` (~13 GB).
- Downloaded + decompressed to `~/.stable_worldmodel/datasets/pusht_expert_train.h5`
  (18,685 episodes). Other tasks: `quentinll/lewm-tworooms`, `-cube`, `-reacher`.
```bash
# download + decompress pattern:
python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('quentinll/lewm-pusht','pusht_expert_train.h5.zst',repo_type='dataset'))"
zstd -d <path>.h5.zst -o ~/.stable_worldmodel/datasets/pusht_expert_train.h5
```

## wandb
- Authenticated via `~/.netrc`. Entity resolves to **`yateslab`**, project **`lewm`**.
- Enabled in `config/train/launcher/local.yaml` (`enabled: True`, `entity: null`).

---

## The model (important for "what can we log")
- `jepa.py` = JEPA: pixels → ViT-tiny CLS embedding → predict **next embedding**; all costs
  in **latent space**. There is **NO pixel decoder** — you cannot log predicted frames.
- "Working like the paper" = **planning/control**: the model is an MPC cost function.
  `eval.py` runs CEM planning in the env and `world.evaluate(video=dir)` writes one
  `env_<i>.mp4` per episode + returns `success_rate`.
- Predictor `num_frames == history_size == 3`; the rollout context window must equal `history_size`.

## What was built this session (committed, a69b02a)
- **`planning.py`** — reusable planning-eval helpers (process/StandardScaler, image transform,
  episode sampling, world+policy build, video rollout) mirroring `eval.py`. `eval.py` untouched.
- **`monitor.py`** — `BehaviorEvalCallback` (Lightning) + `latent_rollout_error()`. Every
  `every_n_epochs` (rank 0 only, defensive try/except so it can't crash training) it logs:
  - `monitor/success_rate` + rollout mp4s as `wandb.Video`
  - `monitor/latent_rollout_mse_step_*` — open-loop multi-step latent prediction error
- **`train.py`** — wires the callback; sets `MUJOCO_GL=egl` for headless render.
- **`config/train/lewm.yaml`** — `monitor:` block (PushT defaults; adjust env_name/callables/
  keys_to_cache for other datasets).
- **`config/train/data/pusht.yaml`** — `name: pusht_expert_train.h5` (was `.lance`).
- ⚠️ **Unverified path:** the in-loop monitor's *wandb logging* never actually fired (run was
  stopped at epoch 5, before the epoch-10 trigger). Every underlying piece is verified
  standalone (the 90% eval below proves the planning machinery). If doing a longer run, watch
  the epoch-10 monitor log.

## Results (epoch-5 checkpoint)
- **PushT planning success: 90%** (45/50 episodes), full paper-style eval (num_eval=50, eval_budget=50).
- Losses: `fit/pred_loss` 0.22→~0.01, `fit/sigreg_loss` 40→~0.85 (stable, no collapse),
  `fit/loss` 3.84→~0.085.
- Checkpoints: `~/.stable_worldmodel/checkpoints/lewm/weights_epoch_{1..5}.pt` (+ config.json).
- Videos: `~/lewm_eval_epoch5/` (50 mp4s); sample logged to wandb run `eval-epoch5`.

---

## Commands

```bash
conda activate lewm && cd ~/repos/le-wm-vi

# Train (single GPU chosen for unattended robustness; avoids DDP + in-loop-eval issues)
python train.py data=pusht trainer.devices=1            # resumes from saved weights if present
#   log -> ~/lewm_train.log ; checkpoints -> ~/.stable_worldmodel/checkpoints/lewm/

# Quick smoke test
python train.py data=pusht trainer.max_epochs=1 trainer.devices=1 \
  +trainer.limit_train_batches=3 +trainer.limit_val_batches=2 num_workers=2

# Evaluate a checkpoint (videos + success_rate). Run on a free GPU while training:
CUDA_VISIBLE_DEVICES=1 python eval.py --config-name=pusht policy=lewm/weights_epoch_5.pt \
  eval.num_eval=50 eval.eval_budget=50
#   NOTE: with multiple .pt in the folder you must name the file (…/weights_epoch_N.pt)

# Other tasks need their HF dataset downloaded + data config name set to .h5 first:
#   data=tworoom | dmc | ogb
```

## Open / possible next steps
- Verify the in-loop monitor actually logs to wandb on a run that reaches epoch 10.
- Set up other datasets (tworoom / dmc / ogb): download HF data, set `.h5` in their data config.
- Consider lowering `max_epochs` (100 ≈ 3 days; ~5–15 epochs already strong).
- Optionally add a short README note documenting the `.h5` setup + hdf5plugin/CUDA pins.

## Memory
Persistent notes saved under the project memory: `lewm-env-setup` and `lewm-user-prefs`.
