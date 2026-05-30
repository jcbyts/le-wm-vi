"""Reusable planning-evaluation helpers.

These mirror the setup in ``eval.py`` (process/transform construction, episode
sampling, world+policy build, video rollout) so the in-training behavioral
monitor produces success numbers comparable to the standalone evaluation.
``eval.py`` itself is intentionally left untouched; this module is the shared
implementation used by the training-time monitor (``monitor.py``).
"""

import glob
from pathlib import Path

import numpy as np
import torch
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

import stable_pretraining as spt
import stable_worldmodel as swm


def _episode_col(dataset):
    return "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"


def planning_img_transform(img_size):
    """Image/goal transform applied to env observations during planning.

    Matches ``eval.py``'s ``img_transform``: to-image, float scale, ImageNet
    normalize, resize.
    """
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=img_size),
        ]
    )


def build_process(dataset, keys_to_cache):
    """Fit a StandardScaler per non-pixel column (+ ``goal_`` aliases).

    Mirrors the ``process`` dict built in ``eval.py``.
    """
    process = {}
    for col in keys_to_cache:
        if col in ("pixels",):
            continue
        scaler = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        scaler.fit(col_data)
        process[col] = scaler
        if col != "action":
            process[f"goal_{col}"] = scaler
    return process


def get_episodes_length(dataset, episodes):
    col = _episode_col(dataset)
    episode_idx = dataset.get_col_data(col)
    step_idx = dataset.get_col_data("step_idx")
    return np.array(
        [np.max(step_idx[episode_idx == ep]) + 1 for ep in episodes]
    )


def sample_eval_points(dataset, num_eval, goal_offset_steps, seed):
    """Pick ``num_eval`` valid (episode, start_step) pairs.

    Replicates the sampling block of ``eval.py``. Returns
    ``(episodes_idx, start_steps)`` as plain Python lists. Call this once and
    reuse the result so the success-rate curve is comparable across epochs.
    """
    col = _episode_col(dataset)
    ep_indices, _ = np.unique(dataset.get_col_data(col), return_index=True)
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - goal_offset_steps - 1
    max_start_idx_dict = {ep: max_start_idx[i] for i, ep in enumerate(ep_indices)}
    max_start_per_row = np.array(
        [max_start_idx_dict[ep] for ep in dataset.get_col_data(col)]
    )
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]

    g = np.random.default_rng(seed)
    chosen = g.choice(len(valid_indices) - 1, size=num_eval, replace=False)
    chosen = np.sort(valid_indices[chosen])

    rows = dataset.get_row_data(chosen)
    return rows[col].tolist(), rows["step_idx"].tolist()


def run_planning_eval(
    model,
    dataset,
    *,
    env_name,
    num_eval,
    eval_budget,
    goal_offset_steps,
    plan_config,
    cem_kwargs,
    callables,
    process,
    transform,
    video_dir,
    episodes_idx=None,
    start_steps=None,
):
    """Build the world + world-model policy, roll out, and return metrics.

    The model is used in-place (set to eval / no-grad and given the
    ``interpolate_pos_encoding`` flag eval expects); the caller is responsible
    for restoring train mode afterwards. Writes one ``env_<i>.mp4`` per env to
    ``video_dir`` and returns ``(metrics_dict, sorted_list_of_video_paths)``.
    """
    if episodes_idx is None or start_steps is None:
        episodes_idx, start_steps = sample_eval_points(
            dataset, num_eval, goal_offset_steps, cem_kwargs.get("seed", 42)
        )

    world = swm.World(
        env_name=env_name,
        num_envs=num_eval,
        image_shape=(224, 224),
        max_episode_steps=2 * eval_budget,
    )

    model.eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    config = swm.PlanConfig(**plan_config)
    solver = swm.solver.CEMSolver(model=model, **cem_kwargs)
    policy = swm.policy.WorldModelPolicy(
        solver=solver, config=config, process=process, transform=transform
    )
    world.set_policy(policy)

    video_dir = Path(video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    try:
        metrics = world.evaluate(
            dataset=dataset,
            start_steps=start_steps,
            goal_offset=goal_offset_steps,
            eval_budget=eval_budget,
            episodes_idx=episodes_idx,
            callables=callables,
            video=str(video_dir),
        )
    finally:
        world.close()

    videos = sorted(glob.glob(str(video_dir / "env_*.mp4")))
    return metrics, videos
