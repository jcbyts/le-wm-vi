from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_args():
    p = argparse.ArgumentParser(description="Visualize marmoset world-model latents and V1 probes")
    p.add_argument("--latents", required=True)
    p.add_argument("--outdir", default=None)
    p.add_argument("--target-hz", type=float, default=120.0)
    return p.parse_args()


def safe_corr(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return np.nan
    return pearsonr(x[ok], y[ok])[0]


def plot_pca(outdir: Path, z, eyepos, speed):
    pca = PCA(n_components=min(8, z.shape[1]), random_state=0)
    pcs = pca.fit_transform(StandardScaler().fit_transform(z))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    colors = [eyepos[:, 0], eyepos[:, 1], speed]
    titles = ["eye x (deg)", "eye y (deg)", "eye speed (deg/s)"]
    for ax, c, title in zip(axes, colors, titles):
        im = ax.scatter(pcs[:, 0], pcs[:, 1], c=c, s=4, cmap="viridis", alpha=0.75)
        ax.set_xlabel("latent PC1")
        ax.set_ylabel("latent PC2")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(outdir / "latent_pca_gaze.png", dpi=170)
    plt.close(fig)
    return pcs, pca.explained_variance_ratio_


def plot_v1_probe(outdir: Path, z, robs, dfs, split):
    train = split == 0
    val = split == 1
    if train.sum() < 20 or val.sum() < 20:
        print("Skipping V1 probe: need both train and val split latents.")
        return
    unit_mask = np.nanmean(robs[train], axis=0) > 0.005
    y_train = robs[train][:, unit_mask]
    y_val = robs[val][:, unit_mask]
    model = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-2, 4, 12)),
    )
    model.fit(z[train], y_train)
    pred = model.predict(z[val])
    corr = np.array([safe_corr(pred[:, i], y_val[:, i]) for i in range(y_val.shape[1])])

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].hist(corr[np.isfinite(corr)], bins=30, color="0.2")
    axes[0].set_xlabel("val Pearson r")
    axes[0].set_ylabel("units")
    axes[0].set_title(f"V1 probe, median={np.nanmedian(corr):.3f}")
    top = np.argsort(np.nan_to_num(corr, nan=-np.inf))[-1]
    axes[1].plot(y_val[:300, top], label="robs", lw=1)
    axes[1].plot(pred[:300, top], label="ridge(latent)", lw=1)
    axes[1].set_title(f"best unit r={corr[top]:.3f}")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "v1_latent_probe.png", dpi=170)
    plt.close(fig)
    np.save(outdir / "v1_probe_correlations.npy", corr)


def plot_perisaccade(outdir: Path, pcs, speed, trial_inds, win=(-20, 40)):
    lo, hi = win
    if len(speed) < hi - lo + 10:
        return
    thresh = np.nanpercentile(speed, 99)
    peaks = []
    for i in range(1, len(speed) - 1):
        if speed[i] > thresh and speed[i] >= speed[i - 1] and speed[i] >= speed[i + 1]:
            if i + lo >= 0 and i + hi < len(speed):
                if np.all(trial_inds[i + lo:i + hi] == trial_inds[i]):
                    peaks.append(i)
    if len(peaks) < 3:
        print("Skipping peri-saccade plot: not enough high-speed events in extracted subset.")
        return
    t = np.arange(lo, hi)
    pc1 = np.stack([pcs[i + lo:i + hi, 0] for i in peaks])
    pc2 = np.stack([pcs[i + lo:i + hi, 1] for i in peaks])
    sp = np.stack([speed[i + lo:i + hi] for i in peaks])
    fig, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
    for ax, arr, title in zip(axes, [sp, pc1, pc2], ["speed", "latent PC1", "latent PC2"]):
        m = np.nanmean(arr, axis=0)
        se = np.nanstd(arr, axis=0) / np.sqrt(arr.shape[0])
        ax.plot(t, m, color="k")
        ax.fill_between(t, m - se, m + se, color="0.7")
        ax.axvline(0, color="r", lw=1)
        ax.set_ylabel(title)
    axes[-1].set_xlabel("bins around high-speed event")
    fig.suptitle(f"Peri-saccade latent summary, n={len(peaks)}, speed>{thresh:.1f}")
    fig.tight_layout()
    fig.savefig(outdir / "peri_saccade_latents.png", dpi=170)
    plt.close(fig)


def main():
    args = parse_args()
    data = np.load(args.latents, allow_pickle=True)
    z = data["code"].astype(np.float32)
    eyepos = data["eyepos"].astype(np.float32)
    action = data["action"].astype(np.float32)
    speed = np.linalg.norm(action, axis=1) * args.target_hz
    robs = data["robs"].astype(np.float32)
    dfs = data["dfs"].astype(np.float32)
    try:
        robs = robs * dfs
    except ValueError:
        if dfs.ndim == 2 and dfs.shape[1] == 1:
            robs = robs * dfs
        else:
            raise
    split = data["split"].astype(np.int64)
    trial_inds = data["trial_inds"].astype(np.int64)

    outdir = Path(args.outdir) if args.outdir else Path(args.latents).with_suffix("")
    outdir.mkdir(parents=True, exist_ok=True)

    pcs, evr = plot_pca(outdir, z, eyepos, speed)
    plot_v1_probe(outdir, z, robs, dfs, split)
    plot_perisaccade(outdir, pcs, speed, trial_inds)
    print(f"saved figures in {outdir}")
    print(f"PCA explained variance first 5: {evr[:5]}")


if __name__ == "__main__":
    main()
