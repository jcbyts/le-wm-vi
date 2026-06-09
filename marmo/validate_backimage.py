from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from marmo.backimage_sequences import BackImagePaths, BackImageSampler, load_backimage_dataset


def parse_args():
    p = argparse.ArgumentParser(description="Validate regenerated BackImage crops against backimage.dset")
    p.add_argument("--session", default="Allen_2022-04-13")
    p.add_argument(
        "--sessions",
        default=None,
        help="Comma-separated sessions, or 'all-allen' to validate every Allen backimage.dset under data-root.",
    )
    p.add_argument("--data-root", default="/mnt/sata/YatesMarmoV1")
    p.add_argument("--dset-path", default=None)
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--seed", type=int, default=1002)
    p.add_argument("--outdir", default="/home/tejas/le-wm-vi/outputs/marmo_validation")
    return p.parse_args()


def discover_sessions(args) -> list[str]:
    if args.sessions is None:
        return [args.session]
    if args.sessions.strip().lower() == "all-allen":
        root = Path(args.data_root) / "processed"
        sessions = []
        for path in sorted(root.glob("Allen_*/datasets/backimage.dset")):
            sessions.append(path.parents[1].name)
        if not sessions:
            raise FileNotFoundError(f"No Allen backimage.dset files found under {root}")
        return sessions
    return [x.strip() for x in args.sessions.split(",") if x.strip()]


def validate_session(args, session: str, outdir: Path) -> dict[str, object]:
    dset_path = Path(args.dset_path) if args.dset_path else BackImagePaths(session, Path(args.data_root)).dset_path
    dset = load_backimage_dataset(dset_path)
    sampler = BackImageSampler(session, dset)
    cov = dset.covariates
    trial_inds = cov["trial_inds"].cpu().numpy().astype(np.int64)
    unique_trials = np.unique(trial_inds)
    missing = {
        int(t): sampler.trial(int(t)).image_file
        for t in unique_trials
        if not sampler.has_image(int(t))
    }
    if missing:
        names = sorted(set(missing.values()))
        print(
            f"WARNING {session}: skipping {len(missing)} trials with missing source images: "
            + ", ".join(names)
        )
    image_ok = np.array([int(t) not in missing for t in trial_inds], dtype=bool)
    valid = np.flatnonzero((cov["dpi_valid"].cpu().numpy() > 0) & image_ok)
    rng = np.random.default_rng(args.seed)
    rows = rng.choice(valid, size=min(args.n, len(valid)), replace=False)
    rows.sort()

    stats = []
    gaze_offsets = []
    worst = None
    for row in rows:
        trial_id = int(cov["trial_inds"][row].item())
        regen = sampler.crop_roi(trial_id, cov["roi"][row].cpu().numpy())
        saved = cov["stim"][row].cpu().numpy()
        diff = np.abs(regen.astype(np.int16) - saved.astype(np.int16))
        rec = {
            "row": int(row),
            "trial": trial_id,
            "mean_abs": float(diff.mean()),
            "max_abs": int(diff.max()),
            "exact_frac": float((diff == 0).mean()),
        }
        stats.append(rec)
        if worst is None or rec["mean_abs"] > worst[0]["mean_abs"]:
            worst = (rec, saved, regen, diff)
        try:
            gaze = sampler.gaze_center_ij(cov, int(row))
            roi = sampler.pyramid_rois_for_row(cov, int(row), crop_sizes=(51,), center_mode="gaze")[0]
            gaze_offsets.append(gaze - roi.mean(axis=1))
        except KeyError:
            pass

    mean_abs = np.array([s["mean_abs"] for s in stats])
    max_abs = np.array([s["max_abs"] for s in stats])
    exact_frac = np.array([s["exact_frac"] for s in stats])
    print(f"Validated {len(stats)} crops from {dset_path}")
    print(f"  mean abs error: mean={mean_abs.mean():.6f}, p95={np.percentile(mean_abs, 95):.6f}, max={mean_abs.max():.6f}")
    print(f"  max abs error: mean={max_abs.mean():.3f}, max={max_abs.max()}")
    print(f"  exact pixel fraction: mean={exact_frac.mean():.6f}, min={exact_frac.min():.6f}")
    gaze_summary = {}
    if gaze_offsets:
        gaze_offsets_arr = np.stack(gaze_offsets, axis=0)
        abs_offsets = np.abs(gaze_offsets_arr)
        gaze_summary = {
            "gaze_median_di": float(np.median(gaze_offsets_arr[:, 0])),
            "gaze_median_dj": float(np.median(gaze_offsets_arr[:, 1])),
            "gaze_max_abs": float(abs_offsets.max()),
        }
        print(
            "  gaze-centered L0 offset: "
            f"median_di={gaze_summary['gaze_median_di']:.3f}px "
            f"median_dj={gaze_summary['gaze_median_dj']:.3f}px "
            f"max_abs={abs_offsets.max():.3f}px"
        )

    if worst is not None:
        rec, saved, regen, diff = worst
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        axes[0].imshow(saved, cmap="gray", vmin=0, vmax=255)
        axes[0].set_title("dset stim")
        axes[1].imshow(regen, cmap="gray", vmin=0, vmax=255)
        axes[1].set_title("regenerated")
        axes[2].imshow(diff, cmap="magma")
        axes[2].set_title(f"abs diff\nrow {rec['row']}")
        for ax in axes:
            ax.set_axis_off()
        fig.tight_layout()
        panel = outdir / f"{session}_worst_crop_validation.png"
        fig.savefig(panel, dpi=160)
        plt.close(fig)
        print(f"  Worst case: {rec}")
        print(f"  Saved panel: {panel}")

    return {
        "session": session,
        "dset_path": str(dset_path),
        "n_rows": int(len(trial_inds)),
        "n_trials": int(len(unique_trials)),
        "n_missing_image_trials": int(len(missing)),
        "n_validated": int(len(stats)),
        "mean_abs_mean": float(mean_abs.mean()) if len(mean_abs) else float("nan"),
        "mean_abs_p95": float(np.percentile(mean_abs, 95)) if len(mean_abs) else float("nan"),
        "mean_abs_max": float(mean_abs.max()) if len(mean_abs) else float("nan"),
        "max_abs_mean": float(max_abs.mean()) if len(max_abs) else float("nan"),
        "max_abs_max": int(max_abs.max()) if len(max_abs) else -1,
        "exact_frac_mean": float(exact_frac.mean()) if len(exact_frac) else float("nan"),
        "exact_frac_min": float(exact_frac.min()) if len(exact_frac) else float("nan"),
        **gaze_summary,
    }


def main():
    args = parse_args()
    sessions = discover_sessions(args)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = [validate_session(args, session, outdir) for session in sessions]
    if rows:
        keys = sorted(set().union(*(row.keys() for row in rows)))
        preferred = [
            "session",
            "n_rows",
            "n_trials",
            "n_missing_image_trials",
            "n_validated",
            "mean_abs_mean",
            "mean_abs_p95",
            "mean_abs_max",
            "max_abs_max",
            "exact_frac_mean",
            "exact_frac_min",
            "gaze_median_di",
            "gaze_median_dj",
            "gaze_max_abs",
            "dset_path",
        ]
        keys = [k for k in preferred if k in keys] + [k for k in keys if k not in preferred]
        path = outdir / "backimage_validation_summary.csv"
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved validation summary: {path}")


if __name__ == "__main__":
    main()
