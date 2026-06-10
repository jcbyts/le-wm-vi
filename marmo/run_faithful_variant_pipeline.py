from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Train/evaluate one faithful Gaussian BackImage LeWM variant")
    p.add_argument("--tag", required=True)
    p.add_argument("--session", default="Allen_2022-04-13")
    p.add_argument("--sessions", default=None)
    p.add_argument("--data-root", default="/mnt/sata/YatesMarmoV1")
    p.add_argument("--outbase", default="/home/tejas/le-wm-vi/outputs/gaussian_variant_search")
    p.add_argument("--device", default="cuda")
    p.add_argument("--cuda-visible-devices", default=None)
    p.add_argument("--center-mode", choices=["dset", "gaze"], default="dset")
    p.add_argument("--crop-sizes", default="51,101,201,401,801,1201")
    p.add_argument("--pyramid-mode", choices=["raw", "gaussian", "hybrid_l0_gaussian", "laplacian"], default="raw")
    p.add_argument("--blur-sigmas", default=None)
    p.add_argument("--laplacian-contrast", type=float, default=1.0)
    p.add_argument("--output-hw", type=int, default=51)
    p.add_argument("--pixel-normalization", choices=["unit", "visioncore"], default="visioncore")
    p.add_argument("--target-mode", choices=["full", "l0"], default="full")
    p.add_argument("--masked-channel-value", type=float, default=None)
    p.add_argument("--split-mode", choices=["numpy", "torch"], default="torch")
    p.add_argument("--robs-downsample-mode", choices=["sample", "sum"], default="sum")
    p.add_argument("--covariate-downsample-mode", choices=["sample", "mean"], default="mean")
    p.add_argument("--validity-downsample-mode", choices=["sample", "all"], default="all")
    p.add_argument("--foveal-dim", type=int, default=0)
    p.add_argument("--context-dim", type=int, default=0)
    p.add_argument("--encoder-kind", choices=["conv", "alexnet_v1", "voneblock", "vonealexnet"], default="conv")
    p.add_argument("--no-neural-pretrained", action="store_true")
    p.add_argument("--train-neural-frontend", action="store_true")
    p.add_argument("--neural-resize-hw", type=int, default=224)
    p.add_argument("--neural-feature-index", type=int, default=2)
    p.add_argument("--neural-pool-hw", type=int, default=1)
    p.add_argument("--neural-pixel-mode", choices=["unit", "visioncore", "auto"], default=None)
    p.add_argument("--vone-simple-channels", type=int, default=128)
    p.add_argument("--vone-complex-channels", type=int, default=128)
    p.add_argument("--vone-ksize", type=int, default=25)
    p.add_argument("--vone-stride", type=int, default=4)
    p.add_argument("--vone-visual-degrees", type=float, default=8.0)
    p.add_argument("--vone-sf-corr", type=float, default=0.75)
    p.add_argument("--vone-sf-max", type=float, default=9.0)
    p.add_argument("--vone-sf-min", type=float, default=0.0)
    p.add_argument("--vone-noise-mode", choices=["none", "gaussian", "neuronal"], default="none")
    p.add_argument("--projector-norm", choices=["batchnorm", "layernorm", "none"], default="layernorm")
    p.add_argument("--sigreg-weight", type=float, default=0.3)
    p.add_argument("--dfs-mode", choices=["none", "valid_nlags", "visioncore"], default="visioncore")
    p.add_argument("--dfs-valid-lags", type=int, default=32)
    p.add_argument("--dfs-missing-threshold", type=float, default=45.0)
    p.add_argument("--action-history", type=int, default=1)
    p.add_argument("--action-smoothed-dim", type=int, default=8)
    p.add_argument("--action-mlp-scale", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=1200)
    p.add_argument("--max-train-windows", type=int, default=16384)
    p.add_argument("--max-val-windows", type=int, default=4096)
    p.add_argument("--val-interval", type=int, default=300)
    p.add_argument("--val-batches", type=int, default=8)
    p.add_argument("--save-interval", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=1002)
    p.add_argument("--extract-full", action="store_true")
    p.add_argument("--extract-max-windows", type=int, default=65536)
    p.add_argument("--extract-window-sample-mode", choices=["random", "first", "last"], default="first")
    p.add_argument("--readout-epochs", type=int, default=50)
    p.add_argument("--readout-patience", type=int, default=10)
    p.add_argument("--readout-feature-keys", default="code,pred_hat,none")
    p.add_argument("--readout-lag-sets", default="3,4,5;2,3,4,5;2,3,4;3,4;4,5")
    p.add_argument("--readout-archs", default="mlp")
    p.add_argument("--readout-behavior-mode", choices=["raw", "visioncore", "raw+visioncore", "none"], default="visioncore")
    p.add_argument("--readout-mlp-hidden-dims", default="64,128")
    p.add_argument("--readout-mlp-depths", default="1,2")
    p.add_argument("--readout-dropouts", default="0.1,0.3")
    p.add_argument("--readout-mlp-weight-decays", default="1e-4,3e-4,1e-3,3e-3")
    p.add_argument("--readout-linear-weight-decays", default="1e-6,1e-5,1e-4,1e-3,1e-2")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-extract", action="store_true")
    p.add_argument("--skip-readout", action="store_true")
    p.add_argument("--extract-precompute-pixels", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def run(cmd: list[str], *, env: dict[str, str], cwd: Path):
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def variant_dir(args) -> Path:
    session_name = "all_allen" if args.sessions is not None else args.session
    return Path(args.outbase) / args.tag / f"{session_name}_faithful_gaussian_{args.center_mode}"


def maybe_add(cmd: list[str], flag: str, value):
    if value is not None:
        cmd.extend([flag, str(value)])


def main():
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    outbase = Path(args.outbase) / args.tag
    outdir = variant_dir(args)

    train_cmd = [
        sys.executable,
        "-m",
        "marmo.train_faithful_marmo",
        "--session",
        args.session,
        "--data-root",
        args.data_root,
        "--center-mode",
        args.center_mode,
        "--crop-sizes",
        args.crop_sizes,
        "--pyramid-mode",
        args.pyramid_mode,
        "--laplacian-contrast",
        str(args.laplacian_contrast),
        "--output-hw",
        str(args.output_hw),
        "--pixel-normalization",
        args.pixel_normalization,
        "--target-mode",
        args.target_mode,
        "--split-mode",
        args.split_mode,
        "--robs-downsample-mode",
        args.robs_downsample_mode,
        "--covariate-downsample-mode",
        args.covariate_downsample_mode,
        "--validity-downsample-mode",
        args.validity_downsample_mode,
        "--dfs-mode",
        args.dfs_mode,
        "--dfs-valid-lags",
        str(args.dfs_valid_lags),
        "--dfs-missing-threshold",
        str(args.dfs_missing_threshold),
        "--foveal-dim",
        str(args.foveal_dim),
        "--context-dim",
        str(args.context_dim),
        "--action-history",
        str(args.action_history),
        "--action-smoothed-dim",
        str(args.action_smoothed_dim),
        "--action-mlp-scale",
        str(args.action_mlp_scale),
        "--embed-dim",
        "192",
        "--encoder-width",
        "64",
        "--encoder-kind",
        args.encoder_kind,
        "--neural-resize-hw",
        str(args.neural_resize_hw),
        "--neural-feature-index",
        str(args.neural_feature_index),
        "--neural-pool-hw",
        str(args.neural_pool_hw),
        "--vone-simple-channels",
        str(args.vone_simple_channels),
        "--vone-complex-channels",
        str(args.vone_complex_channels),
        "--vone-ksize",
        str(args.vone_ksize),
        "--vone-stride",
        str(args.vone_stride),
        "--vone-visual-degrees",
        str(args.vone_visual_degrees),
        "--vone-sf-corr",
        str(args.vone_sf_corr),
        "--vone-sf-max",
        str(args.vone_sf_max),
        "--vone-sf-min",
        str(args.vone_sf_min),
        "--vone-noise-mode",
        args.vone_noise_mode,
        "--predictor-depth",
        "6",
        "--predictor-heads",
        "16",
        "--predictor-mlp-dim",
        "2048",
        "--predictor-dim-head",
        "64",
        "--sigreg-weight",
        str(args.sigreg_weight),
        "--projector-norm",
        args.projector_norm,
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--precompute-pixels",
        "--max-train-windows",
        str(args.max_train_windows),
        "--max-val-windows",
        str(args.max_val_windows),
        "--max-steps",
        str(args.max_steps),
        "--warmup-steps",
        str(max(50, min(500, args.max_steps // 8))),
        "--val-interval",
        str(args.val_interval),
        "--val-batches",
        str(args.val_batches),
        "--save-interval",
        str(args.save_interval),
        "--log-interval",
        "100",
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--outdir",
        str(outbase),
    ]
    if args.masked_channel_value is not None:
        train_cmd.extend(["--masked-channel-value", str(args.masked_channel_value)])
    if args.no_neural_pretrained:
        train_cmd.append("--no-neural-pretrained")
    if args.train_neural_frontend:
        train_cmd.append("--train-neural-frontend")
    maybe_add(train_cmd, "--neural-pixel-mode", args.neural_pixel_mode)
    maybe_add(train_cmd, "--blur-sigmas", args.blur_sigmas)
    maybe_add(train_cmd, "--sessions", args.sessions)
    if not args.skip_train and (args.force or not (outdir / "last.pt").exists()):
        run(train_cmd, env=env, cwd=repo)

    ckpt = outdir / "best.pt"
    if not ckpt.exists():
        ckpt = outdir / "last.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"No checkpoint found in {outdir}")

    extract_limit = 0 if args.extract_full else int(args.extract_max_windows)
    latents = outdir / ("latents_full.npz" if extract_limit <= 0 else "latents_stage.npz")
    if not args.skip_extract and (args.force or not latents.exists()):
        extract_cmd = [
            sys.executable,
            "-m",
            "marmo.extract_faithful_latents",
            "--checkpoint",
            str(ckpt),
            "--data-root",
            args.data_root,
            "--splits",
            "train",
            "val",
            "--max-windows-per-split",
            str(extract_limit),
            "--window-sample-mode",
            args.extract_window_sample_mode,
            "--batch-size",
            "128",
            "--num-workers",
            str(args.num_workers),
            "--device",
            args.device,
            "--out",
            str(latents),
        ]
        if args.extract_precompute_pixels:
            extract_cmd.append("--precompute-pixels")
        run(extract_cmd, env=env, cwd=repo)

    readout_dir = outdir / "readout_predhat_lag_search"
    if not args.skip_readout and (args.force or not (readout_dir / "results.csv").exists()):
        readout_cmd = [
            sys.executable,
            "-m",
            "marmo.train_latent_spike_readout",
            "--latents",
            str(latents),
            "--outdir",
            str(readout_dir),
            "--feature-keys",
            args.readout_feature_keys,
            "--lag-sets",
            args.readout_lag_sets,
            "--archs",
            args.readout_archs,
            "--behavior-mode",
            args.readout_behavior_mode,
            "--include-eye",
            "--include-action",
            "--mlp-hidden-dims",
            args.readout_mlp_hidden_dims,
            "--mlp-depths",
            args.readout_mlp_depths,
            "--dropouts",
            args.readout_dropouts,
            "--mlp-weight-decays",
            args.readout_mlp_weight_decays,
            "--linear-weight-decays",
            args.readout_linear_weight_decays,
            "--epochs",
            str(args.readout_epochs),
            "--patience",
            str(args.readout_patience),
            "--batch-size",
            "4096",
            "--seed",
            str(args.seed + 200),
            "--device",
            args.device,
        ]
        run(readout_cmd, env=env, cwd=repo)

    summary = {
        "tag": args.tag,
        "outdir": str(outdir),
        "checkpoint": str(ckpt),
        "latents": str(latents),
        "readout_dir": str(readout_dir),
        "args": vars(args),
    }
    results = readout_dir / "results.csv"
    if results.exists():
        df = pd.read_csv(results).sort_values("val_mean_bps", ascending=False)
        if not df.empty:
            summary["best_readout"] = df.iloc[0].to_dict()
    metrics = outdir / "metrics.csv"
    if metrics.exists():
        dfm = pd.read_csv(metrics)
        vals = dfm[dfm.get("phase").eq("val")] if "phase" in dfm else pd.DataFrame()
        if not vals.empty:
            summary["last_val"] = vals.iloc[-1].to_dict()
    with (outdir / "variant_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary.get("best_readout", {}), indent=2), flush=True)
    print(f"saved variant summary: {outdir / 'variant_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
