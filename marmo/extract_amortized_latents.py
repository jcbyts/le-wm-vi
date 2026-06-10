from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from marmo.amortized_train_utils import load_amortized_from_checkpoint
from marmo.backimage_sequences import MarmoBackImageSequenceDataset, collate_marmo


def parse_args():
    p = argparse.ArgumentParser(description="Extract amortized world-model latents aligned to BackImage V1 responses")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--session", default=None)
    p.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val", "all"])
    p.add_argument("--center-mode", default=None, choices=["dset", "gaze"])
    p.add_argument("--crop-sizes", default=None)
    p.add_argument("--target-hz", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-windows-per-split", type=int, default=8192)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    return p.parse_args()


def parse_crop_sizes(text):
    return tuple(int(x) for x in text.split(",") if x)


def to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def main():
    args = parse_args()
    model, cfg, extra = load_amortized_from_checkpoint(args.checkpoint, map_location="cpu")
    model.to(args.device).eval()
    session = args.session or extra.get("session", "Allen_2022-04-13")
    crop_sizes = parse_crop_sizes(args.crop_sizes) if args.crop_sizes else tuple(extra.get("crop_sizes", (51, 101, 201)))
    center_mode = args.center_mode or extra.get("center_mode", "dset")
    target_hz = args.target_hz or int(extra.get("target_hz", 120))
    seq_len = cfg.history_size + 1
    arrays = {k: [] for k in [
        "eta", "code", "pred_hat", "robs", "dfs", "eyepos", "action", "t_bins",
        "trial_inds", "row_indices", "split",
    ]}
    split_to_id = {"train": 0, "val": 1, "all": 2}
    for split in args.splits:
        ds = MarmoBackImageSequenceDataset(
            session_name=session,
            split=split,
            target_hz=target_hz,
            seq_len=seq_len,
            crop_sizes=crop_sizes,
            output_hw=cfg.img_hw,
            center_mode=center_mode,
            max_windows=args.max_windows_per_split,
        )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_marmo)
        for batch in loader:
            batch = to_device(batch, args.device)
            with torch.no_grad():
                enc = model.encode(batch)
                eta = enc["emb"].detach()
                code = model.deterministic_code(eta).detach()
                pred = model.predict(eta[:, : cfg.history_size], enc["act_emb"][:, : cfg.history_size]).detach()
            for name, tensor in [
                ("eta", eta[:, -1]),
                ("code", code[:, -1]),
                ("pred_hat", pred[:, -1]),
                ("robs", batch["robs"][:, -1]),
                ("dfs", batch["dfs"][:, -1]),
                ("eyepos", batch["eyepos"][:, -1]),
                ("action", batch["action"][:, -2]),
                ("t_bins", batch["t_bins"][:, -1]),
                ("trial_inds", batch["trial_inds"][:, -1]),
                ("row_indices", batch["row_indices"][:, -1]),
            ]:
                arrays[name].append(tensor.detach().cpu().numpy())
            arrays["split"].append(np.full((eta.shape[0],), split_to_id[split], dtype=np.int64))
        print(f"extracted split={split} windows={len(ds)}")
    out = {k: np.concatenate(v, axis=0) for k, v in arrays.items() if v}
    out["split_names"] = np.array(["train", "val", "all"])
    out_path = Path(args.out) if args.out else Path(args.checkpoint).with_name("latents.npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)
    print(f"saved latents: {out_path}")
    for k, v in out.items():
        if hasattr(v, "shape"):
            print(f"  {k}: {v.shape}")


if __name__ == "__main__":
    main()
