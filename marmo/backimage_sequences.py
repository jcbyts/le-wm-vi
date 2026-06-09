from __future__ import annotations

import math
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset

from DataYatesV1.exp.backimage import BackImageTrial
from DataYatesV1.exp.support import get_backimage_directory
from DataYatesV1.utils.data.datasets import DictDataset
from DataYatesV1.utils.general import nd_cut
from DataYatesV1.utils.io import get_session


LevelSpec = int | str


@dataclass(frozen=True)
class BackImagePaths:
    session_name: str = "Allen_2022-04-13"
    data_root: Path = Path("/mnt/sata/YatesMarmoV1")

    @property
    def dset_path(self) -> Path:
        return self.data_root / "processed" / self.session_name / "datasets" / "backimage.dset"


def split_session_name(session_name: str) -> tuple[str, str]:
    parts = session_name.split("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Expected session name like Allen_2022-04-13, got {session_name!r}")
    return parts[0], parts[1]


def load_backimage_dataset(path: str | Path) -> DictDataset:
    return DictDataset.load(Path(path))


def normalize_level_specs(specs: str | Sequence[LevelSpec]) -> tuple[LevelSpec, ...]:
    if isinstance(specs, str):
        raw = [x.strip() for x in specs.split(",") if x.strip()]
    else:
        raw = list(specs)
    out: list[LevelSpec] = []
    for item in raw:
        if isinstance(item, str):
            low = item.strip().lower()
            if low in {"screen", "full", "full_screen", "fullscreen"}:
                out.append("screen")
            else:
                out.append(int(low))
        else:
            out.append(int(item))
    if not out:
        raise ValueError("At least one pyramid level must be specified")
    return tuple(out)


def level_spec_label(spec: LevelSpec) -> str:
    return "screen" if isinstance(spec, str) else str(int(spec))


def parse_blur_sigmas(sigmas: str | Sequence[float] | None, n_levels: int) -> tuple[float, ...]:
    if sigmas is None or sigmas == "":
        values = [float(2**i) for i in range(n_levels)]
    elif isinstance(sigmas, str):
        values = [float(x.strip()) for x in sigmas.split(",") if x.strip()]
    else:
        values = [float(x) for x in sigmas]
    if len(values) == 1 and n_levels > 1:
        values = values * n_levels
    if len(values) != n_levels:
        raise ValueError(f"Expected {n_levels} blur sigmas, got {len(values)}")
    return tuple(max(0.0, float(x)) for x in values)


def normalize_pyramid_mode(mode: str) -> str:
    mode = str(mode).strip().lower()
    aliases = {
        "raw": "raw",
        "none": "raw",
        "gauss": "gaussian",
        "gaussian": "gaussian",
        "blur": "gaussian",
        "blurred": "gaussian",
        "hybrid": "hybrid_l0_gaussian",
        "hybrid_l0_gaussian": "hybrid_l0_gaussian",
        "sharp_l0_blur": "hybrid_l0_gaussian",
        "lap": "laplacian",
        "laplacian": "laplacian",
        "dog": "laplacian",
    }
    if mode not in aliases:
        raise ValueError("pyramid_mode must be raw, gaussian, hybrid_l0_gaussian, or laplacian")
    return aliases[mode]


def normalize_pixel_values(raw_pixels: np.ndarray, mode: str = "unit") -> np.ndarray:
    """Convert reconstructed BackImage pixels into the model input convention."""
    mode = str(mode).strip().lower()
    pixels = raw_pixels.astype(np.float32)
    if np.issubdtype(raw_pixels.dtype, np.integer):
        pixels01 = pixels / 255.0
    else:
        pixels01 = np.clip(pixels, 0.0, 1.0)
    if mode in {"unit", "zero_one", "0_1"}:
        return pixels01
    if mode in {"visioncore", "pixelnorm", "centered"}:
        return pixels01 - (127.0 / 255.0)
    raise ValueError("pixel_normalization must be 'unit' or 'visioncore'")


class BackImageSampler:
    """Reconstruct BackImage crops from the original displayed image.

    In ``center_mode='dset'``, the level-0 crop uses the saved dset ROI exactly,
    so it can be compared pixel for pixel against ``backimage.dset['stim']``.
    In ``center_mode='gaze'``, every pyramid level is centered on the measured
    eye position, expressed in screen ``ij`` pixels.
    """

    def __init__(self, session_name: str, dset: DictDataset | None = None):
        subject, date = split_session_name(session_name)
        self.session = get_session(subject, date)
        self.exp = self.session.exp
        self.dset = dset
        self._trial_cache: dict[int, BackImageTrial] = {}
        self._image_cache: dict[int, np.ndarray] = {}
        self._padded_screen_cache: OrderedDict[tuple[int, int], Image.Image] = OrderedDict()
        self._padded_screen_cache_max = 8
        self.background_dir = get_backimage_directory()

    def trial(self, trial_id: int) -> BackImageTrial:
        trial_id = int(trial_id)
        if trial_id not in self._trial_cache:
            self._trial_cache[trial_id] = BackImageTrial(self.exp["D"][trial_id], self.exp["S"])
        return self._trial_cache[trial_id]

    def image(self, trial_id: int) -> np.ndarray:
        trial_id = int(trial_id)
        if trial_id not in self._image_cache:
            self._image_cache[trial_id] = self.trial(trial_id).get_image()
        return self._image_cache[trial_id]

    @property
    def screen_shape_ij(self) -> tuple[int, int]:
        rect = np.asarray(self.exp["S"]["screenRect"], dtype=np.float64)
        width = int(round(rect[2] - rect[0]))
        height = int(round(rect[3] - rect[1]))
        return height, width

    def screen_roi(self) -> np.ndarray:
        rect = np.asarray(self.exp["S"]["screenRect"], dtype=np.int64)
        return np.asarray([[rect[1], rect[3]], [rect[0], rect[2]]], dtype=np.int64)

    def screen_canvas(self, trial_id: int) -> np.ndarray:
        trial = self.trial(trial_id)
        image = self.image(trial_id)
        rect = np.asarray(self.exp["S"]["screenRect"], dtype=np.int64)
        screen_x0, screen_y0, screen_x1, screen_y1 = rect.tolist()
        height, width = int(screen_y1 - screen_y0), int(screen_x1 - screen_x0)
        canvas = np.full((height, width), int(trial.bkgnd), dtype=image.dtype)
        left, top, right, bottom = np.asarray(trial.dest_rect, dtype=np.int64)
        r0 = int(top - screen_y0)
        r1 = int(bottom - screen_y0)
        c0 = int(left - screen_x0)
        c1 = int(right - screen_x0)
        dst_i0 = max(0, r0)
        dst_j0 = max(0, c0)
        dst_i1 = min(height, r1)
        dst_j1 = min(width, c1)
        if dst_i1 > dst_i0 and dst_j1 > dst_j0:
            src_i0 = dst_i0 - r0
            src_j0 = dst_j0 - c0
            src_i1 = src_i0 + (dst_i1 - dst_i0)
            src_j1 = src_j0 + (dst_j1 - dst_j0)
            canvas[dst_i0:dst_i1, dst_j0:dst_j1] = image[src_i0:src_i1, src_j0:src_j1]
        return canvas

    def _padded_screen_image(self, trial_id: int, pad: int) -> Image.Image:
        key = (int(trial_id), int(pad))
        cached = self._padded_screen_cache.get(key)
        if cached is not None:
            self._padded_screen_cache.move_to_end(key)
            return cached
        canvas = self.screen_canvas(trial_id)
        bkgnd = int(self.trial(trial_id).bkgnd)
        image = Image.fromarray(canvas)
        if pad > 0:
            padded = Image.new("L", (image.width + 2 * pad, image.height + 2 * pad), bkgnd)
            padded.paste(image, (pad, pad))
        else:
            padded = image
        self._padded_screen_cache[key] = padded
        self._padded_screen_cache.move_to_end(key)
        while len(self._padded_screen_cache) > self._padded_screen_cache_max:
            self._padded_screen_cache.popitem(last=False)
        return padded

    def crop_screen_roi_resized(self, trial_id: int, roi: np.ndarray, output_hw: int, pad: int) -> np.ndarray:
        roi = np.asarray(roi, dtype=np.int64)
        if roi.shape != (2, 2):
            raise ValueError(f"ROI must have shape (2, 2), got {roi.shape}")
        screen_origin = self.screen_roi()[:, 0].astype(np.int64)
        i0, i1 = roi[0]
        j0, j1 = roi[1]
        height = int(i1 - i0)
        width = int(j1 - j0)
        if height <= 0 or width <= 0:
            raise ValueError(f"ROI has non-positive size: {roi.tolist()}")
        canvas = self.screen_canvas(trial_id)
        crop = nd_cut(
            canvas,
            np.asarray([int(i0 - screen_origin[0]), int(j0 - screen_origin[1])], dtype=np.int64),
            (height, width),
            fill_value=int(self.trial(trial_id).bkgnd),
        )
        if crop.shape != (output_hw, output_hw):
            crop_img = Image.fromarray(crop)
            resample = Image.BILINEAR if max(crop.shape[:2]) <= output_hw else Image.LANCZOS
            crop = np.asarray(crop_img.resize((output_hw, output_hw), resample=resample))
        return np.asarray(crop)

    def image_path(self, trial_id: int) -> Path:
        return self.background_dir / self.trial(trial_id).image_file

    def has_image(self, trial_id: int) -> bool:
        return self.image_path(trial_id).exists()

    def crop_roi(self, trial_id: int, roi: np.ndarray) -> np.ndarray:
        trial = self.trial(trial_id)
        image = self.image(trial_id)
        roi = np.asarray(roi, dtype=np.int64)
        if roi.shape != (2, 2):
            raise ValueError(f"ROI must have shape (2, 2), got {roi.shape}")
        height = int(roi[0, 1] - roi[0, 0])
        width = int(roi[1, 1] - roi[1, 0])
        src_pos = np.flipud(trial.dest_rect[:2])
        return nd_cut(
            image,
            roi[:, 0] - src_pos,
            (height, width),
            fill_value=int(trial.bkgnd),
        )

    def saved_stim_resized(self, row: int, output_hw: int) -> np.ndarray:
        if self.dset is None or "stim" not in self.dset.covariates:
            raise RuntimeError("saved BackImage stim is unavailable")
        stim = self.dset.covariates["stim"][int(row)]
        if torch.is_tensor(stim):
            stim = stim.detach().cpu().numpy()
        stim_np = np.asarray(stim)
        if stim_np.shape != (int(output_hw), int(output_hw)):
            resample = Image.BILINEAR if max(stim_np.shape[:2]) <= int(output_hw) else Image.LANCZOS
            stim_np = np.asarray(Image.fromarray(stim_np).resize((int(output_hw), int(output_hw)), resample=resample))
        return stim_np

    @staticmethod
    def centered_roi(center_ij: np.ndarray, crop_size: int) -> np.ndarray:
        center = np.asarray(center_ij, dtype=np.float64)
        lo = np.floor(center - crop_size / 2.0).astype(np.int64)
        hi = lo + int(crop_size)
        return np.stack([lo, hi], axis=1)

    @property
    def screen_center_ij(self) -> np.ndarray:
        return np.asarray(self.exp["S"]["centerPix"], dtype=np.float64)[::-1]

    @property
    def pix_per_deg(self) -> float:
        return float(self.exp["S"]["pixPerDeg"])

    def eyepos_to_screen_ij(self, eyepos_xy_deg: np.ndarray) -> np.ndarray:
        """Convert ``eyepos`` covariate values from ``[x_deg, y_deg]`` to screen ``[i, j]``."""
        eye = np.asarray(eyepos_xy_deg, dtype=np.float64)
        center = self.screen_center_ij
        ppd = self.pix_per_deg
        return np.asarray([center[0] - eye[1] * ppd, center[1] + eye[0] * ppd], dtype=np.float64)

    def gaze_center_ij(self, covariates: dict[str, torch.Tensor], row: int) -> np.ndarray:
        """Return the gaze center in screen ``ij`` pixels, preferring ``eyepos`` over ``dpi_pix``."""
        row = int(row)
        if "eyepos" in covariates:
            eyepos = covariates["eyepos"][row].detach().cpu().numpy().astype(np.float64)
            if eyepos.shape == (2,) and np.isfinite(eyepos).all():
                return self.eyepos_to_screen_ij(eyepos)
        if "dpi_pix" in covariates:
            dpi = covariates["dpi_pix"][row].detach().cpu().numpy().astype(np.float64)
            if dpi.shape == (2,) and np.isfinite(dpi).all():
                return dpi
        raise KeyError("Cannot build gaze-centered crop: missing finite 'eyepos' and 'dpi_pix'")

    @staticmethod
    def _resize_u8(image: np.ndarray, output_hw: int) -> np.ndarray:
        if image.shape == (output_hw, output_hw):
            return image
        resample = Image.BILINEAR if max(image.shape[:2]) <= output_hw else Image.LANCZOS
        return np.asarray(
            Image.fromarray(image).resize((output_hw, output_hw), resample=resample)
        )

    @staticmethod
    def _resize_letterbox_u8(image: np.ndarray, output_hw: int, fill_value: int) -> np.ndarray:
        if image.shape == (output_hw, output_hw):
            return image
        h, w = image.shape[:2]
        scale = min(output_hw / max(1, h), output_hw / max(1, w))
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        resized = np.asarray(Image.fromarray(image).resize((new_w, new_h), resample=Image.BILINEAR))
        canvas = np.full((output_hw, output_hw), int(fill_value), dtype=resized.dtype)
        i0 = (output_hw - new_h) // 2
        j0 = (output_hw - new_w) // 2
        canvas[i0 : i0 + new_h, j0 : j0 + new_w] = resized
        return canvas

    @staticmethod
    def _blur_u8(image: np.ndarray, sigma: float) -> np.ndarray:
        sigma = float(sigma)
        if sigma <= 0:
            return image
        return np.asarray(Image.fromarray(image).filter(ImageFilter.GaussianBlur(radius=sigma)))

    @staticmethod
    def _blur_float01(image: np.ndarray, sigma: float) -> np.ndarray:
        sigma = float(sigma)
        if sigma <= 0:
            return image.astype(np.float32, copy=False)
        u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        return np.asarray(Image.fromarray(u8).filter(ImageFilter.GaussianBlur(radius=sigma))).astype(np.float32) / 255.0

    @classmethod
    def transform_pyramid(
        cls,
        levels: np.ndarray,
        *,
        pyramid_mode: str = "raw",
        blur_sigmas: str | Sequence[float] | None = None,
        laplacian_contrast: float = 1.0,
    ) -> np.ndarray:
        mode = normalize_pyramid_mode(pyramid_mode)
        if mode == "raw":
            return levels
        sigmas = parse_blur_sigmas(blur_sigmas, int(levels.shape[0]))
        if mode == "gaussian":
            return np.stack([cls._blur_u8(level, sigma) for level, sigma in zip(levels, sigmas)], axis=0)
        if mode == "hybrid_l0_gaussian":
            out = [levels[0]]
            out.extend(cls._blur_u8(level, sigma) for level, sigma in zip(levels[1:], sigmas[1:]))
            return np.stack(out, axis=0)
        if mode == "laplacian":
            levels01 = levels.astype(np.float32) / 255.0
            bands = []
            for i, (level, sigma) in enumerate(zip(levels01, sigmas)):
                sigma = max(float(sigma), 1e-6)
                low = cls._blur_float01(level, sigma)
                if i == len(levels01) - 1:
                    bands.append(np.clip(low, 0.0, 1.0))
                else:
                    lower = cls._blur_float01(level, sigma * 2.0)
                    band = low - lower
                    bands.append(np.clip(0.5 + float(laplacian_contrast) * band, 0.0, 1.0))
            return np.stack(bands, axis=0).astype(np.float32)
        raise AssertionError(f"Unhandled pyramid_mode {mode}")

    def pyramid_rois_for_row(
        self,
        covariates: dict[str, torch.Tensor],
        row: int,
        crop_sizes: Sequence[LevelSpec] = (51, 101, 201),
        center_mode: str = "dset",
    ) -> np.ndarray:
        crop_sizes = normalize_level_specs(crop_sizes)
        saved_roi = covariates["roi"][row].cpu().numpy().astype(np.int64)

        if center_mode == "dset":
            center = saved_roi.mean(axis=1)
        elif center_mode == "gaze":
            center = self.gaze_center_ij(covariates, row)
        else:
            raise ValueError("center_mode must be 'dset' or 'gaze'")

        rois = []
        saved_h = int(saved_roi[0, 1] - saved_roi[0, 0])
        saved_w = int(saved_roi[1, 1] - saved_roi[1, 0])
        for i, spec in enumerate(crop_sizes):
            if isinstance(spec, str):
                rois.append(self.screen_roi())
                continue
            crop_size = int(spec)
            use_saved_l0 = center_mode == "dset" and i == 0 and crop_size == saved_h == saved_w
            roi = saved_roi if use_saved_l0 else self.centered_roi(center, crop_size)
            rois.append(roi.astype(np.int64))
        return np.stack(rois, axis=0)

    def pyramid_and_rois_for_row(
        self,
        covariates: dict[str, torch.Tensor],
        row: int,
        crop_sizes: Sequence[LevelSpec] = (51, 101, 201),
        output_hw: int = 64,
        center_mode: str = "dset",
        pyramid_mode: str = "raw",
        blur_sigmas: str | Sequence[float] | None = None,
        laplacian_contrast: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        trial_id = int(covariates["trial_inds"][row].item())
        crop_sizes = normalize_level_specs(crop_sizes)
        rois = self.pyramid_rois_for_row(covariates, row, crop_sizes=crop_sizes, center_mode=center_mode)
        saved_roi = covariates["roi"][row].cpu().numpy().astype(np.int64)
        saved_h = int(saved_roi[0, 1] - saved_roi[0, 0])
        saved_w = int(saved_roi[1, 1] - saved_roi[1, 0])
        levels = []
        numeric_sizes = [int(spec) for spec in crop_sizes if not isinstance(spec, str)]
        pad = max(numeric_sizes) if numeric_sizes else 0
        for level_idx, (spec, roi) in enumerate(zip(crop_sizes, rois)):
            if isinstance(spec, str):
                crop = self.screen_canvas(trial_id)
                levels.append(self._resize_letterbox_u8(crop, output_hw, int(self.trial(trial_id).bkgnd)))
            else:
                crop_size = int(spec)
                use_saved_l0 = (
                    center_mode == "dset"
                    and level_idx == 0
                    and crop_size == saved_h == saved_w
                    and self.dset is not None
                    and "stim" in self.dset.covariates
                )
                if use_saved_l0:
                    levels.append(self.saved_stim_resized(row, output_hw))
                else:
                    levels.append(self.crop_screen_roi_resized(trial_id, roi, output_hw, pad))
        pyramid = np.stack(levels, axis=0)
        pyramid = self.transform_pyramid(
            pyramid,
            pyramid_mode=pyramid_mode,
            blur_sigmas=blur_sigmas,
            laplacian_contrast=laplacian_contrast,
        )
        return pyramid, rois

    def pyramid_for_row(
        self,
        covariates: dict[str, torch.Tensor],
        row: int,
        crop_sizes: Sequence[LevelSpec] = (51, 101, 201),
        output_hw: int = 64,
        center_mode: str = "dset",
        pyramid_mode: str = "raw",
        blur_sigmas: str | Sequence[float] | None = None,
        laplacian_contrast: float = 1.0,
    ) -> np.ndarray:
        pyramid, _ = self.pyramid_and_rois_for_row(
            covariates,
            row,
            crop_sizes=crop_sizes,
            output_hw=output_hw,
            center_mode=center_mode,
            pyramid_mode=pyramid_mode,
            blur_sigmas=blur_sigmas,
            laplacian_contrast=laplacian_contrast,
        )
        return pyramid


class MarmoBackImageSequenceDataset(Dataset):
    """Short contiguous BackImage sequences for FOND/LeWM-style world models."""

    def __init__(
        self,
        session_name: str = "Allen_2022-04-13",
        dset_path: str | Path | None = None,
        split: str = "train",
        train_frac: float = 0.8,
        seed: int = 1002,
        source_hz: int = 240,
        target_hz: int = 120,
        seq_len: int = 4,
        crop_sizes: Sequence[LevelSpec] = (51, 101, 201),
        output_hw: int = 64,
        center_mode: str = "dset",
        pyramid_mode: str = "raw",
        blur_sigmas: str | Sequence[float] | None = None,
        laplacian_contrast: float = 1.0,
        action_history: int = 1,
        max_windows: int | None = None,
        window_sample_mode: str = "random",
        stride: int = 1,
        split_mode: str = "numpy",
        robs_downsample_mode: str = "sample",
        covariate_downsample_mode: str = "sample",
        validity_downsample_mode: str = "sample",
        pixel_normalization: str = "unit",
        dfs_mode: str = "none",
        dfs_valid_lags: int = 32,
        dfs_missing_threshold: float = 45.0,
    ):
        super().__init__()
        if split not in {"train", "val", "all"}:
            raise ValueError("split must be train, val, or all")
        self.session_name = session_name
        self.dset_path = Path(dset_path) if dset_path is not None else BackImagePaths(session_name).dset_path
        self.dset = load_backimage_dataset(self.dset_path)
        self.cov = self.dset.covariates
        self.split = split
        self.train_frac = float(train_frac)
        self.seed = int(seed)
        self.source_hz = int(source_hz)
        self.target_hz = int(target_hz)
        self.downsample = max(1, self.source_hz // self.target_hz)
        if self.source_hz % self.target_hz != 0:
            raise ValueError("target_hz must divide source_hz for this first adapter")
        self.seq_len = int(seq_len)
        self.crop_sizes = normalize_level_specs(crop_sizes)
        self.output_hw = int(output_hw)
        self.center_mode = center_mode
        self.pyramid_mode = normalize_pyramid_mode(pyramid_mode)
        self.blur_sigmas = None if blur_sigmas is None else tuple(parse_blur_sigmas(blur_sigmas, len(self.crop_sizes)))
        self.laplacian_contrast = float(laplacian_contrast)
        self.action_history = max(1, int(action_history))
        self.window_sample_mode = str(window_sample_mode).lower()
        if self.window_sample_mode not in {"random", "first", "last"}:
            raise ValueError("window_sample_mode must be random, first, or last")
        self.stride = max(1, int(stride))
        self.split_mode = str(split_mode).lower()
        if self.split_mode not in {"numpy", "torch"}:
            raise ValueError("split_mode must be numpy or torch")
        self.robs_downsample_mode = str(robs_downsample_mode).lower()
        if self.robs_downsample_mode not in {"sample", "sum"}:
            raise ValueError("robs_downsample_mode must be sample or sum")
        self.covariate_downsample_mode = str(covariate_downsample_mode).lower()
        if self.covariate_downsample_mode not in {"sample", "mean"}:
            raise ValueError("covariate_downsample_mode must be sample or mean")
        self.validity_downsample_mode = str(validity_downsample_mode).lower()
        if self.validity_downsample_mode not in {"sample", "all"}:
            raise ValueError("validity_downsample_mode must be sample or all")
        self.pixel_normalization = str(pixel_normalization).lower()
        if self.pixel_normalization not in {"unit", "zero_one", "0_1", "visioncore", "pixelnorm", "centered"}:
            raise ValueError("pixel_normalization must be unit or visioncore")
        self.sampler = BackImageSampler(session_name, self.dset)
        self.dfs_mode = str(dfs_mode).lower()
        if self.dfs_mode not in {"none", "valid_nlags", "visioncore"}:
            raise ValueError("dfs_mode must be none, valid_nlags, or visioncore")
        self.dfs_valid_lags = max(1, int(dfs_valid_lags))
        self.dfs_missing_threshold = float(dfs_missing_threshold)
        self._dfs_cache: torch.Tensor | None = self._build_dfs_cache()
        self.missing_image_trials = self._missing_image_trials()
        self.windows = self._build_windows(max_windows=max_windows)
        self._pixel_cache: np.ndarray | None = None
        self._pixel_cache_lookup: np.ndarray | None = None

    @property
    def action_dim(self) -> int:
        return 2 * int(self.action_history)

    @property
    def img_ch(self) -> int:
        return len(self.crop_sizes)

    def _uses_saved_l0_only(self) -> bool:
        if self.center_mode != "dset" or "stim" not in self.cov:
            return False
        if len(self.crop_sizes) != 1 or isinstance(self.crop_sizes[0], str):
            return False
        roi0 = self.cov["roi"][0]
        if torch.is_tensor(roi0):
            roi0 = roi0.detach().cpu().numpy()
        roi0 = np.asarray(roi0, dtype=np.int64)
        saved_h = int(roi0[0, 1] - roi0[0, 0])
        saved_w = int(roi0[1, 1] - roi0[1, 0])
        return int(self.crop_sizes[0]) == saved_h == saved_w

    def _trial_split(self, trial_ids: np.ndarray) -> set[int]:
        trial_ids = np.array(sorted(int(t) for t in trial_ids), dtype=np.int64)
        if self.split_mode == "torch":
            generator = torch.Generator().manual_seed(self.seed)
            perm = torch.randperm(len(trial_ids), generator=generator).cpu().numpy()
            trial_ids = trial_ids[perm]
        else:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(trial_ids)
        n_train = int(math.floor(len(trial_ids) * self.train_frac))
        if self.split == "train":
            return set(trial_ids[:n_train].tolist())
        if self.split == "val":
            return set(trial_ids[n_train:].tolist())
        return set(trial_ids.tolist())

    def _missing_image_trials(self) -> dict[int, str]:
        if self._uses_saved_l0_only():
            return {}
        trial_inds = self.cov["trial_inds"].cpu().numpy().astype(np.int64)
        missing = {}
        for trial_id in np.unique(trial_inds):
            if not self.sampler.has_image(int(trial_id)):
                missing[int(trial_id)] = self.sampler.trial(int(trial_id)).image_file
        return missing

    def _build_windows(self, max_windows: int | None = None) -> np.ndarray:
        trial_inds = self.cov["trial_inds"].cpu().numpy().astype(np.int64)
        dpi_valid = self.cov["dpi_valid"].cpu().numpy() > 0
        all_trials = np.unique(trial_inds)
        keep_trials = self._trial_split(all_trials)
        windows: list[np.ndarray] = []
        for trial_id in sorted(keep_trials):
            if int(trial_id) in self.missing_image_trials:
                continue
            trial_rows = np.flatnonzero(trial_inds == trial_id)
            n_bins = len(trial_rows) // self.downsample
            if n_bins <= 0:
                continue
            rows = trial_rows[: n_bins * self.downsample : self.downsample]
            if len(rows) < self.seq_len:
                continue
            for start in range(0, len(rows) - self.seq_len + 1, self.stride):
                win = rows[start : start + self.seq_len]
                if self._window_valid(dpi_valid, trial_inds, win):
                    windows.append(win.astype(np.int64))
        if not windows:
            raise RuntimeError(f"No valid windows found for {self.session_name} split={self.split}")
        windows_arr = np.stack(windows, axis=0)
        if max_windows is not None and len(windows_arr) > int(max_windows):
            if self.window_sample_mode == "first":
                windows_arr = windows_arr[: int(max_windows)]
            elif self.window_sample_mode == "last":
                windows_arr = windows_arr[-int(max_windows) :]
            else:
                rng = np.random.default_rng(self.seed + {"train": 0, "val": 1, "all": 2}[self.split])
                take = rng.choice(len(windows_arr), size=int(max_windows), replace=False)
                take.sort()
                windows_arr = windows_arr[take]
        return windows_arr

    def __len__(self) -> int:
        return len(self.windows)

    def precompute_pixels(self, verbose: bool = True) -> None:
        rows = np.unique(self.windows.reshape(-1))
        first = self.sampler.pyramid_for_row(
            self.cov,
            int(rows[0]),
            crop_sizes=self.crop_sizes,
            output_hw=self.output_hw,
            center_mode=self.center_mode,
            pyramid_mode=self.pyramid_mode,
            blur_sigmas=self.blur_sigmas,
            laplacian_contrast=self.laplacian_contrast,
        )
        cache = np.empty((len(rows), self.img_ch, self.output_hw, self.output_hw), dtype=first.dtype)
        cache[0] = first
        t0 = None
        if verbose:
            import time

            t0 = time.time()
            print(
                f"precomputing {len(rows)} unique rows for split={self.split} "
                f"levels={tuple(level_spec_label(x) for x in self.crop_sizes)}",
                flush=True,
            )
        for i, row in enumerate(rows[1:], start=1):
            cache[i] = self.sampler.pyramid_for_row(
                self.cov,
                int(row),
                crop_sizes=self.crop_sizes,
                output_hw=self.output_hw,
                center_mode=self.center_mode,
                pyramid_mode=self.pyramid_mode,
                blur_sigmas=self.blur_sigmas,
                laplacian_contrast=self.laplacian_contrast,
            )
            if verbose and (i + 1) % 10000 == 0:
                assert t0 is not None
                import time

                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 1e-6)
                print(f"  cached {i + 1}/{len(rows)} rows ({rate:.1f} rows/s)", flush=True)
        lookup = np.full(int(rows.max()) + 1, -1, dtype=np.int64)
        lookup[rows] = np.arange(len(rows), dtype=np.int64)
        self._pixel_cache = cache
        self._pixel_cache_lookup = lookup
        if verbose:
            import time

            assert t0 is not None
            print(f"pixel cache ready in {time.time() - t0:.1f}s, size={cache.nbytes / 1e9:.2f} GB", flush=True)

    def _window_valid(self, dpi_valid: np.ndarray, trial_inds: np.ndarray, rows: np.ndarray) -> bool:
        if not np.all(dpi_valid[rows]):
            return False
        if self.downsample <= 1 or self.validity_downsample_mode == "sample":
            return self._window_dfs_valid(rows)
        offsets = np.arange(self.downsample, dtype=np.int64)
        bin_rows = rows[:, None] + offsets[None, :]
        if int(bin_rows.max()) >= len(dpi_valid):
            return False
        trial_ids = trial_inds[rows]
        same_trial = trial_inds[bin_rows] == trial_ids[:, None]
        return bool(np.all(same_trial) and np.all(dpi_valid[bin_rows]) and self._window_dfs_valid(rows))

    def _window_dfs_valid(self, rows: np.ndarray) -> bool:
        if self._dfs_cache is None:
            return True
        down_idx = int(rows[-1]) // self.downsample
        if down_idx < 0 or down_idx >= int(self._dfs_cache.shape[0]):
            return False
        return bool(torch.any(self._dfs_cache[down_idx] > 0).item())

    def _raw_bin_rows(self, rows: np.ndarray) -> torch.Tensor:
        rows = np.asarray(rows, dtype=np.int64)
        offsets = np.arange(self.downsample, dtype=np.int64)
        return torch.from_numpy(rows[:, None] + offsets[None, :]).long()

    def _build_dfs_cache(self) -> torch.Tensor | None:
        if self.dfs_mode == "none":
            return None
        trial_inds = self.cov["trial_inds"].cpu().numpy()
        dpi_valid = self.cov["dpi_valid"].cpu().numpy().astype(np.float32)
        t_bins = self.cov["t_bins"].cpu().numpy().astype(np.float64)
        n_raw = len(trial_inds)
        n_bins = n_raw // self.downsample
        if n_bins <= 0:
            return None
        raw = np.arange(n_bins * self.downsample, dtype=np.int64).reshape(n_bins, self.downsample)
        down_trials = trial_inds[raw[:, 0]]
        down_valid = dpi_valid[raw].mean(axis=1)
        new_trials = np.empty(n_bins, dtype=bool)
        new_trials[0] = True
        new_trials[1:] = down_trials[1:] != down_trials[:-1]
        dfs = (~new_trials) & (down_valid > 0)
        for lag in range(1, self.dfs_valid_lags):
            shifted = np.zeros_like(dfs)
            shifted[lag:] = dfs[:-lag]
            dfs &= shifted
        dfs_tensor = torch.from_numpy(dfs.astype(np.float32))[:, None]
        if self.dfs_mode != "visioncore":
            return dfs_tensor

        robs = self.cov["robs"]
        n_units = int(robs.shape[1]) if robs.ndim == 2 else 1
        try:
            cids = np.asarray(self.sampler.session.get_cluster_ids())
            if len(cids) != n_units:
                warnings.warn(
                    f"get_cluster_ids returned {len(cids)} ids but robs has {n_units} units; "
                    "using the first robs-width cluster ids for missing_pct dfs",
                    RuntimeWarning,
                )
                cids = cids[:n_units]
            missing_fun = self.sampler.session.get_missing_pct_interp(cids)
            down_t = t_bins[raw].mean(axis=1)
            missing_pct = missing_fun(torch.from_numpy(down_t)).float()
            if missing_pct.ndim == 1:
                missing_pct = missing_pct[:, None]
            if missing_pct.shape[1] != n_units:
                warnings.warn(
                    f"missing_pct returned {missing_pct.shape[1]} units but robs has {n_units}; "
                    "falling back to valid_nlags-only dfs",
                    RuntimeWarning,
                )
                return dfs_tensor
            unit_mask = missing_pct < float(self.dfs_missing_threshold)
            med = torch.median(missing_pct, dim=0).values
            multi_units = med >= float(self.dfs_missing_threshold)
            unit_mask[:, multi_units] = True
            return dfs_tensor * unit_mask.float()
        except Exception as exc:
            warnings.warn(
                f"Could not build missing_pct dfs for {self.session_name}: {exc}; "
                "falling back to valid_nlags-only dfs",
                RuntimeWarning,
            )
            return dfs_tensor

    def _sample_covariate(self, key: str, rows: np.ndarray, *, mode: str = "sample") -> torch.Tensor:
        values = self.cov[key]
        if self.downsample <= 1 or mode == "sample":
            return values[rows]
        bin_rows = self._raw_bin_rows(rows).to(values.device)
        gathered = values[bin_rows]
        if mode == "sum":
            return gathered.sum(dim=1)
        if mode == "mean":
            return gathered.float().mean(dim=1)
        raise ValueError(f"Unsupported downsample mode {mode!r}")

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rows = self.windows[int(idx)]
        if self._pixel_cache is not None and self._pixel_cache_lookup is not None:
            cache_inds = self._pixel_cache_lookup[rows]
            if np.any(cache_inds < 0):
                raise RuntimeError("pixel cache lookup is missing one or more requested rows")
            raw_pixels = self._pixel_cache[cache_inds]
        else:
            pixels = [
                self.sampler.pyramid_for_row(
                    self.cov,
                    int(row),
                    crop_sizes=self.crop_sizes,
                    output_hw=self.output_hw,
                    center_mode=self.center_mode,
                    pyramid_mode=self.pyramid_mode,
                    blur_sigmas=self.blur_sigmas,
                    laplacian_contrast=self.laplacian_contrast,
                )
                for row in rows
            ]
            raw_pixels = np.stack(pixels, axis=0)
        pixels_np = normalize_pixel_values(raw_pixels, self.pixel_normalization)
        eyepos = self._sample_covariate("eyepos", rows, mode=self.covariate_downsample_mode).float()
        base_action = torch.zeros_like(eyepos)
        base_action[:-1] = eyepos[1:] - eyepos[:-1]
        if self.action_history == 1:
            action = base_action
        else:
            parts = []
            for lag in range(self.action_history):
                shifted = torch.zeros_like(base_action)
                if lag == 0:
                    shifted = base_action
                else:
                    shifted[lag:] = base_action[:-lag]
                parts.append(shifted)
            action = torch.cat(parts, dim=-1)
        robs = self._sample_covariate("robs", rows, mode=self.robs_downsample_mode).float()
        if self._dfs_cache is not None:
            down_rows = torch.from_numpy((rows // self.downsample).astype(np.int64))
            dfs = self._dfs_cache[down_rows].float()
        elif "dfs" in self.cov:
            dfs = self._sample_covariate("dfs", rows, mode=self.covariate_downsample_mode).float()
        else:
            dfs = torch.ones((len(rows), 1), dtype=torch.float32)
        t_bins = self._sample_covariate("t_bins", rows, mode=self.covariate_downsample_mode).float()
        return {
            "pixels": torch.from_numpy(pixels_np),
            "action": action,
            "eyepos": eyepos,
            "robs": robs,
            "dfs": dfs,
            "t_bins": t_bins,
            "trial_inds": self.cov["trial_inds"][rows].long(),
            "row_indices": torch.from_numpy(rows.astype(np.int64)),
        }


def _pad_last_dim(tensor: torch.Tensor, width: int) -> torch.Tensor:
    if tensor.shape[-1] == width:
        return tensor
    out = tensor.new_zeros(*tensor.shape[:-1], width)
    out[..., : tensor.shape[-1]] = tensor
    return out


def collate_marmo(batch: Iterable[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = batch[0].keys() if isinstance(batch, list) else None
    if keys is None:
        batch = list(batch)
        keys = batch[0].keys()
    out = {}
    for key in keys:
        values = [b[key] for b in batch]
        try:
            out[key] = torch.stack(values, dim=0)
        except RuntimeError:
            if key not in {"robs", "dfs"}:
                raise
            max_width = max(int(v.shape[-1]) for v in values)
            out[key] = torch.stack([_pad_last_dim(v, max_width) for v in values], dim=0)
    return out
