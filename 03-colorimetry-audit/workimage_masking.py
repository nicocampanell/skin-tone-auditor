"""Mask-aware preprocessing for already-masked workimage pairs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from skimage.color import rgb2lab


@dataclass(frozen=True)
class WorkimagePair:
    identifier: str
    image_path: Path
    mask_path: Path


def discover_pairs(root: Path) -> list[WorkimagePair]:
    pairs: list[WorkimagePair] = []
    for image_path in sorted(root.glob("*_lesion.png")):
        identifier = image_path.name.removesuffix("_lesion.png")
        mask_path = root / f"{identifier}_mask.png"
        if not mask_path.exists():
            raise ValueError(f"Missing mask: {mask_path.name}")
        pairs.append(WorkimagePair(identifier, image_path, mask_path))
    if not pairs:
        raise ValueError(f"No *_lesion.png files found in {root}")
    return pairs


def valid_skin_mask(rgb: np.ndarray, supplied_mask: np.ndarray) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("RGB image must have shape (height, width, 3)")
    if supplied_mask.ndim == 3:
        supplied_mask = supplied_mask[..., 0]
    if supplied_mask.shape != rgb.shape[:2]:
        raise ValueError("Image and mask dimensions do not match")
    pure_black = np.all(rgb[..., :3] == 0, axis=2)
    return (supplied_mask >= 128) & ~pure_black


def masked_pixel_ita(
    rgb: np.ndarray,
    supplied_mask: np.ndarray,
    *,
    minimum_pixels: int = 50,
) -> dict[str, float | int | bool]:
    keep = valid_skin_mask(rgb, supplied_mask)
    valid_pixels = int(keep.sum())
    if valid_pixels < minimum_pixels:
        return {
            "ita": float("nan"),
            "l_star": float("nan"),
            "b_star": float("nan"),
            "valid_pixels": valid_pixels,
            "valid_fraction": float(keep.mean()),
            "valid": False,
        }
    skin_rgb = rgb[..., :3][keep].reshape(-1, 1, 3)
    skin_lab = rgb2lab(skin_rgb).reshape(-1, 3)
    l_star = float(np.median(skin_lab[:, 0]))
    b_star = float(np.median(skin_lab[:, 2]))
    ita = float(np.degrees(np.arctan2(l_star - 50.0, b_star)))
    return {
        "ita": ita,
        "l_star": l_star,
        "b_star": b_star,
        "valid_pixels": valid_pixels,
        "valid_fraction": float(keep.mean()),
        "valid": True,
    }


def neutral_fill(rgb: np.ndarray, keep: np.ndarray) -> np.ndarray:
    if keep.shape != rgb.shape[:2]:
        raise ValueError("Image and valid-skin mask dimensions do not match")
    if not np.any(keep):
        raise ValueError("Cannot fill an image with no valid skin pixels")
    fill_color = np.median(rgb[..., :3][keep], axis=0).astype(np.uint8)
    filled = rgb[..., :3].copy()
    filled[~keep] = fill_color
    return filled
