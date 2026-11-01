"""Mask the lesion, then compute ITA on remaining healthy-skin pixels.

Benčević's published ITA eval used ground-truth masks. The inference path in
external/nn_colorimetry_dermatoscopy/cv_ita_estimation.py removed that step
and always calls extract_skin(..., label=None), so lesion pixels still enter
the color estimate.

Backends:
  kmeans       — no extra weights; Lab clustering (fallback everywhere)
  unet         — ISIC 2018 U-Net (dermoscopy). Not trained on clinical photos.
  sam_adapter  — SAM ViT-B + Melanoma_Photo adapter (dermoscopy only).
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.color import rgb2lab

BACKENDS = ("kmeans", "unet", "sam_adapter")
ISIC_UNET_REPO = "Jesusrodriguezf90/unet-resnet34-isic2018-segmentation"
ISIC_UNET_FILE = "best_unet_resnet34.pth"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MIN_HEALTHY_FRACTION = 0.05


def load_rgb(path: str | Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def ita_deg(l_star: float, b_star: float) -> float:
    return float(np.degrees(np.arctan2(l_star - 50.0, b_star)))


def _ita_on_pixels(lab: np.ndarray, keep: np.ndarray, robust: bool) -> dict:
    n = int(keep.sum())
    if n < 50:
        return {
            "ita": math.nan,
            "l_star": math.nan,
            "b_star": math.nan,
            "n_pixels": n,
            "valid": False,
        }
    l_vals = lab[..., 0][keep]
    b_vals = lab[..., 2][keep]
    if robust:
        l_use = float(np.median(l_vals))
        b_use = float(np.median(b_vals))
        l_sd = float(np.std(l_vals)) or 1.0
        b_sd = float(np.std(b_vals)) or 1.0
        inliers = (np.abs(l_vals - l_use) <= l_sd) & (np.abs(b_vals - b_use) <= b_sd)
        if int(inliers.sum()) >= 50:
            l_use = float(np.median(l_vals[inliers]))
            b_use = float(np.median(b_vals[inliers]))
    else:
        l_use = float(np.mean(l_vals))
        b_use = float(np.mean(b_vals))
    return {
        "ita": ita_deg(l_use, b_use),
        "l_star": l_use,
        "b_star": b_use,
        "n_pixels": n,
        "valid": True,
    }


def mask_lesion_kmeans(rgb: np.ndarray) -> np.ndarray:
    """Mark the darker/more-chromatic Lab cluster as lesion.

    ponytail: 3-cluster Lab k-means, not a trained segmenter. Fails when
    lesion ≈ skin (low contrast, darker tones). Upgrade: sDDI-trained U-Net.
    """
    lab = rgb2lab(rgb).astype(np.float32)
    h, w = lab.shape[:2]
    pixels = lab.reshape(-1, 3)
    dark = pixels[:, 0] < 8.0
    usable = pixels[~dark]
    if len(usable) < 100:
        return np.zeros((h, w), dtype=bool)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.5)
    _, labels, centers = cv2.kmeans(
        usable, 3, None, criteria, 4, cv2.KMEANS_PP_CENTERS
    )
    labels = labels.flatten()
    counts = np.bincount(labels, minlength=3)
    skin_id = int(np.argmax(counts))
    lesion_id = int(np.argmin(centers[:, 0]))
    if lesion_id == skin_id:
        chroma = np.sqrt(centers[:, 1] ** 2 + centers[:, 2] ** 2)
        chroma[skin_id] = -1
        lesion_id = int(np.argmax(chroma))

    full = np.full(h * w, -1, dtype=np.int32)
    full[~dark] = labels
    mask = (full.reshape(h, w) == lesion_id)
    if mask.mean() > 0.45:
        return np.zeros((h, w), dtype=bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def mask_lesion_unet(rgb: np.ndarray) -> np.ndarray:
    """ISIC 2018 ResNet34 U-Net. Dermoscopy only; do not trust on F17k/SCIN."""
    import torch
    import segmentation_models_pytorch as smp
    from huggingface_hub import hf_hub_download

    weights = hf_hub_download(ISIC_UNET_REPO, ISIC_UNET_FILE)
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    h, w = rgb.shape[:2]
    x = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA)
    x = x.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)[0, 0]
        pred = torch.sigmoid(logits).numpy()
    mask = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR) >= 0.5
    return mask.astype(bool)


def _mask_looks_bad(mask: np.ndarray) -> bool:
    frac = float(mask.mean())
    return frac < 0.001 or frac > 0.85


def lesion_mask(rgb: np.ndarray, backend: str = "kmeans") -> tuple[np.ndarray, str]:
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    if backend == "unet":
        try:
            mask = mask_lesion_unet(rgb)
            if not _mask_looks_bad(mask):
                return mask, "unet"
        except Exception:
            pass
        return mask_lesion_kmeans(rgb), "kmeans"
    if backend == "sam_adapter":
        try:
            from .sam_adapter import mask_lesion_sam_adapter

            mask = mask_lesion_sam_adapter(rgb)
            if not _mask_looks_bad(mask):
                return mask, "sam_adapter"
        except Exception:
            pass
        return mask_lesion_kmeans(rgb), "kmeans"
    return mask_lesion_kmeans(rgb), "kmeans"


def measure_ita(rgb: np.ndarray, backend: str = "kmeans") -> dict:
    lab = rgb2lab(rgb)
    not_void = lab[..., 0] >= 8.0
    mask, used = lesion_mask(rgb, backend)
    healthy = not_void & ~mask
    unmasked = _ita_on_pixels(lab, not_void, robust=False)
    masked = _ita_on_pixels(lab, healthy, robust=True)
    healthy_mean = _ita_on_pixels(lab, healthy, robust=False)
    healthy_frac = float(healthy.mean())
    valid = bool(masked["valid"] and healthy_frac >= MIN_HEALTHY_FRACTION)
    return {
        "backend": used,
        "lesion_fraction": float(mask.mean()),
        "healthy_fraction": healthy_frac,
        "ita_unmasked": unmasked["ita"],
        "ita_masked": masked["ita"],
        "ita_healthy_mean": healthy_mean["ita"],
        "l_star_masked": masked["l_star"],
        "b_star_masked": masked["b_star"],
        "valid": valid,
        "lesion_mask": mask,
    }


def overlay_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    out[mask] = (0.55 * out[mask] + np.array([180, 40, 40])).astype(np.uint8)
    return out
