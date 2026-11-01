"""Synthetic check: a dark lesion must not dominate masked ITA."""

from __future__ import annotations

import numpy as np

from skimage.color import rgb2lab

from lesion_masked_ita.core import ita_deg, mask_lesion_kmeans, measure_ita


def fake_derm_photo(size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    rgb[:] = (210, 168, 140)  # light skin
    yy, xx = np.ogrid[:size, :size]
    lesion = (xx - size * 0.5) ** 2 + (yy - size * 0.5) ** 2 <= (size * 0.28) ** 2
    rgb[lesion] = (40, 18, 12)  # dark lesion
    return rgb, lesion


def main() -> None:
    import os

    rgb, true_lesion = fake_derm_photo()
    pred = mask_lesion_kmeans(rgb)
    overlap = (pred & true_lesion).sum() / max(true_lesion.sum(), 1)
    assert overlap > 0.7, f"k-means missed the lesion (overlap={overlap:.2f})"

    result = measure_ita(rgb, backend="kmeans")
    assert result["valid"]
    # Whole-image ITA is pulled down by the dark blob; masked ITA should stay high.
    assert result["ita_unmasked"] < result["ita_masked"] - 8, result
    skin_ita = ita_deg(*_median_lab(rgb, ~true_lesion))
    assert abs(result["ita_masked"] - skin_ita) < 6, result
    assert result["ita_healthy_mean"] > result["ita_unmasked"] + 8, result
    print(
        "ok",
        {
            k: result[k]
            for k in ("ita_unmasked", "ita_masked", "ita_healthy_mean", "lesion_fraction")
        },
    )

    # Heavy: downloads SAM + melanoma adapter. Skip unless explicitly requested.
    if os.environ.get("RUN_SAM_CHECK") == "1":
        sam = measure_ita(rgb, backend="sam_adapter")
        assert sam["backend"] in ("sam_adapter", "kmeans"), sam
        print("ok sam_adapter", {"backend": sam["backend"], "valid": sam["valid"]})


def _median_lab(rgb, keep):
    lab = rgb2lab(rgb)
    return float(np.median(lab[..., 0][keep])), float(np.median(lab[..., 2][keep]))


if __name__ == "__main__":
    main()
