#!/usr/bin/env python3
"""Run mask-exclusive ITA and released neural ensembles on workimage pairs."""

from __future__ import annotations

import argparse
import gc
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from coral_pytorch.dataset import corn_label_from_logits
from model_factory import get_efficientnet_b4_coral, get_efficientnet_b4_lab
from workimage_masking import (
    discover_pairs,
    masked_pixel_ita,
    neutral_fill,
    valid_skin_mask,
)


ITA_LABELS = ["dark", "brown", "tan", "intermediate", "light", "very_light"]
FST_GROUPS = {1: "I-II", 2: "I-II", 3: "III-IV", 4: "III-IV", 5: "V-VI", 6: "V-VI"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("workimage_results"))
    parser.add_argument("--experiments", type=Path, default=Path("experiments"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    pairs = discover_pairs(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = args.output_dir / "filled_previews"
    preview_dir.mkdir(exist_ok=True)

    image_transform = transforms.Compose(
        [
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    rows: list[dict] = []
    tensors: list[torch.Tensor] = []
    for pair in pairs:
        rgb = np.asarray(Image.open(pair.image_path).convert("RGB"), dtype=np.uint8)
        supplied_mask = np.asarray(Image.open(pair.mask_path).convert("L"), dtype=np.uint8)
        measurement = masked_pixel_ita(rgb, supplied_mask)
        keep = valid_skin_mask(rgb, supplied_mask)
        filled = neutral_fill(rgb, keep)
        Image.fromarray(filled).save(preview_dir / f"{pair.identifier}_filled.png")
        tensors.append(image_transform(Image.fromarray(filled)))
        rows.append(
            {
                "id": pair.identifier,
                "image_path": str(pair.image_path),
                "mask_path": str(pair.mask_path),
                "width": int(rgb.shape[1]),
                "height": int(rgb.shape[0]),
                "valid_skin_pixels": int(measurement["valid_pixels"]),
                "valid_skin_fraction": float(measurement["valid_fraction"]),
                "masked_pixel_L": float(measurement["l_star"]),
                "masked_pixel_b": float(measurement["b_star"]),
                "masked_pixel_ita": float(measurement["ita"]),
                "pixel_ita_valid": bool(measurement["valid"]),
            }
        )

    batch = torch.stack(tensors)
    if args.device != "auto":
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Running {len(pairs)} pairs on {device}", flush=True)

    fp_folds = np.zeros((5, len(pairs)), dtype=np.int8)
    lab_folds = np.zeros((5, len(pairs), 3), dtype=np.float32)
    with torch.inference_mode():
        for fold in range(5):
            fp_path = args.experiments / "finetune_mskcc_blur0" / f"fold_{fold}" / "model.pth"
            lab_path = args.experiments / "finetune_mskcc_blur0_lab" / f"fold_{fold}" / "model.pth"
            if not fp_path.exists() or not lab_path.exists():
                raise FileNotFoundError(f"Missing released checkpoint for fold {fold}")

            fp_model = get_efficientnet_b4_coral(num_classes=6, weights=None)
            fp_model.load_state_dict(torch.load(fp_path, map_location="cpu", weights_only=True))
            fp_model = fp_model.eval().to(device)
            lab_model = get_efficientnet_b4_lab(num_outputs=3, weights=None)
            lab_model.load_state_dict(torch.load(lab_path, map_location="cpu", weights_only=True))
            lab_model = lab_model.eval().to(device)

            device_batch = batch.to(device)
            fp_folds[fold] = corn_label_from_logits(fp_model(device_batch)).cpu().numpy() + 1
            lab_folds[fold] = lab_model(device_batch).cpu().numpy()
            print(f"Completed fold {fold + 1}/5", flush=True)

            del fp_model, lab_model, device_batch
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()

    results = pd.DataFrame(rows)
    ita_folds = np.degrees(
        np.arctan2(lab_folds[:, :, 0] - 50.0, lab_folds[:, :, 2])
    )
    for fold in range(5):
        results[f"fitzpatrick_fold_{fold}"] = fp_folds[fold]
        results[f"neural_ita_fold_{fold}"] = ita_folds[fold]

    modes: list[int] = []
    agreements: list[float] = []
    for index in range(len(pairs)):
        counts = Counter(fp_folds[:, index].tolist())
        mode, count = max(counts.items(), key=lambda item: (item[1], -item[0]))
        modes.append(mode)
        agreements.append(count / 5)
    results["fitzpatrick_pred"] = modes
    results["fitzpatrick_group"] = results["fitzpatrick_pred"].map(FST_GROUPS)
    results["fitzpatrick_fold_agreement"] = agreements
    results["neural_ita"] = ita_folds.mean(axis=0)
    results["neural_ita_fold_std"] = ita_folds.std(axis=0)
    results["ita_difference"] = results["neural_ita"] - results["masked_pixel_ita"]
    results["masked_pixel_ita_category"] = pd.cut(
        results["masked_pixel_ita"],
        bins=[-np.inf, -30, 10, 28, 41, 55, np.inf],
        labels=ITA_LABELS,
        include_lowest=True,
    )
    results["review_recommended"] = (
        ~results["pixel_ita_valid"]
        | (results["fitzpatrick_fold_agreement"] < 0.8)
        | (results["neural_ita_fold_std"] > 10)
        | (results["ita_difference"].abs() > 15)
    )

    results_path = args.output_dir / "predictions.csv"
    results.to_csv(results_path, index=False)

    ita_counts = (
        results["masked_pixel_ita_category"]
        .value_counts()
        .reindex(ITA_LABELS, fill_value=0)
    )
    fst_counts = results["fitzpatrick_group"].value_counts().reindex(
        ["I-II", "III-IV", "V-VI"], fill_value=0
    )
    summary = {
        "images": int(len(results)),
        "device": str(device),
        "median_masked_pixel_ita": round(float(results["masked_pixel_ita"].median()), 3),
        "median_neural_ita": round(float(results["neural_ita"].median()), 3),
        "median_absolute_ita_difference": round(float(results["ita_difference"].abs().median()), 3),
        "review_recommended": int(results["review_recommended"].sum()),
        "ita_categories": {str(k): int(v) for k, v in ita_counts.items()},
        "fitzpatrick_groups": {str(k): int(v) for k, v in fst_counts.items()},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    table_rows = []
    for row in results.itertuples():
        table_rows.append(
            f"| {row.id} | {row.valid_skin_fraction:.1%} | {row.masked_pixel_ita:.1f} | "
            f"{row.neural_ita:.1f} | {row.fitzpatrick_pred} | "
            f"{row.fitzpatrick_fold_agreement:.0%} | {'yes' if row.review_recommended else 'no'} |"
        )
    report = f"""# Mask-exclusive workimage audit

## Result

- Images analyzed: {len(results)}
- Median mask-exclusive pixel ITA: {summary['median_masked_pixel_ita']:.2f}°
- Median neural ITA after neutral filling: {summary['median_neural_ita']:.2f}°
- Median absolute difference between methods: {summary['median_absolute_ita_difference']:.2f}°
- Images recommended for review: {summary['review_recommended']}/{len(results)}

Pure-black pixels and black mask regions were excluded from pixel ITA. Before neural inference, excluded pixels were replaced with each image's median retained-skin color, so black was not supplied to the released model.

## Per-image results

| ID | Valid skin | Pixel ITA | Neural ITA | FST | Fold agreement | Review |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
{chr(10).join(table_rows)}

## Interpretation limits

- Mask-exclusive pixel ITA is the primary result.
- Neural ITA and Fitzpatrick are secondary experimental outputs from a model trained for dermoscopy; several source images appear to be clinical photographs.
- Median filling removes black as an input signal but is outside the released model's original validation protocol.
- Review thresholds are operational flags, not calibrated clinical confidence.
- These outputs estimate image appearance, not race, ethnicity, diagnosis, or patient identity.
"""
    (args.output_dir / "REPORT.md").write_text(report)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(
        results["masked_pixel_ita"],
        results["neural_ita"],
        c=results["review_recommended"].map({False: "#3a7d44", True: "#c4513f"}),
    )
    bounds = [
        float(min(results["masked_pixel_ita"].min(), results["neural_ita"].min())),
        float(max(results["masked_pixel_ita"].max(), results["neural_ita"].max())),
    ]
    axes[0].plot(bounds, bounds, "--", color="#777777", linewidth=1)
    axes[0].set_xlabel("Mask-exclusive pixel ITA (°)")
    axes[0].set_ylabel("Neural ITA after neutral fill (°)")
    axes[0].set_title("Method comparison")

    axes[1].bar(ita_counts.index, ita_counts.values, color="#5c6bc0")
    axes[1].set_ylabel("Images")
    axes[1].set_title("Mask-exclusive ITA categories")
    axes[1].tick_params(axis="x", rotation=35)
    figure.tight_layout()
    figure.savefig(args.output_dir / "comparison.png", dpi=180)
    plt.close(figure)
    print(f"Saved results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
