#!/usr/bin/env python3
"""Audit a folder-structured dermoscopy dataset with the released ensembles."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from coral_pytorch.dataset import corn_label_from_logits
from model_factory import get_efficientnet_b4_coral, get_efficientnet_b4_lab


ITA_LABELS = ["dark", "brown", "tan", "intermediate", "light", "very_light"]
FST_GROUPS = {1: "I-II", 2: "I-II", 3: "III-IV", 4: "III-IV", 5: "V-VI", 6: "V-VI"}


class ImageFolderDataset(Dataset):
    def __init__(self, files: list[Path]):
        self.files = files
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.files[index]) as image:
            image = image.convert("RGB")
            return self.transform(image), index


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ita_from_lab(l_value: np.ndarray, b_value: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan2(l_value - 50.0, b_value))


def ita_category(values: pd.Series) -> pd.Categorical:
    return pd.cut(
        values,
        bins=[-np.inf, -30, 10, 28, 41, 55, np.inf],
        labels=ITA_LABELS,
        include_lowest=True,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(path: Path, kind: str, device: torch.device) -> torch.nn.Module:
    if kind == "fp":
        model = get_efficientnet_b4_coral(num_classes=6, weights=None)
    else:
        model = get_efficientnet_b4_lab(num_outputs=3, weights=None)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint)
    return model.eval().to(device)


def distribution_table(df: pd.DataFrame, column: str) -> list[dict]:
    counts = df[column].value_counts(dropna=False, sort=False)
    return [
        {"category": str(category), "count": int(count), "percent": round(100 * count / len(df), 2)}
        for category, count in counts.items()
    ]


def markdown_table(rows: list[dict]) -> str:
    if not rows:
        return "No rows."
    columns = list(rows[0])
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(str(row[column]) for column in columns) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--experiments", type=Path, default=Path("experiments"))
    parser.add_argument("--output-dir", type=Path, default=Path("audit_results/kaggle_melanoma"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    files = sorted(
        path
        for path in args.dataset_root.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"No images found under {args.dataset_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    print(f"Found {len(files):,} images; using {device}", flush=True)

    dataset = ImageFolderDataset(files)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    n_images = len(files)
    fp_folds = np.zeros((5, n_images), dtype=np.int8)
    lab_folds = np.zeros((5, n_images, 3), dtype=np.float32)

    for fold in range(5):
        fp_path = args.experiments / "finetune_mskcc_blur0" / f"fold_{fold}" / "model.pth"
        lab_path = args.experiments / "finetune_mskcc_blur0_lab" / f"fold_{fold}" / "model.pth"
        if not fp_path.exists() or not lab_path.exists():
            raise SystemExit(f"Missing checkpoint for fold {fold}")
        fp_model = load_model(fp_path, "fp", device)
        lab_model = load_model(lab_path, "lab", device)
        print(f"Running fold {fold + 1}/5", flush=True)
        with torch.inference_mode():
            for batch_number, (images, indices) in enumerate(loader, start=1):
                images = images.to(device)
                fp_logits = fp_model(images)
                fp_predictions = corn_label_from_logits(fp_logits).cpu().numpy() + 1
                lab_predictions = lab_model(images).cpu().numpy()
                batch_indices = indices.numpy()
                fp_folds[fold, batch_indices] = fp_predictions
                lab_folds[fold, batch_indices] = lab_predictions
                if batch_number % 50 == 0 or batch_number == len(loader):
                    print(f"  fold {fold + 1}: {batch_number}/{len(loader)} batches", flush=True)
        del fp_model, lab_model
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

    relative_paths = [path.relative_to(args.dataset_root).as_posix() for path in files]
    result = pd.DataFrame(
        {
            "file": [str(path) for path in files],
            "relative_path": relative_paths,
            "split": [path.relative_to(args.dataset_root).parts[0] for path in files],
            "diagnosis_folder": [path.relative_to(args.dataset_root).parts[1] for path in files],
        }
    )
    for fold in range(5):
        result[f"fitzpatrick_fold_{fold}"] = fp_folds[fold]
        result[f"L_fold_{fold}"] = lab_folds[fold, :, 0]
        result[f"a_fold_{fold}"] = lab_folds[fold, :, 1]
        result[f"b_fold_{fold}"] = lab_folds[fold, :, 2]
        result[f"ita_fold_{fold}"] = ita_from_lab(lab_folds[fold, :, 0], lab_folds[fold, :, 2])

    fp_modes = []
    fp_agreement = []
    for index in range(n_images):
        counts = Counter(fp_folds[:, index].tolist())
        mode, count = max(counts.items(), key=lambda item: (item[1], -item[0]))
        fp_modes.append(mode)
        fp_agreement.append(count / 5)
    result["fitzpatrick_pred"] = fp_modes
    result["fitzpatrick_group"] = result["fitzpatrick_pred"].map(FST_GROUPS)
    result["fitzpatrick_fold_agreement"] = fp_agreement

    for component_index, component in enumerate(["L", "a", "b"]):
        result[f"{component}_pred"] = lab_folds[:, :, component_index].mean(axis=0)
        result[f"{component}_fold_std"] = lab_folds[:, :, component_index].std(axis=0)
    ita_columns = [f"ita_fold_{fold}" for fold in range(5)]
    result["ita_pred"] = result[ita_columns].mean(axis=1)
    result["ita_fold_std"] = result[ita_columns].std(axis=1)
    result["ita_category"] = ita_category(result["ita_pred"])
    result["review_recommended"] = (
        (result["fitzpatrick_fold_agreement"] < 0.8) | (result["ita_fold_std"] > 10)
    )

    print("Checking exact duplicate files", flush=True)
    result["sha256"] = [sha256(path) for path in files]
    duplicate_hashes = result.groupby("sha256").filter(lambda group: len(group) > 1)
    cross_split_duplicates = duplicate_hashes.groupby("sha256").filter(
        lambda group: group["split"].nunique() > 1
    )

    predictions_path = args.output_dir / "predictions.csv"
    result.to_csv(predictions_path, index=False)

    fst_rows = distribution_table(result, "fitzpatrick_group")
    ita_rows = distribution_table(result, "ita_category")
    subgroup_rows = []
    for diagnosis, group in result.groupby("diagnosis_folder"):
        subgroup_rows.append(
            {
                "diagnosis_folder": diagnosis,
                "images": int(len(group)),
                "FST_V_VI_percent": round(
                    float(100 * (group["fitzpatrick_group"] == "V-VI").mean()), 2
                ),
                "median_ITA": round(float(group["ita_pred"].median()), 2),
                "review_percent": round(
                    float(100 * group["review_recommended"].mean()), 2
                ),
            }
        )

    summary = {
        "images": n_images,
        "device": str(device),
        "fitzpatrick_groups": fst_rows,
        "ita_categories": ita_rows,
        "diagnosis_subgroups": subgroup_rows,
        "median_ita": round(float(result["ita_pred"].median()), 3),
        "review_recommended_count": int(result["review_recommended"].sum()),
        "review_recommended_percent": round(100 * float(result["review_recommended"].mean()), 2),
        "exact_duplicate_images": int(len(duplicate_hashes)),
        "cross_split_duplicate_images": int(len(cross_split_duplicates)),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fst_dark_percent = next(
        (row["percent"] for row in fst_rows if row["category"] == "V-VI"), 0.0
    )
    warnings = []
    if fst_dark_percent < 10:
        warnings.append(
            f"Predicted FST V-VI coverage is only {fst_dark_percent:.2f}%; darker skin appears underrepresented."
        )
    if summary["review_recommended_percent"] > 10:
        warnings.append(
            f"{summary['review_recommended_percent']:.2f}% of images have high ensemble disagreement and should be reviewed."
        )
    if summary["cross_split_duplicate_images"]:
        warnings.append(
            f"Found {summary['cross_split_duplicate_images']} exact duplicate images spanning train and test folders."
        )
    if not warnings:
        warnings.append("No threshold-based screening warning was triggered.")

    report = f"""# Skin-tone representation screening audit

## Scope

This audit applies the five-fold Fitzpatrick and colorimeter-supervised Lab/ITA ensembles released with `nn_colorimetry_dermatoscopy` to {n_images:,} dermoscopic images. The source dataset has diagnosis-folder labels but no skin-tone ground truth, patient IDs, acquisition metadata, or rateability labels. Predictions therefore describe the model's estimated image distribution; they do not establish demographic identity or equal clinical performance.

## Screening findings

{chr(10).join(f'- {warning}' for warning in warnings)}

- Median predicted ITA: {summary['median_ita']:.2f} degrees.
- Review recommended by the unvalidated ensemble-disagreement heuristic: {summary['review_recommended_count']:,}/{n_images:,} ({summary['review_recommended_percent']:.2f}%).
- Exact duplicate image rows: {summary['exact_duplicate_images']:,}; exact duplicates spanning train/test: {summary['cross_split_duplicate_images']:,}.

### Predicted Fitzpatrick groups

{markdown_table(fst_rows)}

### Predicted ITA categories

{markdown_table(ita_rows)}

### Diagnosis-folder comparison

{markdown_table(subgroup_rows)}

## Interpretation limits

- Fitzpatrick is presented only as compatibility metadata; ITA is the preferred continuous estimate.
- The model was validated on a small MSKCC dermoscopy/colorimeter cohort, not on this Kaggle collection.
- Lesions, black borders, hair, lighting, and acquisition devices can influence estimates.
- Folder names are diagnosis labels, not patient demographics; subgroup differences may reflect lesion appearance or source composition.
- Ensemble agreement is not calibrated confidence and does not replace a dermatologist rateability review.
- A defensible equality claim requires representative patient-level ground truth and downstream diagnostic performance by tone, neither of which this dataset supplies.
"""
    (args.output_dir / "REPORT.md").write_text(report)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fst_order = ["I-II", "III-IV", "V-VI"]
    fst_counts = result["fitzpatrick_group"].value_counts().reindex(fst_order, fill_value=0)
    axes[0].bar(fst_counts.index, 100 * fst_counts.values / n_images, color=["#e8c39e", "#b97945", "#58351f"])
    axes[0].set_ylabel("Images (%)")
    axes[0].set_title("Predicted Fitzpatrick groups")
    axes[0].set_ylim(0, max(5, 1.15 * 100 * fst_counts.max() / n_images))

    ita_counts = result["ita_category"].value_counts().reindex(ITA_LABELS, fill_value=0)
    axes[1].bar(ita_counts.index, 100 * ita_counts.values / n_images, color="#5c6bc0")
    axes[1].set_ylabel("Images (%)")
    axes[1].set_title("Predicted ITA categories")
    axes[1].tick_params(axis="x", rotation=35)
    figure.suptitle("Kaggle melanoma dataset: skin-tone representation screening")
    figure.tight_layout()
    figure.savefig(args.output_dir / "distribution.png", dpi=180)
    plt.close(figure)

    print(f"Saved audit to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
