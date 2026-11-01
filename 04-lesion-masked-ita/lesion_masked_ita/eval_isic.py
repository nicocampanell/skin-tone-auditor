"""Eval lesion backends on ISIC 2016 Part 1 test (Dice + GT-masked ITA error).

Usage:
  python -m lesion_masked_ita.eval_isic /path/to/ISIC \\
      --backends kmeans,unet,sam_adapter --prompt center

Expects MSA-style layout under data_path:
  ISBI2016_ISIC_Part1_Test_GroundTruth.csv
  (columns: index, image_relpath, mask_relpath)
  plus the referenced image/mask files.

Does not download weights or run unless invoked.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import rgb2lab

from lesion_masked_ita.core import BACKENDS, ita_deg, lesion_mask, measure_ita


def _load_pairs(data_path: Path) -> list[tuple[Path, Path]]:
    csv_path = data_path / "ISBI2016_ISIC_Part1_Test_GroundTruth.csv"
    pairs: list[tuple[Path, Path]] = []
    with csv_path.open(newline="", encoding="gbk") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            img = data_path / row[1]
            msk = data_path / row[2]
            pairs.append((img, msk))
    return pairs


def _dice(pred: np.ndarray, gt: np.ndarray) -> float:
    p = pred.astype(bool)
    g = gt.astype(bool)
    inter = int((p & g).sum())
    denom = int(p.sum() + g.sum())
    return 1.0 if denom == 0 else 2.0 * inter / denom


def _ita_healthy(rgb: np.ndarray, lesion: np.ndarray) -> float:
    lab = rgb2lab(rgb)
    keep = (lab[..., 0] >= 8.0) & ~lesion.astype(bool)
    if keep.sum() < 50:
        return float("nan")
    return ita_deg(float(np.median(lab[..., 0][keep])), float(np.median(lab[..., 2][keep])))


def _gt_click_mask(rgb: np.ndarray, gt: np.ndarray, backend: str) -> np.ndarray:
    """For sam_adapter with --prompt gt: temporarily monkey via measure path.

    ponytail: GT click only used here for upper-bound Dice; production uses center.
    When prompt is gt we still call lesion_mask for non-sam backends; for sam we
    import the module and run a one-off forward with a GT-sampled point.
    """
    if backend != "sam_adapter":
        mask, _ = lesion_mask(rgb, backend)
        return mask

    import torch
    import torch.nn.functional as F
    import cv2

    from lesion_masked_ita.sam_adapter import IMAGE_SIZE, _load_net

    ys, xs = np.where(gt)
    if len(xs) == 0:
        mask, _ = lesion_mask(rgb, "sam_adapter")
        return mask
    idx = int(np.random.randint(len(xs)))
    # map original GT click into 1024 space
    h, w = rgb.shape[:2]
    px = xs[idx] * (IMAGE_SIZE / w)
    py = ys[idx] * (IMAGE_SIZE / h)

    net, device = _load_net()
    resized = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized.astype(np.float32).transpose(2, 0, 1)).unsqueeze(0).to(device)
    coords = torch.tensor([[[float(px), float(py)]]], device=device)
    labels = torch.tensor([[1]], dtype=torch.int, device=device)
    with torch.no_grad():
        imgs = net.preprocess(tensor)
        emb = net.image_encoder(imgs)
        sparse, dense = net.prompt_encoder(points=(coords, labels), boxes=None, masks=None)
        pred, _ = net.mask_decoder(
            image_embeddings=emb,
            image_pe=net.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        pred = F.interpolate(pred, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
        prob = torch.sigmoid(pred)[0, 0].detach().cpu().numpy()
    mask = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR) >= 0.5
    return mask.astype(bool)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_path", type=Path)
    parser.add_argument(
        "--backends",
        default="kmeans,unet,sam_adapter",
        help="Comma-separated backends to score",
    )
    parser.add_argument(
        "--prompt",
        choices=("center", "gt"),
        default="center",
        help="sam_adapter prompt: center (deploy) or gt click (upper bound)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max images (0 = all)")
    args = parser.parse_args()

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    for b in backends:
        if b not in BACKENDS:
            raise SystemExit(f"unknown backend {b!r}; choose from {BACKENDS}")

    pairs = _load_pairs(args.data_path)
    if args.limit:
        pairs = pairs[: args.limit]

    scores = {b: {"dice": [], "ita_abs": []} for b in backends}

    for img_path, msk_path in pairs:
        rgb = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        gt = np.asarray(Image.open(msk_path).convert("L")) >= 128
        gt_ita = _ita_healthy(rgb, gt)

        for b in backends:
            if b == "sam_adapter" and args.prompt == "gt":
                pred = _gt_click_mask(rgb, gt, b)
                used = "sam_adapter"
                result_ita = None
            else:
                result = measure_ita(rgb, backend=b)
                pred = result["lesion_mask"]
                used = result["backend"]
                result_ita = result["ita_masked"] if result["valid"] else float("nan")

            scores[b]["dice"].append(_dice(pred, gt))
            if result_ita is None:
                pred_ita = _ita_healthy(rgb, pred)
            else:
                pred_ita = result_ita
            if not (np.isnan(gt_ita) or np.isnan(pred_ita)):
                scores[b]["ita_abs"].append(abs(pred_ita - gt_ita))
            if used != b:
                print(f"note: requested {b}, fell back to {used} on {img_path.name}")

    print(f"{'backend':<14} {'n':>5} {'dice':>8} {'|dITA|':>8}")
    for b in backends:
        d = np.array(scores[b]["dice"], dtype=np.float64)
        e = np.array(scores[b]["ita_abs"], dtype=np.float64)
        dice = float(np.nanmean(d)) if len(d) else float("nan")
        err = float(np.nanmean(e)) if len(e) else float("nan")
        print(f"{b:<14} {len(d):5d} {dice:8.3f} {err:8.2f}")


if __name__ == "__main__":
    main()
