"""CLI: python -m lesion_masked_ita photo.jpg --out overlay.png"""

from __future__ import annotations

import argparse
import csv
import datetime
import math
import urllib.parse
from pathlib import Path

from PIL import Image

from .core import BACKENDS, load_rgb, measure_ita, overlay_mask

EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
HEADER = [
    "image", "link", "added", "backend", "lesion_fraction", "healthy_fraction",
    "ita_unmasked", "ita_masked", "ita_healthy_mean", "l_star_masked",
    "b_star_masked", "valid", "overlay",
]


def collect(inputs: list[str]) -> list[Path]:
    files = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            files.extend(p for p in sorted(path.iterdir()) if p.suffix.lower() in EXTS)
        elif path.is_file() and path.suffix.lower() in EXTS:
            files.append(path)
        else:
            raise SystemExit(f"not an image or folder: {path}")
    return files


def csv_value(value: float) -> float | str:
    return round(value, 3) if math.isfinite(value) else ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mask the lesion, then report ITA on remaining skin pixels."
    )
    parser.add_argument("inputs", nargs="+", help="image files or folders")
    parser.add_argument("--backend", choices=BACKENDS, default="kmeans")
    parser.add_argument("--out", help="Write a red overlay; requires exactly one image")
    parser.add_argument("--csv", type=Path, help="Append results and skip images already listed")
    parser.add_argument("--overlay-dir", type=Path, help="Save one review overlay per image")
    parser.add_argument("--link-prefix", default="", help="URL prefix for input image links")
    parser.add_argument("--overlay-link-prefix", default="", help="URL prefix for overlay links")
    args = parser.parse_args()

    files = collect(args.inputs)
    if args.csv and args.csv.exists():
        with args.csv.open(newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != HEADER:
                raise SystemExit(f"{args.csv} has different columns than this version writes")
            done = {row["image"] for row in reader}
        files = [p for p in files if p.name not in done]
    if not files:
        if args.csv:
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            if not args.csv.exists():
                with args.csv.open("w", newline="") as f:
                    csv.writer(f).writerow(HEADER)
            print("no new images")
            return
        raise SystemExit("no images found")
    if args.out and len(files) != 1:
        raise SystemExit("--out requires exactly one image")

    if args.overlay_dir:
        args.overlay_dir.mkdir(parents=True, exist_ok=True)
    added = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    rows = []
    for path in files:
        rgb = load_rgb(path)
        result = measure_ita(rgb, backend=args.backend)
        overlay_name = ""
        if args.out or args.overlay_dir:
            overlay_path = Path(args.out) if args.out else args.overlay_dir / f"{path.name}.overlay.png"
            Image.fromarray(overlay_mask(rgb, result["lesion_mask"])).save(overlay_path)
            overlay_name = overlay_path.name

        for key, label in (
            ("backend", "backend"),
            ("lesion_fraction", "lesion_fraction"),
            ("healthy_fraction", "healthy_fraction"),
            ("ita_unmasked", "ita_unmasked"),
            ("ita_masked", "ita_masked"),
            ("ita_healthy_mean", "ita_healthy_mean"),
            ("valid", "valid"),
        ):
            value = result[key]
            print(f"{label:<18} {value:.2f}" if isinstance(value, float) else f"{label:<18} {value}")
        if args.out:
            print(f"wrote              {args.out}")

        rows.append([
            path.name,
            args.link_prefix + urllib.parse.quote(path.name),
            added,
            result["backend"],
            csv_value(result["lesion_fraction"]),
            csv_value(result["healthy_fraction"]),
            csv_value(result["ita_unmasked"]),
            csv_value(result["ita_masked"]),
            csv_value(result["ita_healthy_mean"]),
            csv_value(result["l_star_masked"]),
            csv_value(result["b_star_masked"]),
            result["valid"],
            args.overlay_link_prefix + urllib.parse.quote(overlay_name) if overlay_name else "",
        ])

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        new = not args.csv.exists()
        with args.csv.open("a", newline="") as f:
            writer = csv.writer(f)
            if new:
                writer.writerow(HEADER)
            writer.writerows(rows)


if __name__ == "__main__":
    main()
