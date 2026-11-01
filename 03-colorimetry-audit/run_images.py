"""Run the released Fitzpatrick and ITA models on your own images.

Usage:
  colorimetry photo.jpg
  colorimetry photo.jpg another.png
  colorimetry /path/to/folder
  colorimetry https://example.com/photo.jpg
  colorimetry --csv results.csv /path/to/folder   # append new images as rows, skip ones already in the CSV
"""
import argparse
import csv
import datetime
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from model_factory import get_efficientnet_b4_coral, get_efficientnet_b4_lab

EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
ROOT = Path(__file__).resolve().parent
FP_DIR = ROOT / 'experiments/finetune_mskcc_blur0'
LAB_DIR = ROOT / 'experiments/finetune_mskcc_blur0_lab'
PREP = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
HEADER = ['image', 'link', 'added', 'fitzpatrick', 'confidence', 'confidence_within1', 'ita', 'L', 'a', 'b',
          *(col for k in range(5) for col in (f'fold_{k}', f'fold_{k}_conf'))]
MASK_HEADER = ['mask_backend', 'lesion_fraction', 'healthy_fraction', 'ita_unmasked', 'ita_masked',
               'ita_healthy_mean', 'l_star_masked', 'b_star_masked', 'mask_valid', 'overlay']


def collect(args):
    files = []
    for arg in args:
        if arg.startswith(('http://', 'https://')):
            name = Path(urllib.parse.urlparse(arg).path).name or 'image'
            files.append(Path(urllib.request.urlretrieve(arg, Path(tempfile.gettempdir()) / name)[0]))
            continue
        path = Path(arg)
        if path.is_dir():
            files.extend(p for p in sorted(path.iterdir()) if p.suffix.lower() in EXTS)
        elif path.suffix.lower() in EXTS:
            files.append(path)
        else:
            raise SystemExit(f'not an image, folder, or URL: {path}')
    return files


def load_batch(files):
    tensors = []
    for path in files:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f'could not read {path}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        tensors.append(PREP(img))
    return torch.stack(tensors)


def folds(builder, ckpt_dir, images, decode):
    device = images.device
    out = []
    for fold in range(5):
        model = builder()
        state = torch.load(f'{ckpt_dir}/fold_{fold}/model.pth', map_location='cpu', weights_only=True)
        model.load_state_dict(state)
        model.to(device).eval()
        preds = []
        with torch.no_grad():
            for start in range(0, len(images), 16):
                preds.append(decode(model(images[start:start + 16])).detach().cpu().numpy())
        out.append(np.concatenate(preds, axis=0))
        del model
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('inputs', nargs='+', help='images, folders, or URLs')
    ap.add_argument('--csv', type=Path, help='append one row per image not already in this CSV')
    ap.add_argument('--link-prefix', default='', help='prepended to the file name for the link column')
    ap.add_argument('--mask-ita', action='store_true', help='append lesion-masked ITA while retaining the released model outputs')
    ap.add_argument('--mask-backend', choices=('kmeans', 'unet', 'sam_adapter'), default='kmeans')
    ap.add_argument('--overlay-dir', type=Path, help='save lesion-mask overlays for review')
    ap.add_argument('--overlay-link-prefix', default='', help='URL prefix for saved mask overlays')
    args = ap.parse_args()
    header = HEADER + (MASK_HEADER if args.mask_ita else [])
    files = collect(args.inputs)
    if args.csv and args.csv.exists():
        with args.csv.open(newline='') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != header:
                raise SystemExit(f'{args.csv} has different columns than this version writes; rename or delete it')
            done = {row['image'] for row in reader}
        files = [p for p in files if p.name not in done]
        if not files:
            print('no new images')
            return
    if not files:
        if args.csv:
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            with args.csv.open('w', newline='') as f:
                csv.writer(f).writerow(header)
            print('no new images')
            return
        raise SystemExit('no images found')
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    images = load_batch(files).to(device)
    p_gt = folds(  # (fold, image, 5): CORAL's P(type > k) for k = 1..5
        lambda: get_efficientnet_b4_coral(num_classes=6, weights=None),
        FP_DIR, images, torch.sigmoid,
    )
    fp = (np.cumprod(p_gt, axis=-1) > 0.5).sum(axis=-1) + 1  # upstream's corn_label_from_logits rule
    edges = np.concatenate([np.ones(p_gt.shape[:2] + (1,)), p_gt, np.zeros(p_gt.shape[:2] + (1,))], axis=-1)
    probs = edges[..., :-1] - edges[..., 1:]  # (fold, image, 6): P(type = 1..6)
    lab = folds(
        lambda: get_efficientnet_b4_lab(num_outputs=3, weights=None),
        LAB_DIR, images, lambda logits: logits,
    )
    masks = []
    if args.mask_ita:
        from lesion_masked_ita.core import load_rgb, measure_ita, overlay_mask

        if args.overlay_dir:
            args.overlay_dir.mkdir(parents=True, exist_ok=True)
        for path in files:
            rgb = load_rgb(path)
            result = measure_ita(rgb, backend=args.mask_backend)
            overlay_name = ''
            if args.overlay_dir:
                overlay_name = f'{path.name}.overlay.png'
                Image.fromarray(overlay_mask(rgb, result['lesion_mask'])).save(args.overlay_dir / overlay_name)
            masks.append((result, overlay_name))
    added = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    rows = []
    print(f'{"file":<40} {"type":>4} {"conf":>5} {"folds":<14} {"ITA":>7}   L*    a*    b*')
    for i, path in enumerate(files):
        votes = fp[:, i]
        call = int(np.bincount(votes, minlength=7).argmax())
        fold_conf = probs[np.arange(len(votes)), i, votes - 1]
        conf = float(probs[:, i, call - 1].mean())
        within1 = float(probs[:, i, max(call - 2, 0):call + 1].sum(axis=-1).mean())
        mean = lab[:, i].mean(axis=0)
        ita = float(np.degrees(np.arctan2(mean[0] - 50.0, mean[2])))
        fold_txt = ','.join(str(int(v)) for v in votes)
        print(f'{path.name:<40} {call:>4} {conf:5.2f} {fold_txt:<14} {ita:7.1f}  {mean[0]:5.1f} {mean[1]:5.1f} {mean[2]:5.1f}')
        row = [path.name, args.link_prefix + urllib.parse.quote(path.name), added, call,
               round(conf, 3), round(within1, 3), round(ita, 1), *(round(float(v), 1) for v in mean),
               *(x for v, c in zip(votes, fold_conf) for x in (int(v), round(float(c), 3)))]
        if args.mask_ita:
            result, overlay_name = masks[i]
            row.extend([result['backend'], round(result['lesion_fraction'], 3),
                        round(result['healthy_fraction'], 3), round(result['ita_unmasked'], 2),
                        round(result['ita_masked'], 2), round(result['ita_healthy_mean'], 2),
                        round(result['l_star_masked'], 2), round(result['b_star_masked'], 2),
                        result['valid'], args.overlay_link_prefix + urllib.parse.quote(overlay_name) if overlay_name else ''])
        rows.append(row)
    if args.csv:
        new = not args.csv.exists()
        with args.csv.open('a', newline='') as f:
            writer = csv.writer(f)
            if new:
                writer.writerow(header)
            writer.writerows(rows)


if __name__ == '__main__':
    main()
