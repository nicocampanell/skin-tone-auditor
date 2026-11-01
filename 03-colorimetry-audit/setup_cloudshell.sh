#!/usr/bin/env bash
# One-time setup for running the released Fitzpatrick and ITA models in Google Cloud Shell.
# Self-contained: paste this one file into Cloud Shell, nothing else is needed. Safe to re-run.
# Weights go in $HOME (persists, ~0.7 GB). The Python env goes in /tmp because the
# 5 GB home disk can't also hold PyTorch; it is rebuilt automatically after a restart.
set -euo pipefail

repo=$HOME/nn_colorimetry_dermatoscopy
venv=/tmp/colorimetry-venv
release=https://github.com/marinbenc/nn_colorimetry_dermatoscopy/releases/download/preprint

if [ ! -d "$repo/.git" ]; then
  git clone -q --depth 1 https://github.com/marinbenc/nn_colorimetry_dermatoscopy.git "$repo"
fi
[ "$(realpath "$0")" = "$repo/setup_cloudshell.sh" ] || cp "$0" "$repo/setup_cloudshell.sh"

cat > "$repo/run_images.py" <<'PY'
"""Run the released Fitzpatrick and ITA models on your own images.

Usage:
  colorimetry photo.jpg
  colorimetry photo.jpg another.png
  colorimetry /path/to/folder
  colorimetry https://example.com/photo.jpg
"""
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
from coral_pytorch.dataset import corn_label_from_logits
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
    if not files:
        raise SystemExit('no images found')
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
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    files = collect(sys.argv[1:])
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    images = load_batch(files).to(device)
    fp = folds(
        lambda: get_efficientnet_b4_coral(num_classes=6, weights=None),
        FP_DIR, images, lambda logits: corn_label_from_logits(logits),
    ).astype(int) + 1
    lab = folds(
        lambda: get_efficientnet_b4_lab(num_outputs=3, weights=None),
        LAB_DIR, images, lambda logits: logits,
    )
    print(f'{"file":<40} {"type":>4} {"folds":<14} {"ITA":>7}   L*    a*    b*')
    for i, path in enumerate(files):
        votes = fp[:, i]
        call = int(np.bincount(votes, minlength=7).argmax())
        mean = lab[:, i].mean(axis=0)
        ita = float(np.degrees(np.arctan2(mean[0] - 50.0, mean[2])))
        fold_txt = ','.join(str(int(v)) for v in votes)
        print(f'{path.name:<40} {call:>4} {fold_txt:<14} {ita:7.1f}  {mean[0]:5.1f} {mean[1]:5.1f} {mean[2]:5.1f}')


if __name__ == '__main__':
    main()
PY

for name in finetune_mskcc_blur0 finetune_mskcc_blur0_lab; do
  [ -f "$repo/experiments/$name/fold_4/model.pth" ] && continue
  echo "Downloading $name weights..."
  curl -fsSL -o "/tmp/$name.zip" "$release/$name.zip"
  unzip -q -o "/tmp/$name.zip" '*/model.pth' -d "$repo/experiments"
  rm "/tmp/$name.zip"
done

if [ ! -x "$venv/bin/python" ]; then
  echo "Installing PyTorch (CPU) into $venv, about a minute..."
  python3 -m venv "$venv"
  "$venv/bin/pip" install -q --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision
  "$venv/bin/pip" install -q --no-cache-dir opencv-python-headless coral-pytorch pandas tqdm matplotlib scikit-image
fi

line="alias colorimetry='bash $repo/setup_cloudshell.sh && $venv/bin/python $repo/run_images.py'"
grep -qxF "$line" ~/.bashrc || echo "$line" >> ~/.bashrc
