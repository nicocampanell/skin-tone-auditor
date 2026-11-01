"""Medical-SAM-Adapter melanoma masker (dermoscopy only).

Uses frozen SAM ViT-B + Melanoma_Photo_SAM_1024 from Medical-Adapter-Zoo.
Requires external/medical_sam_adapter (ImprintLab, commit 5888e722).

Weights (auto-downloaded on first use into lesion_masked_ita/weights/):
  - sam_vit_b_01ec64.pth  (Meta)
  - Melanoma_Photo_SAM_1024.pth  (KidsWithTokens/Medical-Adapter-Zoo)

ponytail: center-point prompt — derm lesions are usually centered; off-center
lesions need a click/box UI or GT-sampled prompt (see eval_isic --prompt gt).
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

IMAGE_SIZE = 1024
WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
MSA_ROOT = Path(__file__).resolve().parents[1] / "external" / "medical_sam_adapter"
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
SAM_FILE = "sam_vit_b_01ec64.pth"
ADAPTER_REPO = "KidsWithTokens/Medical-Adapter-Zoo"
ADAPTER_FILE = "Melanoma_Photo_SAM_1024.pth"

_net = None
_device = None


def _ensure_msa_on_path() -> None:
    root = str(MSA_ROOT)
    if not MSA_ROOT.is_dir():
        raise FileNotFoundError(
            f"Missing {MSA_ROOT}. Clone ImprintLab/Medical-SAM-Adapter there."
        )
    if root not in sys.path:
        sys.path.insert(0, root)


def _download_sam(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    urllib.request.urlretrieve(SAM_URL, path)


def _download_adapter(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    import shutil

    from huggingface_hub import hf_hub_download

    shutil.copy2(hf_hub_download(ADAPTER_REPO, ADAPTER_FILE), path)


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        mod="sam_adpt",
        mid_dim=None,
        thd=False,
        chunk=None,
        image_size=IMAGE_SIZE,
        out_size=IMAGE_SIZE,
        multimask_output=1,
        encoder="vit_b",
        sam_ckpt=str(WEIGHTS_DIR / SAM_FILE),
    )


def _load_net():
    global _net, _device
    if _net is not None:
        return _net, _device

    import torch

    _ensure_msa_on_path()
    from models.sam.build_sam import sam_model_registry

    sam_path = WEIGHTS_DIR / SAM_FILE
    adapter_path = WEIGHTS_DIR / ADAPTER_FILE
    _download_sam(sam_path)
    _download_adapter(adapter_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = _args()
    net = sam_model_registry["vit_b"](args, checkpoint=str(sam_path))
    try:
        ckpt = torch.load(adapter_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(adapter_path, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    net.load_state_dict(state, strict=False)
    net.to(device)
    net.eval()
    _net, _device = net, device
    return _net, _device


def mask_lesion_sam_adapter(rgb: np.ndarray) -> np.ndarray:
    """Binary lesion mask; input HxWx3 uint8 RGB. Dermoscopy only."""
    import torch
    import torch.nn.functional as F

    net, device = _load_net()
    h, w = rgb.shape[:2]
    resized = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    # MSA train path: ToTensor * 255 → float [0, 255]
    tensor = torch.from_numpy(resized.astype(np.float32).transpose(2, 0, 1)).unsqueeze(0)
    tensor = tensor.to(device)

    # Center positive click in resized coordinates (x, y)
    coords = torch.tensor([[[IMAGE_SIZE / 2.0, IMAGE_SIZE / 2.0]]], device=device)
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
