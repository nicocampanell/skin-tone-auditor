"""
Lesion-Robust Binary Skin Tone Classifier
==========================================
Dermatology Fairness Auditing Infrastructure

Architecture : EfficientNet-B3 (primary) | ConvNeXt-Tiny (ensemble)
Task         : Binary classification — Fitzpatrick 1-3 (Class 0) vs 4-6 (Class 1)
Dataset      : Fitzpatrick17k (~10,000 dermatological images)

Usage
-----
1. Install dependencies:
   pip install torch torchvision timm albumentations opencv-python scikit-learn pandas numpy tqdm

2. Configure paths in CONFIG (bottom of file) then run:
   python skin_tone_classifier.py --mode train
   python skin_tone_classifier.py --mode eval
   python skin_tone_classifier.py --mode predict --image path/to/image.jpg
"""

import os
import cv2
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    cohen_kappa_score, confusion_matrix,
    classification_report
)

# ── Optional: Grad-CAM ─────────────────────────────────────────────────────
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    print("[WARN] pytorch-grad-cam not installed. Attention regularization disabled.")
    print("       Install with: pip install grad-cam")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG  —  Edit these paths before running
# ══════════════════════════════════════════════════════════════════════════════
CONFIG = {
    # --- Paths ---
    "metadata_csv":  "fitzpatrick17k.csv",       # path to metadata CSV
    "image_dir":     "images/",                  # directory containing <md5hash>.jpg
    "mask_dir":      None,                       # optional: directory with <md5hash>.png lesion masks
    "output_dir":    "outputs/",                 # checkpoints, logs, results

    # --- Model ---
    "backbone":      "efficientnet_b3",          # timm model name
    "img_size":      300,                        # 300 for EffNet-B3; use 224 for ConvNeXt/ViT
    "dropout":       0.4,

    # --- Training ---
    "epochs_warmup":  5,
    "epochs_finetune": 40,
    "batch_size":     32,
    "seed":           42,
    "num_workers":    4,

    # --- Loss ---
    "label_smoothing": 0.05,
    "lambda_reg":      0.1,                      # attention regularization weight
    "attention_reg":   True,                     # enable only if mask_dir is set and grad-cam available

    # --- Early Stopping ---
    "patience":        7,

    # --- Ensemble (optional) ---
    "ensemble_backbones": ["efficientnet_b3", "convnext_tiny"],
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA INGESTION & QC FILTERING
# ══════════════════════════════════════════════════════════════════════════════

def load_and_filter(metadata_path: str, image_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load metadata, apply QC filtering, convert to binary labels,
    and flag noisy labels using fitzpatrick_centaur cross-validation.

    Returns
    -------
    df_clean : rows passing all QC checks
    df_noisy : rows with centaur disagreement > 1 grade (for ablation)
    """
    df = pd.read_csv(metadata_path)
    n_raw = len(df)

    # Step 1: QC gate
    df = df[df['qc'] == 'pass'].copy()

    # Step 2: Valid Fitzpatrick range
    df = df[df['fitzpatrick_scale'].between(1, 6)]

    # Step 3: Binary label conversion
    df['label'] = (df['fitzpatrick_scale'] >= 4).astype(int)

    # Step 4: Centaur cross-validation (label noise detection)
    df['centaur_label'] = (df['fitzpatrick_centaur'] >= 4).astype(int)
    df['noisy'] = abs(df['fitzpatrick_scale'] - df['fitzpatrick_centaur']) > 1

    df_clean = df[~df['noisy']].copy()
    df_noisy = df[df['noisy']].copy()

    # Step 5: Verify image files exist on disk
    df_clean['img_path'] = df_clean['md5hash'].apply(
        lambda h: os.path.join(image_dir, str(h) + '.jpg'))
    exists = df_clean['img_path'].apply(os.path.exists)
    df_clean = df_clean[exists].copy()

    print(f"\n[Data] Raw rows       : {n_raw}")
    print(f"[Data] After QC       : {len(df)}")
    print(f"[Data] Clean labels   : {len(df_clean)}")
    print(f"[Data] Noisy labels   : {len(df_noisy)}")
    print(f"[Data] Class 0 (F1-3) : {(df_clean['label']==0).sum()}")
    print(f"[Data] Class 1 (F4-6) : {(df_clean['label']==1).sum()}")

    return df_clean, df_noisy


# ══════════════════════════════════════════════════════════════════════════════
# 2. LEAKAGE-AWARE SPLITTING
# ══════════════════════════════════════════════════════════════════════════════

def make_splits(df: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratify on compound key: skin_tone_label x disease_partition.
    Prevents model exploiting disease-tone correlations present in training set.
    """
    df = df.copy()
    df['strat_key'] = (
        df['label'].astype(str) + '_' + df['nine_partition_label'].astype(str)
    )

    train_df, temp_df = train_test_split(
        df, test_size=0.20, stratify=df['strat_key'], random_state=seed)

    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df['strat_key'], random_state=seed)

    # Sanity: no hash overlap across splits
    assert len(set(train_df.md5hash) & set(val_df.md5hash)) == 0, "Train/Val overlap!"
    assert len(set(train_df.md5hash) & set(test_df.md5hash)) == 0, "Train/Test overlap!"

    print(f"\n[Split] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df


# ══════════════════════════════════════════════════════════════════════════════
# 3. TRANSFORMS & AUGMENTATION
# ══════════════════════════════════════════════════════════════════════════════

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def get_transforms(split: str, img_size: int = 300) -> A.Compose:
    """
    Dermatology-specific augmentation.

    Key constraint: hue augmentation capped at ±0.04 (not the typical 0.2).
    Aggressive hue shifts in the red-yellow range corrupt the melanin signal
    this classifier depends on. Saturation is safe to augment broadly.
    """
    if split == 'train':
        return A.Compose([
            A.RandomResizedCrop(img_size, img_size, scale=(0.7, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(
                brightness=0.3, contrast=0.3,
                saturation=0.2, hue=0.04,   # NOTE: deliberately constrained
                p=0.8
            ),
            A.RandomShadow(p=0.3),           # simulates clinical lighting artifacts
            A.GaussianBlur(blur_limit=3, p=0.2),
            A.ImageCompression(quality_lower=70, p=0.2),  # simulates JPEG archival
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ])
    else:  # val / test
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ])


def get_tta_transforms(img_size: int = 300) -> list[A.Compose]:
    """
    Test-Time Augmentation: 8 views averaged at inference.
    """
    base = [A.Resize(img_size, img_size), A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return [
        A.Compose(base),
        A.Compose([A.HorizontalFlip(p=1.0)] + base),
        A.Compose([A.VerticalFlip(p=1.0)] + base),
        A.Compose([A.RandomRotate90(p=1.0)] + base),
        A.Compose([A.Transpose(p=1.0)] + base),
        A.Compose([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)] + base),
        A.Compose([A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0)] + base),
        A.Compose([A.GaussianBlur(blur_limit=3, p=1.0)] + base),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# 4. DATASET
# ══════════════════════════════════════════════════════════════════════════════

class FitzpatrickDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None, mask_dir: str = None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.mask_dir = mask_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.cvtColor(cv2.imread(row['img_path']), cv2.COLOR_BGR2RGB)

        mask = None
        if self.mask_dir:
            mask_path = os.path.join(self.mask_dir, str(row['md5hash']) + '.png')
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = (mask > 127).astype(np.uint8)  # binarize

        if self.transform:
            if mask is not None:
                aug = self.transform(image=img, mask=mask)
                img, mask = aug['image'], aug['mask']
            else:
                img = self.transform(image=img)['image']

        img_size = img.shape[-1] if isinstance(img, torch.Tensor) else img.shape[0]

        return {
            'image':   img if isinstance(img, torch.Tensor) else torch.tensor(img),
            'label':   torch.tensor(row['label'], dtype=torch.float32),
            'mask':    torch.tensor(mask, dtype=torch.float32) if mask is not None
                       else torch.zeros(img_size, img_size),
            'disease': str(row.get('nine_partition_label', 'unknown')),
            'md5':     str(row['md5hash']),
        }


def make_dataloaders(train_df, val_df, test_df, cfg: dict) -> tuple:
    """Build DataLoaders with stratified batch sampling for class balance."""
    img_size = cfg['img_size']

    train_ds = FitzpatrickDataset(train_df, get_transforms('train', img_size), cfg.get('mask_dir'))
    val_ds   = FitzpatrickDataset(val_df,   get_transforms('val',   img_size), cfg.get('mask_dir'))
    test_ds  = FitzpatrickDataset(test_df,  get_transforms('test',  img_size))

    # Weighted sampler for class balance
    labels = train_df['label'].values
    class_counts = np.bincount(labels)
    weights = 1.0 / class_counts[labels]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'],
                              sampler=sampler, num_workers=cfg['num_workers'],
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg['batch_size'],
                              shuffle=False, num_workers=cfg['num_workers'])
    test_loader  = DataLoader(test_ds,  batch_size=cfg['batch_size'],
                              shuffle=False, num_workers=cfg['num_workers'])

    return train_loader, val_loader, test_loader


# ══════════════════════════════════════════════════════════════════════════════
# 5. MODEL
# ══════════════════════════════════════════════════════════════════════════════

class SkinToneClassifier(nn.Module):
    """
    Binary skin tone classifier built on a pretrained backbone.
    Supported backbones: efficientnet_b3, convnext_tiny, vit_base_patch16_224
    """

    def __init__(self, backbone: str = 'efficientnet_b3',
                 pretrained: bool = True, dropout: float = 0.4):
        super().__init__()
        self.backbone_name = backbone
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.75),
            nn.Linear(256, 1)   # single logit → sigmoid → probability
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)   # (B, feat_dim)
        return self.classifier(features)   # (B, 1)

    def get_gradcam_target_layer(self):
        """Return the last convolutional layer for Grad-CAM."""
        if 'efficientnet' in self.backbone_name:
            return self.backbone.conv_head
        elif 'convnext' in self.backbone_name:
            return self.backbone.stages[-1].blocks[-1].conv_dw
        elif 'vit' in self.backbone_name:
            return self.backbone.blocks[-1].norm1
        return None


def set_finetune_phase(model: SkinToneClassifier, phase: str = 'warmup'):
    """
    Warmup  : freeze all backbone params, train head only.
    Finetune: unfreeze top blocks with layerwise LR (set in optimizer).
    """
    if phase == 'warmup':
        for p in model.backbone.parameters():
            p.requires_grad = False
        for p in model.classifier.parameters():
            p.requires_grad = True
        print("[Phase] Warmup — backbone frozen, training head only")

    elif phase == 'finetune':
        # Unfreeze top blocks only
        finetune_keywords = ['blocks.6', 'blocks.7', 'bn2', 'conv_head',   # EfficientNet
                             'stages.3', 'stages.2',                         # ConvNeXt
                             'blocks.10', 'blocks.11', 'norm']               # ViT
        for name, p in model.backbone.named_parameters():
            p.requires_grad = any(k in name for k in finetune_keywords)
        for p in model.classifier.parameters():
            p.requires_grad = True
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[Phase] Finetune — {n_trainable:,} trainable parameters")


def build_optimizer(model: SkinToneClassifier, phase: str = 'warmup',
                    finetune_keywords: list = None) -> torch.optim.Optimizer:
    if phase == 'warmup':
        return torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)

    if finetune_keywords is None:
        finetune_keywords = ['blocks.6', 'blocks.7', 'bn2', 'conv_head',
                             'stages.3', 'stages.2', 'blocks.10', 'blocks.11', 'norm']
    top_params = [p for n, p in model.backbone.named_parameters()
                  if any(k in n for k in finetune_keywords) and p.requires_grad]

    return torch.optim.AdamW([
        {'params': model.classifier.parameters(), 'lr': 1e-4},
        {'params': top_params,                    'lr': 1e-5},
    ], weight_decay=1e-4)


# ══════════════════════════════════════════════════════════════════════════════
# 6. LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

class LabelSmoothBCE(nn.Module):
    """
    BCE with label smoothing to handle expected annotation noise
    in Fitzpatrick scale labels (smoothing=0.05 recommended).
    Combines with pos_weight for class imbalance correction.
    """
    def __init__(self, smoothing: float = 0.05, pos_weight: torch.Tensor = None):
        super().__init__()
        self.s = smoothing
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets_smooth = targets * (1 - self.s) + 0.5 * self.s
        return self.bce(logits, targets_smooth)


def build_criterion(train_df: pd.DataFrame, smoothing: float = 0.05,
                    device: str = 'cpu') -> LabelSmoothBCE:
    n_neg = (train_df['label'] == 0).sum()
    n_pos = (train_df['label'] == 1).sum()
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
    print(f"[Loss] pos_weight = {pos_weight.item():.3f} (n_neg={n_neg}, n_pos={n_pos})")
    return LabelSmoothBCE(smoothing=smoothing, pos_weight=pos_weight)


# ══════════════════════════════════════════════════════════════════════════════
# 7. ATTENTION REGULARIZATION
# ══════════════════════════════════════════════════════════════════════════════

def attention_reg_loss(gradcam_map: torch.Tensor,
                       lesion_mask: torch.Tensor,
                       lambda_reg: float = 0.1) -> torch.Tensor:
    """
    Novel training objective: penalize model attention in lesion regions.
    Forces classifier to rely on perilesional skin rather than pathological
    discoloration (erythema, inflammation) as a skin tone proxy.

    Novel contribution: cite this loss in paper methods section.

    Parameters
    ----------
    gradcam_map : (B, H, W) normalized Grad-CAM saliency maps
    lesion_mask : (B, H, W) binary masks (1=lesion, 0=perilesional skin)
    lambda_reg  : regularization weight (default 0.1)
    """
    if lesion_mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True)

    # Resize mask to match Grad-CAM spatial dimensions
    mask_resized = F.interpolate(
        lesion_mask.unsqueeze(1).float(),
        size=gradcam_map.shape[-2:],
        mode='nearest'
    ).squeeze(1)

    lesion_attn    = (gradcam_map * mask_resized).sum(dim=[-1, -2])
    perilesion_attn = (gradcam_map * (1 - mask_resized)).sum(dim=[-1, -2])

    # Ratio: penalize when model fixates on lesion vs. perilesional skin
    ratio = lesion_attn / (perilesion_attn + 1e-8)
    return lambda_reg * ratio.mean()


def compute_gradcam(model: SkinToneClassifier, images: torch.Tensor,
                    device: str) -> torch.Tensor | None:
    """Compute Grad-CAM maps for attention regularization."""
    if not GRADCAM_AVAILABLE:
        return None
    target_layer = model.get_gradcam_target_layer()
    if target_layer is None:
        return None
    try:
        cam = GradCAM(model=model, target_layers=[target_layer])
        targets = [BinaryClassifierOutputTarget(1)] * images.shape[0]
        grayscale_cam = cam(input_tensor=images, targets=targets)
        return torch.tensor(grayscale_cam, device=device)
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 8. ITA BASELINE  (perilesional skin tone proxy)
# ══════════════════════════════════════════════════════════════════════════════

def compute_ita(image_bgr: np.ndarray, mask: np.ndarray = None) -> float:
    """
    Individual Typology Angle from CIELAB color space.
    Approximate mapping: ITA > 55 = Fitzpatrick I, ITA < -30 = Fitzpatrick VI.

    If lesion mask provided, compute only on non-lesion (perilesional) pixels.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(float)
    L, b = lab[:, :, 0], lab[:, :, 2]

    if mask is not None:
        roi = (mask == 0)   # perilesional only
        if roi.sum() < 100:  # fallback if mask too large
            roi = np.ones_like(mask, dtype=bool)
        L, b = L[roi], b[roi]

    ita = np.degrees(np.arctan((np.mean(L) - 50) / (np.mean(b) + 1e-8)))
    return float(ita)


def ita_baseline_predict(image_path: str, threshold: float = 28.0) -> int:
    """ITA threshold classifier (null model baseline). Class 1 if ITA < threshold."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    ita = compute_ita(img)
    return int(ita < threshold)


# ══════════════════════════════════════════════════════════════════════════════
# 9. METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(labels: list, probs: list,
                    threshold: float = 0.5) -> dict:
    """Compute full fairness-aware metrics suite."""
    labels = np.array(labels)
    probs  = np.array(probs)
    preds  = (probs >= threshold).astype(int)

    auc_roc = roc_auc_score(labels, probs)
    auc_pr  = average_precision_score(labels, probs)
    kappa   = cohen_kappa_score(labels, preds)

    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    fpr = fp / (fp + tn + 1e-8)
    fnr = fn / (fn + tp + 1e-8)

    return {
        'auc_roc':  round(auc_roc, 4),
        'auc_pr':   round(auc_pr, 4),
        'kappa':    round(kappa, 4),
        'fpr':      round(fpr, 4),
        'fnr':      round(fnr, 4),
        'n':        int(len(labels)),
    }


def compute_metrics_by_disease(labels, probs, diseases, threshold=0.5) -> dict:
    """Equalized odds analysis stratified by disease partition."""
    results = {}
    df = pd.DataFrame({'label': labels, 'prob': probs, 'disease': diseases})
    for disease, grp in df.groupby('disease'):
        if len(grp) < 20:
            continue
        try:
            results[disease] = compute_metrics(
                grp['label'].tolist(), grp['prob'].tolist(), threshold)
        except Exception:
            pass
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 10. TRAINING & EVALUATION LOOPS
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, criterion,
                device, use_attention_reg=False, lambda_reg=0.1) -> dict:
    model.train()
    losses, probs, labels, diseases = [], [], [], []

    for batch in tqdm(loader, desc="Train", leave=False):
        images  = batch['image'].to(device)
        targets = batch['label'].to(device)
        masks   = batch['mask'].to(device)

        optimizer.zero_grad()
        logits = model(images).squeeze(1)
        loss   = criterion(logits, targets)

        # Attention regularization (novel contribution)
        if use_attention_reg and GRADCAM_AVAILABLE:
            gcam = compute_gradcam(model, images, device)
            if gcam is not None:
                loss = loss + attention_reg_loss(gcam, masks, lambda_reg)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(loss.item())
        probs.extend(torch.sigmoid(logits).detach().cpu().numpy())
        labels.extend(targets.cpu().numpy())
        diseases.extend(batch['disease'])

    metrics = compute_metrics(labels, probs)
    metrics['loss'] = round(np.mean(losses), 4)
    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device,
             use_tta=False, img_size=300) -> dict:
    model.eval()
    losses, probs, labels, diseases = [], [], [], []

    tta_transforms = get_tta_transforms(img_size) if use_tta else None

    for batch in tqdm(loader, desc="Eval", leave=False):
        images  = batch['image'].to(device)
        targets = batch['label'].to(device)

        if use_tta and tta_transforms:
            # Average predictions across 8 TTA views
            batch_probs = []
            for t in tta_transforms:
                imgs_np = images.cpu().numpy().transpose(0, 2, 3, 1)
                imgs_t  = torch.stack([
                    t(image=(img * 255).astype(np.uint8))['image']
                    for img in imgs_np
                ]).to(device)
                logits = model(imgs_t).squeeze(1)
                batch_probs.append(torch.sigmoid(logits).cpu().numpy())
            avg_probs = np.mean(batch_probs, axis=0)
            probs.extend(avg_probs)
        else:
            logits = model(images).squeeze(1)
            loss   = criterion(logits, targets)
            losses.append(loss.item())
            probs.extend(torch.sigmoid(logits).cpu().numpy())

        labels.extend(targets.cpu().numpy())
        diseases.extend(batch['disease'])

    metrics = compute_metrics(labels, probs)
    if losses:
        metrics['loss'] = round(np.mean(losses), 4)
    metrics['disease_breakdown'] = compute_metrics_by_disease(labels, probs, diseases)
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# 11. FULL TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def train(cfg: dict):
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    os.makedirs(cfg['output_dir'], exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[Device] {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    df_clean, df_noisy = load_and_filter(cfg['metadata_csv'], cfg['image_dir'])
    train_df, val_df, test_df = make_splits(df_clean, seed=cfg['seed'])
    train_loader, val_loader, test_loader = make_dataloaders(
        train_df, val_df, test_df, cfg)

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = SkinToneClassifier(cfg['backbone'], dropout=cfg['dropout']).to(device)
    criterion = build_criterion(train_df, cfg['label_smoothing'], device)

    use_attention_reg = (
        cfg['attention_reg'] and
        cfg.get('mask_dir') is not None and
        GRADCAM_AVAILABLE
    )
    if use_attention_reg:
        print("[Training] Attention regularization: ENABLED")
    else:
        print("[Training] Attention regularization: DISABLED (no mask_dir or grad-cam)")

    best_auc  = 0.0
    history   = []
    patience  = 0

    # ── Phase 1: Warmup ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"PHASE 1: WARMUP ({cfg['epochs_warmup']} epochs)")
    print(f"{'='*60}")
    set_finetune_phase(model, 'warmup')
    optimizer = build_optimizer(model, 'warmup')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['epochs_warmup'])

    for epoch in range(cfg['epochs_warmup']):
        train_m = train_epoch(model, train_loader, optimizer, criterion, device,
                              use_attention_reg=False)
        val_m   = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        print(f"[Warmup {epoch+1:02d}] "
              f"Train AUC={train_m['auc_roc']:.4f} loss={train_m['loss']:.4f} | "
              f"Val AUC={val_m['auc_roc']:.4f}")
        history.append({'phase': 'warmup', 'epoch': epoch+1, **val_m})

    # ── Phase 2: Fine-tuning ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"PHASE 2: FINE-TUNING (up to {cfg['epochs_finetune']} epochs)")
    print(f"{'='*60}")
    set_finetune_phase(model, 'finetune')
    optimizer = build_optimizer(model, 'finetune')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2)

    for epoch in range(cfg['epochs_finetune']):
        train_m = train_epoch(model, train_loader, optimizer, criterion, device,
                              use_attention_reg=use_attention_reg,
                              lambda_reg=cfg['lambda_reg'])
        val_m   = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"[Epoch {epoch+1:02d}] "
              f"Train AUC={train_m['auc_roc']:.4f} loss={train_m['loss']:.4f} | "
              f"Val AUC={val_m['auc_roc']:.4f} PR={val_m['auc_pr']:.4f} "
              f"Kappa={val_m['kappa']:.4f}")

        history.append({'phase': 'finetune', 'epoch': epoch+1, **val_m})

        if val_m['auc_roc'] > best_auc:
            best_auc = val_m['auc_roc']
            ckpt_path = os.path.join(cfg['output_dir'], 'best_model.pt')
            torch.save({'model_state': model.state_dict(),
                        'cfg': cfg, 'epoch': epoch+1,
                        'val_metrics': val_m}, ckpt_path)
            print(f"  ✓ Saved best model (AUC={best_auc:.4f})")
            patience = 0
        else:
            patience += 1
            if patience >= cfg['patience']:
                print(f"[Early Stop] No improvement for {cfg['patience']} epochs.")
                break

    # ── Final Test Evaluation ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FINAL TEST EVALUATION")
    print(f"{'='*60}")
    ckpt = torch.load(os.path.join(cfg['output_dir'], 'best_model.pt'), map_location=device)
    model.load_state_dict(ckpt['model_state'])
    test_m = evaluate(model, test_loader, criterion, device,
                      use_tta=True, img_size=cfg['img_size'])

    print(f"\nTest Results:")
    print(f"  AUC-ROC : {test_m['auc_roc']:.4f}")
    print(f"  AUC-PR  : {test_m['auc_pr']:.4f}")
    print(f"  Kappa   : {test_m['kappa']:.4f}")
    print(f"  FPR     : {test_m['fpr']:.4f}")
    print(f"  FNR     : {test_m['fnr']:.4f}")

    print(f"\nEqualized Odds by Disease Partition:")
    for disease, dm in test_m.get('disease_breakdown', {}).items():
        print(f"  {disease:<30} AUC={dm['auc_roc']:.4f}  FPR={dm['fpr']:.4f}  FNR={dm['fnr']:.4f}")

    # Save history
    pd.DataFrame(history).to_csv(
        os.path.join(cfg['output_dir'], 'training_history.csv'), index=False)
    print(f"\n[Done] Outputs saved to: {cfg['output_dir']}")


# ══════════════════════════════════════════════════════════════════════════════
# 12. INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_single(image_path: str, checkpoint_path: str,
                   use_tta: bool = True) -> dict:
    """
    Predict skin tone class for a single image.
    Returns probability, predicted class, and ITA baseline for comparison.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg  = ckpt['cfg']

    model = SkinToneClassifier(cfg['backbone'], pretrained=False).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = get_transforms('val', cfg['img_size'])
    tensor = transform(image=img_rgb)['image'].unsqueeze(0).to(device)

    logit = model(tensor).squeeze()
    prob  = torch.sigmoid(logit).item()

    ita = compute_ita(img)

    result = {
        'image':          image_path,
        'prob_class1':    round(prob, 4),
        'predicted_class': int(prob >= 0.5),
        'label_meaning':  'Fitzpatrick 4-6' if prob >= 0.5 else 'Fitzpatrick 1-3',
        'ita_angle':      round(ita, 2),
        'ita_prediction': int(ita < 28.0),
    }
    print(f"\nPrediction for: {image_path}")
    for k, v in result.items():
        print(f"  {k:<20}: {v}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 13. ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Lesion-Robust Skin Tone Classifier")
    parser.add_argument('--mode',       type=str, default='train',
                        choices=['train', 'eval', 'predict'],
                        help='Run mode')
    parser.add_argument('--image',      type=str, default=None,
                        help='Path to image for --mode predict')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint for eval/predict')
    parser.add_argument('--backbone',   type=str, default=None,
                        help='Override backbone (efficientnet_b3, convnext_tiny, vit_base_patch16_224)')
    parser.add_argument('--no-attn-reg', action='store_true',
                        help='Disable attention regularization')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg  = CONFIG.copy()

    if args.backbone:
        cfg['backbone'] = args.backbone
        cfg['img_size'] = 224 if 'vit' in args.backbone or 'convnext' in args.backbone else 300
    if args.no_attn_reg:
        cfg['attention_reg'] = False

    if args.mode == 'train':
        train(cfg)

    elif args.mode == 'eval':
        ckpt_path = args.checkpoint or os.path.join(cfg['output_dir'], 'best_model.pt')
        device    = 'cuda' if torch.cuda.is_available() else 'cpu'
        ckpt      = torch.load(ckpt_path, map_location=device)
        model     = SkinToneClassifier(ckpt['cfg']['backbone'], pretrained=False).to(device)
        model.load_state_dict(ckpt['model_state'])

        df_clean, _ = load_and_filter(cfg['metadata_csv'], cfg['image_dir'])
        _, _, test_df = make_splits(df_clean, seed=cfg['seed'])
        _, _, test_loader = make_dataloaders(
            test_df, test_df, test_df, ckpt['cfg'])
        criterion = build_criterion(test_df, device=device)
        test_m = evaluate(model, test_loader, criterion, device,
                          use_tta=True, img_size=ckpt['cfg']['img_size'])
        print("\nTest Metrics:", {k: v for k, v in test_m.items() if k != 'disease_breakdown'})

    elif args.mode == 'predict':
        if not args.image:
            print("[ERROR] --image required for predict mode")
        else:
            ckpt_path = args.checkpoint or os.path.join(cfg['output_dir'], 'best_model.pt')
            predict_single(args.image, ckpt_path)
