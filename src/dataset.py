"""
============================================================================
  dataset.py  —  BCC Dermoscopy Dataset & Preprocessing Pipeline
  Paper : "Basal Cell Carcinoma Skin Detection Using Deep Learning"
============================================================================
"""

import os
import cv2
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ─────────────────────────────────────────────────────────────────────────────
# 1.  HAIR REMOVAL
# ─────────────────────────────────────────────────────────────────────────────
def remove_hair(image: np.ndarray, radius: int = 15) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    _, mask = cv2.threshold(blackhat, int(0.15 * 255), 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

    return cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_NS)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  COLOUR NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────
def reinhard_normalise(image: np.ndarray,
                       target_mean: Optional[np.ndarray] = None,
                       target_std: Optional[np.ndarray] = None) -> np.ndarray:
    if target_mean is None:
        target_mean = np.array([65.3, 12.1, 14.8], dtype=np.float32)
    if target_std is None:
        target_std = np.array([22.1, 9.6, 8.4], dtype=np.float32)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)

    def norm_channel(ch, t_mean, t_std):
        src_mean, src_std = ch.mean(), ch.std() + 1e-6
        return (ch - src_mean) / src_std * t_std + t_mean

    lab_norm = cv2.merge([
        np.clip(norm_channel(l, target_mean[0], target_std[0]), 0, 255),
        np.clip(norm_channel(a, target_mean[1], target_std[1]), 0, 255),
        np.clip(norm_channel(b, target_mean[2], target_std[2]), 0, 255),
    ]).astype(np.uint8)

    return cv2.cvtColor(lab_norm, cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  AUGMENTATION PIPELINES
# ─────────────────────────────────────────────────────────────────────────────
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_SIZE = 380


def get_train_transforms(is_minority: bool = False) -> A.Compose:
    base_augs = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.7),
        A.ColorJitter(brightness=0.2, contrast=0.15,
                      saturation=0.10, hue=0.05, p=0.6),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.RandomBrightnessContrast(p=0.3),
        A.HueSaturationValue(hue_shift_limit=5,
                             sat_shift_limit=10,
                             val_shift_limit=10, p=0.3),
    ]

    # FIX 6: removed alpha_affine kwarg — it was dropped from ElasticTransform
    #         in recent Albumentations versions (now lives in a separate
    #         Affine transform). Leaving it in raises TypeError on current
    #         installs.
    minority_augs = [
        A.ElasticTransform(alpha=1, sigma=50, p=0.4),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.4),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15,
                           rotate_limit=15, p=0.4),
    ]

    augs = base_augs + (minority_augs if is_minority else [])

    return A.Compose(augs + [
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms() -> A.Compose:
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# 4.  CUTMIX AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────
def cutmix_batch(images: torch.Tensor,
                 labels: torch.Tensor,
                 alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor,
                                               torch.Tensor, float]:
    lam = np.random.beta(alpha, alpha)
    B, C, H, W = images.shape
    rand_idx = torch.randperm(B, device=images.device)

    cut_ratio = np.sqrt(1.0 - lam)
    cut_w, cut_h = int(W * cut_ratio), int(H * cut_ratio)
    cx, cy = np.random.randint(W), np.random.randint(H)

    x1, y1 = np.clip(cx - cut_w // 2, 0, W), np.clip(cy - cut_h // 2, 0, H)
    x2, y2 = np.clip(cx + cut_w // 2, 0, W), np.clip(cy + cut_h // 2, 0, H)

    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[rand_idx, :, y1:y2, x1:x2]

    lam = 1.0 - (x2 - x1) * (y2 - y1) / (H * W)
    return mixed, labels, labels[rand_idx], lam


# ─────────────────────────────────────────────────────────────────────────────
# 5.  DATASET CLASS
# ─────────────────────────────────────────────────────────────────────────────
BINARY_LABELS = {'BCC': 1, 'non-BCC': 0}
SUBTYPE_LABELS = {'nodular': 0, 'superficial': 1,
                  'infiltrative': 2, 'morpheaform': 3, 'basosquamous': 4}
MINORITY_SUBTYPES = {'morpheaform', 'infiltrative', 'basosquamous'}


class BCCDataset(Dataset):
    def __init__(self, csv_path: str, root_dir: str = "",
                 mode: str = 'binary', split: str = 'train',
                 preprocess: bool = True):
        self.df = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.mode = mode
        self.split = split
        self.preprocess = preprocess
        self.is_train = (split == 'train')

        assert mode in ('binary', 'subtype')

        # FIX (2026-07): the previous version only ever flagged minority
        # samples in 'subtype' mode, leaving is_minority permanently False
        # for every sample in 'binary' mode. Since binary mode's actual
        # minority class is BCC itself (657 vs 4782 in the test split),
        # this meant BCC images never received the stronger minority
        # augmentation pipeline (ElasticTransform, GridDistortion,
        # ShiftScaleRotate) — only non-BCC images did, which is backwards.
        if mode == 'subtype' and 'subtype' in self.df.columns:
            self.is_minority = self.df['subtype'].isin(MINORITY_SUBTYPES).tolist()
        elif mode == 'binary' and 'label' in self.df.columns:
            self.is_minority = (self.df['label'] == 'BCC').tolist()
        else:
            self.is_minority = [False] * len(self.df)

    # FIX 2: make get_class_weights mode-aware instead of hardcoding 'label'
    def get_class_weights(self) -> torch.Tensor:
        if self.mode == 'binary':
            label_col = 'label'
            label_map = BINARY_LABELS
        else:
            label_col = 'subtype'
            label_map = SUBTYPE_LABELS

        labels = self.df[label_col].map(label_map)
        class_counts = labels.value_counts().sort_index()
        total = class_counts.sum()
        weights = total / (len(class_counts) * class_counts)
        return torch.tensor(weights.values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.df)

    def _load_and_preprocess(self, row) -> np.ndarray:
        # Check if preprocessed image exists
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        preprocessed_path = os.path.join(project_root, "dataset", "data", "images_preprocessed", row["image_path"])
        
        if os.path.exists(preprocessed_path):
            image = cv2.imread(preprocessed_path)
            if image is not None:
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Use full_path if present, else join root_dir + image_path
        if "full_path" in self.df.columns:
            img_path = row["full_path"]
        else:
            img_path = os.path.join(self.root_dir, row["image_path"])

        image = cv2.imread(str(img_path))

        if image is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")

        if self.preprocess:
            image = remove_hair(image)
            image = reinhard_normalise(image)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    # FIX 3: removed the erroneous duplicate nested `def _get_label` stub
    #         that was syntactically inside this method's body
    def _get_label(self, row) -> int:
        if self.mode == "binary":
            return BINARY_LABELS.get(str(row.get("label", "non-BCC")), 0)
        else:
            return SUBTYPE_LABELS.get(str(row.get("subtype", "nodular")), 0)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = self._load_and_preprocess(row)
        label = self._get_label(row)
        minority = self.is_minority[idx]

        if self.is_train:
            transform = get_train_transforms(is_minority=minority)
        else:
            transform = get_val_transforms()

        augmented = transform(image=image)
        return augmented["image"], label


# ─────────────────────────────────────────────────────────────────────────────
# 6.  DATALOADER FACTORY
# ─────────────────────────────────────────────────────────────────────────────
def build_dataloaders(train_csv: str,
                      val_csv: str,
                      test_csv: str,
                      root_dir: str = "",
                      mode: str = 'binary',
                      batch_size: int = 32,
                      num_workers: int = 8,
                      preprocess: bool = True) -> Dict[str, DataLoader]:
    train_ds = BCCDataset(train_csv, root_dir, mode, 'train', preprocess)
    val_ds   = BCCDataset(val_csv,   root_dir, mode, 'val',   preprocess)
    test_ds  = BCCDataset(test_csv,  root_dir, mode, 'test',  preprocess)

    # FIX 4: get_class_weights is now mode-aware, so this is consistent
    class_weights = train_ds.get_class_weights()

    label_col = 'label' if mode == 'binary' else 'subtype'
    label_map  = BINARY_LABELS if mode == 'binary' else SUBTYPE_LABELS

    sample_weights = []
    for _, row in train_ds.df.iterrows():
        lbl = label_map.get(str(row.get(label_col, '')), 0)
        sample_weights.append(class_weights[lbl].item())

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    loaders = {
        'train': DataLoader(train_ds, batch_size=batch_size,
                            sampler=sampler, num_workers=num_workers,
                            pin_memory=True, drop_last=True),
        'val':   DataLoader(val_ds,   batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True),
        'test':  DataLoader(test_ds,  batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True),
    }

    print(f"[Data] Train={len(train_ds):,}  Val={len(val_ds):,}  "
          f"Test={len(test_ds):,}  Mode={mode}")
    return loaders


# ─────────────────────────────────────────────────────────────────────────────
# 7.  DEMO — generate a dummy CSV for testing without real data
# ─────────────────────────────────────────────────────────────────────────────
def create_demo_csv(out_dir: str = ".", n: int = 100):
    """
    Creates demo CSV files (train/val/test) using synthetic random images.
    Useful for verifying pipeline without real dermoscopic data.

    Args:
        out_dir (str): Directory to save demo CSVs and images.
        n       (int): Number of samples per split.
    """
    import random
    img_dir = os.path.join(out_dir, "demo_images")
    os.makedirs(img_dir, exist_ok=True)

    labels   = ['BCC', 'non-BCC']
    subtypes = ['nodular', 'superficial', 'infiltrative',
                'morpheaform', 'basosquamous']

    rows = []
    for i in range(n):
        fname = f"img_{i:04d}.jpg"
        fpath = os.path.join(img_dir, fname)
        img = (np.random.rand(380, 380, 3) * 255).astype(np.uint8)
        cv2.imwrite(fpath, img)

        lbl     = random.choice(labels)
        subtype = random.choice(subtypes) if lbl == 'BCC' else 'none'
        rows.append({'image_path': os.path.join("demo_images", fname),
                     'label': lbl, 'subtype': subtype})

    df = pd.DataFrame(rows)

    # FIX 5: use direct index arithmetic — correct for any value of n
    splits = [
        ('train', 0,          int(n * 0.70)),
        ('val',   int(n * 0.70), int(n * 0.85)),
        ('test',  int(n * 0.85), n),
    ]
    for name, s, e in splits:
        df.iloc[s:e].to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)

    print(f"[Demo] CSV files written to {out_dir}/")


if __name__ == "__main__":
    create_demo_csv("./demo_data", n=100)
    loaders = build_dataloaders(
        train_csv="./demo_data/train.csv",
        val_csv  ="./demo_data/val.csv",
        test_csv ="./demo_data/test.csv",
        root_dir ="./demo_data",
        mode='binary', batch_size=4, num_workers=0, preprocess=False,
    )
    imgs, lbls = next(iter(loaders['train']))
    print(f"[Test] Batch shape: {imgs.shape}, Labels: {lbls}")