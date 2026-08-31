"""
src/data_pipeline.py
Data cleaning, augmentation, and DataLoader construction for Food-101.
Handles download, integrity checks, and train/val/test splits.
"""

import os
import json
import shutil
import hashlib
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from PIL import Image, ImageFile
from collections import Counter
from configs.food_classes import SELECTED_CLASSES
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.datasets as datasets

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import (
    DATA_DIR, RAW_DATA_DIR, PROCESSED_DIR,
    IMAGE_SIZE, DATASET_MEAN, DATASET_STD,
    BATCH_SIZE, NUM_WORKERS, SEED,
    AUG_HFLIP_P, AUG_ROTATION, AUG_BRIGHTNESS, AUG_CONTRAST, AUG_SATURATION
)

ImageFile.LOAD_TRUNCATED_IMAGES = True
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ─── Transforms ───────────────────────────────────────────────────────────────
def get_train_transforms() -> T.Compose:
    return T.Compose([
        T.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(p=AUG_HFLIP_P),
        T.RandomRotation(degrees=AUG_ROTATION),
        T.ColorJitter(
            brightness=AUG_BRIGHTNESS,
            contrast=AUG_CONTRAST,
            saturation=AUG_SATURATION,
            hue=0.05,
        ),
        T.RandomGrayscale(p=0.05),
        T.ToTensor(),
        T.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
    ])


def get_val_transforms() -> T.Compose:
    return T.Compose([
        T.Resize(int(IMAGE_SIZE * 1.14)),   # 256 for 224 target
        T.CenterCrop(IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
    ])


def get_inference_transforms() -> T.Compose:
    """Same as val transforms — used during Streamlit inference."""
    return get_val_transforms()


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Reverse ImageNet normalisation for visualization."""
    mean = torch.tensor(DATASET_MEAN).view(3, 1, 1)
    std  = torch.tensor(DATASET_STD).view(3, 1, 1)
    return torch.clamp(tensor * std + mean, 0, 1)


# ─── Custom Dataset ────────────────────────────────────────────────────────────


class FoodDataset(Dataset):
    """
    Official Food-101 Dataset Loader

    Expects:

    data/raw/food-101/
        images/
        meta/
            train.txt
            test.txt
    """

    def __init__(
        self,
        root: Path,
        split: str = "train",
        transform=None,
        class_to_idx=None,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform

        self.image_root = self.root / "images"
        self.meta_file = self.root / "meta" / f"{split}.txt"

        if not self.meta_file.exists():
            raise FileNotFoundError(
                f"Could not find {self.meta_file}"
            )

        self.classes = sorted(SELECTED_CLASSES)

        if class_to_idx is None:
            self.class_to_idx = {
                c: i for i, c in enumerate(self.classes)
            }
        else:
            self.class_to_idx = class_to_idx

        self.samples = []

        with open(self.meta_file, "r") as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                cls = line.split("/")[0]

                if cls not in self.class_to_idx:
                    continue

                img_path = self.image_root / f"{line}.jpg"

                if img_path.exists():

                    self.samples.append(
                        (
                            img_path,
                            self.class_to_idx[cls]
                        )
                    )

        log.info(
            f"[{split}] Loaded {len(self.samples)} images "
            f"across {len(self.classes)} classes."
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label, str(img_path)


# ─── Data Cleaning ─────────────────────────────────────────────────────────────
class DataCleaner:
    """
    Runs integrity checks on the Food-101 dataset:
    - Removes corrupted / unreadable images
    - Detects and flags near-duplicates (MD5 hash)
    - Validates image resolution
    - Generates an EDA report
    """
    def __init__(self, data_root: Path, min_size: int = 32):
        self.data_root   = Path(data_root)
        self.min_size    = min_size
        self.report: Dict = {
            "total": 0, "corrupted": 0, "duplicates": 0,
            "too_small": 0, "removed": 0, "clean": 0,
        }

    def run(self, splits: List[str] = ["train", "test"]) -> Dict:
        seen_hashes = set()
        for split in splits:
            split_dir = self.data_root / split
            if not split_dir.exists():
                log.warning(f"Split dir {split_dir} does not exist, skipping.")
                continue

            for img_path in split_dir.rglob("*.jpg"):
                self.report["total"] += 1
                remove = False

                # ── Corruption check ───────────────────────────────────────
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                    with Image.open(img_path) as img:
                        w, h = img.size
                except Exception as e:
                    log.debug(f"Corrupted: {img_path} — {e}")
                    self.report["corrupted"] += 1
                    remove = True
                    w, h = 0, 0

                if not remove:
                    # ── Size check ─────────────────────────────────────────
                    if w < self.min_size or h < self.min_size:
                        self.report["too_small"] += 1
                        remove = True

                    # ── Duplicate check (MD5) ──────────────────────────────
                    if not remove:
                        md5 = hashlib.md5(img_path.read_bytes()).hexdigest()
                        if md5 in seen_hashes:
                            self.report["duplicates"] += 1
                            remove = True
                        else:
                            seen_hashes.add(md5)

                if remove:
                    img_path.unlink(missing_ok=True)
                    self.report["removed"] += 1

        self.report["clean"] = self.report["total"] - self.report["removed"]
        log.info(f"[Cleaner] Report: {self.report}")
        return self.report

    def generate_eda_report(self, split: str = "train") -> pd.DataFrame:
        """
        Build a per-class image-count DataFrame for EDA.
        """
        records = []
        split_dir = self.data_root / split
        if not split_dir.exists():
            return pd.DataFrame()

        for cls_dir in sorted(split_dir.iterdir()):
            if cls_dir.is_dir():
                count = len(list(cls_dir.glob("*.jpg")))
                records.append({"class": cls_dir.name, "count": count})

        df = pd.DataFrame(records).sort_values("count", ascending=False)
        log.info(f"[EDA] {split}: {len(df)} classes, "
                 f"min={df['count'].min()}, max={df['count'].max()}, "
                 f"mean={df['count'].mean():.0f}")
        return df


# ─── DataLoader Factory ────────────────────────────────────────────────────────
def get_dataloaders(
    data_root: Path,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    use_weighted_sampler: bool = True,
    seed: int = SEED,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str], Dict[str, int]]:
    """
    Build train / val / test DataLoaders from the Food-101 directory structure.
    Food-101 comes with predefined train and test splits.
    We carve out 10% of train as validation.

    Returns: (train_loader, val_loader, test_loader, class_names, class_to_idx)
    """
    torch.manual_seed(seed)

    train_full = FoodDataset(
        root=data_root, split="train",
        transform=get_train_transforms(),
    )
    class_names  = train_full.classes
    class_to_idx = train_full.class_to_idx
    n_classes    = len(class_names)

    # ── Val split (10% of train) ───────────────────────────────────────────
    n_total = len(train_full)
    n_val   = int(n_total * 0.10)
    n_train = n_total - n_val
    gen     = torch.Generator().manual_seed(seed)
    train_ds, val_ds = torch.utils.data.random_split(
        train_full, [n_train, n_val], generator=gen
    )
    # Override val augmentations
    val_ds.dataset.transform = get_val_transforms()

    # ── Test set ─────────────────────────────────────────────────────────────
    test_ds = FoodDataset(
        root=data_root, split="test",
        transform=get_val_transforms(),
        class_to_idx=class_to_idx,
    )

    # ── Weighted sampler for class balance ────────────────────────────────────
    train_loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    if use_weighted_sampler:
        labels = [train_full.samples[i][1] for i in train_ds.indices]
        counts = Counter(labels)
        weights = [1.0 / counts[l] for l in labels]
        sampler = WeightedRandomSampler(
            weights=weights, num_samples=len(weights), replacement=True
        )
        train_loader = DataLoader(train_ds, sampler=sampler, **train_loader_kwargs)
    else:
        train_loader = DataLoader(train_ds, shuffle=True, **train_loader_kwargs)

    val_loader  = DataLoader(val_ds,  shuffle=False, batch_size=batch_size,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, shuffle=False, batch_size=batch_size,
                             num_workers=num_workers, pin_memory=True)

    log.info(f"[DataLoader] Train={len(train_ds)} | Val={len(val_ds)} | Test={len(test_ds)}")
    log.info(f"[DataLoader] Classes={n_classes} | Batch={batch_size}")

    return train_loader, val_loader, test_loader, class_names, class_to_idx


# ─── Inference helper ─────────────────────────────────────────────────────────
def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Pre-process a single PIL image for model inference.
    Returns a [1, 3, H, W] tensor.
    """
    transform = get_inference_transforms()
    return transform(image.convert("RGB")).unsqueeze(0)


if __name__ == "__main__":
    # Quick smoke test
    cleaner = DataCleaner(RAW_DATA_DIR / "food-101")
    print("DataCleaner ready. Call cleaner.run() to clean the dataset.")
