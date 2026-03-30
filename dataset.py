# -*- coding: utf-8 -*-
"""Unified medical image segmentation dataset with albumentations augmentation."""

from __future__ import annotations

import os
from glob import glob

import cv2
import numpy as np
from PIL import Image
import albumentations as albu
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensorV2

from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Default augmentation pipelines
# ---------------------------------------------------------------------------

def get_train_transform(image_size: int = 256) -> Compose:
    return Compose([
        albu.Rotate(limit=(-15, 15), p=0.5),
        albu.RandomRotate90(p=0.5),
        albu.Flip(p=0.5),
        albu.HueSaturationValue(p=0.1),
        albu.RandomBrightnessContrast(p=0.1),
        albu.GaussianBlur(p=0.1),
        albu.GaussNoise(p=0.1),
        albu.ElasticTransform(p=0.1),
        albu.Resize(image_size, image_size),
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transform(image_size: int = 256) -> Compose:
    return Compose([
        albu.Resize(image_size, image_size),
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ---------------------------------------------------------------------------
# SegmentationDataset
# ---------------------------------------------------------------------------

class SegmentationDataset(Dataset):
    """Generic medical image segmentation dataset.

    Expected directory layout::

        <dataset_dir>/<dataset_name>/<split>/imgs/*.png
        <dataset_dir>/<dataset_name>/<split>/masks/*.png

    Supported datasets: ACDC, EchoNet-Dynamic, Fetal_HC, Synapse, FIVES,
    Kvasir-SEG, tn3k, cvc, and any dataset following the above layout.
    """

    def __init__(
        self,
        dataset_dir: str,
        dataset_name: str,
        split: str = "train",
        image_size: int = 256,
        transform: Compose | None = None,
    ) -> None:
        assert split in ("train", "val", "test"), f"Invalid split: {split}"
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.split = split
        self.image_size = image_size

        # File extensions per dataset
        if dataset_name in ("Kvasir-SEG", "tn3k"):
            self.img_ext = ".jpg"
            self.msk_ext = ".jpg"
        else:
            self.img_ext = ".png"
            self.msk_ext = ".png"

        dataset_path = os.path.join(dataset_dir, dataset_name)
        self.img_dir = os.path.join(dataset_path, split, "imgs")
        self.mask_dir = os.path.join(dataset_path, split, "masks")

        img_ids = glob(os.path.join(self.img_dir, f"*{self.img_ext}"))
        self.img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

        if split == "train":
            self.transform = transform or get_train_transform(image_size)
        else:
            self.transform = get_val_transform(image_size)

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int):
        file_name = self.img_ids[idx]
        img_path = os.path.join(self.img_dir, file_name + self.img_ext)
        mask_path = os.path.join(self.mask_dir, file_name + self.msk_ext)

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        mask = mask.float() / 255.0  # normalize to [0, 1]
        return image, mask, file_name


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def create_dataloader(
    dataset_dir: str,
    dataset_name: str,
    split: str = "train",
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: int = 256,
    transform: Compose | None = None,
) -> DataLoader:
    dataset = SegmentationDataset(
        dataset_dir, dataset_name, split=split,
        image_size=image_size, transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )
