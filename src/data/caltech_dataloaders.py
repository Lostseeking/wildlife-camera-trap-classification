from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import v2  # type: ignore[import-untyped]

from .caltech_dataset import CaltechCameraTrapDataset

NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)
DEFAULT_TRAIN_CSV = Path("data/processed/splits/train.csv")
DEFAULT_VAL_CSV = Path("data/processed/splits/val.csv")
DEFAULT_TEST_CSV = Path("data/processed/splits/test.csv")
DEFAULT_IMAGE_ROOT = Path("data/raw/caltech_camera_traps/images")

ImageTransform = Callable[[Image.Image], torch.Tensor]


@dataclass(frozen=True)
class CaltechDataLoaders:
    """Container for Caltech split Datasets and DataLoaders."""

    train_dataset: CaltechCameraTrapDataset
    val_dataset: CaltechCameraTrapDataset
    test_dataset: CaltechCameraTrapDataset
    train_loader: DataLoader[dict[str, Any]]
    val_loader: DataLoader[dict[str, Any]]
    test_loader: DataLoader[dict[str, Any]]
    pin_memory: bool


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and torch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def seed_worker(worker_id: int) -> None:
    """Seed DataLoader workers reproducibly from torch's worker seed."""
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_train_transform(image_size: int = 224) -> ImageTransform:
    """Build the baseline training transform for full-scene camera-trap images."""
    return cast(
        ImageTransform,
        v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(size=(image_size, image_size), antialias=True),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomApply(
                    [
                        v2.ColorJitter(
                            brightness=0.15,
                            contrast=0.15,
                            saturation=0.10,
                            hue=0.02,
                        )
                    ],
                    p=0.5,
                ),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
            ]
        ),
    )


def build_eval_transform(image_size: int = 224) -> ImageTransform:
    """Build the deterministic validation/test transform."""
    return cast(
        ImageTransform,
        v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(size=(image_size, image_size), antialias=True),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
            ]
        ),
    )


def build_caltech_dataloaders(
    train_csv: Path = DEFAULT_TRAIN_CSV,
    val_csv: Path = DEFAULT_VAL_CSV,
    test_csv: Path = DEFAULT_TEST_CSV,
    image_root: Path = DEFAULT_IMAGE_ROOT,
    batch_size: int = 32,
    num_workers: int = 0,
    image_size: int = 224,
    seed: int = 42,
    pin_memory: bool | None = None,
) -> CaltechDataLoaders:
    """Build train, validation, and test Datasets plus DataLoaders."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1: {batch_size}")
    if num_workers < 0:
        raise ValueError(f"num_workers must be non-negative: {num_workers}")
    if image_size < 1:
        raise ValueError(f"image_size must be at least 1: {image_size}")

    seed_everything(seed)
    resolved_pin_memory = (
        torch.cuda.is_available() if pin_memory is None else pin_memory
    )
    train_transform = build_train_transform(image_size=image_size)
    eval_transform = build_eval_transform(image_size=image_size)

    train_dataset = CaltechCameraTrapDataset(
        csv_path=train_csv,
        image_root=image_root,
        transform=train_transform,
    )
    val_dataset = CaltechCameraTrapDataset(
        csv_path=val_csv,
        image_root=image_root,
        transform=eval_transform,
    )
    test_dataset = CaltechCameraTrapDataset(
        csv_path=test_csv,
        image_root=image_root,
        transform=eval_transform,
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    eval_generator = torch.Generator()
    eval_generator.manual_seed(seed)

    train_loader: DataLoader[dict[str, Any]] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=resolved_pin_memory,
        generator=train_generator,
        worker_init_fn=seed_worker,
    )
    val_loader: DataLoader[dict[str, Any]] = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=resolved_pin_memory,
        generator=eval_generator,
        worker_init_fn=seed_worker,
    )
    test_loader: DataLoader[dict[str, Any]] = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=resolved_pin_memory,
        generator=eval_generator,
        worker_init_fn=seed_worker,
    )

    return CaltechDataLoaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        pin_memory=resolved_pin_memory,
    )
