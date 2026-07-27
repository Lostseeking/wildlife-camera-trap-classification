# ruff: noqa: E402,I001

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

Path("artifacts/matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(Path("artifacts/matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import torchvision  # type: ignore[import-untyped]

from .caltech_dataloaders import (
    DEFAULT_IMAGE_ROOT,
    DEFAULT_TEST_CSV,
    DEFAULT_TRAIN_CSV,
    DEFAULT_VAL_CSV,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    CaltechDataLoaders,
    build_caltech_dataloaders,
)
from .caltech_dataset import CLASS_TO_INDEX

DEFAULT_OUTPUT_DIR = Path("artifacts/data_checks")
REQUIRED_BATCH_KEYS = (
    "image",
    "label",
    "image_id",
    "file_name",
    "category_name",
    "location",
    "seq_id",
)
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect Caltech Camera Traps MVP DataLoader batches."
    )
    parser.add_argument("--train-csv", default=DEFAULT_TRAIN_CSV, type=Path)
    parser.add_argument("--val-csv", default=DEFAULT_VAL_CSV, type=Path)
    parser.add_argument("--test-csv", default=DEFAULT_TEST_CSV, type=Path)
    parser.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT, type=Path)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--image-size", default=224, type=int)
    parser.add_argument("--samples-per-grid", default=16, type=int)
    return parser.parse_args()


def shorten(text: str, max_length: int = 18) -> str:
    if len(text) <= max_length:
        return text
    return f"{text[: max_length - 3]}..."


def metadata_values(value: Any) -> list[str]:
    if isinstance(value, list | tuple):
        return [str(item) for item in value]
    return [str(value)]


def inverse_normalize(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(NORMALIZE_MEAN, dtype=images.dtype, device=images.device).view(
        1, 3, 1, 1
    )
    std = torch.tensor(NORMALIZE_STD, dtype=images.dtype, device=images.device).view(
        1, 3, 1, 1
    )
    return images * std + mean


def save_batch_grid(
    batch: dict[str, Any],
    output_path: Path,
    samples_per_grid: int,
) -> None:
    images = inverse_normalize(batch["image"]).clamp(0.0, 1.0)
    labels = batch["label"].detach().cpu().tolist()
    categories = metadata_values(batch["category_name"])
    locations = metadata_values(batch["location"])
    file_names = metadata_values(batch["file_name"])
    sample_count = min(samples_per_grid, int(images.shape[0]))
    column_count = max(1, math.ceil(math.sqrt(sample_count)))
    row_count = max(1, math.ceil(sample_count / column_count))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(column_count * 3.0, row_count * 3.4),
        squeeze=False,
    )
    flat_axes = axes.ravel()
    for index, axis in enumerate(flat_axes):
        axis.axis("off")
        if index >= sample_count:
            continue
        image = images[index].detach().cpu().permute(1, 2, 0).numpy()
        axis.imshow(image)
        title = (
            f"{shorten(categories[index], 10)} | y={labels[index]}\n"
            f"loc {shorten(locations[index], 8)}\n"
            f"{shorten(file_names[index], 22)}"
        )
        axis.set_title(title, fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def validate_batch(
    split: str,
    batch: dict[str, Any],
    image_size: int,
    output_path: Path,
    samples_per_grid: int,
) -> dict[str, Any]:
    missing_keys = [key for key in REQUIRED_BATCH_KEYS if key not in batch]
    if missing_keys:
        raise ValueError(f"{split} batch is missing keys: {missing_keys}")

    images = batch["image"]
    labels = batch["label"]
    if not isinstance(images, torch.Tensor):
        raise TypeError(f"{split} image batch must be a torch.Tensor")
    if not isinstance(labels, torch.Tensor):
        raise TypeError(f"{split} label batch must be a torch.Tensor")

    batch_size = int(images.shape[0])
    expected_image_shape = (batch_size, 3, image_size, image_size)
    if tuple(images.shape) != expected_image_shape:
        raise ValueError(
            f"{split} image batch shape mismatch: "
            f"{tuple(images.shape)} != {expected_image_shape}"
        )
    if tuple(labels.shape) != (batch_size,):
        raise ValueError(f"{split} label batch shape mismatch: {tuple(labels.shape)}")
    if images.dtype != torch.float32:
        raise TypeError(f"{split} image dtype must be torch.float32: {images.dtype}")
    if labels.dtype != torch.int64:
        raise TypeError(f"{split} label dtype must be torch.int64: {labels.dtype}")

    nan_count = int(torch.isnan(images).sum().item())
    inf_count = int(torch.isinf(images).sum().item())
    if nan_count or inf_count:
        raise ValueError(f"{split} batch contains NaN={nan_count}, Inf={inf_count}")

    categories = metadata_values(batch["category_name"])
    for key in ("image_id", "file_name", "category_name", "location", "seq_id"):
        values = metadata_values(batch[key])
        if len(values) != batch_size:
            raise ValueError(
                f"{split} metadata length mismatch for {key}: "
                f"{len(values)} != {batch_size}"
            )

    label_values = [int(value) for value in labels.detach().cpu().tolist()]
    consistency = all(
        CLASS_TO_INDEX[category] == label
        for category, label in zip(categories, label_values, strict=True)
    )
    if not consistency:
        raise ValueError(f"{split} numeric labels do not match category_name values")

    save_batch_grid(
        batch=batch,
        output_path=output_path,
        samples_per_grid=samples_per_grid,
    )

    summary = {
        "inspected_batch_size": batch_size,
        "image_batch_shape": list(images.shape),
        "label_batch_shape": list(labels.shape),
        "image_dtype": str(images.dtype),
        "label_dtype": str(labels.dtype),
        "normalized_minimum": float(images.min().item()),
        "normalized_maximum": float(images.max().item()),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "numeric_label_counts": {
            str(label): count for label, count in sorted(Counter(label_values).items())
        },
        "category_name_counts": dict(sorted(Counter(categories).items())),
        "label_category_consistency": consistency,
        "saved_visualization_path": str(output_path),
    }
    LOGGER.info(
        "%s batch: image_shape=%s label_shape=%s image_dtype=%s label_dtype=%s",
        split,
        list(images.shape),
        list(labels.shape),
        images.dtype,
        labels.dtype,
    )
    LOGGER.info("%s batch NaN=%s Inf=%s", split, nan_count, inf_count)
    return summary


def next_batch(loader: Any) -> dict[str, Any]:
    batch = next(iter(loader))
    if not isinstance(batch, dict):
        raise TypeError(f"Expected DataLoader batch dictionary, got {type(batch)}")
    return batch


def validate_transform_behavior(loaders: CaltechDataLoaders) -> dict[str, bool]:
    first_val = loaders.val_dataset[0]["image"]
    second_val = loaders.val_dataset[0]["image"]
    validation_transform_deterministic = bool(torch.equal(first_val, second_val))

    first_train = loaders.train_dataset[0]["image"]
    second_train = loaders.train_dataset[0]["image"]
    train_transform_shape_valid = bool(first_train.shape == second_train.shape)
    train_transform_dtype_valid = bool(
        first_train.dtype == torch.float32 and second_train.dtype == torch.float32
    )

    return {
        "validation_transform_deterministic": validation_transform_deterministic,
        "train_transform_shape_valid": train_transform_shape_valid,
        "train_transform_dtype_valid": train_transform_dtype_valid,
    }


def inspect_dataloaders(
    train_csv: Path = DEFAULT_TRAIN_CSV,
    val_csv: Path = DEFAULT_VAL_CSV,
    test_csv: Path = DEFAULT_TEST_CSV,
    image_root: Path = DEFAULT_IMAGE_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    batch_size: int = 16,
    num_workers: int = 0,
    seed: int = 42,
    image_size: int = 224,
    samples_per_grid: int = 16,
) -> dict[str, Any]:
    loaders = build_caltech_dataloaders(
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        image_root=image_root,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        seed=seed,
        pin_memory=None,
    )
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_lengths = {
        "train": len(loaders.train_dataset),
        "val": len(loaders.val_dataset),
        "test": len(loaders.test_dataset),
    }
    dataloader_batch_counts = {
        "train": len(loaders.train_loader),
        "val": len(loaders.val_loader),
        "test": len(loaders.test_loader),
    }

    LOGGER.info("torch version: %s", torch.__version__)
    LOGGER.info("torchvision version: %s", torchvision.__version__)
    LOGGER.info("CUDA available: %s", torch.cuda.is_available())
    LOGGER.info("pin_memory: %s", loaders.pin_memory)
    LOGGER.info("Dataset lengths: %s", dataset_lengths)
    LOGGER.info("DataLoader batch counts: %s", dataloader_batch_counts)

    split_batches = {
        "train": next_batch(loaders.train_loader),
        "val": next_batch(loaders.val_loader),
        "test": next_batch(loaders.test_loader),
    }
    inspected_splits = {
        split: validate_batch(
            split=split,
            batch=batch,
            image_size=image_size,
            output_path=output_dir / f"{split}_batch.png",
            samples_per_grid=samples_per_grid,
        )
        for split, batch in split_batches.items()
    }
    transform_checks = validate_transform_behavior(loaders)
    fixed_class_mapping_valid = CLASS_TO_INDEX == {
        "empty": 0,
        "deer": 1,
        "coyote": 2,
        "bobcat": 3,
        "bird": 4,
        "opossum": 5,
    }

    validation_passed = bool(
        all(transform_checks.values()) and fixed_class_mapping_valid
    )
    summary: dict[str, Any] = {
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_type": device_type,
        "seed": seed,
        "image_size": image_size,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": loaders.pin_memory,
        "train_csv_path": str(train_csv),
        "val_csv_path": str(val_csv),
        "test_csv_path": str(test_csv),
        "image_root": str(image_root),
        "dataset_lengths": dataset_lengths,
        "dataloader_batch_counts": dataloader_batch_counts,
        "inspected_splits": inspected_splits,
        **transform_checks,
        "fixed_class_mapping_valid": fixed_class_mapping_valid,
        "validation_passed": validation_passed,
    }
    if not validation_passed:
        raise ValueError("DataLoader inspection validation failed")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "dataloader_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Saved summary: %s", summary_path)
    LOGGER.info("Validation transform deterministic: %s", transform_checks)
    LOGGER.info("Final validation status: true")
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    try:
        inspect_dataloaders(
            train_csv=args.train_csv,
            val_csv=args.val_csv,
            test_csv=args.test_csv,
            image_root=args.image_root,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            image_size=args.image_size,
            samples_per_grid=args.samples_per_grid,
        )
    except Exception as exc:
        LOGGER.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
