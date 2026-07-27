from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image
from torch.utils.data import RandomSampler, SequentialSampler

from data.caltech_dataloaders import (
    build_caltech_dataloaders,
    build_eval_transform,
    build_train_transform,
)
from data.inspect_caltech_dataloader import inspect_dataloaders


def make_workspace_test_dir() -> Path:
    test_dir = Path("data/processed") / f"test_dataloaders_{uuid.uuid4().hex}"
    test_dir.mkdir(parents=True)
    return test_dir


def save_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 30), color=color).save(path)


def split_rows(prefix: str, count: int) -> list[dict[str, str]]:
    categories = ("empty", "deer", "coyote", "bobcat", "bird", "opossum")
    return [
        {
            "image_id": f"{prefix}-{index}",
            "file_name": f"{prefix}/{index}.jpg",
            "category_name": categories[index % len(categories)],
            "location": f"loc-{prefix}-{index}",
            "seq_id": f"seq-{prefix}-{index}",
            "split": prefix,
        }
        for index in range(count)
    ]


def write_split(path: Path, rows: list[dict[str, str]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def build_split_fixture() -> tuple[Path, Path, Path, Path, Path]:
    test_dir = make_workspace_test_dir()
    image_root = test_dir / "images"
    train_csv = test_dir / "train.csv"
    val_csv = test_dir / "val.csv"
    test_csv = test_dir / "test.csv"
    all_rows = {
        train_csv: split_rows("train", 8),
        val_csv: split_rows("val", 6),
        test_csv: split_rows("test", 6),
    }
    for rows in all_rows.values():
        for index, row in enumerate(rows):
            save_image(
                image_root / row["file_name"],
                color=(20 + index, 70 + index, 120 + index),
            )
    for path, rows in all_rows.items():
        write_split(path, rows)
    return test_dir, image_root, train_csv, val_csv, test_csv


def test_dataloader_factory_shapes_collation_and_samplers() -> None:
    test_dir, image_root, train_csv, val_csv, test_csv = build_split_fixture()
    try:
        loaders = build_caltech_dataloaders(
            train_csv=train_csv,
            val_csv=val_csv,
            test_csv=test_csv,
            image_root=image_root,
            batch_size=4,
            num_workers=0,
            image_size=224,
            seed=42,
            pin_memory=False,
        )

        train_batch = next(iter(loaders.train_loader))
        val_batch = next(iter(loaders.val_loader))
        test_batch = next(iter(loaders.test_loader))

        assert train_batch["image"].shape == (4, 3, 224, 224)
        assert train_batch["label"].shape == (4,)
        assert val_batch["image"].shape == (4, 3, 224, 224)
        assert val_batch["label"].shape == (4,)
        assert test_batch["image"].shape == (4, 3, 224, 224)
        assert test_batch["label"].shape == (4,)
        assert isinstance(loaders.train_loader.sampler, RandomSampler)
        assert isinstance(loaders.val_loader.sampler, SequentialSampler)
        assert isinstance(loaders.test_loader.sampler, SequentialSampler)
        assert len(train_batch["image_id"]) == 4
        assert train_batch["image_id"] != ["train-0", "train-1", "train-2", "train-3"]
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_eval_transform_is_deterministic_and_train_preserves_shape_dtype() -> None:
    test_dir = make_workspace_test_dir()
    try:
        image_path = test_dir / "image.jpg"
        save_image(image_path, (50, 80, 110))
        with Image.open(image_path) as image:
            rgb = image.convert("RGB").copy()

        eval_transform = build_eval_transform(image_size=224)
        first_eval = eval_transform(rgb)
        second_eval = eval_transform(rgb)
        train_transform = build_train_transform(image_size=224)
        first_train = train_transform(rgb)
        second_train = train_transform(rgb)

        assert torch.equal(first_eval, second_eval)
        assert first_train.shape == (3, 224, 224)
        assert second_train.shape == (3, 224, 224)
        assert first_train.dtype == torch.float32
        assert second_train.dtype == torch.float32
        assert int(torch.isnan(first_eval).sum().item()) == 0
        assert int(torch.isinf(first_eval).sum().item()) == 0
        assert int(torch.isnan(first_train).sum().item()) == 0
        assert int(torch.isinf(first_train).sum().item()) == 0
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_dataloader_factory_uses_correct_transforms() -> None:
    test_dir, image_root, train_csv, val_csv, test_csv = build_split_fixture()
    try:
        loaders = build_caltech_dataloaders(
            train_csv=train_csv,
            val_csv=val_csv,
            test_csv=test_csv,
            image_root=image_root,
            batch_size=2,
            image_size=64,
            seed=7,
            pin_memory=False,
        )

        val_first = loaders.val_dataset[0]["image"]
        val_second = loaders.val_dataset[0]["image"]
        train_first = loaders.train_dataset[0]["image"]

        assert torch.equal(val_first, val_second)
        assert train_first.shape == (3, 64, 64)
        assert val_first.shape == (3, 64, 64)
        assert loaders.train_dataset.transform is not loaders.val_dataset.transform
        assert loaders.val_dataset.transform is loaders.test_dataset.transform
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_inspection_cli_function_creates_outputs_and_valid_summary() -> None:
    test_dir, image_root, train_csv, val_csv, test_csv = build_split_fixture()
    try:
        output_dir = test_dir / "artifacts"
        summary = inspect_dataloaders(
            train_csv=train_csv,
            val_csv=val_csv,
            test_csv=test_csv,
            image_root=image_root,
            output_dir=output_dir,
            batch_size=4,
            num_workers=0,
            seed=42,
            image_size=64,
            samples_per_grid=4,
        )

        for filename in (
            "train_batch.png",
            "val_batch.png",
            "test_batch.png",
            "dataloader_summary.json",
        ):
            assert (output_dir / filename).exists()

        saved_summary = json.loads(
            (output_dir / "dataloader_summary.json").read_text(encoding="utf-8")
        )
        assert summary["validation_passed"] is True
        assert saved_summary["validation_passed"] is True
        assert saved_summary["validation_transform_deterministic"] is True
        assert saved_summary["fixed_class_mapping_valid"] is True
        assert saved_summary["inspected_splits"]["train"]["nan_count"] == 0
        assert saved_summary["inspected_splits"]["train"]["inf_count"] == 0
        assert saved_summary["inspected_splits"]["val"]["image_batch_shape"] == [
            4,
            3,
            64,
            64,
        ]
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_invalid_dataloader_arguments_are_rejected() -> None:
    test_dir, image_root, train_csv, val_csv, test_csv = build_split_fixture()
    try:
        with pytest.raises(ValueError, match="batch_size"):
            build_caltech_dataloaders(
                train_csv=train_csv,
                val_csv=val_csv,
                test_csv=test_csv,
                image_root=image_root,
                batch_size=0,
            )
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)
