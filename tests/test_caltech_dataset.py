from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from data.caltech_dataloaders import build_eval_transform
from data.caltech_dataset import (
    CLASS_NAMES,
    CLASS_TO_INDEX,
    INDEX_TO_CLASS,
    CaltechCameraTrapDataset,
)


def make_workspace_test_dir() -> Path:
    test_dir = Path("data/processed") / f"test_dataset_{uuid.uuid4().hex}"
    test_dir.mkdir(parents=True)
    return test_dir


def image_row(
    image_id: str,
    file_name: str,
    category_name: str,
    location: str = "loc-a",
    seq_id: str = "seq-a",
) -> dict[str, str]:
    return {
        "image_id": image_id,
        "file_name": file_name,
        "category_name": category_name,
        "location": location,
        "seq_id": seq_id,
    }


def save_image(path: Path, mode: str = "RGB") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "L":
        image = Image.new("L", (32, 24), color=128)
    elif mode == "RGBA":
        image = Image.new("RGBA", (32, 24), color=(40, 80, 120, 180))
    else:
        image = Image.new("RGB", (32, 24), color=(40, 80, 120))
    image.save(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def build_dataset_fixture() -> tuple[Path, Path, Path]:
    test_dir = make_workspace_test_dir()
    image_root = test_dir / "images"
    csv_path = test_dir / "split.csv"
    rows = [
        image_row("img-0", "nested/rgb.jpg", "empty", "loc-0", "seq-0"),
        image_row("img-1", "gray.png", "deer", "loc-1", "seq-1"),
        image_row("img-2", "rgba.png", "coyote", "loc-2", "seq-2"),
    ]
    save_image(image_root / "nested" / "rgb.jpg", "RGB")
    save_image(image_root / "gray.png", "L")
    save_image(image_root / "rgba.png", "RGBA")
    write_csv(csv_path, rows)
    return test_dir, image_root, csv_path


def test_fixed_class_mappings_are_exact() -> None:
    assert CLASS_NAMES == ("empty", "deer", "coyote", "bobcat", "bird", "opossum")
    assert CLASS_TO_INDEX == {
        "empty": 0,
        "deer": 1,
        "coyote": 2,
        "bobcat": 3,
        "bird": 4,
        "opossum": 5,
    }
    assert INDEX_TO_CLASS == {index: name for name, index in CLASS_TO_INDEX.items()}


def test_dataset_len_keys_shape_dtype_and_labels() -> None:
    test_dir, image_root, csv_path = build_dataset_fixture()
    try:
        dataset = CaltechCameraTrapDataset(
            csv_path=csv_path,
            image_root=image_root,
            transform=build_eval_transform(),
        )

        first = dataset[0]
        assert len(dataset) == 3
        assert set(first) == {
            "image",
            "label",
            "image_id",
            "file_name",
            "category_name",
            "location",
            "seq_id",
        }
        assert first["image"].shape == (3, 224, 224)
        assert first["image"].dtype == torch.float32
        assert first["label"].dtype == torch.int64
        assert first["label"].shape == torch.Size([])
        assert first["label"].item() == CLASS_TO_INDEX["empty"]
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_grayscale_and_rgba_images_become_three_channel_tensors() -> None:
    test_dir, image_root, csv_path = build_dataset_fixture()
    try:
        dataset = CaltechCameraTrapDataset(
            csv_path=csv_path,
            image_root=image_root,
            transform=build_eval_transform(),
        )

        assert dataset[1]["image"].shape == (3, 224, 224)
        assert dataset[2]["image"].shape == (3, 224, 224)
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_missing_image_file_error_is_clear() -> None:
    test_dir = make_workspace_test_dir()
    try:
        image_root = test_dir / "images"
        csv_path = test_dir / "split.csv"
        write_csv(csv_path, [image_row("missing", "missing.jpg", "empty")])
        dataset = CaltechCameraTrapDataset(
            csv_path=csv_path,
            image_root=image_root,
            transform=build_eval_transform(),
        )

        with pytest.raises(FileNotFoundError) as exc_info:
            _ = dataset[0]

        message = str(exc_info.value)
        assert "missing.jpg" in message
        assert str(image_root.resolve() / "missing.jpg") in message
        assert str(csv_path) in message
        assert "row_index=0" in message
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_missing_required_csv_columns_are_rejected() -> None:
    test_dir = make_workspace_test_dir()
    try:
        csv_path = test_dir / "split.csv"
        write_csv(csv_path, [image_row("img", "img.jpg", "empty")])
        df = pd.read_csv(csv_path).drop(columns=["seq_id"])
        df.to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="missing required columns"):
            CaltechCameraTrapDataset(csv_path=csv_path, image_root=test_dir)
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_null_required_values_are_rejected() -> None:
    test_dir = make_workspace_test_dir()
    try:
        csv_path = test_dir / "split.csv"
        write_csv(csv_path, [image_row("img", "img.jpg", "empty", location="")])

        with pytest.raises(ValueError, match="null values"):
            CaltechCameraTrapDataset(csv_path=csv_path, image_root=test_dir)
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_unknown_categories_are_rejected() -> None:
    test_dir = make_workspace_test_dir()
    try:
        csv_path = test_dir / "split.csv"
        write_csv(csv_path, [image_row("img", "img.jpg", "fox")])

        with pytest.raises(ValueError, match="unknown category_name"):
            CaltechCameraTrapDataset(csv_path=csv_path, image_root=test_dir)
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_dataset_preserves_csv_row_ordering() -> None:
    test_dir, image_root, csv_path = build_dataset_fixture()
    try:
        dataset = CaltechCameraTrapDataset(
            csv_path=csv_path,
            image_root=image_root,
            transform=build_eval_transform(),
        )

        assert [dataset[index]["image_id"] for index in range(len(dataset))] == [
            "img-0",
            "img-1",
            "img-2",
        ]
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_default_dataloader_collation_handles_metadata() -> None:
    test_dir, image_root, csv_path = build_dataset_fixture()
    try:
        dataset = CaltechCameraTrapDataset(
            csv_path=csv_path,
            image_root=image_root,
            transform=build_eval_transform(),
        )
        batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))

        assert batch["image"].shape == (2, 3, 224, 224)
        assert batch["label"].shape == (2,)
        assert batch["image_id"] == ["img-0", "img-1"]
        assert batch["file_name"] == ["nested/rgb.jpg", "gray.png"]
        assert batch["category_name"] == ["empty", "deer"]
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)
