from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import torch
from PIL import Image
from torch.utils.data import Dataset

from .download_caltech_subset import safe_destination_path

CLASS_NAMES = ("empty", "deer", "coyote", "bobcat", "bird", "opossum")
CLASS_TO_INDEX = {
    "empty": 0,
    "deer": 1,
    "coyote": 2,
    "bobcat": 3,
    "bird": 4,
    "opossum": 5,
}
INDEX_TO_CLASS = {index: name for name, index in CLASS_TO_INDEX.items()}
REQUIRED_COLUMNS = ("image_id", "file_name", "category_name", "location", "seq_id")


class CaltechCameraTrapDataset(Dataset[dict[str, Any]]):
    """Map-style Dataset for Caltech Camera Traps split CSV files."""

    def __init__(
        self,
        csv_path: Path,
        image_root: Path,
        transform: Callable[[Image.Image], Any] | None = None,
        strict: bool = True,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.image_root = Path(image_root)
        self.transform = transform
        self.strict = strict
        self.metadata = self._read_and_validate_csv(self.csv_path)
        self._records = cast(
            list[dict[str, Any]],
            self.metadata[list(REQUIRED_COLUMNS)].to_dict(orient="records"),
        )

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self._records[index]
        file_name = str(row["file_name"])
        image_path = self.resolve_image_path(file_name)

        if not image_path.exists():
            raise FileNotFoundError(
                "Image file referenced by split CSV was not found. "
                f"csv_path={self.csv_path}; row_index={index}; "
                f"file_name={file_name}; resolved_image_path={image_path}"
            )

        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB").copy()

        transformed_image: Any
        if self.transform is None:
            transformed_image = self._pil_rgb_to_float_tensor(rgb_image)
        else:
            transformed_image = self.transform(rgb_image)

        if not isinstance(transformed_image, torch.Tensor):
            raise TypeError(
                "Dataset transform must return a torch.Tensor; "
                f"got {type(transformed_image).__name__}"
            )

        category_name = str(row["category_name"])
        label_index = CLASS_TO_INDEX[category_name]
        return {
            "image": transformed_image,
            "label": torch.tensor(label_index, dtype=torch.int64),
            "image_id": str(row["image_id"]),
            "file_name": file_name,
            "category_name": category_name,
            "location": str(row["location"]),
            "seq_id": str(row["seq_id"]),
        }

    def resolve_image_path(self, file_name: str) -> Path:
        """Resolve a manifest file_name using the downloader's path rules."""
        return safe_destination_path(self.image_root, file_name)

    @staticmethod
    def _pil_rgb_to_float_tensor(image: Image.Image) -> torch.Tensor:
        array = np.asarray(image, dtype=np.uint8).copy()
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return tensor.to(dtype=torch.float32).div(255.0)

    @staticmethod
    def _read_and_validate_csv(csv_path: Path) -> pd.DataFrame:
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset CSV does not exist: {csv_path}")
        if not csv_path.is_file():
            raise ValueError(f"Dataset CSV path is not a file: {csv_path}")

        df = pd.read_csv(csv_path)
        missing_columns = [
            column for column in REQUIRED_COLUMNS if column not in df.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Dataset CSV is missing required columns: {missing_columns}"
            )

        null_columns = [
            column for column in REQUIRED_COLUMNS if df[column].isnull().any()
        ]
        if null_columns:
            raise ValueError(
                f"Dataset CSV has null values in required columns: {null_columns}"
            )

        discovered_categories = {
            str(category) for category in df["category_name"].unique()
        }
        unknown_categories = sorted(discovered_categories - set(CLASS_NAMES))
        if unknown_categories:
            raise ValueError(
                "Dataset CSV contains unknown category_name values: "
                f"{unknown_categories}"
            )

        return df.reset_index(drop=True)
