from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pandas as pd
import pytest

from data.split_caltech_mvp import (
    ALLOWED_CATEGORIES,
    REQUIRED_COLUMNS,
    create_splits,
)


def image_row(
    image_id: str,
    category: str,
    location: str,
    seq_id: str,
) -> dict[str, str]:
    return {
        "image_id": image_id,
        "file_name": f"{image_id}.jpg",
        "category_name": category,
        "location": location,
        "seq_id": seq_id,
        "extra_column": f"extra-{image_id}",
    }


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def make_workspace_test_dir() -> Path:
    test_dir = Path("data/processed") / f"test_split_caltech_{uuid.uuid4().hex}"
    test_dir.mkdir(parents=True)
    return test_dir


def read_outputs(output_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "train": pd.read_csv(output_dir / "train.csv"),
        "val": pd.read_csv(output_dir / "val.csv"),
        "test": pd.read_csv(output_dir / "test.csv"),
    }


def balanced_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for location_index in range(12):
        for category in ALLOWED_CATEGORIES:
            image_id = f"loc-{location_index:02d}-{category}"
            rows.append(
                image_row(
                    image_id=image_id,
                    category=category,
                    location=f"loc-{location_index:02d}",
                    seq_id=f"seq-{image_id}",
                )
            )
    return rows


def split_fixture(test_dir: Path) -> tuple[Path, Path, dict[str, pd.DataFrame]]:
    manifest_path = test_dir / "manifest.csv"
    output_dir = test_dir / "splits"
    write_manifest(manifest_path, balanced_rows())
    create_splits(
        manifest_path=manifest_path,
        output_dir=output_dir,
        seed=123,
        num_restarts=40,
    )
    return manifest_path, output_dir, read_outputs(output_dir)


def combined(outputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(outputs.values(), ignore_index=True)


def test_zero_location_overlap() -> None:
    test_dir = make_workspace_test_dir()
    try:
        _manifest_path, _output_dir, outputs = split_fixture(test_dir)

        train_locations = set(outputs["train"]["location"])
        val_locations = set(outputs["val"]["location"])
        test_locations = set(outputs["test"]["location"])

        assert not train_locations & val_locations
        assert not train_locations & test_locations
        assert not val_locations & test_locations
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_all_rows_are_preserved_exactly_once() -> None:
    test_dir = make_workspace_test_dir()
    try:
        manifest_path, _output_dir, outputs = split_fixture(test_dir)

        input_df = pd.read_csv(manifest_path)
        output_df = combined(outputs)

        assert sorted(output_df["image_id"]) == sorted(input_df["image_id"])
        assert len(output_df) == len(input_df)
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_image_id_remains_unique() -> None:
    test_dir = make_workspace_test_dir()
    try:
        _manifest_path, _output_dir, outputs = split_fixture(test_dir)

        assert combined(outputs)["image_id"].is_unique
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_all_rows_from_one_location_remain_together() -> None:
    test_dir = make_workspace_test_dir()
    try:
        _manifest_path, _output_dir, outputs = split_fixture(test_dir)

        output_df = combined(outputs)

        assert output_df.groupby("location")["split"].nunique().max() == 1
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_deterministic_results_for_fixed_seed() -> None:
    test_dir = make_workspace_test_dir()
    try:
        manifest_path = test_dir / "manifest.csv"
        first_output = test_dir / "first"
        second_output = test_dir / "second"
        write_manifest(manifest_path, balanced_rows())

        create_splits(manifest_path=manifest_path, output_dir=first_output, seed=9)
        create_splits(manifest_path=manifest_path, output_dir=second_output, seed=9)

        for filename in ("train.csv", "val.csv", "test.csv"):
            assert (first_output / filename).read_text(encoding="utf-8") == (
                second_output / filename
            ).read_text(encoding="utf-8")
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_sensible_approximate_split_ratios() -> None:
    test_dir = make_workspace_test_dir()
    try:
        _manifest_path, _output_dir, outputs = split_fixture(test_dir)

        total_rows = len(combined(outputs))
        ratios = {
            split: len(split_df) / total_rows for split, split_df in outputs.items()
        }

        assert ratios["train"] == pytest.approx(0.70, abs=0.12)
        assert ratios["val"] == pytest.approx(0.15, abs=0.10)
        assert ratios["test"] == pytest.approx(0.15, abs=0.10)
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_category_coverage_when_feasible() -> None:
    test_dir = make_workspace_test_dir()
    try:
        _manifest_path, _output_dir, outputs = split_fixture(test_dir)

        for split_df in outputs.values():
            assert set(split_df["category_name"]) == set(ALLOWED_CATEGORIES)
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_invalid_ratios_are_rejected() -> None:
    test_dir = make_workspace_test_dir()
    try:
        manifest_path = test_dir / "manifest.csv"
        write_manifest(manifest_path, balanced_rows())

        with pytest.raises(ValueError, match="sum to 1.0"):
            create_splits(
                manifest_path=manifest_path,
                output_dir=test_dir / "out",
                train_ratio=0.8,
                val_ratio=0.1,
                test_ratio=0.2,
            )
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_missing_required_columns_are_rejected() -> None:
    test_dir = make_workspace_test_dir()
    try:
        manifest_path = test_dir / "manifest.csv"
        rows = balanced_rows()
        pd.DataFrame(rows).drop(columns=["seq_id"]).to_csv(manifest_path, index=False)

        with pytest.raises(ValueError, match="missing required columns"):
            create_splits(manifest_path=manifest_path, output_dir=test_dir / "out")
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_null_required_values_are_rejected() -> None:
    test_dir = make_workspace_test_dir()
    try:
        manifest_path = test_dir / "manifest.csv"
        rows = balanced_rows()
        rows[0]["location"] = ""
        write_manifest(manifest_path, rows)

        with pytest.raises(ValueError, match="null values"):
            create_splits(manifest_path=manifest_path, output_dir=test_dir / "out")
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_duplicate_image_id_values_are_rejected() -> None:
    test_dir = make_workspace_test_dir()
    try:
        manifest_path = test_dir / "manifest.csv"
        rows = balanced_rows()
        rows[1]["image_id"] = rows[0]["image_id"]
        write_manifest(manifest_path, rows)

        with pytest.raises(ValueError, match="image_id values must be unique"):
            create_splits(manifest_path=manifest_path, output_dir=test_dir / "out")
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_too_few_locations_are_rejected() -> None:
    test_dir = make_workspace_test_dir()
    try:
        manifest_path = test_dir / "manifest.csv"
        rows = [
            image_row(
                image_id=f"{location}-{category}",
                category=category,
                location=location,
                seq_id=f"{location}-{category}",
            )
            for location in ("loc-a", "loc-b")
            for category in ALLOWED_CATEGORIES
        ]
        write_manifest(manifest_path, rows)

        with pytest.raises(ValueError, match="Too few unique locations"):
            create_splits(manifest_path=manifest_path, output_dir=test_dir / "out")
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_impossible_grouped_split_fails_clearly() -> None:
    test_dir = make_workspace_test_dir()
    try:
        manifest_path = test_dir / "manifest.csv"
        rows: list[dict[str, str]] = []
        for location_index, category in enumerate(ALLOWED_CATEGORIES):
            rows.append(
                image_row(
                    image_id=f"{category}-{location_index}",
                    category=category,
                    location=f"loc-{location_index}",
                    seq_id=f"{category}-{location_index}",
                )
            )
        write_manifest(manifest_path, rows)

        with pytest.raises(ValueError, match="Full category coverage is not feasible"):
            create_splits(
                manifest_path=manifest_path,
                output_dir=test_dir / "out",
                train_ratio=0.34,
                val_ratio=0.33,
                test_ratio=0.33,
                num_restarts=5,
            )
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_output_files_and_summary_json_are_created_correctly() -> None:
    test_dir = make_workspace_test_dir()
    try:
        manifest_path, output_dir, outputs = split_fixture(test_dir)

        for filename in ("train.csv", "val.csv", "test.csv", "split_summary.json"):
            assert (output_dir / filename).exists()

        summary = json.loads(
            (output_dir / "split_summary.json").read_text(encoding="utf-8")
        )
        assert summary["input_manifest_path"] == str(manifest_path)
        assert summary["output_directory"] == str(output_dir)
        assert summary["validation_passed"] is True
        assert summary["total_input_rows"] == len(combined(outputs))
        assert summary["train_val_location_overlap_count"] == 0
        assert summary["train_test_location_overlap_count"] == 0
        assert summary["val_test_location_overlap_count"] == 0
        assert summary["duplicate_image_id_count"] == 0
        assert summary["missing_image_id_count"] == 0
        assert summary["unexpected_image_id_count"] == 0
        assert set(outputs["train"].columns) == {
            *REQUIRED_COLUMNS,
            "extra_column",
            "split",
        }
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)
