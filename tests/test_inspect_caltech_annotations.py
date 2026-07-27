from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pytest

from data.inspect_caltech_annotations import (
    inspect_annotations,
    validate_json_path,
    write_summary,
)


def write_coco_json(path: Path) -> None:
    payload = {
        "images": [
            {
                "id": "img001",
                "file_name": "loc1/image001.jpg",
                "location": "loc1",
                "datetime": "2021-01-01T00:00:00",
                "width": 1920,
                "height": 1080,
            },
            {
                "id": "img002",
                "file_name": "loc1/image002.jpg",
                "camera_id": "cam-a",
            },
        ],
        "annotations": [
            {"id": "ann001", "image_id": "img001", "category_id": 1},
            {"id": "ann002", "image_id": "img002", "category_id": 2},
            {"id": "ann003", "image_id": "img002", "category_id": 2},
        ],
        "categories": [
            {"id": 1, "name": "deer"},
            {"id": 2, "name": "fox"},
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_workspace_test_dir() -> Path:
    test_dir = Path("data/metadata") / f"test_inspect_caltech_{uuid.uuid4().hex}"
    test_dir.mkdir(parents=True)
    return test_dir


def test_missing_file_handling() -> None:
    test_dir = make_workspace_test_dir()
    missing_path = test_dir / "missing.json"

    try:
        with pytest.raises(FileNotFoundError):
            validate_json_path(missing_path)
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_record_counts_and_sampling() -> None:
    test_dir = make_workspace_test_dir()
    json_path = test_dir / "annotations.json"

    try:
        write_coco_json(json_path)
        summary = inspect_annotations(json_path, sample_size=1)

        assert summary["total_record_counts"] == {
            "images": 2,
            "annotations": 3,
            "categories": 2,
        }
        assert len(summary["sampled_records"]["images"]) == 1
        assert len(summary["sampled_records"]["annotations"]) == 1
        assert len(summary["sampled_records"]["categories"]) == 1
        assert summary["sampled_records"]["images"][0]["id"] == "img001"
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_detected_fields_from_sampled_records() -> None:
    test_dir = make_workspace_test_dir()
    json_path = test_dir / "annotations.json"

    try:
        write_coco_json(json_path)
        summary = inspect_annotations(json_path, sample_size=2)

        assert "file_name" in summary["detected_fields"]["images"]
        assert "datetime" in summary["detected_image_metadata_fields"]
        assert "camera_id" in summary["detected_image_metadata_fields"]
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_optional_summary_output() -> None:
    test_dir = make_workspace_test_dir()
    json_path = test_dir / "annotations.json"
    output_path = test_dir / "summary.json"

    try:
        write_coco_json(json_path)
        summary = inspect_annotations(json_path, sample_size=1)
        write_summary(summary, output_path)

        saved_summary = json.loads(output_path.read_text(encoding="utf-8"))
        assert saved_summary["source_path"] == str(json_path)
        assert saved_summary["total_record_counts"]["images"] == 2
        assert len(saved_summary["sampled_records"]["categories"]) == 1
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)
