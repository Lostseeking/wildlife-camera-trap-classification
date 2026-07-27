from __future__ import annotations

import csv
import json
import shutil
import uuid
from pathlib import Path

from data.build_caltech_metadata import build_metadata


def write_coco_json(path: Path) -> None:
    payload = {
        "images": [
            {
                "id": "img001",
                "file_name": "loc1/image001.jpg",
                "location": "loc1",
                "date_captured": "2021-01-01T00:00:00",
                "seq_id": "seq-a",
                "frame_num": 1,
                "seq_num_frames": 2,
                "width": 1920,
                "height": 1080,
            },
            {
                "id": "img002",
                "file_name": "loc1/image002.jpg",
                "location": "loc1",
                "date_captured": "2021-01-01T00:00:01",
                "seq_id": "seq-a",
                "frame_num": 2,
                "seq_num_frames": 2,
                "width": 1920,
                "height": 1080,
            },
            {
                "id": "img003",
                "file_name": "loc2/image003.jpg",
                "location": "loc2",
                "date_captured": "2021-01-02T00:00:00",
                "seq_id": "seq-b",
                "frame_num": 1,
                "seq_num_frames": 1,
                "width": 1280,
                "height": 720,
            },
            {
                "id": "img004",
                "file_name": "loc3/image004.jpg",
            },
        ],
        "annotations": [
            {"id": "ann001", "image_id": "img001", "category_id": 1},
            {"id": "ann002", "image_id": "img002", "category_id": 1},
            {"id": "ann003", "image_id": "img002", "category_id": 2},
            {"id": "ann004", "image_id": "img003", "category_id": 2},
        ],
        "categories": [
            {"id": 1, "name": "deer"},
            {"id": 2, "name": "fox"},
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def make_workspace_test_dir() -> Path:
    test_dir = Path("data/metadata") / f"test_build_caltech_{uuid.uuid4().hex}"
    test_dir.mkdir(parents=True)
    return test_dir


def test_build_metadata_writes_joined_outputs() -> None:
    test_dir = make_workspace_test_dir()
    try:
        json_path = test_dir / "annotations.json"
        metadata_output = test_dir / "caltech_metadata.csv"
        multi_annotation_output = test_dir / "caltech_multi_annotation_images.csv"
        summary_output = test_dir / "caltech_metadata_summary.json"
        write_coco_json(json_path)

        build_metadata(
            json_path=json_path,
            metadata_output=metadata_output,
            multi_annotation_output=multi_annotation_output,
            summary_output=summary_output,
        )

        metadata_rows = read_csv_rows(metadata_output)
        multi_annotation_rows = read_csv_rows(multi_annotation_output)
        summary = json.loads(summary_output.read_text(encoding="utf-8"))

        assert len(metadata_rows) == 5
        assert metadata_rows[0]["image_id"] == "img001"
        assert metadata_rows[0]["category_name"] == "deer"
        assert metadata_rows[2]["image_id"] == "img002"
        assert metadata_rows[2]["category_name"] == "fox"
        assert metadata_rows[4]["image_id"] == "img004"
        assert metadata_rows[4]["category_id"] == ""
        assert len(multi_annotation_rows) == 2
        assert {row["annotation_id"] for row in multi_annotation_rows} == {
            "ann002",
            "ann003",
        }
        assert summary["total_unique_images"] == 4
        assert summary["total_annotations"] == 4
        assert summary["total_categories"] == 2
        assert summary["number_of_unique_locations"] == 2
        assert summary["number_of_unique_sequence_ids"] == 2
        assert summary["number_of_images_with_multiple_annotations"] == 1
        assert summary["category_counts_by_annotation"] == {"deer": 2, "fox": 2}
        assert summary["category_counts_by_unique_image"] == {"deer": 2, "fox": 2}
        assert summary["missing_values_per_field"]["category_id"] == 1
        assert summary["missing_values_per_field"]["location"] == 1
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)
