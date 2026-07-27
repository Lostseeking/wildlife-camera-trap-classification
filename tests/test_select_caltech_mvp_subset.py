from __future__ import annotations

import csv
import json
import shutil
import uuid
from pathlib import Path

from data.select_caltech_mvp_subset import build_mvp_subset

METADATA_COLUMNS = [
    "image_id",
    "file_name",
    "category_id",
    "category_name",
    "location",
    "date_captured",
    "seq_id",
    "frame_num",
    "seq_num_frames",
    "width",
    "height",
]


def make_workspace_test_dir() -> Path:
    test_dir = Path("data/metadata") / f"test_select_caltech_{uuid.uuid4().hex}"
    test_dir.mkdir(parents=True)
    return test_dir


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def image_row(
    image_id: str, category: str, location: str, seq_id: str
) -> dict[str, str]:
    return {
        "image_id": image_id,
        "file_name": f"{image_id}.jpg",
        "category_id": "1" if category == "deer" else "2",
        "category_name": category,
        "location": location,
        "date_captured": "2021-01-01 00:00:00",
        "seq_id": seq_id,
        "frame_num": "1",
        "seq_num_frames": "2",
        "width": "2048",
        "height": "1494",
    }


def test_build_mvp_subset_excludes_multi_annotations_and_limits_sequences() -> None:
    test_dir = make_workspace_test_dir()
    try:
        metadata_path = test_dir / "metadata.csv"
        multi_annotation_path = test_dir / "multi.csv"
        subset_output = test_dir / "subset.csv"
        summary_output = test_dir / "summary.json"
        write_csv(
            metadata_path,
            [
                image_row("deer-a1", "deer", "loc-a", "seq-a"),
                image_row("deer-a2", "deer", "loc-a", "seq-a"),
                image_row("deer-b1", "deer", "loc-b", "seq-b"),
                image_row("deer-c1", "deer", "loc-c", "seq-c"),
                image_row("fox-a1", "fox", "loc-a", "seq-d"),
                image_row("fox-b1", "fox", "loc-b", "seq-e"),
                image_row("fox-c1", "fox", "loc-c", "seq-f"),
            ],
            METADATA_COLUMNS,
        )
        write_csv(
            multi_annotation_path,
            [
                {
                    "annotation_id": "ann-1",
                    "image_id": "fox-b1",
                    "file_name": "fox-b1.jpg",
                    "category_id": "2",
                    "category_name": "fox",
                }
            ],
            [
                "annotation_id",
                "image_id",
                "file_name",
                "category_id",
                "category_name",
            ],
        )

        summary = build_mvp_subset(
            metadata_path=metadata_path,
            multi_annotation_path=multi_annotation_path,
            subset_output=subset_output,
            summary_output=summary_output,
            categories=["deer", "fox"],
            max_images_per_category=3,
            max_images_per_seq=1,
            seed=7,
        )

        subset_rows = read_csv_rows(subset_output)
        selected_ids = {row["image_id"] for row in subset_rows}
        deer_seq_a_count = sum(1 for row in subset_rows if row["seq_id"] == "seq-a")
        saved_summary = json.loads(summary_output.read_text(encoding="utf-8"))

        assert "fox-b1" not in selected_ids
        assert deer_seq_a_count == 1
        assert summary["total_selected_images"] == 5
        assert saved_summary["selected_category_summary"]["deer"] == {
            "original_available_image_count": 4,
            "selected_image_count": 3,
            "unique_locations": 3,
            "unique_sequences": 3,
        }
        assert saved_summary["selected_category_summary"]["fox"] == {
            "original_available_image_count": 2,
            "selected_image_count": 2,
            "unique_locations": 2,
            "unique_sequences": 2,
        }
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_missing_requested_categories_are_reported() -> None:
    test_dir = make_workspace_test_dir()
    try:
        metadata_path = test_dir / "metadata.csv"
        multi_annotation_path = test_dir / "multi.csv"
        subset_output = test_dir / "subset.csv"
        summary_output = test_dir / "summary.json"
        write_csv(
            metadata_path,
            [image_row("deer-a1", "deer", "loc-a", "seq-a")],
            METADATA_COLUMNS,
        )
        write_csv(
            multi_annotation_path,
            [],
            [
                "annotation_id",
                "image_id",
                "file_name",
                "category_id",
                "category_name",
            ],
        )

        summary = build_mvp_subset(
            metadata_path=metadata_path,
            multi_annotation_path=multi_annotation_path,
            subset_output=subset_output,
            summary_output=summary_output,
            categories=["deer", "bear"],
            max_images_per_category=10,
            max_images_per_seq=1,
            seed=1,
        )

        assert summary["missing_requested_categories"] == ["bear"]
        assert summary["selected_categories"] == ["deer"]
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)
