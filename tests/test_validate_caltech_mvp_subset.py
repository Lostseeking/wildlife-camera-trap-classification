from __future__ import annotations

import csv
import json
import shutil
import uuid
from pathlib import Path
from typing import Any

from data.validate_caltech_mvp_subset import (
    DEFAULT_EXPECTED_CATEGORIES,
    REQUIRED_COLUMNS,
    validate_subset,
)

MULTI_ANNOTATION_COLUMNS = [
    "annotation_id",
    "image_id",
    "file_name",
    "category_id",
    "category_name",
]


def make_workspace_test_dir() -> Path:
    test_dir = Path("data/metadata") / f"test_validate_caltech_{uuid.uuid4().hex}"
    test_dir.mkdir(parents=True)
    return test_dir


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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
    }


def summary_for_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    categories = sorted({row["category_name"] for row in rows})
    selected_category_summary = {}
    for category in categories:
        category_rows = [row for row in rows if row["category_name"] == category]
        selected_category_summary[category] = {
            "selected_image_count": len(category_rows),
            "unique_locations": len({row["location"] for row in category_rows}),
            "unique_sequences": len({row["seq_id"] for row in category_rows}),
            "original_available_image_count": len(category_rows),
        }

    return {
        "selected_categories": categories,
        "selected_category_summary": selected_category_summary,
        "total_selected_images": len(rows),
    }


def six_category_rows() -> list[dict[str, str]]:
    return [
        image_row(f"{category}-1", category, f"loc-{index}", f"seq-{index}")
        for index, category in enumerate(DEFAULT_EXPECTED_CATEGORIES)
    ]


def write_fixture(
    test_dir: Path,
    rows: list[dict[str, str]],
    summary: dict[str, Any] | None = None,
    multi_rows: list[dict[str, str]] | None = None,
) -> tuple[Path, Path, Path]:
    manifest_path = test_dir / "subset.csv"
    summary_path = test_dir / "summary.json"
    multi_annotation_path = test_dir / "multi.csv"
    write_csv(manifest_path, rows, REQUIRED_COLUMNS)
    write_json(summary_path, summary if summary is not None else summary_for_rows(rows))
    write_csv(multi_annotation_path, multi_rows or [], MULTI_ANNOTATION_COLUMNS)
    return manifest_path, summary_path, multi_annotation_path


def assert_has_error(errors: list[str], expected_text: str) -> None:
    assert any(expected_text in error for error in errors)


def test_correct_six_category_manifest_passes() -> None:
    test_dir = make_workspace_test_dir()
    try:
        report = validate_subset(*write_fixture(test_dir, six_category_rows()))

        assert report.is_valid
        assert report.total_selected_images == 6
        assert report.categories == sorted(DEFAULT_EXPECTED_CATEGORIES)
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_unexpected_category_fails() -> None:
    test_dir = make_workspace_test_dir()
    try:
        rows = six_category_rows() + [image_row("fox-1", "fox", "loc-x", "seq-x")]
        report = validate_subset(*write_fixture(test_dir, rows))

        assert not report.is_valid
        assert_has_error(report.errors, "manifest categories do not match expected")
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_duplicate_image_ids_fail() -> None:
    test_dir = make_workspace_test_dir()
    try:
        rows = six_category_rows()
        rows[1]["image_id"] = rows[0]["image_id"]
        report = validate_subset(*write_fixture(test_dir, rows))

        assert not report.is_valid
        assert_has_error(report.errors, "duplicate image_id count is 1")
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_multi_annotation_overlap_fails() -> None:
    test_dir = make_workspace_test_dir()
    try:
        rows = six_category_rows()
        multi_rows = [
            {
                "annotation_id": "ann-1",
                "image_id": rows[0]["image_id"],
                "file_name": rows[0]["file_name"],
                "category_id": "1",
                "category_name": rows[0]["category_name"],
            }
        ]
        report = validate_subset(*write_fixture(test_dir, rows, multi_rows=multi_rows))

        assert not report.is_valid
        assert_has_error(report.errors, "manifest overlaps multi-annotation")
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_repeated_sequence_selection_fails_when_maximum_is_one() -> None:
    test_dir = make_workspace_test_dir()
    try:
        rows = six_category_rows()
        rows.append(image_row("empty-2", "empty", "loc-extra", rows[0]["seq_id"]))
        report = validate_subset(*write_fixture(test_dir, rows))

        assert not report.is_valid
        assert_has_error(report.errors, "repeated (category_name, seq_id)")
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_inconsistent_summary_statistics_fail() -> None:
    test_dir = make_workspace_test_dir()
    try:
        rows = six_category_rows()
        summary = summary_for_rows(rows)
        summary["total_selected_images"] = 999
        summary["selected_category_summary"]["empty"]["selected_image_count"] = 999
        report = validate_subset(*write_fixture(test_dir, rows, summary=summary))

        assert not report.is_valid
        assert_has_error(report.errors, "summary total_selected_images")
        assert_has_error(report.errors, "summary empty.selected_image_count")
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)
