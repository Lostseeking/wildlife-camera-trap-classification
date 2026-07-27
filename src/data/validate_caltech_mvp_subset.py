from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MANIFEST_PATH = Path("data/metadata/caltech_mvp_subset.csv")
DEFAULT_SUMMARY_PATH = Path("data/metadata/caltech_mvp_subset_summary.json")
DEFAULT_MULTI_ANNOTATION_PATH = Path(
    "data/metadata/caltech_multi_annotation_images.csv"
)
DEFAULT_EXPECTED_CATEGORIES = [
    "empty",
    "deer",
    "coyote",
    "bobcat",
    "bird",
    "opossum",
]
REQUIRED_COLUMNS = [
    "image_id",
    "file_name",
    "category_name",
    "location",
    "seq_id",
]


@dataclass(frozen=True)
class CategoryStats:
    image_count: int
    unique_locations: int
    unique_sequences: int


@dataclass(frozen=True)
class ValidationReport:
    total_selected_images: int
    categories: list[str]
    category_stats: dict[str, CategoryStats]
    duplicate_image_id_count: int
    duplicate_category_seq_count: int
    multi_annotation_overlap_count: int
    errors: list[str]

    @property
    def is_valid(self) -> bool:
        return not self.errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the Caltech Camera Traps MVP subset manifest."
    )
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST_PATH, type=Path)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH, type=Path)
    parser.add_argument(
        "--multi-annotation-path",
        default=DEFAULT_MULTI_ANNOTATION_PATH,
        type=Path,
    )
    parser.add_argument(
        "--expected-categories",
        nargs="*",
        default=DEFAULT_EXPECTED_CATEGORIES,
        help="Expected category names. Defaults to the Caltech MVP category list.",
    )
    parser.add_argument("--max-images-per-category", default=1000, type=int)
    parser.add_argument("--max-images-per-seq", default=1, type=int)
    return parser.parse_args()


def read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Input path is not a file: {path}")

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader), list(reader.fieldnames or [])


def read_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Summary file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Summary path is not a file: {path}")

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Summary JSON must contain an object: {path}")
    return payload


def read_multi_annotation_image_ids(path: Path) -> set[str]:
    rows, _fieldnames = read_csv_rows(path)
    return {row["image_id"] for row in rows if row.get("image_id")}


def category_stats_from_rows(
    rows: list[dict[str, str]],
) -> dict[str, CategoryStats]:
    categories = sorted({row["category_name"] for row in rows})
    return {
        category: CategoryStats(
            image_count=sum(1 for row in rows if row["category_name"] == category),
            unique_locations=len(
                {
                    row["location"]
                    for row in rows
                    if row["category_name"] == category
                }
            ),
            unique_sequences=len(
                {row["seq_id"] for row in rows if row["category_name"] == category}
            ),
        )
        for category in categories
    }


def validate_summary_consistency(
    summary: dict[str, Any],
    category_stats: dict[str, CategoryStats],
    total_selected_images: int,
) -> list[str]:
    errors: list[str] = []
    categories = sorted(category_stats)

    if summary.get("total_selected_images") != total_selected_images:
        errors.append(
            "summary total_selected_images does not match manifest row count: "
            f"{summary.get('total_selected_images')} != {total_selected_images}"
        )

    if summary.get("selected_categories") != categories:
        errors.append(
            "summary selected_categories does not match manifest categories: "
            f"{summary.get('selected_categories')} != {categories}"
        )

    selected_summary = summary.get("selected_category_summary")
    if not isinstance(selected_summary, dict):
        errors.append("summary selected_category_summary is missing or invalid")
        return errors

    for category, stats in category_stats.items():
        raw_category_summary = selected_summary.get(category)
        if not isinstance(raw_category_summary, dict):
            errors.append(f"summary is missing selected stats for {category}")
            continue

        expected_values = {
            "selected_image_count": stats.image_count,
            "unique_locations": stats.unique_locations,
            "unique_sequences": stats.unique_sequences,
        }
        for key, expected_value in expected_values.items():
            if raw_category_summary.get(key) != expected_value:
                errors.append(
                    f"summary {category}.{key} does not match manifest: "
                    f"{raw_category_summary.get(key)} != {expected_value}"
                )

    unexpected_summary_categories = sorted(set(selected_summary) - set(categories))
    if unexpected_summary_categories:
        errors.append(
            "summary includes categories absent from manifest: "
            f"{unexpected_summary_categories}"
        )

    return errors


def validate_subset(
    manifest_path: Path,
    summary_path: Path,
    multi_annotation_path: Path,
    expected_categories: list[str] | None = None,
    max_images_per_category: int = 1000,
    max_images_per_seq: int = 1,
) -> ValidationReport:
    rows, fieldnames = read_csv_rows(manifest_path)
    summary = read_summary(summary_path)
    multi_annotation_image_ids = read_multi_annotation_image_ids(multi_annotation_path)

    errors: list[str] = []
    expected = sorted(expected_categories or DEFAULT_EXPECTED_CATEGORIES)
    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in fieldnames
    ]
    if missing_columns:
        errors.append(f"manifest is missing required columns: {missing_columns}")

    category_stats = category_stats_from_rows(rows)
    categories = sorted(category_stats)
    if categories != expected:
        errors.append(
            "manifest categories do not match expected: "
            f"{categories} != {expected}"
        )

    over_limit_categories = [
        category
        for category, stats in category_stats.items()
        if stats.image_count > max_images_per_category
    ]
    if over_limit_categories:
        errors.append(
            "categories exceed max-images-per-category "
            f"{max_images_per_category}: {over_limit_categories}"
        )

    image_ids = [row["image_id"] for row in rows]
    duplicate_image_id_count = len(image_ids) - len(set(image_ids))
    if duplicate_image_id_count:
        errors.append(f"duplicate image_id count is {duplicate_image_id_count}")

    category_seq_pairs = [
        (row["category_name"], row["seq_id"]) for row in rows if row.get("seq_id")
    ]
    category_seq_counts = Counter(category_seq_pairs)
    duplicate_category_seq_count = sum(
        count - max_images_per_seq
        for count in category_seq_counts.values()
        if count > max_images_per_seq
    )
    if duplicate_category_seq_count:
        errors.append(
            "repeated (category_name, seq_id) selections above "
            f"max-images-per-seq {max_images_per_seq}: "
            f"{duplicate_category_seq_count}"
        )

    multi_annotation_overlap_count = len(set(image_ids) & multi_annotation_image_ids)
    if multi_annotation_overlap_count:
        errors.append(
            "manifest overlaps multi-annotation image report: "
            f"{multi_annotation_overlap_count}"
        )

    errors.extend(
        validate_summary_consistency(
            summary=summary,
            category_stats=category_stats,
            total_selected_images=len(rows),
        )
    )

    return ValidationReport(
        total_selected_images=len(rows),
        categories=categories,
        category_stats=category_stats,
        duplicate_image_id_count=duplicate_image_id_count,
        duplicate_category_seq_count=duplicate_category_seq_count,
        multi_annotation_overlap_count=multi_annotation_overlap_count,
        errors=errors,
    )


def print_validation_report(report: ValidationReport) -> None:
    print(f"Selected images: {report.total_selected_images}")
    print(
        f"Selected categories ({len(report.categories)}): "
        f"{', '.join(report.categories)}"
    )
    print("Category statistics:")
    for category, stats in report.category_stats.items():
        print(
            f"  {category}: {stats.image_count} images, "
            f"{stats.unique_locations} locations, "
            f"{stats.unique_sequences} sequences"
        )
    print(f"Duplicate image IDs: {report.duplicate_image_id_count}")
    print(
        "Duplicate (category_name, seq_id) selections: "
        f"{report.duplicate_category_seq_count}"
    )
    print(f"Multi-annotation overlap: {report.multi_annotation_overlap_count}")

    if report.errors:
        print("Validation failed:")
        for error in report.errors:
            print(f"  - {error}")
        return

    print("Validation passed.")


def main() -> None:
    args = parse_args()
    report = validate_subset(
        manifest_path=args.manifest_path,
        summary_path=args.summary_path,
        multi_annotation_path=args.multi_annotation_path,
        expected_categories=args.expected_categories,
        max_images_per_category=args.max_images_per_category,
        max_images_per_seq=args.max_images_per_seq,
    )
    print_validation_report(report)
    if not report.is_valid:
        sys.exit(1)


if __name__ == "__main__":
    main()
