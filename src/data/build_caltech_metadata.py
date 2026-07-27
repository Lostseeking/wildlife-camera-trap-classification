from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ijson  # type: ignore[import-untyped]

DEFAULT_JSON_PATH = Path(
    r"D:\wildlife\data\raw\caltech_camera_traps\caltech_images_20210113.json"
)
DEFAULT_METADATA_OUTPUT = Path("data/metadata/caltech_metadata.csv")
DEFAULT_MULTI_ANNOTATION_OUTPUT = Path(
    "data/metadata/caltech_multi_annotation_images.csv"
)
DEFAULT_SUMMARY_OUTPUT = Path("data/metadata/caltech_metadata_summary.json")

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
MULTI_ANNOTATION_COLUMNS = [
    "annotation_id",
    "image_id",
    "file_name",
    "category_id",
    "category_name",
]


@dataclass(frozen=True)
class AnnotationRecord:
    annotation_id: str
    category_id: str
    category_name: str


@dataclass
class MetadataBuildResult:
    metadata_output: Path
    multi_annotation_output: Path
    summary_output: Path
    summary: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Caltech Camera Traps metadata CSVs with streaming JSON reads."
        )
    )
    parser.add_argument("--json-path", default=DEFAULT_JSON_PATH, type=Path)
    parser.add_argument(
        "--metadata-output", default=DEFAULT_METADATA_OUTPUT, type=Path
    )
    parser.add_argument(
        "--multi-annotation-output",
        default=DEFAULT_MULTI_ANNOTATION_OUTPUT,
        type=Path,
    )
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY_OUTPUT, type=Path)
    return parser.parse_args()


def validate_json_path(json_path: Path) -> None:
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file does not exist: {json_path}")
    if not json_path.is_file():
        raise ValueError(f"Path is not a file: {json_path}")
    if json_path.suffix.lower() != ".json":
        raise ValueError(f"Expected a .json file, got: {json_path}")


def normalize_value(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def stream_items(json_path: Path, item_prefix: str) -> Any:
    with json_path.open("rb") as file:
        yield from ijson.items(file, item_prefix)


def read_categories(json_path: Path) -> tuple[dict[str, str], int]:
    categories: dict[str, str] = {}
    total_categories = 0

    for category in stream_items(json_path, "categories.item"):
        total_categories += 1
        category_id = normalize_value(category.get("id"))
        categories[category_id] = normalize_value(category.get("name"))

    return categories, total_categories


def read_annotations(
    json_path: Path, categories: dict[str, str]
) -> tuple[dict[str, list[AnnotationRecord]], Counter[str], int]:
    annotations_by_image: dict[str, list[AnnotationRecord]] = defaultdict(list)
    category_counts_by_annotation: Counter[str] = Counter()
    total_annotations = 0

    for annotation in stream_items(json_path, "annotations.item"):
        total_annotations += 1
        image_id = normalize_value(annotation.get("image_id"))
        category_id = normalize_value(annotation.get("category_id"))
        category_name = categories.get(category_id, "")
        annotation_record = AnnotationRecord(
            annotation_id=normalize_value(annotation.get("id")),
            category_id=category_id,
            category_name=category_name,
        )
        annotations_by_image[image_id].append(annotation_record)
        category_counts_by_annotation[category_name or category_id] += 1

    return dict(annotations_by_image), category_counts_by_annotation, total_annotations


def image_metadata_row(
    image: dict[str, object], annotation: AnnotationRecord | None
) -> dict[str, str]:
    return {
        "image_id": normalize_value(image.get("id")),
        "file_name": normalize_value(image.get("file_name")),
        "category_id": "" if annotation is None else annotation.category_id,
        "category_name": "" if annotation is None else annotation.category_name,
        "location": normalize_value(image.get("location")),
        "date_captured": normalize_value(image.get("date_captured")),
        "seq_id": normalize_value(image.get("seq_id")),
        "frame_num": normalize_value(image.get("frame_num")),
        "seq_num_frames": normalize_value(image.get("seq_num_frames")),
        "width": normalize_value(image.get("width")),
        "height": normalize_value(image.get("height")),
    }


def increment_missing_counts(
    missing_values_by_field: Counter[str], row: dict[str, str]
) -> None:
    for field_name, field_value in row.items():
        if field_value == "":
            missing_values_by_field[field_name] += 1


def write_outputs_and_collect_image_stats(
    json_path: Path,
    metadata_output: Path,
    multi_annotation_output: Path,
    annotations_by_image: dict[str, list[AnnotationRecord]],
) -> tuple[dict[str, Any], Counter[str]]:
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    multi_annotation_output.parent.mkdir(parents=True, exist_ok=True)

    total_unique_images = 0
    locations: set[str] = set()
    sequence_ids: set[str] = set()
    category_counts_by_unique_image: Counter[str] = Counter()
    missing_values_by_field: Counter[str] = Counter()
    multi_annotation_image_ids: set[str] = set()

    with (
        metadata_output.open("w", encoding="utf-8", newline="") as metadata_file,
        multi_annotation_output.open(
            "w", encoding="utf-8", newline=""
        ) as multi_annotation_file,
    ):
        metadata_writer = csv.DictWriter(
            metadata_file, fieldnames=METADATA_COLUMNS, extrasaction="ignore"
        )
        multi_annotation_writer = csv.DictWriter(
            multi_annotation_file,
            fieldnames=MULTI_ANNOTATION_COLUMNS,
            extrasaction="ignore",
        )
        metadata_writer.writeheader()
        multi_annotation_writer.writeheader()

        for image in stream_items(json_path, "images.item"):
            total_unique_images += 1
            image_id = normalize_value(image.get("id"))
            image_annotations = annotations_by_image.get(image_id, [])
            location = normalize_value(image.get("location"))
            sequence_id = normalize_value(image.get("seq_id"))

            if location:
                locations.add(location)
            if sequence_id:
                sequence_ids.add(sequence_id)

            unique_categories_for_image = {
                annotation.category_name or annotation.category_id
                for annotation in image_annotations
            }
            for category in unique_categories_for_image:
                category_counts_by_unique_image[category] += 1

            if not image_annotations:
                row = image_metadata_row(image, None)
                increment_missing_counts(missing_values_by_field, row)
                metadata_writer.writerow(row)
                continue

            if len(image_annotations) > 1:
                multi_annotation_image_ids.add(image_id)

            for annotation in image_annotations:
                row = image_metadata_row(image, annotation)
                increment_missing_counts(missing_values_by_field, row)
                metadata_writer.writerow(row)

                if len(image_annotations) > 1:
                    multi_annotation_writer.writerow(
                        {
                            "annotation_id": annotation.annotation_id,
                            "image_id": image_id,
                            "file_name": row["file_name"],
                            "category_id": annotation.category_id,
                            "category_name": annotation.category_name,
                        }
                    )

    image_stats = {
        "total_unique_images": total_unique_images,
        "number_of_unique_locations": len(locations),
        "number_of_unique_sequence_ids": len(sequence_ids),
        "missing_values_per_field": {
            field: missing_values_by_field[field] for field in METADATA_COLUMNS
        },
        "number_of_images_with_multiple_annotations": len(multi_annotation_image_ids),
    }
    return image_stats, category_counts_by_unique_image


def write_summary(summary: dict[str, Any], summary_output: Path) -> None:
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    with summary_output.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")


def build_metadata(
    json_path: Path = DEFAULT_JSON_PATH,
    metadata_output: Path = DEFAULT_METADATA_OUTPUT,
    multi_annotation_output: Path = DEFAULT_MULTI_ANNOTATION_OUTPUT,
    summary_output: Path = DEFAULT_SUMMARY_OUTPUT,
) -> MetadataBuildResult:
    validate_json_path(json_path)

    categories, total_categories = read_categories(json_path)
    (
        annotations_by_image,
        category_counts_by_annotation,
        total_annotations,
    ) = read_annotations(json_path, categories)
    image_stats, category_counts_by_unique_image = (
        write_outputs_and_collect_image_stats(
            json_path=json_path,
            metadata_output=metadata_output,
            multi_annotation_output=multi_annotation_output,
            annotations_by_image=annotations_by_image,
        )
    )

    summary: dict[str, Any] = {
        "source_path": str(json_path),
        "metadata_output": str(metadata_output),
        "multi_annotation_output": str(multi_annotation_output),
        "summary_output": str(summary_output),
        "total_unique_images": image_stats["total_unique_images"],
        "total_annotations": total_annotations,
        "total_categories": total_categories,
        "number_of_unique_locations": image_stats["number_of_unique_locations"],
        "number_of_unique_sequence_ids": image_stats[
            "number_of_unique_sequence_ids"
        ],
        "category_counts_by_annotation": dict(
            sorted(category_counts_by_annotation.items())
        ),
        "category_counts_by_unique_image": dict(
            sorted(category_counts_by_unique_image.items())
        ),
        "missing_values_per_field": image_stats["missing_values_per_field"],
        "number_of_images_with_multiple_annotations": image_stats[
            "number_of_images_with_multiple_annotations"
        ],
    }
    write_summary(summary, summary_output)

    return MetadataBuildResult(
        metadata_output=metadata_output,
        multi_annotation_output=multi_annotation_output,
        summary_output=summary_output,
        summary=summary,
    )


def print_summary(summary: dict[str, Any]) -> None:
    print(f"Source path: {summary['source_path']}")
    print(f"Metadata output: {summary['metadata_output']}")
    print(f"Multi-annotation output: {summary['multi_annotation_output']}")
    print(f"Summary output: {summary['summary_output']}")
    print(f"Total unique images: {summary['total_unique_images']}")
    print(f"Total annotations: {summary['total_annotations']}")
    print(f"Total categories: {summary['total_categories']}")
    print(f"Unique locations: {summary['number_of_unique_locations']}")
    print(f"Unique sequence IDs: {summary['number_of_unique_sequence_ids']}")
    print(
        "Images with multiple annotations: "
        f"{summary['number_of_images_with_multiple_annotations']}"
    )


def main() -> None:
    args = parse_args()
    result = build_metadata(
        json_path=args.json_path,
        metadata_output=args.metadata_output,
        multi_annotation_output=args.multi_annotation_output,
        summary_output=args.summary_output,
    )
    print_summary(result.summary)


if __name__ == "__main__":
    main()
