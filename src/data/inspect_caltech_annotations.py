from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import ijson  # type: ignore[import-untyped]

PREFIX_BYTES = 256
RECORD_PREFIXES = {
    "images": "images.item",
    "annotations": "annotations.item",
    "categories": "categories.item",
}
METADATA_FIELD_CANDIDATES = (
    "id",
    "file_name",
    "location",
    "location_id",
    "camera_id",
    "datetime",
    "width",
    "height",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect Caltech Camera Traps annotations without loading all JSON."
    )
    parser.add_argument("--json-path", required=True, type=Path)
    parser.add_argument("--sample-size", default=5, type=int)
    parser.add_argument("--output-path", type=Path)
    return parser.parse_args()


def validate_json_path(json_path: Path) -> int:
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file does not exist: {json_path}")
    if not json_path.is_file():
        raise ValueError(f"Path is not a file: {json_path}")
    if json_path.suffix.lower() != ".json":
        raise ValueError(f"Expected a .json file, got: {json_path}")
    return json_path.stat().st_size


def read_prefix_summary(json_path: Path) -> dict[str, str]:
    with json_path.open("rb") as file:
        prefix = file.read(PREFIX_BYTES)

    stripped = prefix.lstrip()
    first_char = stripped[:1].decode("utf-8", errors="replace")
    if first_char == "{":
        root_type = "object"
    elif first_char == "[":
        root_type = "array"
    else:
        root_type = "unknown"

    preview = stripped[:80].decode("utf-8", errors="replace")
    return {
        "first_non_whitespace_characters": preview,
        "appears_to_begin_with": root_type,
    }


def stream_records(
    json_path: Path, item_prefix: str, sample_size: int
) -> tuple[int, list[Any], list[str]]:
    count = 0
    samples: list[Any] = []
    field_names: set[str] = set()

    with json_path.open("rb") as file:
        for record in ijson.items(file, item_prefix):
            count += 1
            if len(samples) < sample_size:
                samples.append(record)
                if isinstance(record, dict):
                    field_names.update(str(field) for field in record)

    return count, samples, sorted(field_names)


def inspect_annotations(json_path: Path, sample_size: int) -> dict[str, Any]:
    if sample_size < 0:
        raise ValueError("sample-size must be greater than or equal to 0")

    file_size_bytes = validate_json_path(json_path)
    samples: dict[str, list[Any]] = {}
    detected_fields: dict[str, list[str]] = {}
    counts: dict[str, int] = {}

    for record_type, item_prefix in RECORD_PREFIXES.items():
        count, record_samples, field_names = stream_records(
            json_path=json_path,
            item_prefix=item_prefix,
            sample_size=sample_size,
        )
        counts[record_type] = count
        samples[record_type] = record_samples
        detected_fields[record_type] = field_names

    image_fields = set(detected_fields["images"])
    detected_metadata_fields = [
        field for field in METADATA_FIELD_CANDIDATES if field in image_fields
    ]

    return {
        "source_path": str(json_path),
        "file_size_bytes": file_size_bytes,
        "file_size_mb": round(file_size_bytes / (1024 * 1024), 2),
        "prefix": read_prefix_summary(json_path),
        "total_record_counts": counts,
        "sampled_records": samples,
        "detected_fields": detected_fields,
        "detected_image_metadata_fields": detected_metadata_fields,
    }


def write_summary(summary: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")


def print_summary(summary: dict[str, Any]) -> None:
    print(f"Source path: {summary['source_path']}")
    print(f"File size: {summary['file_size_mb']} MB")
    print("Prefix:")
    print(
        "  first non-whitespace characters: "
        f"{summary['prefix']['first_non_whitespace_characters']}"
    )
    print(f"  appears to begin with: {summary['prefix']['appears_to_begin_with']}")
    print("Record counts:")
    for record_type, count in summary["total_record_counts"].items():
        print(f"  {record_type}: {count}")
    print("Detected fields:")
    for record_type, fields in summary["detected_fields"].items():
        print(f"  {record_type}: {fields}")
    print(
        f"Detected image metadata fields: {summary['detected_image_metadata_fields']}"
    )
    print("Sample sizes:")
    for record_type, records in summary["sampled_records"].items():
        print(f"  {record_type}: {len(records)}")


def main() -> None:
    args = parse_args()
    summary = inspect_annotations(args.json_path, args.sample_size)
    print_summary(summary)

    if args.output_path is not None:
        write_summary(summary, args.output_path)
        print(f"Summary written to: {args.output_path}")


if __name__ == "__main__":
    main()
