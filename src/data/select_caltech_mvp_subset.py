from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_METADATA_PATH = Path("data/metadata/caltech_metadata.csv")
DEFAULT_MULTI_ANNOTATION_PATH = Path(
    "data/metadata/caltech_multi_annotation_images.csv"
)
DEFAULT_SUBSET_OUTPUT = Path("data/metadata/caltech_mvp_subset.csv")
DEFAULT_SUMMARY_OUTPUT = Path("data/metadata/caltech_mvp_subset_summary.json")

SUBSET_COLUMNS = [
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


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    file_name: str
    category_id: str
    category_name: str
    location: str
    date_captured: str
    seq_id: str
    frame_num: str
    seq_num_frames: str
    width: str
    height: str

    @classmethod
    def from_row(cls, row: dict[str, str]) -> ImageRecord:
        return cls(
            image_id=row["image_id"],
            file_name=row["file_name"],
            category_id=row["category_id"],
            category_name=row["category_name"],
            location=row["location"],
            date_captured=row["date_captured"],
            seq_id=row["seq_id"],
            frame_num=row["frame_num"],
            seq_num_frames=row["seq_num_frames"],
            width=row["width"],
            height=row["height"],
        )

    def to_row(self) -> dict[str, str]:
        return {
            "image_id": self.image_id,
            "file_name": self.file_name,
            "category_id": self.category_id,
            "category_name": self.category_name,
            "location": self.location,
            "date_captured": self.date_captured,
            "seq_id": self.seq_id,
            "frame_num": self.frame_num,
            "seq_num_frames": self.seq_num_frames,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class CategoryDistribution:
    image_count: int
    unique_locations: int
    unique_sequences: int

    def to_dict(self) -> dict[str, int]:
        return {
            "image_count": self.image_count,
            "unique_locations": self.unique_locations,
            "unique_sequences": self.unique_sequences,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a reproducible Caltech Camera Traps MVP subset."
    )
    parser.add_argument("--metadata-path", default=DEFAULT_METADATA_PATH, type=Path)
    parser.add_argument(
        "--multi-annotation-path",
        default=DEFAULT_MULTI_ANNOTATION_PATH,
        type=Path,
    )
    parser.add_argument("--subset-output", default=DEFAULT_SUBSET_OUTPUT, type=Path)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY_OUTPUT, type=Path)
    parser.add_argument(
        "--categories",
        nargs="*",
        help="Category names to include. Defaults to all categories in metadata.",
    )
    parser.add_argument("--max-images-per-category", default=100, type=int)
    parser.add_argument("--max-images-per-seq", default=1, type=int)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def validate_input_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Input path is not a file: {path}")


def read_multi_annotation_image_ids(path: Path) -> set[str]:
    validate_input_path(path)
    image_ids: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            image_id = row.get("image_id", "")
            if image_id:
                image_ids.add(image_id)
    return image_ids


def read_metadata_records(
    metadata_path: Path, excluded_image_ids: set[str]
) -> tuple[list[ImageRecord], dict[str, CategoryDistribution]]:
    validate_input_path(metadata_path)
    records: list[ImageRecord] = []
    category_image_counts: Counter[str] = Counter()
    category_locations: dict[str, set[str]] = defaultdict(set)
    category_sequences: dict[str, set[str]] = defaultdict(set)
    seen_image_ids: set[str] = set()

    with metadata_path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            record = ImageRecord.from_row(row)
            category = record.category_name
            category_image_counts[category] += 1
            if record.location:
                category_locations[category].add(record.location)
            if record.seq_id:
                category_sequences[category].add(record.seq_id)

            if (
                record.image_id in excluded_image_ids
                or record.image_id in seen_image_ids
            ):
                continue

            seen_image_ids.add(record.image_id)
            records.append(record)

    distribution = {
        category: CategoryDistribution(
            image_count=count,
            unique_locations=len(category_locations[category]),
            unique_sequences=len(category_sequences[category]),
        )
        for category, count in sorted(category_image_counts.items())
    }
    return records, distribution


def distribution_for_records(
    records_by_category: dict[str, list[ImageRecord]],
) -> dict[str, CategoryDistribution]:
    distribution: dict[str, CategoryDistribution] = {}
    for category, records in sorted(records_by_category.items()):
        distribution[category] = CategoryDistribution(
            image_count=len(records),
            unique_locations=len({record.location for record in records}),
            unique_sequences=len({record.seq_id for record in records}),
        )
    return distribution


def group_records_by_category(
    records: list[ImageRecord],
) -> dict[str, list[ImageRecord]]:
    records_by_category: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        records_by_category[record.category_name].append(record)
    return dict(records_by_category)


def choose_from_sequence(
    sequence_records: list[ImageRecord], max_images_per_seq: int, rng: random.Random
) -> list[ImageRecord]:
    shuffled_records = sorted(sequence_records, key=lambda record: record.image_id)
    rng.shuffle(shuffled_records)
    return shuffled_records[:max_images_per_seq]


def sample_category_records(
    records: list[ImageRecord],
    max_images: int,
    max_images_per_seq: int,
    rng: random.Random,
) -> list[ImageRecord]:
    records_by_location_seq: dict[str, dict[str, list[ImageRecord]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        records_by_location_seq[record.location][record.seq_id].append(record)

    location_names = sorted(records_by_location_seq)
    rng.shuffle(location_names)
    sequence_names_by_location: dict[str, list[str]] = {}
    for location in location_names:
        sequence_names = sorted(records_by_location_seq[location])
        rng.shuffle(sequence_names)
        sequence_names_by_location[location] = sequence_names

    selected: list[ImageRecord] = []
    while len(selected) < max_images:
        selected_this_round = False
        for location in location_names:
            sequence_names = sequence_names_by_location[location]
            if not sequence_names:
                continue

            seq_id = sequence_names.pop(0)
            sequence_records = records_by_location_seq[location][seq_id]
            selected.extend(
                choose_from_sequence(
                    sequence_records=sequence_records,
                    max_images_per_seq=max_images_per_seq,
                    rng=rng,
                )
            )
            selected_this_round = True

            if len(selected) >= max_images:
                return sorted(selected[:max_images], key=lambda record: record.image_id)

        if not selected_this_round:
            break

    return sorted(selected, key=lambda record: record.image_id)


def select_subset(
    records: list[ImageRecord],
    categories: list[str] | None,
    max_images_per_category: int,
    max_images_per_seq: int,
    seed: int,
) -> tuple[list[ImageRecord], dict[str, dict[str, int]], list[str]]:
    if max_images_per_category < 1:
        raise ValueError("max-images-per-category must be at least 1")
    if max_images_per_seq < 1:
        raise ValueError("max-images-per-seq must be at least 1")

    records_by_category = group_records_by_category(records)
    selected_categories = categories or sorted(records_by_category)
    missing_categories = [
        category
        for category in selected_categories
        if category not in records_by_category
    ]
    rng = random.Random(seed)
    selected_records: list[ImageRecord] = []
    selected_summary: dict[str, dict[str, int]] = {}

    for category in selected_categories:
        category_records = records_by_category.get(category, [])
        if not category_records:
            continue

        category_selection = sample_category_records(
            records=category_records,
            max_images=max_images_per_category,
            max_images_per_seq=max_images_per_seq,
            rng=rng,
        )
        selected_records.extend(category_selection)
        selected_summary[category] = {
            "selected_image_count": len(category_selection),
            "unique_locations": len({record.location for record in category_selection}),
            "unique_sequences": len({record.seq_id for record in category_selection}),
            "original_available_image_count": len(category_records),
        }

    return (
        sorted(
            selected_records,
            key=lambda record: (record.category_name, record.image_id),
        ),
        selected_summary,
        missing_categories,
    )


def write_subset(records: list[ImageRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUBSET_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_row())


def write_summary(summary: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")


def print_category_distribution(
    title: str, distribution: dict[str, CategoryDistribution]
) -> None:
    print(title)
    for category, stats in distribution.items():
        print(
            f"  {category}: {stats.image_count} images, "
            f"{stats.unique_locations} locations, "
            f"{stats.unique_sequences} sequences"
        )


def build_mvp_subset(
    metadata_path: Path = DEFAULT_METADATA_PATH,
    multi_annotation_path: Path = DEFAULT_MULTI_ANNOTATION_PATH,
    subset_output: Path = DEFAULT_SUBSET_OUTPUT,
    summary_output: Path = DEFAULT_SUMMARY_OUTPUT,
    categories: list[str] | None = None,
    max_images_per_category: int = 100,
    max_images_per_seq: int = 1,
    seed: int = 42,
) -> dict[str, Any]:
    excluded_image_ids = read_multi_annotation_image_ids(multi_annotation_path)
    records, original_distribution = read_metadata_records(
        metadata_path=metadata_path,
        excluded_image_ids=excluded_image_ids,
    )
    available_records_by_category = group_records_by_category(records)
    available_distribution = distribution_for_records(available_records_by_category)
    selected_records, selected_summary, missing_categories = select_subset(
        records=records,
        categories=categories,
        max_images_per_category=max_images_per_category,
        max_images_per_seq=max_images_per_seq,
        seed=seed,
    )

    write_subset(selected_records, subset_output)
    summary: dict[str, Any] = {
        "metadata_path": str(metadata_path),
        "multi_annotation_path": str(multi_annotation_path),
        "subset_output": str(subset_output),
        "summary_output": str(summary_output),
        "seed": seed,
        "max_images_per_category": max_images_per_category,
        "max_images_per_seq": max_images_per_seq,
        "requested_categories": categories,
        "selected_categories": sorted(selected_summary),
        "missing_requested_categories": missing_categories,
        "excluded_multi_annotation_image_count": len(excluded_image_ids),
        "category_distribution_original": {
            category: stats.to_dict()
            for category, stats in original_distribution.items()
        },
        "category_distribution_available_after_exclusions": {
            category: stats.to_dict()
            for category, stats in available_distribution.items()
        },
        "selected_category_summary": selected_summary,
        "total_selected_images": len(selected_records),
    }
    write_summary(summary, summary_output)
    return summary


def main() -> None:
    args = parse_args()
    summary = build_mvp_subset(
        metadata_path=args.metadata_path,
        multi_annotation_path=args.multi_annotation_path,
        subset_output=args.subset_output,
        summary_output=args.summary_output,
        categories=args.categories,
        max_images_per_category=args.max_images_per_category,
        max_images_per_seq=args.max_images_per_seq,
        seed=args.seed,
    )

    original_distribution = {
        category: CategoryDistribution(**stats)
        for category, stats in summary["category_distribution_original"].items()
    }
    available_distribution = {
        category: CategoryDistribution(**stats)
        for category, stats in summary[
            "category_distribution_available_after_exclusions"
        ].items()
    }
    print_category_distribution(
        "Original category distribution:", original_distribution
    )
    print_category_distribution(
        "Available category distribution after multi-annotation exclusions:",
        available_distribution,
    )
    print(f"Selected images: {summary['total_selected_images']}")
    print(f"Subset output: {summary['subset_output']}")
    print(f"Summary output: {summary['summary_output']}")


if __name__ == "__main__":
    main()
