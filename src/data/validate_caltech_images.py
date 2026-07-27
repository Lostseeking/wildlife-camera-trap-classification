from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.error import HTTPError, URLError

from PIL import Image, UnidentifiedImageError

from .caltech_dataloaders import (
    DEFAULT_IMAGE_ROOT,
    DEFAULT_TEST_CSV,
    DEFAULT_TRAIN_CSV,
    DEFAULT_VAL_CSV,
)
from .download_caltech_subset import (
    DEFAULT_BASE_URL,
    USER_AGENT,
    download_to_temp_file,
    error_http_status,
    is_retryable_exception,
    retry_delay_seconds,
    safe_destination_path,
    source_url_for_file,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_REPORT_PATH = Path("artifacts/data_checks/caltech_image_integrity.json")
REQUIRED_COLUMNS = ("image_id", "file_name")
ImageStatus = Literal["valid", "missing", "invalid", "repaired", "repair_failed"]
RepairStatus = Literal[
    "not_needed",
    "not_requested",
    "repaired",
    "failed",
]


@dataclass(frozen=True)
class SplitReference:
    split: str
    csv_path: str
    csv_row_number: int
    image_id: str


@dataclass(frozen=True)
class ImageReference:
    file_name: str
    local_path: Path
    source_url: str
    references: list[SplitReference]


@dataclass
class ImageIssue:
    file_name: str
    local_path: str
    source_url: str
    status: ImageStatus
    validation_error: str
    repair_status: RepairStatus
    repair_error: str
    affected_splits: list[str]
    references: list[dict[str, Any]]
    byte_count: int


@dataclass(frozen=True)
class IntegrityConfig:
    train_csv: Path = DEFAULT_TRAIN_CSV
    val_csv: Path = DEFAULT_VAL_CSV
    test_csv: Path = DEFAULT_TEST_CSV
    image_root: Path = DEFAULT_IMAGE_ROOT
    base_url: str = DEFAULT_BASE_URL
    report_path: Path = DEFAULT_REPORT_PATH
    max_attempts: int = 3
    timeout_seconds: float = 30.0
    repair: bool = False
    verbose: bool = False


@dataclass(frozen=True)
class IntegritySummary:
    total_unique_images_checked: int
    valid_image_count: int
    missing_image_count: int
    corrupt_or_truncated_image_count: int
    repaired_image_count: int
    still_invalid_image_count: int
    check_only: bool
    report_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and optionally repair every image referenced by the fixed "
            "Caltech train, validation, and test splits."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--repair", action="store_true")
    parser.add_argument("--train-csv", default=DEFAULT_TRAIN_CSV, type=Path)
    parser.add_argument("--val-csv", default=DEFAULT_VAL_CSV, type=Path)
    parser.add_argument("--test-csv", default=DEFAULT_TEST_CSV, type=Path)
    parser.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT, type=Path)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH, type=Path)
    parser.add_argument("--max-attempts", default=3, type=int)
    parser.add_argument("--timeout", default=30.0, type=float)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> IntegrityConfig:
    return IntegrityConfig(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        image_root=args.image_root,
        base_url=args.base_url,
        report_path=args.report_path,
        max_attempts=args.max_attempts,
        timeout_seconds=args.timeout,
        repair=args.repair,
        verbose=args.verbose,
    )


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def validate_config(config: IntegrityConfig) -> None:
    if config.max_attempts < 1:
        raise ValueError("--max-attempts must be at least 1")
    if config.timeout_seconds <= 0:
        raise ValueError("--timeout must be greater than zero")


def read_split_references(
    split_name: str,
    csv_path: Path,
    references_by_file: dict[str, ImageReference],
    image_root: Path,
    base_url: str,
) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"Split CSV does not exist: {csv_path}")
    if not csv_path.is_file():
        raise ValueError(f"Split CSV path is not a file: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []
        missing_columns = [
            column for column in REQUIRED_COLUMNS if column not in fieldnames
        ]
        if missing_columns:
            raise ValueError(
                f"Split CSV {csv_path} is missing required columns: {missing_columns}"
            )

        for csv_row_number, row in enumerate(reader, start=2):
            file_name = row["file_name"]
            reference = SplitReference(
                split=split_name,
                csv_path=str(csv_path),
                csv_row_number=csv_row_number,
                image_id=row.get("image_id", ""),
            )
            if file_name not in references_by_file:
                references_by_file[file_name] = ImageReference(
                    file_name=file_name,
                    local_path=safe_destination_path(image_root, file_name),
                    source_url=source_url_for_file(base_url, file_name),
                    references=[reference],
                )
            else:
                references_by_file[file_name].references.append(reference)


def collect_image_references(config: IntegrityConfig) -> list[ImageReference]:
    references_by_file: dict[str, ImageReference] = {}
    read_split_references(
        "train",
        config.train_csv,
        references_by_file,
        config.image_root,
        config.base_url,
    )
    read_split_references(
        "val",
        config.val_csv,
        references_by_file,
        config.image_root,
        config.base_url,
    )
    read_split_references(
        "test",
        config.test_csv,
        references_by_file,
        config.image_root,
        config.base_url,
    )
    return sorted(references_by_file.values(), key=lambda item: item.file_name)


def validate_image_file(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing file"
    if not path.is_file():
        return False, "path is not a file"
    byte_count = path.stat().st_size
    if byte_count == 0:
        return False, "zero-byte file"

    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
    except (OSError, UnidentifiedImageError) as exc:
        return False, str(exc)
    return True, ""


def issue_kind(validation_error: str) -> ImageStatus:
    if validation_error == "missing file":
        return "missing"
    return "invalid"


def split_names(reference: ImageReference) -> list[str]:
    return sorted({item.split for item in reference.references})


def issue_from_reference(
    reference: ImageReference,
    status: ImageStatus,
    validation_error: str,
    repair_status: RepairStatus,
    repair_error: str = "",
) -> ImageIssue:
    byte_count = (
        reference.local_path.stat().st_size
        if reference.local_path.exists() and reference.local_path.is_file()
        else 0
    )
    return ImageIssue(
        file_name=reference.file_name,
        local_path=str(reference.local_path),
        source_url=reference.source_url,
        status=status,
        validation_error=validation_error,
        repair_status=repair_status,
        repair_error=repair_error,
        affected_splits=split_names(reference),
        references=[asdict(item) for item in reference.references],
        byte_count=byte_count,
    )


def remove_file_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def repair_image(
    reference: ImageReference, config: IntegrityConfig
) -> tuple[bool, str]:
    destination_path = reference.local_path
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination_path.with_name(
        f"{destination_path.name}.repair-{uuid.uuid4().hex}.tmp"
    )
    last_error = ""
    last_http_status: int | None = None

    for attempt in range(1, config.max_attempts + 1):
        remove_file_if_exists(temp_path)
        try:
            _http_status, byte_count = download_to_temp_file(
                source_url=reference.source_url,
                temp_path=temp_path,
                timeout_seconds=config.timeout_seconds,
            )
            if byte_count <= 0 or temp_path.stat().st_size <= 0:
                last_error = "downloaded temporary file is zero bytes"
                remove_file_if_exists(temp_path)
                if attempt < config.max_attempts:
                    time.sleep(retry_delay_seconds(attempt))
                    continue
                return False, last_error

            is_valid, validation_error = validate_image_file(temp_path)
            if not is_valid:
                last_error = (
                    "downloaded temporary image failed validation: "
                    f"{validation_error}"
                )
                remove_file_if_exists(temp_path)
                if attempt < config.max_attempts:
                    time.sleep(retry_delay_seconds(attempt))
                    continue
                return False, last_error

            os.replace(temp_path, destination_path)
            final_valid, final_error = validate_image_file(destination_path)
            if not final_valid:
                return (
                    False,
                    f"replacement image still failed validation: {final_error}",
                )
            return True, ""
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            remove_file_if_exists(temp_path)
            last_error = str(exc)
            last_http_status = error_http_status(exc)
            if attempt < config.max_attempts and is_retryable_exception(exc):
                time.sleep(retry_delay_seconds(attempt))
                continue
            if last_http_status is not None:
                return False, f"HTTP {last_http_status}: {last_error}"
            return False, last_error

    remove_file_if_exists(temp_path)
    return False, last_error or "repair failed"


def print_bad_image(issue: ImageIssue) -> None:
    reference_text = ", ".join(
        f"{item['split']}:{item['csv_row_number']}:{item['image_id']}"
        for item in issue.references
    )
    print(
        "BAD IMAGE "
        f"status={issue.status} repair_status={issue.repair_status} "
        f"file_name={issue.file_name} local_path={issue.local_path} "
        f"splits=[{reference_text}] error={issue.validation_error}"
    )
    if issue.repair_error:
        print(f"  repair_error={issue.repair_error}")


def write_report(
    config: IntegrityConfig,
    references: list[ImageReference],
    issues: list[ImageIssue],
) -> IntegritySummary:
    missing_count = sum(
        1 for issue in issues if issue.validation_error == "missing file"
    )
    invalid_count = sum(
        1 for issue in issues if issue.validation_error != "missing file"
    )
    repaired_count = sum(1 for issue in issues if issue.repair_status == "repaired")
    still_invalid_count = sum(
        1 for issue in issues if issue.status in {"missing", "invalid", "repair_failed"}
    )
    summary = IntegritySummary(
        total_unique_images_checked=len(references),
        valid_image_count=len(references) - still_invalid_count,
        missing_image_count=missing_count,
        corrupt_or_truncated_image_count=invalid_count,
        repaired_image_count=repaired_count,
        still_invalid_image_count=still_invalid_count,
        check_only=not config.repair,
        report_path=str(config.report_path),
    )
    payload = {
        "summary": asdict(summary),
        "bad_images": [asdict(issue) for issue in issues],
        "user_agent": USER_AGENT,
        "source_base_url": config.base_url,
        "split_csv_paths": {
            "train": str(config.train_csv),
            "val": str(config.val_csv),
            "test": str(config.test_csv),
        },
    }
    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def run_integrity_check(config: IntegrityConfig) -> IntegritySummary:
    validate_config(config)
    references = collect_image_references(config)
    issues: list[ImageIssue] = []

    for index, reference in enumerate(references, start=1):
        is_valid, validation_error = validate_image_file(reference.local_path)
        if is_valid:
            if config.verbose and index % 500 == 0:
                LOGGER.info("Validated %s/%s images", index, len(references))
            continue

        repair_status: RepairStatus = "not_requested"
        repair_error = ""
        status = issue_kind(validation_error)
        if config.repair:
            repaired, repair_error = repair_image(reference, config)
            if repaired:
                repair_status = "repaired"
                status = "repaired"
            else:
                repair_status = "failed"
                status = "repair_failed"

        issue = issue_from_reference(
            reference=reference,
            status=status,
            validation_error=validation_error,
            repair_status=repair_status,
            repair_error=repair_error,
        )
        issues.append(issue)
        print_bad_image(issue)

    summary = write_report(config, references, issues)
    print(f"Total unique images checked: {summary.total_unique_images_checked}")
    print(f"Valid images: {summary.valid_image_count}")
    print(f"Missing images: {summary.missing_image_count}")
    print(f"Corrupt or truncated images: {summary.corrupt_or_truncated_image_count}")
    print(f"Repaired images: {summary.repaired_image_count}")
    print(f"Still invalid images: {summary.still_invalid_image_count}")
    print(f"Integrity report: {summary.report_path}")
    return summary


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    try:
        summary = run_integrity_check(config_from_args(args))
    except Exception as exc:
        LOGGER.error("%s", exc)
        sys.exit(2)
    if summary.still_invalid_image_count:
        sys.exit(1)


if __name__ == "__main__":
    main()
