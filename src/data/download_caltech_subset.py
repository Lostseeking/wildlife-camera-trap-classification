from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from PIL import Image, UnidentifiedImageError

DEFAULT_MANIFEST_PATH = Path("data/metadata/caltech_mvp_subset.csv")
DEFAULT_OUTPUT_DIR = Path("data/raw/caltech_camera_traps/images")
DEFAULT_BASE_URL = (
    "https://lilawildlife.blob.core.windows.net/"
    "lila-wildlife/caltech-unzipped/cct_images"
)
DEFAULT_DOWNLOAD_REPORT_PATH = Path(
    "data/metadata/caltech_mvp_subset_download_report.csv"
)
DEFAULT_FAILURE_REPORT_PATH = Path(
    "data/metadata/caltech_mvp_subset_download_failures.csv"
)
USER_AGENT = "wildlife-caltech-subset-downloader/1.0"
REQUIRED_MANIFEST_COLUMNS = ["image_id", "file_name", "category_name"]
REPORT_COLUMNS = [
    "image_id",
    "file_name",
    "category_name",
    "source_url",
    "destination_path",
    "status",
    "attempts",
    "http_status",
    "downloaded_byte_count",
    "error_message",
]
SUPPORTED_IMAGE_FORMATS = {"JPEG", "PNG", "BMP", "GIF", "TIFF", "WEBP"}
RETRYABLE_HTTP_STATUSES = {429, 500, 502, 503, 504}
DownloadStatus = Literal[
    "downloaded",
    "skipped_valid",
    "redownloaded",
    "failed",
    "dry_run",
]


@dataclass(frozen=True)
class ManifestImage:
    row_index: int
    image_id: str
    file_name: str
    category_name: str


@dataclass(frozen=True)
class DownloadResult:
    row_index: int
    image_id: str
    file_name: str
    category_name: str
    source_url: str
    destination_path: str
    status: DownloadStatus
    attempts: int
    http_status: int | None
    downloaded_byte_count: int
    error_message: str

    def to_row(self) -> dict[str, str]:
        return {
            "image_id": self.image_id,
            "file_name": self.file_name,
            "category_name": self.category_name,
            "source_url": self.source_url,
            "destination_path": self.destination_path,
            "status": self.status,
            "attempts": str(self.attempts),
            "http_status": "" if self.http_status is None else str(self.http_status),
            "downloaded_byte_count": str(self.downloaded_byte_count),
            "error_message": self.error_message,
        }


@dataclass(frozen=True)
class DownloadConfig:
    manifest_path: Path = DEFAULT_MANIFEST_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR
    base_url: str = DEFAULT_BASE_URL
    max_attempts: int = 3
    timeout_seconds: float = 30.0
    workers: int = 4
    limit: int | None = None
    dry_run: bool = False
    force_redownload: bool = False
    download_report_path: Path = DEFAULT_DOWNLOAD_REPORT_PATH
    failure_report_path: Path = DEFAULT_FAILURE_REPORT_PATH
    verbose: bool = False


@dataclass(frozen=True)
class DownloadSummary:
    total_considered: int
    newly_downloaded: int
    skipped_valid: int
    redownloaded: int
    failed: int
    dry_run: int
    total_bytes_downloaded: int
    elapsed_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download images listed in the frozen Caltech MVP subset manifest."
    )
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST_PATH, type=Path)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=(
            "Base image URL. Defaults to the LILA Azure HTTPS folder for "
            "Caltech Camera Traps images."
        ),
    )
    parser.add_argument("--max-attempts", default=3, type=int)
    parser.add_argument("--timeout", default=30.0, type=float)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-redownload", action="store_true")
    parser.add_argument(
        "--download-report-output",
        default=DEFAULT_DOWNLOAD_REPORT_PATH,
        type=Path,
    )
    parser.add_argument(
        "--failure-report-output",
        default=DEFAULT_FAILURE_REPORT_PATH,
        type=Path,
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> DownloadConfig:
    return DownloadConfig(
        manifest_path=args.manifest_path,
        output_dir=args.output_dir,
        base_url=args.base_url,
        max_attempts=args.max_attempts,
        timeout_seconds=args.timeout,
        workers=args.workers,
        limit=args.limit,
        dry_run=args.dry_run,
        force_redownload=args.force_redownload,
        download_report_path=args.download_report_output,
        failure_report_path=args.failure_report_output,
        verbose=args.verbose,
    )


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def read_manifest(path: Path, limit: int | None = None) -> list[ManifestImage]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Manifest path is not a file: {path}")
    if limit is not None and limit < 1:
        raise ValueError("--limit must be at least 1 when provided")

    records: list[ManifestImage] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []
        missing_columns = [
            column
            for column in REQUIRED_MANIFEST_COLUMNS
            if column not in fieldnames
        ]
        if missing_columns:
            raise ValueError(f"Manifest is missing required columns: {missing_columns}")

        for row_index, row in enumerate(reader, start=1):
            records.append(
                ManifestImage(
                    row_index=row_index,
                    image_id=row["image_id"],
                    file_name=row["file_name"],
                    category_name=row["category_name"],
                )
            )
            if limit is not None and len(records) >= limit:
                break
    return records


def safe_destination_path(output_dir: Path, file_name: str) -> Path:
    normalized_file_name = file_name.replace("\\", "/")
    relative_path = PurePosixPath(normalized_file_name)
    if (
        not normalized_file_name
        or relative_path.is_absolute()
        or any(part in {"", ".", ".."} for part in relative_path.parts)
        or any(":" in part for part in relative_path.parts)
    ):
        raise ValueError(f"Unsafe manifest file_name: {file_name}")

    output_root = output_dir.resolve()
    destination_path = (output_root / Path(*relative_path.parts)).resolve()
    try:
        destination_path.relative_to(output_root)
    except ValueError as exc:
        raise ValueError(f"Unsafe manifest file_name: {file_name}") from exc
    return destination_path


def source_url_for_file(base_url: str, file_name: str) -> str:
    normalized_file_name = file_name.replace("\\", "/")
    relative_path = PurePosixPath(normalized_file_name)
    if relative_path.is_absolute() or any(
        part in {"", ".", ".."} or ":" in part for part in relative_path.parts
    ):
        raise ValueError(f"Unsafe manifest file_name: {file_name}")

    quoted_path = "/".join(quote(part) for part in relative_path.parts)
    return f"{base_url.rstrip('/')}/{quoted_path}"


def validate_image(path: Path) -> tuple[bool, str]:
    try:
        with Image.open(path) as image:
            image.verify()
            image_format = image.format
        with Image.open(path) as image:
            width, height = image.size
            reopened_format = image.format
            image.load()
    except (OSError, UnidentifiedImageError) as exc:
        return False, str(exc)

    if image_format not in SUPPORTED_IMAGE_FORMATS:
        return False, f"unsupported image format: {image_format}"
    if reopened_format not in SUPPORTED_IMAGE_FORMATS:
        return False, f"unsupported reopened image format: {reopened_format}"
    if width <= 0 or height <= 0:
        return False, f"invalid image dimensions: {width}x{height}"
    return True, ""


def remove_file_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def retry_delay_seconds(attempt: int) -> float:
    return float(min(8.0, 0.5 * (2 ** (attempt - 1))))


def is_retryable_exception(exc: BaseException) -> bool:
    if isinstance(exc, HTTPError):
        return exc.code in RETRYABLE_HTTP_STATUSES
    if isinstance(exc, URLError):
        return True
    return isinstance(exc, TimeoutError)


def error_http_status(exc: BaseException) -> int | None:
    if isinstance(exc, HTTPError):
        return exc.code
    return None


def download_to_temp_file(
    source_url: str,
    temp_path: Path,
    timeout_seconds: float,
) -> tuple[int, int]:
    request = Request(source_url, headers={"User-Agent": USER_AGENT})
    byte_count = 0
    with urlopen(request, timeout=timeout_seconds) as response:
        status = getattr(response, "status", None) or response.getcode()
        if status >= 400:
            raise HTTPError(
                source_url,
                status,
                f"HTTP {status}",
                response.headers,
                response,
            )
        with temp_path.open("wb") as file:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                file.write(chunk)
                byte_count += len(chunk)
    return int(status), byte_count


def failed_result(
    image: ManifestImage,
    source_url: str,
    destination_path: str,
    attempts: int,
    http_status: int | None,
    downloaded_byte_count: int,
    error_message: str,
) -> DownloadResult:
    return DownloadResult(
        row_index=image.row_index,
        image_id=image.image_id,
        file_name=image.file_name,
        category_name=image.category_name,
        source_url=source_url,
        destination_path=destination_path,
        status="failed",
        attempts=attempts,
        http_status=http_status,
        downloaded_byte_count=downloaded_byte_count,
        error_message=error_message,
    )


def process_image(image: ManifestImage, config: DownloadConfig) -> DownloadResult:
    source_url = ""
    destination_path_text = ""
    try:
        destination_path = safe_destination_path(config.output_dir, image.file_name)
        source_url = source_url_for_file(config.base_url, image.file_name)
        destination_path_text = str(destination_path)
    except ValueError as exc:
        return failed_result(
            image=image,
            source_url=source_url,
            destination_path=destination_path_text,
            attempts=0,
            http_status=None,
            downloaded_byte_count=0,
            error_message=str(exc),
        )

    if config.dry_run:
        return DownloadResult(
            row_index=image.row_index,
            image_id=image.image_id,
            file_name=image.file_name,
            category_name=image.category_name,
            source_url=source_url,
            destination_path=destination_path_text,
            status="dry_run",
            attempts=0,
            http_status=None,
            downloaded_byte_count=0,
            error_message="",
        )

    existed_before = destination_path.exists()
    if existed_before and not config.force_redownload:
        is_valid, validation_error = validate_image(destination_path)
        if is_valid:
            return DownloadResult(
                row_index=image.row_index,
                image_id=image.image_id,
                file_name=image.file_name,
                category_name=image.category_name,
                source_url=source_url,
                destination_path=destination_path_text,
                status="skipped_valid",
                attempts=0,
                http_status=None,
                downloaded_byte_count=0,
                error_message="",
            )
        remove_file_if_exists(destination_path)
        logging.warning(
            "Removed invalid existing image before redownload: %s (%s)",
            destination_path,
            validation_error,
        )

    temp_path = destination_path.with_name(
        f"{destination_path.name}.part-{uuid.uuid4().hex}"
    )
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    last_error = ""
    last_http_status: int | None = None
    total_downloaded_bytes = 0
    for attempt in range(1, config.max_attempts + 1):
        remove_file_if_exists(temp_path)
        try:
            http_status, downloaded_bytes = download_to_temp_file(
                source_url=source_url,
                temp_path=temp_path,
                timeout_seconds=config.timeout_seconds,
            )
            total_downloaded_bytes += downloaded_bytes
            is_valid, validation_error = validate_image(temp_path)
            if not is_valid:
                last_error = f"image validation failed: {validation_error}"
                remove_file_if_exists(temp_path)
                if attempt < config.max_attempts:
                    time.sleep(retry_delay_seconds(attempt))
                    continue
                return failed_result(
                    image=image,
                    source_url=source_url,
                    destination_path=destination_path_text,
                    attempts=attempt,
                    http_status=http_status,
                    downloaded_byte_count=total_downloaded_bytes,
                    error_message=last_error,
                )

            os.replace(temp_path, destination_path)
            status: DownloadStatus = "redownloaded" if existed_before else "downloaded"
            return DownloadResult(
                row_index=image.row_index,
                image_id=image.image_id,
                file_name=image.file_name,
                category_name=image.category_name,
                source_url=source_url,
                destination_path=destination_path_text,
                status=status,
                attempts=attempt,
                http_status=http_status,
                downloaded_byte_count=downloaded_bytes,
                error_message="",
            )
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            remove_file_if_exists(temp_path)
            last_error = str(exc)
            last_http_status = error_http_status(exc)
            if attempt < config.max_attempts and is_retryable_exception(exc):
                time.sleep(retry_delay_seconds(attempt))
                continue
            return failed_result(
                image=image,
                source_url=source_url,
                destination_path=destination_path_text,
                attempts=attempt,
                http_status=last_http_status,
                downloaded_byte_count=total_downloaded_bytes,
                error_message=last_error,
            )

    remove_file_if_exists(temp_path)
    return failed_result(
        image=image,
        source_url=source_url,
        destination_path=destination_path_text,
        attempts=config.max_attempts,
        http_status=last_http_status,
        downloaded_byte_count=total_downloaded_bytes,
        error_message=last_error or "download failed",
    )


def write_report(path: Path, results: list[DownloadResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_row())


def summarize_results(
    results: list[DownloadResult],
    elapsed_seconds: float,
) -> DownloadSummary:
    return DownloadSummary(
        total_considered=len(results),
        newly_downloaded=sum(1 for result in results if result.status == "downloaded"),
        skipped_valid=sum(1 for result in results if result.status == "skipped_valid"),
        redownloaded=sum(1 for result in results if result.status == "redownloaded"),
        failed=sum(1 for result in results if result.status == "failed"),
        dry_run=sum(1 for result in results if result.status == "dry_run"),
        total_bytes_downloaded=sum(
            result.downloaded_byte_count for result in results
        ),
        elapsed_seconds=elapsed_seconds,
    )


def print_summary(summary: DownloadSummary) -> None:
    print(f"Total manifest rows considered: {summary.total_considered}")
    print(f"Newly downloaded: {summary.newly_downloaded}")
    print(f"Valid existing files skipped: {summary.skipped_valid}")
    print(f"Successfully redownloaded: {summary.redownloaded}")
    print(f"Failed: {summary.failed}")
    print(f"Dry run: {summary.dry_run}")
    print(f"Total bytes downloaded: {summary.total_bytes_downloaded}")
    print(f"Elapsed seconds: {summary.elapsed_seconds:.2f}")


def run_download(
    config: DownloadConfig,
) -> tuple[list[DownloadResult], DownloadSummary]:
    if config.max_attempts < 1:
        raise ValueError("--max-attempts must be at least 1")
    if config.timeout_seconds <= 0:
        raise ValueError("--timeout must be greater than 0")
    if config.workers < 1:
        raise ValueError("--workers must be at least 1")

    start_time = time.monotonic()
    images = read_manifest(config.manifest_path, config.limit)
    logging.info("Loaded %s manifest rows", len(images))

    results: list[DownloadResult] = []
    success_count = 0
    failure_count = 0

    if config.workers == 1:
        for image in images:
            result = process_image(image, config)
            results.append(result)
            success_count, failure_count = log_progress(
                processed=len(results),
                total=len(images),
                result=result,
                success_count=success_count,
                failure_count=failure_count,
                verbose=config.verbose,
            )
    else:
        with ThreadPoolExecutor(max_workers=config.workers) as executor:
            futures = [
                executor.submit(process_image, image, config) for image in images
            ]
            for processed, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                results.append(result)
                success_count, failure_count = log_progress(
                    processed=processed,
                    total=len(images),
                    result=result,
                    success_count=success_count,
                    failure_count=failure_count,
                    verbose=config.verbose,
                )

    results = sorted(results, key=lambda result: result.row_index)
    failures = [result for result in results if result.status == "failed"]
    write_report(config.download_report_path, results)
    write_report(config.failure_report_path, failures)

    summary = summarize_results(
        results=results,
        elapsed_seconds=time.monotonic() - start_time,
    )
    print_summary(summary)
    return results, summary


def log_progress(
    processed: int,
    total: int,
    result: DownloadResult,
    success_count: int,
    failure_count: int,
    verbose: bool,
) -> tuple[int, int]:
    if result.status == "failed":
        failure_count += 1
        logging.error(
            "[%s/%s] failed %s: %s",
            processed,
            total,
            result.file_name,
            result.error_message,
        )
    else:
        success_count += 1
        if verbose or result.status in {"downloaded", "redownloaded"}:
            logging.info(
                "[%s/%s] %s %s",
                processed,
                total,
                result.status,
                result.file_name,
            )

    if processed == total or processed % 100 == 0:
        logging.info(
            "Progress %s/%s, successes=%s, failures=%s",
            processed,
            total,
            success_count,
            failure_count,
        )
    return success_count, failure_count


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    try:
        _results, summary = run_download(config_from_args(args))
    except Exception as exc:
        logging.error("%s", exc)
        sys.exit(2)

    if summary.failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
