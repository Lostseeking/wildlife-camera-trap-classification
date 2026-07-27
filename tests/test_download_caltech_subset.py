from __future__ import annotations

import csv
import shutil
import uuid
from email.message import Message
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError

import pytest
from PIL import Image

from data import download_caltech_subset as downloader
from data.download_caltech_subset import DownloadConfig, DownloadResult, run_download

MANIFEST_COLUMNS = ["image_id", "file_name", "category_name"]


def make_workspace_test_dir() -> Path:
    test_dir = Path("data/metadata") / f"test_download_caltech_{uuid.uuid4().hex}"
    test_dir.mkdir(parents=True)
    return test_dir


def image_bytes(color: tuple[int, int, int] = (10, 20, 30)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (2, 3), color=color).save(buffer, format="JPEG")
    return buffer.getvalue()


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def one_row(file_name: str = "image-a.jpg") -> dict[str, str]:
    return {
        "image_id": "image-a",
        "file_name": file_name,
        "category_name": "deer",
    }


def make_config(
    test_dir: Path,
    *,
    rows: list[dict[str, str]] | None = None,
    force_redownload: bool = False,
    dry_run: bool = False,
    max_attempts: int = 3,
) -> DownloadConfig:
    manifest_path = test_dir / "manifest.csv"
    write_manifest(manifest_path, rows if rows is not None else [one_row()])
    return DownloadConfig(
        manifest_path=manifest_path,
        output_dir=test_dir / "images",
        base_url="https://example.test/images",
        max_attempts=max_attempts,
        timeout_seconds=1.0,
        workers=1,
        dry_run=dry_run,
        force_redownload=force_redownload,
        download_report_path=test_dir / "download_report.csv",
        failure_report_path=test_dir / "failure_report.csv",
    )


def patch_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "data.download_caltech_subset.time.sleep",
        lambda _seconds: None,
    )


def http_error(url: str, status: int, reason: str) -> HTTPError:
    return HTTPError(url, status, reason, Message(), None)


def patch_download_bytes(
    monkeypatch: pytest.MonkeyPatch,
    payloads: list[bytes],
    *,
    statuses: list[int] | None = None,
) -> list[Path]:
    calls: list[Path] = []
    status_values = statuses or [200 for _payload in payloads]

    def fake_download(
        source_url: str,
        temp_path: Path,
        timeout_seconds: float,
    ) -> tuple[int, int]:
        del source_url, timeout_seconds
        calls.append(temp_path)
        payload = payloads.pop(0)
        temp_path.write_bytes(payload)
        return status_values.pop(0), len(payload)

    monkeypatch.setattr(downloader, "download_to_temp_file", fake_download)
    return calls


def run_with_cleanup(
    config: DownloadConfig,
) -> tuple[list[DownloadResult], downloader.DownloadSummary]:
    return run_download(config)


def test_successful_download(monkeypatch: pytest.MonkeyPatch) -> None:
    test_dir = make_workspace_test_dir()
    try:
        payload = image_bytes()
        patch_download_bytes(monkeypatch, [payload])
        config = make_config(test_dir)

        results, summary = run_with_cleanup(config)

        destination = config.output_dir / "image-a.jpg"
        assert results[0].status == "downloaded"
        assert destination.exists()
        assert summary.newly_downloaded == 1
        assert summary.total_bytes_downloaded == len(payload)
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_valid_existing_image_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    test_dir = make_workspace_test_dir()
    try:
        config = make_config(test_dir)
        destination = config.output_dir / "image-a.jpg"
        destination.parent.mkdir(parents=True)
        destination.write_bytes(image_bytes())

        def should_not_download(
            source_url: str,
            temp_path: Path,
            timeout_seconds: float,
        ) -> tuple[int, int]:
            del source_url, temp_path, timeout_seconds
            raise AssertionError("download should not run")

        monkeypatch.setattr(downloader, "download_to_temp_file", should_not_download)
        results, summary = run_with_cleanup(config)

        assert results[0].status == "skipped_valid"
        assert summary.skipped_valid == 1
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_corrupted_existing_image_is_redownloaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_dir = make_workspace_test_dir()
    try:
        payload = image_bytes((200, 0, 0))
        patch_download_bytes(monkeypatch, [payload])
        config = make_config(test_dir)
        destination = config.output_dir / "image-a.jpg"
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b"not an image")

        results, summary = run_with_cleanup(config)

        assert results[0].status == "redownloaded"
        assert destination.read_bytes() == payload
        assert summary.redownloaded == 1
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_retry_after_transient_http_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    test_dir = make_workspace_test_dir()
    try:
        patch_sleep(monkeypatch)
        payload = image_bytes()
        calls = 0

        def transient_then_success(
            source_url: str,
            temp_path: Path,
            timeout_seconds: float,
        ) -> tuple[int, int]:
            del timeout_seconds
            nonlocal calls
            calls += 1
            if calls == 1:
                raise http_error(source_url, 503, "busy")
            temp_path.write_bytes(payload)
            return 200, len(payload)

        monkeypatch.setattr(
            downloader,
            "download_to_temp_file",
            transient_then_success,
        )
        config = make_config(test_dir)

        results, summary = run_with_cleanup(config)

        assert results[0].status == "downloaded"
        assert results[0].attempts == 2
        assert calls == 2
        assert summary.newly_downloaded == 1
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_permanent_http_failure_is_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_dir = make_workspace_test_dir()
    try:
        calls = 0

        def permanent_failure(
            source_url: str,
            temp_path: Path,
            timeout_seconds: float,
        ) -> tuple[int, int]:
            del temp_path, timeout_seconds
            nonlocal calls
            calls += 1
            raise http_error(source_url, 404, "missing")

        monkeypatch.setattr(downloader, "download_to_temp_file", permanent_failure)
        config = make_config(test_dir)

        results, summary = run_with_cleanup(config)

        failures = read_csv_rows(config.failure_report_path)
        assert results[0].status == "failed"
        assert results[0].http_status == 404
        assert calls == 1
        assert summary.failed == 1
        assert failures[0]["image_id"] == "image-a"
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_corrupt_downloaded_content_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_dir = make_workspace_test_dir()
    try:
        patch_sleep(monkeypatch)
        calls = patch_download_bytes(
            monkeypatch,
            [b"broken image", b"still broken"],
        )
        config = make_config(test_dir, max_attempts=2)

        results, summary = run_with_cleanup(config)

        assert results[0].status == "failed"
        assert "image validation failed" in results[0].error_message
        assert len(calls) == 2
        assert summary.failed == 1
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_temporary_files_are_cleaned_up(monkeypatch: pytest.MonkeyPatch) -> None:
    test_dir = make_workspace_test_dir()
    try:
        patch_download_bytes(monkeypatch, [b"broken image"])
        config = make_config(test_dir, max_attempts=1)

        results, _summary = run_with_cleanup(config)

        partials = list(config.output_dir.rglob("*.part-*"))
        assert results[0].status == "failed"
        assert partials == []
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_dry_run_performs_no_image_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    test_dir = make_workspace_test_dir()
    try:
        config = make_config(test_dir, dry_run=True)

        def should_not_download(
            source_url: str,
            temp_path: Path,
            timeout_seconds: float,
        ) -> tuple[int, int]:
            del source_url, temp_path, timeout_seconds
            raise AssertionError("download should not run")

        monkeypatch.setattr(downloader, "download_to_temp_file", should_not_download)
        results, summary = run_with_cleanup(config)

        assert results[0].status == "dry_run"
        assert not config.output_dir.exists()
        assert summary.dry_run == 1
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_force_redownload_replaces_valid_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_dir = make_workspace_test_dir()
    try:
        new_payload = image_bytes((0, 200, 0))
        patch_download_bytes(monkeypatch, [new_payload])
        config = make_config(test_dir, force_redownload=True)
        destination = config.output_dir / "image-a.jpg"
        destination.parent.mkdir(parents=True)
        destination.write_bytes(image_bytes((10, 10, 10)))

        results, summary = run_with_cleanup(config)

        assert results[0].status == "redownloaded"
        assert destination.read_bytes() == new_payload
        assert summary.redownloaded == 1
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_nested_relative_paths_are_handled_safely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_dir = make_workspace_test_dir()
    try:
        patch_download_bytes(monkeypatch, [image_bytes()])
        config = make_config(test_dir, rows=[one_row("loc-1/nested/image-a.jpg")])

        results, _summary = run_with_cleanup(config)

        assert results[0].status == "downloaded"
        assert (config.output_dir / "loc-1" / "nested" / "image-a.jpg").exists()
        assert results[0].source_url.endswith("/loc-1/nested/image-a.jpg")
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_path_traversal_filenames_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_dir = make_workspace_test_dir()
    try:
        config = make_config(test_dir, rows=[one_row("../escape.jpg")])

        def should_not_download(
            source_url: str,
            temp_path: Path,
            timeout_seconds: float,
        ) -> tuple[int, int]:
            del source_url, temp_path, timeout_seconds
            raise AssertionError("download should not run")

        monkeypatch.setattr(downloader, "download_to_temp_file", should_not_download)
        results, summary = run_with_cleanup(config)

        assert results[0].status == "failed"
        assert "Unsafe manifest file_name" in results[0].error_message
        assert summary.failed == 1
        assert not (test_dir / "escape.jpg").exists()
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_failure_report_and_summary_counts_are_correct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_dir = make_workspace_test_dir()
    try:
        payload = image_bytes()
        calls = 0

        def mixed_download(
            source_url: str,
            temp_path: Path,
            timeout_seconds: float,
        ) -> tuple[int, int]:
            del timeout_seconds
            nonlocal calls
            calls += 1
            if "missing" in source_url:
                raise http_error(source_url, 404, "missing")
            temp_path.write_bytes(payload)
            return 200, len(payload)

        rows = [
            one_row("ok.jpg"),
            {
                "image_id": "missing",
                "file_name": "missing.jpg",
                "category_name": "bird",
            },
        ]
        monkeypatch.setattr(downloader, "download_to_temp_file", mixed_download)
        config = make_config(test_dir, rows=rows)

        results, summary = run_with_cleanup(config)
        report_rows = read_csv_rows(config.download_report_path)
        failure_rows = read_csv_rows(config.failure_report_path)

        assert [result.status for result in results] == ["downloaded", "failed"]
        assert summary.total_considered == 2
        assert summary.newly_downloaded == 1
        assert summary.failed == 1
        assert len(report_rows) == 2
        assert len(failure_rows) == 1
        assert failure_rows[0]["image_id"] == "missing"
        assert calls == 2
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)
