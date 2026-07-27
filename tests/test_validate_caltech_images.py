from __future__ import annotations

import csv
import json
import shutil
import uuid
from email.message import Message
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError

import pytest
from PIL import Image

from data import validate_caltech_images as validator
from data.validate_caltech_images import (
    IntegrityConfig,
    collect_image_references,
    run_integrity_check,
    validate_image_file,
)

SPLIT_FIELDS = ["image_id", "file_name", "category_name", "location", "seq_id"]


def make_workspace_test_dir() -> Path:
    test_dir = Path("artifacts/data_checks_tests") / f"case_{uuid.uuid4().hex}"
    test_dir.mkdir(parents=True)
    return test_dir


def image_bytes(color: tuple[int, int, int] = (10, 20, 30)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (8, 8), color=color).save(buffer, format="JPEG")
    return buffer.getvalue()


def write_split(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SPLIT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def row(image_id: str, file_name: str, split: str = "train") -> dict[str, str]:
    return {
        "image_id": image_id,
        "file_name": file_name,
        "category_name": "deer",
        "location": f"{split}-loc",
        "seq_id": f"{split}-seq",
    }


def write_image(path: Path, payload: bytes | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(image_bytes() if payload is None else payload)


def build_config(
    test_dir: Path,
    *,
    train_rows: list[dict[str, str]],
    val_rows: list[dict[str, str]] | None = None,
    test_rows: list[dict[str, str]] | None = None,
    repair: bool = False,
) -> IntegrityConfig:
    train_csv = test_dir / "train.csv"
    val_csv = test_dir / "val.csv"
    test_csv = test_dir / "test.csv"
    write_split(train_csv, train_rows)
    write_split(val_csv, val_rows or [])
    write_split(test_csv, test_rows or [])
    return IntegrityConfig(
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        image_root=test_dir / "images",
        base_url="https://example.test/cct_images",
        report_path=test_dir / "report.json",
        max_attempts=2,
        timeout_seconds=1.0,
        repair=repair,
    )


def load_report(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_valid_image_passes() -> None:
    test_dir = make_workspace_test_dir()
    try:
        config = build_config(test_dir, train_rows=[row("valid", "valid.jpg")])
        write_image(config.image_root / "valid.jpg")

        summary = run_integrity_check(config)

        assert summary.valid_image_count == 1
        assert summary.missing_image_count == 0
        assert summary.corrupt_or_truncated_image_count == 0
        assert load_report(config.report_path)["bad_images"] == []
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_missing_image_is_reported() -> None:
    test_dir = make_workspace_test_dir()
    try:
        config = build_config(test_dir, train_rows=[row("missing", "missing.jpg")])

        summary = run_integrity_check(config)
        report = load_report(config.report_path)
        bad_images = report["bad_images"]

        assert summary.missing_image_count == 1
        assert isinstance(bad_images, list)
        assert bad_images[0]["file_name"] == "missing.jpg"
        assert bad_images[0]["validation_error"] == "missing file"
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_zero_byte_image_is_reported() -> None:
    test_dir = make_workspace_test_dir()
    try:
        config = build_config(test_dir, train_rows=[row("zero", "zero.jpg")])
        write_image(config.image_root / "zero.jpg", b"")

        summary = run_integrity_check(config)
        bad_image = load_report(config.report_path)["bad_images"][0]

        assert summary.corrupt_or_truncated_image_count == 1
        assert bad_image["validation_error"] == "zero-byte file"
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_truncated_image_is_reported() -> None:
    test_dir = make_workspace_test_dir()
    try:
        config = build_config(test_dir, train_rows=[row("truncated", "bad.jpg")])
        write_image(config.image_root / "bad.jpg", image_bytes()[:-10])

        summary = run_integrity_check(config)
        bad_image = load_report(config.report_path)["bad_images"][0]

        assert summary.corrupt_or_truncated_image_count == 1
        assert "Truncated" in bad_image["validation_error"]
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_image_that_verifies_but_fails_full_decode_is_reported() -> None:
    test_dir = make_workspace_test_dir()
    try:
        config = build_config(test_dir, train_rows=[row("late-fail", "late.jpg")])
        write_image(config.image_root / "late.jpg", image_bytes()[:-1])

        is_valid, error = validate_image_file(config.image_root / "late.jpg")

        assert not is_valid
        assert "image file is truncated" in error
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_check_only_mode_does_not_modify_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_dir = make_workspace_test_dir()
    try:
        config = build_config(test_dir, train_rows=[row("bad", "bad.jpg")])
        damaged_path = config.image_root / "bad.jpg"
        write_image(damaged_path, b"not an image")
        before = damaged_path.read_bytes()

        def should_not_download(
            source_url: str,
            temp_path: Path,
            timeout_seconds: float,
        ) -> tuple[int, int]:
            del source_url, temp_path, timeout_seconds
            raise AssertionError("check-only mode must not download")

        monkeypatch.setattr(validator, "download_to_temp_file", should_not_download)
        summary = run_integrity_check(config)

        assert summary.still_invalid_image_count == 1
        assert damaged_path.read_bytes() == before
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_repair_mode_replaces_only_damaged_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_dir = make_workspace_test_dir()
    try:
        config = build_config(
            test_dir,
            train_rows=[row("valid", "valid.jpg"), row("bad", "bad.jpg")],
            repair=True,
        )
        valid_path = config.image_root / "valid.jpg"
        damaged_path = config.image_root / "bad.jpg"
        valid_payload = image_bytes((1, 2, 3))
        repaired_payload = image_bytes((200, 20, 30))
        write_image(valid_path, valid_payload)
        write_image(damaged_path, b"broken")
        calls: list[str] = []

        def fake_download(
            source_url: str,
            temp_path: Path,
            timeout_seconds: float,
        ) -> tuple[int, int]:
            del timeout_seconds
            calls.append(source_url)
            temp_path.write_bytes(repaired_payload)
            return 200, len(repaired_payload)

        monkeypatch.setattr(validator, "download_to_temp_file", fake_download)
        summary = run_integrity_check(config)

        assert summary.repaired_image_count == 1
        assert valid_path.read_bytes() == valid_payload
        assert damaged_path.read_bytes() == repaired_payload
        assert calls == ["https://example.test/cct_images/bad.jpg"]
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_valid_images_are_not_redownloaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_dir = make_workspace_test_dir()
    try:
        config = build_config(
            test_dir,
            train_rows=[row("valid", "valid.jpg")],
            repair=True,
        )
        write_image(config.image_root / "valid.jpg")

        def should_not_download(
            source_url: str,
            temp_path: Path,
            timeout_seconds: float,
        ) -> tuple[int, int]:
            del source_url, temp_path, timeout_seconds
            raise AssertionError("valid image should not be redownloaded")

        monkeypatch.setattr(validator, "download_to_temp_file", should_not_download)
        summary = run_integrity_check(config)

        assert summary.repaired_image_count == 0
        assert summary.still_invalid_image_count == 0
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_failed_replacement_preserves_original_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_dir = make_workspace_test_dir()
    try:
        config = build_config(
            test_dir,
            train_rows=[row("bad", "bad.jpg")],
            repair=True,
        )
        damaged_path = config.image_root / "bad.jpg"
        original_payload = b"original broken bytes"
        write_image(damaged_path, original_payload)

        def fake_download(
            source_url: str,
            temp_path: Path,
            timeout_seconds: float,
        ) -> tuple[int, int]:
            del source_url, timeout_seconds
            payload = b"still not an image"
            temp_path.write_bytes(payload)
            return 200, len(payload)

        monkeypatch.setattr(validator, "download_to_temp_file", fake_download)
        summary = run_integrity_check(config)

        assert summary.repaired_image_count == 0
        assert summary.still_invalid_image_count == 1
        assert damaged_path.read_bytes() == original_payload
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_repaired_temporary_image_must_pass_verify_and_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_dir = make_workspace_test_dir()
    try:
        config = build_config(
            test_dir,
            train_rows=[row("bad", "bad.jpg")],
            repair=True,
        )
        damaged_path = config.image_root / "bad.jpg"
        write_image(damaged_path, b"bad")
        payloads = [image_bytes()[:-1], image_bytes((0, 200, 0))]

        def fake_download(
            source_url: str,
            temp_path: Path,
            timeout_seconds: float,
        ) -> tuple[int, int]:
            del source_url, timeout_seconds
            payload = payloads.pop(0)
            temp_path.write_bytes(payload)
            return 200, len(payload)

        monkeypatch.setattr(validator, "download_to_temp_file", fake_download)
        monkeypatch.setattr(validator.time, "sleep", lambda _seconds: None)
        summary = run_integrity_check(config)

        assert summary.repaired_image_count == 1
        assert validate_image_file(damaged_path) == (True, "")
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_duplicate_references_across_splits_are_checked_once() -> None:
    test_dir = make_workspace_test_dir()
    try:
        shared = row("shared-train", "shared.jpg", "train")
        config = build_config(
            test_dir,
            train_rows=[shared],
            val_rows=[row("shared-val", "shared.jpg", "val")],
            test_rows=[row("shared-test", "shared.jpg", "test")],
        )
        write_image(config.image_root / "shared.jpg")

        references = collect_image_references(config)
        summary = run_integrity_check(config)

        assert len(references) == 1
        assert summary.total_unique_images_checked == 1
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_split_csv_files_remain_unchanged() -> None:
    test_dir = make_workspace_test_dir()
    try:
        config = build_config(
            test_dir,
            train_rows=[row("bad", "bad.jpg")],
            val_rows=[row("valid", "valid.jpg", "val")],
        )
        write_image(config.image_root / "bad.jpg", b"bad")
        write_image(config.image_root / "valid.jpg")
        before = {
            path: path.read_bytes()
            for path in (config.train_csv, config.val_csv, config.test_csv)
        }

        _summary = run_integrity_check(config)

        assert {
            path: path.read_bytes()
            for path in (config.train_csv, config.val_csv, config.test_csv)
        } == before
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_report_json_contains_required_fields() -> None:
    test_dir = make_workspace_test_dir()
    try:
        config = build_config(test_dir, train_rows=[row("missing", "missing.jpg")])

        _summary = run_integrity_check(config)
        report = load_report(config.report_path)
        summary = report["summary"]
        bad_image = report["bad_images"][0]

        assert "total_unique_images_checked" in summary
        assert "valid_image_count" in summary
        assert "missing_image_count" in summary
        assert "corrupt_or_truncated_image_count" in summary
        assert "affected_splits" in bad_image
        assert "references" in bad_image
        assert bad_image["references"][0]["csv_row_number"] == 2
        assert bad_image["references"][0]["image_id"] == "missing"
        assert "local_path" in bad_image
        assert "repair_status" in bad_image
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_download_failure_is_reported_clearly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_dir = make_workspace_test_dir()
    try:
        config = build_config(
            test_dir,
            train_rows=[row("missing", "missing.jpg")],
            repair=True,
        )

        def fail_download(
            source_url: str,
            temp_path: Path,
            timeout_seconds: float,
        ) -> tuple[int, int]:
            del temp_path, timeout_seconds
            raise HTTPError(source_url, 404, "missing", Message(), None)

        monkeypatch.setattr(validator, "download_to_temp_file", fail_download)
        summary = run_integrity_check(config)
        bad_image = load_report(config.report_path)["bad_images"][0]

        assert summary.still_invalid_image_count == 1
        assert bad_image["repair_status"] == "failed"
        assert "HTTP 404" in bad_image["repair_error"]
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_atomic_replacement_behavior_removes_temp_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_dir = make_workspace_test_dir()
    try:
        config = build_config(
            test_dir,
            train_rows=[row("bad", "bad.jpg")],
            repair=True,
        )
        damaged_path = config.image_root / "bad.jpg"
        original_payload = b"broken original"
        repaired_payload = image_bytes((9, 8, 7))
        write_image(damaged_path, original_payload)
        replace_calls: list[tuple[Path, Path]] = []

        def fake_download(
            source_url: str,
            temp_path: Path,
            timeout_seconds: float,
        ) -> tuple[int, int]:
            del source_url, timeout_seconds
            temp_path.write_bytes(repaired_payload)
            return 200, len(repaired_payload)

        def fake_replace(source: str | Path, destination: str | Path) -> None:
            replace_calls.append((Path(source), Path(destination)))
            Path(destination).write_bytes(Path(source).read_bytes())
            Path(source).unlink()

        monkeypatch.setattr(validator, "download_to_temp_file", fake_download)
        monkeypatch.setattr(validator.os, "replace", fake_replace)
        summary = run_integrity_check(config)

        assert summary.repaired_image_count == 1
        assert len(replace_calls) == 1
        assert replace_calls[0][1] == damaged_path.resolve()
        assert list(config.image_root.glob("*.repair-*.tmp")) == []
        assert damaged_path.read_bytes() == repaired_payload
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)
