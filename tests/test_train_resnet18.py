from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch
from PIL import Image
from src.models.resnet18_classifier import build_resnet18_classifier
from src.training.train_resnet18 import (
    NUM_CLASSES,
    TrainingConfig,
    build_optimizer,
    confusion_matrix_from_predictions,
    frozen_backbone_unchanged,
    metrics_from_confusion_matrix,
    snapshot_named_parameters,
    train_one_epoch,
    train_resnet18,
    validate_one_epoch,
)
from torch import nn


def make_workspace_test_dir() -> Path:
    test_dir = Path("artifacts/training_tests") / f"resnet18_{uuid.uuid4().hex}"
    test_dir.mkdir(parents=True)
    return test_dir


def save_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (48, 48), color=color).save(path)


def rows_for_split(split: str, count: int) -> list[dict[str, str]]:
    categories = ("empty", "deer", "coyote", "bobcat", "bird", "opossum")
    return [
        {
            "image_id": f"{split}-{index}",
            "file_name": f"{split}/{index}.jpg",
            "category_name": categories[index % len(categories)],
            "location": f"{split}-loc-{index}",
            "seq_id": f"{split}-seq-{index}",
            "split": split,
        }
        for index in range(count)
    ]


def build_training_fixture() -> tuple[Path, Path, Path, Path]:
    test_dir = make_workspace_test_dir()
    image_root = test_dir / "images"
    train_csv = test_dir / "train.csv"
    val_csv = test_dir / "val.csv"
    train_rows = rows_for_split("train", 6)
    val_rows = rows_for_split("val", 4)
    for row_index, row in enumerate(train_rows + val_rows):
        save_image(
            image_root / row["file_name"],
            color=(40 + row_index, 80 + row_index, 120 + row_index),
        )
    pd.DataFrame(train_rows).to_csv(train_csv, index=False)
    pd.DataFrame(val_rows).to_csv(val_csv, index=False)
    return test_dir, image_root, train_csv, val_csv


def tensor_batch(batch_size: int = 2) -> dict[str, Any]:
    return {
        "image": torch.randn(batch_size, 3, 32, 32),
        "label": torch.tensor([0, 1][:batch_size], dtype=torch.int64),
        "image_id": [f"img-{index}" for index in range(batch_size)],
        "file_name": [f"{index}.jpg" for index in range(batch_size)],
        "category_name": ["empty", "deer"][:batch_size],
        "location": [f"loc-{index}" for index in range(batch_size)],
        "seq_id": [f"seq-{index}" for index in range(batch_size)],
    }


def test_one_training_step_changes_trainable_and_preserves_frozen_parameters() -> None:
    model = build_resnet18_classifier(pretrained=False, freeze_backbone=True)
    before = snapshot_named_parameters(model)
    optimizer = build_optimizer(model, learning_rate=0.001, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()

    metrics, gradient_check = train_one_epoch(
        model=model,
        loader=[tensor_batch()],
        criterion=criterion,
        optimizer=optimizer,
        device=torch.device("cpu"),
        image_size=32,
        epoch=1,
        max_batches=1,
        log_interval=1,
    )

    trainable_changed = any(
        parameter.requires_grad
        and not torch.equal(before[name], parameter.detach().cpu())
        for name, parameter in model.named_parameters()
    )
    assert torch.isfinite(torch.tensor(metrics.loss))
    assert gradient_check
    assert trainable_changed
    assert frozen_backbone_unchanged(before, model)


def test_validation_does_not_create_gradients() -> None:
    model = build_resnet18_classifier(pretrained=False, freeze_backbone=True)
    criterion = nn.CrossEntropyLoss()

    metrics = validate_one_epoch(
        model=model,
        loader=[tensor_batch()],
        criterion=criterion,
        device=torch.device("cpu"),
        image_size=32,
        max_batches=1,
    )

    assert metrics.batches == 1
    assert all(parameter.grad is None for parameter in model.parameters())


def test_confusion_matrix_and_metrics_handle_zero_denominators() -> None:
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    predictions = torch.tensor([0, 1, 1, 2], dtype=torch.int64)
    matrix = confusion_matrix_from_predictions(
        predictions=predictions,
        labels=labels,
        num_classes=NUM_CLASSES,
    )
    per_class, macro_f1 = metrics_from_confusion_matrix(matrix)

    assert matrix.shape == (6, 6)
    assert per_class["empty"]["precision"] == pytest.approx(1.0)
    assert per_class["empty"]["recall"] == pytest.approx(0.5)
    assert per_class["deer"]["precision"] == pytest.approx(0.5)
    assert per_class["deer"]["recall"] == pytest.approx(0.5)
    assert per_class["opossum"]["f1"] == 0.0
    expected_empty_f1 = 2 * 1.0 * 0.5 / 1.5
    expected_macro = (expected_empty_f1 + 0.5 + 0.0 + 0.0 + 0.0 + 0.0) / 6.0
    assert macro_f1 == pytest.approx(expected_macro)


def test_max_batch_limits_artifacts_checkpoint_and_smoke_summary() -> None:
    test_dir, image_root, train_csv, val_csv = build_training_fixture()
    try:
        output_dir = test_dir / "training"
        checkpoint_dir = test_dir / "checkpoints"
        summary = train_resnet18(
            TrainingConfig(
                train_csv=train_csv,
                val_csv=val_csv,
                image_root=image_root,
                output_dir=output_dir,
                checkpoint_dir=checkpoint_dir,
                epochs=1,
                batch_size=2,
                image_size=32,
                pretrained=False,
                freeze_backbone=True,
                max_train_batches=1,
                max_val_batches=1,
                run_name="unit_smoke",
            )
        )
        best_checkpoint = checkpoint_dir / "unit_smoke_best.pt"
        last_checkpoint = checkpoint_dir / "unit_smoke_last.pt"
        history_path = output_dir / "unit_smoke" / "history.json"
        summary_path = output_dir / "unit_smoke" / "run_summary.json"
        curves_path = output_dir / "unit_smoke" / "training_curves.png"
        checkpoint = torch.load(best_checkpoint, map_location="cpu")

        assert summary["processed_train_batch_count"] == 1
        assert summary["processed_validation_batch_count"] == 1
        assert summary["smoke_test_mode"] is True
        assert summary["limited_batch_metrics_warning"]
        assert summary["training_validation_passed"] is True
        assert summary["gradient_check_passed"] is True
        assert summary["trainable_parameters_changed"] is True
        assert summary["frozen_backbone_parameters_unchanged"] is True
        assert best_checkpoint.exists()
        assert last_checkpoint.exists()
        assert history_path.exists()
        assert summary_path.exists()
        assert curves_path.exists()
        assert checkpoint["fixed_class_names"] == [
            "empty",
            "deer",
            "coyote",
            "bobcat",
            "bird",
            "opossum",
        ]
        assert checkpoint["class_to_index"]["empty"] == 0
        assert checkpoint["num_classes"] == 6
        assert checkpoint["pretrained"] is False
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_nan_loss_triggers_clear_failure() -> None:
    class NanLoss(nn.Module):
        def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            return logits.sum() * torch.tensor(float("nan"))

    model = build_resnet18_classifier(pretrained=False, freeze_backbone=True)
    optimizer = build_optimizer(model, learning_rate=0.001, weight_decay=0.0)

    with pytest.raises(ValueError, match="training loss is NaN or Inf"):
        train_one_epoch(
            model=model,
            loader=[tensor_batch()],
            criterion=NanLoss(),
            optimizer=optimizer,
            device=torch.device("cpu"),
            image_size=32,
            epoch=1,
            max_batches=1,
            log_interval=1,
        )
