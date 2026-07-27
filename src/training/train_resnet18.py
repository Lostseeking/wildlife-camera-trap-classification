from __future__ import annotations

# ruff: noqa: E402,I001

import argparse
import copy
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

Path("artifacts/matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(Path("artifacts/matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision  # type: ignore[import-untyped]
from torch import nn
from torch.utils.data import DataLoader

from ..data.caltech_dataloaders import (
    DEFAULT_IMAGE_ROOT,
    DEFAULT_TRAIN_CSV,
    DEFAULT_VAL_CSV,
    build_caltech_dataloaders,
    seed_everything,
)
from ..data.caltech_dataset import CLASS_NAMES, CLASS_TO_INDEX
from ..models.resnet18_classifier import (
    build_resnet18_classifier,
    count_total_parameters,
    count_trainable_parameters,
    trainable_parameter_names,
)

LOGGER = logging.getLogger(__name__)
NUM_CLASSES = len(CLASS_NAMES)
DEFAULT_OUTPUT_ROOT = Path("artifacts/training")
DEFAULT_CHECKPOINT_DIR = Path("artifacts/checkpoints")


@dataclass(frozen=True)
class TrainingConfig:
    train_csv: Path = DEFAULT_TRAIN_CSV
    val_csv: Path = DEFAULT_VAL_CSV
    image_root: Path = DEFAULT_IMAGE_ROOT
    output_dir: Path = DEFAULT_OUTPUT_ROOT
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR
    epochs: int = 5
    batch_size: int = 16
    num_workers: int = 0
    image_size: int = 224
    seed: int = 42
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    pretrained: bool = True
    freeze_backbone: bool = True
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    run_name: str = "resnet18_head"
    log_interval: int = 10


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    accuracy: float
    samples: int
    batches: int


@dataclass(frozen=True)
class ValidationMetrics(EpochMetrics):
    confusion_matrix: list[list[int]]
    per_class: dict[str, dict[str, float]]
    macro_f1: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a ResNet18 transfer-learning baseline for Caltech MVP."
    )
    parser.add_argument("--train-csv", default=DEFAULT_TRAIN_CSV, type=Path)
    parser.add_argument("--val-csv", default=DEFAULT_VAL_CSV, type=Path)
    parser.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT, type=Path)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_ROOT, type=Path)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR, type=Path)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--image-size", default=224, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--learning-rate", default=0.001, type=float)
    parser.add_argument("--weight-decay", default=0.0001, type=float)
    parser.add_argument("--pretrained", dest="pretrained", action="store_true")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.set_defaults(pretrained=True)
    parser.add_argument(
        "--freeze-backbone", dest="freeze_backbone", action="store_true"
    )
    parser.add_argument(
        "--no-freeze-backbone", dest="freeze_backbone", action="store_false"
    )
    parser.set_defaults(freeze_backbone=True)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--run-name", default="resnet18_head")
    parser.add_argument("--log-interval", default=10, type=int)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        image_root=args.image_root,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        seed=args.seed,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        run_name=args.run_name,
        log_interval=args.log_interval,
    )


def validate_config(config: TrainingConfig) -> None:
    if config.epochs < 1:
        raise ValueError(f"epochs must be at least 1: {config.epochs}")
    if config.batch_size < 1:
        raise ValueError(f"batch-size must be at least 1: {config.batch_size}")
    if config.num_workers < 0:
        raise ValueError(f"num-workers must be non-negative: {config.num_workers}")
    if config.image_size < 1:
        raise ValueError(f"image-size must be at least 1: {config.image_size}")
    if config.learning_rate <= 0:
        raise ValueError("learning-rate must be greater than zero")
    if config.weight_decay < 0:
        raise ValueError("weight-decay must be non-negative")
    if config.max_train_batches is not None and config.max_train_batches < 1:
        raise ValueError("max-train-batches must be at least 1 when provided")
    if config.max_val_batches is not None and config.max_val_batches < 1:
        raise ValueError("max-val-batches must be at least 1 when provided")
    if config.log_interval < 1:
        raise ValueError("log-interval must be at least 1")


def select_device() -> torch.device:
    """Use CPU for reproducible local milestone training."""
    return torch.device("cpu")


def finite_scalar(value: torch.Tensor, name: str) -> None:
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} is NaN or Inf")


def validate_batch_tensors(
    batch: dict[str, Any], image_size: int, num_classes: int
) -> tuple[torch.Tensor, torch.Tensor]:
    images = batch["image"]
    labels = batch["label"]
    if not isinstance(images, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise TypeError("DataLoader batch must contain tensor image and label values")
    if images.ndim != 4 or tuple(images.shape[1:]) != (3, image_size, image_size):
        raise ValueError(
            "Image batch shape must be [batch, 3, image_size, image_size]; "
            f"got {list(images.shape)}"
        )
    if labels.ndim != 1:
        raise ValueError(f"Label batch shape must be [batch]; got {list(labels.shape)}")
    if labels.dtype != torch.int64:
        raise TypeError(f"Label dtype must be torch.int64; got {labels.dtype}")
    if labels.numel() != images.shape[0]:
        raise ValueError("Image and label batch sizes do not match")
    if labels.numel() and (
        int(labels.min().item()) < 0 or int(labels.max().item()) >= num_classes
    ):
        raise ValueError("Label value is outside the fixed class index range")
    finite_scalar(images, "image batch")
    return images, labels


def validate_logits(logits: torch.Tensor, batch_size: int, num_classes: int) -> None:
    if tuple(logits.shape) != (batch_size, num_classes):
        raise ValueError(
            "Logits shape must be "
            f"[{batch_size}, {num_classes}], got {list(logits.shape)}"
        )
    finite_scalar(logits, "logits")


def build_optimizer(
    model: nn.Module, learning_rate: float, weight_decay: float
) -> torch.optim.Optimizer:
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("No trainable parameters available for optimizer")
    return torch.optim.AdamW(
        trainable_parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def snapshot_named_parameters(model: nn.Module) -> dict[str, torch.Tensor]:
    """Clone current model parameters on CPU for later change checks."""
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }


def any_trainable_parameter_changed(
    before: dict[str, torch.Tensor], model: nn.Module
) -> bool:
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and not torch.equal(
            before[name], parameter.detach().cpu()
        ):
            return True
    return False


def frozen_backbone_unchanged(
    before: dict[str, torch.Tensor], model: nn.Module
) -> bool:
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad and not torch.equal(
            before[name], parameter.detach().cpu()
        ):
            return False
    return True


def finite_gradient_exists(model: nn.Module) -> bool:
    """Return true when at least one trainable parameter has a finite gradient."""
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.grad is not None:
            if torch.isfinite(parameter.grad).all().item():
                return True
    return False


def confusion_matrix_from_predictions(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = NUM_CLASSES,
) -> torch.Tensor:
    """Compute a [num_classes, num_classes] matrix with rows=true, cols=predicted."""
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for true_label, predicted_label in zip(
        labels.view(-1), predictions.view(-1), strict=True
    ):
        matrix[int(true_label.item()), int(predicted_label.item())] += 1
    return matrix


def metrics_from_confusion_matrix(
    confusion_matrix: torch.Tensor,
) -> tuple[dict[str, dict[str, float]], float]:
    """Compute per-class precision, recall, F1 and macro-F1.

    Zero denominators use a zero-score policy so limited smoke-test batches with
    absent classes remain valid and explicit.
    """
    per_class: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []
    for index, class_name in enumerate(CLASS_NAMES):
        true_positive = float(confusion_matrix[index, index].item())
        predicted_positive = float(confusion_matrix[:, index].sum().item())
        actual_positive = float(confusion_matrix[index, :].sum().item())
        precision = (
            true_positive / predicted_positive if predicted_positive > 0 else 0.0
        )
        recall = true_positive / actual_positive if actual_positive > 0 else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": actual_positive,
        }
        f1_values.append(f1)
    return per_class, float(sum(f1_values) / len(f1_values))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    image_size: int,
    epoch: int,
    max_batches: int | None,
    log_interval: int,
) -> tuple[EpochMetrics, bool]:
    model.train()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    processed_batches = 0
    gradient_check_passed = False

    for batch_index, batch in enumerate(loader, start=1):
        images, labels = validate_batch_tensors(
            batch=batch, image_size=image_size, num_classes=NUM_CLASSES
        )
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        validate_logits(logits, batch_size=labels.shape[0], num_classes=NUM_CLASSES)
        loss = criterion(logits, labels)
        finite_scalar(loss, "training loss")
        loss.backward()
        if batch_index == 1:
            gradient_check_passed = finite_gradient_exists(model)
            if not gradient_check_passed:
                raise ValueError(
                    "No trainable parameter received a finite gradient on "
                    "the first batch"
                )
        optimizer.step()

        predictions = logits.argmax(dim=1)
        batch_size = int(labels.shape[0])
        total_loss += float(loss.item()) * batch_size
        correct += int((predictions == labels).sum().item())
        total_samples += batch_size
        processed_batches += 1
        if batch_index % log_interval == 0:
            LOGGER.info(
                "epoch=%s train_batch=%s loss=%.6f",
                epoch,
                batch_index,
                float(loss.item()),
            )
        if max_batches is not None and processed_batches >= max_batches:
            break

    if total_samples == 0:
        raise ValueError("Training processed zero samples")
    return (
        EpochMetrics(
            loss=total_loss / total_samples,
            accuracy=correct / total_samples,
            samples=total_samples,
            batches=processed_batches,
        ),
        gradient_check_passed,
    )


def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    criterion: nn.Module,
    device: torch.device,
    image_size: int,
    max_batches: int | None,
) -> ValidationMetrics:
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    processed_batches = 0
    confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)

    with torch.inference_mode():
        for batch in loader:
            images, labels = validate_batch_tensors(
                batch=batch, image_size=image_size, num_classes=NUM_CLASSES
            )
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            validate_logits(logits, batch_size=labels.shape[0], num_classes=NUM_CLASSES)
            loss = criterion(logits, labels)
            finite_scalar(loss, "validation loss")

            predictions = logits.argmax(dim=1)
            batch_size = int(labels.shape[0])
            total_loss += float(loss.item()) * batch_size
            correct += int((predictions == labels).sum().item())
            total_samples += batch_size
            processed_batches += 1
            confusion += confusion_matrix_from_predictions(
                predictions=predictions.cpu(),
                labels=labels.cpu(),
                num_classes=NUM_CLASSES,
            )
            if max_batches is not None and processed_batches >= max_batches:
                break

    if total_samples == 0:
        raise ValueError("Validation processed zero samples")
    per_class, macro_f1 = metrics_from_confusion_matrix(confusion)
    if not math.isfinite(macro_f1):
        raise ValueError("Validation macro-F1 is not finite")
    return ValidationMetrics(
        loss=total_loss / total_samples,
        accuracy=correct / total_samples,
        samples=total_samples,
        batches=processed_batches,
        confusion_matrix=confusion.tolist(),
        per_class=per_class,
        macro_f1=macro_f1,
    )


def safe_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_validation_macro_f1: float,
    validation_metrics: ValidationMetrics,
    config: TrainingConfig,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_validation_macro_f1": best_validation_macro_f1,
        "current_validation_metrics": asdict(validation_metrics),
        "fixed_class_names": list(CLASS_NAMES),
        "class_to_index": dict(CLASS_TO_INDEX),
        "model_name": "resnet18",
        "num_classes": NUM_CLASSES,
        "pretrained": config.pretrained,
        "freeze_backbone": config.freeze_backbone,
        "image_size": config.image_size,
        "seed": config.seed,
        "training_configuration": serializable_config(config),
        "torch_version": str(torch.__version__),
        "torchvision_version": str(torchvision.__version__),
    }
    safe_torch_save(checkpoint, path)


def serializable_config(config: TrainingConfig) -> dict[str, Any]:
    payload = asdict(config)
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in payload.items()
    }


def save_history(history: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def save_training_curves(history: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [record["epoch"] for record in history]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), squeeze=False)
    loss_axis = axes[0][0]
    metric_axis = axes[0][1]
    loss_axis.plot(epochs, [record["train_loss"] for record in history], marker="o")
    loss_axis.plot(
        epochs, [record["validation_loss"] for record in history], marker="o"
    )
    loss_axis.set_title("Loss")
    loss_axis.set_xlabel("Epoch")
    loss_axis.legend(["train", "val"])
    metric_axis.plot(
        epochs, [record["train_accuracy"] for record in history], marker="o"
    )
    metric_axis.plot(
        epochs, [record["validation_accuracy"] for record in history], marker="o"
    )
    metric_axis.plot(
        epochs, [record["validation_macro_f1"] for record in history], marker="o"
    )
    metric_axis.set_title("Metrics")
    metric_axis.set_xlabel("Epoch")
    metric_axis.legend(["train acc", "val acc", "val macro-F1"])
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def write_run_summary(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def train_resnet18(config: TrainingConfig) -> dict[str, Any]:
    validate_config(config)
    seed_everything(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    start_time = datetime.now(UTC)
    device = select_device()
    run_output_dir = config.output_dir / config.run_name
    best_checkpoint_path = config.checkpoint_dir / f"{config.run_name}_best.pt"
    last_checkpoint_path = config.checkpoint_dir / f"{config.run_name}_last.pt"
    history_path = run_output_dir / "history.json"
    summary_path = run_output_dir / "run_summary.json"
    curves_path = run_output_dir / "training_curves.png"

    loaders = build_caltech_dataloaders(
        train_csv=config.train_csv,
        val_csv=config.val_csv,
        test_csv=config.val_csv,
        image_root=config.image_root,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        image_size=config.image_size,
        seed=config.seed,
        pin_memory=False,
    )
    model = build_resnet18_classifier(
        num_classes=NUM_CLASSES,
        pretrained=config.pretrained,
        freeze_backbone=config.freeze_backbone,
    ).to(device)
    total_parameters = count_total_parameters(model)
    trainable_parameters = count_trainable_parameters(model)
    trainable_percentage = (
        trainable_parameters / total_parameters if total_parameters else 0.0
    )
    optimizer = build_optimizer(
        model=model,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    history: list[dict[str, Any]] = []
    best_macro_f1 = -1.0
    best_epoch = 0
    gradient_check_passed = False
    trainable_parameters_changed = False
    frozen_backbone_remained_unchanged = True
    initial_parameter_snapshot = snapshot_named_parameters(model)
    smoke_test_mode = (
        config.max_train_batches is not None or config.max_val_batches is not None
    )
    limited_warning = (
        "Metrics were computed on limited batches and are not final model performance."
        if smoke_test_mode
        else ""
    )

    LOGGER.info("device=%s", device.type)
    LOGGER.info(
        "pretrained=%s freeze_backbone=%s",
        config.pretrained,
        config.freeze_backbone,
    )
    LOGGER.info(
        "parameters total=%s trainable=%s percentage=%.6f",
        total_parameters,
        trainable_parameters,
        trainable_percentage,
    )

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.monotonic()
        train_metrics, epoch_gradient_check = train_one_epoch(
            model=model,
            loader=loaders.train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            image_size=config.image_size,
            epoch=epoch,
            max_batches=config.max_train_batches,
            log_interval=config.log_interval,
        )
        gradient_check_passed = gradient_check_passed or epoch_gradient_check
        validation_metrics = validate_one_epoch(
            model=model,
            loader=loaders.val_loader,
            criterion=criterion,
            device=device,
            image_size=config.image_size,
            max_batches=config.max_val_batches,
        )
        if validation_metrics.macro_f1 > best_macro_f1:
            best_macro_f1 = validation_metrics.macro_f1
            best_epoch = epoch
            save_checkpoint(
                path=best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_validation_macro_f1=best_macro_f1,
                validation_metrics=validation_metrics,
                config=config,
            )
        save_checkpoint(
            path=last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_validation_macro_f1=best_macro_f1,
            validation_metrics=validation_metrics,
            config=config,
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_metrics.loss,
            "train_accuracy": train_metrics.accuracy,
            "validation_loss": validation_metrics.loss,
            "validation_accuracy": validation_metrics.accuracy,
            "validation_macro_f1": validation_metrics.macro_f1,
            "per_class_validation_metrics": validation_metrics.per_class,
            "epoch_duration_seconds": time.monotonic() - epoch_start,
            "train_batches_processed": train_metrics.batches,
            "validation_batches_processed": validation_metrics.batches,
        }
        history.append(epoch_record)
        save_history(history, history_path)
        LOGGER.info(
            "epoch=%s train_loss=%.6f train_acc=%.4f val_loss=%.6f "
            "val_acc=%.4f val_macro_f1=%.4f",
            epoch,
            train_metrics.loss,
            train_metrics.accuracy,
            validation_metrics.loss,
            validation_metrics.accuracy,
            validation_metrics.macro_f1,
        )

    save_training_curves(history, curves_path)
    trainable_parameters_changed = any_trainable_parameter_changed(
        initial_parameter_snapshot, model
    )
    frozen_backbone_remained_unchanged = frozen_backbone_unchanged(
        initial_parameter_snapshot, model
    )
    final_record = history[-1]
    end_time = datetime.now(UTC)
    summary: dict[str, Any] = {
        "configuration": serializable_config(config),
        "run_name": config.run_name,
        "start_timestamp": start_time.isoformat(),
        "end_timestamp": end_time.isoformat(),
        "device": device.type,
        "torch_version": str(torch.__version__),
        "torchvision_version": str(torchvision.__version__),
        "cuda_available": torch.cuda.is_available(),
        "pretrained": config.pretrained,
        "freeze_backbone": config.freeze_backbone,
        "total_parameter_count": total_parameters,
        "trainable_parameter_count": trainable_parameters,
        "trainable_parameter_percentage": trainable_percentage,
        "trainable_parameter_names": trainable_parameter_names(model),
        "train_dataset_size": len(loaders.train_dataset),
        "validation_dataset_size": len(loaders.val_dataset),
        "configured_train_batch_count": len(loaders.train_loader),
        "configured_validation_batch_count": len(loaders.val_loader),
        "processed_train_batch_count": final_record["train_batches_processed"],
        "processed_validation_batch_count": final_record[
            "validation_batches_processed"
        ],
        "image_size": config.image_size,
        "batch_size": config.batch_size,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_macro_f1,
        "best_checkpoint_path": str(best_checkpoint_path),
        "last_checkpoint_path": str(last_checkpoint_path),
        "history_path": str(history_path),
        "training_curves_path": str(curves_path),
        "smoke_test_mode": smoke_test_mode,
        "limited_batch_metrics_warning": limited_warning,
        "gradient_check_passed": gradient_check_passed,
        "trainable_parameters_changed": trainable_parameters_changed,
        "frozen_backbone_parameters_unchanged": frozen_backbone_remained_unchanged,
        "output_logits_shape": [config.batch_size, NUM_CLASSES],
        "final_epoch_metrics": copy.deepcopy(final_record),
        "training_validation_passed": bool(
            gradient_check_passed
            and trainable_parameters_changed
            and frozen_backbone_remained_unchanged
            and best_checkpoint_path.exists()
            and last_checkpoint_path.exists()
            and history_path.exists()
            and curves_path.exists()
        ),
    }
    write_run_summary(summary, summary_path)
    if not summary["training_validation_passed"]:
        raise ValueError("Training smoke validation did not pass")
    LOGGER.info("best checkpoint: %s", best_checkpoint_path)
    LOGGER.info("last checkpoint: %s", last_checkpoint_path)
    LOGGER.info("history: %s", history_path)
    LOGGER.info("run summary: %s", summary_path)
    LOGGER.info("training curves: %s", curves_path)
    LOGGER.info("training_validation_passed=true")
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        summary = train_resnet18(config_from_args(parse_args()))
    except Exception as exc:
        LOGGER.error("%s", exc)
        sys.exit(1)
    LOGGER.info(
        "final train_loss=%.6f val_loss=%.6f val_macro_f1=%.6f",
        summary["final_epoch_metrics"]["train_loss"],
        summary["final_epoch_metrics"]["validation_loss"],
        summary["final_epoch_metrics"]["validation_macro_f1"],
    )


if __name__ == "__main__":
    main()
