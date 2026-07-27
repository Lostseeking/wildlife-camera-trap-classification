from __future__ import annotations

from collections.abc import Iterable
from typing import cast

from torch import nn
from torchvision.models import (  # type: ignore[import-untyped]
    ResNet18_Weights,
    resnet18,
)


def freeze_parameters(parameters: Iterable[nn.Parameter]) -> None:
    """Freeze a collection of model parameters in-place."""
    for parameter in parameters:
        parameter.requires_grad = False


def unfreeze_parameters(parameters: Iterable[nn.Parameter]) -> None:
    """Unfreeze a collection of model parameters in-place."""
    for parameter in parameters:
        parameter.requires_grad = True


def freeze_backbone_parameters(model: nn.Module) -> None:
    """Freeze all parameters except the final ResNet classification head."""
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("fc.")


def unfreeze_all_parameters(model: nn.Module) -> None:
    """Make every model parameter trainable."""
    unfreeze_parameters(model.parameters())


def count_total_parameters(model: nn.Module) -> int:
    """Return the total number of scalar parameters."""
    return sum(parameter.numel() for parameter in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of scalar parameters with gradients enabled."""
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def trainable_parameter_names(model: nn.Module) -> list[str]:
    """Return names of parameters that will be optimized."""
    return [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]


def verify_final_output_dimension(model: nn.Module, num_classes: int) -> bool:
    """Verify that model.fc is a Linear layer with the requested output size."""
    final_layer = getattr(model, "fc", None)
    return (
        isinstance(final_layer, nn.Linear)
        and final_layer.out_features == num_classes
    )


def build_resnet18_classifier(
    num_classes: int = 6,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    """Build an ImageNet ResNet18 classifier with a replaced final layer.

    Pretrained requests are never silently downgraded to random initialization.
    If torchvision cannot load the requested weights, the original exception is
    preserved as context in a RuntimeError.
    """
    if num_classes < 1:
        raise ValueError(f"num_classes must be at least 1: {num_classes}")

    try:
        model = cast(
            nn.Module,
            resnet18(
                weights=ResNet18_Weights.DEFAULT if pretrained else None,
            ),
        )
    except Exception as exc:
        if pretrained:
            raise RuntimeError(
                "pretrained=True was requested, but torchvision could not load "
                "the official ResNet18 pretrained weights. Check network access "
                "or the local torch weights cache."
            ) from exc
        raise

    if not isinstance(model.fc, nn.Linear):
        raise TypeError("Expected torchvision ResNet18 model.fc to be torch.nn.Linear")
    original_in_features = model.fc.in_features
    model.fc = nn.Linear(original_in_features, num_classes)

    if freeze_backbone:
        freeze_backbone_parameters(model)
    else:
        unfreeze_all_parameters(model)

    if count_trainable_parameters(model) < 1:
        raise ValueError("Model has no trainable parameters after configuration")
    if not verify_final_output_dimension(model, num_classes=num_classes):
        raise ValueError(f"Model final output dimension is not {num_classes}")
    return model
