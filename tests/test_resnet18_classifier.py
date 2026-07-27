from __future__ import annotations

import pytest
import torch
from src.models.resnet18_classifier import (
    build_resnet18_classifier,
    count_total_parameters,
    count_trainable_parameters,
    trainable_parameter_names,
    verify_final_output_dimension,
)
from torch import nn


def test_model_output_shape_and_final_layer_features() -> None:
    model = build_resnet18_classifier(num_classes=6, pretrained=False)
    model.eval()

    with torch.inference_mode():
        logits = model(torch.randn(2, 3, 64, 64))

    assert logits.shape == (2, 6)
    assert isinstance(model.fc, nn.Linear)
    assert model.fc.out_features == 6
    assert verify_final_output_dimension(model, 6)


def test_pretrained_false_uses_no_network_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.models.resnet18_classifier as module

    captured_weights = []
    original_resnet18 = module.resnet18

    def spy_resnet18(*args: object, **kwargs: object) -> nn.Module:
        captured_weights.append(kwargs.get("weights"))
        return original_resnet18(weights=None)

    monkeypatch.setattr(module, "resnet18", spy_resnet18)

    _model = build_resnet18_classifier(num_classes=6, pretrained=False)

    assert captured_weights == [None]


def test_freeze_backbone_true_only_keeps_fc_trainable() -> None:
    model = build_resnet18_classifier(
        num_classes=6,
        pretrained=False,
        freeze_backbone=True,
    )

    trainable_names = trainable_parameter_names(model)

    assert trainable_names == ["fc.weight", "fc.bias"]
    assert model.fc.weight.requires_grad
    assert model.fc.bias.requires_grad
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith("fc.")
    )


def test_freeze_backbone_false_leaves_all_parameters_trainable() -> None:
    model = build_resnet18_classifier(
        num_classes=6,
        pretrained=False,
        freeze_backbone=False,
    )

    assert all(parameter.requires_grad for parameter in model.parameters())


def test_parameter_count_helpers_are_sensible() -> None:
    frozen_model = build_resnet18_classifier(
        num_classes=6,
        pretrained=False,
        freeze_backbone=True,
    )
    unfrozen_model = build_resnet18_classifier(
        num_classes=6,
        pretrained=False,
        freeze_backbone=False,
    )

    assert count_total_parameters(frozen_model) > 0
    assert count_trainable_parameters(frozen_model) > 0
    assert count_trainable_parameters(frozen_model) < count_total_parameters(
        frozen_model
    )
    assert count_trainable_parameters(unfrozen_model) == count_total_parameters(
        unfrozen_model
    )


def test_model_returns_raw_logits_and_cross_entropy_accepts_them() -> None:
    model = build_resnet18_classifier(num_classes=6, pretrained=False)
    logits = model(torch.randn(4, 3, 64, 64))
    labels = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    probabilities_sum = logits.sum(dim=1)
    loss = nn.CrossEntropyLoss()(logits, labels)

    assert not torch.allclose(probabilities_sum, torch.ones_like(probabilities_sum))
    assert torch.isfinite(loss)
