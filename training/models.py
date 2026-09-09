"""Factory for the six ImageNet-pretrained RobustDRNet benchmarks."""

from __future__ import annotations

from torch import nn
from torchvision import models


SUPPORTED_MODELS = (
    "resnet34",
    "resnet50",
    "densenet121",
    "efficientnet_b7",
    "convnext_tiny",
    "vit_b_16",
)


def build_model(
    name: str,
    *,
    num_classes: int = 5,
    pretrained: bool = True,
) -> nn.Module:
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model {name!r}; choose from {SUPPORTED_MODELS}")

    weights = "DEFAULT" if pretrained else None
    if name == "resnet34":
        model = models.resnet34(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "resnet50":
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "densenet121":
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif name == "efficientnet_b7":
        model = models.efficientnet_b7(weights=weights)
        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features,
            num_classes,
        )
    elif name == "convnext_tiny":
        model = models.convnext_tiny(weights=weights)
        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features,
            num_classes,
        )
    else:
        model = models.vit_b_16(weights=weights)
        model.heads.head = nn.Linear(
            model.heads.head.in_features,
            num_classes,
        )
    return model
