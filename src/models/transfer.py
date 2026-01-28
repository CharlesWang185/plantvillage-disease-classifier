# src/models/transfer.py
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torchvision import models


# -----------------------------
# Helpers for freezing/unfreezing
# -----------------------------
def freeze_backbone(model: nn.Module, keep_classifier_trainable: bool = True) -> None:
    """
    Freeze all parameters (requires_grad=False). Optionally keep the classifier head trainable.
    This is typically used for "phase 1" training (train head only).
    """
    for p in model.parameters():
        p.requires_grad = False

    if keep_classifier_trainable:
        # Common classifier attribute names across torchvision models
        for attr in ("fc", "classifier", "head"):
            if hasattr(model, attr):
                head = getattr(model, attr)
                for p in head.parameters():
                    p.requires_grad = True


def unfreeze_all(model: nn.Module) -> None:
    """Make all parameters trainable."""
    for p in model.parameters():
        p.requires_grad = True


def unfreeze_last_n_children(model: nn.Module, n: int) -> None:
    """
    Unfreeze only the last N "children" modules of the model (high-level blocks).
    Useful for fine-tuning only the deeper layers.

    Example: unfreeze_last_n_children(resnet18, n=2) will unfreeze the last 2 children modules
    (often includes layer4 + fc, depending on model structure).
    """
    children = list(model.children())
    if n <= 0:
        return

    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze last n children
    for child in children[-n:]:
        for p in child.parameters():
            p.requires_grad = True


def count_trainable_params(model: nn.Module) -> int:
    """Return number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -----------------------------
# Model builders
# -----------------------------
def build_resnet18(num_classes: int, pretrained: bool = True, dropout: float = 0.0) -> nn.Module:
    """
    Build a ResNet-18 with a classifier head for num_classes.
    """
    if pretrained:
        try:
            weights = models.ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
        except Exception:
            # Fallback for older torchvision
            model = models.resnet18(pretrained=True)
    else:
        model = models.resnet18(weights=None)

    in_features = model.fc.in_features
    if dropout > 0:
        model.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))
    else:
        model.fc = nn.Linear(in_features, num_classes)

    return model


def build_resnet50(num_classes: int, pretrained: bool = True, dropout: float = 0.0) -> nn.Module:
    """
    Build a ResNet-50 with a classifier head for num_classes.
    """
    if pretrained:
        try:
            weights = models.ResNet50_Weights.DEFAULT
            model = models.resnet50(weights=weights)
        except Exception:
            model = models.resnet50(pretrained=True)
    else:
        model = models.resnet50(weights=None)

    in_features = model.fc.in_features
    if dropout > 0:
        model.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))
    else:
        model.fc = nn.Linear(in_features, num_classes)

    return model


def build_efficientnet_b0(num_classes: int, pretrained: bool = True, dropout: float = 0.2) -> nn.Module:
    """
    Build an EfficientNet-B0 with a classifier head for num_classes.
    """
    if pretrained:
        try:
            weights = models.EfficientNet_B0_Weights.DEFAULT
            model = models.efficientnet_b0(weights=weights)
        except Exception:
            model = models.efficientnet_b0(pretrained=True)
    else:
        model = models.efficientnet_b0(weights=None)

    # torchvision EfficientNet classifier is typically: Sequential(Dropout, Linear)
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        # Replace last Linear
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
    else:
        # Very defensive fallback (unlikely)
        in_features = getattr(model, "classifier").in_features
        model.classifier = nn.Linear(in_features, num_classes)

    return model


def build_model(name: str, num_classes: int, pretrained: bool = True, dropout: float = 0.0) -> nn.Module:
    """
    Convenience factory.
    name examples:
      - "resnet18"
      - "resnet50"
      - "efficientnet_b0"
    """
    name = name.lower().strip()
    if name == "resnet18":
        return build_resnet18(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    if name == "resnet50":
        return build_resnet50(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    if name in {"efficientnet_b0", "effnet_b0"}:
        return build_efficientnet_b0(num_classes=num_classes, pretrained=pretrained, dropout=max(dropout, 0.0))

    raise ValueError(f"Unknown model name: {name}")
