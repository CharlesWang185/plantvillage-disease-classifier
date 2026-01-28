# src/models/cnn_baseline.py
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    A small reusable building block:
      Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        pool_kernel: int = 2,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_kernel),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNNBaseline(nn.Module):
    """
    A simple CNN baseline for image classification.

    Input:  (B, 3, H, W)   (e.g. 224x224)
    Output: (B, num_classes) logits

    Notes:
    - No Softmax here. CrossEntropyLoss expects raw logits.
    - GlobalAveragePooling makes it robust to different H/W.
    """

    def __init__(self, num_classes: int, dropout: float = 0.3) -> None:
        super().__init__()

        # Feature extractor
        self.features = nn.Sequential(
            ConvBlock(3, 32),    # -> (B, 32, H/2, W/2)
            ConvBlock(32, 64),   # -> (B, 64, H/4, W/4)
            ConvBlock(64, 128),  # -> (B, 128, H/8, W/8)
            ConvBlock(128, 256), # -> (B, 256, H/16, W/16)
        )

        # Global average pooling: (B, C, h, w) -> (B, C, 1, 1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),                 # (B, 256)
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),  # (B, num_classes)
        )

        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x
