from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from src.cats_dogs.config import ModelConfig


class SimpleCNN(nn.Module):
    def __init__(self, filters: list[int], dropout: float, activation: str) -> None:
        super().__init__()
        act: nn.Module
        if activation == "leaky_relu":
            act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        else:
            act = nn.ReLU(inplace=True)

        blocks = []
        in_channels = 3
        for out_channels in filters:
            blocks.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    act,
                    nn.MaxPool2d(kernel_size=2),
                ]
            )
            in_channels = out_channels

        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters[-1], 128),
            act,
            nn.Dropout(p=dropout),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


class TransferResNet18(nn.Module):
    def __init__(self, pretrained: bool) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 2)

    def freeze_backbone(self) -> None:
        for name, param in self.backbone.named_parameters():
            param.requires_grad = name.startswith("fc")

    def unfreeze_all(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_model(cfg: ModelConfig) -> nn.Module:
    if cfg.type == "cnn":
        return SimpleCNN(cfg.cnn.filters, cfg.cnn.dropout, cfg.cnn.activation)
    return TransferResNet18(pretrained=cfg.transfer.pretrained)
