from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml


@dataclass(slots=True)
class PathsConfig:
    dataset_dir: Path
    report_dir: Path
    model_dir: Path


@dataclass(slots=True)
class DataConfig:
    class_names: list[str]
    img_size: int
    val_size: float
    test_size: float
    random_seed: int
    num_workers: int


@dataclass(slots=True)
class AugmentationConfig:
    horizontal_flip: bool
    rotation_degrees: int
    color_jitter: bool


@dataclass(slots=True)
class CNNConfig:
    filters: list[int]
    activation: Literal["relu", "leaky_relu"]
    dropout: float


@dataclass(slots=True)
class TransferConfig:
    backbone: Literal["resnet18"]
    freeze_backbone_epochs: int
    pretrained: bool


@dataclass(slots=True)
class ModelConfig:
    type: Literal["cnn", "transfer"]
    cnn: CNNConfig
    transfer: TransferConfig


@dataclass(slots=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: Literal["sgd", "adam"]
    weight_decay: float
    momentum: float
    checkpoint_name: str


@dataclass(slots=True)
class CatsDogsConfig:
    project_name: str
    experiment_name: str
    paths: PathsConfig
    data: DataConfig
    augmentation: AugmentationConfig
    model: ModelConfig
    training: TrainingConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CatsDogsConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)

        return cls(
            project_name=payload["project_name"],
            experiment_name=payload["experiment_name"],
            paths=PathsConfig(
                dataset_dir=Path(payload["paths"]["dataset_dir"]),
                report_dir=Path(payload["paths"]["report_dir"]),
                model_dir=Path(payload["paths"]["model_dir"]),
            ),
            data=DataConfig(
                class_names=payload["data"]["class_names"],
                img_size=int(payload["data"]["img_size"]),
                val_size=float(payload["data"]["val_size"]),
                test_size=float(payload["data"]["test_size"]),
                random_seed=int(payload["data"]["random_seed"]),
                num_workers=int(payload["data"]["num_workers"]),
            ),
            augmentation=AugmentationConfig(
                horizontal_flip=bool(payload["augmentation"]["horizontal_flip"]),
                rotation_degrees=int(payload["augmentation"]["rotation_degrees"]),
                color_jitter=bool(payload["augmentation"]["color_jitter"]),
            ),
            model=ModelConfig(
                type=payload["model"]["type"],
                cnn=CNNConfig(
                    filters=[int(v) for v in payload["model"]["cnn"]["filters"]],
                    activation=payload["model"]["cnn"]["activation"],
                    dropout=float(payload["model"]["cnn"]["dropout"]),
                ),
                transfer=TransferConfig(
                    backbone=payload["model"]["transfer"]["backbone"],
                    freeze_backbone_epochs=int(
                        payload["model"]["transfer"]["freeze_backbone_epochs"]
                    ),
                    pretrained=bool(payload["model"]["transfer"]["pretrained"]),
                ),
            ),
            training=TrainingConfig(
                epochs=int(payload["training"]["epochs"]),
                batch_size=int(payload["training"]["batch_size"]),
                learning_rate=float(payload["training"]["learning_rate"]),
                optimizer=payload["training"]["optimizer"],
                weight_decay=float(payload["training"]["weight_decay"]),
                momentum=float(payload["training"]["momentum"]),
                checkpoint_name=payload["training"]["checkpoint_name"],
            ),
        )
