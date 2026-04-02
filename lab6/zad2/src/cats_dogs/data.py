from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.cats_dogs.config import AugmentationConfig, DataConfig

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class CatsDogsDataset(Dataset):
    def __init__(self, files: list[Path], labels: list[int], transform: transforms.Compose) -> None:
        self.files = files
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple:
        image = Image.open(self.files[index]).convert("RGB")
        return self.transform(image), self.labels[index], self.files[index].name


@dataclass(slots=True)
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader


def _build_transforms(img_size: int, augmentation: AugmentationConfig) -> tuple:
    train_items = [transforms.Resize((img_size, img_size))]
    if augmentation.horizontal_flip:
        train_items.append(transforms.RandomHorizontalFlip())
    if augmentation.rotation_degrees > 0:
        train_items.append(transforms.RandomRotation(augmentation.rotation_degrees))
    if augmentation.color_jitter:
        train_items.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))

    train_items.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    eval_items = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]

    return transforms.Compose(train_items), transforms.Compose(eval_items)


def _extract_label(path: Path) -> int:
    prefix = path.name.split(".")[0].lower()
    if prefix == "cat":
        return 0
    if prefix == "dog":
        return 1
    raise ValueError(f"Unsupported filename for class extraction: {path.name}")


def build_dataloaders(
    dataset_dir: Path,
    data_cfg: DataConfig,
    aug_cfg: AugmentationConfig,
    batch_size: int,
) -> DataBundle:
    files = sorted([p for p in dataset_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    labels = [_extract_label(path) for path in files]

    train_files, tmp_files, train_labels, tmp_labels = train_test_split(
        files,
        labels,
        test_size=data_cfg.val_size + data_cfg.test_size,
        stratify=labels,
        random_state=data_cfg.random_seed,
    )

    relative_test_size = data_cfg.test_size / (data_cfg.val_size + data_cfg.test_size)
    val_files, test_files, val_labels, test_labels = train_test_split(
        tmp_files,
        tmp_labels,
        test_size=relative_test_size,
        stratify=tmp_labels,
        random_state=data_cfg.random_seed,
    )

    train_tf, eval_tf = _build_transforms(data_cfg.img_size, aug_cfg)

    train_ds = CatsDogsDataset(train_files, train_labels, train_tf)
    val_ds = CatsDogsDataset(val_files, val_labels, eval_tf)
    test_ds = CatsDogsDataset(test_files, test_labels, eval_tf)

    return DataBundle(
        train_loader=DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=data_cfg.num_workers,
            pin_memory=True,
        ),
        val_loader=DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=data_cfg.num_workers,
            pin_memory=True,
        ),
        test_loader=DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=data_cfg.num_workers,
            pin_memory=True,
        ),
    )
