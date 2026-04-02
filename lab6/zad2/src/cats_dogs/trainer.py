from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from torch import nn
from torch.optim import Adam, SGD

from src.cats_dogs.config import CatsDogsConfig
from src.cats_dogs.data import DataBundle, build_dataloaders
from src.cats_dogs.models import TransferResNet18, build_model


def _ensure_dirs(cfg: CatsDogsConfig) -> None:
    cfg.paths.report_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.model_dir.mkdir(parents=True, exist_ok=True)


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_optimizer(cfg: CatsDogsConfig, model: nn.Module) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.training.optimizer == "sgd":
        return SGD(
            params,
            lr=cfg.training.learning_rate,
            momentum=cfg.training.momentum,
            weight_decay=cfg.training.weight_decay,
        )
    return Adam(params, lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)


def _run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    avg_acc = accuracy_score(all_labels, all_preds)
    return avg_loss, avg_acc


def _collect_predictions(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int], list[str]]:
    model.eval()
    preds: list[int] = []
    labels: list[int] = []
    names: list[str] = []

    with torch.no_grad():
        for images, batch_labels, batch_names in loader:
            images = images.to(device)
            logits = model(images)
            batch_preds = torch.argmax(logits, dim=1).cpu().tolist()
            preds.extend(batch_preds)
            labels.extend(batch_labels.tolist())
            names.extend(batch_names)

    return preds, labels, names


def _plot_curves(history: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=160)
    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["epoch"], history["train_acc"], label="train")
    axes[1].plot(history["epoch"], history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_confusion_matrix(cm: np.ndarray, labels: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=160)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_misclassified_grid(
    names: list[str],
    y_true: list[int],
    y_pred: list[int],
    cfg: CatsDogsConfig,
    out_path: Path,
) -> int:
    mistakes = [(n, t, p) for n, t, p in zip(names, y_true, y_pred) if t != p]
    if not mistakes:
        return 0

    rows, cols = 3, 4
    count = min(rows * cols, len(mistakes))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 9), dpi=140)

    for idx, ax in enumerate(axes.flat):
        ax.axis("off")
        if idx >= count:
            continue

        name, true_idx, pred_idx = mistakes[idx]
        img = plt.imread(cfg.paths.dataset_dir / name)
        ax.imshow(img)
        ax.set_title(
            f"true={cfg.data.class_names[true_idx]}\npred={cfg.data.class_names[pred_idx]}",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return len(mistakes)


def run_experiment(cfg: CatsDogsConfig) -> dict:
    _ensure_dirs(cfg)
    _set_seed(cfg.data.random_seed)
    device = _select_device()

    data_bundle: DataBundle = build_dataloaders(
        dataset_dir=cfg.paths.dataset_dir,
        data_cfg=cfg.data,
        aug_cfg=cfg.augmentation,
        batch_size=cfg.training.batch_size,
    )

    model = build_model(cfg.model).to(device)
    if isinstance(model, TransferResNet18):
        model.freeze_backbone()

    criterion = nn.CrossEntropyLoss()
    optimizer = _build_optimizer(cfg, model)

    checkpoint_path = cfg.paths.model_dir / cfg.training.checkpoint_name
    history_rows: list[dict] = []
    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(1, cfg.training.epochs + 1):
        if isinstance(model, TransferResNet18) and epoch == cfg.model.transfer.freeze_backbone_epochs + 1:
            model.unfreeze_all()
            optimizer = _build_optimizer(cfg, model)

        train_loss, train_acc = _run_epoch(model, data_bundle.train_loader, criterion, device, optimizer)
        val_loss, val_acc = _run_epoch(model, data_bundle.val_loader, criterion, device, optimizer=None)

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": asdict(cfg),
                },
                checkpoint_path,
            )

    # Local checkpoint contains config metadata, so load full pickle payload.
    best_state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best_state["model_state_dict"])

    history_df = pd.DataFrame(history_rows)
    history_path = cfg.paths.report_dir / "training_history.csv"
    curves_path = cfg.paths.report_dir / "learning_curves.png"
    history_df.to_csv(history_path, index=False)
    _plot_curves(history_df, curves_path)

    y_pred, y_true, names = _collect_predictions(model, data_bundle.test_loader, device)
    test_acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    confusion_path = cfg.paths.report_dir / "confusion_matrix.png"
    _plot_confusion_matrix(cm, cfg.data.class_names, confusion_path)

    mistakes_path = cfg.paths.report_dir / "misclassified_examples.png"
    mistake_count = _plot_misclassified_grid(names, y_true, y_pred, cfg, mistakes_path)

    preds_df = pd.DataFrame({
        "filename": names,
        "y_true": y_true,
        "y_pred": y_pred,
        "true_label": [cfg.data.class_names[i] for i in y_true],
        "pred_label": [cfg.data.class_names[i] for i in y_pred],
    })
    preds_df.to_csv(cfg.paths.report_dir / "test_predictions.csv", index=False)

    summary = {
        "project_name": cfg.project_name,
        "experiment_name": cfg.experiment_name,
        "device": str(device),
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "test_acc": test_acc,
        "misclassified_count": mistake_count,
        "total_test_samples": len(y_true),
        "confusion_matrix": cm.tolist(),
        "history_csv": str(history_path),
        "learning_curves": str(curves_path),
        "confusion_matrix_png": str(confusion_path),
        "misclassified_examples_png": str(mistakes_path),
        "checkpoint": str(checkpoint_path),
    }

    with (cfg.paths.report_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    return summary
