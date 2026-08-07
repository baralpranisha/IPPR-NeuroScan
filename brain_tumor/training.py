from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import CLASS_NAMES, DEFAULT_IMAGE_SIZE
from .data import BrainTumorDataset, ImageRecord, split_records
from .metrics import calculate_metrics, save_json
from .model import BrainTumorCNN


@dataclass
class TrainingConfig:
    dataset_dir: str
    output_dir: str = "outputs"
    image_size: int = DEFAULT_IMAGE_SIZE
    epochs: int = 12
    batch_size: int = 32
    learning_rate: float = 0.0005
    weight_decay: float = 0.0001
    seed: int = 42
    num_workers: int = 0
    patience: int = 4


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(
    records: list[ImageRecord],
    config: TrainingConfig,
    train: bool,
) -> DataLoader:
    dataset = BrainTumorDataset(records, image_size=config.image_size, train=train)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=train,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        if training:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item()) * labels.size(0)
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += labels.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


@torch.inference_mode()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int], list[float]]:
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    confidences: list[float] = []
    for images, labels in loader:
        logits = model(images.to(device))
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted = probabilities.max(dim=1)
        targets.extend(labels.tolist())
        predictions.extend(predicted.cpu().tolist())
        confidences.extend(confidence.cpu().tolist())
    return targets, predictions, confidences


def train_model(config: TrainingConfig) -> dict:
    seed_everything(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    from .data import discover_images, manifest_dataframe

    records = discover_images(config.dataset_dir)
    splits = split_records(records, seed=config.seed)
    manifest_dataframe(records).to_csv(output_dir / "dataset_manifest.csv", index=False)
    for split_name, split_records_list in splits.items():
        manifest_dataframe(split_records_list).to_csv(
            output_dir / f"{split_name}_manifest.csv", index=False
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BrainTumorCNN(num_classes=len(CLASS_NAMES)).to(device)
    train_loader = _loader(splits["train"], config, train=True)
    validation_loader = _loader(splits["validation"], config, train=False)
    test_loader = _loader(splits["test"], config, train=False)

    class_counts = np.bincount([record.label_id for record in splits["train"]], minlength=2)
    class_weights = len(splits["train"]) / (2 * np.maximum(class_counts, 1))
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )

    history: list[dict[str, float | int]] = []
    best_validation_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience_left = config.patience
    for epoch in range(1, config.epochs + 1):
        train_loss, train_accuracy = _run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        validation_loss, validation_accuracy = _run_epoch(
            model, validation_loader, criterion, device, None
        )
        scheduler.step(validation_loss)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience_left = config.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_targets, test_predictions, test_confidences = predict_loader(
        model, test_loader, device
    )
    metrics = calculate_metrics(test_targets, test_predictions, CLASS_NAMES)
    metrics["mean_confidence"] = float(np.mean(test_confidences))
    metrics["device"] = str(device)
    metrics["split_sizes"] = {name: len(values) for name, values in splits.items()}

    model_path = output_dir / "brain_tumor_cnn.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": CLASS_NAMES,
            "image_size": config.image_size,
            "config": asdict(config),
        },
        model_path,
    )
    (output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    save_json(metrics, output_dir / "metrics.json")
    summary = {
        "model_path": str(model_path),
        "metrics_path": str(output_dir / "metrics.json"),
        "history_path": str(output_dir / "training_history.json"),
        "dataset_size": len(records),
        "epochs_completed": len(history),
        "duration_seconds": round(time.perf_counter() - started, 2),
        **metrics,
    }
    save_json(summary, output_dir / "run_summary.json")
    return summary