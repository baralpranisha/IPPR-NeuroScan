from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


def generate_result_plots(output_dir: str | Path) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    history_path = output / "training_history.json"
    metrics_path = output / "metrics.json"
    generated: list[Path] = []
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        epochs = [row["epoch"] for row in history]
        figure, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train")
        axes[0].plot(epochs, [row["validation_loss"] for row in history], label="Validation")
        axes[0].set_title("Loss curve")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Cross-entropy loss")
        axes[0].legend()
        axes[1].plot(
            epochs, [row["train_accuracy"] for row in history], label="Train"
        )
        axes[1].plot(
            epochs, [row["validation_accuracy"] for row in history], label="Validation"
        )
        axes[1].set_title("Accuracy curve")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_ylim(0, 1)
        axes[1].legend()
        figure.tight_layout()
        path = output / "training_curves.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        generated.append(path)
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        matrix = metrics.get("confusion_matrix")
        if matrix:
            figure, axis = plt.subplots(figsize=(5, 4))
            sns.heatmap(
                matrix,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                xticklabels=["healthy", "tumor"],
                yticklabels=["healthy", "tumor"],
                ax=axis,
            )
            axis.set_xlabel("Predicted label")
            axis.set_ylabel("True label")
            axis.set_title("Test-set confusion matrix")
            figure.tight_layout()
            path = output / "confusion_matrix.png"
            figure.savefig(path, dpi=160)
            plt.close(figure)
            generated.append(path)
    return generated