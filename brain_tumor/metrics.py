from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def calculate_metrics(
    targets: Iterable[int],
    predictions: Iterable[int],
    class_names: tuple[str, ...] = ("healthy", "tumor"),
) -> dict:
    target_array = np.asarray(list(targets))
    prediction_array = np.asarray(list(predictions))
    precision, recall, f1, _ = precision_recall_fscore_support(
        target_array,
        prediction_array,
        labels=list(range(len(class_names))),
        average="binary" if len(class_names) == 2 else "weighted",
        zero_division=0,
    )
    report = classification_report(
        target_array,
        prediction_array,
        labels=list(range(len(class_names))),
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(target_array, prediction_array)),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": confusion_matrix(
            target_array,
            prediction_array,
            labels=list(range(len(class_names))),
        ).tolist(),
        "classification_report": report,
        "support": int(len(target_array)),
    }


def save_json(data: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")