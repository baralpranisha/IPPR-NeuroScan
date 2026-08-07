from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from PIL import Image

from .config import DEFAULT_OUTPUT_DIR

HISTORY_LOG_PATH = DEFAULT_OUTPUT_DIR / "prediction_history.json"
HISTORY_IMAGE_DIR = DEFAULT_OUTPUT_DIR / "history"


def _ensure_history_dir() -> None:
    HISTORY_IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def load_history() -> list[dict]:
    """Return all recorded predictions, oldest first. Never raises on a missing
    or corrupted log file — just returns an empty list instead."""
    if not HISTORY_LOG_PATH.exists():
        return []
    try:
        return json.loads(HISTORY_LOG_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _save_history(records: list[dict]) -> None:
    HISTORY_LOG_PATH.write_text(json.dumps(records, indent=2), encoding="utf-8")


def record_prediction(image: Image.Image, filename: str, result: dict) -> dict:
    """Persist an uploaded image and its prediction (plus Grad-CAM heatmap, if
    present in `result`) to disk, so it shows up on the History page even after
    the app restarts. Returns the saved record.
    """
    _ensure_history_dir()
    record_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"

    image_path = HISTORY_IMAGE_DIR / f"{record_id}_input.png"
    image.convert("RGB").save(image_path)

    gradcam_path: str | None = None
    gradcam_image = result.get("gradcam_image")
    if gradcam_image is not None:
        gradcam_file = HISTORY_IMAGE_DIR / f"{record_id}_gradcam.png"
        gradcam_image.convert("RGB").save(gradcam_file)
        gradcam_path = str(gradcam_file)

    record = {
        "id": record_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "filename": filename,
        "label": result["label"],
        "confidence": result["confidence"],
        "probabilities": result["probabilities"],
        "image_path": str(image_path),
        "gradcam_path": gradcam_path,
    }

    records = load_history()
    records.append(record)
    _save_history(records)
    return record


def clear_history() -> None:
    """Delete the entire history log and all saved history images."""
    if HISTORY_LOG_PATH.exists():
        HISTORY_LOG_PATH.unlink()
    if HISTORY_IMAGE_DIR.exists():
        shutil.rmtree(HISTORY_IMAGE_DIR)
