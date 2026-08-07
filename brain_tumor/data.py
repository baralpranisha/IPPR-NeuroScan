from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter, ImageOps
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

from .config import CLASS_NAMES, IMAGE_EXTENSIONS


LABEL_ALIASES = {
    "healthy": "healthy",
    "normal": "healthy",
    "no_tumor": "healthy",
    "notumor": "healthy",
    "no tumor": "healthy",
    "tumor": "tumor",
    "brain tumor": "tumor",
    "braintumor": "tumor",
    "cancer": "tumor",
    # 4-class Kaggle "brain-tumor-mri-dataset" folder names (glioma,
    # meningioma, pituitary, notumor) collapsed into this project's
    # 2-class healthy/tumor scheme.
    "glioma": "tumor",
    "meningioma": "tumor",
    "pituitary": "tumor",
    "pituitary tumor": "tumor",
}


@dataclass(frozen=True)
class ImageRecord:
    path: str
    label: str
    label_id: int
    file_hash: str
    width: int
    height: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def canonical_label(value: str) -> str | None:
    normalized = re.sub(r"[_\-/]+", " ", str(value).strip().lower())
    normalized = re.sub(r"\s+", " ", normalized)
    return LABEL_ALIASES.get(normalized)


def _metadata_label_map(dataset_root: Path) -> dict[str, str]:
    metadata_files = list(dataset_root.rglob("metadata.csv"))
    if not metadata_files:
        return {}
    metadata = pd.read_csv(metadata_files[0])
    if "image" not in metadata.columns or "class" not in metadata.columns:
        return {}
    labels: dict[str, str] = {}
    for _, row in metadata.iterrows():
        label = canonical_label(row["class"])
        if label:
            labels[Path(str(row["image"])).name.lower()] = label
    return labels


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_images(dataset_root: str | Path) -> list[ImageRecord]:
    root = Path(dataset_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    metadata_labels = _metadata_label_map(root)
    records: list[ImageRecord] = []
    skipped = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = canonical_label(path.parent.name) or metadata_labels.get(path.name.lower())
        if label is None:
            skipped += 1
            continue
        try:
            with Image.open(path) as image:
                width, height = image.size
                image.verify()
            records.append(
                ImageRecord(
                    path=str(path),
                    label=label,
                    label_id=CLASS_NAMES.index(label),
                    file_hash=_file_digest(path),
                    width=width,
                    height=height,
                )
            )
        except (OSError, ValueError):
            skipped += 1
    if not records:
        raise ValueError(
            "No labeled images were found. Expected class folders named "
            "'Healthy' and 'Brain Tumor'/'Tumor', or a metadata.csv file."
        )
    if skipped:
        print(f"Skipped {skipped} unreadable or unlabeled files.")
    return records


def manifest_dataframe(records: Iterable[ImageRecord]) -> pd.DataFrame:
    return pd.DataFrame([record.to_dict() for record in records])


def _record_group_key(record: ImageRecord) -> str:
    """Group common format variants of the same named source image."""
    stem = re.sub(r"[^a-z0-9]+", "", Path(record.path).stem.lower())
    return f"{record.label_id}:{stem}"


def _split_group_ids(
    records: list[ImageRecord],
    test_size: float,
    val_size: float,
    seed: int,
) -> tuple[set[str], set[str], set[str]]:
    frame = manifest_dataframe(records)
    frame["group_key"] = [_record_group_key(record) for record in records]
    groups = frame.groupby("group_key", as_index=False)["label_id"].first()
    class_group_counts = groups["label_id"].value_counts()
    if groups["label_id"].nunique() < 2:
        raise ValueError("Both healthy and tumor classes are required for a stratified split.")
    # Hash grouping is preferred because the Kaggle data can contain format variants
    # of the same image. Very small or highly duplicated datasets can collapse to
    # too few groups for a three-way stratified split, so use file-level stratification
    # as an explicit, safe fallback rather than failing with a cryptic sklearn error.
    if class_group_counts.min() < 3:
        file_ids = frame[["path", "label_id"]]
        train_files, test_files = train_test_split(
            file_ids,
            test_size=test_size,
            random_state=seed,
            stratify=file_ids["label_id"],
        )
        relative_val = val_size / (1.0 - test_size)
        train_files, val_files = train_test_split(
            train_files,
            test_size=relative_val,
            random_state=seed,
            stratify=train_files["label_id"],
        )
        return (
            set(train_files["path"]),
            set(val_files["path"]),
            set(test_files["path"]),
        )
    train_groups, test_groups = train_test_split(
        groups,
        test_size=test_size,
        random_state=seed,
        stratify=groups["label_id"],
    )
    relative_val = val_size / (1.0 - test_size)
    train_groups, val_groups = train_test_split(
        train_groups,
        test_size=relative_val,
        random_state=seed,
        stratify=train_groups["label_id"],
    )
    return (
        set(train_groups["group_key"]),
        set(val_groups["group_key"]),
        set(test_groups["group_key"]),
    )


def split_records(
    records: list[ImageRecord],
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> dict[str, list[ImageRecord]]:
    train_ids, val_ids, test_ids = _split_group_ids(records, test_size, val_size, seed)
    group_keys = {_record_group_key(record) for record in records}
    grouped = all(identifier in group_keys for identifier in train_ids)
    if grouped:
        split_map = {
            "train": [record for record in records if _record_group_key(record) in train_ids],
            "validation": [record for record in records if _record_group_key(record) in val_ids],
            "test": [record for record in records if _record_group_key(record) in test_ids],
        }
    else:
        split_map = {
            "train": [record for record in records if record.path in train_ids],
            "validation": [record for record in records if record.path in val_ids],
            "test": [record for record in records if record.path in test_ids],
        }
    if any(not values for values in split_map.values()):
        raise ValueError("The dataset is too small to populate train, validation, and test splits.")
    return split_map


def _crop_to_brain_region(image: Image.Image, threshold: int = 10, padding: int = 4) -> Image.Image:
    """Crop out the black background/border surrounding the brain scan.

    Without this, a CNN can learn to key off border shape, corner artifacts,
    or scanner-specific framing instead of the actual brain tissue -- a known
    shortcut-learning failure mode with this dataset. This keeps only the
    bounding box of non-background pixels (with a small padding margin), so
    background can no longer carry predictive signal.
    """
    grayscale = ImageOps.grayscale(image)
    array = np.asarray(grayscale)
    mask = array > threshold
    if not mask.any():
        return image
    row_indices = np.where(mask.any(axis=1))[0]
    col_indices = np.where(mask.any(axis=0))[0]
    top = max(int(row_indices[0]) - padding, 0)
    bottom = min(int(row_indices[-1]) + padding, array.shape[0] - 1)
    left = max(int(col_indices[0]) - padding, 0)
    right = min(int(col_indices[-1]) + padding, array.shape[1] - 1)
    return image.crop((left, top, right + 1, bottom + 1))


def preprocess_image(
    image: Image.Image,
    image_size: int = 224,
    threshold: bool = False,
) -> Image.Image:
    """Apply the IPPR preprocessing stages used by the classifier."""
    stages = preprocessing_stages(image, image_size)
    resized = stages["resized"]
    if threshold:
        array = np.asarray(resized)
        threshold_value = float(array.mean())
        resized = Image.fromarray((array > threshold_value).astype(np.uint8) * 255)
    return resized.convert("L")


def preprocessing_stages(image: Image.Image, image_size: int = 224) -> dict[str, Image.Image]:
    """Return the visual stages required by the IPPR preprocessing pipeline."""
    original = image.convert("RGB")
    cropped = _crop_to_brain_region(original)
    grayscale = ImageOps.grayscale(cropped)
    denoised = grayscale.filter(ImageFilter.MedianFilter(size=3))
    equalized = ImageOps.equalize(denoised)
    resized = equalized.resize((image_size, image_size), Image.Resampling.LANCZOS)
    return {
        "original": original,
        "cropped": cropped,
        "grayscale": grayscale,
        "denoised": denoised,
        "equalized": equalized,
        "resized": resized,
    }


class BrainTumorDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        records: list[ImageRecord],
        image_size: int = 224,
        train: bool = False,
    ) -> None:
        self.records = records
        self.image_size = image_size
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=8),
                transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
            if train
            else [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        with Image.open(record.path) as image:
            processed = preprocess_image(image, self.image_size)
        return self.transform(processed), record.label_id