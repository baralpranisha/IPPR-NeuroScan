from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .data import preprocess_image
from .gradcam import GradCAM, overlay_heatmap
from .model import BrainTumorCNN


def load_model(model_path: str | Path) -> tuple[BrainTumorCNN, tuple[str, ...], int, torch.device]:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    class_names = tuple(checkpoint.get("class_names", ("healthy", "tumor")))
    image_size = int(checkpoint.get("image_size", 224))
    model = BrainTumorCNN(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names, image_size, torch.device("cpu")


def _preprocess_to_tensor(image: Image.Image, image_size: int, device: torch.device) -> tuple[Image.Image, torch.Tensor]:
    processed = preprocess_image(image, image_size=image_size)
    tensor = torch.from_numpy(
        np.asarray(processed, dtype="float32") / 255.0
    ).unsqueeze(0).unsqueeze(0)
    tensor = (tensor - 0.5) / 0.5
    return processed, tensor.to(device)


@torch.inference_mode()
def predict_image(
    model: BrainTumorCNN,
    image: Image.Image,
    class_names: tuple[str, ...],
    image_size: int,
    device: torch.device,
) -> dict:
    processed, tensor = _preprocess_to_tensor(image, image_size, device)
    probabilities = torch.softmax(model(tensor), dim=1)[0]
    predicted_id = int(probabilities.argmax().item())
    return {
        "label": class_names[predicted_id],
        "confidence": float(probabilities[predicted_id].item()),
        "probabilities": {
            name: float(probabilities[index].item())
            for index, name in enumerate(class_names)
        },
        "processed_image": processed,
    }


def predict_image_with_explanation(
    model: BrainTumorCNN,
    image: Image.Image,
    class_names: tuple[str, ...],
    image_size: int,
    device: torch.device,
) -> dict:
    """Same as predict_image, but also returns a Grad-CAM heatmap showing which
    pixels the model actually used to make its decision. Use this to check
    whether the model is focusing on brain tissue versus scan borders,
    corners, or other dataset-specific artifacts unrelated to the tumor.
    """
    processed, tensor = _preprocess_to_tensor(image, image_size, device)

    cam_tool = GradCAM(model)
    try:
        cam, predicted_id = cam_tool(tensor)
    finally:
        cam_tool.close()

    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]

    heatmap_image = overlay_heatmap(processed.convert("RGB"), cam)

    return {
        "label": class_names[predicted_id],
        "confidence": float(probabilities[predicted_id].item()),
        "probabilities": {
            name: float(probabilities[index].item())
            for index, name in enumerate(class_names)
        },
        "processed_image": processed,
        "gradcam_image": heatmap_image,
    }