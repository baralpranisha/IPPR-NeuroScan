from __future__ import annotations

import numpy as np
import torch
from torch import nn
from PIL import Image


def _find_last_conv2d(model: nn.Module) -> nn.Module:
    """Auto-detect the last Conv2d layer in the network for Grad-CAM hooking.

    Works regardless of the exact architecture in model.py, as long as it's a
    standard CNN (conv blocks followed by pooling/linear layers).
    """
    last_conv: nn.Module | None = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError(
            "No nn.Conv2d layer found in the model. Grad-CAM requires at least "
            "one convolutional layer to hook activations/gradients from."
        )
    return last_conv


class GradCAM:
    """Gradient-weighted Class Activation Mapping.

    Produces a heatmap over the input showing which spatial regions most
    influenced the model's predicted class. Use this to check whether the
    model is focusing on the brain tissue itself versus scan borders,
    corners, or other dataset-specific artifacts.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None) -> None:
        self.model = model
        self.target_layer = target_layer or _find_last_conv2d(model)
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._forward_handle = self.target_layer.register_forward_hook(self._save_activations)
        self._backward_handle = self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
        self._activations = output.detach()

    def _save_gradients(self, module: nn.Module, grad_input: tuple, grad_output: tuple) -> None:
        self._gradients = grad_output[0].detach()

    def close(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()

    def __call__(self, input_tensor: torch.Tensor, target_class: int | None = None) -> tuple[np.ndarray, int]:
        """Run Grad-CAM on a single (1, C, H, W) input tensor.

        Returns (cam, target_class) where cam is a float32 array in [0, 1]
        at the spatial resolution of the target conv layer's feature map.
        """
        self.model.eval()
        was_grad_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        try:
            input_tensor = input_tensor.clone().requires_grad_(True)
            logits = self.model(input_tensor)
            if target_class is None:
                target_class = int(logits.argmax(dim=1).item())
            self.model.zero_grad(set_to_none=True)
            score = logits[0, target_class]
            score.backward()

            if self._activations is None or self._gradients is None:
                raise RuntimeError(
                    "Grad-CAM hooks did not capture activations/gradients. "
                    "Check that the target layer is actually used in the forward pass."
                )

            activations = self._activations[0]  # (channels, h, w)
            gradients = self._gradients[0]  # (channels, h, w)
            weights = gradients.mean(dim=(1, 2))  # (channels,) global average pool

            cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
            for channel_index, weight in enumerate(weights):
                cam += weight * activations[channel_index]
            cam = torch.relu(cam)

            cam_min, cam_max = float(cam.min()), float(cam.max())
            if cam_max - cam_min > 1e-8:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = torch.zeros_like(cam)

            return cam.cpu().numpy(), target_class
        finally:
            torch.set_grad_enabled(was_grad_enabled)


def overlay_heatmap(base_image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Resize a Grad-CAM map to match base_image and blend it on as a red-hot heatmap."""
    width, height = base_image.size
    cam_image = Image.fromarray((cam * 255).astype(np.uint8)).resize((width, height), Image.Resampling.BILINEAR)
    cam_array = np.asarray(cam_image, dtype=np.float32) / 255.0

    heatmap = np.zeros((height, width, 3), dtype=np.float32)
    heatmap[..., 0] = np.clip(1.5 * cam_array, 0, 1)  # red channel, hottest where cam is highest
    heatmap[..., 1] = np.clip(1.5 * (cam_array - 0.5), 0, 1)  # green kicks in for mid-high activation
    heatmap[..., 2] = np.clip(1.0 - 1.5 * cam_array, 0, 1)  # blue fades out as activation rises

    base_array = np.asarray(base_image.convert("RGB"), dtype=np.float32) / 255.0
    blended = (1 - alpha) * base_array + alpha * heatmap
    blended = np.clip(blended * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)
