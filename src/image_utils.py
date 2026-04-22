import numpy as np
import torch


def tensor_to_rgb_numpy(img: torch.Tensor) -> np.ndarray:

    arr = img.detach().cpu().numpy().astype(np.float32)

    if arr.ndim != 3:
        raise ValueError(f"Expected [C,H,W], got {arr.shape}")

    if arr.shape[0] == 1:
        arr = np.repeat(arr, 3, axis=0)

    arr = np.transpose(arr, (1, 2, 0))
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def rgb_to_gray(img_rgb: np.ndarray) -> np.ndarray:

    if img_rgb.ndim == 2:
        return img_rgb.astype(np.float32)

    gray = 0.2989 * img_rgb[..., 0] + 0.5870 * img_rgb[..., 1] + 0.1140 * img_rgb[..., 2]
    return np.clip(gray, 0.0, 1.0).astype(np.float32)


def clip01(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0.0, 1.0).astype(np.float32)