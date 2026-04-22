import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, normalized_root_mse


def compute_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict:
    ref = reference.astype(np.float32)
    cand = candidate.astype(np.float32)

    mse = float(np.mean((ref - cand) ** 2))
    mae = float(np.mean(np.abs(ref - cand)))
    nrmse = float(normalized_root_mse(ref, cand))
    psnr = float(peak_signal_noise_ratio(ref, cand, data_range=1.0))

    if ref.ndim == 2:
        ssim = float(structural_similarity(ref, cand, data_range=1.0))
    else:
        ssim = float(structural_similarity(ref, cand, data_range=1.0, channel_axis=-1))

    return {
        "mse": mse,
        "mae": mae,
        "nrmse": nrmse,
        "psnr": psnr,
        "ssim": ssim,
    }