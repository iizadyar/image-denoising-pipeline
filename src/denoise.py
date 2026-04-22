import numpy as np
from bm3d import bm3d
from skimage.restoration import denoise_wavelet


def _bm3d_per_channel(img: np.ndarray, sigma: float) -> np.ndarray:

    if img.ndim == 2:
        out = bm3d(img, sigma_psd=sigma)
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    out = np.zeros_like(img, dtype=np.float32)
    for c in range(img.shape[2]):
        out[..., c] = bm3d(img[..., c], sigma_psd=sigma)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _wavelet_gaussian(img: np.ndarray, sigma: float, cfg) -> np.ndarray:
    if img.ndim == 2:
        out = denoise_wavelet(
            img,
            sigma=sigma,
            wavelet=cfg.WAVELET_NAME,
            method=cfg.WAVELET_METHOD,
            mode=cfg.WAVELET_MODE,
            rescale_sigma=True,
            channel_axis=None,
        )
    else:
        out = denoise_wavelet(
            img,
            sigma=sigma,
            wavelet=cfg.WAVELET_NAME,
            method=cfg.WAVELET_METHOD,
            mode=cfg.WAVELET_MODE,
            rescale_sigma=True,
            channel_axis=-1,
            convert2ycbcr=False,
        )
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _anscombe_forward_counts(y_counts: np.ndarray) -> np.ndarray:
    return 2.0 * np.sqrt(np.maximum(y_counts, 0.0) + 3.0 / 8.0)


def _anscombe_inverse_counts(z: np.ndarray) -> np.ndarray:
    return np.maximum((z / 2.0) ** 2 - 3.0 / 8.0, 0.0)


def _poisson_to_vst_norm(img: np.ndarray, peak: float):

    counts = np.clip(img, 0.0, 1.0) * peak
    z = _anscombe_forward_counts(counts)
    zmax = 2.0 * np.sqrt(peak + 3.0 / 8.0)
    z_norm = z / zmax
    return z_norm.astype(np.float32), zmax


def _poisson_from_vst_norm(z_norm: np.ndarray, zmax: float, peak: float) -> np.ndarray:
    z = np.maximum(z_norm, 0.0) * zmax
    counts_hat = _anscombe_inverse_counts(z)
    img_hat = counts_hat / peak
    return np.clip(img_hat, 0.0, 1.0).astype(np.float32)


def denoise_bm3d(img: np.ndarray, noise_type: str, cfg) -> np.ndarray:
    if noise_type == "gaussian":
        return _bm3d_per_channel(img, cfg.BM3D_SIGMA_GAUSSIAN)

    if noise_type == "poisson":
        z_norm, zmax = _poisson_to_vst_norm(img, cfg.POISSON_PEAK)
        sigma_norm = cfg.BM3D_SIGMA_ANSCOMBE / zmax
        z_hat_norm = _bm3d_per_channel(z_norm, sigma_norm)
        return _poisson_from_vst_norm(z_hat_norm, zmax, cfg.POISSON_PEAK)

    raise ValueError(f"Unsupported noise type: {noise_type}")


def denoise_wavelet_image(img: np.ndarray, noise_type: str, cfg) -> np.ndarray:
    if noise_type == "gaussian":
        return _wavelet_gaussian(img, cfg.GAUSSIAN_SIGMA, cfg)

    if noise_type == "poisson":
        z_norm, zmax = _poisson_to_vst_norm(img, cfg.POISSON_PEAK)
        sigma_norm = cfg.BM3D_SIGMA_ANSCOMBE / zmax
        z_hat_norm = _wavelet_gaussian(z_norm, sigma_norm, cfg)
        return _poisson_from_vst_norm(z_hat_norm, zmax, cfg.POISSON_PEAK)

    raise ValueError(f"Unsupported noise type: {noise_type}")