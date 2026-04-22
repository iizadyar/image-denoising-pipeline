import numpy as np


def add_gaussian_noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(loc=0.0, scale=sigma, size=img.shape).astype(np.float32)
    noisy = img + noise
    return np.clip(noisy, 0.0, 1.0).astype(np.float32)


def add_poisson_noise(img: np.ndarray, peak: float, rng: np.random.Generator) -> np.ndarray:
    counts = np.clip(img, 0.0, 1.0) * peak
    noisy_counts = rng.poisson(counts).astype(np.float32)
    noisy = noisy_counts / peak
    return np.clip(noisy, 0.0, 1.0).astype(np.float32)


def apply_noise(img: np.ndarray, noise_type: str, cfg, rng: np.random.Generator) -> np.ndarray:
    if noise_type == "gaussian":
        return add_gaussian_noise(img, cfg.GAUSSIAN_SIGMA, rng)
    if noise_type == "poisson":
        return add_poisson_noise(img, cfg.POISSON_PEAK, rng)
    raise ValueError(f"Unsupported noise type: {noise_type}")