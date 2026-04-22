import hashlib
import random
from time import perf_counter

import numpy as np
import pandas as pd
import torch

import config as cfg
from dataset_loader import get_dataset
from image_utils import tensor_to_rgb_numpy, rgb_to_gray
from noise import apply_noise
from denoise import denoise_bm3d, denoise_wavelet_image
from metrics import compute_metrics
from visualization import (
    save_dataset_overview,
    save_examples_grid,
    save_metric_comparison,
    save_boxplot_by_noise,
    save_summary_table,
)


def set_global_determinism(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_stable_rng(base_seed: int, image_id: int, noise_type: str, mode_name: str) -> np.random.Generator:
    key = f"{base_seed}|{image_id}|{noise_type}|{mode_name}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    local_seed = int.from_bytes(digest[:8], byteorder="little", signed=False) % (2 ** 32)
    return np.random.default_rng(local_seed)


def main():
    set_global_determinism(cfg.RANDOM_SEED)

    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset, info = get_dataset(
        flag=cfg.DATASET_FLAG,
        size=cfg.IMAGE_SIZE,
        split=cfg.SPLIT,
        root=cfg.DATA_DIR,
        max_images=cfg.MAX_IMAGES,
    )

    print("=" * 80)
    print("Dataset:", cfg.DATASET_FLAG)
    print("Python class:", info["python_class"])
    print("Source channels:", info["n_channels"])
    print("Image size:", cfg.IMAGE_SIZE)
    print("Images used:", len(dataset))
    print("Noise types:", cfg.NOISE_TYPES)
    print("Modes:", cfg.MODES)
    print("Methods:", cfg.METHODS)
    print("Random seed:", cfg.RANDOM_SEED)
    print("=" * 80)

    overview_images = []
    example_bank = {(noise_type, mode): [] for noise_type in cfg.NOISE_TYPES for mode in cfg.MODES}
    records = []

    for image_id, (img_tensor, target) in enumerate(dataset):
        rgb_clean = tensor_to_rgb_numpy(img_tensor)
        gray_clean = rgb_to_gray(rgb_clean)

        if image_id < 12:
            overview_images.append(gray_clean)

        for noise_type in cfg.NOISE_TYPES:
            for mode_name, clean in [("rgb", rgb_clean), ("gray", gray_clean)]:
                rng = make_stable_rng(cfg.RANDOM_SEED, image_id, noise_type, mode_name)

                noisy = apply_noise(clean, noise_type, cfg, rng)
                noisy_metrics = compute_metrics(clean, noisy)

                t0 = perf_counter()
                bm3d_img = denoise_bm3d(noisy, noise_type, cfg)
                bm3d_runtime = perf_counter() - t0
                bm3d_metrics = compute_metrics(clean, bm3d_img)

                t0 = perf_counter()
                wavelet_img = denoise_wavelet_image(noisy, noise_type, cfg)
                wavelet_runtime = perf_counter() - t0
                wavelet_metrics = compute_metrics(clean, wavelet_img)

                for method_name, den_metrics, runtime_sec in [
                    ("bm3d", bm3d_metrics, bm3d_runtime),
                    ("wavelet", wavelet_metrics, wavelet_runtime),
                ]:
                    records.append({
                        "image_id": image_id,
                        "noise_type": noise_type,
                        "mode": mode_name,
                        "method": method_name,

                        "noisy_mse": noisy_metrics["mse"],
                        "noisy_mae": noisy_metrics["mae"],
                        "noisy_nrmse": noisy_metrics["nrmse"],
                        "noisy_psnr": noisy_metrics["psnr"],
                        "noisy_ssim": noisy_metrics["ssim"],

                        "den_mse": den_metrics["mse"],
                        "den_mae": den_metrics["mae"],
                        "den_nrmse": den_metrics["nrmse"],
                        "den_psnr": den_metrics["psnr"],
                        "den_ssim": den_metrics["ssim"],

                        "mse_drop": noisy_metrics["mse"] - den_metrics["mse"],
                        "mae_drop": noisy_metrics["mae"] - den_metrics["mae"],
                        "nrmse_drop": noisy_metrics["nrmse"] - den_metrics["nrmse"],
                        "psnr_gain": den_metrics["psnr"] - noisy_metrics["psnr"],
                        "ssim_gain": den_metrics["ssim"] - noisy_metrics["ssim"],

                        "runtime_sec": runtime_sec,
                    })

                key = (noise_type, mode_name)
                if len(example_bank[key]) < cfg.EXAMPLE_ROWS:
                    example_bank[key].append(
                        (f"img_{image_id}", clean, noisy, bm3d_img, wavelet_img)
                    )

    detail_df = pd.DataFrame(records)
    detail_path = cfg.OUTPUT_DIR / "detailed_metrics.csv"
    detail_df.to_csv(detail_path, index=False)

    summary_df = (
        detail_df.groupby(["noise_type", "mode", "method"], as_index=False)
        .agg(
            noisy_psnr_mean=("noisy_psnr", "mean"),
            noisy_psnr_std=("noisy_psnr", "std"),
            noisy_ssim_mean=("noisy_ssim", "mean"),
            noisy_ssim_std=("noisy_ssim", "std"),

            den_psnr_mean=("den_psnr", "mean"),
            den_psnr_std=("den_psnr", "std"),
            den_ssim_mean=("den_ssim", "mean"),
            den_ssim_std=("den_ssim", "std"),

            den_mse_mean=("den_mse", "mean"),
            den_mse_std=("den_mse", "std"),
            den_mae_mean=("den_mae", "mean"),
            den_mae_std=("den_mae", "std"),
            den_nrmse_mean=("den_nrmse", "mean"),
            den_nrmse_std=("den_nrmse", "std"),

            psnr_gain_mean=("psnr_gain", "mean"),
            psnr_gain_std=("psnr_gain", "std"),
            ssim_gain_mean=("ssim_gain", "mean"),
            ssim_gain_std=("ssim_gain", "std"),
            mse_drop_mean=("mse_drop", "mean"),
            mse_drop_std=("mse_drop", "std"),

            runtime_mean=("runtime_sec", "mean"),
            runtime_std=("runtime_sec", "std"),
        )
    )

    summary_path = cfg.OUTPUT_DIR / "summary_metrics.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\nSummary metrics:")
    print(summary_df.round(4).to_string(index=False))

    # Image figures
    save_dataset_overview(
        overview_images,
        cfg.OUTPUT_DIR / "dataset_overview.png",
        title=f"{cfg.DATASET_FLAG} overview",
        ncols=4,
        dpi=cfg.SAVE_DPI,
    )

    for (noise_type, mode_name), rows in example_bank.items():
        save_examples_grid(
            rows,
            cfg.OUTPUT_DIR / f"examples_{noise_type}_{mode_name}.png",
            title=f"{cfg.DATASET_FLAG} | {noise_type.upper()} | {mode_name.upper()}",
            dpi=cfg.SAVE_DPI,
        )

    # Clean metric comparison plots
    colors = {
        "bm3d": cfg.COLOR_BM3D,
        "wavelet": cfg.COLOR_WAVELET
    }

    save_metric_comparison(
        summary_df,
        mean_col="den_psnr_mean",
        std_col="den_psnr_std",
        title="Denoised PSNR Comparison",
        ylabel="PSNR (dB)",
        path=cfg.OUTPUT_DIR / "plot_den_psnr.png",
        dpi=cfg.SAVE_DPI,
        value_fmt="{:.2f}",
        colors=colors
    )

    save_metric_comparison(
        summary_df,
        mean_col="den_ssim_mean",
        std_col="den_ssim_std",
        title="Denoised SSIM Comparison",
        ylabel="SSIM",
        path=cfg.OUTPUT_DIR / "plot_den_ssim.png",
        dpi=cfg.SAVE_DPI,
        value_fmt="{:.3f}",
        colors=colors
    )

    save_metric_comparison(
        summary_df,
        mean_col="den_mse_mean",
        std_col="den_mse_std",
        title="Denoised MSE Comparison",
        ylabel="MSE",
        path=cfg.OUTPUT_DIR / "plot_den_mse.png",
        dpi=cfg.SAVE_DPI,
        value_fmt="{:.4f}",
        colors=colors
    )

    save_metric_comparison(
        summary_df,
        mean_col="psnr_gain_mean",
        std_col="psnr_gain_std",
        title="PSNR Gain After Denoising",
        ylabel="PSNR Gain (dB)",
        path=cfg.OUTPUT_DIR / "plot_psnr_gain.png",
        dpi=cfg.SAVE_DPI,
        value_fmt="{:.2f}",
        colors=colors
    )

    save_metric_comparison(
        summary_df,
        mean_col="ssim_gain_mean",
        std_col="ssim_gain_std",
        title="SSIM Gain After Denoising",
        ylabel="SSIM Gain",
        path=cfg.OUTPUT_DIR / "plot_ssim_gain.png",
        dpi=cfg.SAVE_DPI,
        value_fmt="{:.3f}",
        colors=colors
    )

    save_metric_comparison(
        summary_df,
        mean_col="runtime_mean",
        std_col="runtime_std",
        title="Average Runtime per Image",
        ylabel="Seconds",
        path=cfg.OUTPUT_DIR / "plot_runtime.png",
        dpi=cfg.SAVE_DPI,
        value_fmt="{:.3f}",
        colors=colors
    )

    # Boxplots
    save_boxplot_by_noise(
        detail_df,
        metric_col="den_psnr",
        title="Distribution of Denoised PSNR",
        ylabel="PSNR (dB)",
        path=cfg.OUTPUT_DIR / "boxplot_den_psnr.png",
        dpi=cfg.SAVE_DPI,
        colors=colors
    )

    save_boxplot_by_noise(
        detail_df,
        metric_col="den_ssim",
        title="Distribution of Denoised SSIM",
        ylabel="SSIM",
        path=cfg.OUTPUT_DIR / "boxplot_den_ssim.png",
        dpi=cfg.SAVE_DPI,
        colors=colors
    )

    save_boxplot_by_noise(
        detail_df,
        metric_col="runtime_sec",
        title="Distribution of Runtime",
        ylabel="Seconds",
        path=cfg.OUTPUT_DIR / "boxplot_runtime.png",
        dpi=cfg.SAVE_DPI,
        colors=colors
    )

    # Summary table image
    save_summary_table(
        summary_df,
        cfg.OUTPUT_DIR / "summary_table.png",
        dpi=cfg.SAVE_DPI
    )

    print("\nSaved files:")
    print(f"- {detail_path}")
    print(f"- {summary_path}")
    print(f"- figures in: {cfg.OUTPUT_DIR}")
    print("\nDone.")


if __name__ == "__main__":
    main()