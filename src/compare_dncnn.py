import hashlib
import random
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import config as cfg
from dataset_loader import get_dataset
from denoise import denoise_bm3d, denoise_wavelet_image
from dncnn_model import DnCNN
from image_utils import tensor_to_rgb_numpy, rgb_to_gray
from metrics import compute_metrics
from noise import apply_noise


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_stable_rng(base_seed: int, image_id: int, noise_type: str):
    key = f"{base_seed}|{image_id}|{noise_type}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    local_seed = int.from_bytes(digest[:8], byteorder="little", signed=False) % (2 ** 32)
    return np.random.default_rng(local_seed)


def set_plot_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.titleweight": "bold",
        "axes.titlesize": 11,
        "font.size": 10,
    })


def show_img(ax, img, caption=""):
    if img.ndim == 2:
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    else:
        ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(caption, fontsize=9, labelpad=8)


def save_example_grid(example_rows, path, title):
    if not example_rows:
        return

    set_plot_style()
    nrows = len(example_rows)
    fig, axes = plt.subplots(nrows, 5, figsize=(17, 3.6 * nrows), constrained_layout=True)

    if nrows == 1:
        axes = np.array([axes])

    col_headers = ["Ground Truth", "Noisy", "BM3D", "Wavelet", "DnCNN"]

    for c in range(5):
        axes[0, c].set_title(col_headers[c], fontsize=12, fontweight="bold")

    for r, row in enumerate(example_rows):
        img_id, clean, noisy, bm3d_img, wavelet_img, dncnn_img = row

        show_img(axes[r, 0], clean, caption=f"{img_id} | clean")
        show_img(axes[r, 1], noisy, caption=f"{img_id} | noisy")
        show_img(axes[r, 2], bm3d_img, caption=f"{img_id} | bm3d")
        show_img(axes[r, 3], wavelet_img, caption=f"{img_id} | wavelet")
        show_img(axes[r, 4], dncnn_img, caption=f"{img_id} | dncnn")

    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.savefig(path, dpi=cfg.SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    set_seed(cfg.RANDOM_SEED)

    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    docs_fig_dir = cfg.ROOT / "docs" / "figures"
    docs_fig_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = cfg.CHECKPOINT_DIR / cfg.DN_SAVE_NAME
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\nPlease run train_dncnn.py first."
        )

    print("Loading test dataset...")
    test_ds, info = get_dataset(
        cfg.DATASET_FLAG,
        cfg.IMAGE_SIZE,
        "test",
        cfg.DATA_DIR,
        max_images=cfg.MAX_IMAGES
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = DnCNN(in_channels=1).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    print(f"Loaded checkpoint: {ckpt_path}")

    records = []
    example_bank = {noise_type: [] for noise_type in cfg.NOISE_TYPES}

    for image_id, (img_tensor, _) in enumerate(test_ds):
        rgb = tensor_to_rgb_numpy(img_tensor)
        clean = rgb_to_gray(rgb)

        for noise_type in cfg.NOISE_TYPES:
            rng = make_stable_rng(cfg.RANDOM_SEED, image_id, noise_type)
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

            noisy_t = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).float().to(device)

            t0 = perf_counter()
            with torch.no_grad():
                dncnn_out = model(noisy_t).squeeze().cpu().numpy().astype(np.float32)
            dncnn_runtime = perf_counter() - t0

            dncnn_out = np.clip(dncnn_out, 0.0, 1.0)
            dncnn_metrics = compute_metrics(clean, dncnn_out)

            for method_name, den_metrics, runtime_sec in [
                ("bm3d", bm3d_metrics, bm3d_runtime),
                ("wavelet", wavelet_metrics, wavelet_runtime),
                ("dncnn", dncnn_metrics, dncnn_runtime),
            ]:
                records.append({
                    "image_id": image_id,
                    "noise_type": noise_type,
                    "method": method_name,
                    "noisy_psnr": noisy_metrics["psnr"],
                    "noisy_ssim": noisy_metrics["ssim"],
                    "den_psnr": den_metrics["psnr"],
                    "den_ssim": den_metrics["ssim"],
                    "den_mse": den_metrics["mse"],
                    "den_mae": den_metrics["mae"],
                    "den_nrmse": den_metrics["nrmse"],
                    "psnr_gain": den_metrics["psnr"] - noisy_metrics["psnr"],
                    "ssim_gain": den_metrics["ssim"] - noisy_metrics["ssim"],
                    "runtime_sec": runtime_sec,
                })

            if len(example_bank[noise_type]) < cfg.EXAMPLE_ROWS:
                example_bank[noise_type].append(
                    (f"img_{image_id}", clean, noisy, bm3d_img, wavelet_img, dncnn_out)
                )

    df = pd.DataFrame(records)
    detail_path = cfg.OUTPUT_DIR / "compare_dncnn_test.csv"
    df.to_csv(detail_path, index=False)

    summary = (
        df.groupby(["noise_type", "method"], as_index=False)
        .agg(
            den_psnr_mean=("den_psnr", "mean"),
            den_psnr_std=("den_psnr", "std"),
            den_ssim_mean=("den_ssim", "mean"),
            den_ssim_std=("den_ssim", "std"),
            den_mse_mean=("den_mse", "mean"),
            den_mae_mean=("den_mae", "mean"),
            den_nrmse_mean=("den_nrmse", "mean"),
            psnr_gain_mean=("psnr_gain", "mean"),
            ssim_gain_mean=("ssim_gain", "mean"),
            runtime_mean=("runtime_sec", "mean"),
        )
    )

    summary_path = cfg.OUTPUT_DIR / "compare_dncnn_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nComparison summary:")
    print(summary.round(4).to_string(index=False))

    save_example_grid(
        example_bank["gaussian"],
        cfg.OUTPUT_DIR / "dncnn_examples_gaussian.png",
        "DnCNN vs Classical Methods | Gaussian Noise"
    )

    save_example_grid(
        example_bank["poisson"],
        cfg.OUTPUT_DIR / "dncnn_examples_poisson.png",
        "DnCNN vs Classical Methods | Poisson Noise"
    )

    save_example_grid(
        example_bank["gaussian"],
        docs_fig_dir / "dncnn_examples_gaussian.png",
        "DnCNN vs Classical Methods | Gaussian Noise"
    )

    save_example_grid(
        example_bank["poisson"],
        docs_fig_dir / "dncnn_examples_poisson.png",
        "DnCNN vs Classical Methods | Poisson Noise"
    )

    print("\nSaved files:")
    print(f"- {detail_path}")
    print(f"- {summary_path}")
    print(f"- {cfg.OUTPUT_DIR / 'dncnn_examples_gaussian.png'}")
    print(f"- {cfg.OUTPUT_DIR / 'dncnn_examples_poisson.png'}")


if __name__ == "__main__":
    main()