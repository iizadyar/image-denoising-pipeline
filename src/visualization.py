from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def set_plot_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "font.size": 10,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })


def _show(ax, img, title=""):
    if img.ndim == 2:
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    else:
        ax.imshow(img)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def _annotate_bars(ax, bars, fmt="{:.2f}"):
    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin if ymax > ymin else 1.0

    for bar in bars:
        h = bar.get_height()
        if np.isnan(h):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.012 * yspan,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=8
        )


def save_dataset_overview(images, path: Path, title="Dataset overview", ncols=4, dpi=220):
    if not images:
        return

    set_plot_style()
    n = len(images)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 3.1 * nrows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for i, img in enumerate(images):
        _show(axes[i], img, title=f"sample {i}")

    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_examples_grid(example_rows, path: Path, title: str, dpi=220):
    if not example_rows:
        return

    set_plot_style()
    nrows = len(example_rows)

    fig, axes = plt.subplots(
        nrows, 4,
        figsize=(16, 3.4 * nrows),
        constrained_layout=True
    )

    if nrows == 1:
        axes = np.array([axes])

    for r, row in enumerate(example_rows):
        img_id, clean, noisy, bm3d_img, wavelet_img = row
        _show(axes[r, 0], clean, title=f"{img_id} | Ground truth")
        _show(axes[r, 1], noisy, title=f"{img_id} | Noisy")
        _show(axes[r, 2], bm3d_img, title=f"{img_id} | BM3D")
        _show(axes[r, 3], wavelet_img, title=f"{img_id} | Wavelet")

    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_metric_comparison(
    summary_df: pd.DataFrame,
    mean_col: str,
    std_col: str,
    title: str,
    ylabel: str,
    path: Path,
    dpi=220,
    value_fmt="{:.2f}",
    colors=None
):
    """
    Creates one clean figure with two subplots:
    - left: Gaussian
    - right: Poisson
    Bars: BM3D vs Wavelet
    X-axis: RGB vs GRAY
    """
    set_plot_style()

    if colors is None:
        colors = {"bm3d": "#4E79A7", "wavelet": "#59A14F"}

    noise_order = ["gaussian", "poisson"]
    mode_order = ["rgb", "gray"]
    method_order = ["bm3d", "wavelet"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True, constrained_layout=True)

    x = np.arange(len(mode_order))
    width = 0.34

    handles = [
        Patch(facecolor=colors["bm3d"], label="BM3D"),
        Patch(facecolor=colors["wavelet"], label="Wavelet")
    ]

    # Determine a stable upper bound across both subplots
    all_vals = []
    for noise_type in noise_order:
        df_n = summary_df[summary_df["noise_type"] == noise_type]
        for method in method_order:
            for mode in mode_order:
                row = df_n[(df_n["mode"] == mode) & (df_n["method"] == method)]
                if len(row) > 0:
                    val = float(row.iloc[0][mean_col])
                    err = float(row.iloc[0][std_col])
                    all_vals.append(val + err)

    upper = max(all_vals) * 1.18 if len(all_vals) > 0 else None

    for ax, noise_type in zip(axes, noise_order):
        df_n = summary_df[summary_df["noise_type"] == noise_type]

        for j, method in enumerate(method_order):
            vals = []
            errs = []

            for mode in mode_order:
                row = df_n[(df_n["mode"] == mode) & (df_n["method"] == method)]
                if len(row) == 0:
                    vals.append(np.nan)
                    errs.append(np.nan)
                else:
                    vals.append(float(row.iloc[0][mean_col]))
                    errs.append(float(row.iloc[0][std_col]))

            bars = ax.bar(
                x + (j - 0.5) * width,
                vals,
                width=width,
                yerr=errs,
                capsize=4,
                color=colors[method],
                alpha=0.95
            )
            _annotate_bars(ax, bars, fmt=value_fmt)

        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in mode_order])
        ax.set_xlabel("Mode")
        ax.set_title(noise_type.upper())
        ax.set_ylabel(ylabel)

        if upper is not None:
            ax.set_ylim(0, upper)

    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_boxplot_by_noise(
    detail_df: pd.DataFrame,
    metric_col: str,
    title: str,
    ylabel: str,
    path: Path,
    dpi=220,
    colors=None
):
    set_plot_style()

    if colors is None:
        colors = {"bm3d": "#4E79A7", "wavelet": "#59A14F"}

    noise_order = ["gaussian", "poisson"]
    order = [
        ("rgb", "bm3d"),
        ("rgb", "wavelet"),
        ("gray", "bm3d"),
        ("gray", "wavelet"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True, constrained_layout=True)

    for ax, noise_type in zip(axes, noise_order):
        df_n = detail_df[detail_df["noise_type"] == noise_type]

        groups = []
        labels = []
        box_colors = []

        for mode, method in order:
            grp = df_n[(df_n["mode"] == mode) & (df_n["method"] == method)]
            if len(grp) > 0:
                groups.append(grp[metric_col].values)
                labels.append(f"{mode.upper()}\n{method.upper()}")
                box_colors.append(colors[method])

        bp = ax.boxplot(groups, labels=labels, showfliers=False, patch_artist=True)

        for patch, c in zip(bp["boxes"], box_colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)

        ax.set_title(noise_type.upper())
        ax.set_ylabel(ylabel)

    handles = [
        Patch(facecolor=colors["bm3d"], label="BM3D", alpha=0.75),
        Patch(facecolor=colors["wavelet"], label="Wavelet", alpha=0.75),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_summary_table(summary_df: pd.DataFrame, path: Path, dpi=220):

    set_plot_style()

    table_df = summary_df.copy()

    display_df = table_df[[
        "noise_type", "mode", "method",
        "den_psnr_mean", "den_ssim_mean",
        "psnr_gain_mean", "ssim_gain_mean",
        "runtime_mean"
    ]].copy()

    display_df.columns = [
        "Noise", "Mode", "Method",
        "PSNR", "SSIM",
        "PSNR Gain", "SSIM Gain",
        "Runtime (s)"
    ]

    display_df["PSNR"] = display_df["PSNR"].round(2)
    display_df["SSIM"] = display_df["SSIM"].round(3)
    display_df["PSNR Gain"] = display_df["PSNR Gain"].round(2)
    display_df["SSIM Gain"] = display_df["SSIM Gain"].round(3)
    display_df["Runtime (s)"] = display_df["Runtime (s)"].round(3)

    fig_height = 0.55 * len(display_df) + 1.5
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis("off")

    tbl = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center"
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.5)

    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#3B3B3B")
        else:
            if row % 2 == 0:
                cell.set_facecolor("#F7F7F7")
            else:
                cell.set_facecolor("white")

    plt.title("Compact Summary Table", fontsize=16, fontweight="bold", pad=14)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)