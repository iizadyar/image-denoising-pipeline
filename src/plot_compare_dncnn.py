from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as cfg


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


def grouped_bar(ax, df_sub, metric_mean, metric_std, title, ylabel, colors, methods, fmt="{:.2f}"):
    noise_types = ["gaussian", "poisson"]
    x = np.arange(len(noise_types))
    width = 0.23

    for j, method in enumerate(methods):
        vals = []
        errs = []

        for noise in noise_types:
            row = df_sub[(df_sub["noise_type"] == noise) & (df_sub["method"] == method)]
            if len(row) == 0:
                vals.append(np.nan)
                errs.append(np.nan)
            else:
                vals.append(float(row.iloc[0][metric_mean]))
                errs.append(float(row.iloc[0][metric_std]) if metric_std else 0.0)

        bars = ax.bar(
            x + (j - 1) * width,
            vals,
            width=width,
            yerr=errs,
            capsize=4,
            color=colors[method],
            edgecolor="black",
            linewidth=0.8,
            alpha=0.95,
            label=method.upper()
        )

        ymin, ymax = ax.get_ylim()
        yspan = max(ymax - ymin, 1e-8)

        for bar in bars:
            h = bar.get_height()
            if np.isnan(h):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.015 * yspan,
                fmt.format(h),
                ha="center",
                va="bottom",
                fontsize=8
            )

    ax.set_xticks(x)
    ax.set_xticklabels([n.upper() for n in noise_types])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Noise Type")


def save_dashboard(summary_df, out_path):
    set_plot_style()

    methods = ["bm3d", "wavelet", "dncnn"]
    colors = cfg.METHOD_COLORS

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    grouped_bar(
        axes[0, 0], summary_df,
        "den_psnr_mean", "den_psnr_std",
        "Denoised PSNR", "PSNR (dB)",
        colors, methods, fmt="{:.2f}"
    )

    grouped_bar(
        axes[0, 1], summary_df,
        "den_ssim_mean", "den_ssim_std",
        "Denoised SSIM", "SSIM",
        colors, methods, fmt="{:.3f}"
    )

    grouped_bar(
        axes[1, 0], summary_df,
        "psnr_gain_mean", None,
        "PSNR Gain", "Gain (dB)",
        colors, methods, fmt="{:.2f}"
    )

    grouped_bar(
        axes[1, 1], summary_df,
        "runtime_mean", None,
        "Runtime per Image", "Seconds",
        colors, methods, fmt="{:.3f}"
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3)

    fig.suptitle("DnCNN vs Classical Methods", fontsize=17, fontweight="bold")
    fig.savefig(out_path, dpi=cfg.SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def save_metric_plot(summary_df, metric_mean, metric_std, out_path, title, ylabel, fmt):
    set_plot_style()

    methods = ["bm3d", "wavelet", "dncnn"]
    colors = cfg.METHOD_COLORS

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    grouped_bar(ax, summary_df, metric_mean, metric_std, title, ylabel, colors, methods, fmt=fmt)
    ax.legend()
    fig.savefig(out_path, dpi=cfg.SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def save_compact_table(summary_df, out_path):
    set_plot_style()

    table_df = summary_df.copy()
    table_df = table_df[[
        "noise_type", "method",
        "den_psnr_mean", "den_ssim_mean",
        "psnr_gain_mean", "runtime_mean"
    ]].copy()

    table_df.columns = ["Noise", "Method", "PSNR", "SSIM", "PSNR Gain", "Runtime (s)"]

    table_df["PSNR"] = table_df["PSNR"].map(lambda x: f"{x:.2f}")
    table_df["SSIM"] = table_df["SSIM"].map(lambda x: f"{x:.3f}")
    table_df["PSNR Gain"] = table_df["PSNR Gain"].map(lambda x: f"{x:.2f}")
    table_df["Runtime (s)"] = table_df["Runtime (s)"].map(lambda x: f"{x:.3f}")

    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.axis("off")

    tbl = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center"
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.6)

    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#404040")
        else:
            cell.set_facecolor("#F2F2F2")

    plt.title("Compact DnCNN Comparison Table", fontsize=15, fontweight="bold", pad=18)
    fig.savefig(out_path, dpi=cfg.SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    docs_fig_dir = cfg.ROOT / "docs" / "figures"
    docs_fig_dir.mkdir(parents=True, exist_ok=True)

    summary_path = cfg.OUTPUT_DIR / "compare_dncnn_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"{summary_path} not found. Run compare_dncnn.py first.")

    summary_df = pd.read_csv(summary_path)

    save_dashboard(summary_df, cfg.OUTPUT_DIR / "dncnn_dashboard.png")
    save_metric_plot(
        summary_df, "den_psnr_mean", "den_psnr_std",
        cfg.OUTPUT_DIR / "dncnn_psnr.png",
        "PSNR Comparison", "PSNR (dB)", "{:.2f}"
    )
    save_metric_plot(
        summary_df, "den_ssim_mean", "den_ssim_std",
        cfg.OUTPUT_DIR / "dncnn_ssim.png",
        "SSIM Comparison", "SSIM", "{:.3f}"
    )
    save_compact_table(summary_df, cfg.OUTPUT_DIR / "dncnn_summary_table.png")

    save_dashboard(summary_df, docs_fig_dir / "dncnn_dashboard.png")
    save_metric_plot(
        summary_df, "den_psnr_mean", "den_psnr_std",
        docs_fig_dir / "dncnn_psnr.png",
        "PSNR Comparison", "PSNR (dB)", "{:.2f}"
    )
    save_metric_plot(
        summary_df, "den_ssim_mean", "den_ssim_std",
        docs_fig_dir / "dncnn_ssim.png",
        "SSIM Comparison", "SSIM", "{:.3f}"
    )
    save_compact_table(summary_df, docs_fig_dir / "dncnn_summary_table.png")

    print("Saved plots successfully.")


if __name__ == "__main__":
    main()