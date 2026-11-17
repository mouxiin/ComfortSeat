"""Standalone plotter for the shaker disturbance hardware sweep.

The script reads CSV files from this experiment directory only and writes PDF
figures in the style of the original frequency sweep plots.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
FIGURE_DIR = SCRIPT_DIR / "figures"
MPL_CACHE_DIR = SCRIPT_DIR / ".mpl_cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR.resolve()))

import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


METHOD_ORDER = ("passive", "adrc", "mpc", "deepc", "ar_ppo")
DISPLAY_NAMES = {
    "passive": "Passive",
    "adrc": "ADRC",
    "mpc": "MPC",
    "deepc": "DeePC",
    "ar_ppo": "AR-PPO",
}
RMS_BAR_COLORS = (
    (0 / 255, 114 / 255, 178 / 255),
    (213 / 255, 94 / 255, 0 / 255),
    (0 / 255, 158 / 255, 115 / 255),
    (204 / 255, 121 / 255, 167 / 255),
    (230 / 255, 159 / 255, 0 / 255),
)
ALL_CONTOUR_METRICS = (
    ("x2_rms", "$x_2$ RMS [m]"),
    ("v2_rms", "$v_2$ RMS [m/s]"),
    ("a2_rms", "$a_2$ RMS [m/s$^2$]"),
)

JOURNAL_FONT_SIZE = 8
CONTOUR_LABEL_FONT_SIZE = 6
JOURNAL_AXIS_WIDTH = 0.7
JOURNAL_GRID_WIDTH = 0.35
JOURNAL_GRID_ALPHA = 0.18
JOURNAL_TICK_WIDTH = 0.6
JOURNAL_TICK_LENGTH = 3.0


def read_rows(path: Path) -> list[dict]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "font.size": JOURNAL_FONT_SIZE,
            "axes.titlesize": JOURNAL_FONT_SIZE,
            "axes.labelsize": JOURNAL_FONT_SIZE,
            "xtick.labelsize": JOURNAL_FONT_SIZE,
            "ytick.labelsize": JOURNAL_FONT_SIZE,
            "legend.fontsize": JOURNAL_FONT_SIZE,
            "axes.linewidth": JOURNAL_AXIS_WIDTH,
            "lines.linewidth": 0.8,
            "grid.linewidth": JOURNAL_GRID_WIDTH,
            "grid.alpha": JOURNAL_GRID_ALPHA,
            "grid.linestyle": "--",
        }
    )


def apply_axis_style(ax, show_grid=False) -> None:
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        length=JOURNAL_TICK_LENGTH,
        width=JOURNAL_TICK_WIDTH,
        top=True,
        right=True,
    )
    ax.minorticks_off()
    for spine in ax.spines.values():
        spine.set_linewidth(JOURNAL_AXIS_WIDTH)
    if show_grid:
        ax.grid(True, linewidth=JOURNAL_GRID_WIDTH, alpha=JOURNAL_GRID_ALPHA, linestyle="--")


def load_metric_grids(path: Path):
    rows = read_rows(path)
    if not rows:
        raise FileNotFoundError(f"No aggregate rows found in {path.resolve()}")
    frequencies = np.asarray(sorted({float(r["frequency_hz"]) for r in rows}), dtype=np.float64)
    amplitudes = np.asarray(sorted({float(r["force_amplitude_n"]) for r in rows}), dtype=np.float64)
    methods = [m for m in METHOD_ORDER if any(r["controller"] == m for r in rows)]
    metric_grids = {}
    for method in methods:
        metric_grids[method] = {
            "x2_rms": np.full((len(amplitudes), len(frequencies)), np.nan, dtype=np.float64),
            "v2_rms": np.full((len(amplitudes), len(frequencies)), np.nan, dtype=np.float64),
            "a2_rms": np.full((len(amplitudes), len(frequencies)), np.nan, dtype=np.float64),
        }
    freq_index = {float(v): i for i, v in enumerate(frequencies)}
    amp_index = {float(v): i for i, v in enumerate(amplitudes)}
    for row in rows:
        method = row["controller"]
        if method not in metric_grids:
            continue
        ai = amp_index[float(row["force_amplitude_n"])]
        fi = freq_index[float(row["frequency_hz"])]
        for metric in ("x2_rms", "v2_rms", "a2_rms"):
            metric_grids[method][metric][ai, fi] = float(row[f"{metric}_mean"])
    return amplitudes, frequencies, metric_grids


def common_limits(method_metrics, metric_name):
    values = np.stack([method_metrics[method][metric_name] for method in method_metrics], axis=0)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if np.isclose(vmin, vmax):
        delta = max(abs(vmin) * 1.0e-6, 1.0e-6)
        vmin -= delta
        vmax += delta
    return vmin, vmax


def draw_panel(ax, amplitudes, frequencies, values, vmin, vmax):
    frequencies = np.asarray(frequencies, dtype=float)
    amplitudes = np.asarray(amplitudes, dtype=float)
    values = np.asarray(values, dtype=float)
    image = ax.imshow(
        np.clip(values, vmin, vmax),
        origin="lower",
        extent=[
            float(np.min(frequencies)),
            float(np.max(frequencies)),
            float(np.min(amplitudes)),
            float(np.max(amplitudes)),
        ],
        cmap="jet",
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        interpolation="bilinear",
    )
    if len(amplitudes) >= 2 and len(frequencies) >= 2 and np.isfinite(values).any():
        freq_grid, amp_grid = np.meshgrid(frequencies, amplitudes)
        local_min = float(np.nanmin(values))
        local_max = float(np.nanmax(values))
        if not np.isclose(local_min, local_max):
            levels = np.linspace(local_min, local_max, 7)
            lines = ax.contour(freq_grid, amp_grid, values, levels=levels, colors="black", linewidths=0.35, alpha=0.75)
            ax.clabel(lines, inline=True, fontsize=CONTOUR_LABEL_FONT_SIZE, fmt="%.2g")
    ax.set_xlim(float(np.min(frequencies)), float(np.max(frequencies)))
    ax.set_ylim(float(np.min(amplitudes)), float(np.max(amplitudes)))
    ax.set_box_aspect(1)
    ax.set_xlabel("Freq. [Hz]")
    apply_axis_style(ax, show_grid=False)
    return image


def plot_a2_v2_contours_all(save_prefix: str, output_dir: Path, amplitudes, frequencies, method_metrics):
    methods = list(method_metrics.keys())
    fig, axes = plt.subplots(len(ALL_CONTOUR_METRICS), len(methods), figsize=(7.0, 3.5), squeeze=False)
    for col, method in enumerate(methods):
        axes[0, col].set_title(DISPLAY_NAMES.get(method, method))
    for row, (metric_name, label) in enumerate(ALL_CONTOUR_METRICS):
        vmin, vmax = common_limits(method_metrics, metric_name)
        for col, method in enumerate(methods):
            ax = axes[row, col]
            draw_panel(ax, amplitudes, frequencies, method_metrics[method][metric_name], vmin, vmax)
            if row != len(ALL_CONTOUR_METRICS) - 1:
                ax.set_xlabel("")
                ax.tick_params(axis="x", labelbottom=False)
            if col == 0:
                ax.set_ylabel("Amp. [N]")
            else:
                ax.tick_params(axis="y", labelleft=False)
        cax = axes[row, -1].inset_axes([1.08, 0.0, 0.07, 1.0])
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        mappable = cm.ScalarMappable(norm=norm, cmap="jet")
        mappable.set_array([])
        colorbar = fig.colorbar(mappable, cax=cax)
        colorbar.set_label(label, labelpad=5)
        colorbar.locator = MaxNLocator(nbins=5)
        colorbar.formatter = FormatStrFormatter("%.3g")
        colorbar.update_ticks()
        colorbar.ax.tick_params(direction="in", length=JOURNAL_TICK_LENGTH, width=JOURNAL_TICK_WIDTH, labelsize=JOURNAL_FONT_SIZE)
        colorbar.outline.set_linewidth(JOURNAL_AXIS_WIDTH)
    fig.subplots_adjust(left=0.07, right=0.89, top=0.93, bottom=0.10, wspace=0.12, hspace=0.06)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{save_prefix}_plot_a2_v2_contours_all.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_rms_all_bar(save_prefix: str, output_dir: Path, method_metrics):
    metrics = (
        ("x2_rms", "$x_2$ RMS [m]"),
        ("v2_rms", "$v_2$ RMS [m/s]"),
        ("a2_rms", "$a_2$ RMS [m/s$^2$]"),
    )
    methods = list(method_metrics.keys())
    x = np.arange(len(methods), dtype=float)
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.5, 2.0), sharey=False)
    bar_colors = [RMS_BAR_COLORS[idx % len(RMS_BAR_COLORS)] for idx in range(len(methods))]
    for ax, (metric_name, label) in zip(axes, metrics):
        means = []
        stds = []
        for method in methods:
            values = np.asarray(method_metrics[method][metric_name], dtype=float).reshape(-1)
            values = values[np.isfinite(values)]
            means.append(float(np.mean(values)) if values.size else 0.0)
            stds.append(float(np.std(values, ddof=1)) if values.size > 1 else 0.0)
        ax.bar(
            x,
            means,
            0.58,
            color=bar_colors,
            edgecolor="black",
            linewidth=JOURNAL_AXIS_WIDTH,
            yerr=stds,
            error_kw={"elinewidth": JOURNAL_AXIS_WIDTH, "ecolor": "black", "capsize": 2},
        )
        ax.set_xticks(x)
        ax.set_xticklabels([DISPLAY_NAMES.get(method, method) for method in methods], rotation=25, ha="right", fontstyle="italic", fontsize=6)
        ax.set_ylabel(label)
        upper = max(mean + std for mean, std in zip(means, stds)) if means else 1.0
        ax.set_ylim(0.0, max(upper * 1.12, 1.0e-12))
        apply_axis_style(ax, show_grid=True)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.27, top=0.94, wspace=0.35)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{save_prefix}_plot_rms_all_bar.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot shaker disturbance sweep PDFs from CSV summary data.")
    parser.add_argument("--summary", type=str, default=str(DATA_DIR / "aggregate_statistics.csv"))
    parser.add_argument("--prefix", type=str, default="frequency_test_all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_plot_style()
    amplitudes, frequencies, method_metrics = load_metric_grids(Path(args.summary))
    saved = [
        plot_a2_v2_contours_all(args.prefix, FIGURE_DIR, amplitudes, frequencies, method_metrics),
        plot_rms_all_bar(args.prefix, FIGURE_DIR, method_metrics),
    ]
    for path in saved:
        print(f"Saved figure: {path.resolve()}")


if __name__ == "__main__":
    main()
