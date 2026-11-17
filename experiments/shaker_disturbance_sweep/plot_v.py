from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
FIGURES_DIR = SCRIPT_DIR / "figures_v"
MPL_CACHE_DIR = SCRIPT_DIR / ".mpl_cache"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR.resolve()))

import matplotlib

matplotlib.use("Agg")
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
BAR_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")
METRICS = (
    ("x2_rms", r"$q_2$ RMS [m]"),
    ("v2_rms", r"$\dot{q}_2$ RMS [m/s]"),
    ("a2_rms", r"$\ddot{q}_2$ RMS [m/s$^2$]"),
)
FONT_SIZE = 8
AXIS_WIDTH = 0.7
GRID_WIDTH = 0.35
GRID_ALPHA = 0.18


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
            "axes.linewidth": AXIS_WIDTH,
            "lines.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(ax: plt.Axes, grid: bool = False) -> None:
    ax.minorticks_off()
    ax.tick_params(axis="both", direction="in", length=3.0, width=0.6, top=True, right=True)
    if grid:
        ax.grid(True, linewidth=GRID_WIDTH, alpha=GRID_ALPHA, linestyle="--")
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_WIDTH)


def read_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_metric_grids(path: Path):
    rows = read_rows(path)
    freqs = np.asarray(sorted({float(row["frequency_hz"]) for row in rows}), dtype=float)
    amps = np.asarray(sorted({float(row["force_amplitude_n"]) for row in rows}), dtype=float)
    methods = [m for m in METHOD_ORDER if any(row["controller"] == m for row in rows)]
    grids = {
        method: {metric: np.full((len(amps), len(freqs)), np.nan, dtype=float) for metric, _ in METRICS}
        for method in methods
    }
    freq_index = {float(v): idx for idx, v in enumerate(freqs)}
    amp_index = {float(v): idx for idx, v in enumerate(amps)}
    for row in rows:
        method = row["controller"]
        if method not in grids:
            continue
        ai = amp_index[float(row["force_amplitude_n"])]
        fi = freq_index[float(row["frequency_hz"])]
        for metric, _ in METRICS:
            grids[method][metric][ai, fi] = float(row[f"{metric}_mean"])
    return amps, freqs, grids


def common_limits(grids: dict, metric: str):
    values = np.stack([grids[method][metric] for method in grids], axis=0)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if np.isclose(vmin, vmax):
        delta = max(abs(vmin) * 1.0e-6, 1.0e-6)
        return vmin - delta, vmax + delta
    return vmin, vmax


def draw_panel(ax: plt.Axes, amps: np.ndarray, freqs: np.ndarray, values: np.ndarray, vmin: float, vmax: float):
    image = ax.imshow(
        np.clip(values, vmin, vmax),
        origin="lower",
        extent=[float(freqs.min()), float(freqs.max()), float(amps.min()), float(amps.max())],
        cmap="jet",
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        interpolation="bilinear",
    )
    if len(amps) >= 2 and len(freqs) >= 2 and np.isfinite(values).any():
        freq_grid, amp_grid = np.meshgrid(freqs, amps)
        local_min = float(np.nanmin(values))
        local_max = float(np.nanmax(values))
        if not np.isclose(local_min, local_max):
            levels = np.linspace(local_min, local_max, 7)
            lines = ax.contour(freq_grid, amp_grid, values, levels=levels, colors="black", linewidths=0.35, alpha=0.75)
            ax.clabel(lines, inline=True, fontsize=6, fmt="%.2g")
    ax.set_xlim(float(freqs.min()), float(freqs.max()))
    ax.set_ylim(float(amps.min()), float(amps.max()))
    ax.set_box_aspect(1)
    ax.set_xlabel("Freq. [Hz]")
    style_axis(ax, grid=False)
    return image


def plot_contours(amps: np.ndarray, freqs: np.ndarray, grids: dict) -> Path:
    methods = list(grids.keys())
    fig, axes = plt.subplots(len(METRICS), len(methods), figsize=(7.0, 3.5), squeeze=False)
    for col, method in enumerate(methods):
        axes[0, col].set_title(DISPLAY_NAMES.get(method, method))
    for row_idx, (metric, label) in enumerate(METRICS):
        vmin, vmax = common_limits(grids, metric)
        for col, method in enumerate(methods):
            ax = axes[row_idx, col]
            draw_panel(ax, amps, freqs, grids[method][metric], vmin, vmax)
            if row_idx != len(METRICS) - 1:
                ax.set_xlabel("")
                ax.tick_params(axis="x", labelbottom=False)
            if col == 0:
                ax.set_ylabel("Amp. [N]")
            else:
                ax.tick_params(axis="y", labelleft=False)
        cax = axes[row_idx, -1].inset_axes([1.08, 0.0, 0.07, 1.0])
        mappable = cm.ScalarMappable(norm=mcolors.Normalize(vmin=vmin, vmax=vmax), cmap="jet")
        mappable.set_array([])
        cbar = fig.colorbar(mappable, cax=cax)
        cbar.set_label(label, labelpad=5)
        cbar.locator = MaxNLocator(nbins=5)
        cbar.formatter = FormatStrFormatter("%.3g")
        cbar.update_ticks()
        cbar.ax.tick_params(direction="in", length=3.0, width=0.6, labelsize=FONT_SIZE)
        cbar.outline.set_linewidth(AXIS_WIDTH)
    fig.subplots_adjust(left=0.07, right=0.89, top=0.93, bottom=0.10, wspace=0.12, hspace=0.06)
    out = FIGURES_DIR / "frequency_test_all_plot_a2_v2_contours_all.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_rms_bars(grids: dict) -> Path:
    methods = list(grids.keys())
    x = np.arange(len(methods), dtype=float)
    fig, axes = plt.subplots(3, 1, figsize=(3.5, 3.35), sharex=True)
    colors = [BAR_COLORS[idx % len(BAR_COLORS)] for idx in range(len(methods))]
    for ax, (metric, ylabel) in zip(axes, METRICS):
        means = []
        stds = []
        for method in methods:
            values = np.asarray(grids[method][metric], dtype=float).reshape(-1)
            values = values[np.isfinite(values)]
            means.append(float(np.mean(values)) if values.size else 0.0)
            stds.append(float(np.std(values, ddof=1)) if values.size > 1 else 0.0)
        ax.bar(
            x,
            means,
            0.58,
            color=colors,
            edgecolor="black",
            linewidth=AXIS_WIDTH,
            yerr=stds,
            error_kw={"elinewidth": AXIS_WIDTH, "ecolor": "black", "capsize": 2},
        )
        ax.set_ylabel(ylabel)
        upper = max(mean + std for mean, std in zip(means, stds)) if means else 1.0
        ax.set_ylim(0.0, max(upper * 1.12, 1.0e-12))
        style_axis(ax, grid=True)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([DISPLAY_NAMES.get(method, method) for method in methods], rotation=25, ha="right", fontsize=7)
    axes[-1].set_xlabel("Controller")
    for ax in axes[:-1]:
        ax.tick_params(axis="x", labelbottom=False)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.16, top=0.98, hspace=0.10)
    out = FIGURES_DIR / "frequency_test_all_plot_rms_all_bar.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    apply_style()
    amps, freqs, grids = load_metric_grids(DATA_DIR / "aggregate_statistics.csv")
    saved = [plot_contours(amps, freqs, grids), plot_rms_bars(grids)]
    for path in saved:
        print(f"Saved figure: {path.resolve()}")


if __name__ == "__main__":
    main()
