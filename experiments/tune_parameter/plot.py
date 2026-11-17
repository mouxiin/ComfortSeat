from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MPL_CACHE_DIR = SCRIPT_DIR / ".mpl_cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR.resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {
    "passive": "#000000",
    "adrc": "#D55E00",
    "mpc": "#009E73",
    "deepc": "#CC79A7",
}
LABELS = {
    "passive": "Passive",
    "adrc": "ADRC",
    "mpc": "MPC",
    "deepc": "DeePC",
}


def apply_style():
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.7,
        }
    )


def style_axis(ax, grid_axis="both"):
    ax.tick_params(axis="both", direction="in", length=3.0, width=0.6, top=True, right=True)
    ax.grid(True, axis=grid_axis, linewidth=0.35, alpha=0.18)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)


def read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(row, key):
    return float(row[key])


def plot_comparison():
    rows = read_csv(DATA_DIR / "comparison_rollouts.csv")
    grouped = {}
    for row in rows:
        grouped.setdefault(row["method"], []).append(row)
    methods = [method for method in ("passive", "adrc", "mpc", "deepc") if method in grouped]
    fig, axes = plt.subplots(3, 2, figsize=(6.5, 4.5), sharex="col", gridspec_kw={"wspace": 0.30, "hspace": 0.10})
    axes = axes.reshape(-1)
    for method in methods:
        data = grouped[method]
        t = np.asarray([f(row, "time") for row in data])
        color = COLORS.get(method, "0.4")
        label = LABELS.get(method, method)
        axes[0].plot(t, 1.0e3 * np.asarray([f(row, "x2") for row in data]), color=color, linewidth=0.8, label=label)
        axes[1].plot(t, np.asarray([f(row, "v2") for row in data]), color=color, linewidth=0.8, label=label)
        axes[2].plot(t, np.asarray([f(row, "a2") for row in data]), color=color, linewidth=0.8, label=label)
        axes[3].plot(t, np.asarray([f(row, "u") for row in data]), color=color, linewidth=0.8, label=label)
    first = grouped[methods[0]]
    t = np.asarray([f(row, "time") for row in first])
    d = np.asarray([f(row, "shaker_force_n") for row in first])
    axes[4].plot(t, d, color="#5B3886", linewidth=0.8, label="Shaker")
    best = read_csv(DATA_DIR / "best_by_method.csv")
    methods_bar = [row["method"] for row in best]
    x = np.arange(len(best))
    axes[5].bar(x, [f(row, "objective") for row in best], color=[COLORS.get(m, "0.4") for m in methods_bar], edgecolor="black", linewidth=0.7)
    axes[5].set_xticks(x)
    axes[5].set_xticklabels([LABELS.get(m, m) for m in methods_bar], rotation=25, ha="right")

    axes[0].set_ylabel("$x_2$ [mm]")
    axes[1].set_ylabel("$v_2$ [m/s]")
    axes[2].set_ylabel("$a_2$ [m/s$^2$]")
    axes[3].set_ylabel("Action [N]")
    axes[4].set_ylabel("Shaker [N]")
    axes[5].set_ylabel("Objective")
    axes[4].set_xlabel("Time [s]")
    axes[5].set_xlabel("Method")
    for ax in axes:
        style_axis(ax)
    axes[0].legend(loc="upper right", ncol=2, frameon=True, framealpha=0.88, handlelength=1.5, columnspacing=0.65)
    axes[4].legend(loc="upper right", frameon=True, framealpha=0.88, handlelength=1.5)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.12, top=0.97)
    out = FIGURES_DIR / "tune_parameter_comparison.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved figure: {out.resolve()}")


def plot_fixed_shaker_disturbance():
    rows = read_csv(DATA_DIR / "fixed_shaker_disturbance.csv")
    t = np.asarray([f(row, "time") for row in rows])
    d = np.asarray([f(row, "shaker_force_n") for row in rows])

    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    ax.plot(t, d, color="#5B3886", linewidth=0.8)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Disturbance [N]")
    style_axis(ax)
    fig.subplots_adjust(left=0.15, right=0.985, bottom=0.18, top=0.97)
    out = FIGURES_DIR / "fixed_shaker_disturbance.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved figure: {out.resolve()}")


def main():
    apply_style()
    plot_comparison()
    plot_fixed_shaker_disturbance()


if __name__ == "__main__":
    main()
