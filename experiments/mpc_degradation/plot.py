from __future__ import annotations

import os
import csv
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
MPL_CACHE_DIR = SCRIPT_DIR / ".mpl_cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib.pyplot as plt

DATA_PATH = SCRIPT_DIR / "data" / "mpc_degradation.csv"
FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PLOT_TIME_MIN = 12.0
PLOT_TIME_MAX = 12.5

plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

MAIN_LINE_WIDTH = 0.8
AXIS_LINE_WIDTH = 0.7
GRID_LINE_WIDTH = 0.35
GRID_ALPHA = 0.18


def get_case_style(case_name):
    normalized = str(case_name).lower().strip()
    if "ideal" in normalized:
        return (0.0, 0.0, 0.0), "-", "Ideal plant"
    if normalized == "variant i":
        return (0 / 255.0, 114 / 255.0, 178 / 255.0), "--", "Variant I"
    if normalized == "variant ii":
        return (213 / 255.0, 94 / 255.0, 0 / 255.0), "-.", "Variant II"
    if normalized == "variant iii":
        return (0 / 255.0, 158 / 255.0, 115 / 255.0), ":", "Variant III"
    if normalized == "variant iv":
        return (204 / 255.0, 121 / 255.0, 167 / 255.0), "-", "Variant IV"
    return "0.5", "-", str(case_name)


def style_axis(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_LINE_WIDTH)
    ax.tick_params(axis="both", which="major", direction="in", length=3.0, width=0.6, top=True, right=True)
    ax.minorticks_off()
    ax.grid(True, linewidth=GRID_LINE_WIDTH, alpha=GRID_ALPHA)


def main(data_path=DATA_PATH, save_path=None):
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Run main.py first; missing {data_path}")
    rows = []
    with data_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        raise ValueError(f"Empty data file: {data_path}")
    names = []
    for row in rows:
        if row["case"] not in names:
            names.append(row["case"])
    time_s = np.asarray([float(row["time"]) for row in rows if row["case"] == names[0]], dtype=np.float64)
    x2 = np.zeros((len(names), time_s.size), dtype=np.float64)
    v2 = np.zeros_like(x2)
    a2 = np.zeros_like(x2)
    for idx, name in enumerate(names):
        case_rows = [row for row in rows if row["case"] == name]
        x2[idx] = [float(row["x2_filtered"]) for row in case_rows]
        v2[idx] = [float(row["v2_filtered"]) for row in case_rows]
        a2[idx] = [float(row["a2_filtered"]) for row in case_rows]

    mask = (time_s >= PLOT_TIME_MIN) & (time_s <= PLOT_TIME_MAX)
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.0), gridspec_kw={"wspace": 0.32})
    series = [(1.0e3 * x2, "$x_2$ [mm]"), (v2, "$v_2$ [m/s]"), (a2, "$a_2$ [m/s$^2$]")]

    for ax, (values, ylabel) in zip(axes, series):
        for idx, name in enumerate(names):
            color, linestyle, label = get_case_style(name)
            ax.plot(time_s[mask], values[idx, mask], color=color, linestyle=linestyle, linewidth=MAIN_LINE_WIDTH, label=label)
        ax.set_xlim(PLOT_TIME_MIN, PLOT_TIME_MAX)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(ylabel)
        style_axis(ax)

    axes[0].set_ylim(-10.0, 17.0)
    axes[0].legend(
        loc="upper right",
        frameon=True,
        framealpha=0.85,
        borderpad=0.25,
        labelspacing=0.20,
        handlelength=1.5,
        handletextpad=0.40,
        columnspacing=0.55,
        ncol=2,
    )
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.26, top=0.94)

    if save_path is None:
        save_path = FIGURES_DIR / "mpc_degradation_x2_v2_a2.pdf"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved figure: {save_path.resolve()}")
    return save_path


if __name__ == "__main__":
    main()
