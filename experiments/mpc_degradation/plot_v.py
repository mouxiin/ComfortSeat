from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "data" / "mpc_degradation.csv"
FIGURES_DIR = SCRIPT_DIR / "figures_v"
MPL_CACHE_DIR = SCRIPT_DIR / ".mpl_cache"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR.resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, MaxNLocator


TIME_MIN = 10.0
TIME_MAX = 11.0
FONT_SIZE = 8
LINE_WIDTH = 0.75
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
            "legend.fontsize": FONT_SIZE - 1,
            "axes.linewidth": AXIS_WIDTH,
            "lines.linewidth": LINE_WIDTH,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.minorticks_off()
    ax.tick_params(axis="both", direction="in", length=3.0, width=0.6, top=True, right=True)
    ax.grid(True, linewidth=GRID_WIDTH, alpha=GRID_ALPHA, linestyle="--")
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_WIDTH)


def case_style(case_name: str):
    key = case_name.lower().strip()
    if "ideal" in key:
        return "#000000", "-", "Ideal plant"
    if key == "variant i":
        return "#0072B2", "--", "Variant I"
    if key == "variant ii":
        return "#D55E00", "-.", "Variant II"
    if key == "variant iii":
        return "#009E73", ":", "Variant III"
    if key == "variant iv":
        return "#CC79A7", "-", "Variant IV"
    return "0.45", "-", case_name


def read_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> Path:
    apply_style()
    rows = read_rows(DATA_PATH)
    cases = []
    for row in rows:
        if row["case"] not in cases:
            cases.append(row["case"])

    first = [row for row in rows if row["case"] == cases[0]]
    time = np.asarray([float(row["time"]) for row in first], dtype=float)
    mask = (time >= TIME_MIN) & (time <= TIME_MAX)
    data = {}
    for case in cases:
        case_rows = [row for row in rows if row["case"] == case]
        data[case] = {
            "q2": np.asarray([float(row["x2_filtered"]) for row in case_rows], dtype=float),
            "dq2": np.asarray([float(row["v2_filtered"]) for row in case_rows], dtype=float),
            "ddq2": np.asarray([float(row["a2_filtered"]) for row in case_rows], dtype=float),
        }

    fig, axes = plt.subplots(3, 1, figsize=(3.5, 2.85), sharex=True)
    panels = [
        ("q2", r"$q_2$ [m]"),
        ("dq2", r"$\dot{q}_2$ [m/s]"),
        ("ddq2", r"$\ddot{q}_2$ [m/s$^2$]"),
    ]
    for ax, (key, ylabel) in zip(axes, panels):
        for case in cases:
            color, linestyle, label = case_style(case)
            ax.plot(time[mask], data[case][key][mask], color=color, linestyle=linestyle, label=label)
        ax.set_ylabel(ylabel)
        ax.set_xlim(TIME_MIN, TIME_MAX)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        style_axis(ax)
    axes[-1].set_xlabel("Time [s]")
    axes[-1].xaxis.set_major_locator(FixedLocator([10.0, 10.5, 11.0]))
    axes[0].set_ylim(-0.006, 0.013)
    axes[0].legend(
        loc="upper right",
        ncol=3,
        framealpha=0.92,
        borderpad=0.20,
        labelspacing=0.18,
        handlelength=1.45,
        columnspacing=0.55,
        handletextpad=0.35,
    )
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.12, top=0.98, hspace=0.10)
    out = FIGURES_DIR / "mpc_degradation_x2_v2_a2.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved figure: {out.resolve()}")
    return out


if __name__ == "__main__":
    main()
