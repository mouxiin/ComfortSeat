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


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.7,
            "lines.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.minorticks_off()
    ax.tick_params(axis="both", direction="in", length=3.0, width=0.6, top=True, right=True)
    ax.grid(True, linewidth=0.35, alpha=0.18, linestyle="--")
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)


def read_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> Path:
    apply_style()
    rows = read_rows(DATA_DIR / "fixed_shaker_disturbance.csv")
    time = np.asarray([float(row["time"]) for row in rows], dtype=float)
    disturbance = np.asarray([float(row["shaker_force_n"]) for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(3.5, 1.85))
    ax.plot(time, disturbance, color="#000000", linewidth=0.8)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Shaker-side disturbance [N]")
    style_axis(ax)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.17, top=0.98)
    out = FIGURES_DIR / "fixed_shaker_disturbance.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved figure: {out.resolve()}")
    return out


if __name__ == "__main__":
    main()
