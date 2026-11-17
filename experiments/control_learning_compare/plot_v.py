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
from matplotlib.lines import Line2D


DISPLAY_NAMES = {
    "passive": "Passive",
    "adrc": "ADRC",
    "non_residual": "NR-PPO",
    "fixed_residual": "FR-PPO",
    "adaptive_residual": "AR-PPO",
}
METHOD_ORDER = ["passive", "adrc", "non_residual", "fixed_residual", "adaptive_residual"]
CURVE_METHODS = ["non_residual", "fixed_residual", "adaptive_residual"]
BASELINE_METHODS = ["passive", "adrc"]
COLORS = {
    "passive": "#000000",
    "adrc": "#0072B2",
    "non_residual": "#D55E00",
    "fixed_residual": "#009E73",
    "adaptive_residual": "#CC79A7",
}
LINESTYLES = {
    "passive": "-",
    "adrc": "-",
    "non_residual": "--",
    "fixed_residual": "-.",
    "adaptive_residual": ":",
}


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
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "lines.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.minorticks_off()
    ax.tick_params(axis="both", direction="in", length=3.0, width=0.6, top=True, right=True)
    ax.grid(True, linewidth=0.35, alpha=0.22, linestyle="--")
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)


def read_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        out.append(
            {
                "episode": int(float(row["episode"])),
                "method": row["method"],
                "average_return": float(row["average_return"]),
            }
        )
    return out


def grouped_curve(rows: list[dict], method: str):
    by_episode: dict[int, list[float]] = {}
    for row in rows:
        if row["method"] == method:
            by_episode.setdefault(row["episode"], []).append(row["average_return"])
    episodes = np.asarray(sorted(by_episode), dtype=float)
    if episodes.size == 0:
        return episodes, episodes, episodes
    mean = np.asarray([np.mean(by_episode[int(ep)]) for ep in episodes], dtype=float)
    std = np.asarray([np.std(by_episode[int(ep)]) for ep in episodes], dtype=float)
    return episodes, mean, std


def main() -> Path:
    apply_style()
    rows = read_rows(DATA_DIR / "average_return_comparison.csv")
    fig, ax = plt.subplots(figsize=(3.5, 2.05))
    values = [row["average_return"] for row in rows]
    if values:
        pad = max(0.05, 0.04 * (max(values) - min(values) + 1.0e-9))
        ax.set_ylim(min(values) - pad, max(values) + pad)
    for method in BASELINE_METHODS:
        vals = [row["average_return"] for row in rows if row["method"] == method]
        if vals:
            ax.axhline(np.mean(vals), color=COLORS[method], linestyle=LINESTYLES[method], linewidth=0.8)
    for method in CURVE_METHODS:
        episodes, mean, std = grouped_curve(rows, method)
        if episodes.size == 0:
            continue
        ax.plot(episodes, mean, color=COLORS[method], linestyle=LINESTYLES[method], linewidth=0.8)
        if np.any(std > 0):
            ax.fill_between(episodes, mean - std, mean + std, color=COLORS[method], alpha=0.10, linewidth=0)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Average return")
    style_axis(ax)
    present = {row["method"] for row in rows}
    handles = [
        Line2D([0], [0], color=COLORS[m], linestyle=LINESTYLES[m], linewidth=0.8, label=DISPLAY_NAMES[m])
        for m in METHOD_ORDER
        if m in present
    ]
    ax.legend(
        handles=handles,
        loc="lower right",
        ncol=2,
        framealpha=0.92,
        borderpad=0.20,
        labelspacing=0.16,
        handlelength=1.35,
        columnspacing=0.55,
        handletextpad=0.32,
    )
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.16, top=0.98)
    out = FIGURES_DIR / "average_return_comparison.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved figure: {out.resolve()}")
    return out


if __name__ == "__main__":
    main()
