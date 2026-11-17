from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
FIGURES_DIR = SCRIPT_DIR / "figures"
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

SINGLE_FIGSIZE = (3.0, 2.0)
MAIN_LINEWIDTH_PT = 0.8
AXIS_LINEWIDTH_PT = 0.7
GRID_LINEWIDTH_PT = 0.35
TICK_LINEWIDTH_PT = 0.6
TICK_LENGTH_PT = 3.0
FONT_SIZE_PT = 8


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": FONT_SIZE_PT,
            "axes.labelsize": FONT_SIZE_PT,
            "axes.titlesize": FONT_SIZE_PT,
            "legend.fontsize": FONT_SIZE_PT,
            "xtick.labelsize": FONT_SIZE_PT,
            "ytick.labelsize": FONT_SIZE_PT,
            "axes.linewidth": AXIS_LINEWIDTH_PT,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": TICK_LENGTH_PT,
            "ytick.major.size": TICK_LENGTH_PT,
            "xtick.major.width": TICK_LINEWIDTH_PT,
            "ytick.major.width": TICK_LINEWIDTH_PT,
            "xtick.minor.visible": False,
            "ytick.minor.visible": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.format": "pdf",
        }
    )


def apply_axes(ax: plt.Axes) -> None:
    ax.grid(True, which="major", color="#BDBDBD", linestyle="-", linewidth=GRID_LINEWIDTH_PT, alpha=0.32)
    ax.minorticks_off()
    ax.tick_params(which="major", direction="in", length=TICK_LENGTH_PT, width=TICK_LINEWIDTH_PT, top=True, right=True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(AXIS_LINEWIDTH_PT)


def read_rows(path: Path) -> list[dict]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        try:
            out.append(
                {
                    "episode": int(float(row["episode"])),
                    "method": str(row["method"]),
                    "seed": int(float(row.get("seed", 0))),
                    "average_return": float(row["average_return"]),
                    "return_std": float(row.get("return_std", 0.0) or 0.0),
                }
            )
        except Exception:
            continue
    return out


def grouped_curve(rows: list[dict], method: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def plot_average_return(rows: list[dict], output_path: Path) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE)
    values = [row["average_return"] for row in rows]
    if values:
        pad = max(0.05, 0.04 * (max(values) - min(values) + 1.0e-9))
        ax.set_ylim(min(values) - pad, max(values) + pad)
    for method in BASELINE_METHODS:
        vals = [row["average_return"] for row in rows if row["method"] == method]
        if vals:
            ax.axhline(np.mean(vals), color=COLORS[method], linestyle=LINESTYLES[method], linewidth=MAIN_LINEWIDTH_PT)
    for method in CURVE_METHODS:
        episodes, mean, std = grouped_curve(rows, method)
        if episodes.size == 0:
            continue
        ax.plot(episodes, mean, color=COLORS[method], linestyle=LINESTYLES[method], linewidth=MAIN_LINEWIDTH_PT)
        if np.any(std > 0):
            ax.fill_between(episodes, mean - std, mean + std, color=COLORS[method], alpha=0.10, linewidth=0)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Average return")
    apply_axes(ax)
    present = {row["method"] for row in rows}
    handles = [
        Line2D([0], [0], color=COLORS[m], linestyle=LINESTYLES[m], linewidth=MAIN_LINEWIDTH_PT, label=DISPLAY_NAMES[m])
        for m in METHOD_ORDER
        if m in present
    ]
    if handles:
        ax.legend(
            handles=handles,
            loc="lower right",
            ncol=2,
            framealpha=0.92,
            fontsize=FONT_SIZE_PT,
            handlelength=1.35,
            columnspacing=0.60,
            handletextpad=0.35,
            borderpad=0.20,
            labelspacing=0.18,
        )
    fig.subplots_adjust(left=0.22, right=0.985, bottom=0.23, top=0.96)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot control-learning average return comparison.")
    parser.add_argument("--input", type=Path, default=DATA_DIR / "average_return_comparison.csv")
    parser.add_argument("--output", type=Path, default=FIGURES_DIR / "average_return_comparison.pdf")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input)
    if not rows:
        raise ValueError(f"No rows found in {args.input}")
    out = plot_average_return(rows, args.output)
    print(f"Saved figure: {out.resolve()}")


if __name__ == "__main__":
    main()
