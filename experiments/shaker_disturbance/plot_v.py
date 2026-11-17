from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ROLLOUT_DIR = DATA_DIR / "rollouts"
FIGURES_DIR = ROOT / "figures_v"
MPL_CACHE_DIR = ROOT / ".mpl_cache"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR.resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONTROLLERS = ["passive", "adrc", "mpc", "deepc", "ar_ppo"]
DISPLAY_NAMES = {
    "passive": "Passive",
    "adrc": "ADRC",
    "mpc": "MPC",
    "deepc": "DeePC",
    "ar_ppo": "AR-PPO",
}
COLORS = {
    "passive": "#000000",
    "adrc": "#D55E00",
    "mpc": "#009E73",
    "deepc": "#CC79A7",
    "ar_ppo": "#E69F00",
}
LINESTYLES = {
    "passive": "-",
    "adrc": "-.",
    "mpc": ":",
    "deepc": "--",
    "ar_ppo": "-",
}
FONT_SIZE = 8
LINE_WIDTH = 0.55
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
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.5)


def read_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def numeric(rows: list[dict], key: str) -> np.ndarray:
    return np.asarray([float(row.get(key, 0.0) or 0.0) for row in rows], dtype=float)


def load_rollouts() -> dict[str, list[dict[str, np.ndarray]]]:
    out: dict[str, list[dict[str, np.ndarray]]] = {}
    for controller in CONTROLLERS:
        runs = []
        for path in sorted(ROLLOUT_DIR.glob(f"{controller}_repeat_*.csv")):
            rows = read_rows(path)
            if not rows:
                continue
            runs.append(
                {
                    "time": numeric(rows, "time"),
                    "q2": numeric(rows, "x2"),
                    "dq2": numeric(rows, "v2"),
                    "ddq2": numeric(rows, "a2"),
                    "u": numeric(rows, "u_applied"),
                }
            )
        if runs:
            out[controller] = runs
    return out


def mean_std(runs: list[dict[str, np.ndarray]], key: str):
    n = min(len(run[key]) for run in runs)
    stack = np.vstack([run[key][:n] for run in runs])
    mean = np.mean(stack, axis=0)
    std = np.std(stack, axis=0, ddof=1) if len(runs) > 1 else np.zeros_like(mean)
    return runs[0]["time"][:n], mean, std


def plot_band(ax: plt.Axes, t: np.ndarray, mean: np.ndarray, std: np.ndarray, controller: str) -> None:
    ax.plot(t, mean, color=COLORS[controller], linestyle=LINESTYLES[controller], linewidth=LINE_WIDTH, label=DISPLAY_NAMES[controller])
    if np.any(std > 0.0):
        ax.fill_between(t, mean - std, mean + std, color=COLORS[controller], alpha=0.12, linewidth=0.0)


def main() -> Path:
    apply_style()
    rollouts = load_rollouts()
    fig, axes = plt.subplots(2, 2, figsize=(3.5, 2.65), sharex=True)
    panels = [
        (axes[0, 0], "q2", r"$q_2$ [m]"),
        (axes[0, 1], "dq2", r"$\dot{q}_2$ [m/s]"),
        (axes[1, 0], "ddq2", r"$\ddot{q}_2$ [m/s$^2$]"),
        (axes[1, 1], "u", "Action [N]"),
    ]
    for controller in CONTROLLERS:
        if controller not in rollouts:
            continue
        for ax, key, _ in panels:
            t, mean, std = mean_std(rollouts[controller], key)
            plot_band(ax, t, mean, std, controller)
    for ax, _, ylabel in panels:
        ax.set_ylabel(ylabel)
        ax.set_xlim(0.0, 2.0)
        style_axis(ax)
    for ax in axes[1, :]:
        ax.set_xlabel("Time [s]")
    for ax in axes[0, :]:
        ax.tick_params(axis="x", labelbottom=False)
    axes[0, 0].legend(
        loc="lower left",
        ncol=2,
        framealpha=0.92,
        borderpad=0.18,
        labelspacing=0.12,
        handlelength=1.25,
        columnspacing=0.42,
        handletextpad=0.26,
        fontsize=6.5,
    )
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.16, top=0.98, wspace=0.5, hspace=0.10)
    out = FIGURES_DIR / "fixed_trajectory_comparison.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved figure: {out.resolve()}")
    return out


if __name__ == "__main__":
    main()
