"""Standalone hardware comparison plotter.

Reads only CSV files from this experiment directory and writes PDF figures.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ROLLOUT_DIR = DATA_DIR / "rollouts"
FIGURE_DIR = ROOT / "figures"
MPL_CACHE_DIR = ROOT / ".mpl_cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR.resolve()))

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

JOURNAL_SHOWCASE_FIGSIZE = (6.50, 3.00)
JOURNAL_FONT_SIZE = 8
JOURNAL_LINE_WIDTH = 0.55
JOURNAL_AXIS_WIDTH = 0.7
JOURNAL_GRID_WIDTH = 0.35
JOURNAL_GRID_ALPHA = 0.18
JOURNAL_TICK_LENGTH = 3.0
JOURNAL_TICK_WIDTH = 0.6
SHOWCASE_ADJUST = {
    "left": 0.090,
    "right": 0.965,
    "bottom": 0.145,
    "top": 0.970,
    "wspace": 0.40,
    "hspace": 0.0,
}


def read_csv_rows(path: Path) -> list[dict]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": JOURNAL_FONT_SIZE,
            "axes.labelsize": JOURNAL_FONT_SIZE,
            "axes.titlesize": JOURNAL_FONT_SIZE,
            "xtick.labelsize": JOURNAL_FONT_SIZE,
            "ytick.labelsize": JOURNAL_FONT_SIZE,
            "legend.fontsize": JOURNAL_FONT_SIZE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.linewidth": JOURNAL_AXIS_WIDTH,
            "lines.linewidth": JOURNAL_LINE_WIDTH,
            "lines.solid_capstyle": "round",
            "lines.dash_capstyle": "round",
            "grid.linewidth": JOURNAL_GRID_WIDTH,
            "grid.alpha": JOURNAL_GRID_ALPHA,
            "grid.linestyle": "--",
        }
    )


def style_axes(ax, show_grid=True) -> None:
    ax.minorticks_off()
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        length=JOURNAL_TICK_LENGTH,
        width=JOURNAL_TICK_WIDTH,
        labelsize=JOURNAL_FONT_SIZE,
        top=True,
        right=True,
    )
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    for spine in ax.spines.values():
        spine.set_linewidth(JOURNAL_AXIS_WIDTH)
    if show_grid:
        ax.grid(True, linewidth=JOURNAL_GRID_WIDTH, alpha=JOURNAL_GRID_ALPHA, linestyle="--")


def numeric_column(rows: list[dict], key: str) -> np.ndarray:
    return np.asarray([float(r.get(key, 0.0) or 0.0) for r in rows], dtype=np.float64)


def load_rollouts() -> dict[str, list[dict[str, np.ndarray]]]:
    out: dict[str, list[dict[str, np.ndarray]]] = {}
    for controller in CONTROLLERS:
        runs = []
        for path in sorted(ROLLOUT_DIR.glob(f"{controller}_repeat_*.csv")):
            rows = read_csv_rows(path)
            if not rows:
                continue
            runs.append(
                {
                    "time": numeric_column(rows, "time"),
                    "x2": numeric_column(rows, "x2"),
                    "v2": numeric_column(rows, "v2"),
                    "a2": numeric_column(rows, "a2"),
                    "u": numeric_column(rows, "u_applied"),
                    "disturbance": numeric_column(rows, "shaker_force_n"),
                    "compute": numeric_column(rows, "computation_time_ms"),
                }
            )
        if runs:
            out[controller] = runs
    if not out:
        raise FileNotFoundError(f"No rollout CSV files found in {ROLLOUT_DIR.resolve()}")
    return out


def mean_std(runs: list[dict[str, np.ndarray]], key: str) -> tuple[np.ndarray, np.ndarray]:
    length = min(len(run[key]) for run in runs)
    stack = np.vstack([run[key][:length] for run in runs])
    mean = np.mean(stack, axis=0)
    std = np.std(stack, axis=0, ddof=1) if len(runs) > 1 else np.zeros_like(mean)
    return mean, std


def plot_with_band(ax, t, mean, std, controller, label=None):
    color = COLORS[controller]
    ax.plot(
        t,
        mean,
        LINESTYLES[controller],
        color=color,
        lw=JOURNAL_LINE_WIDTH,
        label=label or DISPLAY_NAMES[controller],
    )
    if np.any(std > 0.0):
        ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.12, linewidth=0.0)


def plot_fixed_trajectory(rollouts: dict[str, list[dict[str, np.ndarray]]], time_window_s: float) -> None:
    setup_style()
    fig, grid_axes = plt.subplots(
        2,
        3,
        figsize=JOURNAL_SHOWCASE_FIGSIZE,
        sharex="col",
        gridspec_kw={
            "wspace": SHOWCASE_ADJUST["wspace"],
            "hspace": SHOWCASE_ADJUST["hspace"],
        },
    )
    axes = [grid_axes[0, 0], grid_axes[0, 1], grid_axes[0, 2], grid_axes[1, 0], grid_axes[1, 1]]
    legend_ax = grid_axes[1, 2]
    legend_ax.axis("off")

    first_controller = next(iter(rollouts))
    for controller in CONTROLLERS:
        if controller not in rollouts:
            continue
        runs = rollouts[controller]
        t = mean_std(runs, "time")[0]
        for idx, key in enumerate(["x2", "v2", "a2", "u"]):
            mean, std = mean_std(runs, key)
            plot_with_band(axes[idx], t[: len(mean)], mean, std, controller)

    t_dist = mean_std(rollouts[first_controller], "time")[0]
    d_mean, d_std = mean_std(rollouts[first_controller], "disturbance")
    axes[4].plot(t_dist[: len(d_mean)], d_mean, "-", color="#5B3886", lw=JOURNAL_LINE_WIDTH, label="Disturbance")
    if np.any(d_std > 0.0):
        axes[4].fill_between(t_dist[: len(d_mean)], d_mean - d_std, d_mean + d_std, color="#5B3886", alpha=0.12, linewidth=0.0)

    axes[0].set_ylabel(r"$x_2$ [m]")
    axes[1].set_ylabel(r"$v_2$ [m/s]")
    axes[2].set_ylabel(r"$a_2$ [m/s$^2$]")
    axes[3].set_ylabel("Action [N]")
    axes[4].set_ylabel("Disturbance [N]")
    axes[3].set_xlabel("Time [s]")
    axes[4].set_xlabel("Time [s]")
    for ax in grid_axes[0, :].reshape(-1):
        ax.tick_params(axis="x", labelbottom=False)
    for ax in axes:
        if time_window_s > 0.0:
            ax.set_xlim(0.0, float(time_window_s))
        style_axes(ax, show_grid=True)
        ax.axhline(y=0, color="black", linestyle="--", linewidth=JOURNAL_LINE_WIDTH)
    handles, labels = axes[0].get_legend_handles_labels()
    disturbance_handles, disturbance_labels = axes[4].get_legend_handles_labels()
    handles += disturbance_handles
    labels += disturbance_labels
    legend_ax.legend(
        handles,
        labels,
        ncol=2,
        framealpha=0.92,
        loc="center",
        fontsize=JOURNAL_FONT_SIZE,
        handlelength=1.6,
        columnspacing=0.7,
        handletextpad=0.45,
    )
    fig.subplots_adjust(**SHOWCASE_ADJUST)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / "fixed_trajectory_comparison.pdf")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot hardware controller comparison PDFs from rollout CSV files.")
    parser.add_argument("--time-window-s", type=float, default=2.0, help="Use 0 or negative to show the full trajectory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rollouts = load_rollouts()
    plot_fixed_trajectory(rollouts, args.time_window_s)
    print(f"Saved figure: {FIGURE_DIR / 'fixed_trajectory_comparison.pdf'}")


if __name__ == "__main__":
    main()
