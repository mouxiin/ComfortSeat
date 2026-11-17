from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ROLLOUT_DIR = SCRIPT_DIR / "data" / "rollouts"
FIGURES_DIR = SCRIPT_DIR / "figures_v"
MPL_CACHE_DIR = SCRIPT_DIR / ".mpl_cache"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR.resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CASE_ID = "fm20_d30"
TIME_WINDOW = (1.5, 4.5)
METHOD_ORDER = ["passive", "adrc", "mpc", "deepc", "ar_ppo", "dgs_ar_ppo"]
DISPLAY_NAMES = {
    "passive": "Passive",
    "adrc": "ADRC",
    "mpc": "MPC",
    "deepc": "DeePC",
    "ar_ppo": "AR-PPO",
    "dgs_ar_ppo": "DGS-AR-PPO",
}
COLORS = {
    "passive": "#000000",
    "adrc": "#D55E00",
    "mpc": "#009E73",
    "deepc": "#CC79A7",
    "ar_ppo": "#E69F00",
    "dgs_ar_ppo": "#0072B2",
    "unknown": "#C81E1E",
}
LINESTYLES = {
    "passive": "-",
    "adrc": "-.",
    "mpc": ":",
    "deepc": "--",
    "ar_ppo": "-",
    "dgs_ar_ppo": "-",
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


def numeric(rows: list[dict], key: str, default: float = 0.0) -> np.ndarray:
    values = []
    for row in rows:
        try:
            values.append(float(row.get(key, default)))
        except (TypeError, ValueError):
            values.append(default)
    return np.asarray(values, dtype=float)


def load_rollouts(case_id: str) -> dict[str, list[dict[str, np.ndarray]]]:
    case_dir = ROLLOUT_DIR / case_id
    out: dict[str, list[dict[str, np.ndarray]]] = {}
    for controller in METHOD_ORDER:
        runs = []
        for path in sorted(case_dir.glob(f"{controller}_repeat_*.csv")):
            rows = read_rows(path)
            if not rows:
                continue
            runs.append(
                {
                    "time": numeric(rows, "time"),
                    "q2": numeric(rows, "x2"),
                    "dq2": numeric(rows, "v2"),
                    "ddq2": numeric(rows, "a2"),
                    "u": numeric(rows, "u_cmd"),
                    "unknown": numeric(rows, "F_ud_true"),
                }
            )
        if runs:
            out[controller] = runs
    if not out:
        raise FileNotFoundError(f"No rollout CSV files found in {case_dir.resolve()}")
    return out


def aligned_stack(runs: list[dict[str, np.ndarray]], key: str):
    n = min(len(run[key]) for run in runs)
    return runs[0]["time"][:n], np.vstack([run[key][:n] for run in runs])


def mean_std(runs: list[dict[str, np.ndarray]], key: str):
    t, stack = aligned_stack(runs, key)
    mean = np.nanmean(stack, axis=0)
    std = np.nanstd(stack, axis=0, ddof=1) if len(runs) > 1 else np.zeros_like(mean)
    return t, mean, std


def plot_band(ax: plt.Axes, t: np.ndarray, mean: np.ndarray, std: np.ndarray, controller: str) -> None:
    ax.plot(t, mean, color=COLORS[controller], linestyle=LINESTYLES[controller], linewidth=LINE_WIDTH, label=DISPLAY_NAMES[controller])
    if np.any(np.isfinite(std) & (std > 0.0)):
        ax.fill_between(t, mean - std, mean + std, color=COLORS[controller], alpha=0.12, linewidth=0.0)


def main() -> Path:
    apply_style()
    rollouts = load_rollouts(CASE_ID)
    fig, axes = plt.subplots(2, 2, figsize=(3.5, 2.65), sharex=True)
    panels = [
        (axes[0, 0], "q2", r"$q_2$ [m]"),
        (axes[0, 1], "dq2", r"$\dot{q}_2$ [m/s]"),
        (axes[1, 0], "ddq2", r"$\ddot{q}_2$ [m/s$^2$]"),
        (axes[1, 1], "u", "Action [N]"),
    ]
    for controller in METHOD_ORDER:
        if controller not in rollouts:
            continue
        for ax, key, _ in panels:
            t, mean, std = mean_std(rollouts[controller], key)
            plot_band(ax, t, mean, std, controller)

    ref_runs = rollouts.get("dgs_ar_ppo") or rollouts.get("ar_ppo") or next(iter(rollouts.values()))
    t_unknown, unknown_stack = aligned_stack(ref_runs, "unknown")
    unknown_mean = np.nanmean(unknown_stack, axis=0)
    unknown_std = np.nanstd(unknown_stack, axis=0, ddof=1) if len(ref_runs) > 1 else np.zeros_like(unknown_mean)
    injected_line = axes[1, 1].plot(
        t_unknown,
        unknown_mean,
        color=COLORS["unknown"],
        linestyle=(0, (2.0, 1.4)),
        linewidth=0.75,
        label=r"Injected $w_k^u$",
    )[0]
    if np.any(np.isfinite(unknown_std) & (unknown_std > 0.0)):
        axes[1, 1].fill_between(t_unknown, unknown_mean - unknown_std, unknown_mean + unknown_std, color=COLORS["unknown"], alpha=0.10, linewidth=0.0)

    for ax, _, ylabel in panels:
        ax.set_ylabel(ylabel)
        ax.set_xlim(*TIME_WINDOW)
        style_axis(ax)
    for ax in axes[1, :]:
        ax.set_xlabel("Time [s]")
    for ax in axes[0, :]:
        ax.tick_params(axis="x", labelbottom=False)
    axes[0, 0].set_ylim(-0.019, 0.01)
    axes[1, 1].set_ylim(-2.9, 2.1)
    axes[0, 0].legend(
        loc="lower right",
        ncol=2,
        framealpha=0.92,
        borderpad=0.18,
        labelspacing=0.12,
        handlelength=1.20,
        columnspacing=0.34,
        handletextpad=0.24,
        fontsize=6.0,
    )
    axes[1, 1].legend(
        handles=[injected_line],
        labels=[r"Injected $w^u$"],
        loc="lower right",
        ncol=1,
        framealpha=0.92,
        borderpad=0.18,
        labelspacing=0.12,
        handlelength=1.20,
        columnspacing=0.34,
        handletextpad=0.24,
        fontsize=6.5,
    )
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.12, top=0.98, wspace=0.5, hspace=0.10)
    out = FIGURES_DIR / "actuator_disturbance_pulse_sweep_fm20_d30_time_response_comparison.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved figure: {out.resolve()}")
    return out


if __name__ == "__main__":
    main()
