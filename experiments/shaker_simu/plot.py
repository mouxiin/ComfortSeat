from __future__ import annotations

import csv
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".mpl_cache"))

import matplotlib.pyplot as plt
import numpy as np


DATA_DIR = SCRIPT_DIR / "data"
ROLLOUT_DIR = DATA_DIR / "rollouts"
FIGURES_DIR = SCRIPT_DIR / "figures"

CONTROLLER_ORDER = ["passive", "adrc", "mpc", "deepc", "ar_ppo"]
CONTROLLER_LABELS = {
    "passive": "Passive",
    "adrc": "ADRC",
    "mpc": "MPC",
    "deepc": "DeePC",
    "ar_ppo": "AR-PPO",
}
COLORS = {
    "passive": "#555555",
    "adrc": "#1f77b4",
    "mpc": "#2ca02c",
    "deepc": "#9467bd",
    "ar_ppo": "#d62728",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.5,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def values(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def ordered_comparison_rows() -> list[dict[str, str]]:
    rows = read_rows(DATA_DIR / "simulation_vs_hardware.csv")
    by_controller = {row["controller"]: row for row in rows}
    return [by_controller[name] for name in CONTROLLER_ORDER if name in by_controller]


def plot_rms_comparison() -> Path:
    rows = ordered_comparison_rows()
    labels = [CONTROLLER_LABELS[row["controller"]] for row in rows]
    x = np.arange(len(rows), dtype=float)
    metrics = [
        ("x2_rms", "x2 RMS (m)"),
        ("v2_rms", "v2 RMS (m/s)"),
        ("a2_rms", "a2 RMS (m/s$^2$)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.2), constrained_layout=True)
    width = 0.34
    for ax, (metric, ylabel) in zip(axes, metrics):
        sim = values(rows, f"{metric}_sim")
        hardware_mean = values(rows, f"{metric}_hardware_mean")
        hardware_std = values(rows, f"{metric}_hardware_std")
        ax.bar(
            x - width / 2,
            hardware_mean,
            width,
            yerr=hardware_std,
            capsize=2.5,
            color="#b9b9b9",
            edgecolor="#333333",
            linewidth=0.45,
            label="Hardware mean +/- std",
        )
        sim_colors = [COLORS[row["controller"]] for row in rows]
        ax.bar(
            x + width / 2,
            sim,
            width,
            color=sim_colors,
            edgecolor="#333333",
            linewidth=0.45,
            label="DDM simulation",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=28, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", color="#dddddd", linewidth=0.6, alpha=0.85)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, loc="upper left")
    path = FIGURES_DIR / "shaker_simu_vs_hardware_rms.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def load_rollout(controller: str) -> list[dict[str, str]]:
    return read_rows(ROLLOUT_DIR / f"{controller}.csv")


def plot_time_response() -> Path:
    rollouts = {name: load_rollout(name) for name in CONTROLLER_ORDER if (ROLLOUT_DIR / f"{name}.csv").exists()}
    if not rollouts:
        raise FileNotFoundError(f"No rollout CSV files found in {ROLLOUT_DIR}")

    fig, axes = plt.subplots(5, 1, figsize=(8.0, 7.2), sharex=True, constrained_layout=True)
    signal_defs = [
        ("x2", "x2 (m)"),
        ("v2", "v2 (m/s)"),
        ("a2", "a2 (m/s$^2$)"),
        ("u_applied", "u (N)"),
    ]

    for controller, rows in rollouts.items():
        t = values(rows, "time")
        label = CONTROLLER_LABELS[controller]
        color = COLORS[controller]
        for ax, (key, ylabel) in zip(axes[:4], signal_defs):
            ax.plot(t, values(rows, key), color=color, linewidth=1.1, label=label)
            ax.set_ylabel(ylabel)

    first_rows = next(iter(rollouts.values()))
    t = values(first_rows, "time")
    axes[4].plot(t, values(first_rows, "shaker_force_n"), color="#222222", linewidth=1.1)
    axes[4].set_ylabel("d (N)")
    axes[4].set_xlabel("Time (s)")

    for ax in axes:
        ax.grid(True, color="#dddddd", linewidth=0.6, alpha=0.85)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.38))

    path = FIGURES_DIR / "shaker_simu_time_response.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths = [plot_rms_comparison(), plot_time_response()]
    for path in paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
