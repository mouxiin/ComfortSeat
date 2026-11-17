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


METHOD_ORDER = [
    "passive",
    "adrc",
    "mpc",
    "deepc",
    "ar_ppo",
    "dgs_ar_ppo",
]
LABELS = {
    "passive": "Passive",
    "adrc": "ADRC",
    "mpc": "MPC",
    "deepc": "DeePC",
    "ar_ppo": "AR-PPO",
    "dgs_ar_ppo": "DGS-AR-PPO",
}
COLORS = {
    "passive": "#8A8A8A",
    "adrc": "#D55E00",
    "mpc": "#009E73",
    "deepc": "#CC79A7",
    "ar_ppo": "#E69F00",
    "dgs_ar_ppo": "#0072B2",
}


def read_csv_dicts(path: Path) -> list[dict]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
        }
    )


def sorted_rows(rows: list[dict]) -> list[dict]:
    order = {name: i for i, name in enumerate(METHOD_ORDER)}
    return sorted(rows, key=lambda row: order.get(row["controller"], 99))


def plot_event_metric_bars(rows: list[dict], out_path: Path, title: str = "") -> None:
    rows = sorted_rows(rows)
    metrics = [
        ("x2_event_rms_mean", "$x_2$ event RMS [m]"),
        ("v2_event_rms_mean", "$v_2$ event RMS [m/s]"),
        ("a2_event_rms_mean", "$a_2$ event RMS [m/s$^2$]"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.7))
    labels = [LABELS.get(row["controller"], row["controller"]) for row in rows]
    x = np.arange(len(rows))
    for ax, (metric, ylabel) in zip(axes, metrics):
        values = np.asarray([float(row.get(metric, "nan")) for row in rows], dtype=float)
        std_values = np.asarray([float(row.get(metric.replace("_mean", "_std"), 0.0)) for row in rows], dtype=float)
        ax.bar(
            x,
            values,
            yerr=std_values,
            capsize=2.0,
            color=[COLORS.get(row["controller"], "#8A8A8A") for row in rows],
            edgecolor="black",
            linewidth=0.6,
        )
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.25)
    if title:
        fig.suptitle(title)
        fig.subplots_adjust(top=0.86)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_time_response(case_dir: Path, out_path: Path, duration_steps: int = 10) -> None:
    rollout_root = DATA_DIR / "rollouts"
    available = []
    for controller in METHOD_ORDER:
        path = case_dir / f"{controller}_repeat_01.csv"
        if path.exists():
            available.append((controller, read_csv_dicts(path)))
    if not available:
        return
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 5.2), sharex=True)
    for controller, rows in available:
        time_s = np.asarray([float(row["time"]) for row in rows], dtype=float)
        x2 = np.asarray([float(row["x2"]) for row in rows], dtype=float)
        v2 = np.asarray([float(row["v2"]) for row in rows], dtype=float)
        u_app = np.asarray([float(row["u_app"]) for row in rows], dtype=float)
        label = LABELS.get(controller, controller).replace("\n", " ")
        color = COLORS.get(controller, None)
        axes[0].plot(time_s, x2, label=label, color=color, linewidth=0.9)
        axes[1].plot(time_s, v2, color=color, linewidth=0.9)
        axes[2].plot(time_s, u_app, color=color, linewidth=0.9)
    axes[0].set_ylabel("$x_2$ [m]")
    axes[1].set_ylabel("$v_2$ [m/s]")
    axes[2].set_ylabel("$u_{app}$ [N]")
    axes[2].set_xlabel("Time [s]")
    for ax in axes:
        ax.axvspan(2.0, 2.0 + int(duration_steps) * 0.03, color="#D55E00", alpha=0.10, linewidth=0)
        ax.grid(True, linestyle="--", alpha=0.22)
        ax.set_xlim(1.6, 3.2)
    axes[0].legend(loc="upper right", ncol=2, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def case_duration(case_id: str) -> int:
    try:
        return int(case_id.rsplit("_d", 1)[1])
    except Exception:
        return 10


def main() -> None:
    apply_style()
    aggregate_path = DATA_DIR / "aggregate_statistics.csv"
    if aggregate_path.exists():
        rows = read_csv_dicts(aggregate_path)
        if rows and "case_id" in rows[0]:
            for case in sorted({row["case_id"] for row in rows}):
                case_rows = [row for row in rows if row["case_id"] == case]
                plot_event_metric_bars(
                    case_rows,
                    FIGURES_DIR / f"{case}_event_metric_comparison.pdf",
                    title=case,
                )
                plot_time_response(
                    DATA_DIR / "rollouts" / case,
                    FIGURES_DIR / f"{case}_time_response_repeat_01.pdf",
                    duration_steps=case_duration(case),
                )
        else:
            plot_event_metric_bars(rows, FIGURES_DIR / "event_metric_comparison.pdf")
            plot_time_response(DATA_DIR / "rollouts", FIGURES_DIR / "time_response_repeat_01.pdf")
    print(f"Saved figures to {FIGURES_DIR.resolve()}")


if __name__ == "__main__":
    main()
