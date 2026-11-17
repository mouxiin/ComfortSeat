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
from matplotlib.ticker import MaxNLocator


MODEL_COLORS = {
    "M0": "#0072B2",
    "M1": "#D55E00",
    "M2": "#009E73",
    "M3": "#CC79A7",
    "M4": "#E69F00",
}
MODEL_STYLES = {
    "M0": (0, (4.0, 2.0)),
    "M1": "-.",
    "M2": ":",
    "M3": "-",
    "M4": "-",
}
MEASURED_COLOR = "#000000"
ACTION_COLOR = "#E95E56"
DISTURBANCE_COLOR = "#5B3886"
FONT_SIZE = 8
LINE_WIDTH = 0.75
THIN_LINE_WIDTH = 0.45
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


def style_axis(ax: plt.Axes, grid: bool = True) -> None:
    ax.minorticks_off()
    ax.tick_params(axis="both", direction="in", length=3.0, width=0.6, top=True, right=True)
    if grid:
        ax.grid(True, linewidth=GRID_WIDTH, alpha=GRID_ALPHA, linestyle="--")
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_WIDTH)


def read_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def has_number(row: dict, key: str) -> bool:
    try:
        float(row.get(key, ""))
        return True
    except (TypeError, ValueError):
        return False


def open_loop_metric(rows: list[dict]):
    if rows and all(has_number(row, "test_olr_weighted_nrmse_mean") for row in rows):
        return "test_olr_weighted_nrmse_mean", "test_olr_weighted_nrmse_std", "Open-loop SW-NRMSE"
    if rows and all(has_number(row, "test_olr_nrmse_mean") for row in rows):
        return "test_olr_nrmse_mean", "test_olr_nrmse_std", "Open-loop NRMSE"
    return "test_olr_rmse_mean", "test_olr_rmse_std", "Open-loop RMSE"


def model_order(rows: list[dict]) -> list[dict]:
    order = {f"M{i}": i for i in range(5)}
    return sorted(rows, key=lambda row: order.get(row["model_id"], 99))


def plot_ablation() -> Path:
    rows = model_order(read_rows(DATA_DIR / "ablation_summary.csv"))
    metric, std_metric, ylabel = open_loop_metric(rows)
    labels = [row["model_id"] for row in rows]
    values = np.asarray([float(row[metric]) for row in rows], dtype=float)
    std = np.asarray([float(row.get(std_metric, 0.0) or 0.0) for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    ax.bar(
        labels,
        values,
        yerr=std,
        capsize=2.5,
        color=[MODEL_COLORS.get(label, "#E69F00") for label in labels],
        edgecolor="black",
        linewidth=AXIS_WIDTH,
        error_kw={"elinewidth": LINE_WIDTH, "capthick": 0.6},
    )
    ax.set_xlabel("Ablation model")
    ax.set_ylabel(ylabel)
    style_axis(ax, grid=False)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.16, top=0.98)
    out = FIGURES_DIR / "ablation_bar_olr.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def selected_history_steps(rows: list[dict]) -> int:
    for row in rows:
        if str(row.get("is_selected_history_steps", "")).lower() in {"true", "1"}:
            return int(float(row["history_steps"]))
    for row in rows:
        if str(row.get("is_proposed_history_steps", "")).lower() in {"true", "1"}:
            return int(float(row["history_steps"]))
    metric, _, _ = open_loop_metric(rows)
    return int(float(min(rows, key=lambda row: float(row[metric]))["history_steps"]))


def plot_history() -> Path:
    rows = sorted(read_rows(DATA_DIR / "history_sensitivity.csv"), key=lambda row: int(float(row["history_steps"])))
    metric, std_metric, ylabel = open_loop_metric(rows)
    x = np.asarray([int(float(row["history_steps"])) for row in rows], dtype=int)
    y = np.asarray([float(row[metric]) for row in rows], dtype=float)
    yerr = np.asarray([float(row.get(std_metric, 0.0) or 0.0) for row in rows], dtype=float)
    selected = selected_history_steps(rows)
    selected_y = y[x.tolist().index(selected)] if selected in x else float(np.nanmin(y))

    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        color="#009E73",
        marker="o",
        markersize=3.6,
        linewidth=LINE_WIDTH,
        elinewidth=LINE_WIDTH,
        capsize=2.0,
    )
    ax.axvline(selected, color="0.70", linestyle="--", linewidth=LINE_WIDTH, zorder=0)
    ax.axhline(selected_y, color="0.70", linestyle="--", linewidth=LINE_WIDTH, zorder=0)
    ax.set_xlabel(r"History length $N_h$")
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(0, max(x) + 1, 3))
    ax.set_ylim(0.0, max(1.2, float(np.nanmax(y + yerr)) * 1.12))
    style_axis(ax, grid=False)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.16, top=0.97)
    out = FIGURES_DIR / "history_sensitivity_olr.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def load_error_stats(rows: list[dict], prefix: str):
    return {
        "min": np.asarray([float(row[f"{prefix}_min"]) for row in rows], dtype=float),
        "max": np.asarray([float(row[f"{prefix}_max"]) for row in rows], dtype=float),
        "mid": np.asarray([float(row[f"{prefix}_mid"]) for row in rows], dtype=float),
    }


def plot_error_panel(ax: plt.Axes, time: np.ndarray, stats: dict, ylabel: str, legend: bool) -> None:
    specs = {
        "open": {"label": "OLRE", "color": "#9DB7D5"},
        "one": {"label": "OSPE", "color": "#D9A4B3"},
    }
    handles = []
    labels = []
    for key in ("open", "one"):
        spec = specs[key]
        fill = ax.fill_between(
            time,
            stats[key]["min"],
            stats[key]["max"],
            color=spec["color"],
            alpha=0.34,
            linewidth=0,
            label=spec["label"],
        )
        handles.append(fill)
        labels.append(spec["label"])
    ax.set_ylabel(ylabel)
    style_axis(ax, grid=True)
    if legend:
        ax.legend(
            handles,
            labels,
            loc="upper right",
            ncol=2,
            framealpha=0.92,
            borderpad=0.20,
            labelspacing=0.16,
            handlelength=1.45,
            columnspacing=0.60,
        )


def load_threshold_stats() -> dict | None:
    path = DATA_DIR / "nh07_one_step_residual_threshold_stats.csv"
    if not path.exists():
        return None
    rows = read_rows(path)
    if not rows:
        return None
    return rows[0]


def annotate_error_thresholds(axes: np.ndarray, stats: dict | None) -> None:
    if stats is None:
        return
    x2_value = float(stats["x2_equivalent_abs_threshold_m"])
    v2_value = float(stats["v2_equivalent_abs_threshold_m_per_s"])
    labels = (
        rf"$|e_{{q_2}}|={x2_value:.1e}$ m",
        rf"$|e_{{\dot{{q}}_2}}|={v2_value:.3f}$ m/s",
    )
    for ax, value, text in zip(axes, (x2_value, v2_value), labels):
        ax.axhline(value, color="#C43C39", lw=AXIS_WIDTH, linestyle=(0, (2.0, 1.6)))
        ax.axhline(-value, color="#C43C39", lw=AXIS_WIDTH, linestyle=(0, (2.0, 1.6)))
        ax.text(
            0.03,
            0.06,
            text,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=FONT_SIZE - 1,
            color="#C43C39",
        )


def plot_error_envelope() -> Path:
    rows = read_rows(DATA_DIR / "nh07_open_vs_one_error_envelope.csv")
    time = np.asarray([float(row["time"]) for row in rows], dtype=float)
    mask = time <= 6.0
    stats = {
        "x2": {
            "open": load_error_stats(rows, "x2_open"),
            "one": load_error_stats(rows, "x2_one"),
        },
        "v2": {
            "open": load_error_stats(rows, "v2_open"),
            "one": load_error_stats(rows, "v2_one"),
        },
    }
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 2.55), sharex=True)
    plot_error_panel(
        axes[0],
        time[mask],
        {key: {name: values[mask] for name, values in value.items()} for key, value in stats["x2"].items()},
        r"$q_2$ error [m]",
        True,
    )
    plot_error_panel(
        axes[1],
        time[mask],
        {key: {name: values[mask] for name, values in value.items()} for key, value in stats["v2"].items()},
        r"$\dot{q}_2$ error [m/s]",
        False,
    )
    annotate_error_thresholds(axes, load_threshold_stats())
    axes[1].set_xlabel("Time [s]")
    axes[0].tick_params(axis="x", labelbottom=False)
    for ax in axes:
        ax.set_xlim(0.0, 6.0)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.16, top=0.98, hspace=0.10)
    out = FIGURES_DIR / "nh07_open_vs_one_error_envelope_6s.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def grouped_trajectory_rows(path: Path):
    rows = read_rows(path)
    ids = ["Measured", "M0", "M1", "M2", "M3", "M4"]
    return {model_id: [row for row in rows if row["model_id"] == model_id] for model_id in ids}


def series(rows: list[dict], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def plot_trajectory_247() -> Path:
    csv_path = DATA_DIR / "subfigure_predictions" / "single_plot" / "trajectory_247.csv"
    if not csv_path.exists():
        csv_path = DATA_DIR / "subfigure_predictions" / "test" / "trajectory_247.csv"
    grouped = grouped_trajectory_rows(csv_path)
    measured = grouped["Measured"]
    time = series(measured, "time")
    input_time = time[:-1]
    states = np.column_stack([series(measured, key) for key in ("x1", "v1", "x2", "v2")])
    states[:, 0] -= states[0, 0]
    states[:, 2] -= states[0, 2]
    predictions = {}
    for model_id in ("M0", "M1", "M2", "M3", "M4"):
        rows = grouped[model_id]
        if not rows:
            continue
        pred = np.column_stack([series(rows, key) for key in ("x1", "v1", "x2", "v2")])
        pred[:, 0] -= pred[0, 0]
        pred[:, 2] -= pred[0, 2]
        predictions[model_id] = pred

    fig, axes = plt.subplots(3, 2, figsize=(3.5, 3.25), sharex=True)
    state_panels = [
        (axes[0, 0], r"$q_1$ [m]", 0),
        (axes[0, 1], r"$\dot{q}_1$ [m/s]", 1),
        (axes[1, 0], r"$q_2$ [m]", 2),
        (axes[1, 1], r"$\dot{q}_2$ [m/s]", 3),
    ]
    for idx, (ax, ylabel, state_idx) in enumerate(state_panels):
        ax.plot(time, states[:, state_idx], color=MEASURED_COLOR, linewidth=THIN_LINE_WIDTH, label="True")
        for model_id, pred in predictions.items():
            ax.plot(
                time,
                pred[:, state_idx],
                color=MODEL_COLORS[model_id],
                linestyle=MODEL_STYLES[model_id],
                linewidth=THIN_LINE_WIDTH,
                alpha=0.95,
                label=model_id,
            )
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.55)
        ax.set_ylabel(ylabel)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
        style_axis(ax, grid=True)
        if idx == 0:
            ax.legend(
                loc="lower left",
                ncol=3,
                framealpha=0.92,
                borderpad=0.18,
                labelspacing=0.12,
                handlelength=1.25,
                columnspacing=0.45,
                handletextpad=0.28,
                fontsize=6.5,
            )
            ax.set_ylim(-0.004, 0.003)

    action = series(measured[:-1], "voice_force_n")
    disturbance = series(measured[:-1], "shaker_force_n")
    axes[2, 0].plot(input_time, action, color=ACTION_COLOR, linewidth=LINE_WIDTH)
    axes[2, 0].set_ylabel("Action [N]")
    style_axis(axes[2, 0], grid=True)
    axes[2, 1].plot(input_time, disturbance, color=MEASURED_COLOR, linewidth=LINE_WIDTH)
    axes[2, 1].set_ylabel("Shaker-side disturbance [N]")
    style_axis(axes[2, 1], grid=True)
    for ax in axes[:2, :].reshape(-1):
        ax.tick_params(axis="x", labelbottom=False)
    for ax in axes[2, :]:
        ax.set_xlabel("Time [s]")
    for ax in axes.reshape(-1):
        ax.set_xlim(0.0, min(2.0, float(time[-1])))
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.12, top=0.98, wspace=0.5, hspace=0.1)
    out = FIGURES_DIR / "trajectory_247.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    apply_style()
    saved = [plot_ablation(), plot_history(), plot_error_envelope(), plot_trajectory_247()]
    for path in saved:
        print(f"Saved figure: {path.resolve()}")


if __name__ == "__main__":
    main()
