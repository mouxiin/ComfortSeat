"""Standalone plotter for structured actuator-side pulse disturbance sweep."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
ROLLOUT_DIR = DATA_DIR / "rollouts"
FIGURE_DIR = SCRIPT_DIR / "figures"
MPL_CACHE_DIR = SCRIPT_DIR / ".mpl_cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR.resolve()))

import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


METHOD_ORDER = ["passive", "adrc", "mpc", "deepc", "ar_ppo", "ar_ppo_est", "dgs_ar_ppo"]
DISPLAY_NAMES = {
    "passive": "Passive",
    "adrc": "ADRC",
    "mpc": "MPC",
    "deepc": "DeePC",
    "ar_ppo": "AR-PPO",
    "ar_ppo_est": "AR-PPO-Est",
    "dgs_ar_ppo": "DGS-AR-PPO",
}
COLORS = {
    "passive": "#000000",
    "adrc": "#D55E00",
    "mpc": "#009E73",
    "deepc": "#CC79A7",
    "ar_ppo": "#E69F00",
    "ar_ppo_est": "#E95E56",
    "dgs_ar_ppo": "#0072B2",
    "disturbance": "#5B3886",
    "unknown": "#C81E1E",
    "estimate": "#D97706",
}
LINESTYLES = {
    "passive": "-",
    "adrc": "-.",
    "mpc": ":",
    "deepc": "--",
    "ar_ppo": "-",
    "ar_ppo_est": "-",
    "dgs_ar_ppo": "-",
}
ALL_CONTOUR_METRICS = (
    ("x2_event_rms", "$x_2$ event RMS [m]"),
    ("v2_event_rms", "$v_2$ event RMS [m/s]"),
    ("a2_event_rms", "$a_2$ event RMS [m/s$^2$]"),
)
RMS_BAR_COLORS = (
    (0 / 255, 114 / 255, 178 / 255),
    (213 / 255, 94 / 255, 0 / 255),
    (0 / 255, 158 / 255, 115 / 255),
    (204 / 255, 121 / 255, 167 / 255),
    (230 / 255, 159 / 255, 0 / 255),
    (233 / 255, 94 / 255, 86 / 255),
)
JOURNAL_FONT_SIZE = 8
CONTOUR_LABEL_FONT_SIZE = 6
JOURNAL_SHOWCASE_FIGSIZE = (6.50, 3.00)
JOURNAL_LINE_WIDTH = 0.55
JOURNAL_AXIS_WIDTH = 0.7
JOURNAL_GRID_WIDTH = 0.35
JOURNAL_GRID_ALPHA = 0.18
JOURNAL_TICK_WIDTH = 0.6
JOURNAL_TICK_LENGTH = 3.0
SHOWCASE_ADJUST = {
    "left": 0.090,
    "right": 0.965,
    "bottom": 0.145,
    "top": 0.970,
    "wspace": 0.40,
    "hspace": 0.0,
}


def read_rows(path: Path) -> list[dict]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def numeric(rows: list[dict], key: str, default=np.nan) -> np.ndarray:
    vals = []
    for row in rows:
        try:
            vals.append(float(row.get(key, default)))
        except Exception:
            vals.append(default)
    return np.asarray(vals, dtype=np.float64)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "font.size": JOURNAL_FONT_SIZE,
            "axes.labelsize": JOURNAL_FONT_SIZE,
            "axes.titlesize": JOURNAL_FONT_SIZE,
            "xtick.labelsize": JOURNAL_FONT_SIZE,
            "ytick.labelsize": JOURNAL_FONT_SIZE,
            "legend.fontsize": JOURNAL_FONT_SIZE,
            "axes.linewidth": JOURNAL_AXIS_WIDTH,
            "lines.linewidth": JOURNAL_LINE_WIDTH,
            "lines.solid_capstyle": "round",
            "lines.dash_capstyle": "round",
            "grid.linewidth": JOURNAL_GRID_WIDTH,
            "grid.alpha": JOURNAL_GRID_ALPHA,
            "grid.linestyle": "--",
        }
    )


def apply_axis_style(ax, show_grid=False) -> None:
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        length=JOURNAL_TICK_LENGTH,
        width=JOURNAL_TICK_WIDTH,
        top=True,
        right=True,
    )
    ax.minorticks_off()
    for spine in ax.spines.values():
        spine.set_linewidth(JOURNAL_AXIS_WIDTH)
    if show_grid:
        ax.grid(True, linewidth=JOURNAL_GRID_WIDTH, alpha=JOURNAL_GRID_ALPHA, linestyle="--")


def discover_case_dirs(cases: list[str] | None = None) -> list[Path]:
    if not ROLLOUT_DIR.exists():
        return []
    wanted = {str(case) for case in cases} if cases else None
    case_dirs = []
    for path in sorted(ROLLOUT_DIR.iterdir()):
        if not path.is_dir() or not any(path.glob("*_repeat_*.csv")):
            continue
        if wanted is not None and path.name not in wanted:
            continue
        case_dirs.append(path)
    return case_dirs


def load_case_rollouts(case_dir: Path) -> dict[str, list[dict[str, np.ndarray]]]:
    out: dict[str, list[dict[str, np.ndarray]]] = {}
    for controller in METHOD_ORDER:
        runs = []
        for path in sorted(case_dir.glob(f"{controller}_repeat_*.csv")):
            rows = read_rows(path)
            if not rows:
                continue
            runs.append(
                {
                    "time": numeric(rows, "time", 0.0),
                    "x2": numeric(rows, "x2", 0.0),
                    "v2": numeric(rows, "v2", 0.0),
                    "a2": numeric(rows, "a2", 0.0),
                    "u_cmd": numeric(rows, "u_cmd", 0.0),
                    "d_t": numeric(rows, "d_t", 0.0),
                    "F_ud_true": numeric(rows, "F_ud_true", 0.0),
                    "F_ud_est": numeric(rows, "F_ud_est", np.nan),
                    "detection_flag": numeric(rows, "detection_flag", 0.0),
                }
            )
        if runs:
            out[controller] = runs
    if not out:
        raise FileNotFoundError(f"No rollout CSV files found in {case_dir.resolve()}")
    return out


def aligned_stack(runs: list[dict[str, np.ndarray]], key: str) -> tuple[np.ndarray, np.ndarray]:
    length = min(len(run[key]) for run in runs)
    return runs[0]["time"][:length], np.vstack([run[key][:length] for run in runs])


def mean_std(runs: list[dict[str, np.ndarray]], key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t, stack = aligned_stack(runs, key)
    mean = np.nanmean(stack, axis=0)
    std = np.nanstd(stack, axis=0, ddof=1) if len(runs) > 1 else np.zeros_like(mean)
    return t, mean, std


def plot_with_band(ax, t, mean, std, controller, label=None) -> None:
    color = COLORS[controller]
    ax.plot(
        t,
        mean,
        LINESTYLES[controller],
        color=color,
        lw=JOURNAL_LINE_WIDTH,
        label=label or DISPLAY_NAMES[controller],
    )
    if np.any(np.isfinite(std) & (std > 0.0)):
        ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.12, linewidth=0.0)


def plot_case_time_response(
    case_id: str,
    rollouts: dict[str, list[dict[str, np.ndarray]]],
    time_window: tuple[float, float],
    prefix: str,
    output_dir: Path,
) -> Path:
    fig, grid_axes = plt.subplots(
        2,
        3,
        figsize=JOURNAL_SHOWCASE_FIGSIZE,
        sharex="col",
        gridspec_kw={"wspace": SHOWCASE_ADJUST["wspace"], "hspace": SHOWCASE_ADJUST["hspace"]},
    )
    axes = [grid_axes[0, 0], grid_axes[0, 1], grid_axes[0, 2], grid_axes[1, 0], grid_axes[1, 1]]
    legend_ax = grid_axes[1, 2]
    legend_ax.axis("off")

    ref_runs = rollouts.get("dgs_ar_ppo") or rollouts.get("ar_ppo_est") or next(iter(rollouts.values()))
    t_ref, fud_stack = aligned_stack(ref_runs, "F_ud_true")
    fud_mean = np.nanmean(fud_stack, axis=0)
    active = np.abs(fud_mean) > 1.0e-6
    if np.any(active):
        active_idx = np.where(active)[0]
        start_t = t_ref[active_idx[0]]
        end_t = t_ref[min(active_idx[-1] + 1, len(t_ref) - 1)]
        for ax in axes:
            ax.axvspan(start_t, end_t, color="#E0E0E0", alpha=0.55, zorder=0, linewidth=0)

    for controller in METHOD_ORDER:
        if controller not in rollouts:
            continue
        runs = rollouts[controller]
        for ax, key in zip(axes[:4], ["x2", "v2", "a2", "u_cmd"]):
            t, mean, std = mean_std(runs, key)
            plot_with_band(ax, t, mean, std, controller)

    t_d, d_stack = aligned_stack(ref_runs, "d_t")
    d_mean = np.nanmean(d_stack, axis=0)
    d_std = np.nanstd(d_stack, axis=0, ddof=1) if len(ref_runs) > 1 else np.zeros_like(d_mean)
    axes[4].plot(t_d, d_mean, color=COLORS["disturbance"], lw=JOURNAL_LINE_WIDTH, label="Shaker")
    if np.any(d_std > 0.0):
        axes[4].fill_between(t_d, d_mean - d_std, d_mean + d_std, color=COLORS["disturbance"], alpha=0.12, linewidth=0.0)
    fud_std = np.nanstd(fud_stack, axis=0, ddof=1) if len(ref_runs) > 1 else np.zeros_like(fud_mean)
    axes[4].plot(t_ref, fud_mean, color=COLORS["unknown"], lw=JOURNAL_LINE_WIDTH, label=r"True $F_{ud}$")
    if np.any(fud_std > 0.0):
        axes[4].fill_between(t_ref, fud_mean - fud_std, fud_mean + fud_std, color=COLORS["unknown"], alpha=0.12, linewidth=0.0)
    if "ar_ppo_est" in rollouts:
        t_e, est_stack = aligned_stack(rollouts["ar_ppo_est"], "F_ud_est")
        finite_est = np.isfinite(est_stack)
        if np.any(finite_est):
            count = np.sum(finite_est, axis=0)
            est_mean = np.full(est_stack.shape[1], np.nan, dtype=np.float64)
            valid_mean = count > 0
            est_mean[valid_mean] = np.nansum(est_stack[:, valid_mean], axis=0) / count[valid_mean]
            est_std = np.zeros_like(est_mean)
            valid_std = count > 1
            if np.any(valid_std):
                centered = np.where(finite_est[:, valid_std], est_stack[:, valid_std] - est_mean[valid_std], 0.0)
                est_std[valid_std] = np.sqrt(np.sum(centered**2, axis=0) / (count[valid_std] - 1))
            axes[4].plot(t_e, est_mean, color=COLORS["estimate"], linestyle="--", lw=JOURNAL_LINE_WIDTH, label=r"Estimated $F_{ud}$")
            finite_band = np.isfinite(est_std) & (est_std > 0.0)
            if np.any(finite_band):
                axes[4].fill_between(t_e, est_mean - est_std, est_mean + est_std, color=COLORS["estimate"], alpha=0.12, linewidth=0.0)

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
        ax.set_xlim(float(time_window[0]), float(time_window[1]))
        apply_axis_style(ax, show_grid=True)
        ax.axhline(y=0, color="black", linestyle="--", lw=JOURNAL_LINE_WIDTH)

    handles, labels = axes[0].get_legend_handles_labels()
    force_handles, force_labels = axes[4].get_legend_handles_labels()
    handles += force_handles
    labels += force_labels
    legend_ax.legend(
        handles,
        labels,
        ncol=2,
        loc="center",
        framealpha=0.92,
        fontsize=JOURNAL_FONT_SIZE,
        handlelength=1.6,
        handletextpad=0.45,
        columnspacing=0.7,
    )

    fig.subplots_adjust(**SHOWCASE_ADJUST)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{prefix}_{case_id}_time_response_comparison.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def default_time_window(rollouts: dict[str, list[dict[str, np.ndarray]]], t_start: float, t_end: float) -> tuple[float, float]:
    if float(t_end) > float(t_start):
        return float(t_start), float(t_end)
    first_runs = next(iter(rollouts.values()))
    t0 = min(float(run["time"][0]) for run in first_runs)
    t1 = max(float(run["time"][-1]) for run in first_runs)
    return t0, t1


def plot_time_responses_all(prefix: str, output_dir: Path, cases: list[str] | None, t_start: float, t_end: float) -> list[Path]:
    saved = []
    for case_dir in discover_case_dirs(cases):
        rollouts = load_case_rollouts(case_dir)
        time_window = default_time_window(rollouts, t_start, t_end)
        saved.append(plot_case_time_response(case_dir.name, rollouts, time_window, prefix, output_dir))
    return saved


def load_metric_grids(path: Path):
    rows = read_rows(path)
    if not rows:
        raise FileNotFoundError(f"No aggregate rows found in {path.resolve()}")
    forces = np.asarray(sorted({float(r["unknown_force_n"]) for r in rows}), dtype=np.float64)
    durations = np.asarray(sorted({float(r["unknown_duration_steps"]) for r in rows}), dtype=np.float64)
    methods = [m for m in METHOD_ORDER if any(r["controller"] == m for r in rows)]
    metrics = {}
    for method in methods:
        metrics[method] = {
            "x2_rms": np.full((len(forces), len(durations)), np.nan, dtype=np.float64),
            "v2_rms": np.full((len(forces), len(durations)), np.nan, dtype=np.float64),
            "a2_rms": np.full((len(forces), len(durations)), np.nan, dtype=np.float64),
            "x2_event_rms": np.full((len(forces), len(durations)), np.nan, dtype=np.float64),
            "v2_event_rms": np.full((len(forces), len(durations)), np.nan, dtype=np.float64),
            "a2_event_rms": np.full((len(forces), len(durations)), np.nan, dtype=np.float64),
        }
    force_index = {float(v): i for i, v in enumerate(forces)}
    duration_index = {float(v): i for i, v in enumerate(durations)}
    for row in rows:
        method = row["controller"]
        if method not in metrics:
            continue
        fi = force_index[float(row["unknown_force_n"])]
        di = duration_index[float(row["unknown_duration_steps"])]
        for metric in ("x2_rms", "v2_rms", "a2_rms", "x2_event_rms", "v2_event_rms", "a2_event_rms"):
            metrics[method][metric][fi, di] = float(row[f"{metric}_mean"])
    return forces, durations, metrics


def common_limits(metrics, methods, metric_name):
    values = np.stack([metrics[method][metric_name] for method in methods], axis=0)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if np.isclose(vmin, vmax):
        delta = max(abs(vmin) * 1.0e-6, 1.0e-6)
        vmin -= delta
        vmax += delta
    return vmin, vmax


def draw_contour_panel(ax, forces, durations, values, vmin, vmax):
    forces = np.asarray(forces, dtype=float)
    durations = np.asarray(durations, dtype=float)
    values = np.asarray(values, dtype=float)
    image = ax.imshow(
        np.clip(values, vmin, vmax),
        origin="lower",
        extent=[float(np.min(durations)), float(np.max(durations)), float(np.min(forces)), float(np.max(forces))],
        cmap="jet",
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        interpolation="nearest",
    )
    if len(forces) >= 2 and len(durations) >= 2 and np.isfinite(values).any():
        duration_grid, force_grid = np.meshgrid(durations, forces)
        local_min = float(np.nanmin(values))
        local_max = float(np.nanmax(values))
        if not np.isclose(local_min, local_max):
            levels = np.linspace(local_min, local_max, 7)
            lines = ax.contour(duration_grid, force_grid, values, levels=levels, colors="black", linewidths=0.35, alpha=0.75)
            ax.clabel(lines, inline=True, fontsize=CONTOUR_LABEL_FONT_SIZE, fmt="%.2g")
    ax.set_xlim(float(np.min(durations)), float(np.max(durations)))
    ax.set_ylim(float(np.min(forces)), float(np.max(forces)))
    ax.set_box_aspect(1)
    ax.set_xlabel("Duration [steps]")
    apply_axis_style(ax, show_grid=False)
    return image


def plot_contours_all(prefix: str, output_dir: Path, forces, durations, metrics):
    methods = [m for m in METHOD_ORDER if m in metrics]
    fig, axes = plt.subplots(len(ALL_CONTOUR_METRICS), len(methods), figsize=(7.0, 3.5), squeeze=False)
    for col, method in enumerate(methods):
        axes[0, col].set_title(DISPLAY_NAMES.get(method, method))
    for row, (metric_name, label) in enumerate(ALL_CONTOUR_METRICS):
        vmin, vmax = common_limits(metrics, methods, metric_name)
        for col, method in enumerate(methods):
            ax = axes[row, col]
            draw_contour_panel(ax, forces, durations, metrics[method][metric_name], vmin, vmax)
            if row != len(ALL_CONTOUR_METRICS) - 1:
                ax.set_xlabel("")
                ax.tick_params(axis="x", labelbottom=False)
            if col == 0:
                ax.set_ylabel("Force [N]")
            else:
                ax.tick_params(axis="y", labelleft=False)
        cax = axes[row, -1].inset_axes([1.08, 0.0, 0.07, 1.0])
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        mappable = cm.ScalarMappable(norm=norm, cmap="jet")
        mappable.set_array([])
        colorbar = fig.colorbar(mappable, cax=cax)
        colorbar.set_label(label, labelpad=5)
        colorbar.locator = MaxNLocator(nbins=5)
        colorbar.formatter = FormatStrFormatter("%.3g")
        colorbar.update_ticks()
        colorbar.ax.tick_params(direction="in", length=JOURNAL_TICK_LENGTH, width=JOURNAL_TICK_WIDTH, labelsize=JOURNAL_FONT_SIZE)
        colorbar.outline.set_linewidth(JOURNAL_AXIS_WIDTH)
    fig.subplots_adjust(left=0.07, right=0.89, top=0.93, bottom=0.10, wspace=0.12, hspace=0.06)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{prefix}_a2_v2_contours_all.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_rms_all_bar(prefix: str, output_dir: Path, metrics):
    metric_list = (
        ("x2_event_rms", "$x_2$ event RMS [m]"),
        ("v2_event_rms", "$v_2$ event RMS [m/s]"),
        ("a2_event_rms", "$a_2$ event RMS [m/s$^2$]"),
    )
    methods = [m for m in METHOD_ORDER if m in metrics]
    x = np.arange(len(methods), dtype=float)
    fig, axes = plt.subplots(1, len(metric_list), figsize=(6.5, 2.0), sharey=False)
    bar_colors = [RMS_BAR_COLORS[idx % len(RMS_BAR_COLORS)] for idx in range(len(methods))]
    for ax, (metric_name, label) in zip(axes, metric_list):
        means = []
        stds = []
        for method in methods:
            values = np.asarray(metrics[method][metric_name], dtype=float).reshape(-1)
            values = values[np.isfinite(values)]
            means.append(float(np.mean(values)) if values.size else 0.0)
            stds.append(float(np.std(values, ddof=1)) if values.size > 1 else 0.0)
        ax.bar(
            x,
            means,
            0.58,
            color=bar_colors,
            edgecolor="black",
            linewidth=JOURNAL_AXIS_WIDTH,
            yerr=stds,
            error_kw={"elinewidth": JOURNAL_AXIS_WIDTH, "ecolor": "black", "capsize": 2},
        )
        ax.set_xticks(x)
        ax.set_xticklabels([DISPLAY_NAMES.get(method, method) for method in methods], rotation=25, ha="right", fontstyle="italic", fontsize=6)
        ax.set_ylabel(label)
        upper = max(mean + std for mean, std in zip(means, stds)) if means else 1.0
        ax.set_ylim(0.0, max(upper * 1.12, 1.0e-12))
        apply_axis_style(ax, show_grid=True)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.27, top=0.94, wspace=0.35)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{prefix}_rms_all_bar.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot structured actuator-side pulse sweep PDFs from CSV summary data.")
    parser.add_argument("--summary", type=str, default=str(DATA_DIR / "aggregate_statistics.csv"))
    parser.add_argument("--prefix", type=str, default="actuator_disturbance_pulse_sweep")
    parser.add_argument("--cases", nargs="+", default=None, help="Optional case ids such as fm10_d10 fm20_d30.")
    parser.add_argument("--t-start", type=float, default=0.0)
    parser.add_argument("--t-end", type=float, default=0.0, help="Use 0 or negative to show the full trajectory.")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--skip-time-response", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_style()
    saved = []
    summary_path = Path(args.summary)
    if not args.skip_summary:
        if summary_path.exists():
            forces, durations, metrics = load_metric_grids(summary_path)
            saved.extend(
                [
                    plot_contours_all(args.prefix, FIGURE_DIR, forces, durations, metrics),
                    plot_rms_all_bar(args.prefix, FIGURE_DIR, metrics),
                ]
            )
        else:
            print(f"Skipping summary figures: {summary_path.resolve()} not found")
    if not args.skip_time_response:
        saved.extend(
            plot_time_responses_all(
                args.prefix,
                FIGURE_DIR / "time_response_cases",
                args.cases,
                args.t_start,
                args.t_end,
            )
        )
    if not saved:
        raise FileNotFoundError(f"No figures were generated. Check {DATA_DIR.resolve()} for summary or rollout CSV files.")
    for path in saved:
        print(f"Saved figure: {path.resolve()}")


if __name__ == "__main__":
    main()
