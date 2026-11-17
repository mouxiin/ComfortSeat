from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
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
OKABE_ITO_ORANGE = "#E69F00"
LIGHT_MARKER_GRAY = "#B8B8B8"
DETECTOR_X2_SCALE = 0.02
DETECTOR_V2_SCALE = 0.6
DETECTOR_THRESHOLD_QUANTILE = 0.95
DETECTOR_CALIBRATION_SPLITS = ("train", "val")

JOURNAL_SINGLE_FIGSIZE = (3.00, 2.00)
JOURNAL_DOUBLE_FIGSIZE = (5.50, 2.00)
JOURNAL_SHOWCASE_FIGSIZE = (6.50, 3.00)
JOURNAL_FONT_SIZE = 8
JOURNAL_LINE_WIDTH = 0.8
SHOWCASE_LINE_WIDTH = 0.4
JOURNAL_AXIS_WIDTH = 0.7
JOURNAL_GRID_WIDTH = 0.35
JOURNAL_GRID_ALPHA = 0.18
JOURNAL_TICK_LENGTH = 3.0
JOURNAL_TICK_WIDTH = 0.6
SINGLE_ADJUST = {"left": 0.24, "right": 0.96, "bottom": 0.24, "top": 0.91}
DOUBLE_ADJUST = {"left": 0.13, "right": 0.985, "bottom": 0.26, "top": 0.94, "wspace": 0.24}
SHOWCASE_ADJUST = {"left": 0.090, "right": 0.965, "bottom": 0.145, "top": 0.970, "wspace": 0.40, "hspace": 0.0}


def main():
    parser = argparse.ArgumentParser(description="DDM ablation plotting")
    parser.add_argument(
        "--trajectory",
        "-t",
        default=None,
        help="Plot a single trajectory showcase (e.g. 247, trajectory_247).",
    )
    parser.add_argument(
        "--skip-prediction",
        action="store_true",
        help="Reuse existing prediction CSV (only with --trajectory).",
    )
    args = parser.parse_args()

    apply_journal_style()

    if args.trajectory:
        plot_single_trajectory(
            args.trajectory, skip_prediction=args.skip_prediction
        )
        return

    summary = read_csv_dicts(DATA_DIR / "ablation_summary.csv")
    history = read_csv_dicts(DATA_DIR / "history_sensitivity.csv")
    plot_history_sensitivity(history, FIGURES_DIR / "history_sensitivity_olr.pdf")
    plot_ablation_olr(summary, FIGURES_DIR / "ablation_bar_olr.pdf")
    selected_nh = selected_history_steps(history)
    envelope_csv = DATA_DIR / f"nh{selected_nh:02d}_open_vs_one_error_envelope.csv"
    if not envelope_csv.exists():
        envelope_csv = DATA_DIR / "open_vs_one_error_envelope.csv"
    threshold_stats = calibrate_one_step_residual_threshold(selected_nh)
    plot_open_vs_one(
        envelope_csv,
        FIGURES_DIR / f"nh{selected_nh:02d}_open_vs_one_error_envelope.pdf",
        threshold_stats=threshold_stats,
    )
    plot_open_vs_one(
        envelope_csv,
        FIGURES_DIR / f"nh{selected_nh:02d}_open_vs_one_error_envelope_6s.pdf",
        t_max=6.0,
        threshold_stats=threshold_stats,
    )
    refresh_stale_subfigure_predictions()
    plot_all_subfigures()
    print(f"Saved PDF figures to {FIGURES_DIR.resolve()}")


def read_csv_dicts(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV data file: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def style_axes(ax, show_grid=False):
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
    ax.tick_params(axis="both", which="minor", bottom=False, top=False, left=False, right=False)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    for spine in ax.spines.values():
        spine.set_linewidth(JOURNAL_AXIS_WIDTH)
    if show_grid:
        ax.grid(True, linewidth=JOURNAL_GRID_WIDTH, alpha=JOURNAL_GRID_ALPHA, linestyle="--")


def apply_journal_style():
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.size": JOURNAL_FONT_SIZE,
            "axes.labelsize": JOURNAL_FONT_SIZE,
            "axes.titlesize": JOURNAL_FONT_SIZE,
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


def model_order(rows):
    order = {f"M{i}": i for i in range(5)}
    return sorted(rows, key=lambda row: order.get(row["model_id"], 99))


def has_number(row, key):
    try:
        float(row.get(key, ""))
        return True
    except (TypeError, ValueError):
        return False


def open_loop_plot_metric(rows):
    if rows and all(has_number(row, "test_olr_weighted_nrmse_mean") for row in rows):
        return "test_olr_weighted_nrmse_mean", "test_olr_weighted_nrmse_std", "Open-loop SW-NRMSE"
    if rows and all(has_number(row, "test_olr_nrmse_mean") for row in rows):
        return "test_olr_nrmse_mean", "test_olr_nrmse_std", "Test open-loop rollout NRMSE"
    return "test_olr_rmse_mean", "test_olr_rmse_std", "Test open-loop rollout RMSE"


def plot_ablation_olr(rows, out_path):
    rows = model_order(rows)
    labels = [row["model_id"] for row in rows]
    metric, std_metric, ylabel = open_loop_plot_metric(rows)
    vals = np.asarray([float(row[metric]) for row in rows], dtype=float)
    stds = np.asarray([float(row.get(std_metric, 0.0)) for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=JOURNAL_SINGLE_FIGSIZE)
    ax.bar(
        labels,
        vals,
        yerr=stds,
        capsize=2.5,
        color=[MODEL_COLORS.get(label, OKABE_ITO_ORANGE) for label in labels],
        edgecolor="black",
        linewidth=JOURNAL_AXIS_WIDTH,
        error_kw={"elinewidth": JOURNAL_LINE_WIDTH, "capthick": JOURNAL_TICK_WIDTH},
    )
    ax.set_xlabel("Ablation model")
    ax.set_ylabel(ylabel)
    style_axes(ax, show_grid=False)
    fig.subplots_adjust(**SINGLE_ADJUST)
    fig.savefig(out_path)
    plt.close(fig)


def infer_proposed_history_steps(rows):
    for row in rows:
        if str(row.get("is_selected_history_steps", "")).lower() in {"true", "1"}:
            return int(float(row["history_steps"]))
    for row in rows:
        if str(row.get("is_proposed_history_steps", "")).lower() in {"true", "1"}:
            return int(float(row["history_steps"]))
    return 8


def selected_history_steps(rows):
    for row in rows:
        if str(row.get("is_selected_history_steps", "")).lower() in {"true", "1"}:
            return int(float(row["history_steps"]))
    for row in rows:
        if str(row.get("is_proposed_history_steps", "")).lower() in {"true", "1"}:
            return int(float(row["history_steps"]))
    if rows:
        metric, _, _ = open_loop_plot_metric(rows)
        best = min(rows, key=lambda row: float(row[metric]))
        return int(float(best["history_steps"]))
    return 8


def plot_history_sensitivity(rows, out_path):
    rows = sorted(rows, key=lambda row: int(float(row["history_steps"])))
    x = np.asarray([int(float(row["history_steps"])) for row in rows], dtype=int)
    metric, std_metric, ylabel = open_loop_plot_metric(rows)
    y = np.asarray([float(row[metric]) for row in rows], dtype=float)
    yerr = np.asarray([float(row.get(std_metric, 0.0)) for row in rows], dtype=float)
    proposed = infer_proposed_history_steps(rows)

    fig, ax = plt.subplots(figsize=JOURNAL_SINGLE_FIGSIZE)
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        color="#009E73",
        marker="o",
        markersize=4,
        linewidth=JOURNAL_LINE_WIDTH,
        elinewidth=JOURNAL_LINE_WIDTH,
        capsize=2.0,
    )
    y_at_proposed = y[x.tolist().index(proposed)] if proposed in x else np.nanmin(y)
    ax.axvline(proposed, color=LIGHT_MARKER_GRAY, linestyle="--", linewidth=JOURNAL_LINE_WIDTH, zorder=0)
    ax.axhline(y_at_proposed, color=LIGHT_MARKER_GRAY, linestyle="--", linewidth=JOURNAL_LINE_WIDTH, zorder=0)
    ax.set_xlabel(r"History length $N_h$")
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(0, max(x) + 1, 3))
    upper = max(1.2, float(np.nanmax(y + yerr)) * 1.12)
    ax.set_ylim(0, upper)
    style_axes(ax, show_grid=False)
    fig.subplots_adjust(**SINGLE_ADJUST)
    fig.savefig(out_path)
    plt.close(fig)


def parse_trajectory_name(value: str) -> str:
    """Normalize trajectory id input.

    ``"247"``, ``"trajectory_247"``, and ``"trajectory_247.csv"`` all return
    ``"trajectory_247"``.
    """
    text = str(value).strip()
    if text.startswith("trajectory_"):
        return text[:-4] if text.endswith(".csv") else text
    return f"trajectory_{int(text):03d}"


def grouped_showcase_rows(path: Path):
    rows = read_csv_dicts(path)
    model_ids = ["Measured", "M0", "M1", "M2", "M3", "M4"]
    grouped = {model_id: [row for row in rows if row["model_id"] == model_id] for model_id in model_ids}
    return grouped


def series(rows, key):
    return np.asarray([float(row[key]) for row in rows], dtype=np.float32)


def plot_showcase(csv_path, out_path):
    grouped = grouped_showcase_rows(csv_path)
    measured = grouped["Measured"]
    if not measured:
        raise ValueError(f"{csv_path} does not contain Measured rows.")
    out_name = Path(out_path).name
    is_special = out_name.endswith("_best.pdf") or out_name.endswith("_median.pdf") or out_name.endswith("_worst.pdf")
    time_axis = series(measured, "time")
    input_time = time_axis[:-1]
    states = np.column_stack([series(measured, key) for key in ("x1", "v1", "x2", "v2")])
    states_plot = states.copy()
    states_plot[:, 0] -= states_plot[0, 0]
    states_plot[:, 2] -= states_plot[0, 2]

    pred_plots = {}
    for model_id in ("M0", "M1", "M2", "M3", "M4"):
        rows = grouped[model_id]
        if not rows:
            continue
        pred = np.column_stack([series(rows, key) for key in ("x1", "v1", "x2", "v2")])
        pred[:, 0] -= pred[0, 0]
        pred[:, 2] -= pred[0, 2]
        pred_plots[model_id] = pred

    fig, axes = plt.subplots(
        2,
        3,
        figsize=JOURNAL_SHOWCASE_FIGSIZE,
        sharex="col",
        gridspec_kw={"wspace": SHOWCASE_ADJUST["wspace"], "hspace": SHOWCASE_ADJUST["hspace"]},
    )
    state_panels = [
        (axes[0, 0], "$x_1$ [m]", 0),
        (axes[0, 1], "$v_1$ [m/s]", 1),
        (axes[0, 2], "$x_2$ [m]", 2),
        (axes[1, 0], "$v_2$ [m/s]", 3),
    ]
    for panel_idx, (ax, ylabel, state_idx) in enumerate(state_panels):
        ax.plot(time_axis, states_plot[:, state_idx], color=MEASURED_COLOR, lw=SHOWCASE_LINE_WIDTH, label="True")
        for model_id, pred in pred_plots.items():
            ax.plot(
                time_axis,
                pred[:, state_idx],
                color=MODEL_COLORS.get(model_id, "0.25"),
                linestyle=MODEL_STYLES.get(model_id, "-"),
                lw=SHOWCASE_LINE_WIDTH,
                label=model_id,
                alpha=0.95,
            )
        ax.set_ylabel(ylabel)
        style_axes(ax, show_grid=True)
        if panel_idx == 0:
            ax.legend(
                loc="lower right" if is_special else "upper right",
                ncol=3,
                framealpha=0.92,
                fontsize=JOURNAL_FONT_SIZE,
                handlelength=1.6,
                columnspacing=0.7,
                handletextpad=0.45,
            )

    action = series(measured[:-1], "voice_force_n")
    disturbance = series(measured[:-1], "shaker_force_n")
    axes[1, 1].plot(input_time, action, color=ACTION_COLOR, lw=SHOWCASE_LINE_WIDTH, label="Action")
    axes[1, 1].set_ylabel("Action [N]")
    axes[1, 1].legend(loc="upper left", framealpha=0.92, fontsize=JOURNAL_FONT_SIZE)
    style_axes(axes[1, 1], show_grid=True)

    axes[1, 2].plot(input_time, disturbance, color=DISTURBANCE_COLOR, lw=SHOWCASE_LINE_WIDTH, label="Disturbance")
    axes[1, 2].set_ylabel("Disturbance [N]")
    axes[1, 2].set_ylim(-9.0, 12.0)
    axes[1, 2].legend(loc="upper right", framealpha=0.92, fontsize=JOURNAL_FONT_SIZE)
    style_axes(axes[1, 2], show_grid=True)

    for ax in axes[0, :].reshape(-1):
        ax.tick_params(axis="x", labelbottom=False)
    for ax in axes[1, :].reshape(-1):
        ax.set_xlabel("Time [s]")
    for ax in axes.reshape(-1):
        ax.set_xlim(0.0, min(2.0, float(time_axis[-1])))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))

    fig.subplots_adjust(**SHOWCASE_ADJUST)
    fig.savefig(out_path)
    plt.close(fig)


def plot_all_subfigures():
    pred_root = DATA_DIR / "subfigure_predictions"
    if not pred_root.exists():
        raise FileNotFoundError(
            f"Missing {pred_root}. Run ddm_ablation/main.py once to generate all subfigure prediction CSV files."
        )
    for split_name in ("train", "eval", "test"):
        in_dir = pred_root / split_name
        if not in_dir.exists():
            continue
        out_dir = FIGURES_DIR / "subfigures" / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for csv_path in sorted(in_dir.glob("trajectory_*.csv")):
            plot_showcase(csv_path, out_dir / f"{csv_path.stem}.pdf")


def plot_single_trajectory(
    trajectory_id: str,
    split_name: str | None = None,
    skip_prediction: bool = False,
) -> None:
    """Plot a single trajectory showcase figure.

    Parameters
    ----------
    trajectory_id:
        Trajectory identifier — ``"247"``, ``"trajectory_247"``, or
        ``"trajectory_247.csv"`` are all accepted.
    split_name:
        Optional split label written into the prediction CSV.
        Defaults to the split recorded in *dataset_split.csv*.
    skip_prediction:
        If True, reuse an existing prediction CSV instead of recomputing
        from trained models.
    """
    trajectory_name = parse_trajectory_name(trajectory_id)
    split_rows = read_csv_dicts(DATA_DIR / "dataset_split.csv")
    row = None
    for r in split_rows:
        if r.get("trajectory") == trajectory_name:
            row = r
            break
    if row is None:
        raise FileNotFoundError(
            f"Cannot find {trajectory_name} in {DATA_DIR / 'dataset_split.csv'}. "
            "Run main.py once, or check the trajectory id."
        )

    source_path = Path(row["path"])
    if not source_path.exists():
        raise FileNotFoundError(f"Missing source trajectory CSV: {source_path}")

    out_split = split_name or ("eval" if row["split"] == "val" else row["split"])

    pred_dir = DATA_DIR / "subfigure_predictions" / "single_plot"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / f"{trajectory_name}.csv"

    if not skip_prediction or not pred_path.exists():
        ddm = load_training_module()
        torch = import_torch()
        device = torch.device("cpu")
        results = load_ablation_models(ddm, torch, device)
        traj = ddm.load_trajectory(source_path)
        rows = build_prediction_rows(ddm, torch, device, results, traj, out_split)
        write_csv(pred_path, rows)

    out_dir = FIGURES_DIR / "subfigures" / "single_plot"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{trajectory_name}.pdf"
    plot_showcase(pred_path, out_path)
    print(f"Saved single trajectory PDF: {out_path.resolve()}")


def refresh_stale_subfigure_predictions():
    split_path = DATA_DIR / "dataset_split.csv"
    pred_root = DATA_DIR / "subfigure_predictions"
    if not split_path.exists() or not pred_root.exists():
        return

    split_rows = read_csv_dicts(split_path)
    stale_rows = []
    newest_checkpoint_mtime = newest_ablation_checkpoint_mtime()
    for row in split_rows:
        split_name = "eval" if row["split"] == "val" else row["split"]
        source_path = Path(row["path"])
        pred_path = pred_root / split_name / f"{row['trajectory']}.csv"
        if not source_path.exists():
            continue
        pred_mtime = pred_path.stat().st_mtime if pred_path.exists() else -1.0
        if (not pred_path.exists()) or max(source_path.stat().st_mtime, newest_checkpoint_mtime) > pred_mtime:
            stale_rows.append((row, split_name, source_path, pred_path))

    if not stale_rows:
        return

    ddm = load_training_module()
    torch = import_torch()
    device = torch.device("cpu")
    results = load_ablation_models(ddm, torch, device)
    for row, split_name, source_path, pred_path in stale_rows:
        traj = ddm.load_trajectory(source_path)
        rows = build_prediction_rows(ddm, torch, device, results, traj, split_name)
        write_csv(pred_path, rows)
        print(f"Refreshed prediction CSV: {pred_path}")


def load_training_module():
    module_path = SCRIPT_DIR / "main.py"
    spec = importlib.util.spec_from_file_location("ddm_ablation_main_for_plot", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def import_torch():
    import torch

    return torch


def newest_ablation_checkpoint_mtime():
    mtimes = []
    for model_id in ("M0", "M1", "M2", "M3", "M4"):
        path = SCRIPT_DIR / "models" / f"ablation_{model_id}.pt"
        if path.exists():
            mtimes.append(path.stat().st_mtime)
    return max(mtimes) if mtimes else -1.0


def load_ablation_models(ddm, torch, device):
    results = {}
    for model_id in ("M0", "M1", "M2", "M3", "M4"):
        ckpt_path = SCRIPT_DIR / "models" / f"ablation_{model_id}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing trained model checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        spec = ddm.ModelSpec(**ckpt["spec"])
        input_dim = int(ckpt["summary"]["input_dim"])
        hidden_dim = int(ckpt.get("hidden_dim", ckpt["summary"]["hidden_dim"]))
        model = ddm.make_model(input_dim, hidden_dim, spec.residual_blocks).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        results[model_id] = {
            "model": model,
            "stats": ckpt["stats"],
            "spec": spec,
            "history_steps": int(ckpt["history_steps"]),
        }
    return results


def load_history_model(ddm, torch, device, history_steps):
    ckpt_path = SCRIPT_DIR / "models" / f"history_sensitivity_Nh{int(history_steps):02d}.pt"
    if not ckpt_path.exists():
        ckpt_path = SCRIPT_DIR / "models" / "history_sensitivity_selected.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing selected history checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    spec = ddm.ModelSpec(**ckpt["spec"])
    input_dim = int(ckpt["summary"]["input_dim"])
    hidden_dim = int(ckpt.get("hidden_dim", ckpt["summary"]["hidden_dim"]))
    model = ddm.make_model(input_dim, hidden_dim, spec.residual_blocks).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return {
        "model": model,
        "stats": ckpt["stats"],
        "spec": spec,
        "history_steps": int(ckpt["history_steps"]),
        "checkpoint": ckpt_path.name,
    }


def calibrate_one_step_residual_threshold(history_steps):
    ddm = load_training_module()
    torch = import_torch()
    device = torch.device("cpu")
    result = load_history_model(ddm, torch, device, history_steps)
    split = ddm.split_trajectories(ddm.load_trajectories(DATA_DIR), seed=666)
    rows = []
    residuals_by_split = {}
    scores_by_split = {}
    for split_name, trajectories in split.items():
        arrays = ddm.build_arrays(
            trajectories,
            result["history_steps"],
            result["spec"].history_input,
            result["spec"].residual_target,
        )
        pred_target = ddm.predict_targets(result["model"], arrays, result["stats"], device)
        pred_next = arrays["current_states"] + pred_target if result["spec"].residual_target else pred_target
        err = pred_next - arrays["next_states"]
        score = np.sqrt((err[:, 2] / DETECTOR_X2_SCALE) ** 2 + (err[:, 3] / DETECTOR_V2_SCALE) ** 2)
        residuals_by_split[split_name] = err
        scores_by_split[split_name] = score

    calibration_err = np.concatenate([residuals_by_split[name] for name in DETECTOR_CALIBRATION_SPLITS], axis=0)
    calibration_score = np.concatenate([scores_by_split[name] for name in DETECTOR_CALIBRATION_SPLITS], axis=0)
    threshold = float(np.quantile(calibration_score, DETECTOR_THRESHOLD_QUANTILE))

    def append_summary(name, err, score):
        x2 = err[:, 2]
        v2 = err[:, 3]
        ax2 = np.abs(x2)
        av2 = np.abs(v2)
        rows.append(
            {
                "split": name,
                "history_steps": result["history_steps"],
                "checkpoint": result["checkpoint"],
                "samples": len(score),
                "score_definition": "sqrt((e_x2/0.02)^2+(e_v2/0.6)^2)",
                "threshold_quantile": DETECTOR_THRESHOLD_QUANTILE,
                "selected_score_threshold": threshold,
                "x2_equivalent_abs_threshold_m": threshold * DETECTOR_X2_SCALE,
                "v2_equivalent_abs_threshold_m_per_s": threshold * DETECTOR_V2_SCALE,
                "score_mean": float(np.mean(score)),
                "score_std": float(np.std(score, ddof=1)),
                "score_q99": float(np.quantile(score, 0.99)),
                "score_q995": float(np.quantile(score, 0.995)),
                "score_max": float(np.max(score)),
                "x2_error_mean_m": float(np.mean(x2)),
                "x2_error_std_m": float(np.std(x2, ddof=1)),
                "x2_abs_error_mean_m": float(np.mean(ax2)),
                "x2_abs_error_std_m": float(np.std(ax2, ddof=1)),
                "x2_abs_error_q995_m": float(np.quantile(ax2, 0.995)),
                "x2_abs_error_max_m": float(np.max(ax2)),
                "v2_error_mean_m_per_s": float(np.mean(v2)),
                "v2_error_std_m_per_s": float(np.std(v2, ddof=1)),
                "v2_abs_error_mean_m_per_s": float(np.mean(av2)),
                "v2_abs_error_std_m_per_s": float(np.std(av2, ddof=1)),
                "v2_abs_error_q995_m_per_s": float(np.quantile(av2, 0.995)),
                "v2_abs_error_max_m_per_s": float(np.max(av2)),
                "exceedance_rate_at_selected_threshold": float(np.mean(score > threshold)),
            }
        )

    append_summary("calibration_train_val", calibration_err, calibration_score)
    for split_name in ("train", "val", "test"):
        append_summary(split_name, residuals_by_split[split_name], scores_by_split[split_name])
    out_csv = DATA_DIR / f"nh{int(result['history_steps']):02d}_one_step_residual_threshold_stats.csv"
    write_csv(out_csv, rows)
    return rows[0]


def build_prediction_rows(ddm, torch, device, results, traj, split_name):
    rows = []

    def append_state(model_id, sample, state):
        rows.append(
            {
                "split": split_name,
                "trajectory": traj.name,
                "model_id": model_id,
                "sample": sample,
                "time": float(traj.time[min(sample, len(traj.time) - 1)]),
                "x1": float(state[0]),
                "v1": float(state[1]),
                "x2": float(state[2]),
                "v2": float(state[3]),
                "voice_force_n": float(traj.actions[min(sample, traj.num_steps - 1), 0]),
                "shaker_force_n": float(traj.disturbances[min(sample, traj.num_steps - 1), 0]),
            }
        )

    for sample, state in enumerate(traj.states):
        append_state("Measured", sample, state)
    for model_id, result in results.items():
        pred = ddm.open_loop_rollout(
            result["model"],
            traj,
            result["stats"],
            result["history_steps"],
            result["spec"].history_input,
            result["spec"].residual_target,
            device,
        )
        for sample, state in enumerate(pred):
            append_state(model_id, sample, state)
    return rows


def plot_open_vs_one(csv_path, out_path, t_max=2.0, threshold_stats=None):
    rows = read_csv_dicts(csv_path)
    time_axis = np.asarray([float(row["time"]) for row in rows], dtype=np.float32)
    x2 = {
        "open": load_error_stats(rows, "x2_open"),
        "one": load_error_stats(rows, "x2_one"),
    }
    v2 = {
        "open": load_error_stats(rows, "v2_open"),
        "one": load_error_stats(rows, "v2_one"),
    }
    fig, axes = plt.subplots(1, 2, figsize=JOURNAL_DOUBLE_FIGSIZE, sharex=True)
    plot_error_panel(axes[0], time_axis, x2, "$x_2$ error [m]", True)
    plot_error_panel(axes[1], time_axis, v2, "$v_2$ error [m/s]", False)
    if threshold_stats is not None:
        annotate_detector_threshold(axes, threshold_stats)
    axes[0].set_xlabel("Time [s]")
    axes[1].set_xlabel("Time [s]")
    for ax in axes:
        ax.set_xlim(0.0, min(t_max, float(time_axis[-1])))
    fig.subplots_adjust(**DOUBLE_ADJUST)
    fig.savefig(out_path)
    plt.close(fig)


def annotate_detector_threshold(axes, stats):
    score_threshold = float(stats["selected_score_threshold"])
    x2_equiv = float(stats["x2_equivalent_abs_threshold_m"])
    v2_equiv = float(stats["v2_equivalent_abs_threshold_m_per_s"])
    line_color = "#C43C39"
    for ax, value in ((axes[0], x2_equiv), (axes[1], v2_equiv)):
        ax.axhline(value, color=line_color, lw=JOURNAL_AXIS_WIDTH, linestyle=(0, (2.0, 1.6)))
        ax.axhline(-value, color=line_color, lw=JOURNAL_AXIS_WIDTH, linestyle=(0, (2.0, 1.6)))
    axes[0].text(
        0.03,
        0.05,
        rf"$\tau_{{95}}={score_threshold:.3f}$" + "\n" + rf"$|e_{{x_2}}|={x2_equiv:.1e}$ m",
        transform=axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=JOURNAL_FONT_SIZE - 1,
        color=line_color,
    )
    axes[1].text(
        0.03,
        0.05,
        rf"$|e_{{v_2}}|={v2_equiv:.3f}$ m/s",
        transform=axes[1].transAxes,
        ha="left",
        va="bottom",
        fontsize=JOURNAL_FONT_SIZE - 1,
        color=line_color,
    )


def load_error_stats(rows, prefix):
    return {
        "min": np.asarray([float(row[f"{prefix}_min"]) for row in rows], dtype=np.float32),
        "max": np.asarray([float(row[f"{prefix}_max"]) for row in rows], dtype=np.float32),
        "mid": np.asarray([float(row[f"{prefix}_mid"]) for row in rows], dtype=np.float32),
    }


def plot_error_panel(ax, time_axis, data, ylabel, show_legend):
    specs = {
        "open": {"label": "OLRE", "fill": "#9DB7D5"},
        "one": {"label": "OSPE", "fill": "#D9A4B3"},
    }
    handles, labels = [], []
    for key in ("open", "one"):
        spec = specs[key]
        fill = ax.fill_between(
            time_axis,
            data[key]["min"],
            data[key]["max"],
            color=spec["fill"],
            alpha=0.35,
            linewidth=0,
            label=spec["label"],
        )
        handles.append(fill)
        labels.append(spec["label"])
    ax.axhline(0.0, color="0.45", lw=JOURNAL_AXIS_WIDTH, linestyle="--")
    ax.set_ylabel(ylabel)
    style_axes(ax, show_grid=True)
    if show_legend:
        ax.legend(handles, labels, loc="upper right", ncol=2, framealpha=0.92, fontsize=JOURNAL_FONT_SIZE, columnspacing=1.0, handlelength=1.8)


if __name__ == "__main__":
    main()
