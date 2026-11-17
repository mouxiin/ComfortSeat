from __future__ import annotations

import os
import csv
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
MPL_CACHE_DIR = SCRIPT_DIR / ".mpl_cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib.pyplot as plt

DATA_DIR = SCRIPT_DIR / "data"
FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def style_axis(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
    ax.tick_params(axis="both", which="major", direction="in", length=3.0, width=0.6, top=True, right=True)
    ax.grid(True, linewidth=0.35, alpha=0.18)


def load_file(path: Path) -> dict:
    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        raise ValueError(f"Empty CSV file: {path}")
    numeric = [
        "time",
        "x1_filtered",
        "x2_filtered",
        "v1_filtered",
        "v2_filtered",
        "a1_filtered",
        "a2_filtered",
        "shaker_force_n",
        "voice_force_n",
    ]
    out = {key: np.asarray([float(row[key]) for row in rows], dtype=np.float64) for key in numeric}
    for key in ("block", "shaker_type", "voice_type", "chirp_direction"):
        out[key] = rows[0].get(key, "")
    return out


def choose_preview_files(paths: list[Path], max_files: int = 6) -> list[Path]:
    if len(paths) <= max_files:
        return paths
    chosen = []
    used_blocks = set()
    for path in paths:
        try:
            item = load_file(path)
        except Exception:
            continue
        block = str(item.get("block", ""))
        if block not in used_blocks:
            chosen.append(path)
            used_blocks.add(block)
        if len(chosen) >= max_files:
            return chosen
    for path in paths:
        if path not in chosen:
            chosen.append(path)
        if len(chosen) >= max_files:
            break
    return chosen


def plot_preview(paths: list[Path], save_path: Path):
    n = len(paths)
    if n == 0:
        raise FileNotFoundError(f"No trajectory_*.csv files found in {DATA_DIR}")
    fig, axes = plt.subplots(n, 4, figsize=(7.2, max(1.7, 1.35 * n)), sharex=False)
    if n == 1:
        axes = axes[None, :]

    for row_idx, path in enumerate(paths):
        item = load_file(path)
        time_s = np.asarray(item["time"], dtype=np.float64)
        title = (
            f"{path.stem} {item.get('block', '')}: "
            f"d={item.get('shaker_type', '')}, u={item.get('voice_type', '')}"
        )

        axes[row_idx, 0].plot(time_s, 1.0e3 * item["x1_filtered"], label="$x_1$", linewidth=0.8)
        axes[row_idx, 0].plot(time_s, 1.0e3 * item["x2_filtered"], label="$x_2$", linewidth=0.8)
        axes[row_idx, 0].set_ylabel("mm")
        axes[row_idx, 0].set_title(title)
        axes[row_idx, 0].legend(loc="upper right", frameon=False)

        axes[row_idx, 1].plot(time_s, item["v1_filtered"], label="$v_1$", linewidth=0.8)
        axes[row_idx, 1].plot(time_s, item["v2_filtered"], label="$v_2$", linewidth=0.8)
        axes[row_idx, 1].set_ylabel("m/s")
        axes[row_idx, 1].legend(loc="upper right", frameon=False)

        axes[row_idx, 2].plot(time_s, item["a1_filtered"], label="$a_1$", linewidth=0.8)
        axes[row_idx, 2].plot(time_s, item["a2_filtered"], label="$a_2$", linewidth=0.8)
        axes[row_idx, 2].set_ylabel("m/s$^2$")
        axes[row_idx, 2].legend(loc="upper right", frameon=False)

        axes[row_idx, 3].plot(time_s, item["shaker_force_n"], label="shaker", linewidth=0.8)
        axes[row_idx, 3].plot(time_s, item["voice_force_n"], label="voice", linewidth=0.8)
        axes[row_idx, 3].set_ylabel("N")
        axes[row_idx, 3].legend(loc="upper right", frameon=False)

        for ax in axes[row_idx]:
            ax.set_xlabel("Time [s]")
            style_axis(ax)

    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.08, top=0.95, hspace=0.55, wspace=0.32)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved figure: {save_path.resolve()}")
    return save_path


def main():
    paths = sorted(DATA_DIR.glob("trajectory_*.csv"))
    preview_paths = choose_preview_files(paths, max_files=6)
    return plot_preview(preview_paths, FIGURES_DIR / "hardware_protocol_preview.pdf")


if __name__ == "__main__":
    main()
