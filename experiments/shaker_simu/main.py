from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils.filters import CUTOFF_HZ, lowpass_alpha


DATA_DIR = SCRIPT_DIR / "data"
ROLLOUT_DIR = DATA_DIR / "rollouts"
FIGURES_DIR = SCRIPT_DIR / "figures"
HARDWARE_AGGREGATE_CSV = PROJECT_DIR / "shaker_disturbance" / "data" / "aggregate_statistics.csv"

DT = 0.03
STATE_DIM = 4
U_MAX = 2.0
DEFAULT_CONTROLLERS = ["passive", "adrc", "mpc", "deepc", "ar_ppo"]
CONTROLLER_LABELS = {
    "passive": "Passive",
    "adrc": "ADRC",
    "mpc": "MPC",
    "deepc": "DeePC",
    "ar_ppo": "AR-PPO",
}

for directory in (DATA_DIR, ROLLOUT_DIR, FIGURES_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def load_actuator_support():
    support_path = PROJECT_DIR / "actuator_disturbance" / "main.py"
    spec = importlib.util.spec_from_file_location("actuator_support_for_shaker_simu", support_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load actuator support from {support_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SUPPORT = load_actuator_support()


def read_rows(path: Path) -> list[dict]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
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


def step_feature(state: np.ndarray, action: float, disturbance: float) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(state, dtype=np.float32).reshape(STATE_DIM),
            np.asarray([float(action), float(disturbance)], dtype=np.float32),
        ]
    ).astype(np.float32)


def load_disturbance(duration_s: float | None) -> np.ndarray:
    time_s, disturbance = SUPPORT.load_fixed_disturbance()
    disturbance = np.asarray(disturbance, dtype=np.float32).reshape(-1)
    if duration_s is not None and float(duration_s) > 0.0:
        keep = int(round(float(duration_s) / DT)) + 1
        disturbance = disturbance[: min(len(disturbance), keep)]
        time_s = time_s[: len(disturbance)]
    rows = [
        {"sample": int(i), "time": float(time_s[i] if i < len(time_s) else i * DT), "shaker_force_n": float(value)}
        for i, value in enumerate(disturbance)
    ]
    write_csv(DATA_DIR / "fixed_test_disturbance.csv", rows)
    return disturbance.astype(np.float32)


def rollout_controller(controller_name: str, controller, dynamics, disturbance: np.ndarray) -> list[dict]:
    controller.reset()
    state = np.zeros(STATE_DIM, dtype=np.float32)
    history = dynamics.zero_history()
    previous_action = 0.0
    a2_filtered = 0.0
    alpha = lowpass_alpha(DT, CUTOFF_HZ)
    rows = []
    for sample, shaker_force in enumerate(np.asarray(disturbance, dtype=np.float32)):
        obs = {
            "state": state.astype(np.float64),
            "disturbance": float(shaker_force),
            "u_prev": float(previous_action),
            "dt": DT,
        }
        start = time.perf_counter()
        u_cmd, info = controller.act(obs)
        control_ms = (time.perf_counter() - start) * 1000.0
        u_app = float(np.clip(float(u_cmd), -U_MAX, U_MAX))
        rows.append(
            {
                "controller": controller_name,
                "controller_label": CONTROLLER_LABELS[controller_name],
                "sample": int(sample),
                "time": float(sample * DT),
                "cutoff_hz": CUTOFF_HZ,
                "x1": float(state[0]),
                "v1": float(state[1]),
                "a1": 0.0,
                "x2": float(state[2]),
                "v2": float(state[3]),
                "a2": float(a2_filtered),
                "x1_filtered": float(state[0]),
                "v1_filtered": float(state[1]),
                "a1_filtered": 0.0,
                "x2_filtered": float(state[2]),
                "v2_filtered": float(state[3]),
                "a2_filtered": float(a2_filtered),
                "shaker_force_n": float(shaker_force),
                "u_raw": float(info.get("u_raw", u_cmd)),
                "u_applied": float(u_app),
                "voice_force_n": float(u_app),
                "base_u": float(info.get("base_u", 0.0)),
                "ppo_residual": float(info.get("ppo_residual", 0.0)),
                "saturation_flag": int(info.get("saturation_flag", 0)),
                "fallback_flag": int(info.get("fallback_flag", 0)),
                "computation_time_ms": float(control_ms),
                "solve_time_ms": float(info.get("solve_time_ms", control_ms)),
            }
        )
        next_state = dynamics.predict_next(state, u_app, float(shaker_force), history)
        history = dynamics.append_history(history, step_feature(state, u_app, float(shaker_force)))
        a2_raw = (float(next_state[3]) - float(state[3])) / DT
        a2_filtered = float(a2_filtered + alpha * (a2_raw - a2_filtered))
        state = next_state.astype(np.float32)
        previous_action = float(u_app)
    return rows


def metric_array(rows: list[dict], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def rollout_metrics(controller_name: str, rows: list[dict], path: Path) -> dict:
    out = {
        "controller": controller_name,
        "controller_label": CONTROLLER_LABELS[controller_name],
        "path": str(path),
    }
    for key in ("x2", "v2", "a2"):
        values = metric_array(rows, key)
        out[f"{key}_rms"] = float(np.sqrt(np.mean(values**2)))
        out[f"{key}_mean_abs"] = float(np.mean(np.abs(values)))
        out[f"{key}_peak_abs"] = float(np.max(np.abs(values)))
    compute = metric_array(rows, "computation_time_ms")
    out["compute_time_mean_ms"] = float(np.mean(compute))
    out["compute_time_max_ms"] = float(np.max(compute))
    out["saturation_count"] = int(np.sum(metric_array(rows, "saturation_flag") > 0.5))
    out["fallback_count"] = int(np.sum(metric_array(rows, "fallback_flag") > 0.5))
    return out


def copy_hardware_aggregate() -> list[dict]:
    rows = read_rows(HARDWARE_AGGREGATE_CSV)
    write_csv(DATA_DIR / "hardware_aggregate_statistics.csv", rows)
    return rows


def compare_with_hardware(sim_rows: list[dict], hardware_rows: list[dict]) -> list[dict]:
    hardware_by_controller = {row["controller"]: row for row in hardware_rows}
    out = []
    for sim in sim_rows:
        controller = sim["controller"]
        hardware = hardware_by_controller.get(controller)
        if hardware is None:
            continue
        row = {
            "controller": controller,
            "controller_label": sim["controller_label"],
        }
        for metric in ("x2_rms", "v2_rms", "a2_rms"):
            sim_value = float(sim[metric])
            hw_mean = float(hardware[f"{metric}_mean"])
            hw_std = float(hardware[f"{metric}_std"])
            row[f"{metric}_sim"] = sim_value
            row[f"{metric}_hardware_mean"] = hw_mean
            row[f"{metric}_hardware_std"] = hw_std
            row[f"{metric}_sim_over_hardware"] = float(sim_value / hw_mean) if abs(hw_mean) > 1.0e-12 else np.nan
            row[f"{metric}_sim_minus_hardware"] = float(sim_value - hw_mean)
        out.append(row)
    return out


def run(args: argparse.Namespace) -> None:
    controllers = [str(item) for item in args.controllers]
    disturbance = load_disturbance(args.duration_s)
    dynamics = SUPPORT.DDMDynamics()
    best_params = SUPPORT.CONTROL_SUPPORT.load_best_params()
    sim_metric_rows = []
    for idx, controller_name in enumerate(controllers, start=1):
        print(f"Simulating {CONTROLLER_LABELS[controller_name]} ({idx}/{len(controllers)})")
        controller = SUPPORT.make_controller(controller_name, best_params, dynamics)
        rows = rollout_controller(controller_name, controller, dynamics, disturbance)
        path = ROLLOUT_DIR / f"{controller_name}.csv"
        write_csv(path, rows)
        sim_metric_rows.append(rollout_metrics(controller_name, rows, path))
    write_csv(DATA_DIR / "simulated_metrics.csv", sim_metric_rows)
    hardware_rows = copy_hardware_aggregate()
    comparison_rows = compare_with_hardware(sim_metric_rows, hardware_rows)
    write_csv(DATA_DIR / "simulation_vs_hardware.csv", comparison_rows)
    print(f"Saved {DATA_DIR / 'simulation_vs_hardware.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run shaker_disturbance controllers inside the selected DDM surrogate.")
    parser.add_argument("--controllers", nargs="+", default=DEFAULT_CONTROLLERS, choices=DEFAULT_CONTROLLERS)
    parser.add_argument("--duration-s", type=float, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
