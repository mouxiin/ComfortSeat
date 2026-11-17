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

from utils.filters import CUTOFF_HZ, OnlineKinematicsFilter
from utils.hardware_io import (
    close_devices,
    finish_hardware_trial,
    import_hardware_module,
    prepare_hardware_for_trial,
    quiet_put_voice_coil_force,
    shaker_force_to_input_voltage,
    stop_outputs,
    wait_until_still,
)


def load_controller_support():
    support_path = PROJECT_DIR / "shaker_disturbance" / "main.py"
    spec = importlib.util.spec_from_file_location("shaker_disturbance_support", support_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load controller support from {support_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONTROL_SUPPORT = load_controller_support()

DATA_DIR = SCRIPT_DIR / "data"
ROLLOUT_DIR = DATA_DIR / "rollouts"
FIGURES_DIR = SCRIPT_DIR / "figures"
for directory in (DATA_DIR, ROLLOUT_DIR, FIGURES_DIR):
    directory.mkdir(parents=True, exist_ok=True)

DT = 0.03
DURATION_S = 9.0
STEPS = int(round(DURATION_S / DT)) + 1
SHAKER_NEWTON_PER_VOLT = 3.0
SHAKER_FORCE_LIMIT_N = 9.0
SHAKER_SIGNED_VOLTAGE_LIMIT = 3.0
DEFAULT_FREQUENCIES_HZ = [1.0, 2.0, 3.0, 4.0, 5.0]
DEFAULT_VOLTAGE_AMPLITUDES_V = [0.6, 1.2, 1.8, 2.4, 3.0]
DEFAULT_CONTROLLERS = ["passive", "adrc", "mpc", "deepc", "ar_ppo"]
CONTROLLER_LABELS = dict(CONTROL_SUPPORT.CONTROLLER_LABELS)


class RolloutAbort(RuntimeError):
    pass


def case_id(frequency_hz: float, voltage_amplitude_v: float) -> str:
    voltage_token = f"{float(voltage_amplitude_v):.1f}".replace(".", "d")
    return f"f{int(round(float(frequency_hz))):02d}_v{voltage_token}"


def rollout_path(controller: str, frequency_hz: float, voltage_amplitude_v: float, repeat: int) -> Path:
    return ROLLOUT_DIR / case_id(frequency_hz, voltage_amplitude_v) / f"{controller}_repeat_{int(repeat):02d}.csv"


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


def read_csv_rows(path: Path) -> list[dict]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def valid_rollout(path: Path, min_rows: int = STEPS) -> bool:
    if not path.exists():
        return False
    try:
        rows = read_csv_rows(path)
        return len(rows) == int(min_rows) and "x2" in rows[0] and "computation_time_ms" in rows[0]
    except Exception:
        return False


def generate_sine_disturbance(frequency_hz: float, voltage_amplitude_v: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time_s = np.arange(STEPS, dtype=np.float64) * DT
    voltage = float(voltage_amplitude_v) * np.sin(2.0 * np.pi * float(frequency_hz) * time_s)
    voltage = np.clip(voltage, -SHAKER_SIGNED_VOLTAGE_LIMIT, SHAKER_SIGNED_VOLTAGE_LIMIT)
    force = SHAKER_NEWTON_PER_VOLT * voltage
    force = np.clip(force, -SHAKER_FORCE_LIMIT_N, SHAKER_FORCE_LIMIT_N)
    return time_s, force.astype(np.float64), voltage.astype(np.float64)


def save_sweep_plan(frequencies: list[float], voltage_amplitudes: list[float]) -> None:
    rows = []
    for frequency in frequencies:
        for voltage in voltage_amplitudes:
            rows.append(
                {
                    "case_id": case_id(frequency, voltage),
                    "frequency_hz": float(frequency),
                    "voltage_amplitude_v": float(voltage),
                    "force_amplitude_n": float(SHAKER_NEWTON_PER_VOLT * voltage),
                    "duration_s": DURATION_S,
                    "dt": DT,
                    "steps": STEPS,
                }
            )
    write_csv(
        DATA_DIR / "sweep_plan.csv",
        rows,
        ["case_id", "frequency_hz", "voltage_amplitude_v", "force_amplitude_n", "duration_s", "dt", "steps"],
    )


def save_case_disturbance(frequency_hz: float, voltage_amplitude_v: float, time_s: np.ndarray, force: np.ndarray, voltage: np.ndarray) -> None:
    rows = [
        {
            "sample": int(i),
            "time": float(t),
            "frequency_hz": float(frequency_hz),
            "voltage_amplitude_v": float(voltage_amplitude_v),
            "force_amplitude_n": float(SHAKER_NEWTON_PER_VOLT * voltage_amplitude_v),
            "shaker_force_n": float(f),
            "shaker_voltage_v": float(v),
        }
        for i, (t, f, v) in enumerate(zip(time_s, force, voltage))
    ]
    write_csv(
        DATA_DIR / "disturbances" / f"{case_id(frequency_hz, voltage_amplitude_v)}.csv",
        rows,
        [
            "sample",
            "time",
            "frequency_hz",
            "voltage_amplitude_v",
            "force_amplitude_n",
            "shaker_force_n",
            "shaker_voltage_v",
        ],
    )


def save_rollout(path: Path, rows: list[dict]) -> None:
    fields = [
        "case_id",
        "controller",
        "controller_label",
        "repeat",
        "frequency_hz",
        "voltage_amplitude_v",
        "force_amplitude_n",
        "sample",
        "time",
        "timestamp_abs",
        "dt_sample",
        "cutoff_hz",
        "x1_raw",
        "x2_raw",
        "x1",
        "v1",
        "a1",
        "x2",
        "v2",
        "a2",
        "x1_filtered",
        "v1_filtered",
        "a1_filtered",
        "x2_filtered",
        "v2_filtered",
        "a2_filtered",
        "shaker_force_n",
        "shaker_voltage_v",
        "u_raw",
        "u_applied",
        "voice_force_n",
        "base_u",
        "ppo_residual",
        "saturation_flag",
        "fallback_flag",
        "computation_time_ms",
        "solve_time_ms",
    ]
    write_csv(path, rows, fields)


def collect_rollout(
    controller_name: str,
    repeat: int,
    frequency_hz: float,
    voltage_amplitude_v: float,
    controller,
    force: np.ndarray,
    voltage: np.ndarray,
    hardware,
    x_task,
    x_ser,
    y_ser,
    arduino_task,
    *,
    settle_time_s: float,
    post_wait_s: float,
    jump_abort_m: float,
    abs_displacement_limit_m: float,
) -> Path:
    controller.reset()
    prepare_hardware_for_trial(x_task, x_ser, y_ser, arduino_task)
    offset = wait_until_still(hardware.get_xy_by_serial, x_ser, y_ser, duration_s=settle_time_s, sample_dt=DT)
    filt = OnlineKinematicsFilter(cutoff_hz=CUTOFF_HZ, dimensions=2)
    rows: list[dict] = []
    prev_raw = None
    u_prev = 0.0
    cid = case_id(frequency_hz, voltage_amplitude_v)
    force_amplitude_n = SHAKER_NEWTON_PER_VOLT * float(voltage_amplitude_v)

    try:
        x_task.start()
    except Exception:
        pass

    start_time = time.perf_counter()
    filt.reset(np.zeros(2, dtype=np.float64), start_time)
    try:
        for step in range(len(force)):
            target = start_time + step * DT
            while True:
                remaining = target - time.perf_counter()
                if remaining <= 0.0:
                    break
                time.sleep(min(remaining, 0.002))

            now = time.perf_counter()
            raw_xy = np.asarray(hardware.get_xy_by_serial(x_ser, y_ser), dtype=np.float64) - offset
            if prev_raw is not None and np.any(np.abs(raw_xy - prev_raw) > float(jump_abort_m)):
                raise RolloutAbort(f"sudden displacement jump at sample {step}: {raw_xy - prev_raw}")
            if np.any(np.abs(raw_xy) > float(abs_displacement_limit_m)):
                raise RolloutAbort(f"absolute displacement limit exceeded at sample {step}: {raw_xy}")

            x_f, v_f, a_f, dt_sample = filt.update(raw_xy, now)
            shaker_force = float(force[step])
            shaker_voltage = float(voltage[step])
            expected_voltage = shaker_force_to_input_voltage(shaker_force)
            if abs(expected_voltage - shaker_voltage) > 1.0e-6:
                shaker_voltage = expected_voltage
            obs = {
                "state": np.asarray([x_f[0], v_f[0], x_f[1], v_f[1]], dtype=np.float64),
                "disturbance": shaker_force,
                "u_prev": u_prev,
                "dt": float(dt_sample),
            }
            control_start = time.perf_counter()
            u_cmd, info = controller.act(obs)
            computation_ms = (time.perf_counter() - control_start) * 1000.0
            u_applied = quiet_put_voice_coil_force(arduino_task, u_cmd)
            x_task.write(shaker_voltage, auto_start=False)

            rows.append(
                {
                    "case_id": cid,
                    "controller": controller_name,
                    "controller_label": CONTROLLER_LABELS.get(controller_name, controller_name),
                    "repeat": int(repeat),
                    "frequency_hz": float(frequency_hz),
                    "voltage_amplitude_v": float(voltage_amplitude_v),
                    "force_amplitude_n": float(force_amplitude_n),
                    "sample": int(step),
                    "time": float(now - start_time),
                    "timestamp_abs": float(now),
                    "dt_sample": float(dt_sample),
                    "cutoff_hz": CUTOFF_HZ,
                    "x1_raw": float(raw_xy[0]),
                    "x2_raw": float(raw_xy[1]),
                    "x1": float(x_f[0]),
                    "v1": float(v_f[0]),
                    "a1": float(a_f[0]),
                    "x2": float(x_f[1]),
                    "v2": float(v_f[1]),
                    "a2": float(a_f[1]),
                    "x1_filtered": float(x_f[0]),
                    "v1_filtered": float(v_f[0]),
                    "a1_filtered": float(a_f[0]),
                    "x2_filtered": float(x_f[1]),
                    "v2_filtered": float(v_f[1]),
                    "a2_filtered": float(a_f[1]),
                    "shaker_force_n": shaker_force,
                    "shaker_voltage_v": float(shaker_voltage),
                    "u_raw": float(info.get("u_raw", u_cmd)),
                    "u_applied": float(u_applied),
                    "voice_force_n": float(u_applied),
                    "base_u": float(info.get("base_u", 0.0)),
                    "ppo_residual": float(info.get("ppo_residual", 0.0)),
                    "saturation_flag": int(info.get("saturation_flag", 0)),
                    "fallback_flag": int(info.get("fallback_flag", 0)),
                    "computation_time_ms": float(computation_ms),
                    "solve_time_ms": float(info.get("solve_time_ms", computation_ms)),
                }
            )
            prev_raw = raw_xy.copy()
            u_prev = float(u_applied)
    finally:
        finish_hardware_trial(x_task, x_ser, y_ser, arduino_task, post_wait_s=post_wait_s)

    if len(rows) != len(force):
        raise RuntimeError(f"Expected {len(force)} rows, collected {len(rows)} rows.")
    path = rollout_path(controller_name, frequency_hz, voltage_amplitude_v, repeat)
    save_rollout(path, rows)
    return path


def rollout_metrics(path: Path) -> dict:
    rows = read_csv_rows(path)
    first = rows[0]
    out = {
        "case_id": first["case_id"],
        "controller": first["controller"],
        "controller_label": first.get("controller_label", first["controller"]),
        "repeat": int(float(first["repeat"])),
        "frequency_hz": float(first["frequency_hz"]),
        "voltage_amplitude_v": float(first["voltage_amplitude_v"]),
        "force_amplitude_n": float(first["force_amplitude_n"]),
        "path": str(path),
    }
    for key in ("x2", "v2", "a2"):
        arr = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
        out[f"{key}_rms"] = float(np.sqrt(np.mean(arr**2)))
        out[f"{key}_mean_abs"] = float(np.mean(np.abs(arr)))
        out[f"{key}_peak_abs"] = float(np.max(np.abs(arr)))
    compute = np.asarray([float(r["computation_time_ms"]) for r in rows], dtype=np.float64)
    out["compute_time_mean_ms"] = float(np.mean(compute))
    out["compute_time_std_ms"] = float(np.std(compute, ddof=1)) if len(compute) > 1 else 0.0
    out["compute_time_max_ms"] = float(np.max(compute))
    out["saturation_count"] = int(sum(int(float(r.get("saturation_flag", 0))) for r in rows))
    out["fallback_count"] = int(sum(int(float(r.get("fallback_flag", 0))) for r in rows))
    return out


def refresh_metric_tables(controllers: list[str], frequencies: list[float], voltages: list[float], repeats: int) -> None:
    repeat_rows = []
    for frequency in frequencies:
        for voltage in voltages:
            for controller in controllers:
                for repeat in range(1, int(repeats) + 1):
                    path = rollout_path(controller, frequency, voltage, repeat)
                    if path.exists():
                        repeat_rows.append(rollout_metrics(path))
    repeat_fields = [
        "case_id",
        "controller",
        "controller_label",
        "repeat",
        "frequency_hz",
        "voltage_amplitude_v",
        "force_amplitude_n",
        "path",
        "x2_rms",
        "x2_mean_abs",
        "x2_peak_abs",
        "v2_rms",
        "v2_mean_abs",
        "v2_peak_abs",
        "a2_rms",
        "a2_mean_abs",
        "a2_peak_abs",
        "compute_time_mean_ms",
        "compute_time_std_ms",
        "compute_time_max_ms",
        "saturation_count",
        "fallback_count",
    ]
    write_csv(DATA_DIR / "repeat_metrics.csv", repeat_rows, repeat_fields)

    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in repeat_rows:
        grouped.setdefault((row["case_id"], row["controller"]), []).append(row)
    aggregate_rows = []
    metrics = [field for field in repeat_fields if field not in {"case_id", "controller", "controller_label", "repeat", "frequency_hz", "voltage_amplitude_v", "force_amplitude_n", "path"}]
    for (cid, controller), group in sorted(grouped.items()):
        first = group[0]
        out = {
            "case_id": cid,
            "controller": controller,
            "controller_label": first["controller_label"],
            "frequency_hz": first["frequency_hz"],
            "voltage_amplitude_v": first["voltage_amplitude_v"],
            "force_amplitude_n": first["force_amplitude_n"],
            "n_repeats": len(group),
        }
        for metric in metrics:
            values = np.asarray([float(r[metric]) for r in group], dtype=np.float64)
            out[f"{metric}_mean"] = float(np.mean(values))
            out[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        aggregate_rows.append(out)
    aggregate_fields = []
    for row in aggregate_rows:
        for key in row:
            if key not in aggregate_fields:
                aggregate_fields.append(key)
    write_csv(DATA_DIR / "aggregate_statistics.csv", aggregate_rows, aggregate_fields)


def collect_all(args: argparse.Namespace) -> None:
    frequencies = [float(v) for v in args.frequencies]
    voltages = [float(v) for v in args.voltage_amplitudes]
    controllers = [str(c) for c in args.controllers]
    save_sweep_plan(frequencies, voltages)
    best_params = CONTROL_SUPPORT.load_best_params()
    controller_cache = {name: CONTROL_SUPPORT.make_controller(name, best_params) for name in controllers}

    total_jobs = len(frequencies) * len(voltages) * len(controllers) * int(args.repeats)
    completed = 0
    for frequency in frequencies:
        for voltage in voltages:
            for controller in controllers:
                for repeat in range(1, int(args.repeats) + 1):
                    completed += int(valid_rollout(rollout_path(controller, frequency, voltage, repeat)))
    print(f"Sweep plan: {DATA_DIR / 'sweep_plan.csv'}")
    print(f"Completed rollouts: {completed}/{total_jobs}")

    hardware = import_hardware_module()
    x_task = x_ser = y_ser = arduino_task = None
    try:
        x_task, x_ser, y_ser, arduino_task = hardware.all_devices_setup_on_demand()
        stop_outputs(x_task, arduino_task)
        job_index = 0
        for frequency in frequencies:
            for voltage in voltages:
                time_s, force, voltage_signal = generate_sine_disturbance(frequency, voltage)
                save_case_disturbance(frequency, voltage, time_s, force, voltage_signal)
                for controller_name in controllers:
                    controller = controller_cache[controller_name]
                    for repeat in range(1, int(args.repeats) + 1):
                        job_index += 1
                        path = rollout_path(controller_name, frequency, voltage, repeat)
                        if valid_rollout(path) and not args.overwrite:
                            print(
                                f"Skipping {case_id(frequency, voltage)} {controller_name} "
                                f"repeat {repeat}/{args.repeats}: already saved"
                            )
                            continue
                        attempt = 0
                        while True:
                            attempt += 1
                            print(
                                f"Collecting {case_id(frequency, voltage)} "
                                f"freq={frequency:g}Hz amp={voltage:g}V/{3.0 * voltage:g}N "
                                f"{CONTROLLER_LABELS.get(controller_name, controller_name)} "
                                f"repeat {repeat}/{args.repeats} "
                                f"(job {job_index}/{total_jobs}) attempt={attempt}"
                            )
                            try:
                                saved = collect_rollout(
                                    controller_name,
                                    repeat,
                                    frequency,
                                    voltage,
                                    controller,
                                    force,
                                    voltage_signal,
                                    hardware,
                                    x_task,
                                    x_ser,
                                    y_ser,
                                    arduino_task,
                                    settle_time_s=args.settle_time_s,
                                    post_wait_s=args.post_wait_s,
                                    jump_abort_m=args.jump_abort_m,
                                    abs_displacement_limit_m=args.abs_displacement_limit_m,
                                )
                                print(f"Saved: {saved.relative_to(SCRIPT_DIR)}")
                                refresh_metric_tables(controllers, frequencies, voltages, args.repeats)
                                break
                            except RolloutAbort as exc:
                                print(f"Restarting {case_id(frequency, voltage)} {controller_name} repeat {repeat}: {exc}")
                                if attempt >= int(args.max_retries):
                                    raise RuntimeError(
                                        f"{case_id(frequency, voltage)} {controller_name} repeat {repeat} "
                                        f"exceeded max_retries={args.max_retries}"
                                    ) from exc
    finally:
        close_devices(x_task, x_ser, y_ser, arduino_task)
    refresh_metric_tables(controllers, frequencies, voltages, args.repeats)
    print(f"Saved repeat metrics: {DATA_DIR / 'repeat_metrics.csv'}")
    print(f"Saved aggregate statistics: {DATA_DIR / 'aggregate_statistics.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hardware shaker disturbance frequency/amplitude sweep.")
    parser.add_argument("--controllers", nargs="+", default=DEFAULT_CONTROLLERS, choices=DEFAULT_CONTROLLERS)
    parser.add_argument("--frequencies", type=float, nargs="+", default=DEFAULT_FREQUENCIES_HZ)
    parser.add_argument("--voltage-amplitudes", type=float, nargs="+", default=DEFAULT_VOLTAGE_AMPLITUDES_V)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-retries", type=int, default=10)
    parser.add_argument("--settle-time-s", type=float, default=2.0)
    parser.add_argument("--post-wait-s", type=float, default=2.0)
    parser.add_argument("--jump-abort-m", type=float, default=0.008)
    parser.add_argument("--abs-displacement-limit-m", type=float, default=0.04)
    return parser.parse_args()


if __name__ == "__main__":
    collect_all(parse_args())
