from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def load_sweep_support():
    support_path = PROJECT_DIR / "actuator_disturbance_sweep" / "support.py"
    spec = importlib.util.spec_from_file_location("actuator_disturbance_sweep_support_for_test", support_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load actuator disturbance support from {support_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SUPPORT = load_sweep_support()

DATA_DIR = SCRIPT_DIR / "data"
ROLLOUT_DIR = DATA_DIR / "rollouts"
FIGURES_DIR = SCRIPT_DIR / "figures"
UNKNOWN_FORCE_DIR = DATA_DIR / "unknown_forces"
for directory in (DATA_DIR, ROLLOUT_DIR, FIGURES_DIR, UNKNOWN_FORCE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

DT = SUPPORT.DT
DEFAULT_FORCES_N = [-1.0, -1.5, -2.0]
DEFAULT_DURATIONS_STEPS = [10, 20, 30]
DEFAULT_START_S = 2.0
DEFAULT_CONTROLLERS = [
    "passive",
    "adrc",
    "mpc",
    "deepc",
    "ar_ppo",
    "dgs_ar_ppo",
]
AVAILABLE_CONTROLLERS = list(DEFAULT_CONTROLLERS)


def force_token(force_n: float) -> str:
    return f"fm{int(round(abs(float(force_n)) * 10)):02d}"


def case_id(force_n: float, duration_steps: int) -> str:
    return f"{force_token(force_n)}_d{int(duration_steps):02d}"


def rollout_path(controller: str, force_n: float, duration_steps: int, repeat: int) -> Path:
    return ROLLOUT_DIR / case_id(force_n, duration_steps) / f"{controller}_repeat_{int(repeat):02d}.csv"


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


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(float(item.strip())) for item in str(text).split(",") if item.strip()]


def valid_rollout(path: Path, force_n: float, duration_steps: int, start_s: float, min_rows: int, detect_threshold: float) -> bool:
    if not path.exists():
        return False
    try:
        rows = read_csv_rows(path)
        force_values = SUPPORT.num_col(rows, "F_ud_requested", 0.0)
        active_force = force_values[np.abs(force_values) > 1.0e-9]
        measured_force = float(active_force[0]) if active_force.size else 0.0
        return (
            len(rows) == int(min_rows)
            and abs(measured_force - float(force_n)) < 1.0e-9
            and abs(float(rows[0].get("unknown_force_start_s", "nan")) - float(start_s)) < 1.0e-9
            and int(float(rows[0].get("unknown_force_duration_steps", -1))) == int(duration_steps)
            and abs(float(rows[0].get("diag_detect_threshold", "nan")) - float(detect_threshold)) < 1.0e-9
            and abs(float(rows[0].get("diag_dgs_recovery_threshold", "nan")) - float(SUPPORT.DGS_RECOVERY_THRESHOLD)) < 1.0e-9
            and int(float(rows[0].get("dgs_min_adrc_steps", -1))) == int(SUPPORT.DGS_MIN_ADRC_STEPS)
            and int(float(rows[0].get("dgs_recovery_consecutive_steps", -1))) == int(SUPPORT.DGS_RECOVERY_CONSECUTIVE_STEPS)
            and int(float(rows[0].get("dgs_refractory_steps", -1))) == int(SUPPORT.DGS_REFRACTORY_STEPS)
        )
    except Exception:
        return False


def save_test_plan(forces: list[float], durations: list[int], start_s: float, repeats: int, controllers: list[str]) -> None:
    rows = []
    for force in forces:
        for duration in durations:
            for controller in controllers:
                rows.append(
                    {
                        "case_id": case_id(force, duration),
                        "unknown_force_n": float(force),
                        "unknown_start_s": float(start_s),
                        "unknown_duration_steps": int(duration),
                        "unknown_duration_s": float(duration * DT),
                        "repeats": int(repeats),
                        "controller": controller,
                        "controller_label": SUPPORT.CONTROLLER_LABELS.get(controller, controller),
                        "detect_threshold": float(SUPPORT.DETECT_THRESHOLD),
                        "recovery_threshold": float(SUPPORT.DGS_RECOVERY_THRESHOLD),
                        "min_fallback_dwell_steps": int(SUPPORT.DGS_MIN_ADRC_STEPS),
                        "recovery_consecutive_steps": int(SUPPORT.DGS_RECOVERY_CONSECUTIVE_STEPS),
                        "refractory_steps": int(SUPPORT.DGS_REFRACTORY_STEPS),
                    }
                )
    write_csv(DATA_DIR / "test_plan.csv", rows)


def save_unknown_force(force: np.ndarray, force_n: float, duration_steps: int, start_s: float) -> None:
    rows = [
        {
            "sample": int(i),
            "time": float(i * DT),
            "unknown_force_n": float(force_n),
            "unknown_start_s": float(start_s),
            "unknown_duration_steps": int(duration_steps),
            "unknown_force_requested_n": float(value),
        }
        for i, value in enumerate(force)
    ]
    write_csv(UNKNOWN_FORCE_DIR / f"{case_id(force_n, duration_steps)}.csv", rows)


def metric_row(path: Path, force_n: float, duration_steps: int, start_s: float) -> dict:
    row = SUPPORT.rollout_metrics(path)
    rows = read_csv_rows(path)
    row["case_id"] = case_id(force_n, duration_steps)
    row["unknown_force_n"] = float(force_n)
    row["unknown_start_s"] = float(start_s)
    row["unknown_duration_steps"] = int(duration_steps)
    row["dgs_fallback_policy"] = rows[0].get("dgs_fallback_policy", "")
    return row


def refresh_metric_tables(
    controllers: list[str],
    forces: list[float],
    durations: list[int],
    repeats: int,
    start_s: float,
    min_rows: int,
    detect_threshold: float,
) -> None:
    repeat_rows = []
    for force in forces:
        for duration in durations:
            for controller in controllers:
                for repeat in range(1, int(repeats) + 1):
                    path = rollout_path(controller, force, duration, repeat)
                    if valid_rollout(path, force, duration, start_s, min_rows, detect_threshold):
                        repeat_rows.append(metric_row(path, force, duration, start_s))
    write_csv(DATA_DIR / "repeat_metrics.csv", repeat_rows)

    aggregate_rows = []
    keys = sorted({row["case_id"] for row in repeat_rows})
    for case in keys:
        for controller in controllers:
            group = [row for row in repeat_rows if row["case_id"] == case and row["controller"] == controller]
            if not group:
                continue
            out = {
                "case_id": case,
                "controller": controller,
                "controller_label": SUPPORT.CONTROLLER_LABELS.get(controller, controller),
                "dgs_fallback_policy": group[0].get("dgs_fallback_policy", ""),
                "unknown_force_n": float(group[0]["unknown_force_n"]),
                "unknown_start_s": float(start_s),
                "unknown_duration_steps": int(group[0]["unknown_duration_steps"]),
                "n_repeats": len(group),
            }
            for key in group[0]:
                if key in {"case_id", "controller", "controller_label", "path", "dgs_fallback_policy"}:
                    continue
                values = []
                for row in group:
                    try:
                        value = float(row[key])
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(value):
                        values.append(value)
                if values:
                    arr = np.asarray(values, dtype=np.float64)
                    out[f"{key}_mean"] = float(np.mean(arr))
                    out[f"{key}_std"] = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            aggregate_rows.append(out)
    write_csv(DATA_DIR / "aggregate_statistics.csv", aggregate_rows)


def collect_all(args: argparse.Namespace) -> None:
    forces = parse_float_list(args.forces)
    durations = parse_int_list(args.durations)
    controllers = [str(c) for c in args.controllers]
    _, disturbance = SUPPORT.load_fixed_disturbance()
    save_test_plan(forces, durations, args.unknown_start_s, args.repeats, controllers)
    for force in forces:
        for duration in durations:
            unknown_force, _ = SUPPORT.generate_unknown_force(force, args.unknown_start_s, duration, len(disturbance))
            save_unknown_force(unknown_force, force, duration, args.unknown_start_s)

    best_params = SUPPORT.CONTROL_SUPPORT.load_best_params()
    needs_ddm = any(c in SUPPORT.DGS_CONTROLLER_NAMES or c == "ar_ppo_est" for c in controllers)
    dynamics = SUPPORT.DDMDynamics() if needs_ddm else None
    controller_cache = {name: SUPPORT.make_controller(name, best_params, dynamics) for name in controllers}

    total_jobs = len(forces) * len(durations) * len(controllers) * int(args.repeats)
    completed = sum(
        valid_rollout(rollout_path(c, f, d, r), f, d, args.unknown_start_s, len(disturbance), args.detect_threshold)
        for f in forces
        for d in durations
        for c in controllers
        for r in range(1, int(args.repeats) + 1)
    )
    print(f"Test cases: forces={forces}, durations={durations}, start={args.unknown_start_s} s")
    print(f"Controllers: {', '.join(controllers)}")
    print(f"Completed rollouts: {completed}/{total_jobs}")

    hardware = SUPPORT.import_hardware_module()
    x_task = x_ser = y_ser = arduino_task = None
    try:
        x_task, x_ser, y_ser, arduino_task = hardware.all_devices_setup_on_demand()
        SUPPORT.stop_outputs(x_task, arduino_task)
        job_index = 0
        for repeat in range(1, int(args.repeats) + 1):
            for force in forces:
                for duration in durations:
                    unknown_force, meta = SUPPORT.generate_unknown_force(force, args.unknown_start_s, duration, len(disturbance))
                    for controller_name in controllers:
                        job_index += 1
                        path = rollout_path(controller_name, force, duration, repeat)
                        if valid_rollout(path, force, duration, args.unknown_start_s, len(disturbance), args.detect_threshold) and not args.overwrite:
                            print(f"Skipping {path.relative_to(SCRIPT_DIR)}: already saved")
                            continue
                        attempt = 0
                        while True:
                            attempt += 1
                            print(
                                f"Collecting {case_id(force, duration)} {SUPPORT.CONTROLLER_LABELS.get(controller_name, controller_name)} "
                                f"repeat {repeat}/{args.repeats} (job {job_index}/{total_jobs}) attempt={attempt}"
                            )
                            try:
                                saved = SUPPORT.collect_rollout(
                                    controller_name,
                                    repeat,
                                    controller_cache[controller_name],
                                    dynamics if controller_name in SUPPORT.DGS_CONTROLLER_NAMES or controller_name == "ar_ppo_est" else None,
                                    disturbance,
                                    unknown_force,
                                    hardware,
                                    x_task,
                                    x_ser,
                                    y_ser,
                                    arduino_task,
                                    settle_time_s=args.settle_time_s,
                                    post_wait_s=args.post_wait_s,
                                    jump_abort_m=args.jump_abort_m,
                                    abs_displacement_limit_m=args.abs_displacement_limit_m,
                                    detect_threshold=args.detect_threshold,
                                    unknown_force_start_s=meta["start_time"],
                                    unknown_force_duration_steps=duration,
                                    output_path=path,
                                )
                                print(f"Saved: {saved.relative_to(SCRIPT_DIR)}")
                                refresh_metric_tables(controllers, forces, durations, args.repeats, args.unknown_start_s, len(disturbance), args.detect_threshold)
                                break
                            except SUPPORT.RolloutAbort as exc:
                                print(f"Restarting {controller_name} repeat {repeat}/{args.repeats}: {exc}")
                                if attempt >= int(args.max_retries):
                                    raise RuntimeError(f"{controller_name} repeat {repeat} exceeded max_retries={args.max_retries}") from exc
    finally:
        SUPPORT.close_devices(x_task, x_ser, y_ser, arduino_task)

    refresh_metric_tables(controllers, forces, durations, args.repeats, args.unknown_start_s, len(disturbance), args.detect_threshold)
    print(f"Saved repeat metrics: {DATA_DIR / 'repeat_metrics.csv'}")
    print(f"Saved aggregate statistics: {DATA_DIR / 'aggregate_statistics.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Actuator-side disturbance smoke-test sweep.")
    parser.add_argument("--controllers", nargs="+", default=DEFAULT_CONTROLLERS, choices=AVAILABLE_CONTROLLERS)
    parser.add_argument("--forces", default=",".join(str(v) for v in DEFAULT_FORCES_N))
    parser.add_argument("--durations", default=",".join(str(v) for v in DEFAULT_DURATIONS_STEPS))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-retries", type=int, default=10)
    parser.add_argument("--settle-time-s", type=float, default=2.0)
    parser.add_argument("--post-wait-s", type=float, default=2.0)
    parser.add_argument("--jump-abort-m", type=float, default=0.008)
    parser.add_argument("--abs-displacement-limit-m", type=float, default=0.04)
    parser.add_argument("--unknown-start-s", type=float, default=DEFAULT_START_S)
    parser.add_argument("--detect-threshold", type=float, default=SUPPORT.DETECT_THRESHOLD)
    return parser.parse_args()


if __name__ == "__main__":
    collect_all(parse_args())
