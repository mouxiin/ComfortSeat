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


def load_local_support():
    support_path = SCRIPT_DIR / "support.py"
    spec = importlib.util.spec_from_file_location("actuator_disturbance_sweep_support", support_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load actuator disturbance sweep support from {support_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SUPPORT = load_local_support()

DATA_DIR = SCRIPT_DIR / "data"
ROLLOUT_DIR = DATA_DIR / "rollouts"
FIGURES_DIR = SCRIPT_DIR / "figures"
for directory in (DATA_DIR, ROLLOUT_DIR, FIGURES_DIR):
    directory.mkdir(parents=True, exist_ok=True)

DT = SUPPORT.DT
DURATION_S = SUPPORT.DURATION_S
STEPS = SUPPORT.STEPS
UNKNOWN_START_S = SUPPORT.UNKNOWN_START_S
DEFAULT_TEST_FORCES_N = [-1.0, -1.5, -2.0]
DEFAULT_TEST_DURATION_STEPS = [10, 20, 30]
DISTURBANCE_METRIC_POST_S = SUPPORT.DISTURBANCE_METRIC_POST_S
ESTIMATOR_MODE = SUPPORT.ESTIMATOR_MODE
ACTUATOR_DISTURBANCE_MODEL = SUPPORT.ACTUATOR_DISTURBANCE_MODEL
DETECTOR_GATE_MODE = SUPPORT.DETECTOR_GATE_MODE
DGS_RECOVERY_THRESHOLD = SUPPORT.DGS_RECOVERY_THRESHOLD
DGS_RECOVERY_CONSECUTIVE_STEPS = SUPPORT.DGS_RECOVERY_CONSECUTIVE_STEPS
DGS_MIN_ADRC_STEPS = SUPPORT.DGS_MIN_ADRC_STEPS
DGS_REFRACTORY_STEPS = SUPPORT.DGS_REFRACTORY_STEPS
DEFAULT_CONTROLLERS = list(SUPPORT.DEFAULT_CONTROLLERS)
AVAILABLE_CONTROLLERS = list(SUPPORT.AVAILABLE_CONTROLLERS)
CONTROLLER_LABELS = dict(SUPPORT.CONTROLLER_LABELS)


def force_token(force_n: float) -> str:
    return f"fm{int(round(abs(float(force_n)) * 10)):02d}"


def case_id(force_n: float, duration_steps: int) -> str:
    return f"{force_token(force_n)}_d{int(duration_steps):02d}"


def rollout_path(controller: str, force_n: float, duration_steps: int, repeat: int) -> Path:
    return ROLLOUT_DIR / case_id(force_n, duration_steps) / f"{controller}_repeat_{int(repeat):02d}.csv"


def read_csv_rows(path: Path) -> list[dict]:
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


def valid_rollout(
    path: Path,
    min_rows: int = STEPS,
    *,
    force_n: float | None = None,
    duration_steps: int | None = None,
    start_s: float | None = None,
) -> bool:
    if not path.exists():
        return False
    try:
        rows = read_csv_rows(path)
        force_values = SUPPORT.num_col(rows, "F_ud_requested", 0.0)
        active_force = force_values[np.abs(force_values) > 1.0e-9]
        measured_force = float(active_force[0]) if active_force.size else 0.0
        return (
            len(rows) == int(min_rows)
            and "x2" in rows[0]
            and "F_ud_true" in rows[0]
            and "unknown_force_start_s" in rows[0]
            and "unknown_force_duration_steps" in rows[0]
            and rows[0].get("detector_gate_mode") == DETECTOR_GATE_MODE
            and rows[0].get("estimator_mode") == ESTIMATOR_MODE
            and rows[0].get("actuator_disturbance_model") == ACTUATOR_DISTURBANCE_MODEL
            and abs(float(rows[0].get("unknown_compensation_gain", "nan")) - float(SUPPORT.UNKNOWN_COMPENSATION_GAIN)) < 1.0e-9
            and int(float(rows[0].get("unknown_allow_retrigger", -1))) == int(bool(SUPPORT.UNKNOWN_ALLOW_RETRIGGER))
            and int(float(rows[0].get("dgs_min_adrc_steps", -1))) == int(DGS_MIN_ADRC_STEPS)
            and int(float(rows[0].get("dgs_recovery_consecutive_steps", -1))) == int(DGS_RECOVERY_CONSECUTIVE_STEPS)
            and int(float(rows[0].get("dgs_refractory_steps", -1))) == int(DGS_REFRACTORY_STEPS)
            and (force_n is None or abs(measured_force - float(force_n)) < 1.0e-9)
            and (duration_steps is None or int(float(rows[0].get("unknown_force_duration_steps", -1))) == int(duration_steps))
            and (start_s is None or abs(float(rows[0].get("unknown_force_start_s", "nan")) - float(start_s)) < 1.0e-9)
            and "actuator_gain_requested" not in rows[0]
        )
    except Exception:
        return False


def generate_unknown_pulse(
    force_n: float,
    duration_steps: int,
    *,
    start_s: float = UNKNOWN_START_S,
    steps: int = STEPS,
) -> tuple[np.ndarray, np.ndarray, dict]:
    time_s = np.arange(int(steps), dtype=np.float64) * DT
    force = np.zeros(int(steps), dtype=np.float64)
    start_idx = int(round(float(start_s) / DT))
    end_idx = min(int(steps), start_idx + max(1, int(duration_steps)))
    force[start_idx:end_idx] = float(np.clip(force_n, -SUPPORT.U_MAX, SUPPORT.U_MAX))
    meta = {
        "unknown_force_n": float(force_n),
        "unknown_start_s": float(start_s),
        "unknown_start_idx": int(start_idx),
        "unknown_duration_steps": int(end_idx - start_idx),
        "unknown_duration_s": float((end_idx - start_idx) * DT),
    }
    return time_s, force.astype(np.float64), meta


def save_sweep_plan(forces: list[float], durations: list[int], start_s: float) -> None:
    rows = []
    for force in forces:
        for duration in durations:
            rows.append(
                {
                    "case_id": case_id(force, duration),
                    "unknown_force_n": float(force),
                    "unknown_start_s": float(start_s),
                    "unknown_duration_steps": int(duration),
                    "unknown_duration_s": float(int(duration) * DT),
                    "duration_s": DURATION_S,
                    "dt": DT,
                    "steps": STEPS,
                }
            )
    write_csv(
        DATA_DIR / "sweep_plan.csv",
        rows,
        ["case_id", "unknown_force_n", "unknown_start_s", "unknown_duration_steps", "unknown_duration_s", "duration_s", "dt", "steps"],
    )


def save_case_unknown_force(force_n: float, duration_steps: int, start_s: float, time_s: np.ndarray, force: np.ndarray) -> None:
    rows = [
        {
            "sample": int(i),
            "time": float(t),
            "unknown_force_n": float(force_n),
            "unknown_start_s": float(start_s),
            "unknown_duration_steps": int(duration_steps),
            "unknown_force_requested_n": float(f),
        }
        for i, (t, f) in enumerate(zip(time_s, force))
    ]
    write_csv(
        DATA_DIR / "unknown_forces" / f"{case_id(force_n, duration_steps)}.csv",
        rows,
        ["sample", "time", "unknown_force_n", "unknown_start_s", "unknown_duration_steps", "unknown_force_requested_n"],
    )


def make_controller(name: str, best_params: dict, dynamics):
    return SUPPORT.make_controller(name, best_params, dynamics)


def collect_rollout(
    controller_name: str,
    repeat: int,
    force_n: float,
    duration_steps: int,
    unknown_start_s: float,
    controller,
    dynamics,
    disturbance: np.ndarray,
    unknown_force: np.ndarray,
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
    detect_threshold: float,
) -> Path:
    return SUPPORT.collect_rollout(
        controller_name,
        repeat,
        controller,
        dynamics,
        disturbance,
        unknown_force,
        hardware,
        x_task,
        x_ser,
        y_ser,
        arduino_task,
        settle_time_s=settle_time_s,
        post_wait_s=post_wait_s,
        jump_abort_m=jump_abort_m,
        abs_displacement_limit_m=abs_displacement_limit_m,
        detect_threshold=detect_threshold,
        unknown_force_start_s=unknown_start_s,
        unknown_force_duration_steps=duration_steps,
        output_path=rollout_path(controller_name, force_n, duration_steps, repeat),
    )


def rollout_metrics(
    path: Path,
    *,
    force_n: float | None = None,
    duration_steps: int | None = None,
    start_s: float | None = None,
) -> dict:
    rows = read_csv_rows(path)
    first = rows[0]
    time_s = SUPPORT.num_col(rows, "time", 0.0)
    event_start_s = float(
        start_s
        if start_s is not None
        else first.get("unknown_force_start_s", first.get("unknown_start_s", UNKNOWN_START_S)) or UNKNOWN_START_S
    )
    duration_steps_value = int(
        duration_steps
        if duration_steps is not None
        else float(first.get("unknown_force_duration_steps", first.get("unknown_duration_steps", 0)) or 0)
    )
    force_value = force_n
    if force_value is None:
        requested = SUPPORT.num_col(rows, "F_ud_requested", 0.0)
        active_requested = requested[np.abs(requested) > 1.0e-9]
        force_value = float(active_requested[0]) if active_requested.size else 0.0
    event_end_s = event_start_s + float(duration_steps_value) * DT + float(DISTURBANCE_METRIC_POST_S)
    event_mask = (time_s >= event_start_s) & (time_s <= event_end_s)
    if not np.any(event_mask):
        event_mask = np.ones_like(time_s, dtype=bool)
    out = {
        "case_id": first.get("case_id", path.parent.name if path.parent != ROLLOUT_DIR else case_id(float(force_value), duration_steps_value)),
        "controller": first["controller"],
        "controller_label": first.get("controller_label", CONTROLLER_LABELS.get(first["controller"], first["controller"])),
        "repeat": int(float(first["repeat"])),
        "unknown_force_n": float(force_value),
        "unknown_start_s": event_start_s,
        "unknown_duration_steps": duration_steps_value,
        "event_window_start_s": event_start_s,
        "event_window_end_s": event_end_s,
        "path": str(path),
    }
    for key in ("x2", "v2", "a2"):
        arr = SUPPORT.num_col(rows, key, 0.0)
        out[f"{key}_rms"] = float(np.sqrt(np.mean(arr**2)))
        out[f"{key}_mean_abs"] = float(np.mean(np.abs(arr)))
        out[f"{key}_peak_abs"] = float(np.max(np.abs(arr)))
        event_arr = arr[event_mask]
        out[f"{key}_event_rms"] = float(np.sqrt(np.mean(event_arr**2)))
        out[f"{key}_event_mean_abs"] = float(np.mean(np.abs(event_arr)))
        out[f"{key}_event_peak_abs"] = float(np.max(np.abs(event_arr)))
    compute = SUPPORT.num_col(rows, "computation_time_ms", 0.0)
    out["compute_time_mean_ms"] = float(np.mean(compute))
    out["compute_time_std_ms"] = float(np.std(compute, ddof=1)) if len(compute) > 1 else 0.0
    out["compute_time_max_ms"] = float(np.max(compute))
    out["saturation_count"] = int(np.sum(SUPPORT.num_col(rows, "saturation_flag", 0.0) > 0.5))
    out["fallback_count"] = int(np.sum(SUPPORT.num_col(rows, "fallback_flag", 0.0) > 0.5))
    out["detection_count"] = int(np.sum(SUPPORT.num_col(rows, "detection_flag", 0.0) > 0.5))
    out["event_trigger_count"] = int(np.sum(SUPPORT.num_col(rows, "event_trigger_flag", 0.0) > 0.5))
    out["compensation_active_count"] = int(np.sum(SUPPORT.num_col(rows, "compensation_active", 0.0) > 0.5))
    out["dgs_adrc_active_count"] = int(np.sum(SUPPORT.num_col(rows, "dgs_fallback_active", 0.0) > 0.5))
    out["dgs_switch_to_adrc_count"] = int(np.sum(SUPPORT.num_col(rows, "dgs_switch_to_adrc_flag", 0.0) > 0.5))
    out["dgs_fallback_active_count"] = out["dgs_adrc_active_count"]
    out["dgs_switch_to_fallback_count"] = out["dgs_switch_to_adrc_count"]
    out["dgs_switch_to_arppo_count"] = int(np.sum(SUPPORT.num_col(rows, "dgs_switch_to_arppo_flag", 0.0) > 0.5))
    true_unknown = SUPPORT.num_col(rows, "F_ud_true", np.nan)
    est_unknown = SUPPORT.num_col(rows, "F_ud_est", np.nan)
    valid = np.isfinite(true_unknown) & np.isfinite(est_unknown)
    out["mean_abs_Fud_error"] = float(np.mean(np.abs(est_unknown[valid] - true_unknown[valid]))) if np.any(valid) else np.nan
    return out


def refresh_metric_tables(
    controllers: list[str],
    forces: list[float],
    durations: list[int],
    repeats: int,
    *,
    start_s: float = UNKNOWN_START_S,
) -> None:
    repeat_rows = []
    for force in forces:
        for duration in durations:
            for controller in controllers:
                for repeat in range(1, int(repeats) + 1):
                    path = rollout_path(controller, force, duration, repeat)
                    if valid_rollout(path, force_n=force, duration_steps=duration, start_s=start_s):
                        repeat_rows.append(rollout_metrics(path, force_n=force, duration_steps=duration, start_s=start_s))
    repeat_fields = [
        "case_id",
        "controller",
        "controller_label",
        "repeat",
        "unknown_force_n",
        "unknown_start_s",
        "unknown_duration_steps",
        "event_window_start_s",
        "event_window_end_s",
        "path",
        "x2_rms",
        "x2_mean_abs",
        "x2_peak_abs",
        "x2_event_rms",
        "x2_event_mean_abs",
        "x2_event_peak_abs",
        "v2_rms",
        "v2_mean_abs",
        "v2_peak_abs",
        "v2_event_rms",
        "v2_event_mean_abs",
        "v2_event_peak_abs",
        "a2_rms",
        "a2_mean_abs",
        "a2_peak_abs",
        "a2_event_rms",
        "a2_event_mean_abs",
        "a2_event_peak_abs",
        "compute_time_mean_ms",
        "compute_time_std_ms",
        "compute_time_max_ms",
        "saturation_count",
        "fallback_count",
        "detection_count",
        "event_trigger_count",
        "compensation_active_count",
        "dgs_adrc_active_count",
        "dgs_switch_to_adrc_count",
        "dgs_fallback_active_count",
        "dgs_switch_to_fallback_count",
        "dgs_switch_to_arppo_count",
        "mean_abs_Fud_error",
    ]
    write_csv(DATA_DIR / "repeat_metrics.csv", repeat_rows, repeat_fields)

    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in repeat_rows:
        grouped.setdefault((row["case_id"], row["controller"]), []).append(row)
    aggregate_rows = []
    metrics = [
        f
        for f in repeat_fields
        if f
        not in {
            "case_id",
            "controller",
            "controller_label",
            "repeat",
            "unknown_force_n",
            "unknown_start_s",
            "unknown_duration_steps",
            "path",
        }
    ]
    for (cid, controller), group in sorted(grouped.items()):
        first = group[0]
        out = {
            "case_id": cid,
            "controller": controller,
            "controller_label": first["controller_label"],
            "unknown_force_n": first["unknown_force_n"],
            "unknown_start_s": first["unknown_start_s"],
            "unknown_duration_steps": first["unknown_duration_steps"],
            "n_repeats": len(group),
        }
        for metric in metrics:
            values = np.asarray([float(r[metric]) for r in group if r[metric] == r[metric]], dtype=np.float64)
            out[f"{metric}_mean"] = float(np.mean(values)) if values.size else np.nan
            out[f"{metric}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        aggregate_rows.append(out)
    aggregate_fields = []
    for row in aggregate_rows:
        for key in row:
            if key not in aggregate_fields:
                aggregate_fields.append(key)
    write_csv(DATA_DIR / "aggregate_statistics.csv", aggregate_rows, aggregate_fields)


def collect_all(args: argparse.Namespace) -> None:
    forces = [float(v) for v in args.forces]
    durations = [int(v) for v in args.durations]
    controllers = [str(c) for c in args.controllers]
    save_sweep_plan(forces, durations, args.unknown_start_s)
    _, disturbance = SUPPORT.load_fixed_disturbance()
    write_csv(
        DATA_DIR / "fixed_test_disturbance.csv",
        [{"sample": i, "time": i * DT, "shaker_force_n": float(v)} for i, v in enumerate(disturbance)],
        ["sample", "time", "shaker_force_n"],
    )
    best_params = SUPPORT.CONTROL_SUPPORT.load_best_params()
    dynamics = SUPPORT.DDMDynamics() if any(c in {"ar_ppo_est"} or c in SUPPORT.DGS_CONTROLLER_NAMES for c in controllers) else None
    controller_cache = {name: make_controller(name, best_params, dynamics) for name in controllers}

    total_jobs = len(forces) * len(durations) * len(controllers) * int(args.repeats)
    completed = 0
    for force in forces:
        for duration in durations:
            for controller in controllers:
                for repeat in range(1, int(args.repeats) + 1):
                    completed += int(
                        valid_rollout(
                            rollout_path(controller, force, duration, repeat),
                            len(disturbance),
                            force_n=force,
                            duration_steps=duration,
                            start_s=args.unknown_start_s,
                        )
                    )
    print(f"Sweep plan: {DATA_DIR / 'sweep_plan.csv'}")
    print(f"Completed rollouts: {completed}/{total_jobs}")

    hardware = SUPPORT.import_hardware_module()
    x_task = x_ser = y_ser = arduino_task = None
    try:
        x_task, x_ser, y_ser, arduino_task = hardware.all_devices_setup_on_demand()
        SUPPORT.stop_outputs(x_task, arduino_task)
        job_index = 0
        for force in forces:
            for duration in durations:
                time_s, unknown_force, _unknown_meta = generate_unknown_pulse(
                    force,
                    duration,
                    start_s=args.unknown_start_s,
                    steps=len(disturbance),
                )
                save_case_unknown_force(force, duration, args.unknown_start_s, time_s, unknown_force)
                for controller_name in controllers:
                    controller = controller_cache[controller_name]
                    for repeat in range(1, int(args.repeats) + 1):
                        job_index += 1
                        path = rollout_path(controller_name, force, duration, repeat)
                        if (
                            valid_rollout(
                                path,
                                len(disturbance),
                                force_n=force,
                                duration_steps=duration,
                                start_s=args.unknown_start_s,
                            )
                            and not args.overwrite
                        ):
                            print(
                                f"Skipping {case_id(force, duration)} {controller_name} "
                                f"repeat {repeat}/{args.repeats}: already saved"
                            )
                            continue
                        attempt = 0
                        while True:
                            attempt += 1
                            print(
                                f"Collecting {case_id(force, duration)} force={force:g}N duration={duration} steps "
                                f"{CONTROLLER_LABELS.get(controller_name, controller_name)} "
                                f"repeat {repeat}/{args.repeats} "
                                f"(job {job_index}/{total_jobs}) attempt={attempt}"
                            )
                            try:
                                saved = collect_rollout(
                                    controller_name,
                                    repeat,
                                    force,
                                    duration,
                                    args.unknown_start_s,
                                    controller,
                                    dynamics if controller_name in {"ar_ppo_est"} or controller_name in SUPPORT.DGS_CONTROLLER_NAMES else None,
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
                                )
                                print(f"Saved: {saved.relative_to(SCRIPT_DIR)}")
                                refresh_metric_tables(controllers, forces, durations, args.repeats, start_s=args.unknown_start_s)
                                break
                            except SUPPORT.RolloutAbort as exc:
                                print(f"Restarting {case_id(force, duration)} {controller_name} repeat {repeat}: {exc}")
                                if attempt >= int(args.max_retries):
                                    raise RuntimeError(
                                        f"{case_id(force, duration)} {controller_name} repeat {repeat} "
                                        f"exceeded max_retries={args.max_retries}"
                                    ) from exc
    finally:
        SUPPORT.close_devices(x_task, x_ser, y_ser, arduino_task)
    refresh_metric_tables(controllers, forces, durations, args.repeats, start_s=args.unknown_start_s)
    print(f"Saved repeat metrics: {DATA_DIR / 'repeat_metrics.csv'}")
    print(f"Saved aggregate statistics: {DATA_DIR / 'aggregate_statistics.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hardware sweep under structured pulse actuator-side disturbances.")
    parser.add_argument("--controllers", nargs="+", default=DEFAULT_CONTROLLERS, choices=AVAILABLE_CONTROLLERS)
    parser.add_argument("--forces", type=float, nargs="+", default=DEFAULT_TEST_FORCES_N)
    parser.add_argument("--durations", type=int, nargs="+", default=DEFAULT_TEST_DURATION_STEPS)
    parser.add_argument("--unknown-start-s", type=float, default=UNKNOWN_START_S)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-retries", type=int, default=10)
    parser.add_argument("--settle-time-s", type=float, default=2.0)
    parser.add_argument("--post-wait-s", type=float, default=2.0)
    parser.add_argument("--jump-abort-m", type=float, default=0.008)
    parser.add_argument("--abs-displacement-limit-m", type=float, default=0.04)
    parser.add_argument("--detect-threshold", type=float, default=SUPPORT.DETECT_THRESHOLD)
    return parser.parse_args()


if __name__ == "__main__":
    collect_all(parse_args())
