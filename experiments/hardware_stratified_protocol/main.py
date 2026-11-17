from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils.filters import OnlineKinematicsFilter
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
from utils.signals import (
    DT,
    SHAKER_FORCE_LIMIT_N,
    STEPS,
    VOICE_FORCE_LIMIT_N,
    build_channel_signal,
    load_or_create_plan,
)


DATA_DIR = SCRIPT_DIR / "data"
FIGURES_DIR = SCRIPT_DIR / "figures"
PLAN_PATH = DATA_DIR / "protocol_plan.csv"
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

JUMP_ABORT_M = 0.008
SETTLE_TIME_S = 2.0
POST_WAIT_S = 2.0
CUTOFF_HZ = 8.0
VOICE_WRITE_DELAY_S = 0.0
DEBUG_LOG_PATH = DATA_DIR / "hardware_runtime_debug.log"


class TrajectoryAbort(RuntimeError):
    pass


def append_debug(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")


class DebugSerialWriter:
    def __init__(self, wrapped, max_logs: int = 180):
        self._wrapped = wrapped
        self._max_logs = int(max_logs)
        self._count = 0

    def write(self, data):
        if self._count < self._max_logs:
            append_debug(f"serial_write[{self._count:03d}] {data!r}")
        self._count += 1
        return self._wrapped.write(data)

    def flush(self):
        return self._wrapped.flush()

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


def trajectory_path(item: dict) -> Path:
    return DATA_DIR / f"trajectory_{int(item['index']):03d}.csv"


def valid_saved_trajectory(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return len(rows) == STEPS and "time" in rows[0] and int(float(rows[0]["index"])) > 0
    except Exception:
        return False


def generate_inputs(item: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(item["seed"]))
    direction = str(item.get("chirp_direction", "up"))
    if direction == "none":
        direction = "up"
    shaker_force = build_channel_signal(
        rng,
        str(item["shaker_type"]),
        STEPS,
        DT,
        SHAKER_FORCE_LIMIT_N,
        chirp_direction=direction,
        allow_prbs=False,
        smooth_edges=True,
    )
    voice_force = build_channel_signal(
        rng,
        str(item["voice_type"]),
        STEPS,
        DT,
        VOICE_FORCE_LIMIT_N,
        chirp_direction=direction,
        allow_prbs=True,
    )
    shaker_voltage = np.array([shaker_force_to_input_voltage(force) for force in shaker_force], dtype=np.float32)
    return shaker_force.astype(np.float32), shaker_voltage, voice_force.astype(np.float32)


def save_trajectory(path: Path, item: dict, rows: list[dict], shaker_force, shaker_voltage, voice_force) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "index",
        "total",
        "block",
        "purpose",
        "shaker_type",
        "voice_type",
        "chirp_direction",
        "sample",
        "duration_s",
        "dt_nominal",
        "steps",
        "seed",
        "cutoff_hz",
        "time",
        "timestamp_abs",
        "dt_sample",
        "x1_raw",
        "x2_raw",
        "x1_filtered",
        "v1_filtered",
        "a1_filtered",
        "x2_filtered",
        "v2_filtered",
        "a2_filtered",
        "shaker_force_n",
        "shaker_voltage_v",
        "voice_force_n",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for sample, row in enumerate(rows):
            payload = {
                "index": int(item["index"]),
                "total": int(item["total"]),
                "block": str(item["block"]),
                "purpose": str(item["purpose"]),
                "shaker_type": str(item["shaker_type"]),
                "voice_type": str(item["voice_type"]),
                "chirp_direction": str(item.get("chirp_direction", "none")),
                "sample": sample,
                "duration_s": float(item["duration_s"]),
                "dt_nominal": float(item["dt"]),
                "steps": int(item["steps"]),
                "seed": int(item["seed"]),
                "cutoff_hz": CUTOFF_HZ,
            }
            payload.update(row)
            writer.writerow(payload)


def collect_one_trajectory(item: dict, hardware, x_task, x_ser, y_ser, arduino_task) -> Path:
    shaker_force, shaker_voltage, voice_force = generate_inputs(item)
    path = trajectory_path(item)
    debug_this = int(item["index"]) == 76
    if debug_this:
        append_debug(
            f"start trajectory {int(item['index'])}: "
            f"block={item['block']} shaker={item['shaker_type']} voice={item['voice_type']}"
        )
        arduino_task = DebugSerialWriter(arduino_task)

    prepare_hardware_for_trial(x_task, x_ser, y_ser, arduino_task)
    offset = wait_until_still(
        hardware.get_xy_by_serial,
        x_ser,
        y_ser,
        duration_s=SETTLE_TIME_S,
        sample_dt=DT,
    )

    filt = OnlineKinematicsFilter(cutoff_hz=CUTOFF_HZ, dimensions=2)
    rows: list[dict] = []
    prev_raw = None

    try:
        x_task.start()
    except Exception:
        pass

    start_time = time.perf_counter()
    filt.reset(np.zeros(2, dtype=np.float64), start_time)

    try:
        for step in range(STEPS):
            target = start_time + step * DT
            while True:
                remaining = target - time.perf_counter()
                if remaining <= 0.0:
                    break
                time.sleep(min(remaining, 0.002))

            now = time.perf_counter()
            raw_xy = np.asarray(hardware.get_xy_by_serial(x_ser, y_ser), dtype=np.float64) - offset
            if prev_raw is not None and np.any(np.abs(raw_xy - prev_raw) > JUMP_ABORT_M):
                if debug_this:
                    append_debug(
                        f"abort trajectory {int(item['index'])} at step={step}: "
                        f"jump={(raw_xy - prev_raw).tolist()}"
                    )
                raise TrajectoryAbort(f"sudden displacement jump at sample {step}: {raw_xy - prev_raw}")

            x_f, v_f, a_f, dt_sample = filt.update(raw_xy, now)
            sh_v = float(shaker_voltage[step])
            sh_f = float(shaker_force[step])
            vc_f = float(voice_force[step])

            x_task.write(sh_v, auto_start=False)
            if VOICE_WRITE_DELAY_S > 0.0:
                time.sleep(VOICE_WRITE_DELAY_S)
            if debug_this and step < 60:
                ch = 1 if vc_f >= 0.0 else 2
                code = int(round((abs(vc_f) / 5.0) * 30))
                code = int(np.clip(code, 0, 30))
                append_debug(
                    f"step={step:03d} sh_v={sh_v:+.4f} voice={vc_f:+.4f} "
                    f"cmd={ch} {code} raw_xy={raw_xy.tolist()}"
                )
            quiet_put_voice_coil_force(arduino_task, vc_f)

            rows.append(
                {
                    "time": float(now - start_time),
                    "timestamp_abs": float(now),
                    "dt_sample": float(dt_sample),
                    "x1_raw": float(raw_xy[0]),
                    "x2_raw": float(raw_xy[1]),
                    "x1_filtered": float(x_f[0]),
                    "v1_filtered": float(v_f[0]),
                    "a1_filtered": float(a_f[0]),
                    "x2_filtered": float(x_f[1]),
                    "v2_filtered": float(v_f[1]),
                    "a2_filtered": float(a_f[1]),
                    "shaker_force_n": sh_f,
                    "shaker_voltage_v": sh_v,
                    "voice_force_n": vc_f,
                }
            )
            prev_raw = raw_xy.copy()
    finally:
        finish_hardware_trial(x_task, x_ser, y_ser, arduino_task, post_wait_s=POST_WAIT_S)
        try:
            x_task.stop()
        except Exception:
            pass

    if len(rows) != STEPS:
        raise RuntimeError(f"Expected {STEPS} rows, collected {len(rows)} rows.")

    save_trajectory(path, item, rows, shaker_force, shaker_voltage, voice_force)
    if debug_this:
        append_debug(f"saved trajectory {int(item['index'])}: {path.resolve()}")
    return path


def collect_all(max_retries: int = 20, seed: int = 20260516) -> None:
    plan = load_or_create_plan(PLAN_PATH, seed=seed)
    total = len(plan)
    completed = sum(1 for item in plan if valid_saved_trajectory(trajectory_path(item)))
    print(f"Protocol plan: {PLAN_PATH.resolve()}")
    print(f"Completed trajectories: {completed}/{total}")

    hardware = import_hardware_module()
    x_task = x_ser = y_ser = arduino_task = None
    reset_before_first_collect = False
    try:
        x_task, x_ser, y_ser, arduino_task = hardware.all_devices_setup_on_demand()
        stop_outputs(x_task, arduino_task)
        for item in plan:
            idx = int(item["index"])
            path = trajectory_path(item)
            if valid_saved_trajectory(path):
                print(f"Skipping trajectory {idx}/{total}: already saved")
                continue

            if not reset_before_first_collect:
                print(f"Resetting hardware before collecting trajectory {idx}/{total}...")
                append_debug(f"reset hardware before first pending trajectory {idx}")
                close_devices(x_task, x_ser, y_ser, arduino_task)
                time.sleep(2.0)
                x_task, x_ser, y_ser, arduino_task = hardware.all_devices_setup_on_demand()
                stop_outputs(x_task, arduino_task)
                time.sleep(0.5)
                reset_before_first_collect = True

            retries = 0
            while True:
                print(
                    f"Collecting trajectory {idx}/{total} "
                    f"{item['block']} shaker={item['shaker_type']} voice={item['voice_type']} "
                    f"attempt={retries + 1}"
                )
                try:
                    saved_path = collect_one_trajectory(item, hardware, x_task, x_ser, y_ser, arduino_task)
                    print(f"Saved trajectory {idx}/{total}: {saved_path.name}")
                    break
                except TrajectoryAbort as exc:
                    retries += 1
                    print(f"Restarting trajectory {idx}/{total}: {exc}")
                    if retries >= int(max_retries):
                        raise RuntimeError(f"Trajectory {idx} exceeded max_retries={max_retries}") from exc
    finally:
        close_devices(x_task, x_ser, y_ser, arduino_task)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect B1/B2/B3 stratified comfort-seat hardware data.")
    parser.add_argument("--max-retries", type=int, default=20, help="Maximum restarts for one trajectory after jump aborts.")
    parser.add_argument("--seed", type=int, default=20260516, help="Seed used only when creating a new protocol plan.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    collect_all(max_retries=args.max_retries, seed=args.seed)
