from __future__ import annotations

import time
from pathlib import Path

import numpy as np


SHAKER_FORCE_LIMIT_N = 9.0
VOICE_FORCE_LIMIT_N = 2.0
SHAKER_INPUT_MIN_V = -3.0
SHAKER_INPUT_MAX_V = 3.0
SHAKER_SIGNED_VOLTAGE_LIMIT = 3.0
SHAKER_NEWTON_PER_SIGNED_VOLT = 3.0
DEFAULT_PRE_FLUSH_WAIT_S = 0.15
DEFAULT_POST_WAIT_S = 2.0


def shaker_force_to_input_voltage(force_n: float) -> float:
    force_n = float(np.clip(force_n, -SHAKER_FORCE_LIMIT_N, SHAKER_FORCE_LIMIT_N))
    input_voltage = force_n / SHAKER_NEWTON_PER_SIGNED_VOLT
    return float(np.clip(input_voltage, SHAKER_INPUT_MIN_V, SHAKER_INPUT_MAX_V))


def input_voltage_to_shaker_force(input_voltage: float) -> float:
    input_voltage = float(np.clip(input_voltage, SHAKER_INPUT_MIN_V, SHAKER_INPUT_MAX_V))
    return float(SHAKER_NEWTON_PER_SIGNED_VOLT * input_voltage)


def quiet_put_voice_coil_force(arduino_task, force_n: float) -> float:
    force_n = float(np.clip(force_n, -VOICE_FORCE_LIMIT_N, VOICE_FORCE_LIMIT_N))
    if force_n > 0:
        ch = 1
        mag = force_n
    elif force_n < 0:
        ch = 2
        mag = -force_n
    else:
        ch = 1
        mag = 0.0
    code = int(round((mag / 5.0) * 30))
    code = int(np.clip(code, 0, 30))
    other_ch = 2 if ch == 1 else 1
    arduino_task.write(f"{other_ch} 0\n".encode("ascii"))
    arduino_task.write(f"{ch} {code}\n".encode("ascii"))
    try:
        arduino_task.flush()
    except Exception:
        pass
    return force_n


def import_hardware_module():
    import importlib
    import sys

    utils_dir = Path(__file__).resolve().parent
    if str(utils_dir) not in sys.path:
        sys.path.insert(0, str(utils_dir))
    return importlib.import_module("Hardware")


def stop_outputs(x_task=None, arduino_task=None) -> None:
    if arduino_task is not None:
        try:
            quiet_put_voice_coil_force(arduino_task, 0.0)
        except Exception:
            pass
    if x_task is not None:
        try:
            x_task.write(shaker_force_to_input_voltage(0.0), auto_start=False)
        except Exception:
            pass


def clear_serial_buffers(*serials) -> None:
    for ser in serials:
        if ser is None:
            continue
        for method in ("reset_input_buffer", "reset_output_buffer", "flushInput", "flushOutput"):
            try:
                getattr(ser, method)()
            except Exception:
                pass
        try:
            ser.flush()
        except Exception:
            pass


def prepare_hardware_for_trial(
    x_task=None,
    x_ser=None,
    y_ser=None,
    arduino_task=None,
    pre_flush_wait_s: float = DEFAULT_PRE_FLUSH_WAIT_S,
) -> None:
    stop_outputs(x_task, arduino_task)
    clear_serial_buffers(x_ser, y_ser, arduino_task)
    if pre_flush_wait_s > 0.0:
        time.sleep(float(pre_flush_wait_s))


def finish_hardware_trial(
    x_task=None,
    x_ser=None,
    y_ser=None,
    arduino_task=None,
    post_wait_s: float = DEFAULT_POST_WAIT_S,
) -> None:
    stop_outputs(x_task, arduino_task)
    clear_serial_buffers(x_ser, y_ser, arduino_task)
    if post_wait_s > 0.0:
        time.sleep(float(post_wait_s))


def close_devices(x_task=None, x_ser=None, y_ser=None, arduino_task=None) -> None:
    finish_hardware_trial(x_task, x_ser, y_ser, arduino_task, post_wait_s=0.0)
    if x_task is not None:
        for method in ("stop", "close"):
            try:
                getattr(x_task, method)()
            except Exception:
                pass
    for ser in (x_ser, y_ser, arduino_task):
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass


def wait_until_still(
    get_xy,
    x_ser,
    y_ser,
    duration_s: float = 2.0,
    sample_dt: float = 0.03,
    max_span_m: float = 8.0e-4,
) -> np.ndarray:
    samples = []
    count = max(3, int(round(float(duration_s) / float(sample_dt))))
    for _ in range(count):
        samples.append(get_xy(x_ser, y_ser))
        time.sleep(float(sample_dt))
    arr = np.asarray(samples, dtype=np.float64)
    span = np.ptp(arr, axis=0)
    if np.any(span > float(max_span_m)):
        print(f"Warning: settling span is {span}; using mean offset anyway.")
    return np.mean(arr, axis=0)
