"""
calibrate.py
Output a sine disturbance signal for power amplifier gain calibration.

Usage:
    python calibrate.py
"""

import math
import threading
import time


def main():
    try:
        from Hardware import config_usb_ao_x_task_instant
    except Exception as exc:
        raise RuntimeError(
            "Hardware imports failed. Please run on the machine with NI-DAQ drivers installed."
        ) from exc

    amplitude_v = 2.0
    freq_hz = 5.0
    sample_dt = 0.002

    print("Calibration started.")
    print(
        f"Outputting sine disturbance: amplitude={amplitude_v:.1f} V, frequency={freq_hz:.1f} Hz."
    )
    print("Please manually adjust the power supply to 0.8 V and 0.54 A")
    print("Press Enter to stop output and exit.")

    stop_event = threading.Event()

    def _wait_for_enter():
        input()
        stop_event.set()

    x_task = config_usb_ao_x_task_instant()
    try:
        x_task.start()
        waiter = threading.Thread(target=_wait_for_enter, daemon=True)
        waiter.start()
        t0 = time.perf_counter()
        while not stop_event.is_set():
            t = time.perf_counter() - t0
            v = amplitude_v * math.sin(2.0 * math.pi * freq_hz * t)
            x_task.write(float(v), auto_start=False)
            time.sleep(sample_dt)
    finally:
        try:
            x_task.write(0.0, auto_start=False)
        except Exception:
            pass
        try:
            x_task.stop()
        except Exception:
            pass
        try:
            x_task.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
