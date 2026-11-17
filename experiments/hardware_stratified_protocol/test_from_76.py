from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import main as protocol
from utils.hardware_io import (
    close_devices,
    import_hardware_module,
    quiet_put_voice_coil_force,
    stop_outputs,
)
from utils.signals import DT, STEPS, VOICE_FORCE_LIMIT_N, load_or_create_plan


TEST_DATA_DIR = SCRIPT_DIR / "data" / "test_from_76"
TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)


def test_trajectory_path(item: dict) -> Path:
    return TEST_DATA_DIR / f"trajectory_{int(item['index']):03d}.csv"


def install_debug_hooks(print_first: int = 30) -> None:
    original_trajectory_path = protocol.trajectory_path

    def debug_trajectory_path(item: dict) -> Path:
        return test_trajectory_path(item)

    def debug_put_voice_coil_force(arduino_task, force_n: float) -> float:
        force_n = float(max(-VOICE_FORCE_LIMIT_N, min(VOICE_FORCE_LIMIT_N, force_n)))
        ch = 1 if force_n >= 0.0 else 2
        code = int(round((abs(force_n) / 5.0) * 30))
        code = int(max(0, min(30, code)))
        count = getattr(debug_put_voice_coil_force, "count", 0)
        if count < int(print_first):
            print(f"voice_cmd[{count:03d}]: force={force_n:+.4f} N -> {ch} {code}")
        setattr(debug_put_voice_coil_force, "count", count + 1)
        return quiet_put_voice_coil_force(arduino_task, force_n)

    protocol.trajectory_path = debug_trajectory_path
    protocol.quiet_put_voice_coil_force = debug_put_voice_coil_force
    protocol._original_trajectory_path_for_test = original_trajectory_path


def print_item_preview(item: dict) -> None:
    shaker_force, shaker_voltage, voice_force = protocol.generate_inputs(item)
    print(
        f"Selected trajectory {int(item['index'])}/{int(item['total'])}: "
        f"block={item['block']} shaker={item['shaker_type']} voice={item['voice_type']} "
        f"direction={item.get('chirp_direction', 'none')}"
    )
    print(
        f"steps={STEPS}, dt={DT}, "
        f"voice_force=[{float(voice_force.min()):+.4f}, {float(voice_force.max()):+.4f}] N, "
        f"shaker_voltage=[{float(shaker_voltage.min()):+.4f}, {float(shaker_voltage.max()):+.4f}] V"
    )
    preview = []
    for force in voice_force[:12]:
        ch = 1 if float(force) >= 0.0 else 2
        code = int(round((abs(float(force)) / 5.0) * 30))
        preview.append(f"{ch} {code}")
    print(f"first 12 voice commands: {', '.join(preview)}")


def run_test(start_index: int = 76, count: int = 1, print_first: int = 30) -> None:
    plan = load_or_create_plan(protocol.PLAN_PATH)
    selected = [item for item in plan if int(item["index"]) >= int(start_index)]
    selected = selected[: int(count)]
    if not selected:
        raise ValueError(f"No protocol items found at or after index {start_index}.")

    install_debug_hooks(print_first=print_first)
    print(f"Test data will be saved under: {TEST_DATA_DIR.resolve()}")

    hardware = import_hardware_module()
    x_task = x_ser = y_ser = arduino_task = None
    try:
        x_task, x_ser, y_ser, arduino_task = hardware.all_devices_setup_on_demand()
        stop_outputs(x_task, arduino_task)
        for item in selected:
            print_item_preview(item)
            saved_path = protocol.collect_one_trajectory(
                item,
                hardware,
                x_task,
                x_ser,
                y_ser,
                arduino_task,
            )
            print(f"Saved test trajectory: {saved_path.resolve()}")
            stop_outputs(x_task, arduino_task)
            time.sleep(protocol.SETTLE_TIME_S)
    finally:
        close_devices(x_task, x_ser, y_ser, arduino_task)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hardware protocol test directly from trajectory 76.")
    parser.add_argument("--start-index", type=int, default=76, help="Protocol trajectory index to start from.")
    parser.add_argument("--count", type=int, default=1, help="Number of trajectories to run.")
    parser.add_argument("--print-first", type=int, default=30, help="Number of voice commands to print.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_test(
        start_index=args.start_index,
        count=args.count,
        print_first=args.print_first,
    )
