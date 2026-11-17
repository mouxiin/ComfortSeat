import nidaqmx
import serial
import sys
import os
import time
from nidaqmx.constants import AcquisitionType, TerminalConfiguration, RegenerationMode
from nidaqmx.stream_writers import AnalogSingleChannelWriter
import numpy as np

sys.path.append(os.path.dirname(os.getcwd()))

# for real environment
CALIBRATION_NUM = 100  # run the laser for this number of times to calibrate
ANALOG_OUTPUT_SAMPLE_RATE = 5000
X_STILL_HEIGHT_485 = -9999  # mm
Y_STILL_HEIGHT_485 = -9999  # mm

# DAQ port configuration
SHAKER_ANALOG_PORT = "Dev1/ao0"
X_SERIAL_PORT = "COM3"  # read x displacement
Y_SERIAL_PORT = "COM4"  # read y displacement
ARDUINO_UNO_PORT = "COM13"  # send control signal to arduino, which then sends PWM to the power supply
SERIAL485_BRUDRATE = 1250000
ARDUINO_SERIAL_BRUDRATE = 2000000
SERIAL_GET_SIGNAL = b"\x01\x04\x00\x64\x00\x01\x70\x15"

# X parameters
MAX_STEP = 800
DT = 0.03
MAX_PERIOD = MAX_STEP * DT
ACTION_LIMIT = 5

# --------------- Device connection ---------------
# NI6003 analog output to control disturbance.
# 485 serial to read x/y displacement.
# Arduino Uno to send PWM signal to the power supply.


# disturbance output
def config_usb_ao_x_task_preinput(x_ls=[0.0] * MAX_STEP):
    x_task = nidaqmx.Task()
    voltages = np.array(x_ls)
    x_task.ao_channels.add_ao_voltage_chan(SHAKER_ANALOG_PORT,
                                           min_val=-10.0,
                                           max_val=10.0)
    x_task.timing.cfg_samp_clk_timing(
        rate=ANALOG_OUTPUT_SAMPLE_RATE,
        sample_mode=AcquisitionType.CONTINUOUS,
        samps_per_chan=len(voltages),
    )
    x_task.out_stream.regen_mode = RegenerationMode.ALLOW_REGENERATION
    x_writer = AnalogSingleChannelWriter(x_task.out_stream)
    x_writer.write_many_sample(voltages)
    return x_task, x_writer


def config_usb_ao_x_task_instant():
    """
    Configure AO task for on-demand single-sample writes.
    This allows disturbance to be updated each control cycle in software.
    """
    x_task = nidaqmx.Task()
    x_task.ao_channels.add_ao_voltage_chan(SHAKER_ANALOG_PORT,
                                           min_val=-10.0,
                                           max_val=10.0)
    return x_task


# x, y measurement portal config
def config_serial_di_xy_task():
    x_ser = serial.Serial(
        port=X_SERIAL_PORT,
        baudrate=SERIAL485_BRUDRATE,
        timeout=1,
    )
    y_ser = serial.Serial(
        port=Y_SERIAL_PORT,
        baudrate=SERIAL485_BRUDRATE,
        timeout=1,
    )
    x_ser.flushInput()
    y_ser.flushInput()

    xs, ys = [], []
    for _ in range(CALIBRATION_NUM):
        x, y = communicate_xy_serial(x_ser, y_ser)
        xs.append(x)
        ys.append(y)

    global X_STILL_HEIGHT_485, Y_STILL_HEIGHT_485
    X_STILL_HEIGHT_485 = np.mean(xs)
    Y_STILL_HEIGHT_485 = np.mean(ys)
    print(f"X_STILL_HEIGHT_485: {X_STILL_HEIGHT_485}", )
    print(f"Y_STILL_HEIGHT_485: {Y_STILL_HEIGHT_485}")

    return x_ser, y_ser


# control signal output
def config_serial_arduino_task():
    arduino_task = serial.Serial(
        port=ARDUINO_UNO_PORT,
        baudrate=ARDUINO_SERIAL_BRUDRATE,
        timeout=1,
    )
    time.sleep(2.0)
    try:
        arduino_task.reset_input_buffer()
        arduino_task.reset_output_buffer()
    except Exception:
        arduino_task.flushInput()
    return arduino_task


def all_devices_setup(x_ls):
    x_task, ao_writer = config_usb_ao_x_task_preinput(x_ls)
    x_ser, y_ser = config_serial_di_xy_task()
    arduino_task = config_serial_arduino_task()

    return x_task, ao_writer, x_ser, y_ser, arduino_task


def all_devices_setup_on_demand():
    """
    Setup devices with AO configured for on-demand writes.
    """
    x_task = config_usb_ao_x_task_instant()
    x_ser, y_ser = config_serial_di_xy_task()
    arduino_task = config_serial_arduino_task()

    return x_task, x_ser, y_ser, arduino_task


def communicate_xy_serial(x_ser, y_ser):

    def raw_to_height(raw):
        third_byte = raw[3]
        fourth_byte = raw[4]

        value = (third_byte << 8) | fourth_byte
        if value & 0x8000:
            value = value - 0x10000
        return value

    x_ser.write(SERIAL_GET_SIGNAL)
    y_ser.write(SERIAL_GET_SIGNAL)

    x_raw = x_ser.read(7)
    y_raw = y_ser.read(7)

    if len(x_raw) != 7 or len(y_raw) != 7:
        raise Exception("Serial communication error")
    else:
        x = raw_to_height(x_raw)
        y = raw_to_height(y_raw)
        return x, y


def get_xy_by_serial(x_ser, y_ser):

    x, y = communicate_xy_serial(x_ser, y_ser)
    x = (X_STILL_HEIGHT_485 - x) / 100000  # convert to m
    y = (Y_STILL_HEIGHT_485 - y) / 100000
    return x, y


def put_pwm_by_arduino_serial(arduino_task, act_value):
    """
    Send PWM command to Arduino over a serial-like object.

    Params:
        arduino_task: a pyserial Serial instance (opened), or any object with write(bytes) method
        act_value: float in [-3.5, 3.5]
    """
    try:
        v = float(act_value)
    except (TypeError, ValueError):
        raise ValueError

    if v > 0:
        ch = 1
        mag = v
    elif v < 0:
        ch = 2
        mag = -v
    else:
        ch = 1
        mag = 0.0

    code = int(round((mag / ACTION_LIMIT) * 30))
    if code < 0:
        code = 0
    elif code > 30:
        code = 30

    line = f"{ch} {code}\n"
    print(f"Sending to Arduino: {line.strip()}")
    arduino_task.write(line.encode("ascii"))


def test_task_y(params):
    xy_task, x_ser, y_ser = params
    _, y = get_xy_by_serial(x_ser, y_ser)
    # _, ys = get_xys_by_usb(xy_task)
    global ys_serial, ys_ls_usb, ts
    ts.append(time.perf_counter_ns() / 1e9)  # seconds
    # ys_ls_usb.append(ys)
    ys_serial.append(y)


def test_pwm(step=0.1, period_s=0.003):
    """
    Sweep act_value from -3.5 to +3.5 and back, updating every 3 ms.

    Params:
        step: increment per update (default 0.1). Adjust for desired resolution.
        period_s: update period in seconds (default 0.003 = 3 ms)
    """
    arduino_task = config_serial_arduino_task()
    lo, hi = -3.5, 3.5
    v = lo
    direction = +1

    try:
        while True:
            t0 = time.perf_counter()

            put_pwm_by_arduino_serial(arduino_task, v)

            v += direction * step
            if v >= hi:
                v = hi
                direction = -1
            elif v <= lo:
                v = lo
                direction = +1

            elapsed = time.perf_counter() - t0
            remaining = period_s - elapsed
            if remaining > 0:
                time.sleep(remaining)
            else:
                pass
    except KeyboardInterrupt:
        try:
            put_pwm_by_arduino_serial(arduino_task, 0.0)
        except Exception:
            pass


# Timer part
import multiprocessing
import signal

DT = 0.03


def millisecond_delay_s(delay_s):
    delay_s *= 1e9
    start_time = time.perf_counter_ns()
    while True:
        current_time = time.perf_counter_ns()
        elapsed_time = current_time - start_time
        if elapsed_time >= delay_s:
            break


def timer_process(pipe, delay_s=DT):
    # Ignore Ctrl+C
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    while True:
        millisecond_delay_s(delay_s)
        pipe.send(0)


def testing_loop_task(params):
    print("hello")
    millisecond_delay_s(0.00)
    print("hi")


def testing_recall_task(params):
    print("bye")


def timer_interrupt(
    loop_func,
    loop_params=None,
    start_func=None,
    start_params=None,
    callback_func=None,
    callback_params=None,
    times=-1,
    delay_s=DT,
):
    parent_conn, child_conn = multiprocessing.Pipe()
    p = multiprocessing.Process(target=timer_process,
                                args=(child_conn, delay_s))
    p.start()

    start_flag = False

    print("Task started")
    t0 = time.perf_counter_ns()
    if times < 0:
        try:
            while True:
                _ = parent_conn.recv()
                if not start_flag:
                    if start_func is not None:
                        start_func(start_params)
                    start_flag = True
                # print(time.perf_counter_ns() - t0)
                # t0 = time.perf_counter_ns()
                loop_func(loop_params)
        except KeyboardInterrupt:
            print("Task stopped by Ctrl+C")
        finally:
            if callback_func is not None:
                callback_func(callback_params)
    else:
        try:
            for _ in range(times):
                _ = parent_conn.recv()
                if not start_flag:
                    if start_func is not None:
                        start_func(start_params)
                    start_flag = True
                loop_func(loop_params)
        except KeyboardInterrupt:
            print("Task stopped by Ctrl+C")
        finally:
            if callback_func is not None:
                callback_func(callback_params)

    p.terminate()


if __name__ == "__main__":
    # get_y()
    # get_y_with_action()
    test_pwm()
