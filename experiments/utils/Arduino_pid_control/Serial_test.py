import serial
import time

PORT = "COM13"  # 串口5
BAUDRATE = 2000000  # 与单片机代码一致
HOLD_TIME = 0.03  # 每个挡位持续 1 秒


def send_cmd(ser, ch, value):
    cmd = f"{ch} {value}\n"
    ser.write(cmd.encode("utf-8"))
    print(f"[{time.strftime('%H:%M:%S')}] 发送: {cmd.strip()}")


def main():
    ser = serial.Serial(port=PORT, baudrate=BAUDRATE, timeout=0.1)

    # 某些开发板打开串口后会复位，等一会更稳妥
    time.sleep(2)

    print("开始测试：")
    print("1号口 0 -> 30，每挡 1 秒")
    print("2号口 30 -> 0，每挡 1 秒")
    print("按 Ctrl+C 停止")

    try:
        while True:
            # 1号口从 0 到 30
            for value in range(0, 50):
                send_cmd(ser, 1, value)
                time.sleep(HOLD_TIME)

            # 2号口从 30 到 0
            # for value in range(30, -1, -1):
            #     send_cmd(ser, 2, value)
            #     time.sleep(HOLD_TIME)

    except KeyboardInterrupt:
        print("\n用户停止测试")

    finally:
        # 停止前关闭所有输出
        try:
            send_cmd(ser, 1, 0)
            send_cmd(ser, 2, 0)
            time.sleep(0.1)
        except Exception:
            pass

        ser.close()
        print("串口已关闭")


if __name__ == "__main__":
    main()
