"""【概要】実機の制御ボードにつながったモーターを、ID ごとに 1 個ずつ呼びかけて、どれが応答するかを表示する。腕は動かさない(モデル番号を読むだけ)。
どのモーターが応答しないかを調べるための診断用。real_read_joints.py は「6 個そろわないと」エラーで止まるので、1 個ずつ見たいときはこちらを使う。
応答の読み方: 全部が「応答なし」→ モーターの電源・ボードとの配線・通信速度のどれか(共通の原因)。途中の ID から先だけ「応答なし」→ その手前のケーブル・モーターの不良。

この .venv ではなく、LeRobot の環境で実行する:
    ~/lerobot-work/lerobot/.venv/bin/python scripts/real_scan_motors.py [--port /dev/ttyACM0]
"""
import argparse

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

parser = argparse.ArgumentParser()
parser.add_argument("--port", default="/dev/ttyACM0")
parser.add_argument("--max-id", type=int, default=10, help="1 からこの ID まで呼びかける(SO-101 は 1〜6)")
args = parser.parse_args()

NAMES = {1: "shoulder_pan", 2: "shoulder_lift", 3: "elbow_flex", 4: "wrist_flex", 5: "wrist_roll", 6: "gripper"}
# ID 1〜max_id のモーターが「あるつもり」でバスを作る。モーター名は、ID から作った仮の名前(ping は ID で呼びかけるだけで、名前は使わない)
motors = {f"m{i}": Motor(i, "sts3215", MotorNormMode.RANGE_M100_100) for i in range(1, args.max_id + 1)}
bus = FeetechMotorsBus(port=args.port, motors=motors)
bus.connect(handshake=False)  # handshake=False: 「6 個そろっているか」の確認を省く(1 個ずつ調べるため)。通常の接続では使わない
try:
    # 通信速度(bps)は、SO-101 の標準が 1000000。ほかの速度は、モーターの設定が変わっていた場合に見つけるため(通常は 1000000 で見つかる)
    for baud in (1_000_000, 115_200, 57_600, 500_000, 250_000, 128_000):
        bus.set_baudrate(baud)
        found = {i: bus.ping(f"m{i}") for i in range(1, args.max_id + 1)}  # ping: 応答すればモデル番号(STS3215 は 777)、無ければ None
        ok = {i: m for i, m in found.items() if m is not None}
        print(f"{baud:>8} bps: " + (", ".join(f"ID {i} ({NAMES.get(i, '?')}, モデル {m})" for i, m in ok.items()) if ok else "応答なし"))
        if ok and baud == 1_000_000:
            break
finally:
    bus.disconnect(disable_torque=False)  # disable_torque=False: トルクの設定には触れずに閉じる
