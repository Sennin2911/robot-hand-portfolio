"""【概要】実機の SO-101 の、今の関節角を「読むだけ」で表示し、outputs/real_joints.json に保存する。**指令は一切出さない。トルクにも触れない。**
目的: 手で作った姿勢の関節角を取り、同じ姿勢で撮った画像(view_real_board.py)の手先の位置と、sim の順運動学を比べるため。
robot.connect() は使わない(使うとトルクが入り、腕が固まる)。代わりに、バス(モーターとの通信路)だけを開いて、現在位置(Present_Position)を読む。
終了のときも、トルクを書き換えない(disconnect(disable_torque=False))。

この .venv ではなく、LeRobot の環境で実行する:
    ~/lerobot-work/lerobot/.venv/bin/python scripts/real_read_joints.py
単位: 腕の 5 関節は「度」(キャリブレーションの範囲の中央が 0)、gripper は 0〜100。OFFSET_DEG の補正は、まだ引いていない生の値。
"""
import json
import time
from pathlib import Path

from lerobot.robots.so_follower import SO101Follower, SOFollowerRobotConfig

OUT = Path(__file__).resolve().parents[1] / "outputs" / "real_joints.json"  # Path(__file__).resolve().parents[1] = このファイルの 1 つ上のフォルダ(プロジェクトの根)。[頻度] 定番

robot = SO101Follower(SOFollowerRobotConfig(port="/dev/ttyACM0", id="my_follower_arm"))  # 設定を作るだけ。まだ接続しない。キャリブレーションは id から読まれる
robot.bus.connect()  # バス(通信路)だけを開く。robot.connect() と違い、トルクを入れない
try:
    pos = robot.bus.sync_read("Present_Position")  # 全モーターの今の位置。{名前: 値}(度、gripper は 0〜100)
finally:
    # try/finally: エラーでも必ず閉じる。[頻度] 機器を扱うときの定番。disable_torque=False = トルクの設定に触れない
    robot.bus.disconnect(disable_torque=False)
for name, value in pos.items():
    print(f"  {name:14s} {value:8.2f}")
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "joints": pos}, ensure_ascii=False, indent=2))
print(f"保存: {OUT}")
