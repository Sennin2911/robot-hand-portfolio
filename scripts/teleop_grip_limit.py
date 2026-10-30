"""【概要】リーダーアームでフォロワーを遠隔操作する(lerobot-teleoperate と同じ動き)。
ただし、フォロワーのグリッパー(指)が「閉じる側」へ行きすぎないように、指令の下限(--grip-min)を設ける。
古典制御で 30 mm の立方体を掴んだときの閉じる指令(plan_pick_test.py の REAL_GRIP_CLOSE = 20.0)を、既定の下限にしてある。
リーダーのトリガーをそれ以上に握り込んでも、フォロワーの指は、この値より閉じない(立方体や机を強く挟まないための安全装置)。

使い方(LeRobot の環境で実行する。この .venv ではない):
    ~/lerobot-work/lerobot/.venv/bin/python scripts/teleop_grip_limit.py
    # 下限を変える: --grip-min 15  /  ポートを変える: --robot-port ... --leader-port ...
終了は Ctrl+C。始める前に、リーダーの姿勢をフォロワーの今の姿勢に近づけておく(始めた瞬間に、フォロワーがリーダーの姿勢へ動く)。

単位: gripper.pos は 0〜100(キャリブレーションで決めた範囲の百分率)。大きいほど開く。
      古典制御の実測(新しい指): 全開 99.9 / 空で閉じ切り 9.4 / 立方体に触れた読み 21.9 / 閉じる指令 20.0。
"""
import argparse  # コマンドラインの引数(--grip-min など)を読む標準ライブラリ。[頻度] スクリプトを作るとき定番
import time

# ポートは、番号(/dev/ttyACM0 など)だと差し直しで入れ替わるので、固定の名前(by-id)を既定にしている
ROBOT_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B91043920-if00"   # フォロワー
LEADER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B91026562-if00"  # リーダー
GRIPPER_KEY = "gripper.pos"  # LeRobot の指令(辞書)での、グリッパーのキー。他の関節は "shoulder_pan.pos" のような名前


def clamp_gripper(action, grip_min):
    """指令の辞書 action のグリッパーを、grip_min 以上にした新しい辞書を返す。下限に当たったら、2 つ目の値を True にする。"""
    clamped = dict(action)  # dict(...) = 辞書のコピー。元の辞書(リーダーの読み)は変えない
    hit = clamped[GRIPPER_KEY] < grip_min
    if hit:
        clamped[GRIPPER_KEY] = grip_min
    return clamped, hit


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # ArgumentParser = 引数の定義を集める入れ物
    ap.add_argument("--grip-min", type=float, default=20.0,
                    help="フォロワーのグリッパー指令の下限(0〜100)。既定 20.0 = 古典制御で 30 mm の立方体を掴んだ閉じる指令")
    ap.add_argument("--robot-port", default=ROBOT_PORT, help="フォロワーのポート")
    ap.add_argument("--leader-port", default=LEADER_PORT, help="リーダーのポート")
    ap.add_argument("--fps", type=int, default=60, help="1 秒あたりの更新回数(既定 60。lerobot-teleoperate と同じ)")
    args = ap.parse_args()

    # lerobot の部品は、実行するときだけ必要なので、ここで import する(--help だけなら lerobot が無くても動く)
    from lerobot.robots.so_follower import SO101Follower, SOFollowerRobotConfig
    from lerobot.teleoperators.so_leader import SO101Leader, SOLeaderTeleopConfig

    # id は、キャリブレーションの保存名(~/.cache/huggingface/lerobot/calibration/ の中の <id>.json)
    robot = SO101Follower(SOFollowerRobotConfig(port=args.robot_port, id="my_follower_arm"))
    leader = SO101Leader(SOLeaderTeleopConfig(port=args.leader_port, id="my_leader_arm"))
    leader.connect()  # connect() で、保存済みのキャリブレーションを読んでサーボに書く
    robot.connect()  # フォロワーは、ここでトルクが入る(腕が固まる)
    print(f"遠隔操作を始める。グリッパーの指令の下限 = {args.grip_min}(これ以上は閉じない)。終了は Ctrl+C")
    period = 1.0 / args.fps  # 1 回の更新の間隔(秒)
    hits = 0  # 下限に当たった回数(リーダーを握り込みすぎたとき)
    try:
        while True:
            t0 = time.perf_counter()  # perf_counter = 経過時間を測る用の高精度の時計。[頻度] 時間を測るときの定番
            action = leader.get_action()  # リーダーの今の関節の読み。{"shoulder_pan.pos": 値, ..., "gripper.pos": 値}
            action, hit = clamp_gripper(action, args.grip_min)
            hits += hit  # True は 1、False は 0 として足せる(bool は整数の仲間)。[頻度] たまに使う
            robot.send_action(action)  # フォロワーへ指令を送る
            # 次の更新の時刻まで待つ(処理が速すぎて、60 回/秒を超えないようにする)
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))
    except KeyboardInterrupt:  # Ctrl+C で終了したときの処理。[頻度] 無限ループを止める定番
        pass
    finally:
        # try/finally: エラーや Ctrl+C でも必ず実行される。[頻度] 機器を扱うときの定番
        print(f"\n終了。グリッパーが下限 {args.grip_min} に当たった更新は {hits} 回。")
        robot.disconnect()  # フォロワーのトルクを切って、通信を閉じる
        leader.disconnect()


if __name__ == "__main__":  # このファイルを直接実行したときだけ main() を呼ぶ決まり文句。[頻度] 定番
    main()
