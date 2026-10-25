"""実機 SO-101 を、1関節ずつ「+delta → 戻る → -delta → 戻る」と少しだけ動かす確認用スクリプト。

目的: 各関節の動く向き(符号)と、サーボが指令どおりに動くかを、小さな動きで確かめる。
この .venv ではなく、LeRobot の環境 (~/lerobot-work/lerobot の .venv) で実行する。

    cd ~/lerobot-work/lerobot && source .venv/bin/activate
    python <このファイルの絶対パス>                      # 全関節を 5 度ずつ
    python <このファイルの絶対パス> --joint wrist_roll    # 1関節だけ
    python <このファイルの絶対パス> --delta 3 --seconds 2

単位: 腕の5関節は「度」、gripper は 0〜100 (閉じる〜開く側の動き量。LeRobot の決め)。
安全: 動かす前に、腕を手で「関節が端に当たっていない、中ほどの姿勢」にしておく。
      (休めの折りたたみ姿勢では、関節が可動域の端にあり、片側にしか動けない。)
"""
import argparse  # コマンドラインの引数 (--joint など) を読む標準ライブラリ。[頻度] スクリプトでは定番
import time

from lerobot.robots.so_follower import SO101Follower, SOFollowerRobotConfig

RATE_HZ = 50  # 指令を送る回数/秒。目標を少しずつずらして送ると、急に動かず滑らかに動く。
MARGIN = 3.0  # 可動域の端から、これだけ手前で止める [度]。端に押し当てるとサーボに無理がかかるため。
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def joint_limits(robot, name):
    """関節 name が動いてよい範囲 (下限, 上限) を、キャリブレーション結果から求める。

    キャリブレーションは、関節の端から端までを生のカウント値 (0〜4095 = 1周) で保存している。
    度への変換は LeRobot の内部と同じ式: 度 = (カウント - 範囲の中央) * 360 / 4095。
    つまり、範囲の中央がちょうど 0 度になる。gripper だけは 0〜100 の別の尺度。
    """
    if name == "gripper":
        return 0.0, 100.0
    cal = robot.bus.calibration[name]  # robot.bus.calibration: 関節名 -> 保存された範囲などの辞書
    half = (cal.range_max - cal.range_min) / 2 * 360 / 4095  # 中央から端までの角度 [度]
    return -half, half


def move_to(robot, start, goal, seconds, smooth=True):
    """全関節の指令を、start から goal まで seconds 秒かけて、滑らかに変える(smooth=False なら、一定の速さの直線)。

    start, goal : {"関節名.pos": 値} の辞書。キーの形 "shoulder_pan.pos" は LeRobot の決まり。
    動かさない関節は、start と goal を同じ値にしておけば、その位置を保つ。
    """
    steps = max(1, int(seconds * RATE_HZ))
    for i in range(1, steps + 1):
        t = i / steps  # 0 → 1 に進む割合
        if smooth:
            # 最小ジャーク曲線(5 次の補間): 動き出しと止まりの速度・加速度が 0 になる(カクッと動き出したり、止まったりしない)。
            # 真ん中の速さは、直線補間の約 1.9 倍になる(同じ秒数で、同じ距離を動くため)。[頻度] ロボットの軌道の滑らかさでは定番だが、初心者向けの教材ではあまり見ない
            t = 10 * t**3 - 15 * t**4 + 6 * t**5
        # 辞書内包表記: 各キーについて、start から goal へ割合 t だけ進んだ値を作る。[頻度] Python で非常によく使う
        action = {k: start[k] + (goal[k] - start[k]) * t for k in start}
        robot.send_action(action)  # サーボに「この角度へ」という目標を送る (動かすのはサーボ内部の制御)
        time.sleep(1 / RATE_HZ)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint", choices=JOINTS, help="動かす関節。省略すると全関節を順に動かす")
    parser.add_argument("--delta", type=float, default=5.0, help="動かす量 (度、gripper は 0〜100 の量)")
    parser.add_argument("--seconds", type=float, default=2.0, help="片道にかける秒数")
    args = parser.parse_args()

    robot = SO101Follower(SOFollowerRobotConfig(port="/dev/ttyACM0", id="my_follower_arm"))
    robot.connect()  # 接続するとトルクが入り、今の姿勢を保つ
    try:
        obs = robot.get_observation()
        start = {k: v for k, v in obs.items() if k.endswith(".pos")}  # カメラ等を除き、関節だけ残す
        print("開始姿勢:", {k: round(v, 1) for k, v in start.items()})

        for name in [args.joint] if args.joint else JOINTS:
            lo, hi = joint_limits(robot, name)
            key = f"{name}.pos"
            for sign in (+1, -1):
                target = start[key] + sign * args.delta
                if not (lo + MARGIN <= target <= hi - MARGIN):
                    # 端に近すぎる向きは、動かさずに飛ばす。
                    print(f"{name}: {sign * args.delta:+.1f} は可動域 [{lo:.1f}, {hi:.1f}] の端に近いので飛ばす")
                    continue
                goal = dict(start)  # dict(...): 辞書のコピー。他の関節は開始姿勢のまま
                goal[key] = target
                print(f"{name}: {start[key]:.1f} -> {target:.1f}")
                move_to(robot, start, goal, args.seconds)
                time.sleep(0.5)  # 止まるのを待って、実際の角度を読む
                now = robot.get_observation()[key]
                print(f"  指令 {target:.1f} / 実測 {now:.1f} (差 {now - target:+.1f})")
                move_to(robot, goal, start, args.seconds)  # 開始姿勢に戻る
                time.sleep(0.5)
    finally:
        # try/finally: 途中でエラーや Ctrl+C になっても、必ず切断する。[頻度] 機器を扱うときの定番
        robot.disconnect()  # トルクが切れるので、腕は落ちる。手で支えておく。


if __name__ == "__main__":
    main()
