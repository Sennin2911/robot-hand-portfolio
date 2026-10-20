"""【概要】実機の SO-101 を、指定した姿勢(6関節の値)へゆっくり動かして、しばらく保つ。
view_sim_pose.py が出力するコマンドをそのまま貼って使う。実機の動きと sim の見た目を見比べるのが目的。
安全のため、(1) 目標が可動域の端(MARGIN 以内)に近い関節があれば何も動かさず止まる、
(2) 動かす前に各関節の移動量を表示して、Enter を押すまで待つ。

使い方(LeRobot の環境で実行。腕を片手で支える。終了するとトルクが切れて腕が落ちる):
    python scripts/real_send_pose.py --pose 0 0 0 0 0 9.1 --seconds 5 --hold 5
値の並び: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper(度、gripper は 0〜100)
"""
import argparse
import time

# 同じフォルダの real_move_joints.py から、可動域を求める関数と、滑らかに動かす関数を借りる。
# `from real_move_joints import ...` が動くのは、「実行したスクリプトのあるフォルダ」が import の
# 探索先に最初から入っているため。[頻度] 同じフォルダのスクリプトを読むときの簡易な書き方(パッケージにするまでもない小さな場合)
from real_move_joints import JOINTS, MARGIN, SOFollowerRobotConfig, SO101Follower, joint_limits, move_to


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose", type=float, nargs=6, required=True, help="6関節の目標 (度、gripper は 0〜100)")
    parser.add_argument("--seconds", type=float, default=5.0, help="今の姿勢から目標まで動かす秒数")
    parser.add_argument("--hold", type=float, default=5.0, help="目標の姿勢を保つ秒数")
    args = parser.parse_args()
    goal_values = dict(zip(JOINTS, args.pose))  # dict(zip(キー, 値)): 2つの並びから辞書を作る。[頻度] 定番

    robot = SO101Follower(SOFollowerRobotConfig(port="/dev/ttyACM0", id="my_follower_arm"))
    robot.connect()  # 接続するとトルクが入り、今の姿勢を保つ
    try:
        obs = robot.get_observation()
        start = {k: v for k, v in obs.items() if k.endswith(".pos")}

        # 可動域の確認: 1つでも端に近ければ、全体をやめる(一部だけ動かすと姿勢が崩れるため)。
        too_close = []
        for name in JOINTS:
            lo, hi = joint_limits(robot, name)
            if not (lo + MARGIN <= goal_values[name] <= hi - MARGIN):
                too_close.append(f"{name}: 目標 {goal_values[name]:.1f} は可動域 [{lo:.1f}, {hi:.1f}] の端に近い")
        if too_close:
            print("動かさずに終了します。\n  " + "\n  ".join(too_close))
            return

        print("関節ごとの移動量(今 → 目標):")
        for name in JOINTS:
            now, to = start[f"{name}.pos"], goal_values[name]
            print(f"  {name:14s} {now:7.1f} -> {to:7.1f}  ({to - now:+.1f})")
        input("腕を支えて、周りに物がないことを確認したら Enter (やめるなら Ctrl+C): ")

        goal = {f"{name}.pos": goal_values[name] for name in JOINTS}
        move_to(robot, start, goal, args.seconds)
        time.sleep(args.hold)  # 姿勢を保つ。この間に、sim の画面と腕の形を見比べる
        now = robot.get_observation()
        print("実測と目標の差(実測 − 目標):")
        for name in JOINTS:
            print(f"  {name:14s} {now[f'{name}.pos'] - goal_values[name]:+.1f}")
        move_to(robot, goal, start, args.seconds)  # 開始姿勢に戻る
    finally:
        # try/finally: 途中でエラーや Ctrl+C になっても、必ず切断する。[頻度] 機器を扱うときの定番
        robot.disconnect()  # トルクが切れるので、腕は落ちる。手で支えておく。


if __name__ == "__main__":
    main()
