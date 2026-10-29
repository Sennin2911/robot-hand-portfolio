"""【概要】実機の「降下(掴む姿勢へ下がる)」と「持ち上げ(真上へ戻る)」を、かける秒数を変えて動かし、指令と関節の読みを記録して、揺れ・カクつきを数値で比べる診断。
立方体は使わない(置かない)。最新の計画 outputs/pick_plan_<色>.json の段階 1(真上)と段階 2(掴む姿勢)の間を往復する。結果は outputs/vertical_motion_*.csv と、表示。
指標(関節の読みを、指令と同じ 50 Hz で記録して求める):
  stall   : 指令は動いているのに、読みが前回と全く同じだった割合 [%]。大きいと、読みが階段状に進む(サーボの分解能 約 0.088° より、1 回の指令の刻みが小さいとき)
  err_max : 指令と読みの差の最大 [度]。大きいと、サーボが指令に遅れている
  jerk    : 読みの 2 階差分(加速度に近い量)の RMS [度/tick²]。同じ距離を動くなら、小さいほど滑らか。急な加減速・カクつきで大きくなる
  over    : 止まったあと 1 秒間の、最終位置からの最大のずれ [度]。大きいと、止まる瞬間に行き過ぎて揺れている

使い方(LeRobot の環境で実行。腕を休めの姿勢にして、机の上を空ける):
    ~/lerobot-work/lerobot/.venv/bin/python scripts/diagnose_vertical_motion.py --color green
"""
import argparse
import csv
import json
import time
from pathlib import Path

from real_move_joints import RATE_HZ, SOFollowerRobotConfig, SO101Follower, move_to  # move_to: 滑らかに動かす関数(診断以外の移動に使う)

ROOT = Path(__file__).resolve().parents[1]
ARM_KEYS = ("shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos")  # 上下の動きで主に動く 3 関節
parser = argparse.ArgumentParser()
parser.add_argument("--color", default="green", choices=["red", "green", "blue"])
parser.add_argument("--down", type=float, nargs="+", default=[4.0, 2.5, 1.5], help="降下にかける秒数(複数。順に試す)")
parser.add_argument("--up", type=float, nargs="+", default=[0.8, 1.5, 2.5], help="持ち上げにかける秒数(複数。順に試す)")
args = parser.parse_args()

plan = json.loads((ROOT / "outputs" / f"pick_plan_{args.color}.json").read_text())
above, grasp = plan["stages"][0]["cmd"], plan["stages"][1]["cmd"]  # 段階 1(真上)と段階 2(掴む姿勢)の指令


def recorded_move(robot, start, goal, seconds, label, rows):
    """move_to と同じ軌道(5 次の補間)で動かしながら、毎回の指令と読みを rows に足す。止まったあと 1 秒も記録する。戻り値: 指標の辞書。"""
    steps = max(1, int(seconds * RATE_HZ))
    log = []  # (指令, 読み, 動いている最中か)
    for i in range(1, steps + 1 + RATE_HZ):  # 動く steps 回 + 止まったあとの 1 秒(RATE_HZ 回)
        t = min(i / steps, 1.0)
        t = 10 * t**3 - 15 * t**4 + 6 * t**5
        action = {k: start[k] + (goal[k] - start[k]) * t for k in start}
        robot.send_action(action)
        time.sleep(1 / RATE_HZ)
        obs = robot.get_observation()
        log.append(({k: action[k] for k in ARM_KEYS}, {k: obs[k] for k in ARM_KEYS}, i <= steps))
    for n, (cmd, act, moving) in enumerate(log):
        rows.append([label, n, int(moving)] + [round(cmd[k], 3) for k in ARM_KEYS] + [round(act[k], 3) for k in ARM_KEYS])
    stall = tot = 0
    jerk2 = []
    err_max = 0.0
    for k in ARM_KEYS:
        a = [x[1][k] for x in log]  # [k] で辞書から取り出す
        c = [x[0][k] for x in log]
        for n in range(1, steps):
            if abs(c[n] - c[n - 1]) > 0.05:  # 指令が 0.05° 以上動いた tick だけ数える
                tot += 1
                stall += (a[n] == a[n - 1])
        jerk2 += [(a[n + 1] - 2 * a[n] + a[n - 1]) ** 2 for n in range(1, steps)]
        err_max = max(err_max, max(abs(c[n] - a[n]) for n in range(steps)))
    tail = log[steps:]  # 止まったあとの 1 秒
    over = max(max(abs(x[1][k] - tail[-1][1][k]) for x in tail) for k in ARM_KEYS)  # tail[-1]: 最後の要素(最終位置)。[頻度] [-1] は定番
    return {"stall": 100 * stall / max(tot, 1), "err_max": err_max, "jerk": (sum(jerk2) / len(jerk2)) ** 0.5, "over": over}


robot = SO101Follower(SOFollowerRobotConfig(port="/dev/ttyACM0", id="my_follower_arm"))
robot.connect()  # 接続するとトルクが入る。腕を手で支える
rows, results = [], []
try:
    start = {k: v for k, v in robot.get_observation().items() if k.endswith(".pos")}
    input("腕を休めの姿勢にして、机の上と腕の動く範囲を空けた(立方体なし)。Enter で、真上の姿勢へゆっくり移動(やめるなら Ctrl+C): ")
    move_to(robot, start, above, 8.0)
    time.sleep(1.0)
    for d, u in zip(args.down, args.up):
        results.append(("降下", d, recorded_move(robot, above, grasp, d, f"down_{d}", rows)))
        time.sleep(0.5)
        results.append(("持ち上げ", u, recorded_move(robot, grasp, above, u, f"up_{u}", rows)))
        time.sleep(0.5)
    now = {k: v for k, v in robot.get_observation().items() if k.endswith(".pos")}
    move_to(robot, now, start, 8.0)  # 休めの姿勢へ戻る
    input("腕を支えたら Enter(このあとトルクが切れて、腕は落ちる): ")
finally:
    robot.disconnect()
if rows:
    out = ROOT / "outputs" / f"vertical_motion_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with open(out, "w", newline="") as f:  # with ... as: ブロックを抜けると、自動でファイルを閉じる。[頻度] ファイルを扱うときの定番
        w = csv.writer(f)
        w.writerow(["label", "tick", "moving"] + [f"cmd_{k}" for k in ARM_KEYS] + [f"act_{k}" for k in ARM_KEYS])
        w.writerows(rows)
    print(f"\n記録: {out}")
    print(f"{'動き':8s}{'秒':>6s}{'stall%':>9s}{'err_max°':>10s}{'jerk':>8s}{'over°':>8s}")
    for name, sec, r in results:
        print(f"{name:8s}{sec:6.1f}{r['stall']:9.1f}{r['err_max']:10.2f}{r['jerk']:8.3f}{r['over']:8.2f}")
