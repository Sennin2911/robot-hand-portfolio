"""【概要】plan_reach_test.py が作った計画(outputs/reach_plan.json など)のとおりに、実機を動かし、各点で、関節角の読み取りと撮影を行って記録する。
各点の流れ: 指令値へ動かす(関節ごとに直線で、計画の秒数をかけて)→ 止まるのを待つ → 関節角を読む → record_hand_sample.py(この .venv の別プロセス)で撮影・記録(名前は「<tag>_<番号>」)。
全部の点が終わると、休めの姿勢へ戻り、「腕を支えたら Enter」を待ってから、トルクを切る(切ると腕は落ちる)。
安全:
  (1) 動かす前に、今の姿勢が「休めの姿勢」(計画の assumed_start)に近いか確認する(関節ごとに 10° 以内)。違えば、何も動かさず止まる
      (計画の経路の安全は、この姿勢から始めることを前提に、sim で確認してあるため)。
  (2) 動かす前に、計画(各点の指令値・秒数)を表示して、Enter を待つ。腕を支え、動く範囲(机、板、カメラ、ケーブル、手先の下)に物が無いことを確かめてから押す。
  (3) 途中で Ctrl+C を押しても、休めの姿勢へ戻ってから終わる(real_pan_sweep.py と同じ)。
  (4) 始める前に、ウェブカメラが認識されているか確認する。

使い方(LeRobot の環境で実行):
    cd ~/ドキュメント/robot-hand-portfolio
    ~/lerobot-work/lerobot/.venv/bin/python scripts/real_run_plan.py                                  # outputs/reach_plan.json
    ~/lerobot-work/lerobot/.venv/bin/python scripts/real_run_plan.py --plan outputs/reach_plan2.json  # 補正した計画
    ~/lerobot-work/lerobot/.venv/bin/python scripts/real_run_plan.py --only 1 2                       # 番号 1 と 2 だけ(先頭から続く番号だけ指定できる。最初の試しに)
    ~/lerobot-work/lerobot/.venv/bin/python scripts/real_run_plan.py --plan outputs/reach_plan_r1b.json --refine 1   # 関節の読みによる補正を 1 回入れる
    ~/lerobot-work/lerobot/.venv/bin/python scripts/real_run_plan.py --plan outputs/pad_plan.json --refine 1 --no-record   # 置き場の真上へ(撮影せず、目で確認)
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

# 同じフォルダの real_move_joints.py から、滑らかに動かす関数などを借りる(詳しい意味は real_send_pose.py を参照)。
from real_move_joints import JOINTS, SOFollowerRobotConfig, SO101Follower, move_to

ROOT = Path(__file__).resolve().parents[1]
JOINTS_FILE = ROOT / "outputs" / "real_joints.json"

parser = argparse.ArgumentParser()
parser.add_argument("--plan", default=str(ROOT / "outputs" / "reach_plan.json"))
parser.add_argument("--only", type=int, nargs="+", help="この番号の点だけ動かす(最初の試し用)")
parser.add_argument("--refine", type=int, default=0,
                    help="関節の読みによる補正の回数。動いたあと、実際の読みと指令の差(サーボが指令に届かない分。重さによる定常偏差)を、指令に足して、もう一度送る。0 = しない")
parser.add_argument("--no-record", action="store_true", help="撮影・記録をしない(手の板がカメラに映らない位置、たとえば置き場の試験で使う)")
parser.add_argument("--settle", type=float, default=1.5, help="動いたあと、止まるのを待つ秒数")
parser.add_argument("--venv-python", default=str(ROOT / ".venv" / "bin" / "python"), help="撮影・記録に使う、このプロジェクトの環境の python")
args = parser.parse_args()

plan = json.loads(Path(args.plan).read_text())
targets = [t for t in plan["targets"] if not args.only or t["id"] in args.only]
if not targets:
    raise SystemExit("動かす点が無い")
home = plan["home"]
# --only は、計画の「先頭から続く番号」だけ許す。計画の経路は、番号の順に動くことを前提に、sim で安全を確認してあるため(飛ばすと、確認していない経路になる)。
all_ids = [t["id"] for t in plan["targets"]]
if [t["id"] for t in targets] != all_ids[:len(targets)]:
    raise SystemExit(f"--only は、先頭から続く番号(例: {all_ids[:2]})だけ指定できます。経路の安全は、番号の順に動くことを前提に確認してあります")


def webcam_present():
    out = subprocess.run(["v4l2-ctl", "--list-devices"], capture_output=True, text=True).stdout
    return "WebCamera" in out


if not args.no_record and not webcam_present():
    raise SystemExit("ウェブカメラが USB で認識されていない。差し直してから、もう一度実行する。腕は動かしていない")

robot = SO101Follower(SOFollowerRobotConfig(port="/dev/ttyACM0", id="my_follower_arm"))
robot.connect()  # 接続するとトルクが入り、今の姿勢を保つ。接続中は、腕を手で支える
current = None
try:
    start = {k: v for k, v in robot.get_observation().items() if k.endswith(".pos")}
    far = [f"{k.split('.')[0]}: 今 {start[k]:.1f}、想定 {plan['assumed_start'][k]:.1f}" for k in plan["assumed_start"]
           if k != "gripper.pos" and abs(start[k] - plan["assumed_start"][k]) > 10]
    if far:
        print("今の姿勢が、休めの姿勢(計画の前提)から離れているので、何も動かさずに終了します。\n  " + "\n  ".join(far))
        print("腕を、休めの姿勢(畳んだ姿勢)に戻してから、もう一度実行してください。")
        raise SystemExit(1)
    print(f"計画 {plan['tag']}: {len(targets)} 点。手先の高さ {plan['tip_z'] * 100:.0f} cm")
    print(f"{'番号':>4s}  " + "  ".join(f"{n[:9]:>9s}" for n in JOINTS[:5]) + "  秒数")
    for t in targets:
        print(f"{t['id']:4d}  " + "  ".join(f"{t['cmd'][n + '.pos']:9.1f}" for n in JOINTS[:5]) + f"  {t['seconds']:4.0f}")
    input("腕を支え、動く範囲(机、板、カメラ、ケーブル、手先の下)に物が無いことを確認したら Enter(やめるなら Ctrl+C): ")

    current = dict(start)
    n_ok = 0
    for t in targets:
        goal = dict(start)
        goal.update(t["cmd"])
        goal["gripper.pos"] = start["gripper.pos"]  # gripper は動かさない
        print(f"\n--- 点 {t['id']}(半径 {t['r'] * 100:.0f} cm、方位 {t['theta_right_deg']}°)へ {t['seconds']:.0f} 秒で動かす")
        move_to(robot, current, goal, t["seconds"])
        current = goal
        time.sleep(args.settle)  # 止まるのを待つ
        for it in range(args.refine):
            # 関節の読みによる補正(エンコーダの読みで、サーボの定常偏差を打ち消す)。pan・lift・elbow・wrist_flex だけ。roll と gripper は触らない。
            now = robot.get_observation()
            diff = {k: goal[k] - now[k] for k in ("shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos")}
            if max(abs(v) for v in diff.values()) < 0.3:
                break  # 0.3° 未満なら、十分に届いている
            corr = dict(current)
            for k, v in diff.items():
                corr[k] = goal[k] + max(-3.0, min(3.0, v))  # 差を、指令に足す。一度に 3° までに抑える(max/min で範囲を切る。[頻度] numpy を使わない簡単な書き方)
            print("  補正" + str(it + 1) + ": " + ", ".join(f"{k.split('.')[0][:9]} {v:+.1f}°" for k, v in diff.items()) + " を指令に足して、もう一度送る")
            move_to(robot, current, corr, 1.5)
            current = corr
            time.sleep(1.0)
        obs = robot.get_observation()
        JOINTS_FILE.write_text(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "joints": {n: float(obs[f"{n}.pos"]) for n in JOINTS}},
                                          ensure_ascii=False, indent=2))
        if args.no_record:
            input(f"  点 {t['id']}({t.get('color', '')})に着いた。手先が目標の真上に来ているか、目で見て確かめたら Enter: ")
            n_ok += 1
            continue
        label = f"{plan['tag']}_{t['id']}"
        r = subprocess.run([args.venv_python, str(ROOT / "scripts" / "record_hand_sample.py"), "--label", label, "--joints-file-only"],
                           capture_output=True, text=True)  # 別プロセスで、撮影と記録を行う
        print(r.stdout.strip() or r.stderr.strip())  # 失敗しても、次の点へ進む
        n_ok += r.returncode == 0
    print(f"\n記録に成功: {n_ok} / {len(targets)} 点")
finally:
    # 終了: 休めの姿勢へ戻り、腕を支えてもらってから、トルクを切る(切ると腕は落ちる)。
    try:
        try:
            if current is not None:
                move_to(robot, current, {**home, "gripper.pos": current["gripper.pos"]}, plan.get("return_seconds", 8.0))
        except NameError:
            pass
        input("\n腕を支えたら Enter(このあとトルクが切れて、腕は落ちる): ")
    except (KeyboardInterrupt, EOFError):
        print("入力が中断された。このまま切断する(トルクが切れて、腕は落ちる)")
    finally:
        robot.disconnect()  # 入力の途中で Ctrl+C を押しても、必ず切断する
