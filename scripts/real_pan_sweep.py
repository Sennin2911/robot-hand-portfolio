"""【概要】実機の肩のモーター(shoulder_pan)だけを、プログラムで少しずつ回し、各位置で「関節角の読み取り + カメラの撮影 + 手先の記録」を自動で行う。
fit_pan_sweep.py(肩の軸の位置を、測定に頼らず求める)に必要な、pan を変えた複数の姿勢の記録を、手で角度を測らずに作るためのもの。
動かすのは pan だけ(他の関節は、開始時の姿勢のまま、トルクで保つ)。開始時の pan から、右へ delta 度ずつ(既定 0, 15, 30, 45, 60)回す。pan の + は右(実機)。
安全:
  (1) 動かす前に、可動域の端(MARGIN 以内)に近い目標があれば、何も動かさず止まる。
  (2) 動かす前に、計画を表示して、Enter を押すまで待つ。腕を支え、周りの物(机の板、カメラ、ケーブル)に当たらないか確かめてから押す。
  (3) 終了時は、開始の姿勢に戻ってから、「腕を支えて Enter」を待ち、そのあとトルクを切る(切ると腕は落ちる)。途中で Ctrl+C を押しても、同じ手順で終わる。
撮影と記録は、この .venv の record_hand_sample.py を別プロセスで呼ぶ(カメラの処理は、こちらの環境にあるため)。

使い方(LeRobot の環境で実行。先に、手で、腕を前へ伸ばした姿勢にしておく。pan は、ほぼ正面):
    cd ~/ドキュメント/robot-hand-portfolio
    ~/lerobot-work/lerobot/.venv/bin/python scripts/real_pan_sweep.py
    ~/lerobot-work/lerobot/.venv/bin/python scripts/real_pan_sweep.py --deltas 0 20 40 60 --seconds 4
    ~/lerobot-work/lerobot/.venv/bin/python scripts/real_pan_sweep.py --tag B --pose -7 2 95 --deltas 0 30 60   # 形(lift, elbow, wrist_flex)を指定して、自動で動かし、記録する
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

# 同じフォルダの real_move_joints.py から、可動域を求める関数と、滑らかに動かす関数を借りる(詳しい意味は real_send_pose.py を参照)。
from real_move_joints import JOINTS, MARGIN, SOFollowerRobotConfig, SO101Follower, joint_limits, move_to

ROOT = Path(__file__).resolve().parents[1]  # このファイルの 1 つ上のフォルダ = プロジェクトの根
JOINTS_FILE = ROOT / "outputs" / "real_joints.json"

parser = argparse.ArgumentParser()
parser.add_argument("--deltas", type=float, nargs="+", default=[0, 15, 30, 45, 60], help="開始時の pan からの、右への回転量(度)。先頭は 0 にする")
parser.add_argument("--tag", default="sweep", help="この腕の形の名前(例: B、down)。サンプルの名前は「<tag>_+0」のようになる。形ごとに変えること(同じ名前は上書きされる)")
parser.add_argument("--pose", type=float, nargs=3, metavar=("LIFT", "ELBOW", "WRIST_FLEX"),
                    help="腕の形(shoulder_lift, elbow_flex, wrist_flex の読み。度。real_read_joints.py と同じ値)。指定すると、まずこの形まで自動で動かす(pan は --pan0、wrist_roll は 0、gripper はそのまま)。"
                         "省略すると、今の形のまま pan だけを振る")
parser.add_argument("--pan0", type=float, default=0.0, help="--pose を使うときの、pan の開始の値(度)。0 = 正面(OFFSET の分、右へ 4.4° 向く)")
parser.add_argument("--pose-seconds", type=float, default=8.0, help="--pose の形まで動かす秒数(長いほど、ゆっくり動く)")
parser.add_argument("--seconds", type=float, default=3.0, help="1 つの位置へ動かす秒数")
parser.add_argument("--settle", type=float, default=1.0, help="動いたあと、止まるのを待つ秒数")
parser.add_argument("--venv-python", default=str(ROOT / ".venv" / "bin" / "python"), help="撮影・記録に使う、このプロジェクトの環境の python")
args = parser.parse_args()


def webcam_present():
    """ウェブカメラが USB に認識されているか。v4l2-ctl --list-devices の出力に "WebCamera" があるかで調べる(外部コマンドを呼ぶ: subprocess.run)。"""
    out = subprocess.run(["v4l2-ctl", "--list-devices"], capture_output=True, text=True).stdout
    return "WebCamera" in out


def joints_of(obs):
    """get_observation の結果から、関節の値だけを {名前: 値} にして返す(カメラ等は除く)。"""
    return {name: float(obs[f"{name}.pos"]) for name in JOINTS}


if not webcam_present():
    # 動かし始めてから撮影に失敗すると、腕を動かした意味がない。先に確かめる(2026-10-24 に、USB が切れたまま動かして、記録が 0 個になった)
    raise SystemExit("ウェブカメラが USB で認識されていない。差し直してから(v4l2-ctl --list-devices に WebCamera が出るまで)もう一度実行する。腕は動かしていない")
robot = SO101Follower(SOFollowerRobotConfig(port="/dev/ttyACM0", id="my_follower_arm"))
robot.connect()  # 接続するとトルクが入り、今の姿勢を保つ。接続中は、腕を手で支える
try:
    start = {k: v for k, v in robot.get_observation().items() if k.endswith(".pos")}
    pan0 = start["shoulder_pan.pos"]
    lo, hi = joint_limits(robot, "shoulder_pan")
    pan_base = args.pan0 if args.pose is not None else pan0  # --pose のときは、動かしたあとの pan(--pan0)が基準
    bad = [d for d in args.deltas if not (lo + MARGIN <= pan_base + d <= hi - MARGIN)]
    if bad:
        print(f"動かさずに終了します。pan の目標 {[round(pan_base + d, 1) for d in bad]} は、可動域 [{lo:.1f}, {hi:.1f}] の端に近い。開始の pan({pan0:.1f})を変えるか、--deltas を減らす")
        raise SystemExit(1)
    home = dict(start)  # 最初の姿勢。終了時に、ここへ戻る
    if args.pose is not None:
        # 指定された形の、可動域の確認(端から MARGIN 以内なら、何も動かさず止まる)。
        shape = dict(start)
        shape.update({"shoulder_pan.pos": args.pan0, "shoulder_lift.pos": args.pose[0], "elbow_flex.pos": args.pose[1],
                      "wrist_flex.pos": args.pose[2], "wrist_roll.pos": 0.0})
        too_close = []
        for name in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"):
            lo_j, hi_j = joint_limits(robot, name)
            if not (lo_j + MARGIN <= shape[f"{name}.pos"] <= hi_j - MARGIN):
                too_close.append(f"{name}: {shape[f'{name}.pos']:.1f} は可動域 [{lo_j:.1f}, {hi_j:.1f}] の端に近い")
        if too_close:
            print("動かさずに終了します。\n  " + "\n  ".join(too_close))
            raise SystemExit(1)
        print("開始姿勢 → 指定の形(度):")
        for name in JOINTS:
            print(f"  {name:14s} {start[f'{name}.pos']:7.1f} -> {shape[f'{name}.pos']:7.1f}")
        print(f"この形まで {args.pose_seconds:.0f} 秒かけて動かし、そこから pan を右へ " + ", ".join(f"{d:.0f}°" for d in args.deltas) + " 回して記録する。終わると、最初の姿勢へ戻る")
        input("腕を支え、動く範囲(机、板、カメラ、ケーブル)に物が無いことを確認したら Enter(やめるなら Ctrl+C): ")
        move_to(robot, start, shape, args.pose_seconds)  # 関節ごとに直線で動かす(sim で、机から 76 mm 以上離れることを確認済みの形)
        time.sleep(1.5)
        start = shape  # 以降の pan 振りは、この形が基準
        pan0 = shape["shoulder_pan.pos"]
    else:
        print("開始姿勢:", {k: round(v, 1) for k, v in start.items()})
        print("計画(pan だけを動かす。他の関節は保つ): " + ", ".join(f"{pan0 + d:.1f}°" for d in args.deltas))
        input("腕を支え、動く範囲に物が無いことを確認したら Enter(やめるなら Ctrl+C): ")

    current = dict(start)
    n_ok = 0
    for d in args.deltas:
        goal = dict(start)  # dict(...): 辞書のコピー。他の関節は開始時のまま
        goal["shoulder_pan.pos"] = pan0 + d
        move_to(robot, current, goal, args.seconds)
        current = goal
        time.sleep(args.settle)  # 止まるのを待つ
        obs = robot.get_observation()
        JOINTS_FILE.write_text(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "joints": joints_of(obs)}, ensure_ascii=False, indent=2))
        label = f"{args.tag}_{d:+.0f}"
        print(f"\n--- pan {obs['shoulder_pan.pos']:.1f}°(指令 {goal['shoulder_pan.pos']:.1f}°)で撮影・記録: {label}")
        r = subprocess.run([args.venv_python, str(ROOT / "scripts" / "record_hand_sample.py"), "--label", label, "--joints-file-only"],
                           capture_output=True, text=True)  # 別プロセスで、撮影と記録を行う。capture_output: 出力を受け取る
        print(r.stdout.strip() or r.stderr.strip())  # 成功なら標準出力、失敗なら標準エラーを表示。失敗しても、次の位置へ進む
        n_ok += r.returncode == 0  # True は 1 として足せる(成功した数を数える)。[頻度] 数を数えるときの、ちょっとした書き方
    print(f"\n記録に成功: {n_ok} / {len(args.deltas)} 位置")
finally:
    # 終了: 開始の姿勢に戻り、腕を支えてもらってから、トルクを切る(切ると腕は落ちる)。
    try:
        try:
            move_to(robot, current, home, args.pose_seconds if args.pose is not None else args.seconds)
        except NameError:
            pass  # 動き出す前に終わった(current がまだ無い)ときは、戻す必要がない
        input("\n腕を支えたら Enter(このあとトルクが切れて、腕は落ちる): ")
    except (KeyboardInterrupt, EOFError):
        print("入力が中断された。このまま切断する(トルクが切れて、腕は落ちる)")
    finally:
        robot.disconnect()  # 入力の途中で Ctrl+C を押しても、必ず切断する(入れ子の try/finally)
