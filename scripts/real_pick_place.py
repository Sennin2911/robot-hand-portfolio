"""【概要】plan_pick_test.py が作った計画(outputs/pick_plan_<色>.json)のとおりに、実機で立方体を掴んで置き場へ運ぶ。段階ごとに Enter を待つ。
段階: 1 立方体の真上へ(指を開く)→ 2 降下 → 3 指を閉じる → 4 持ち上げる → 5 置き場の真上へ → 6 降下 → 7 指を開く → 8 上がる。
--until N で、段階 N までで止める(最初の試しは --until 1、次に --until 2(降下して、指が立方体をまたぐか目で確認)、…と、少しずつ進める)。
止めたあとは、真上の姿勢(段階 1 または 5 の関節角)を経由して、休めの姿勢へ戻る。
安全:
  (1) 動かす前に、今の姿勢が「休めの姿勢」に近いか確認する(関節ごとに 10° 以内)。違えば、何も動かさず止まる。
  (2) 動かす前に、計画を表示して、Enter を待つ。降下の段階(2、6)と、指を閉じる段階(3)は、その直前にも Enter を待つ(腕を支え、目で見る)。
  (3) 途中で Ctrl+C を押しても、真上の姿勢を経由して休めの姿勢へ戻ってから終わる。
  (4) 指を閉じる指令は、計画の値(約 4)。立方体に当たって止まる(サーボの過負荷の保護が働く)。立方体が無い状態で閉じても、指が互いに当たるだけ。

使い方(LeRobot の環境で実行):
    cd ~/ドキュメント/robot-hand-portfolio
    ~/lerobot-work/lerobot/.venv/bin/python scripts/real_pick_place.py --color red --until 1          # 真上まで
    ~/lerobot-work/lerobot/.venv/bin/python scripts/real_pick_place.py --color red --until 2 --refine 1   # 降下まで
    ~/lerobot-work/lerobot/.venv/bin/python scripts/real_pick_place.py --color red                  # 最後(段階 8)まで
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# 同じフォルダの real_move_joints.py から、滑らかに動かす関数などを借りる(詳しい意味は real_send_pose.py を参照)。
from real_move_joints import JOINTS, SOFollowerRobotConfig, SO101Follower, move_to

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--color", required=True, choices=["red", "green", "blue"])
parser.add_argument("--until", type=int, default=8, help="この段階までで止める(1〜8)")
parser.add_argument("--refine", type=int, default=0, help="関節の読みによる補正の回数(動いたあと、読みと指令の差を指令に足す)。0 = しない")
parser.add_argument("--record", action="store_true",
                    help="段階 1、2、3 で止まったとき、手の板をカメラで撮って、手先の実際の位置と立方体の位置を記録する(labelは PICK_<色>_<段階>)。"
                         "analyze_grasp_offset.py(掴む前)と update_grasp_offset.py(段階 3 = 掴んだ直後。指の面のずれを貯める)が読む。手の板が映る位置だけ")
parser.add_argument("--auto", action="store_true",
                    help="試験用。最初の 1 回の Enter だけを待ち、あとは段階の確認を待たずに最後まで進む(終了時の、腕を支える確認も省く。--confirm-end で残せる)。"
                         "目で見る代わりに、安全装置が働く: 関節の読みが指令から --max-joint-err 度以上ずれたら、指を閉じたあとに何も掴めていなければ、真上へ戻って中止する。結果は outputs/real_trial_result.json")
parser.add_argument("--confirm-end", action="store_true", help="--auto のとき、終了時の「腕を支えたら Enter(トルクが切れて腕が落ちる)」を残す(慣れるまでの最初の数回向け)")
parser.add_argument("--max-joint-err", type=float, default=5.0, help="--auto の安全装置。腕が動く段階のあと、関節の読みと指令の差がこれ(度)を超えたら中止する")
parser.add_argument("--grasp-margin", type=float, default=0.8,
                    help="--auto の安全装置。指を閉じたあとの指の読みが「閉じる指令 + この値」未満なら、何も掴めていない(指が閉じ切った)とみなして中止する。"
                         "立方体を掴んだ実測は、閉じる指令より 1.6 ほど大きい(5.5 など)")
parser.add_argument("--speed", type=float, default=1.0, help="動かす速さの倍率(2 なら、計画の秒数の半分で動かす)。降下(2、6)は、安全のため 1.5 倍までに抑える")
parser.add_argument("--settle", type=float, default=1.5, help="動いたあと、止まるのを待つ秒数")
args = parser.parse_args()

plan_file = ROOT / "outputs" / f"pick_plan_{args.color}.json"
if not plan_file.exists():  # 計画が無いまま実行したときに、分かりやすい案内を出す(腕は動かしていない)
    raise SystemExit(f"{plan_file.name} が無い。先に、立方体を検出して、計画を作る:\n"
                     f"  .venv/bin/python scripts/detect_cubes_world.py\n  MUJOCO_GL=egl .venv/bin/python scripts/plan_pick_test.py --color {args.color}")
plan = json.loads(plan_file.read_text())
stages = plan["stages"][:args.until]
ARM_KEYS = ("shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos")
(ROOT / "outputs" / "real_trial_result.json").unlink(missing_ok=True)  # 前の試験の結果が残って、今回のものと取り違えないように、先に消す
robot = SO101Follower(SOFollowerRobotConfig(port="/dev/ttyACM0", id="my_follower_arm"))
robot.connect()  # 接続するとトルクが入り、今の姿勢を保つ。接続中は、腕を手で支える
current = None
last = 0  # 最後に終えた段階の番号
abort_reason = None  # --auto の安全装置などで中止した理由(None = 最後まで進んだ)。real_trial_result.json に書く
grip_after_close = None  # 段階 3 のあとの指の読み(立方体を掴めたかの目安)
max_joint_err = 0.0  # 腕が動く段階のあとの、関節の読みと指令の差の最大 [度]
t_start = time.time()  # 動き始めの時刻(最初の Enter のあとに、もう一度取り直す)


class TrialAbort(Exception):
    """--auto の安全装置が働いたときに投げる。class で自分用の例外を作る書き方。[頻度] 「この条件になったら止める」を呼び出し元へ伝えたいときの定番"""


try:
    start = {k: v for k, v in robot.get_observation().items() if k.endswith(".pos")}
    far = [f"{k.split('.')[0]}: 今 {start[k]:.1f}、想定 {plan['assumed_start'][k]:.1f}" for k in plan["assumed_start"]
           if k != "gripper.pos" and abs(start[k] - plan["assumed_start"][k]) > 10]
    if far:
        print("今の姿勢が、休めの姿勢(計画の前提)から離れているので、何も動かさずに終了します。\n  " + "\n  ".join(far))
        abort_reason = "not_at_rest: 開始の姿勢が休めの姿勢から離れている"
        raise SystemExit(1)
    c = plan["cube"]
    print(f"計画 {plan['tag']}: 立方体 ({c['x_m'] * 1000:.0f}, {c['y_m'] * 1000:.0f}) mm、手の向き {plan['hand_yaw_deg']:.0f}°。段階 {args.until} まで")
    print(f"{'段階':28s}" + "".join(f"{n[:6]:>7s}" for n in JOINTS) + "  秒")
    for st in stages:
        print(f"{st['name']:24s}" + "".join(f"{st['cmd'][n + '.pos']:7.1f}" for n in JOINTS) + f"  {st['seconds']:.1f}")
    input("立方体を置いた。腕の動く範囲(机、板、カメラ、ケーブル、置き場)に手や物が無いことを確認したら Enter(やめるなら Ctrl+C): " if args.auto
          else "腕を支え、動く範囲(机、板、カメラ、ケーブル、立方体、置き場)を確認したら Enter(やめるなら Ctrl+C): ")
    t_start = time.time()  # 人が Enter を押すまでの待ち時間を、1 回あたりの時間(cycle_s)に含めない

    current = dict(start)
    for i, st in enumerate(stages, start=1):
        if st["name"][0] in "236" and not args.auto:
            input(f"\n[{st['name']}] 動かす前の確認(目で見て、問題なければ Enter): ")
        else:
            print(f"\n--- {st['name']}")
        goal = dict(st["cmd"])
        speed = min(args.speed, 1.5) if st["name"][0] in "26" else args.speed
        move_to(robot, current, goal, max(0.8, st["seconds"] / speed))  # 速くしすぎないよう、1 回の動きは 0.8 秒以上かける
        current = goal
        time.sleep(args.settle)
        if st["name"][0] in "1256":  # 腕が動く段階だけ、関節の読みで補正(指だけ動かす 3、7 と、持ち上げ・上がる 4、8 は、しない)
            for it in range(args.refine):
                now = robot.get_observation()
                diff = {k: goal[k] - now[k] for k in ARM_KEYS}
                if max(abs(v) for v in diff.values()) < 0.3:
                    break  # 0.3° 未満なら、十分に届いている
                corr = dict(current)
                for k, v in diff.items():
                    corr[k] = goal[k] + max(-3.0, min(3.0, v))  # 差を指令に足す。一度に 3° まで
                print("  補正" + str(it + 1) + ": " + ", ".join(f"{k.split('.')[0][:9]} {v:+.1f}°" for k, v in diff.items()))
                move_to(robot, current, corr, 1.5)
                current = corr
                time.sleep(1.0)
        last = i
        obs = robot.get_observation()
        if args.record and st["name"][0] in "123":
            (ROOT / "outputs" / "real_joints.json").write_text(json.dumps(
                {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "joints": {n: float(obs[f"{n}.pos"]) for n in JOINTS}}, ensure_ascii=False, indent=2))
            r = subprocess.run([str(ROOT / ".venv" / "bin" / "python"), str(ROOT / "scripts" / "record_hand_sample.py"),
                                "--label", f"{plan['tag']}_{st['name'][0]}", "--joints-file-only", "--detect-cube", args.color], capture_output=True, text=True)  # 別プロセスで、撮影と記録を行う
            print("  記録: " + (r.stdout.strip() or r.stderr.strip()).splitlines()[-1])
        print("  関節の読み: " + ", ".join(f"{n[:6]} {obs[n + '.pos']:.1f}" for n in JOINTS))
        if st["name"][0] in "124568":  # 腕が動いた段階: 指令に届いているか(机や物に当たって止まっていないか)
            err = max(abs(st["cmd"][k] - obs[k]) for k in ARM_KEYS)
            max_joint_err = max(max_joint_err, err)
            if args.auto and err > args.max_joint_err:
                raise TrialAbort(f"joint_error: 段階 {st['name'][0]} で、関節の読みが指令から {err:.1f}° ずれている(上限 {args.max_joint_err}°)")
        if st["name"][0] == "3":
            grip_after_close = float(obs["gripper.pos"])
            if args.auto and grip_after_close < plan["gripper_close"] + args.grasp_margin:
                raise TrialAbort(f"grasp_missed: 指の読み {grip_after_close:.1f} が閉じ切りに近い(閉じる指令 {plan['gripper_close']:.1f})。何も掴めていない")
        if (i == args.until or st["name"][0] in "237") and not args.auto:
            input("  目で見て、様子(位置・立方体・指)を確認したら Enter: ")
except TrialAbort as e:
    abort_reason = str(e)
    print(f"\n中止: {abort_reason}\n真上へ戻って、休めの姿勢へ戻る")
except KeyboardInterrupt:
    abort_reason = "interrupted: Ctrl+C"
    raise  # 後始末(下の finally)のあと、そのまま終了する
finally:
    # 終了: 真上の姿勢(段階 1 か 5)を経由して、休めの姿勢へ戻り、腕を支えてもらってから、トルクを切る。
    try:
        try:
            if current is not None:
                current = {k: v for k, v in robot.get_observation().items() if k.endswith(".pos")}  # 途中で中断したときのため、今の実際の姿勢から動かす
                via = plan["stages"][4 if last >= 5 else 0]["cmd"]
                if last >= 2:  # 降下した後は、まず真上へ(指の状態はそのまま)
                    up = {**via, "gripper.pos": current["gripper.pos"]}
                    move_to(robot, current, up, 3.0)
                    current = up
                    if current["gripper.pos"] < plan["gripper_open"] - 5:  # 立方体を掴んだまま(指が閉じている)なら、真上で放す(5 cm ほどの高さ。休めの姿勢へ畳むと、立方体を挟みつぶすため)
                        if not args.auto:
                            input("\n立方体を掴んだままです。真上で放します。下に手などが無いことを確認したら Enter: ")
                        opened = {**current, "gripper.pos": plan["gripper_open"]}
                        move_to(robot, current, opened, 1.5)
                        current = opened
                move_to(robot, current, {**plan["home"], "gripper.pos": current["gripper.pos"]}, plan.get("return_seconds", 8.0))
        except NameError:  # current が未定義(接続直後の失敗)のとき
            pass
        (ROOT / "outputs" / "real_trial_result.json").write_text(json.dumps(  # 試験の判定(judge_real_trial.py)が読む
            {"tag": plan["tag"], "time": time.strftime("%Y-%m-%d %H:%M:%S"), "last_stage": last, "total_stages": len(plan["stages"]), "abort_reason": abort_reason,
             "grip_after_close": grip_after_close, "max_joint_err_deg": round(max_joint_err, 2), "cycle_s": round(time.time() - t_start, 1)}, ensure_ascii=False, indent=2))
        if not args.auto or args.confirm_end:
            input("\n腕を支えたら Enter(このあとトルクが切れて、腕は落ちる): ")
    except (KeyboardInterrupt, EOFError):
        print("入力が中断された。このまま切断する(トルクが切れて、腕は落ちる)")
    finally:
        robot.disconnect()  # 入力の途中で Ctrl+C を押しても、必ず切断する
