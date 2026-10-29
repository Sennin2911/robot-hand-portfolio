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
import statistics
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
parser.add_argument("--no-feedforward", action="store_true",
                    help="段階ごとの「前回までの補正量」を、最初から指令へ足す仕組み(下の BIAS_FILE)を使わない。使うと、動いたあとの微調整の動き(止まる → 待つ → 小さく動く)がほぼ無くなり、滑らかになる")
parser.add_argument("--refine-tol", type=float, default=0.4, help="動いたあとの関節の読みと指令の差が、これ(度)未満なら、微調整の動きをしない(--refine が 1 以上のとき)")
parser.add_argument("--record", action="store_true",
                    help="段階 1、2、3 で止まったとき、手の板をカメラで撮って、手先の実際の位置と立方体の位置を記録する(labelは PICK_<色>_<段階>)。"
                         "analyze_grasp_offset.py(掴む前)と update_grasp_offset.py(段階 3 = 掴んだ直後。指の面のずれを貯める)が読む。手の板が映る位置だけ")
parser.add_argument("--auto", action="store_true",
                    help="試験用。最初の 1 回の Enter だけを待ち、あとは段階の確認を待たずに最後まで進む(終了時の、腕を支える確認も省く。--confirm-end で残せる)。"
                         "目で見る代わりに、安全装置が働く: 関節の読みが指令から --max-joint-err 度以上ずれたら、指を閉じたあとに何も掴めていなければ、真上へ戻って中止する。結果は outputs/real_trial_result.json")
parser.add_argument("--confirm-end", action="store_true", help="--auto のとき、終了時の「腕を支えたら Enter(トルクが切れて腕が落ちる)」を残す(慣れるまでの最初の数回向け)")
parser.add_argument("--max-joint-err", type=float, default=5.0, help="--auto の安全装置。腕が動く段階のあと、関節の読みと指令の差がこれ(度)を超えたら中止する")
parser.add_argument("--grasp-margin", type=float, default=3.0,
                    help="--auto の安全装置。指を閉じたあとの指の読みが「空で閉じ切った読み + この値」未満なら、何も掴めていないとみなして中止する。"
                         "クッション付きの実測: 空で閉じ切り 9.4、立方体を掴むと約 20〜22(計画に gripper_empty が無いときは、閉じる指令 + この値)")
parser.add_argument("--speed", type=float, default=1.0, help="動かす速さの倍率(2 なら、計画の秒数の半分で動かす)。降下(2、6)は、安全のため 1.5 倍までに抑える")
parser.add_argument("--settle", type=float, default=1.0, help="動いたあと、止まるのを待つ秒数")
args = parser.parse_args()

# --- 補正量の先回し(フィードフォワード)---
# サーボは、重力や摩擦で、指令から毎回ほぼ同じ量だけずれて止まる(定常偏差)。段階ごとに「最終的に届いた指令」を覚えておき、次回は最初からその分を足して動く。
# 動いたあとに「読んで、差を見て、もう一度小さく動く」補正をしなくてよくなるので、動きが途切れない。ずれ方は姿勢で変わるので、直近 10 回の中央値(外れ値に強い)を使う。
# 上限は 3° (これ以上の補正量は、サーボ以外の原因を疑うべきなので、足さない)。[頻度] 制御では定番の考え方。このファイルの書き方(json で貯める)は、この計画の独自のもの
BIAS_FILE = ROOT / "calibration" / "servo_bias.json"
BIAS_MAX_DEG, BIAS_KEEP = 3.0, 10


def load_bias():
    """{段階の番号: {"関節.pos": [過去の補正量, ...]}} を返す。ファイルが無い・読めないときは空。try/except: 失敗しても動かすのを止めない。[頻度] 補助的なファイルの読み込みでは定番"""
    try:
        return json.loads(BIAS_FILE.read_text())
    except (OSError, ValueError):
        return {}


def predicted_bias(table, stage):
    """この段階で足す補正量(関節ごと、度)。履歴の中央値。履歴が無ければ 0。"""
    hist = table.get(stage, {})
    return {k: max(-BIAS_MAX_DEG, min(BIAS_MAX_DEG, statistics.median(hist[k]))) if hist.get(k) else 0.0 for k in ARM_KEYS}


def wait_settled(robot, max_seconds):
    """腕が止まる(読みの変化が 0.15° 未満)まで待つ。最長 max_seconds。固定の時間待ちより、止まった瞬間に次へ進めるので速い。戻り値: 最後の読み。"""
    deadline, prev = time.time() + max_seconds, None
    time.sleep(0.2)
    while True:
        obs = robot.get_observation()
        if (prev is not None and max(abs(obs[k] - prev[k]) for k in ARM_KEYS) < 0.15) or time.time() > deadline:
            return obs
        prev = obs
        time.sleep(0.1)


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
    bias_table = load_bias()
    for i, st in enumerate(stages, start=1):
        if st["name"][0] in "236" and not args.auto:
            input(f"\n[{st['name']}] 動かす前の確認(目で見て、問題なければ Enter): ")
        else:
            print(f"\n--- {st['name']}")
        goal = dict(st["cmd"])
        stage = st["name"][0]
        arm_stage = stage in "124568"  # 腕が動く段階
        speed = min(args.speed, 1.5) if stage in "26" else args.speed
        bias = predicted_bias(bias_table, stage) if (arm_stage and not args.no_feedforward) else {k: 0.0 for k in ARM_KEYS}
        if arm_stage:
            sent = {**goal, **{k: goal[k] + bias[k] for k in ARM_KEYS}}  # 補正量を足した指令(足す量が 0 なら goal と同じ)。{**a, **b}: 辞書を合成し、同じキーは後ろが勝つ。[頻度] 定番
        else:
            sent = {**goal, **{k: current[k] for k in ARM_KEYS}}  # 指だけ動かす段階(3、7)は、腕を今の指令のままにする(goal に戻すと、足した補正量の分、腕が動いてしまう)
        if any(abs(v) >= 0.3 for v in bias.values()):
            print("  先回し: " + ", ".join(f"{k.split('.')[0][:9]} {v:+.1f}°" for k, v in bias.items()))
        move_to(robot, current, sent, max(0.8, st["seconds"] / speed))  # 速くしすぎないよう、1 回の動きは 0.8 秒以上かける
        current = sent
        obs = wait_settled(robot, args.settle if stage in "1256" else min(args.settle, 0.3))  # 止まったらすぐ次へ(最長は --settle)
        if stage in "1256":  # 腕が動く段階だけ、関節の読みで補正(指だけ動かす 3、7 と、持ち上げ・上がる 4、8 は、しない)
            for it in range(args.refine):
                diff = {k: goal[k] - obs[k] for k in ARM_KEYS}
                if max(abs(v) for v in diff.values()) < args.refine_tol:
                    break  # 十分に届いている(先回しが効いていれば、ここで終わる。動きは途切れない)
                corr = dict(current)
                for k, v in diff.items():
                    corr[k] = current[k] + max(-3.0, min(3.0, v))  # 差を、今の指令に足す。一度に 3° まで
                print("  補正" + str(it + 1) + ": " + ", ".join(f"{k.split('.')[0][:9]} {v:+.1f}°" for k, v in diff.items()))
                move_to(robot, current, corr, 1.5)
                current = corr
                obs = wait_settled(robot, 1.0)
        if arm_stage and not args.no_feedforward:
            # この段階で「届いた」指令 = 最後に送った指令 + (目標 − 読み)。これと目標の差が、次回に足す補正量
            for k in ARM_KEYS:
                needed = current[k] + (goal[k] - obs[k]) - goal[k]
                hist = bias_table.setdefault(stage, {}).setdefault(k, [])
                hist.append(round(needed, 2))
                del hist[:-BIAS_KEEP]  # 古いものから捨てて、直近 BIAS_KEEP 個だけ残す。[頻度] 定番のスライス
            BIAS_FILE.parent.mkdir(exist_ok=True)
            BIAS_FILE.write_text(json.dumps(bias_table, indent=1))
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
            floor = plan.get("gripper_empty", plan["gripper_close"]) + args.grasp_margin  # plan.get(キー, 既定値): キーが無ければ既定値を返す(古い計画ファイル向け)。[頻度] 定番
            if args.auto and grip_after_close < floor:
                raise TrialAbort(f"grasp_missed: 指の読み {grip_after_close:.1f} が閉じ切りに近い(空で閉じた読み + 余裕 = {floor:.1f} 未満)。何も掴めていない")
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
                move_to(robot, current, {**plan["home"], "gripper.pos": current["gripper.pos"]}, max(3.0, plan.get("return_seconds", 8.0) / args.speed))  # 戻りも --speed に従う(ただし 3 秒以上)
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
