"""【概要】実機の試験 1 回ぶんの成否を判定して、outputs/real_trials.csv に 1 行足す(run_real_trial.sh が最後に呼ぶ)。
判定に使うもの: 「掴む前」の検出(outputs/trial_before.json)、「置いたあと」の検出(outputs/trial_after.json。休めの姿勢に戻ってから撮る)、
実機の動きの結果(outputs/real_trial_result.json。real_pick_place.py --auto が書く)、計画(outputs/pick_plan_<色>.json)。
成功 = 立方体が、その色の置き場の中心から --tol-mm 以内にあり、ほかの立方体が --others-fail-mm(10 mm)以上動いていない。
ほかの立方体の動きは、成功率とは別に「接触」として記録する(contact 列。--touch-mm(5 mm)以上 = 軽い接触、--others-fail-mm 以上 = 明らかに動かした)。
失敗の分類: 検出失敗 / 届かない / 隣接 / 安全停止 / 把持失敗 / 運搬中の落下 / 置き損じ / 他を動かした / 最終位置不明(詳しくは classify)。
列は sim の試験(outputs/sorting_*.csv)に合わせてあるので、sim と実機の失敗の内訳を同じ集計で比べられる。
使い方: .venv/bin/python scripts/judge_real_trial.py --color green
        .venv/bin/python scripts/judge_real_trial.py --color green --stage-failed detect   (動かす前の段階で失敗したとき。run_real_trial.sh が使う)
"""
import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)

import numpy as np  # noqa: E402
from hand_sim.cube_world import PADS  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--color", required=True, choices=["red", "green", "blue"])
parser.add_argument("--stage-failed", choices=["detect", "plan", "neighbor"],
                    help="動かす前に失敗したとき。detect = 板が見えない・立方体が見つからない → 検出失敗。plan = 計画が作れない → 届かない(検出済みの色が無ければ検出失敗)。neighbor = 隣の立方体が近すぎて、どの向きでも掴めない → 隣接")
parser.add_argument("--tol-mm", type=float, default=15.0, help="置き場の中心から、この距離(mm)以内に立方体の中心があれば、置けたとみなす")
parser.add_argument("--touch-mm", type=float, default=5.0, help="ほかの立方体が、これ(mm)以上動いたら「軽い接触」(失敗にはしない。測定の誤差が 1〜2 mm あるので、これより小さい動きは見分けない)")
parser.add_argument("--others-fail-mm", type=float, default=10.0, help="ほかの立方体が、これ(mm)以上動いたら「明らかに動かした」(失敗の分類「他を動かした」)")
parser.add_argument("--note", default=os.environ.get("TRIAL_NOTE", ""), help="メモ(照明を変えた、など)を CSV の最後の列に残す")
args = parser.parse_args()

OUT = ROOT / "outputs"
# 記録先の CSV: 環境変数 TRIAL_LOG で切り替える(例: TRIAL_LOG=real_trials_phase6.csv)。無ければ従来の real_trials.csv。
# 試験番号は、その CSV の行数から数えるので、新しい CSV では 1 から始まる。os.environ.get(名前, 既定値): 環境変数を読む。[頻度] シェル経由で設定を渡す定番
LOG = OUT / os.environ.get("TRIAL_LOG", "real_trials.csv")
FIELDS = ["trial", "time", "color", "success", "fail_class", "true_x", "true_y", "true_yaw_deg", "final_x", "final_y", "final_err_mm", "moved_others_mm", "contact", "avoid",
          "cycle_s", "grip_after_close", "max_joint_err_deg", "board_markers", "board_reproj_px", "detail", "note",
          "stages_done", "ended", "start", "detect", "detect_reason"]  # 後ろの 5 列: 動きを最後まで終えたか(8 段階のうち何まで)、終わり方、始まった姿勢、検出の方法(連続仕分け)  # true_* は「掴む前に検出した位置 [m]」(sim の列名に合わせた)


def read_json(name):
    f = OUT / name
    return json.loads(f.read_text()) if f.exists() else None


def xy(d, color):
    """検出結果 d(detect_cubes_world.py の JSON)から、色 color の (x, y) [m]。無ければ None。"""
    return None if d is None or color not in d["cubes"] else np.array([d["cubes"][color]["x_m"], d["cubes"][color]["y_m"]])


def contact_level(moved_mm):
    """ほかの立方体の動き(mm)から、接触の程度を返す。成功率とは別に集計するための記録。"""
    return "明らかに動かした" if moved_mm >= args.others_fail_mm else ("軽い接触" if moved_mm >= args.touch_mm else "")


def classify(color):
    """(成功か, 失敗の分類, 詳しい説明, 追加の列の辞書) を返す。上から順に当てはめる。"""
    before, after, res = read_json("trial_before.json"), read_json("trial_after.json"), read_json("real_trial_result.json")
    plan = read_json(f"pick_plan_{color}.json")
    extra = {"board_markers": "", "board_reproj_px": "", "true_x": "", "true_y": "", "true_yaw_deg": "", "final_x": "", "final_y": "", "final_err_mm": "",
             "moved_others_mm": "", "contact": "", "cycle_s": "", "grip_after_close": "", "max_joint_err_deg": "",
             "stages_done": "", "ended": "", "start": "", "detect": "", "detect_reason": ""}
    extra["avoid"] = "隣を避けて向きを選んだ" if plan and plan.get("avoided_neighbor") else ""
    if before is not None:
        extra["board_markers"], extra["board_reproj_px"] = before.get("board_markers", ""), round(before.get("board_reproj_px", 0), 2)
        c = before["cubes"].get(color)
        if c:
            extra.update(true_x=round(c["x_m"], 4), true_y=round(c["y_m"], 4), true_yaw_deg=round(c["yaw_deg"], 1))
    if res is None:
        return False, "安全停止", "real_trial_result.json が無い(実機の実行が、結果を書く前に終わった)", extra
    # 動きをどこまで終えたか(段階 1〜8)。完了 = 8 段階とも終えた。中止 = 安全装置などで途中で止まり、真上から休めの姿勢へ戻った
    extra.update(stages_done=f"{res['last_stage']}/{res['total_stages']}",
                 ended="完了" if (res["abort_reason"] is None and res["last_stage"] >= res["total_stages"]) else "中止: " + str(res["abort_reason"] or f"段階 {res['last_stage']} で終わった"),
                 start=res.get("start_mode", ""), detect=res.get("detect_mode", ""), detect_reason=res.get("detect_reason", ""))
    extra.update(cycle_s=res["cycle_s"], grip_after_close=res["grip_after_close"] if res["grip_after_close"] is not None else "", max_joint_err_deg=res["max_joint_err_deg"])
    reason = res["abort_reason"]
    finished = reason is None and res["last_stage"] >= res["total_stages"]
    pos = xy(after, color)
    if pos is not None:
        extra.update(final_x=round(float(pos[0]), 4), final_y=round(float(pos[1]), 4))
        extra["final_err_mm"] = round(float(np.linalg.norm(pos - np.array(PADS[color]))) * 1000, 1)
    # ほかの立方体の動き(掴む前 → 置いたあと)。どちらでも見えている色だけ比べる
    moved = 0.0
    for other in ("red", "green", "blue"):
        if other != color and xy(before, other) is not None and xy(after, other) is not None:
            moved = max(moved, float(np.linalg.norm(xy(before, other) - xy(after, other))) * 1000)
    extra["moved_others_mm"] = round(moved, 1)
    extra["contact"] = contact_level(moved)

    if reason and reason.startswith(("grasp_missed", "grasp_wide")):  # startswith(タプル): どれかで始まれば True。grasp_wide = 角などで挟んだ(幅が広い)
        return False, "把持失敗", reason, extra
    if reason:
        return False, "安全停止", reason, extra
    if not finished:
        return False, "安全停止", f"段階 {res['last_stage']} で終わった(最後は {res['total_stages']})", extra
    if pos is None:
        return False, "最終位置不明", "置いたあとの画像で、その色の立方体が見つからなかった(隠れた、暗い、机の外へ落ちた)", extra
    d_pad = extra["final_err_mm"]
    d_orig = float(np.linalg.norm(pos - np.array([plan["cube"]["x_m"], plan["cube"]["y_m"]]))) * 1000
    if d_pad <= args.tol_mm:
        if moved >= args.others_fail_mm:
            return False, "他を動かした", f"ほかの立方体が {moved:.1f} mm 動いた", extra
        return True, "", "", extra
    if d_orig <= args.tol_mm:
        return False, "把持失敗", f"立方体が元の位置から {d_orig:.1f} mm しか動いていない(掴めていない)", extra
    if d_pad <= 3 * args.tol_mm:
        return False, "置き損じ", f"置き場の中心から {d_pad:.1f} mm(許容 {args.tol_mm:.0f} mm)", extra
    return False, "運搬中の落下", f"置き場から {d_pad:.1f} mm、元の位置から {d_orig:.1f} mm の、どちらでもない場所にある", extra


# 腕が動かずに終わった試験(実行が結果を書く前に終わった、開始の姿勢が休めの姿勢から離れていた、最初の確認で Ctrl+C、など)は、
# ロボットの動作の結果ではないので、記録しない。--stage-failed(動かす前に失敗)の場合は、下で従来どおり扱う。
if not args.stage_failed:
    _res = read_json("real_trial_result.json")
    if _res is None or _res.get("arm_moved") is False:
        print("\n【記録しない】腕が動かずに終わりました(real_trial_result.json が無い、または、腕に指令を送る前に止まった)。ロボットの動作の結果ではないので、試験として数えません。")
        sys.exit(0)

if args.stage_failed:
    before = read_json("trial_before.json")
    if args.stage_failed == "neighbor":
        ok, cls, detail, extra = False, "隣接", "隣の立方体が近すぎて、開いた指がどの向きでもぶつかる(隣を離して置き直す)", {}
        extra.update(true_x=round(float(xy(before, args.color)[0]), 4), true_y=round(float(xy(before, args.color)[1]), 4))
    elif args.stage_failed == "detect" or xy(before, args.color) is None:
        # 原因を、検出のログ(run_real_trial.sh が outputs/trial_detect.log に保存)から書く。ログと画像は、あとで見返せるよう日時つきで残す
        log_path = ROOT / "outputs" / "trial_detect.log"
        log_text = log_path.read_text() if log_path.exists() else ""
        if "板のマーカーが" in log_text and "個しか見えない" in log_text:
            reason = next(l for l in log_text.splitlines() if "個しか見えない" in l)  # next(ジェネレータ式): 条件に合う最初の行を取る。[頻度] 「最初の 1 つだけ欲しい」ときの定番
        elif f"{args.color:5s}: 見つからない" in log_text:
            reason = f"板は見えたが、{args.color} の立方体が見つからない(暗い・影・探す範囲の外・マーカーに隠れた、など)"
        else:
            reason = "原因不明(ログが無い)"
        detail = f"{reason}。ログ: outputs/detect_failures/"
        stamp = time.strftime("%Y%m%d_%H%M%S")
        (ROOT / "outputs" / "detect_failures").mkdir(exist_ok=True)
        for src, ext in ((log_path, "log"), (ROOT / "outputs" / "trial_before.png", "png")):
            if src.exists():  # 画像は、検出が最後まで進んだときだけある。shutil.copy: ファイルのコピー。[頻度] 定番
                shutil.copy(src, ROOT / "outputs" / "detect_failures" / f"{stamp}_{args.color}.{ext}")
        ok, cls, extra = False, "検出失敗", {}
    else:
        ok, cls, detail, extra = False, "届かない", "計画が作れなかった(遠すぎる、可動域の端、机に近い経路など。上の表示を参照)", {}
        extra.update(true_x=round(float(xy(before, args.color)[0]), 4), true_y=round(float(xy(before, args.color)[1]), 4))
else:
    ok, cls, detail, extra = classify(args.color)

if LOG.exists():
    with open(LOG, newline="") as f:
        header = next(csv.reader(f), [])
    if header != FIELDS:  # 古い列のままの CSV に、新しい列を足す。バックアップを残してから、全部の行を書き直す(古い行の新しい列は空)
        shutil.copy(LOG, LOG.with_name(LOG.stem + "_backup_before_columns.csv"))
        old_rows = list(csv.DictReader(open(LOG)))
        with open(LOG, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, restval="")
            w.writeheader()
            w.writerows(old_rows)
        print(f"(記録の列を増やした: {LOG.name}。元は {LOG.stem}_backup_before_columns.csv)")
n = 1
if LOG.exists():
    n = len(list(csv.DictReader(open(LOG)))) + 1  # 行数 + 1 が、今回の試験番号。csv.DictReader: 1 行目を列名として、各行を辞書で読む。[頻度] CSV を読む定番
row = {"trial": n, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "color": args.color, "success": ok, "fail_class": cls, "detail": detail, "note": args.note, **extra}
new_file = not LOG.exists()
with open(LOG, "a", newline="") as f:  # "a": 末尾に追記する。newline="": csv モジュールに改行を任せる決まり文句
    w = csv.DictWriter(f, fieldnames=FIELDS, restval="")  # restval: 辞書に無い列は空欄にする
    if new_file:
        w.writeheader()
    w.writerow(row)

# 試験ごとの検出画像と検出結果(向き・IoU を含む)を残す。あとから「その試験で、向きが正しく検出されていたか」を確かめるため
# (trial_before.* は、次の試験で消えるので、残さないと、原因を調べられない)
(OUT / "trial_archive").mkdir(exist_ok=True)
for _name in ("trial_before.png", "trial_before.json", "trial_after.png"):
    if (OUT / _name).exists():
        shutil.copy(OUT / _name, OUT / "trial_archive" / f"{LOG.stem}_{n:03d}_{_name}")

print(f"\n試験 {n}: {args.color}  →  {'成功' if ok else '失敗(' + cls + ')'}  {detail}")
if extra.get("final_err_mm") not in (None, ""):
    print(f"  置いたあとの位置: ({extra['final_x'] * 1000:.0f}, {extra['final_y'] * 1000:.0f}) mm、置き場の中心から {extra['final_err_mm']} mm")
if extra.get("moved_others_mm") not in (None, ""):
    print(f"  今回のほかの立方体の動き: {extra.get('moved_others_mm', '')} mm" + (f"({extra['contact']})" if extra.get("contact") else "(接触なし)"))
rows = [r for r in csv.DictReader(open(LOG)) if not r["note"].startswith("無効")]  # メモが「無効」で始まる試験(装置の都合で測れなかったもの)は、通算に数えない
ok_n = sum(r["success"] == "True" for r in rows)
print(f"  通算: {ok_n} / {len(rows)} 成功(記録: {LOG.relative_to(ROOT)})")
print(f"  通算の、ほかの立方体への接触: 軽い {sum(r['contact'] == '軽い接触' for r in rows)} 回、明らかに動かした {sum(r['contact'] == '明らかに動かした' for r in rows)} 回")
