"""【概要】real_pick_place.py --record で記録した「指を閉じた直後(段階 3)」の、手の位置と立方体の位置から、閉じた指の中での立方体の位置(手の座標)を出し、
sim(CUBE_IN_HAND)との差を、calibration/grasp_site_offset.json に貯める。次に plan_pick_test.py を実行すると、貯めた値(中央値)が自動で使われる。
指を閉じると立方体は固定側の指に押しつけられるので、開閉の向き(z)の位置は、掴む前のずれによらず決まる(詳しくは grasp_calibration.py)。
x(指先の向き)と y(指の幅の向き)は、指では決まらない(y は立方体が滑る方向)ので、参考として表示するだけで、補正には使わない。
pick_color.sh が、掴んだあとに自動で実行する。手動で実行するなら、.venv/bin/python scripts/update_grasp_offset.py --color green
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)

import numpy as np  # noqa: E402
from hand_sim.grasp_calibration import add_sample, load_site_offset  # noqa: E402
from hand_sim.so101_grasp import CUBE_IN_HAND  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--color", required=True)
args = parser.parse_args()

plan = json.loads((ROOT / "outputs" / f"pick_plan_{args.color}.json").read_text())
samples = {s["label"]: s for s in json.loads((ROOT / "outputs" / "hand_samples.json").read_text())}
label = f"{plan['tag']}_3"
if label not in samples:
    sys.exit(f"{label} の記録が無い(段階 3 まで進んでいない、または記録に失敗した)。校正は更新しない")
s = samples[label]


def log_row(**extra):
    """測定の結果を、校正に使う・使わないにかかわらず、calibration/grasp_log.csv に 1 行足す(立方体までの距離で、ずれが変わるかを後で調べるため)。
    csv.writer: 表(CSV ファイル)に行を書く標準の部品。open(..., "a"): 末尾に追記するモード。[頻度] 記録を貯めるときの定番"""
    log = ROOT / "calibration" / "grasp_log.csv"
    row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "tag": plan["tag"], "plan_x_mm": round(plan["cube"]["x_m"] * 1000, 1), "plan_y_mm": round(plan["cube"]["y_m"] * 1000, 1),
           "reach_mm": round(float(np.hypot(plan["cube"]["x_m"] + 0.0388, plan["cube"]["y_m"])) * 1000, 1),  # 肩の軸からの水平距離(原点は肩の軸の 38.8 mm 後ろ)
           "hand_yaw_deg": round(plan["hand_yaw_deg"], 1), "board_markers": s["board_markers"], "board_reproj_px": round(s["board_reproj_px"], 2),
           "cube_found": s.get("cube_pos_m") is not None, "cube_area_px": s["cube_area_px"], "dx": "", "dy": "", "dz": "", "used": False, "note": ""}
    row.update(extra)
    new_file = not log.exists()
    with open(log, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new_file:
            w.writeheader()
        w.writerow(row)


if s.get("cube_pos_m") is not None:  # 校正に使えない測定(板の誤差が大きいなど)でも、手の座標でのずれは記録に残す
    _cube, _hand, _R = np.array(s["cube_pos_m"]), np.array(s["hand_pos_m"]), np.array(s["hand_R"])
    _d = (_R.T @ (_cube - _hand) - CUBE_IN_HAND) * 1000
    log_row(dx=round(float(_d[0]), 1), dy=round(float(_d[1]), 1), dz=round(float(_d[2]), 1))
else:
    log_row(note="cube not found")
if s.get("cube_pos_m") is None:
    sys.exit(f"{label}: 手の板は読めたが、立方体が見つからなかった。校正は更新しない")
if s["board_reproj_px"] > 1.5:
    sys.exit(f"{label}: 板の再投影誤差が {s['board_reproj_px']:.2f} 画素と大きい(1.5 まで)。校正は更新しない")
planned = np.array([plan["cube"]["x_m"], plan["cube"]["y_m"]])
cube = np.array(s["cube_pos_m"])
moved = np.linalg.norm(cube[:2] - planned)
if moved > 0.030:  # 閉じると、立方体は固定側へ最大 16 mm ほど滑る。それより大きく動いていたら、別の物を見ている
    sys.exit(f"{label}: 立方体が、計画時の位置から {moved * 1000:.0f} mm も離れている(30 mm まで)。別の物を検出した、または掴みに失敗したので、校正は更新しない")

hand, R = np.array(s["hand_pos_m"]), np.array(s["hand_R"])
cube_in_hand = R.T @ (cube - hand)  # 世界座標の差を、手の座標に直す(R.T = 逆回転)
diff = cube_in_hand - CUBE_IN_HAND
print(f"閉じた指の中の立方体(手の座標): {np.round(cube_in_hand * 1000, 1)} mm。sim {np.round(CUBE_IN_HAND * 1000, 1)} mm → 差 x = {diff[0] * 1000:+.1f}, y = {diff[1] * 1000:+.1f}, z = {diff[2] * 1000:+.1f} mm"
      f"(立方体の見えた面積 {s['cube_area_px']} 画素)")
_rows = list(csv.DictReader(open(ROOT / "calibration" / "grasp_log.csv")))  # 今の行(いま足した最後の行)に、「校正に使った」印を付ける
new, why = add_sample(float(diff[2]), {"tag": plan["tag"], "time": s["time"], "cube_in_hand_mm": np.round(cube_in_hand * 1000, 1).tolist(),
                                       "hand_yaw_deg": plan["hand_yaw_deg"], "cube_area_px": s["cube_area_px"]})
if why:
    sys.exit(why)
n = len(json.loads((ROOT / "calibration" / "grasp_site_offset.json").read_text())["samples"])
print(f"校正に追加(合計 {n} 回)。次の計画で使うずれ: z = {load_site_offset()[2] * 1000:+.1f} mm(手の座標。固定側の指の面が、sim より +z 側にあるなら正)")

if not why:  # 上で足した行の used を True に書き直す(CSV は、読み込み → 最後の行だけ変更 → 全体を書き直し、で更新する)
    _rows[-1]["used"] = True
    with open(ROOT / "calibration" / "grasp_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(_rows[0]))
        w.writeheader()
        w.writerows(_rows)
