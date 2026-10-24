"""【概要】real_pick_place.py --record で記録した「指を閉じた直後(段階 3)」の、手の位置と立方体の位置から、閉じた指の中での立方体の位置(手の座標)を出し、
sim(CUBE_IN_HAND)との差を、calibration/grasp_site_offset.json に貯める。次に plan_pick_test.py を実行すると、貯めた値(中央値)が自動で使われる。
指を閉じると立方体は固定側の指に押しつけられるので、開閉の向き(z)の位置は、掴む前のずれによらず決まる(詳しくは grasp_calibration.py)。
x(指先の向き)と y(指の幅の向き)は、指では決まらない(y は立方体が滑る方向)ので、参考として表示するだけで、補正には使わない。
pick_color.sh が、掴んだあとに自動で実行する。手動で実行するなら、.venv/bin/python scripts/update_grasp_offset.py --color green
"""
import argparse
import json
import sys
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
new, why = add_sample(float(diff[2]), {"tag": plan["tag"], "time": s["time"], "cube_in_hand_mm": np.round(cube_in_hand * 1000, 1).tolist(),
                                       "hand_yaw_deg": plan["hand_yaw_deg"], "cube_area_px": s["cube_area_px"]})
if why:
    sys.exit(why)
n = len(json.loads((ROOT / "calibration" / "grasp_site_offset.json").read_text())["samples"])
print(f"校正に追加(合計 {n} 回)。次の計画で使うずれ: z = {load_site_offset()[2] * 1000:+.1f} mm(手の座標。固定側の指の面が、sim より +z 側にあるなら正)")
