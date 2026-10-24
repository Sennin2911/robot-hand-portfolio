"""【概要】real_pick_place.py --record で記録した、掴む直前(段階 2: 降下)の手先の実際の位置と、カメラで検出した立方体の位置から、
「立方体が、手(gripperframe)の座標系で、どこにあるか」を出し、sim が想定した位置(CUBE_IN_SITE)との差を、軸ごとに見る。
手の座標: x = 指先の方向、y = 指の幅の方向、z = 指が開閉する方向(固定側 → 動く側が +)。
  差の x が大きい = 指先が、立方体の手前 / 奥で止まった(指の先で、立方体の端をつまむ)
  差の y が大きい = 指の幅の方向にずれた(立方体の面の端に、指が当たる)
  差の z が大きい = 開閉の方向にずれた(片方の指に寄る。固定側に寄ると、固定側の指が立方体の上に乗ることがある)
手の板は、右側の位置(カメラに映る)でしか測れない。立方体の高さは、中心を 15 mm とする。

使い方(この .venv で実行): .venv/bin/python scripts/analyze_grasp_offset.py --color green
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)

import numpy as np  # noqa: E402
from hand_sim.so101_grasp import CUBE_IN_SITE  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--color", required=True)
args = parser.parse_args()
plan = json.loads((ROOT / "outputs" / f"pick_plan_{args.color}.json").read_text())
samples = {s["label"]: s for s in json.loads((ROOT / "outputs" / "hand_samples.json").read_text())}
cube = np.array([plan["cube"]["x_m"], plan["cube"]["y_m"], 0.015])
print(f"立方体(計画時の検出): ({cube[0] * 1000:.0f}, {cube[1] * 1000:.0f}) mm。想定(CUBE_IN_SITE): {np.round(CUBE_IN_SITE * 1000, 1)} mm(手の座標 x, y, z)")
for stage in ("1", "2"):
    label = f"{plan['tag']}_{stage}"
    if label not in samples:
        print(f"{label}: 記録が無い")
        continue
    s = samples[label]
    hand, R = np.array(s["hand_pos_m"]), np.array(s["hand_R"])
    cube_in_hand = R.T @ (cube - hand)  # 世界座標の差を、手の座標に直す(R.T = 逆回転)
    if stage == "1":
        cube_in_hand = R.T @ ((cube + [0, 0, 0.05]) - hand)  # 段階 1 は、立方体の 5 cm 上(HOVER)を目標にしている
    diff = (cube_in_hand - CUBE_IN_SITE) * 1000
    print(f"{label}: 手先 ({hand[0] * 1000:.0f}, {hand[1] * 1000:.0f}, {hand[2] * 1000:.0f}) mm → 立方体は、手の座標で {np.round(cube_in_hand * 1000, 1)} mm。"
          f"想定との差 x = {diff[0]:+.1f}, y = {diff[1]:+.1f}, z = {diff[2]:+.1f} mm(大きさ {np.linalg.norm(diff):.1f} mm)")
