"""【概要】隣の立方体を避けて、掴む向きを選び直す仕組み(grasp_clearance.py、so101_grasp.plan_for_cube の neighbors)の答え合わせ(sim のみ。実機は動かさない)。
確かめること: (1) 隣が遠ければ、向きは今まで通り(選び直しなし)/ (2) 隣が指の開閉の向きの近くにあれば、向きを選び直して、ぶつからない向きになる /
(3) 立方体の四方を囲まれたら NeighborCollision / (4) 実機の試験 31、32 に近い配置(隣が 45 mm)で、どの向きを選ぶか。
使い方: MUJOCO_GL=egl .venv/bin/python scripts/check_neighbor_avoid.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")  # mujoco を import する「前」に設定する(import の時点で描画方式が決まるため)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from hand_sim.grasp_clearance import collides, hand_outline_mm  # noqa: E402
from hand_sim.so101_grasp import GRIP_OPEN, NeighborCollision, plan_for_cube  # noqa: E402

model = mujoco.MjModel.from_xml_path(str(ROOT / "models/so101/so101_new_calib.xml"))
PAD = np.array([0.14, 0.17])
fails = 0


def check(name, cond, detail=""):
    global fails
    fails += not cond
    print(f"{'OK  ' if cond else 'NG  '}{name}  {detail}")


def plan(cube_xy, yaw_deg, neighbors):
    info = {}
    poses, yaw = plan_for_cube(model, np.array([*cube_xy, 0.015]), PAD, np.radians(yaw_deg), neighbors=[(x, y, np.radians(a)) for x, y, a in neighbors], info=info)
    return poses, np.degrees(yaw), info


# 手の輪郭の大きさ(立方体の中心を原点にして、手の向きに合わせた座標で。指の開閉の向き z と、幅 y)を確かめる
poses, yaw, _ = plan((0.20, 0.0), 0, [])
o = hand_outline_mm(model, poses["grasp"], GRIP_OPEN) - np.array([200.0, 0.0])
d = np.array([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])  # 指の開閉の向き(世界の x, y)
along, across = o @ d, o @ np.array([-d[1], d[0]])
print(f"開いた手の輪郭(立方体の中心から): 開閉の向きに {along.min():.0f}〜{along.max():.0f} mm、幅の向きに {across.min():.0f}〜{across.max():.0f} mm(yaw {yaw:.0f}°)")

base_yaw = yaw
p, yw, info = plan((0.20, 0.0), 0, [(0.20, 0.12, 0)])
check("(1) 隣が遠い(120 mm)→ 選び直さない", info["skipped"] == 0 and abs(yw - base_yaw) < 1e-6, f"yaw {yw:.0f}°、隣との距離 {info['clearance_mm']:.0f} mm")

# 隣を、手の開閉の向きの、固定側・動く側に置く(中心間 45 mm = 立方体の間のすき間 15 mm)
for label, sign in (("固定側(-)", -1), ("動く側(+)", +1)):
    nb = (200 + sign * 45 * d[0], 0 + sign * 45 * d[1])
    p, yw, info = plan((0.20, 0.0), 0, [(nb[0] / 1000, nb[1] / 1000, 0)])
    hit, dist = collides(model, p["grasp"], GRIP_OPEN, [(nb[0] / 1000, nb[1] / 1000, 0)])
    print(f"    隣が {label}に 45 mm: 選んだ yaw {yw:.0f}°、飛ばした向き {info['skipped']}、隣との距離 {dist:.1f} mm")
    check(f"(2) 隣が {label}に近い → ぶつからない向きを選ぶ", not hit)

# (3) 四方を、近い隣で囲む
ring = [(0.20 + 0.045, 0.0, 0), (0.20 - 0.045, 0.0, 0), (0.20, 0.045, 0), (0.20, -0.045, 0)]
try:
    plan((0.20, 0.0), 0, ring)
    check("(3) 四方を囲まれたら、掴めない", False, "例外が出なかった")
except NeighborCollision as e:
    check("(3) 四方を囲まれたら、掴めない", True, "NeighborCollision")

# (4) 実機の試験 31 に近い配置(掴む (212, 56) mm、隣 (257, 54) mm)
p, yw, info = plan((0.212, 0.056), 0, [(0.257, 0.054, np.radians(10))])
print(f"(4) 試験 31 に近い配置: 選んだ yaw {yw:.0f}°、飛ばした向き {info['skipped']}、隣との距離 {info['clearance_mm']:.1f} mm")
print("全て OK" if not fails else f"NG が {fails} 件")
sys.exit(1 if fails else 0)
