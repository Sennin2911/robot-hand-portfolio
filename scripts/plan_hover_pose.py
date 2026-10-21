"""【概要】sim で「手先を (x, y, z) に持っていく関節角」を IK で求め、その姿勢を実機に送るための値と、
実機の手先が来るはずの位置(測って比べるための予測値)を表示する。実機は動かさない。
流れ: 目標位置 → solve_ik(so101_ik.py)→ 関節角 [rad] → sim_to_real(sim_real.py)→ 実機の値(度、gripper は 0〜100)。
表示されたコマンドを順に実行すると、sim の画面で確認 → 実機で再現、ができる。

使い方(この .venv で実行):
    .venv/bin/python scripts/plan_hover_pose.py                    # 既定: 立方体(0.22, -0.12)の真上 0.10 m
    .venv/bin/python scripts/plan_hover_pose.py --xy 0.20 0.0 --z 0.10
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)。
sys.path.insert(0, str(ROOT / "src"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from hand_sim.sim_real import JOINTS, sim_to_real  # noqa: E402
from hand_sim.visual_servo import forward_tip, plan_hover  # noqa: E402

HOME = [0, -1.7, 1.6, 1.0, 0]  # IK の出発点: たたんだ姿勢(check_visual_servo.py と同じ)

parser = argparse.ArgumentParser()
parser.add_argument("--xy", type=float, nargs=2, default=[0.22, -0.12], help="手先の水平位置 x y [m]。既定は scene_vs.xml の立方体の位置")
parser.add_argument("--z", type=float, default=0.10, help="手先の高さ [m]。床(机)の面が 0")
parser.add_argument("--gripper", type=float, default=10.0, help="gripper の値 (実機の 0〜100 の尺度)。0 が閉。端(0, 100)の近くは real_send_pose.py が拒否するので避ける")
args = parser.parse_args()

model = mujoco.MjModel.from_xml_path(str(ROOT / "models" / "so101" / "scene.xml"))
q, ok = plan_hover(model, args.xy, args.z, HOME)
tip = forward_tip(model, q)  # 求めた関節角のときの手先の位置 [m](IK が本当に届いたかの確認にもなる)
print(f"IK: {'成功' if ok else '失敗(届かない目標。この姿勢は使わない)'} / 目標 {np.round([*args.xy, args.z], 3)} m → 手先 {np.round(tip, 4)} m")
if not ok:
    sys.exit(1)

# gripper は IK の対象外。実機の尺度(0〜100)で指定された値を、sim の rad に直して6関節ぶんにそろえる。
glo, ghi = model.joint("gripper").range
q6 = np.append(q, glo + args.gripper / 100.0 * (ghi - glo))
real = sim_to_real(model, q6)
values = [real[f"{n}.pos"] for n in JOINTS]
pose = " ".join(f"{v:.1f}" for v in values)
print("実機の値 (度/0-100):", {n: round(v, 1) for n, v in zip(JOINTS, values)})

# 実機で測って比べるための予測。モデルの原点(base)は机の上、肩の回転軸は原点から x 方向に少しずれた所にある。
scratch = mujoco.MjData(model)
mujoco.mj_forward(model, scratch)
pan_axis = scratch.xanchor[model.joint("shoulder_pan").id][:2]  # xanchor: 関節の回転軸の世界座標。[:2] は x, y だけ取る
d = tip[:2] - pan_axis
print("\n【実機で測って比べる値(sim の予測)】")
print(f"  手先の高さ(机の面から)        : {tip[2] * 100:.1f} cm")
print(f"  肩の回転軸から手先までの水平距離: {np.linalg.norm(d) * 100:.1f} cm")
print(f"  肩の回転軸から見た向き          : 正面(前)から {np.degrees(np.arctan2(d[1], d[0])):+.1f} 度(+ は左)")
print("  (手先 = gripperframe = 固定側の指(動かない方の指)の先端。画像は scripts/render_sim_pose.py で見られる)")

print("\n【手順】")
print("  1) sim の画面で確認:")
print(f"     .venv/bin/python scripts/view_sim_pose.py --pose {pose}")
print("  2) 実機で再現(lerobot の環境。腕を支える):")
print(f"     python {ROOT}/scripts/real_send_pose.py --pose {pose}")
