"""【概要】視覚サーボを1周通して確認する。
立方体を色々な位置に置き、overhead カメラで見つけ → IK で関節角を求め → 腕を動かして、
手先が立方体の真上に来たか(水平方向のずれ)を調べる。全部通れば終了コード0、そうでなければ1。
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# 他のスクリプトと同じ形(詳しい意味は check_install.py を参照)。
sys.path.insert(0, str(ROOT / "src"))
if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "egl")  # ★mujoco を import する前に設定(理由は check_install.py)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from hand_sim.visual_servo import gripper_position, locate_cube, move_arm, plan_hover  # noqa: E402

W, H = 640, 480
HOVER_Z = 0.10  # 手先を止める高さ [m]。立方体の上面(0.03)より上にして、ぶつからないようにする。
TOL = 0.01  # 許容する水平方向のずれ [m] (1cm)
HOME = [0, -1.7, 1.6, 1.0, 0, 0]  # たたんだ姿勢。撮影時はこの姿勢(カメラの視界を遮らない)
PLACES = [(0.22, -0.12), (0.2, 0.0), (0.25, 0.12), (0.3, -0.05), (0.15, 0.1)]

model = mujoco.MjModel.from_xml_path(str(ROOT / "models" / "so101" / "scene_vs.xml"))
data = mujoco.MjData(model)
cube_adr = model.joint("cube_free").qposadr[0]  # 立方体の qpos の先頭番号(詳しくは check_vision.py)

n_ng = 0
with mujoco.Renderer(model, H, W) as renderer:
    for x, y in PLACES:
        mujoco.mj_resetData(model, data)
        data.qpos[:6] = HOME
        data.ctrl[:6] = HOME  # 初期状態から勝手に動かないよう、目標も HOME にそろえる
        data.qpos[cube_adr:cube_adr + 3] = [x, y, 0.015]
        mujoco.mj_forward(model, data)

        xy = locate_cube(model, data, renderer)
        if xy is None:
            n_ng += 1
            print(f"NG  置いた位置=({x:+.3f}, {y:+.3f}) 立方体が見つからなかった")
            continue
        q, ok_ik = plan_hover(model, xy, HOVER_Z, HOME[:5])
        move_arm(model, data, q)
        tip = gripper_position(model, data)
        gap = np.linalg.norm(tip[:2] - np.array([x, y]))  # 水平方向のずれ(z は無視)
        ok = ok_ik and gap < TOL and abs(tip[2] - HOVER_Z) < TOL
        n_ng += 0 if ok else 1
        print(f"{'OK' if ok else 'NG'}  立方体=({x:+.3f}, {y:+.3f})  カメラ推定=({xy[0]:+.3f}, {xy[1]:+.3f})  "
              f"手先=({tip[0]:+.3f}, {tip[1]:+.3f}, {tip[2]:.3f})  水平ずれ={gap * 1000:.1f} mm  IK成功={ok_ik}")

print("全部OK" if n_ng == 0 else f"{n_ng} 件NGです")
sys.exit(0 if n_ng == 0 else 1)
