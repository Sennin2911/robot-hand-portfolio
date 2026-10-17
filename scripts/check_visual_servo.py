"""【概要】視覚サーボを1周通して確認する。
立方体を色々な位置に置き、overhead カメラで見つけ → IK で関節角を求め → 腕を動かして、
手先が立方体の真上に来たか(水平方向のずれ)を調べる。全部通れば終了コード0、そうでなければ1。
ずれの合計を、3つの段階の誤差に分けて表示する(視覚 / IK / 腕の動き)。
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
from hand_sim.visual_servo import forward_tip, gripper_position, locate_cube, move_arm, plan_hover  # noqa: E402

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
        gap = np.linalg.norm(tip[:2] - np.array([x, y]))  # 水平方向のずれ(z は無視)。3つの段階の合計
        ok = ok_ik and gap < TOL and abs(tip[2] - HOVER_Z) < TOL
        n_ng += 0 if ok else 1

        # ずれの内訳。各段階が「入力として受け取った位置」と「出した位置」の距離を、mm で出す。
        #   視覚 : 本当の位置 (x, y)       と カメラ推定 (xy)         の距離
        #   IK   : IK への目標 (xy, HOVER_Z) と q から計算した手先位置 の距離(q が目標に届いているか)
        #   動き : q から計算した手先位置   と 実際に動かした後の手先  の距離(PD 制御・重力などの残り)
        # 3つを足しても gap と一致するとは限らない(向きが逆なら打ち消し合う)。大きい段階を見つける目安。
        tip_ik = forward_tip(model, q)
        e_vision = np.linalg.norm(np.asarray(xy) - np.array([x, y]))
        e_ik = np.linalg.norm(tip_ik - np.array([xy[0], xy[1], HOVER_Z]))
        e_move = np.linalg.norm(tip - tip_ik)
        print(f"{'OK' if ok else 'NG'}  立方体=({x:+.3f}, {y:+.3f})  手先=({tip[0]:+.3f}, {tip[1]:+.3f}, {tip[2]:.3f})  "
              f"水平ずれ(合計)={gap * 1000:.1f} mm  内訳: 視覚={e_vision * 1000:.2f} IK={e_ik * 1000:.2f} 動き={e_move * 1000:.2f} mm")

print("全部OK" if n_ng == 0 else f"{n_ng} 件NGです")
sys.exit(0 if n_ng == 0 else 1)
