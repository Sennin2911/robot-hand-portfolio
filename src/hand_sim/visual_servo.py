"""【概要】視覚サーボを1周つなぐ(第2週-4)。
  overhead カメラで撮影 → find_red_centroid → pixel_to_world → solve_ik → ctrl に渡して腕を動かす。
vision.py(目)と so101_ik.py(関節角の計算)を、1本の流れにつなぐだけのファイル。
"画像で位置を測って1回だけ動く" 方式(open-loop)。動いた後に撮り直して補正する本物のサーボ(closed-loop)は次の段階。
"""
import mujoco
import numpy as np

from hand_sim.so101_ik import N_ARM, solve_ik
from hand_sim.vision import find_red_centroid, pixel_to_world

CUBE_TOP_Z = 0.03  # 立方体の上面の高さ [m]。カメラは上面を見ているので、奥行きの計算にこれを使う。


def locate_cube(model, data, renderer, camera="overhead"):
    """カメラで撮影し、赤い立方体の世界座標 (x, y) [m] を返す。見つからなければ None。

    renderer : mujoco.Renderer。画像の幅・高さは renderer.width / renderer.height から分かる。
    """
    renderer.update_scene(data, camera=camera)
    img = renderer.render()
    uv = find_red_centroid(img)
    if uv is None:
        return None
    cam = model.camera(camera)
    return pixel_to_world(uv[0], uv[1], cam.pos, model.cam_fovy[cam.id], renderer.width, renderer.height, CUBE_TOP_Z)


def plan_hover(model, xy, hover_z, q_init):
    """立方体の (x, y) の真上 hover_z [m] に手先を持っていく関節角 (長さ5) を IK で求める。

    戻り値: (q, ok)。solve_ik は渡された data の qpos を書き換えるので、動かしている本体の data を
    汚さないよう、この関数の中で計算専用の data を新しく作って使う。[頻度] 「計算用の作業コピー」は定番
    """
    scratch = mujoco.MjData(model)
    return solve_ik(model, scratch, [xy[0], xy[1], hover_z], q_init)


def move_arm(model, data, q, seconds=3.0):
    """関節角 q (長さ5) を ctrl(目標角度)に入れ、seconds 秒ぶんシミュレーションを進める。

    <position> アクチュエータなので、ctrl に角度 [rad] を入れるだけで内部で PD 制御して追従する。
    gripper(ctrl の6番目)は今回は動かさないので、今の値のままにする。
    """
    data.ctrl[:N_ARM] = q
    for _ in range(int(seconds / model.opt.timestep)):  # model.opt.timestep: 1ステップの時間 [秒]
        mujoco.mj_step(model, data)


def gripper_position(model, data):
    """手先(site gripperframe)の今の世界座標 (x, y, z) [m]。"""
    return data.site_xpos[model.site("gripperframe").id].copy()  # .copy(): 後で data が変わっても値が変わらないように
