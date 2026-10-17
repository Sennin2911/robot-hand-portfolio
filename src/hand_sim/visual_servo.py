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
    # find_red_centroid(vision.py): 画像から「赤い画素」だけを選び、その範囲の中心の画素位置 (u, v) を返す。
    #   u = 横(左端0、右へ増える)、v = 縦(上端0、下へ増える)。単位は画素。赤が1つも無ければ None。
    uv = find_red_centroid(img)
    if uv is None:
        return None
    cam = model.camera(camera)
    # pixel_to_world(vision.py) の引数の意味(画素 → 世界座標 (x, y) [m] に直す):
    #   uv[0], uv[1]        : 赤い物体の画像上の位置 u, v [画素]。上の find_red_centroid の結果。
    #   cam.pos             : カメラの世界座標 (x, y, z) [m]。XML の <camera pos=...> の値。
    #   model.cam_fovy[...] : カメラの縦方向の視野角 [度]。画素と実寸の対応(焦点距離)を決める。
    #   renderer.width/height : 画像の幅・高さ [画素]。画像の中心(= カメラの真下)を求めるのに使う。
    #   CUBE_TOP_Z          : 物体の見えている面の高さ [m]。カメラからこの面までの距離が「奥行き」になる。
    return pixel_to_world(uv[0], uv[1], cam.pos, model.cam_fovy[cam.id], renderer.width, renderer.height, CUBE_TOP_Z)


def plan_hover(model, xy, hover_z, q_init):
    """立方体の (x, y) の真上 hover_z [m] に手先を持っていく関節角 (長さ5) を IK で求める。

    戻り値: (q, ok)。solve_ik は渡された data の qpos を書き換えるので、動かしている本体の data を
    汚さないよう、この関数の中で計算専用の data を新しく作って使う。[頻度] 「計算用の作業コピー」は定番
    """
    scratch = mujoco.MjData(model)
    return solve_ik(model, scratch, [xy[0], xy[1], hover_z], q_init)


def forward_tip(model, q):
    """関節角 q (長さ5) のときの手先の世界座標 (x, y, z) [m] を、腕を動かさずに計算だけで求める(順運動学)。

    plan_hover と同じく計算専用の data を使う。IK が出した q が、本当に目標に届く角度かを確かめるのに使う。
    """
    scratch = mujoco.MjData(model)
    scratch.qpos[:N_ARM] = q
    mujoco.mj_forward(model, scratch)  # qpos から site の位置などを計算する(時間は進めない)
    return gripper_position(model, scratch)


def move_arm(model, data, q, seconds=3.0):
    """関節角 q (長さ5) を ctrl(目標角度)に入れ、seconds 秒ぶんシミュレーションを進める。

    <position> アクチュエータなので、ctrl に角度 [rad] を入れるだけで内部で PD 制御して追従する。
    gripper(ctrl の6番目)は今回は動かさないので、今の値のままにする。
    """
    # ctrl(目標角度)は、ここで1回入れたら、ループの間ずっと同じ値。ステップごとに変えてはいない。
    # 一方、実際の関節角(data.qpos)はステップごとに変わる。アクチュエータが毎ステップ
    #   力 = kp*(ctrl - 今の角度) - kv*速度
    # を計算して関節を押すので、目標に向かって少しずつ近づいていく(PD制御)。
    # ヤコビアンを使うのは solve_ik の中(動かす前の計画)。「手先を目標に届かせる最終的な関節角 q」を
    # 探すために、仮の姿勢で何度も J を計算し直している。つまり役割は
    #   ヤコビアン(IK): 目標の手先位置 → 到達したい関節角 q を決める
    #   ctrl + mj_step : その q へ、実際に関節を動かす
    # ステップごとに ctrl を変えるのは、「動かしながら撮り直して、そのたびに IK をやり直す」closed-loop の場合。
    data.ctrl[:N_ARM] = q
    for _ in range(int(seconds / model.opt.timestep)):  # model.opt.timestep: 1ステップの時間 [秒]
        mujoco.mj_step(model, data)


def gripper_position(model, data):
    """手先(site gripperframe)の今の世界座標 (x, y, z) [m]。"""
    return data.site_xpos[model.site("gripperframe").id].copy()  # .copy(): 後で data が変わっても値が変わらないように
