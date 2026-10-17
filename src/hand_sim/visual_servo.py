"""【概要】視覚サーボを1周つなぐ(第2週-4)。
  overhead カメラで撮影 → find_red_centroid → pixel_to_world → solve_ik → ctrl に渡して腕を動かす。
vision.py(目)と so101_ik.py(関節角の計算)を、1本の流れにつなぐだけのファイル。
open-loop: 画像で位置を測って1回だけ動く(locate_cube → plan_hover → move_arm を1回ずつ呼ぶ)。
closed-loop: 動いたあとに撮り直し、「立方体」と「手先の青い目印」のずれを見て、ずれが縮むまで繰り返す(servo_closed_loop)。
"""
import mujoco
import numpy as np

from hand_sim.so101_ik import N_ARM, solve_ik
from hand_sim.vision import find_blue_centroid, find_red_centroid, pixel_to_world

CUBE_TOP_Z = 0.03  # 立方体の上面の高さ [m]。カメラは上面を見ているので、奥行きの計算にこれを使う。


def _locate(model, data, renderer, finder, z_obj, camera, cam_pos):
    """撮影 → finder で画素位置 → 世界座標 (x, y) [m]。locate_cube / locate_marker の共通部分。

    finder  : 画像から (u, v) を返す関数(find_red_centroid か find_blue_centroid)。
    z_obj   : 見つけた物体の(見えている面の)高さ [m]。奥行きの計算に使う。
    cam_pos : 「プログラムが信じているカメラの位置」(x, y, z) [m]。None なら XML の本当の値を使う。
              実機ではカメラの位置を測って求めるので、ずれる。わざとずらした値を渡して、そのずれに
              どれだけ耐えられるかを試すために引数にしてある。
    """
    renderer.update_scene(data, camera=camera)
    img = renderer.render()
    # finder(vision.py): 画像から目的の色の画素だけを選び、その範囲の中心の画素位置 (u, v) を返す。
    #   u = 横(左端0、右へ増える)、v = 縦(上端0、下へ増える)。単位は画素。見つからなければ None。
    uv = finder(img)
    if uv is None:
        return None
    cam = model.camera(camera)
    if cam_pos is None:
        cam_pos = cam.pos
    # pixel_to_world(vision.py) の引数の意味(画素 → 世界座標 (x, y) [m] に直す):
    #   uv[0], uv[1]        : 物体の画像上の位置 u, v [画素]。上の finder の結果。
    #   cam_pos             : カメラの世界座標 (x, y, z) [m]。XML の <camera pos=...> の値。
    #   model.cam_fovy[...] : カメラの縦方向の視野角 [度]。画素と実寸の対応(焦点距離)を決める。
    #   renderer.width/height : 画像の幅・高さ [画素]。画像の中心(= カメラの真下)を求めるのに使う。
    #   z_obj               : 物体の見えている面の高さ [m]。カメラからこの面までの距離が「奥行き」になる。
    return pixel_to_world(uv[0], uv[1], cam_pos, model.cam_fovy[cam.id], renderer.width, renderer.height, z_obj)


def locate_cube(model, data, renderer, camera="overhead", cam_pos=None):
    """カメラで撮影し、赤い立方体の世界座標 (x, y) [m] を返す。見つからなければ None。

    renderer : mujoco.Renderer。画像の幅・高さは renderer.width / renderer.height から分かる。
    """
    return _locate(model, data, renderer, find_red_centroid, CUBE_TOP_Z, camera, cam_pos)


def locate_marker(model, data, renderer, z_obj, camera="overhead", cam_pos=None):
    """カメラで撮影し、手先の青い目印の世界座標 (x, y) [m] を返す。見つからなければ None。

    z_obj : 目印の今の高さ [m]。関節角から計算できる(forward_tip)ので、呼ぶ側が渡す。
    """
    return _locate(model, data, renderer, find_blue_centroid, z_obj, camera, cam_pos)


# 青い目印を付ける位置(gripper リンクの座標系 [m])。
# 手先(gripperframe)そのものに付けると、真上のカメラからはグリッパ本体の陰になって、ほとんど見えない。
# 腕が立方体の上に来る姿勢では、gripper リンクの +x 方向が世界の「上」を向くので、そちら側に出して
# 上から見えるようにした。手先とは位置がずれるが、そのずれは関節角から計算できる(forward_sites)。
MARKER_POS = (0.035, 0.0, -0.04)


def load_model_with_marker(xml_path, radius=0.012):
    """シーンを読み込み、gripper リンクに青い球の目印(と、その位置を調べる site)を足した model を返す。

    手先をカメラで見つけるための目印。原本の XML は書き換えず、読み込んだあとにプログラムで足す。
    mujoco.MjSpec: XML を「編集できる設計図」として読み込む仕組み。[頻度] あまり使わない
    (今回のように、読み込んだあとにモデルへ部品を足したいときだけ使う)。通常は XML に直接書く。
    """
    spec = mujoco.MjSpec.from_file(str(xml_path))
    body = spec.body("gripper")  # gripperframe が付いているリンク
    body.add_geom(
        name="tip_marker", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[radius, 0, 0], pos=MARKER_POS,
        rgba=[0, 0, 1, 1],
        contype=0, conaffinity=0,  # 他の物体と衝突しない(見た目だけの目印にする)
        mass=1e-4,  # ほぼ重さ無し(腕の動きに影響させない)
    )
    # site: 位置だけを持つ目印(形も重さも無い)。目印の世界座標を、関節角から計算で出すために使う。
    body.add_site(name="tip_marker_site", pos=MARKER_POS, size=[0.002, 0, 0])
    return spec.compile()


def plan_hover(model, xy, hover_z, q_init, history=None):
    """立方体の (x, y) の真上 hover_z [m] に手先を持っていく関節角 (長さ5) を IK で求める。

    戻り値: (q, ok)。solve_ik は渡された data の qpos を書き換えるので、動かしている本体の data を
    汚さないよう、この関数の中で計算専用の data を新しく作って使う。[頻度] 「計算用の作業コピー」は定番
    """
    scratch = mujoco.MjData(model)
    return solve_ik(model, scratch, [xy[0], xy[1], hover_z], q_init, history=history)  # history: solve_ik の記録用リスト(任意)


def forward_sites(model, q):
    """関節角 q (長さ5) のときの、手先と青い目印の世界座標 (x, y, z) [m] を、腕を動かさず計算だけで求める(順運動学)。

    戻り値: (手先, 目印)。plan_hover と同じく計算専用の data を使う。
    IK が出した q が本当に目標に届くかの確認や、closed-loop で「手先と目印のずれ」を求めるのに使う。
    """
    scratch = mujoco.MjData(model)
    scratch.qpos[:N_ARM] = q
    mujoco.mj_forward(model, scratch)  # qpos から site の位置などを計算する(時間は進めない)
    tip = scratch.site_xpos[model.site("gripperframe").id].copy()
    marker = scratch.site_xpos[model.site("tip_marker_site").id].copy()
    return tip, marker


def forward_tip(model, q):
    """forward_sites の、手先だけ版。目印の無い(load_model_with_marker を通さない)model でも使える。"""
    scratch = mujoco.MjData(model)
    scratch.qpos[:N_ARM] = q
    mujoco.mj_forward(model, scratch)
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


def servo_closed_loop(model, data, renderer, hover_z, cam_pos=None, gain=0.8, max_iter=15, settle=0.5, tol=0.002,
                      first_move=3.0, on_step=None):
    """closed-loop の視覚サーボ。立方体の真上 hover_z [m] に、撮り直しながら手先を合わせていく。

    毎回の手順:
      1. 手先の位置 m を、立方体の位置 c(最初に1回だけ測った値)と同じカメラ・同じ計算で測る(m は手先の青い目印から求める)。
      2. ずれ e = c - m [m]。tol 未満なら終了。
      3. 目標位置を gain * e だけ動かし、IK で関節角を求め直して、settle 秒だけ動かす。
    最初の1回だけは、たたんだ姿勢から大きく動くので first_move 秒かけてゆっくり動かす。
    (短い時間で急に動かすと、腕が勢いで下に振れて立方体にぶつかり、弾いてしまったため)
    tol のめやす: 画素は整数単位なので、測定そのものに 1mm 前後の誤差がある。それより小さくしても収束しない。
    ポイント: c と m を「同じカメラ・同じ計算」で測るので、カメラの位置のずれなどは両方に同じように
    乗り、引き算(c - m)でほぼ打ち消される。open-loop は c だけを信じるので、ずれがそのまま残る。
    gain: 1回で埋めるずれの割合。1 だと測定誤差も全部拾って揺れるので、1 未満にして慎重に近づく。

    on_step: 関数を渡すと、腕を動かすたびに on_step(回数, IKの反復ごとの距離のリスト) を呼ぶ(様子を記録する用)。

    戻り値: 各回のずれの大きさ [m] のリスト(最後は動かした後の測定値)。
    """
    history = []
    # 立方体は、腕が視界に入る前(最初)に1回だけ測る。腕が近づくと、グリッパの陰で立方体が部分的に隠れ、
    # 毎回の測定が 5mm ほど揺れて、目標が往復して収束しなかったため。立方体は動かない前提の割り切り
    # (動く物体を追う場合は、隠れても位置が崩れない検出方法が別途必要になる)。
    cube_xy = locate_cube(model, data, renderer, cam_pos=cam_pos)
    if cube_xy is None:
        return history
    target_xy = np.array(cube_xy)  # 最初は open-loop と同じ: 立方体の推定位置を目標にする
    q = data.qpos[:N_ARM].copy()
    for i in range(max_iter):
        ik_hist = []
        q, _ok = plan_hover(model, target_xy, hover_z, q, history=ik_hist)
        move_arm(model, data, q, first_move if i == 0 else settle)
        if on_step is not None:
            on_step(i, ik_hist)
        # 手先と目印の位置関係は、いまの関節角から計算できる(実機ならエンコーダの値から。カメラは使わない)。
        tip_fk, marker_fk = forward_sites(model, data.qpos[:N_ARM])
        marker_xy = locate_marker(model, data, renderer, marker_fk[2], cam_pos=cam_pos)  # 目印の高さで奥行きを計算
        if marker_xy is None:
            break  # 目印が隠れて見えない。無理に動かさず、そこで止める
        # 目印は手先からずれた位置にあるので、「画像で測った目印の位置」に、手先と目印の差を足して、
        # 「画像から見た手先の位置」に直す。この差は関節角から出すので、カメラのずれの影響を受けない。
        hand_xy = np.asarray(marker_xy) + (tip_fk[:2] - marker_fk[:2])
        e = np.asarray(cube_xy) - hand_xy
        history.append(float(np.linalg.norm(e)))
        if history[-1] < tol:
            break
        target_xy = target_xy + gain * e
    return history
