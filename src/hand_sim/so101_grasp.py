"""【概要】SO-101 で立方体を「真上から掴んで、持ち上げて、別の場所へ置く」動作(sim)。
流れ(状態機械): 指を開く → 立方体の真上へ → 降下 → 指を閉じる → 持ち上げる → 置き場の真上へ → 降下 → 指を開く → 上がる。
手先を真下に向け、指の開閉の向きを机と平行にして、立方体を両側から挟む。そのための姿勢は solve_ik_pose(so101_ik.py)で求める。
詳しい考え方は docs/grasp_basics.md を参照。単位は m(長さ)、rad(角度)。
"""
import mujoco
import numpy as np

from hand_sim.so101_ik import N_ARM, solve_ik_pose

# --- 指(gripper)の値。gripper の関節角 [rad]。大きいほど開く(sim の可動域は −0.17〜1.75)---
GRIP_OPEN = 0.5      # 開いた角度。固定側の指との隙間が約 46 mm で、一辺 30 mm の立方体より広い
GRIP_CLOSE = -0.1    # 閉じる指令。立方体に当たって途中で止まる(位置アクチュエータは、目標との差に応じた力で押し続ける)

# --- 掴む姿勢 ---
# 掴む位置: 立方体の中心が、gripperframe(固定側の指の先端)から見てどこにあるか(gripperframe の座標系 [m])。
#   x = −0.005: 指の先より 5 mm 手前(指の根もと側)。小さくしすぎると、指の先が机にぶつかる。
#   z = +0.029: 指が開閉する向きの位置。2 本の指の間の、どこに立方体を入れるか。
#     固定側の指の内側の面は z ≈ −0.002、開いた(0.5 rad)動く側の指の面は z ≈ +0.045。中央(z ≈ +0.022)ではなく少し動く側に寄せたのは、
#     試して決めた値: 立方体の位置が ±14 mm ずれても、4 方向すべてで掴めた(0.017 だと、固定側へ 3 mm ずれただけで、
#     固定側の指が立方体の上に乗って失敗した)。閉じるとき、動く側の指が立方体を固定側へ押して、中心は約 +0.011 に落ち着く。
CUBE_IN_SITE = np.array([-0.005, 0.0, 0.029])
# 閉じたあとの、立方体の中心の位置(gripperframe の座標系 [m])。閉じると、動く側の指が立方体を固定側の指へ押しつけるので、
# z 方向は「固定側の指の面 + 半辺」に落ち着く(= 掴む前の位置のずれによらない。実測: 0.0148)。x は指の先からの深さのまま。
# 置くときは、立方体の中心(この位置)が置き場の中心に来るように、手の位置を決める(掴む前の位置のままだと 12 mm ずれた)。
CUBE_IN_HAND = np.array([-0.0054, 0.0, 0.0148])
HOVER = 0.05  # 掴む姿勢から、真上へ上がる高さ [m]。立方体(高さ 3 cm)の上を通れる余裕。これ以上高くすると、手首(wrist_flex)が可動域の端(95°)に届く
CUBE_HALF = 0.015  # 立方体の一辺の半分 [m](scene_vs.xml の size と同じ)

# --- 指の当たり判定の作り直し(load_grasp_model)で使う値 ---
# 指の先に近い「細い板(ブレード)」だけを切り出す範囲。gripperframe の座標系 [m]。
#   固定側: x が −0.06 より先(指の先の側)で、板の幅(y)が ±0.015 以内の頂点。
#   動く側: x が −0.05 より先で、幅が ±0.015 以内の頂点。
BLADE_X_MIN = {"wrist_roll_follower_so101_v1": -0.06, "moving_jaw_so101_v1": -0.05}
BLADE_Y_MAX = 0.015
BLADE_FRICTION = [1.0, 0.005, 1e-4]  # 滑り摩擦、ねじり摩擦、転がり摩擦(MuJoCo の既定値)。3D プリントの樹脂のつもり
# 指で挟む力の上限 [N·m](gripper の関節のトルク)。元のモデルの上限 3.35 N·m だと、指が立方体(20 g)に深くめり込む
# (実測: 深さ 11 mm、そのまま押し出されて掴めない)。0.3 N·m にすると、立方体を片側 約 6 N で挟み、めり込みは 0.2 mm 以下。
# 立方体の重さ(0.2 N)を、摩擦で十分に支えられる。
GRIP_TORQUE_MAX = 0.3
# 接触の硬さ(solref の「時定数」[秒])。小さいほど硬い。既定は 0.02 で、軽い物体(20 g)との接触だと、とても柔らかくなり、めり込む。
# 時間刻み(0.002 秒)の 2 倍以上でないと、計算が不安定になる。
CONTACT_TIMECONST = 0.004
# 手の目印(較正用)。gripperframe の座標系 [m] での位置と、球の半径。手を真下に向けたとき、x が負 = 手首の側 = 上、z = 指の開閉の向きに出た位置になる。
# 位置は、手の上や横に付けて試した 6 通りのうち、手を真下に向けた 12 の姿勢の全部で、真上のカメラから見えた位置(腕や手の陰に隠れにくい位置)。
MARKER_IN_SITE = np.array([-0.04, 0.0, 0.075])
MARKER_RADIUS = 0.009
# 肘(body lower_arm、肘の軸が原点)と手首(body wrist、手首の軸が原点)に付ける目印の、各リンクの座標系 [m] での位置。
# 関節の軸は、各リンクの座標系の z 軸なので、z 方向(= 軸に沿って、リンクの横へ)に出した位置にして、真上から見えるようにする。
# 色は、目印ごとに変える(紫 = 肘、オレンジ = 手首、手は マゼンタ)。実機でも、目印ごとに色や ID を変えて、どの関節の目印かを見分ける。
LINK_MARKERS = {"lower_arm": (0.0, 0.0, 0.045), "wrist": (0.0, 0.0, 0.045)}
LINK_MARKER_RGBA = {"lower_arm": [0.55, 0.0, 1.0, 1], "wrist": [1.0, 0.5, 0.0, 1]}


def load_grasp_model(xml_path, cube_geoms=("cube_geom",), hand_marker=False, link_markers=False):
    """シーンを読み込み、gripper の指の当たり判定を「指の先の細い板」だけに作り直した model を返す。

    なぜ要るか: MuJoCo は、メッシュどうしの接触を「凸包」(でこぼこを埋めて、出っ張りだけを結んだ形)で判定する。
    固定側の指は、手のひらと指の先がつながった 1 つのメッシュで、凸包にすると、指と手のひらの間のすき間が埋まってしまう
    (実測: 立方体を掴む位置で、凸包が立方体に 7.6 mm めり込んだ)。見た目のメッシュは、そのままにしておく。
    やること: 元の指のメッシュの当たり判定を切り、指の先の板だけの頂点から、新しい当たり判定の形(これも凸包)を作って足す。
    板は、ほぼ凸な形なので、凸包にしても形が変わらない。
    cube_geoms: 掴む対象の立方体の geom の名前(接触を硬くする対象)。立方体が複数あるシーンでは、全部の名前を渡す。
    hand_marker: True なら、手(gripper)にマゼンタ(赤紫)の小さな球の目印と、その位置を調べる site "hand_marker_site" を足す。
                 カメラの較正で、「手の目印が画像のどこに写るか」と「関節角から計算した目印の位置」を比べるために使う。
                 立方体の色(赤・緑・青)とは違う色にして、色検出が取り違えないようにしてある。当たり判定は無い(見た目だけ)。
    mujoco.MjSpec: XML を「編集できる設計図」として読み込む仕組み。[頻度] あまり使わない(読み込んだあとにモデルを編集したいときだけ)。
    """
    base = mujoco.MjModel.from_xml_path(str(xml_path))  # 頂点の位置を調べるための、編集前の model
    data = mujoco.MjData(base)
    mujoco.mj_forward(base, data)
    site_pos = data.site_xpos[base.site("gripperframe").id]
    site_R = data.site_xmat[base.site("gripperframe").id].reshape(3, 3)

    spec = mujoco.MjSpec.from_file(str(xml_path))
    for geom in spec.geoms:
        if geom.meshname in BLADE_X_MIN and geom.group == 3:  # group 3 = 当たり判定用(group 2 は見た目用)
            geom.contype, geom.conaffinity = 0, 0  # 他の物体と衝突しない(見た目用の geom と同じ扱い)

    # 元の当たり判定 geom(group 3)の頂点から、板の部分を切り出して、body の座標系で保存する。
    for gi in range(base.ngeom):
        mesh_name = base.mesh(base.geom_dataid[gi]).name if base.geom_dataid[gi] >= 0 else ""
        if mesh_name not in BLADE_X_MIN or base.geom_group[gi] != 3:
            continue
        mid = base.geom_dataid[gi]
        a, n = base.mesh_vertadr[mid], base.mesh_vertnum[mid]
        # メッシュの頂点(メッシュ自身の座標系)→ 世界座標 → gripperframe の座標系(板の範囲を決めるため)
        world = data.geom_xpos[gi] + base.mesh_vert[a:a + n] @ data.geom_xmat[gi].reshape(3, 3).T
        in_site = (world - site_pos) @ site_R
        keep = (in_site[:, 0] > BLADE_X_MIN[mesh_name]) & (np.abs(in_site[:, 1]) < BLADE_Y_MAX)
        # さらに、その body の座標系に直す(body に付ける geom の頂点は、body の座標系で書く)。
        body_id = base.geom_bodyid[gi]
        local = (world[keep] - data.xpos[body_id]) @ data.xmat[body_id].reshape(3, 3)
        name = f"blade_{mesh_name}"
        mesh = spec.add_mesh(name=name)
        mesh.uservert = local.flatten().tolist()  # uservert: 頂点の座標を、x y z x y z ... と並べた 1 次元のリストで渡す
        spec.body(base.body(body_id).name).add_geom(
            name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name, group=3, friction=BLADE_FRICTION,
            solref=[CONTACT_TIMECONST, 1])
    if hand_marker:
        # 目印の位置: gripperframe の座標系で MARKER_IN_SITE。手を真下に向けたとき、手の側面から上へ(真上のカメラから見える位置に)出る。
        # spec に足す geom の位置は body(gripper)の座標系で書くので、編集前の model で、body の座標系に直しておく。
        body = base.body("gripper")
        world = site_pos + site_R @ MARKER_IN_SITE
        local = (world - data.xpos[body.id]) @ data.xmat[body.id].reshape(3, 3)
        gripper = spec.body("gripper")
        gripper.add_geom(name="hand_marker", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[MARKER_RADIUS, 0, 0], pos=local.tolist(),
                         rgba=[1, 0, 1, 1], contype=0, conaffinity=0, mass=1e-6)  # 当たり判定なし、ほぼ重さなし
        gripper.add_site(name="hand_marker_site", pos=local.tolist(), size=[0.002, 0, 0])
    if link_markers:
        # 肘と手首の関節の近くにも、手の目印と違う色の球を付ける(腕のずれを、関節ごとに分けて測るため)。位置は、各リンクの座標系 LINK_MARKERS。
        for body_name, local in LINK_MARKERS.items():
            link = spec.body(body_name)
            link.add_geom(name=f"{body_name}_marker", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[MARKER_RADIUS, 0, 0], pos=list(local),
                          rgba=LINK_MARKER_RGBA[body_name], contype=0, conaffinity=0, mass=1e-6)
            link.add_site(name=f"{body_name}_marker_site", pos=list(local), size=[0.002, 0, 0])
    for name in cube_geoms:
        spec.geom(name).solref = [CONTACT_TIMECONST, 1]  # 立方体側も硬くする(接触は、2つの geom の値を混ぜて決まる)
    spec.actuator("gripper").forcerange = [-GRIP_TORQUE_MAX, GRIP_TORQUE_MAX]  # 挟む力の上限
    return spec.compile()


# 待機の姿勢(関節角 5 つ [rad])。腕をたたみ、shoulder_pan を −90°(左へ)回して、立方体を置く範囲(x ≥ 0.14 m)から外す。
# カメラで立方体を探すとき、腕が立方体を隠さないための姿勢。(pan = 0 のたたんだ姿勢だと、手が x = 0.2 m に来て、視野の真ん中を隠した)
HOME_Q = np.array([-1.57, -1.7, 1.6, 1.0, 0.0])


# 試験用: 腕の関節が、指令どおりでなく、この角度 [rad] だけずれて動くようにする(実機の関節のオフセットの再現)。通常は 0。
# move_to が ctrl(目標角度)に足す。sim の腕が「実機のように、sim のモデルとずれた腕」になるので、
# 較正や掴む動作が、腕のずれにどれだけ耐えるかを調べられる。set_arm_error(度) で設定する。
ARM_ERROR = np.zeros(N_ARM)
# 試験用: 制御側が、「腕のずれ」を知っているとして、指令から引く補正 [rad]。実機の sim_real.py の OFFSET_DEG(指令から引く)に当たる。通常は 0。
ARM_COMP = np.zeros(N_ARM)


def set_arm_error(deg):
    """腕の 5 関節のずれ(度、長さ 5 のリスト。0 を渡すと、ずれなし)を設定する。"""
    global ARM_ERROR  # global: 関数の外の変数を書き換える宣言。[頻度] 試験用の設定でだけ使う
    ARM_ERROR = np.radians(np.broadcast_to(np.asarray(deg, dtype=float), N_ARM)).copy()  # broadcast_to: 1 つの数を、5 つぶんに広げる


def set_arm_compensation(deg):
    """制御側が使う、腕のずれの補正(度、長さ 5。指令から、この角度を引く)を設定する。腕のずれ(ARM_ERROR)を推定して、打ち消すのに使う。"""
    global ARM_COMP
    ARM_COMP = np.radians(np.broadcast_to(np.asarray(deg, dtype=float), N_ARM)).copy()


def go_home(model, data, seconds=2.0, on_step=None):
    """腕を待機の姿勢(HOME_Q)に戻す。指は開く。"""
    move_to(model, data, HOME_Q, GRIP_OPEN, seconds, on_step)


def grasp_rotation(yaw):
    """掴むときの gripperframe の向き(3×3 の回転行列)。列 = gripperframe の x、y、z 軸が、世界座標でどちらを向くか。

    x 軸(指の先へ向かう向き)= 真下 (0, 0, −1)。
    z 軸(指が開閉する向き)  = 机と平行で、世界の x 軸から yaw [rad] だけ回した向き(cos yaw, sin yaw, 0)。
    y 軸は、右手系になるよう外積で決まる(y = z × x)。
    """
    x = np.array([0.0, 0.0, -1.0])
    z = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    y = np.cross(z, x)  # np.cross: 外積。2つのベクトルに垂直なベクトルが得られる。[頻度] 3次元の向きの計算でよく使う
    return np.stack([x, y, z], axis=1)  # np.stack(..., axis=1): ベクトルを列として並べて 3×3 にする。[頻度] たまに使う


def grasp_site_pose(cube_pos, yaw, in_site=CUBE_IN_SITE):
    """立方体の中心 cube_pos [m] を掴むときの、gripperframe の位置と向き (pos, R) を返す。
    gripperframe から見た立方体の位置は in_site(掴む前は CUBE_IN_SITE、掴んだあとは CUBE_IN_HAND)なので、
    逆に、立方体の位置から gripperframe の位置を引き算で出す。"""
    R = grasp_rotation(yaw)
    pos = np.asarray(cube_pos) - R @ in_site  # R @ v: gripperframe の座標系のベクトル v を、世界座標に直す
    return pos, R


def plan_pick_place(model, cube_pos, place_xy, yaw=0.0, q_init=(0, -1.7, 1.6, 1.0, 0), site_offset=(0.0, 0.0, 0.0), hold_offset=(0.0, 0.0, 0.0)):
    """掴む → 置く の途中の姿勢(関節角 5 つ)を、IK で全部求めて辞書で返す。解けなかった姿勢の名前は、例外で知らせる。

    site_offset: 実機の指の面の位置が sim と違うぶん [m]。手の座標(gripperframe の x, y, z)で、CUBE_IN_SITE と CUBE_IN_HAND の両方に足す
      (指に対する立方体の位置を、そのぶんずらす)。実機で掴んだ立方体の位置から求めた値(grasp_calibration.py)を渡す。既定は 0(sim どおり)。

    hold_offset: 「掴んだあと」(CUBE_IN_HAND)にだけ足すずれ [m]。指のクッションは、固定側の面を動く側へ押し出す(厚み − つぶれ)ので、
      掴んだ立方体は sim より動く側に寄る。掴む前の位置(CUBE_IN_SITE)に足すと、開いた指の動く側に近づきすぎるので、別にしてある。

    返す姿勢: above(立方体の真上)、grasp(掴む)、place_above(置き場の真上)、place(置く)。
    前の姿勢の解を、次の IK の出発点にする(近い姿勢から探すと、関節が急に大きく動く別の解に飛びにくい)。
    解く順は grasp → above → place → place_above(理由は下のコメント)。
    """
    cube_pos = np.asarray(cube_pos, dtype=float)
    site_offset = np.asarray(site_offset, dtype=float)
    place_pos = np.array([place_xy[0], place_xy[1], cube_pos[2] + 0.003])  # 置くときは 3 mm 浮かせて、そっと放す
    scratch = mujoco.MjData(model)  # 計算専用の data(動かしている本体の data を汚さないため)
    q = np.array(q_init, dtype=float)
    poses = {}

    def solve(name, target, in_site):
        nonlocal q  # nonlocal: 外側の関数の変数 q を、この中から書き換える宣言。[頻度] あまり使わない(ここでは「前の解」を引き継ぐため)
        pos, R = grasp_site_pose(target, yaw, in_site)
        q, ok = solve_ik_pose(model, scratch, pos, R, q)
        if not ok:
            raise ValueError(f"IK が解けない姿勢: {name}(立方体の中心 {np.round(target, 3)} m)")
        poses[name] = q

    # 先に「低い姿勢(掴む・置く)」を解き、その解を出発点にして「真上の姿勢」を解く。
    # 真上の姿勢は、折りたたんだ姿勢から遠く、そこから直接だと解けないことがある。低い姿勢のほうが、手が向きを作りやすい。
    # 掴む前(grasp、above)は立方体が CUBE_IN_SITE にあり、掴んだあと(place、place_above)は CUBE_IN_HAND にある。
    in_site, in_hand = CUBE_IN_SITE + site_offset, CUBE_IN_HAND + site_offset + np.asarray(hold_offset, dtype=float)
    solve("grasp", cube_pos, in_site)
    solve("above", cube_pos + [0, 0, HOVER], in_site)
    solve("place", place_pos, in_hand)
    solve("place_above", place_pos + [0, 0, HOVER], in_hand)
    return poses


class NeighborCollision(ValueError):
    """IK が解ける向きは有るが、どの向きでも、開いた指が隣の立方体にぶつかるときに投げる。(ValueError の仲間にして、既存の except ValueError でも受けられる)"""


def plan_for_cube(model, cube_pos, place_xy, cube_yaw=0.0, site_offset=(0.0, 0.0, 0.0), neighbors=(), info=None, hold_offset=(0.0, 0.0, 0.0), grip_open=GRIP_OPEN):
    """立方体が机の上で cube_yaw [rad] だけ回っているとき、指を立方体の面に合わせた掴み方を探して、(姿勢の辞書, 指の向き yaw) を返す。

    指の開閉の向きは、立方体の面に垂直でないと、角(かど)に当たって立方体が回ってしまう。
    立方体は 90° ごとに同じ形なので、候補は cube_yaw + 90° × k。手首の回転(wrist_roll)や他の関節の可動域に収まる候補を、
    yaw が 0 に近い順(= 手首の回転が小さい順)に試して、IK が解けた最初のものを使う。

    neighbors: 隣にある、ほかの立方体 [(x [m], y [m], yaw [rad]), ...]。指定すると、開いた指が隣にぶつかる向きは飛ばして、次の候補を使う
      (grasp_clearance.py。手の向きは 90° ごとに 4 通りあり、指の開閉の向きに隣が無い向きを選ぶ)。どの向きでもぶつかるなら NeighborCollision。
    hold_offset: plan_pick_place と同じ(掴んだあとの位置だけのずれ。クッション用)。
    grip_open: 隣との距離を調べるときの、指を開く角度 [rad]。実機で広く開くとき(クッション付き)は、その角度を渡す(輪郭が広がるため)。既定は GRIP_OPEN。
    info: 辞書を渡すと、選んだ経過を書き込む({"skipped": 飛ばした向きの数, "clearance_mm": 選んだ向きの、隣との距離})。呼び出し側が、記録や表示に使う。
    """
    from hand_sim.grasp_clearance import collides  # 隣を調べるときだけ使うので、ここで import する(sim 側の読み込みを軽く保つ)

    candidates = [cube_yaw + k * np.pi / 2 for k in range(-4, 5)]
    # np.arctan2(sin, cos): 角度を −π〜π の範囲に直す。[頻度] 角度の周期(360°)をそろえるときの定番
    candidates.sort(key=lambda a: abs(np.arctan2(np.sin(a), np.cos(a))))
    skipped, solved_any, tried = 0, False, set()
    for yaw in candidates:
        key = round(float(np.arctan2(np.sin(yaw), np.cos(yaw))), 4)  # 360° ずれた同じ向きを、2 回試さないため
        if key in tried:
            continue
        tried.add(key)
        try:
            poses = plan_pick_place(model, cube_pos, place_xy, yaw, site_offset=site_offset, hold_offset=hold_offset)
        except ValueError:  # IK が解けなかった。次の候補へ
            continue
        solved_any = True
        if len(neighbors):
            hit, d = collides(model, poses["grasp"], grip_open, neighbors)
            if hit:
                skipped += 1
                continue
            if info is not None:
                info["clearance_mm"] = d
        if info is not None:
            info["skipped"] = skipped
        return poses, yaw
    if solved_any:
        raise NeighborCollision(f"IK が解ける向きはあるが、どの向きでも、開いた指が隣の立方体にぶつかる(立方体 {np.round(cube_pos, 3)} m)")
    raise ValueError(f"どの向きでも IK が解けない(立方体 {np.round(cube_pos, 3)} m、向き {np.degrees(cube_yaw):.0f}°)")


def move_to(model, data, q_arm, grip, seconds, on_step=None):
    """腕の目標角度 q_arm(長さ5)と指の目標 grip [rad] へ、seconds 秒かけてなめらかに動かす。

    ctrl(目標角度)を、今の値から目標へ少しずつ変えていく(なめらかな S 字の補間)。
    いきなり目標を入れると、PD 制御が最大の力で飛びつくので、掴む動作では急な動きを避ける。
    on_step: 1 ステップごとに呼ぶ関数(動画のフレーム保存や、ログの記録に使う)。None なら何もしない。
    """
    start = data.ctrl[: N_ARM + 1].copy()
    goal = np.append(np.asarray(q_arm) - ARM_COMP + ARM_ERROR, grip)  # 腕の目標: 補正を引いて(制御側)、試験用のずれ(通常は 0)を足す(実際の腕)
    n = max(1, int(seconds / model.opt.timestep))  # model.opt.timestep: 1 ステップの時間 [秒]
    for i in range(1, n + 1):
        s = i / n
        s = s * s * (3 - 2 * s)  # smoothstep: 0→1 を、始めと終わりがゆっくりになるようにする曲線。[頻度] 動きをなめらかにするときの定番
        data.ctrl[: N_ARM + 1] = start + (goal - start) * s
        mujoco.mj_step(model, data)
        if on_step is not None:
            on_step()


def set_cube_pose(model, data, pos, yaw=0.0, cube="cube"):
    """立方体(body の名前 cube)を pos [m] に、机と平行に yaw [rad] だけ回して置く(freejoint の qpos を直接書き換える)。
    freejoint の名前は「<cube>_free」(例: cube → cube_free、cube_red → cube_red_free)という約束。
    freejoint の qpos は 7 つ: 位置 (x, y, z) と、向きのクォータニオン (w, x, y, z)。
    クォータニオン: 3次元の回転を 4 つの数で表す方法。z 軸まわりに yaw 回すなら (cos(yaw/2), 0, 0, sin(yaw/2))。[頻度] ロボットでは定番"""
    adr = model.jnt_qposadr[model.joint(f"{cube}_free").id]  # この関節の値が、qpos の何番目から始まるか
    data.qpos[adr:adr + 7] = [*pos, np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
    mujoco.mj_forward(model, data)


def pick_and_place(model, data, cube_pos, place_xy, cube_yaw=0.0, on_step=None, cube="cube"):
    """立方体(机の上で cube_yaw [rad] だけ回っている)を掴んで、置き場へ運ぶ。
    data は、腕が初期姿勢(全関節 0)で、立方体が cube_pos にある状態(set_cube_pose で作る)。

    戻り値: 動作の各段階の記録の辞書(段階ごとの立方体の位置)。成功の判定は、呼び出し側(check スクリプト)で行う。
    """
    poses, _ = plan_for_cube(model, cube_pos, place_xy, cube_yaw)
    cube_id = model.body(cube).id
    log = {}

    def cube_xyz():
        return data.xpos[cube_id].copy()  # data.xpos: 各 body の世界座標の位置。copy() しないと、あとの更新で値が変わる

    # 開いて、立方体の真上へ(最初だけ、遠い姿勢から動くので時間をかける)
    move_to(model, data, poses["above"], GRIP_OPEN, 3.0, on_step)
    move_to(model, data, poses["grasp"], GRIP_OPEN, 1.5, on_step)  # 降下
    log["before_close"] = cube_xyz()
    move_to(model, data, poses["grasp"], GRIP_CLOSE, 1.0, on_step)  # 閉じる
    log["after_close"] = cube_xyz()
    move_to(model, data, poses["above"], GRIP_CLOSE, 1.5, on_step)  # 持ち上げる(指は閉じたまま)
    log["lifted"] = cube_xyz()
    move_to(model, data, poses["place_above"], GRIP_CLOSE, 2.0, on_step)  # 置き場の真上へ
    log["carried"] = cube_xyz()
    move_to(model, data, poses["place"], GRIP_CLOSE, 1.5, on_step)  # 降下
    move_to(model, data, poses["place"], GRIP_OPEN, 1.0, on_step)  # 開いて放す
    move_to(model, data, poses["place_above"], GRIP_OPEN, 1.5, on_step)  # 上がる
    log["placed"] = cube_xyz()
    return log
