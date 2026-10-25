"""【概要】立方体を掴みに降りるとき、開いた指が、隣の立方体にぶつからないかを調べる部品(掴む向きを選び直すために使う)。
やること: 掴む姿勢(関節角)で手を開いたときの、机の近く(立方体の高さまで)にある手の部分を、真上から見た形(輪郭)にする。
その輪郭から margin [m] 以内に、隣の立方体が入っていれば「ぶつかる」とみなす。
立方体は正方形なので、手の向きは 90° ごとに 4 通りある。so101_grasp.plan_for_cube が、ぶつからない向きを選ぶ。
(障害物回避や経路計画ではなく、「掴む向きを 4 通りから選ぶ」だけの工夫。降りる動きは、真上からの垂直のままで変えない。)
"""
import cv2
import mujoco
import numpy as np

CUBE_SIDE = 0.030  # 立方体の 1 辺 [m]
Z_MAX = 0.033  # 机からこの高さより下にある手の部分だけを見る [m]。立方体の高さ(30 mm)+ 3 mm。これより上の部分は、隣の立方体の上を通る
MARGIN = 0.005  # 手の輪郭と隣の立方体の間に、最低限あけたい距離 [m]。検出の誤差(1〜2 mm)と、動きのずれの余裕


def hand_outline_mm(model, q_arm, grip_open):
    """掴む姿勢 q_arm(関節角 5 つ)で、指を grip_open(rad)まで開いたときの、机の近く(Z_MAX より低い)にある手の部分を、
    真上から見た凸な輪郭にして返す(世界座標の x, y を mm にした、N×2 の float32 配列)。低い部分が無ければ None。
    手の部分 = 腕のリンクのメッシュ(group 2。plan_pick_test.py の「机からの高さ」の確認と同じ選び方)の頂点。"""
    data = mujoco.MjData(model)
    data.qpos[:6] = np.r_[q_arm, grip_open]
    mujoco.mj_forward(model, data)
    pts = []
    for g in range(model.ngeom):
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or model.geom_group[g] != 2 or model.geom_bodyid[g] < 3:
            continue
        adr, num = model.mesh_vertadr[model.geom_dataid[g]], model.mesh_vertnum[model.geom_dataid[g]]
        world = model.mesh_vert[adr:adr + num] @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]  # メッシュの頂点を、世界座標に直す
        pts.append(world[world[:, 2] < Z_MAX][:, :2])
    pts = np.vstack(pts) if pts else np.zeros((0, 2))
    if len(pts) < 3:
        return None
    return cv2.convexHull((pts * 1000).astype(np.float32)).reshape(-1, 2)  # cv2.convexHull: 点の集まりを囲む、最小の凸な多角形


def clearance_mm(outline_mm, neighbors):
    """手の輪郭 outline_mm と、隣の立方体 neighbors(各要素 (x [m], y [m], yaw [rad]))の間の、いちばん近い距離 [mm] を返す(正 = 離れている、負 = 重なっている)。
    隣の立方体の正方形の中に、格子状に点を取って、輪郭までの距離を測る(cv2.pointPolygonTest: 輪郭の外なら負、中なら正の距離を返す)。"""
    if outline_mm is None or not len(neighbors):
        return np.inf
    contour = outline_mm.reshape(-1, 1, 2).astype(np.float32)
    g = np.linspace(-CUBE_SIDE / 2, CUBE_SIDE / 2, 7) * 1000
    grid = np.array([(a, b) for a in g for b in g])
    best = np.inf
    for x, y, yaw in neighbors:
        c, s = np.cos(yaw), np.sin(yaw)
        pts = grid @ np.array([[c, s], [-s, c]]) + np.array([x, y]) * 1000  # 格子を yaw だけ回して、隣の立方体の位置へ
        for p in pts:
            best = min(best, -cv2.pointPolygonTest(contour, (float(p[0]), float(p[1])), True))  # 輪郭の外なら +(離れている距離)になるよう、符号を反転
    return float(best)


def collides(model, q_arm, grip_open, neighbors, margin=MARGIN):
    """開いた手が、隣の立方体 neighbors に margin 以内に近づくなら True。(True か False, 最も近い距離 [mm]) を返す。"""
    d = clearance_mm(hand_outline_mm(model, q_arm, grip_open), neighbors)
    return d < margin * 1000, d
