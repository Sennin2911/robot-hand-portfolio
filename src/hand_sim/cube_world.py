"""【概要】実機の画像から、赤・緑・青の立方体の、机の上の位置(世界座標 x, y [m])を求める部品。板のマーカー(ID 1〜6)の周りだけを探す。
detect_cubes_world.py と同じ考え方(色相で探す → 立方体の中心の高さの面に当てて換算する)を、record_hand_sample.py(掴んだ瞬間の記録)からも使えるようにしたもの。
"""
import cv2
import numpy as np

from hand_sim.fixed_board import board_marker_corners_world
from hand_sim.real_camera import pixel_to_world_plane_K, project_points_K
from hand_sim.vision import find_cube_pose_hsv

CUBE_M = 0.030  # 立方体の 1 辺
PADS = {"green": (0.14, 0.17), "red": (0.07, 0.17), "blue": (0.21, 0.12)}  # 色ごとの置き場の中心(世界座標 x, y [m])。plan_pick_test.py(運ぶ先)と judge_real_trial.py(置けたかの判定)が共有する


def marker_mask(shape, cam_pos, cam_R, K, margin_px=4):
    """板のマーカーの黒い四角(まわりに margin_px 画素の余裕)が True の、画像と同じ大きさのマスクを返す。立方体を探す範囲から除くために使う。
    除く理由: マーカーの黒に、カメラの色の偏りで青みが残ると、「青い立方体」と誤検出されたため
    (立方体がマーカーの上に乗ると、マーカーが隠れて、板の検出が 4 個未満になりうるが、乗せてもよい運用なので、余裕は 4 画素にとどめる。余裕が大きいと、マーカーの上の立方体が欠けて、形の判定で落ちた)。"""
    mask = np.zeros(shape[:2], np.uint8)
    for quad in board_marker_corners_world().values():
        cv2.fillConvexPoly(mask, project_points_K(quad, cam_pos, cam_R, K).astype(np.int32), 1)
    return cv2.dilate(mask, np.ones((2 * margin_px + 1, 2 * margin_px + 1), np.uint8)).astype(bool)  # cv2.dilate: 白い部分(True)を、まわりに太らせる


WORKSPACE_X_M = (0.07, 0.40)  # 立方体を探す範囲の、前後(世界座標 x [m])。0.07 より後ろは、ロボットの土台(橙色の部品などが、「赤」と誤検出された)
WORKSPACE_Y_M = (-0.30, 0.30)  # 左右(世界座標 y [m])
EXTRA_X_M = (0.09, 0.34)  # 探す範囲に必ず入れる、前後の端(search_hull)。立方体が枠の縁にまたがらないため
EXTRA_Y_M = (-0.20, 0.20)  # 同、左右の端


def workspace_mask(shape, cam_pos, cam_R, K):
    """机の上の、立方体を置いてよい範囲(世界座標の長方形。立方体の中心の高さ 15 mm の面)が True の、画像と同じ大きさのマスクを返す。
    ロボット本体や、机の外の物の色を、立方体と間違えないために、探す範囲を、この中に絞る。"""
    (x0, x1), (y0, y1) = WORKSPACE_X_M, WORKSPACE_Y_M
    rect = np.array([[x0, y0, 0.015], [x1, y0, 0.015], [x1, y1, 0.015], [x0, y1, 0.015]])
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, project_points_K(rect, cam_pos, cam_R, K).astype(np.int32), 1)
    return mask.astype(bool)


def search_hull(shape, cam_pos, cam_R, K, scale=1.3, extend_down=0):
    """立方体を探す範囲の外枠(画像上の凸多角形の頂点、画素)を返す。板の 6 個のマーカーの角と、3 つの置き場の四角(1 辺 9 cm、高さ 15 mm)を全部含む最小の凸な枠を、
    中心から scale 倍に広げたもの。置き場も含める理由: 試験の最後に、置き場の上の立方体の位置を測るため(以前は、板のマーカーだけを囲んだので、カメラを遠ざけると置き場が範囲の外になった)。
    extend_down: 枠の下側の頂点だけを、画像の下方向へ、この画素数だけ広げる(既定 0)。"""
    pads = [np.array([[x - h, y - h, 0.015], [x + h, y - h, 0.015], [x + h, y + h, 0.015], [x - h, y + h, 0.015]]) for (x, y) in PADS.values() for h in (0.045,)]
    pts = project_points_K(np.vstack(list(board_marker_corners_world().values()) + pads), cam_pos, cam_R, K).astype(np.float32)
    hull = cv2.convexHull(pts).reshape(-1, 2)
    hull = hull.mean(axis=0) + (hull - hull.mean(axis=0)) * scale
    hull[hull[:, 1] > hull[:, 1].mean(), 1] += extend_down  # 画像の下側の頂点(v が平均より大きい点)だけを、下へ動かす
    # 板の左手前(世界座標で y が負、x が小さい側)も、立方体を置く範囲。ここは、マーカーの角から離れていて、枠の縁が立方体を横切り、塊が欠けた
    # (試験 1、32、33: (113, −150) mm。IoU 0.39 で、向きが 0° でも 45° でも 67〜70° と出て、斜めに掴んで失敗した)。
    # 枠は、画像の中心から広げるので、場所ごとの余裕が不ぞろいになる。だから、広げたあとの枠に、世界座標で決めた点(立方体の上面の高さ 30 mm)を足して、凸な枠を作り直す。
    extra = project_points_K(np.array([[x, y, 0.03] for x in EXTRA_X_M for y in EXTRA_Y_M]), cam_pos, cam_R, K).astype(np.float32)
    hull = cv2.convexHull(np.vstack([hull.astype(np.float32), extra])).reshape(-1, 2)  # np.vstack: 配列を縦に積む。[頻度] 点を足すときの定番
    hull[:, 1] = np.clip(hull[:, 1], 0, shape[0] - 1)  # 画像の外へはみ出さないようにする
    return hull


def search_roi(shape, cam_pos, cam_R, K, scale=1.3, extend_down=0):
    """search_hull の枠の内側が True の、画像と同じ大きさのマスクを返す(立方体を探す範囲)。マーカーの部分は除き、机の上の置ける範囲(workspace_mask)に絞る。"""
    roi = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(roi, search_hull(shape, cam_pos, cam_R, K, scale, extend_down).astype(np.int32), 1)
    return roi.astype(bool) & ~marker_mask(shape, cam_pos, cam_R, K) & workspace_mask(shape, cam_pos, cam_R, K)  # ~ : True と False の反転(マーカーの部分を除く)


def white_balance_by_paper(img_rgb, cam_pos, cam_R, K):
    """板の白い紙が無彩色(白〜灰色)になるように、画像の色を補正して、(補正した画像, 色ごとの倍率 [R, G, B]) を返す。
    なぜ要るか: 部屋の照明(昼の窓の光、夕方の光、電灯)で、カメラの色が青・赤紫などに偏ると、立方体の色相が、判定の範囲(vision.py の _HSV_RANGES)から外れる
    (緑のブロックが青緑になり、紙のマーカーの白が「青」と判定された)。カメラのホワイトバランスの設定は、緑 ↔ 赤紫の偏りを直せないので、ソフトで直す。
    板の位置は、すでに分かっている(カメラの位置から、板の画像上の範囲が計算できる)ので、その範囲の「明るい半分(= 紙。黒いマーカーを除く)」の中央値を、白の基準にする。
    いちばん明るい色の倍率を 1 にして、ほかの色を持ち上げる(全体が暗くならないように)。"""
    pts = project_points_K(np.vstack(list(board_marker_corners_world().values())), cam_pos, cam_R, K).astype(np.float32)
    inside = np.zeros(img_rgb.shape[:2], np.uint8)
    cv2.fillConvexPoly(inside, cv2.convexHull(pts).astype(np.int32), 1)
    pix = img_rgb[inside.astype(bool)].astype(float)  # 板の範囲の画素(N×3)。bool の配列(マスク)で、True の画素だけを取り出す書き方。[頻度] 画像処理の定番
    bright = pix.sum(axis=1)
    paper = pix[bright >= np.median(bright)]  # 明るい半分 = 紙(黒いマーカーと、立方体の暗い面を外す)
    ref = np.median(paper, axis=0)
    gains = ref.max() / np.maximum(ref, 1.0)
    return np.clip(img_rgb.astype(float) * gains, 0, 255).astype(np.uint8), gains


def locate_cube(undistorted_rgb, color, cam_pos, cam_R, K):
    """歪みを直した画像から、色 color の立方体の中心の世界座標 (x, y) [m] と、見えている面積(画素数)を返す。見つからなければ (None, 0)。
    立方体の中心の高さ(半辺 = 15 mm)の水平な面に、画素の位置を当てて換算する。"""
    roi = search_roi(undistorted_rgb.shape, cam_pos, cam_R, K)
    balanced, _gains = white_balance_by_paper(undistorted_rgb, cam_pos, cam_R, K)
    pose, mask = find_cube_pose_hsv(balanced, color, roi, return_mask=True)
    if pose is None:
        return None, 0
    u, v, _yaw = pose
    xy = pixel_to_world_plane_K(u, v, cam_pos, cam_R, K, CUBE_M / 2)[:2]
    return xy, int(mask.sum())


def fit_cube_yaw(mask, cam_pos, cam_R, K, center_xy, search_mm=8.0):
    """見えている立方体の塊(mask)に、立方体の 3 次元の形を重ねて合わせ、机の上での向き yaw [度、0〜90]・中心 (x, y) [m]・重なり(IoU)を返す。
    なぜ要るか: 立方体の上面と側面は同じ色で、明るさでは分けられない(明るい上位 40% の画素から向きを出す旧方式は、同じ画像でも 41° ↔ 67° と揺れた)。
    形から合わせれば、側面が見えている分も計算に入るので、上面と側面を取り違えない。
    やり方: 向き(1° 刻み)と、中心の位置(center_xy のまわり ±search_mm、2 mm 刻み)を変え、立方体の 8 つの角を画像に映して、囲んだ形(凸包)と mask の IoU
    (重なった面積 ÷ 合わせた面積。1 に近いほどよく合う)が最大のものを選ぶ。戻り値は (yaw, (x, y), IoU, contrast)。contrast は向きの確からしさ(下を参照)。[頻度] 「モデルを映して、見えたものと重ねる」は、物体の姿勢を推定する定番の考え方(analysis-by-synthesis)。
    center_xy: 中心の最初の見積もり(世界座標 [m])。mask の外接長方形の中心を、立方体の半分の高さに当てた値でよい。"""
    v_idx, u_idx = np.nonzero(mask)
    u0, u1, v0, v1 = u_idx.min() - 30, u_idx.max() + 30, v_idx.min() - 30, v_idx.max() + 30  # 計算を軽くするため、塊のまわりだけ切り出す
    target = mask[max(v0, 0):v1, max(u0, 0):u1].astype(np.uint8)
    off = np.array([max(u0, 0), max(v0, 0)], dtype=float)
    area_t = target.sum()
    h = CUBE_M / 2
    corners = np.array([[sx * h, sy * h, sz * h + h] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])  # 立方体の中心を原点にした 8 つの角(底面が z = 0)

    def iou(yaw_rad, cx, cy):
        c, s = np.cos(yaw_rad), np.sin(yaw_rad)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        uv = project_points_K(corners @ R.T + [cx, cy, 0.0], cam_pos, cam_R, K) - off
        canvas = np.zeros_like(target)
        cv2.fillConvexPoly(canvas, cv2.convexHull(uv.astype(np.float32)).astype(np.int32), 1)  # 8 つの角を囲む凸な形を塗る
        inter = (canvas & target).sum()
        return inter / max(canvas.sum() + area_t - inter, 1)

    best = (-1.0, 0.0, center_xy[0], center_xy[1])
    steps = np.arange(-search_mm, search_mm + 0.1, 2.0) / 1000
    best_per_yaw = []  # 向きごとの、最良の IoU(向きの確からしさを出すため)
    for yaw in range(0, 90, 3):  # 粗い探索(3° 刻み)
        best_here = 0.0
        for dx in steps:
            for dy in steps:
                score = iou(np.radians(yaw), center_xy[0] + dx, center_xy[1] + dy)
                best_here = max(best_here, score)
                if score > best[0]:
                    best = (score, yaw, center_xy[0] + dx, center_xy[1] + dy)
        best_per_yaw.append(best_here)
    score, yaw0, cx, cy = best
    for yaw in range(int(yaw0) - 3, int(yaw0) + 4):  # 細かい探索(1° 刻み)
        for dx in np.arange(-0.002, 0.0021, 0.001):
            for dy in np.arange(-0.002, 0.0021, 0.001):
                sc = iou(np.radians(yaw), cx + dx, cy + dy)
                if sc > best[0]:
                    best = (sc, yaw, cx + dx, cy + dy)
    score, yaw, cx, cy = best
    # 向きの確からしさ(contrast): 最良の IoU から、全部の向きの最良 IoU の中央値を引いた値。山がはっきりしていれば大きく、なだらか(どの向きでも同じくらいに重なる)なら 0 に近い。
    # 0 に近いと、塊が少し崩れるだけで、別の向きが勝つ(緑の暗い立方体で、同じ画像から 74° と 55° が出た)。np.median: 中央値。[頻度] 外れ値に強い「ふつうの値」として定番
    contrast = float(score - np.median(best_per_yaw))
    return float(yaw % 90), (cx, cy), float(score), contrast
