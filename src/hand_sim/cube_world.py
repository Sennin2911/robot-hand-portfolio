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
