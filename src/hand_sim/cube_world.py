"""【概要】実機の画像から、赤・緑・青の立方体の、机の上の位置(世界座標 x, y [m])を求める部品。板のマーカー(ID 1〜6)の周りだけを探す。
detect_cubes_world.py と同じ考え方(色相で探す → 立方体の中心の高さの面に当てて換算する)を、record_hand_sample.py(掴んだ瞬間の記録)からも使えるようにしたもの。
"""
import cv2
import numpy as np

from hand_sim.fixed_board import board_marker_corners_world
from hand_sim.real_camera import pixel_to_world_plane_K, project_points_K
from hand_sim.vision import find_cube_pose_hsv

CUBE_M = 0.030  # 立方体の 1 辺


def search_roi(shape, cam_pos, cam_R, K, scale=2.2):
    """板の 6 個のマーカーの角が写る範囲を scale 倍に広げた四角形の内側が True の、画像と同じ大きさのマスクを返す(立方体を探す範囲)。"""
    pts = project_points_K(np.vstack(list(board_marker_corners_world().values())), cam_pos, cam_R, K).astype(np.float32)
    hull = cv2.convexHull(pts).reshape(-1, 2)
    hull = hull.mean(axis=0) + (hull - hull.mean(axis=0)) * scale
    roi = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(roi, hull.astype(np.int32), 1)
    return roi.astype(bool)


def locate_cube(undistorted_rgb, color, cam_pos, cam_R, K):
    """歪みを直した画像から、色 color の立方体の中心の世界座標 (x, y) [m] と、見えている面積(画素数)を返す。見つからなければ (None, 0)。
    立方体の中心の高さ(半辺 = 15 mm)の水平な面に、画素の位置を当てて換算する。"""
    roi = search_roi(undistorted_rgb.shape, cam_pos, cam_R, K)
    pose, mask = find_cube_pose_hsv(undistorted_rgb, color, roi, return_mask=True)
    if pose is None:
        return None, 0
    u, v, _yaw = pose
    xy = pixel_to_world_plane_K(u, v, cam_pos, cam_R, K, CUBE_M / 2)[:2]
    return xy, int(mask.sum())
