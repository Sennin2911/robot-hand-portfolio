"""【概要】実機の立方体の色検出の位置が、定規で測った位置と合うかを調べる。
手順: ウェブカメラで 1 枚撮る → 机の板(ID 1〜6)からカメラの位置を求める → 色検出(HSV)で、各立方体の画像の中心を出す
→ 作者が測った位置(板の十字から、紙の右へ x・紙の上へ y [mm])から、立方体の 8 つの角が画像のどこに写るはずかを計算し、検出と比べる。
比べ方を 2 通り出す:
  (1) 検出の中心 → 机の上の位置への換算(立方体の中心の高さ = 1 辺の半分の面に当てる)と、測った位置との差 [mm]
  (2) 測った位置から予想した、画像の中心と、検出した中心との差 [画素]。換算の仮定(高さ)を含まないので、検出そのものの確認になる
作者の定規の精度は数 mm なので、差が数 mm なら「合っている」と見る。

使い方(この .venv で実行): .venv/bin/python scripts/check_cube_localization.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from hand_sim.aruco_marker import detect_markers  # noqa: E402
from hand_sim.fixed_board import BOARD_XY_M, BOARD_YAW_RAD, SIDE_MM, camera_pose_from_board  # noqa: E402
from hand_sim.real_camera import load_camera, pixel_to_world_plane_K, project_points_K, undistort_image  # noqa: E402
from hand_sim.vision import find_cube_pose_hsv  # noqa: E402
from hand_sim.webcam import grab_frame, list_cameras  # noqa: E402

CUBE_MM = 30.0  # 立方体の 1 辺(作者がノギスで測った値)
# 作者の測定: 板の十字から、(紙の右へ x, 紙の上へ y) [mm]。立方体の中心(真上から見た位置)
MEASURED = {"blue": (-25.0, 37.0), "red": (-21.0, 137.0), "green": (80.0, 140.0)}

cam = load_camera()
device = next(devs[0] for name, devs in list_cameras() if "WebCamera" in name)
fixed = undistort_image(grab_frame(device), cam["K"], cam["dist"])
found = detect_markers(fixed, cam["K"], np.zeros(5), SIDE_MM / 1000)  # 補正済みの画像なので歪みは 0
board = {i: m for i, m in found.items() if 1 <= i <= 6}
cam_pos, cam_R, reproj, n = camera_pose_from_board(board, cam["K"], np.zeros(5), SIDE_MM)
print(f"板のマーカー {n} 個、再投影誤差 {reproj:.2f} 画素、カメラの位置 (x, y, z) = {np.round(cam_pos * 100, 1)} cm")

# 板の探す範囲(view_real_detection.py と同じ)
pts = np.vstack([m["corners"] for m in board.values()]).astype(np.float32)
hull = cv2.convexHull(pts).reshape(-1, 2)
hull = hull.mean(axis=0) + (hull - hull.mean(axis=0)) * 2.2
roi = np.zeros(fixed.shape[:2], np.uint8)
cv2.fillConvexPoly(roi, hull.astype(np.int32), 1)
roi = roi.astype(bool)

c, s = np.cos(BOARD_YAW_RAD), np.sin(BOARD_YAW_RAD)


def board_to_world(x_mm, y_mm):
    """板の座標(x = 紙の右、y = 紙の上)→ 世界座標 [m]。fixed_board.board_marker_centers_world と同じ計算。"""
    x, y = x_mm * 1e-3, y_mm * 1e-3
    return np.array([BOARD_XY_M[0] + c * x - s * y, BOARD_XY_M[1] + s * x + c * y])


h = CUBE_MM * 1e-3
print(f"\n{'色':6s} {'換算の位置との差 (x, y) [mm]':>30s} {'距離':>7s} {'画像の差 (u, v) [画素]':>24s}")
dists = []
for color, (mx, my) in MEASURED.items():
    pose = find_cube_pose_hsv(fixed, color, roi)
    if pose is None:
        print(f"{color:6s} 見つからない")
        continue
    u, v, _ = pose
    truth = board_to_world(mx, my)
    # (1) 検出の中心を、高さ h/2 の面に当てる
    est = pixel_to_world_plane_K(u, v, cam_pos, cam_R, cam["K"], h / 2)[:2]
    diff = (est - truth) * 1e3
    # (2) 立方体の 8 つの角を投影し、画像での外接する四角形の中心を、予想の中心にする
    corners = np.array([[truth[0] + dx * h / 2, truth[1] + dy * h / 2, z] for dx in (-1, 1) for dy in (-1, 1) for z in (0.0, h)])
    uv = project_points_K(corners, cam_pos, cam_R, cam["K"])
    pred = (uv.min(axis=0) + uv.max(axis=0)) / 2
    d = np.hypot(*diff)
    dists.append(d)
    print(f"{color:6s} ({diff[0]:+6.1f}, {diff[1]:+6.1f}) {d:11.1f} mm   ({u - pred[0]:+6.1f}, {v - pred[1]:+6.1f})")
if dists:
    print(f"\n換算の差の平均: {np.mean(dists):.1f} mm(定規の精度は数 mm)")
