"""【概要】今のカメラの位置で、3 つの置き場(赤・緑・青。cube_world.PADS)と、立方体を置く範囲が、画像の中に写るかを確かめる(腕は動かさない)。
撮影 → 板のマーカーからカメラの位置を求める → 置き場の中心(立方体の中心の高さ 15 mm)を画像に当てる。結果を outputs/pads_visible.png に描く(黄色の円 = 置き場)。
カメラを動かしたあと、試験(run_real_trial.sh)の前に実行して、置き場が画像の中に入っているかを見る。
OK の基準: 画像の端から 60 画素以上内側、かつ下端は v <= 1000(画像の下の端はレンズの較正が届いていなくて、位置の精度が落ちるため)。
使い方: .venv/bin/python scripts/check_pads_visible.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from hand_sim.aruco_marker import detect_markers  # noqa: E402
from hand_sim.cube_world import PADS, WORKSPACE_X_M  # noqa: E402
from hand_sim.fixed_board import SIDE_MM, camera_pose_from_board  # noqa: E402
from hand_sim.real_camera import load_camera, project_points_K, undistort_image  # noqa: E402
from hand_sim.webcam import grab_frame, list_cameras, set_manual_exposure  # noqa: E402

cam = load_camera()
device = next(devs[0] for name, devs in list_cameras() if "WebCamera" in name)
set_manual_exposure(device, exposure=300, white_balance=4500, gain=32)
img = undistort_image(grab_frame(device), cam["K"], cam["dist"])
board = {i: m for i, m in detect_markers(img, cam["K"], np.zeros(5), SIDE_MM / 1000).items() if 1 <= i <= 6}
if len(board) < 4:
    sys.exit(f"板のマーカーが {len(board)} 個しか見えない(4 個以上要る)")
cam_pos, cam_R, reproj, n = camera_pose_from_board(board, cam["K"], np.zeros(5), SIDE_MM)
h, w = img.shape[:2]
out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
all_ok = True
for color, (x, y) in PADS.items():
    u, v = project_points_K(np.array([[x, y, 0.015]]), cam_pos, cam_R, cam["K"])[0]
    ok = 60 <= u <= w - 60 and 60 <= v <= min(h - 60, 1000)
    all_ok &= ok
    print(f"{color:5s} 置き場 ({x * 1000:.0f}, {y * 1000:.0f}) mm → 画素 ({u:.0f}, {v:.0f})  {'OK' if ok else '画像の外、または端に近い'}")
    if 0 <= u < w and 0 <= v < h:
        cv2.circle(out, (int(u), int(v)), 30, (0, 255, 255), 3)
        cv2.putText(out, color, (int(u) + 35, int(v)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
cv2.imwrite(str(ROOT / "outputs" / "pads_visible.png"), out)
print(f"板のマーカー {n} 個、再投影誤差 {reproj:.2f} 画素。画像: outputs/pads_visible.png")
print("すべての置き場が写っている。試験に進める" if all_ok else "写っていない置き場がある。カメラを動かして(置き場側 = ロボットの左側が入るように)、もう一度確かめる")
