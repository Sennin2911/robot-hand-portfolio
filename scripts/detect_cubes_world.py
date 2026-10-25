"""【概要】ウェブカメラで 1 枚撮り、机の板からカメラの位置を求め、赤・緑・青の立方体の、机の上の位置(世界座標 x, y [m])と向き yaw を出して、
outputs/cubes_detected.json に保存する(掴む計画 plan_pick_test.py が読む)。画像は outputs/real_detect.png(印と探す範囲を描いたもの)。
立方体の中心の高さは、1 辺(30 mm)の半分(15 mm)の面に当てて換算する。板のマーカーが 4 個未満しか見えなければ、何も保存しない。

使い方(この .venv で実行): .venv/bin/python scripts/detect_cubes_world.py [--extend-down 120]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from hand_sim.aruco_marker import detect_markers  # noqa: E402
from hand_sim.fixed_board import SIDE_MM, board_marker_corners_world, camera_pose_from_board  # noqa: E402
from hand_sim.cube_world import marker_mask, white_balance_by_paper, workspace_mask  # noqa: E402
from hand_sim.real_camera import load_camera, pixel_to_world_plane_K, project_points_K, undistort_image  # noqa: E402
from hand_sim.vision import find_cube_pose_hsv, square_yaw_from_points  # noqa: E402
from hand_sim.webcam import grab_frame, list_cameras, set_manual_exposure  # noqa: E402

# コマンドラインの引数を受け取る仕組み。[頻度] スクリプトに「設定を変えて実行したい」ときの定番
parser = argparse.ArgumentParser()
parser.add_argument("--extend-down", type=int, default=0, metavar="PX",
                    help="探す範囲(黄色の枠)の下の辺を、画像の下方向へ PX 画素広げる(ロボットの左側 = 画像の下。既定 0)。"
                         "画像の下の端は、レンズの較正が届いていない範囲なので、広げるほど位置の精度が落ちうる")
args = parser.parse_args()

CUBE_M = 0.030  # 立方体の 1 辺(ノギスで測った値)
# 前回の検出結果を、先に消す。今回の検出が途中で失敗しても(暗くて見つからない、カメラが読めない、マーカーが足りないなど)、
# 古い位置が残って、plan_pick_test.py がそれで計画を作ってしまわないため。成功した色だけが、最後に書き直される。
# unlink(missing_ok=True): ファイルを消す。無くてもエラーにしない。[頻度] 「あれば消す」の定番
(ROOT / "outputs" / "cubes_detected.json").unlink(missing_ok=True)
cam = load_camera()
device = next(devs[0] for name, devs in list_cameras() if "WebCamera" in name)
# 撮る前に、露出・ホワイトバランス・ゲインを毎回この値に固定する。カメラの設定は、前に動いたスクリプトの値が残るため
# (record_hand_sample.py は、手の板を読むために露出を 100〜1600 の間で変え、その値を残す)。設定しないと、実行のたびに画像の明るさが変わる。
# 値は、立方体の色検出が合う条件(露出 300、ホワイトバランス 4500、ゲイン 32)。[頻度] 色で検出するスクリプトでは、毎回固定するのが定番
set_manual_exposure(device, exposure=300, white_balance=4500, gain=32)
fixed = undistort_image(grab_frame(device), cam["K"], cam["dist"])
board = {i: m for i, m in detect_markers(fixed, cam["K"], np.zeros(5), SIDE_MM / 1000).items() if 1 <= i <= 6}
if len(board) < 4:
    sys.exit(f"板のマーカーが {len(board)} 個しか見えない(4 個以上要る)。カメラ・明るさ・立方体が隠していないかを確認する")
cam_pos, cam_R, reproj, n = camera_pose_from_board(board, cam["K"], np.zeros(5), SIDE_MM)
print(f"板のマーカー {n} 個、再投影誤差 {reproj:.2f} 画素")

# 探す範囲: 板の 6 個のマーカーの角が、画像のどこに写るはずかを計算して使う(立方体がマーカーを隠していても、範囲が変わらない)
pts = project_points_K(np.vstack(list(board_marker_corners_world().values())), cam_pos, cam_R, cam["K"]).astype(np.float32)
hull = cv2.convexHull(pts).reshape(-1, 2)
hull = hull.mean(axis=0) + (hull - hull.mean(axis=0)) * 2.2  # 探す範囲(view_real_detection.py と同じ)
hull[hull[:, 1] > hull[:, 1].mean(), 1] += args.extend_down  # 画像の下側の頂点(v が平均より大きい点)だけを、下へ動かす
hull[:, 1] = np.clip(hull[:, 1], 0, fixed.shape[0] - 1)  # 画像の外へはみ出さないようにする
roi = np.zeros(fixed.shape[:2], np.uint8)
cv2.fillConvexPoly(roi, hull.astype(np.int32), 1)
roi = roi.astype(bool) & ~marker_mask(fixed.shape, cam_pos, cam_R, cam["K"]) & workspace_mask(fixed.shape, cam_pos, cam_R, cam["K"])
# 上: 板のマーカーの部分は探さない(黒に青みが残ると「青」と誤検出するため)。ロボットの土台など、立方体を置けない範囲も探さない(詳しくは marker_mask、workspace_mask)

# 色の判定用に、板の白い紙が白になるよう色を補正した画像を使う(照明やカメラの設定で色が偏っても、立方体の色相が判定の範囲に入るように)。表示は、元の画像のまま
balanced, gains = white_balance_by_paper(fixed, cam_pos, cam_R, cam["K"])
print(f"色の補正(紙を白にする倍率 R, G, B): {np.round(gains, 2)}")

annotated = cv2.cvtColor(fixed, cv2.COLOR_RGB2BGR)
cv2.polylines(annotated, [hull.astype(np.int32)], True, (0, 255, 255), 2)
draw = {"red": (0, 0, 255), "green": (0, 200, 0), "blue": (255, 0, 0)}
out = {}
for color in ("red", "green", "blue"):
    pose, mask = find_cube_pose_hsv(balanced, color, roi, return_mask=True)
    if pose is None:
        print(f"{color:5s}: 見つからない")
        continue
    u, v, yaw = pose
    xy = pixel_to_world_plane_K(u, v, cam_pos, cam_R, cam["K"], CUBE_M / 2)[:2]
    # 向き: 画像の向きは斜め上からの見え方でゆがむので、世界座標で出す。立方体の側面(暗い)を避け、塊の中で明るい上位 40% の画素 = 上面を、
    # 上面の高さ(30 mm)の面に当てて、その点の集まりの向きを見る。
    v_idx, u_idx = np.nonzero(mask)
    bright = fixed[v_idx, u_idx].astype(float).sum(axis=1)
    top = bright >= np.quantile(bright, 0.6)
    top_xy = np.array([pixel_to_world_plane_K(uu, vv, cam_pos, cam_R, cam["K"], CUBE_M)[:2] for uu, vv in zip(u_idx[top][::10], v_idx[top][::10])])
    yaw_world = square_yaw_from_points(top_xy)
    out[color] = {"x_m": float(xy[0]), "y_m": float(xy[1]), "yaw_deg": yaw_world, "uv": [float(u), float(v)]}
    print(f"{color:5s}: 世界座標 x = {xy[0] * 1000:6.1f} mm、y = {xy[1] * 1000:6.1f} mm、向き(世界座標、0〜90°)= {yaw_world:.0f}°(画素 ({u:.0f}, {v:.0f}))")
    cv2.drawMarker(annotated, (int(u), int(v)), draw[color], cv2.MARKER_CROSS, 40, 3)
    cv2.putText(annotated, color, (int(u) + 25, int(v) - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, draw[color], 2)
cv2.imwrite(str(ROOT / "outputs" / "real_detect.png"), annotated)
if out:
    (ROOT / "outputs" / "cubes_detected.json").write_text(json.dumps({"cam_pos_m": cam_pos.tolist(), "cubes": out}, ensure_ascii=False, indent=2))
    print("保存: outputs/cubes_detected.json、outputs/real_detect.png")
