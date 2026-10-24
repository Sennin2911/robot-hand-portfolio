"""【概要】実機のウェブカメラの画像で、机に固定した板(ID 1〜6)を探し、カメラの世界座標での位置・向きを求める。
同じ画像に手の板(ID 0)も写っていれば、その手先(gripperframe)の世界座標での位置も出す(腕のずれの確認に使う)。
板の置き方は src/hand_sim/fixed_board.py の BOARD_XY_M、BOARD_YAW_RAD、BOARD_Z_M(作者の測定)。
出力: 板の再投影誤差(小さいほど良い)、カメラの位置 [cm](世界座標: x = 正面、y = 左、z = 上)、手先の位置。画像は outputs/real_board.png。

使い方(この .venv で実行):
    .venv/bin/python scripts/view_real_board.py --manual --exposure 600     # ウェブカメラから 1 枚撮る
    .venv/bin/python scripts/view_real_board.py --image 画像のパス           # 保存した画像で試す
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from hand_sim.aruco_marker import detect_markers, marker_pose_in_world  # noqa: E402
from hand_sim.fixed_board import SIDE_MM, camera_pose_from_board  # noqa: E402
from hand_sim.hand_marker import hand_pose_from_marker  # noqa: E402
from hand_sim.real_camera import distort_pixels, load_camera, project_points_K  # noqa: E402
from hand_sim.webcam import grab_frame, list_cameras, set_manual_exposure  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--side", type=float, default=SIDE_MM, help="マーカーの 1 辺 [mm](板も手の板も同じ 30 mm。ノギスで測った値)")
parser.add_argument("--image", help="カメラの代わりに読む、保存した画像")
parser.add_argument("--device", help="カメラのファイル名。省略すると、名前に WebCamera を含むカメラを探す")
parser.add_argument("--manual", action="store_true", help="露出とホワイトバランスを固定にする")
parser.add_argument("--exposure", type=int, default=600, help="--manual のときの露出時間(0.1 ミリ秒単位)")
args = parser.parse_args()

cam = load_camera()
if args.image:
    img = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)  # OpenCV は BGR で読むので RGB に直す
else:
    device = args.device or next((devs[0] for name, devs in list_cameras() if "WebCamera" in name), None)
    if device is None:
        sys.exit("ウェブカメラが見つからない(v4l2-ctl --list-devices で確認する)")
    if args.manual:
        set_manual_exposure(device, args.exposure)
    print(f"カメラ: {device}")
    img = grab_frame(device)

found = detect_markers(img, cam["K"], cam["dist"], args.side / 1000)
print(f"見つかった ID: {sorted(found)}")
annotated = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
for i, m in found.items():
    cv2.polylines(annotated, [m["corners"].astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 2)
    cv2.putText(annotated, f"ID {i}", tuple(m["corners"][0].astype(int) - [0, 10]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
result = camera_pose_from_board(found, cam["K"], cam["dist"], args.side)
if result is None:
    print("板のマーカー(ID 1〜6)が見つからない")
else:
    cam_pos, cam_R, err, n = result
    print(f"板のマーカー {n} 個で、カメラの位置を計算。再投影誤差の平均 {err:.2f} 画素(1 画素前後以下なら良い)")
    print(f"カメラの位置(世界座標、cm): x = {cam_pos[0] * 100:+.1f}(正面)、y = {cam_pos[1] * 100:+.1f}(左)、z = {cam_pos[2] * 100:+.1f}(上)")
    if 0 in found:
        R_m, p_m = marker_pose_in_world(found[0]["R"], found[0]["t"], cam_pos, cam_R)
        R_h, p_h = hand_pose_from_marker(R_m, p_m)
        print(f"手先(gripperframe)の位置(世界座標、cm): x = {p_h[0] * 100:+.1f}、y = {p_h[1] * 100:+.1f}、z = {p_h[2] * 100:+.1f}")
        print(f"手先の x 軸(指先の向き)の世界での向き: {np.round(R_h[:, 0], 2)}")
    # 求めた座標系が合っているかを、目で確かめる: 世界の原点(肩の軸の真下の机の点)と、軸(長さ 10 cm)、肩の軸の高さ(6.2 cm)の点を、画像に描く。
    # 原点が、土台の肩の軸の真下に重なれば、板の置き方(位置・向き)とカメラの位置が合っている。
    pts = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1], [0, 0, 0.0624]])
    uv = project_points_K(pts, cam_pos, cam_R, cam["K"])  # 歪みのない画素
    du, dv = distort_pixels(uv[:, 0], uv[:, 1], cam["K"], cam["dist"])  # 実際の(歪んだ)画像の画素に直す
    px = np.stack([du, dv], axis=1).astype(int)
    for k, (color, name) in enumerate([((0, 0, 255), "X"), ((0, 255, 0), "Y"), ((255, 0, 0), "Z")], start=1):  # OpenCV の色は BGR の順(赤 = x、緑 = y、青 = z)
        cv2.line(annotated, tuple(px[0]), tuple(px[k]), color, 3)
        cv2.putText(annotated, name, tuple(px[k]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.circle(annotated, tuple(px[0]), 8, (0, 255, 255), -1)  # 世界の原点(黄)
    cv2.circle(annotated, tuple(px[4]), 8, (255, 0, 255), 2)  # 肩の軸の高さ(紫の輪)
out = ROOT / "outputs" / "real_board.png"
cv2.imwrite(str(out), annotated)
print(f"保存: {out.relative_to(ROOT)}")
