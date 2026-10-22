"""【概要】実機のウェブカメラの画像を 1 枚撮り(または、保存した画像を読み)、歪みを補正して、sim と同じ色検出で、立方体(赤・緑・青)を探す。
見つけた位置に印を付けた画像を outputs/real_detect.png に保存し、画素位置 (u, v) と向き yaw を表示する。
カメラの置き場所(位置・向き)は、まだ求めていないので、世界座標(机の上の位置 [m])は出さない。置き場所を較正したあとで、
real_camera.pixel_to_world_plane_K(u, v, cam_pos, cam_R, K, 立方体の上面の高さ) をつなぐ。

使い方(この .venv で実行。カメラの番号は、差し直しで入れ替わるので、名前(WebCamera)で探す):
    .venv/bin/python scripts/view_real_detection.py                       # ウェブカメラから 1 枚撮る
    .venv/bin/python scripts/view_real_detection.py --image 画像のパス     # 保存した画像で試す(カメラ不要)
    .venv/bin/python scripts/view_real_detection.py --manual --exposure 60  # 露出と色を固定してから撮る(撮影のときと同じ)
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)。
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from hand_sim.real_camera import load_camera, undistort_image  # noqa: E402
from hand_sim.vision import find_cube_pose  # noqa: E402
from hand_sim.webcam import grab_frame, list_cameras, set_manual_exposure  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--image", help="カメラの代わりに読む、保存した画像(歪みの補正の前の、撮ったままの画像)")
parser.add_argument("--device", help="カメラのファイル名(例: /dev/video0)。省略すると、名前に WebCamera を含むカメラを探す")
parser.add_argument("--manual", action="store_true", help="露出とホワイトバランスを固定にする")
parser.add_argument("--exposure", type=int, default=300, help="--manual のときの露出時間(0.1 ミリ秒単位、1〜5000)")
args = parser.parse_args()

cam = load_camera()
if args.image:
    img = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)  # OpenCV は BGR の順で読むので、RGB(sim・vision.py と同じ順)に直す
else:
    device = args.device
    if device is None:
        # 名前に "WebCamera" を含むカメラの、最初の /dev/video を使う(next(...): 条件に合う最初の 1 つを取る。無ければ None)。[頻度] 「最初に見つかった 1 つ」を取るときの定番
        device = next((devs[0] for name, devs in list_cameras() if "WebCamera" in name), None)
        if device is None:
            sys.exit("ウェブカメラが見つからない(v4l2-ctl --list-devices で確認する。MiraBox はキャプチャカードで、ウェブカメラではない)")
    if args.manual:
        set_manual_exposure(device, args.exposure)
    print(f"カメラ: {device}")
    img = grab_frame(device)
fixed = undistort_image(img, cam["K"], cam["dist"])  # 歪みを補正(以降は、sim のカメラと同じ、ピンホールカメラの画像として扱える)

annotated = cv2.cvtColor(fixed, cv2.COLOR_RGB2BGR)  # 保存・描画は OpenCV なので、BGR に戻す
draw_color = {"red": (0, 0, 255), "green": (0, 200, 0), "blue": (255, 0, 0)}  # BGR の順での、印の色
found = 0
for color in ("red", "green", "blue"):
    pose = find_cube_pose(fixed, color)
    if pose is None:
        print(f"{color:5s}: 見つからない")
        continue
    u, v, yaw = pose
    found += 1
    print(f"{color:5s}: 画素 (u, v) = ({u:.1f}, {v:.1f})、向き yaw = {yaw:.0f}°")
    cv2.drawMarker(annotated, (int(u), int(v)), draw_color[color], cv2.MARKER_CROSS, 40, 3)
    cv2.putText(annotated, f"{color} yaw {yaw:.0f}", (int(u) + 25, int(v) - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, draw_color[color], 2)
out = ROOT / "outputs" / "real_detect.png"
cv2.imwrite(str(out), annotated)
print(f"保存: {out.relative_to(ROOT)}(歪みを補正した画像に、見つけた位置を描いた。{found} 色)")
print("※ 色検出は、sim の画像で決めた条件(その色のチャンネルが 100 超で、他が 0.4 倍未満)。実機の照明では、色が合わないことがある(その場合は、露出・条件を調整する)")
