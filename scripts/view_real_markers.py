"""【概要】実機のウェブカメラの画像を 1 枚撮り(または、保存した画像を読み)、ArUco マーカーを探して、
ID・カメラからの距離・マーカーの面の向き(カメラに対する傾き)を表示する。見つけた位置に、枠・ID・座標軸を描いた画像を outputs/real_markers.png に保存する。
カメラの置き場所(世界座標での位置・向き)は、まだ求めていないので、出すのは「カメラから見た位置」まで。
世界座標にするには、aruco_marker.marker_pose_in_world に、カメラの置き場所を渡す(カメラの固定位置と較正のあと)。

使い方(この .venv で実行。カメラの番号は、差し直しで入れ替わるので、名前(WebCamera)で探す):
    .venv/bin/python scripts/view_real_markers.py                         # ウェブカメラから 1 枚撮る(マーカーの 1 辺は 30 mm)
    .venv/bin/python scripts/view_real_markers.py --side 40               # 印刷したマーカーの 1 辺が 40 mm のとき(定規で測った値)
    .venv/bin/python scripts/view_real_markers.py --image 画像のパス       # 保存した画像で試す(カメラ不要)
    .venv/bin/python scripts/view_real_markers.py --manual --exposure 300 # 露出と色を固定してから撮る
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)。
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from hand_sim.aruco_marker import detect_markers  # noqa: E402
from hand_sim.real_camera import load_camera  # noqa: E402
from hand_sim.webcam import grab_frame, list_cameras, set_manual_exposure  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--side", type=float, default=30.0, help="印刷したマーカーの 1 辺 [mm](黒い枠の外側から外側。定規で測った値)")
parser.add_argument("--image", help="カメラの代わりに読む、保存した画像(撮ったままの画像)")
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
        # 名前に "WebCamera" を含むカメラの、最初の /dev/video を使う(無ければ None)。[頻度] 「最初に見つかった 1 つ」を取るときの定番
        device = next((devs[0] for name, devs in list_cameras() if "WebCamera" in name), None)
        if device is None:
            sys.exit("ウェブカメラが見つからない(v4l2-ctl --list-devices で確認する。MiraBox はキャプチャカードで、ウェブカメラではない)")
    if args.manual:
        set_manual_exposure(device, args.exposure)
    print(f"カメラ: {device}")
    img = grab_frame(device)

# 歪みの補正はしない: detect_markers は、歪んだままの画像と dist を受け取り、中で考慮する。描く画像も、同じ(歪んだまま)ので、枠が、実際の像に重なる。
side_m = args.side / 1000
found = detect_markers(img, cam["K"], cam["dist"], side_m)
annotated = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # 保存・描画は OpenCV なので、BGR に戻す
if not found:
    print("マーカーが見つからない(明るさ・ピント・マーカーの大きさ(画像の中で 40 画素以上が目安)・白い余白が切れていないかを確かめる)")
for marker_id, m in sorted(found.items()):
    R, t = m["R"], m["t"]
    distance = np.linalg.norm(t) * 100  # カメラの光学中心からマーカーの中心まで [cm]
    # 面の傾き: マーカーの面の垂直(R の 3 列目)と、カメラ → マーカーの向きの、なす角。0° = 面がカメラのほうを真正面に向いている。
    # 面が裏向き(カメラに背を向けている)なら、検出されない。
    normal = R[:, 2]
    tilt = np.degrees(np.arccos(np.clip(-normal @ t / np.linalg.norm(t), -1, 1)))
    side_px = np.linalg.norm(m["corners"][0] - m["corners"][1])
    print(f"ID {marker_id}: カメラから {distance:.1f} cm(カメラ座標 x={t[0] * 100:+.1f}, y={t[1] * 100:+.1f}, z={t[2] * 100:+.1f} cm)、"
          f"面の傾き {tilt:.0f}°、1 辺 {side_px:.0f} 画素")
    cv2.polylines(annotated, [m["corners"].astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 2)  # マーカーの枠(緑)
    rvec, _ = cv2.Rodrigues(R)
    cv2.drawFrameAxes(annotated, cam["K"], cam["dist"], rvec, t.reshape(3, 1), side_m * 0.8, 3)  # 座標軸(赤 = x、緑 = y、青 = z)
    top_left = m["corners"][0].astype(int)
    cv2.putText(annotated, f"ID {marker_id} {distance:.0f}cm", (int(top_left[0]), int(top_left[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
out = ROOT / "outputs" / "real_markers.png"
out.parent.mkdir(exist_ok=True)
cv2.imwrite(str(out), annotated)
print(f"保存: {out.relative_to(ROOT)}(撮ったままの画像に、見つけたマーカーを描いた。{len(found)} 個)")
