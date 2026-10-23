"""【概要】実機のウェブカメラの画像を 1 枚撮り(または、保存した画像を読み)、手に貼った色のシール(赤・黄・緑・青)を探して、
色ごとの「中心の画素位置」と「塊の面積(画素)」を表示する。4 色見つかれば、面の位置・向き(カメラから見た位置、面の傾き)も表示する。
見つけた位置に印を付けた画像を outputs/real_dots.png に保存する。**面積が小さい(目安 100 画素未満)と、検出が不安定**なので、5 mm のシールで足りるかの確認に使う。
シールの貼り方(各色の中心の位置)は、既定は 2×2、間隔 30 mm(dot_marker.DEFAULT_LAYOUT)。実際に貼った位置を測って、--layout で渡す(JSON ファイル)。
    例: {"red": [-15.0, 15.0], "yellow": [15.0, 15.0], "green": [-15.0, -15.0], "blue": [15.0, -15.0]}   (x = 面をカメラ側から見て右、y = 上。単位 mm)
貼り位置の測りの精度は、姿勢の精度に直結する(0.5 mm の測り誤差で、奥行きが約 1 cm ずれる。docs/dot_markers.md)。

使い方(この .venv で実行。カメラの番号は、差し直しで入れ替わるので、名前(WebCamera)で探す):
    .venv/bin/python scripts/view_real_dots.py --manual --exposure 300      # ウェブカメラから 1 枚撮る(露出と色を固定)
    .venv/bin/python scripts/view_real_dots.py --layout my_layout.json      # 測った貼り位置を使う
    .venv/bin/python scripts/view_real_dots.py --image 画像のパス             # 保存した画像で試す(カメラ不要)
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)。
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from hand_sim.dot_marker import DEFAULT_LAYOUT, detect_dot_marker, find_color_dots  # noqa: E402
from hand_sim.real_camera import load_camera  # noqa: E402
from hand_sim.webcam import grab_frame, list_cameras, set_manual_exposure  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--layout", help="貼り位置の JSON ファイル({色: [x, y]}、mm)。省略すると、既定の 2×2・30 mm")
parser.add_argument("--image", help="カメラの代わりに読む、保存した画像(撮ったままの画像)")
parser.add_argument("--device", help="カメラのファイル名(例: /dev/video2)。省略すると、名前に WebCamera を含むカメラを探す")
parser.add_argument("--manual", action="store_true", help="露出とホワイトバランスを固定にする")
parser.add_argument("--exposure", type=int, default=300, help="--manual のときの露出時間(0.1 ミリ秒単位、1〜5000)")
args = parser.parse_args()

layout = DEFAULT_LAYOUT
if args.layout:
    with open(args.layout, encoding="utf-8") as f:
        layout = {c: tuple(xy) for c, xy in json.load(f).items()}  # JSON のリスト [x, y] を、(x, y) に直す

cam = load_camera()
if args.image:
    img = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)  # OpenCV は BGR の順で読むので、RGB に直す
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

annotated = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # 保存・描画は OpenCV なので、BGR に戻す
draw = {"red": (0, 0, 255), "yellow": (0, 220, 255), "green": (0, 200, 0), "blue": (255, 0, 0)}  # BGR の順での、印の色
dots = find_color_dots(img, tuple(layout), min_area=5)  # 面積の確認のため、小さい塊も拾う(姿勢を出すときは、min_area = 15 以上だけ使う)
for color in layout:
    if color not in dots:
        print(f"{color:6s}: 見つからない(色の条件に合う塊が無い。明るさ・色の条件を確かめる)")
        continue
    u, v = dots[color]["uv"]
    area = dots[color]["area"]
    flag = "" if area >= 100 else "  ← 面積が小さい(目安 100 画素以上。検出が不安定)"
    print(f"{color:6s}: 画素 (u, v) = ({u:.1f}, {v:.1f})、面積 {area} 画素{flag}")
    cv2.drawMarker(annotated, (int(u), int(v)), draw[color], cv2.MARKER_CROSS, 30, 2)
    cv2.putText(annotated, f"{color} {area}px", (int(u) + 15, int(v) - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, draw[color], 2)

res = detect_dot_marker(img, cam["K"], cam["dist"], layout)
if res is None:
    print("姿勢: 求められない(4 色が見つかっていない、または面積が 15 画素未満の色がある)")
else:
    R, t = res["R"], res["t"]
    distance = np.linalg.norm(t) * 100
    # 面の傾き: 面の垂直(R の 3 列目)と、カメラ → 面の向きの、なす角。0° = 面がカメラのほうを真正面に向いている。
    tilt = np.degrees(np.arccos(np.clip(-R[:, 2] @ t / np.linalg.norm(t), -1, 1)))
    print(f"姿勢: カメラから {distance:.1f} cm(カメラ座標 x={t[0] * 100:+.1f}, y={t[1] * 100:+.1f}, z={t[2] * 100:+.1f} cm)、面の傾き {tilt:.0f}°")
    print(f"      再投影の誤差 {res['reproj_px']:.2f} 画素(目安: 1 画素以下。大きければ、色の取り違い・貼り位置の測り違いを疑う)")
    rvec, _ = cv2.Rodrigues(R)
    cv2.drawFrameAxes(annotated, cam["K"], cam["dist"], rvec, t.reshape(3, 1), 0.02, 3)  # 座標軸(赤 = x、緑 = y、青 = z)。長さ 2 cm
out = ROOT / "outputs" / "real_dots.png"
out.parent.mkdir(exist_ok=True)
cv2.imwrite(str(out), annotated)
print(f"保存: {out.relative_to(ROOT)}(撮ったままの画像に、見つけた位置を描いた)")
print("※ 色の条件(色相の範囲、彩度・明るさの下限)は dot_marker.py の HUE_RANGES など。実機の照明・シールの色に合わせて調整が要ることがある")
