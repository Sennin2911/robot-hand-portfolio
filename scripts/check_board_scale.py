"""【概要】机の板(ID 1〜6)の、マーカー中心どうしの間隔を、画像から出して、設計値と比べる。レンズの補正(内部パラメータ)と、板の大きさの確認用。
マーカー 1 個ずつ、独立に位置を出し(カメラから見た 3D 位置)、2 個の中心の距離を計算する。印刷物の間隔(設計値)は fixed_board.py の配置(90 mm と 80 mm の格子)。
また、各マーカーが、画像のどこに写っているか(0〜1 に直した位置)も出す。内部パラメータの較正で、角が届いた範囲は、横 0.09〜0.94、縦 0.16〜0.89。そこから外れた位置のマーカーは、補正の精度が未確認。
カメラが USB で認識されない間は、撮れるまで最大 60 秒待つ。

使い方(この .venv で実行):
    .venv/bin/python scripts/check_board_scale.py [--image 画像のパス]
"""
import argparse
import itertools
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from hand_sim.aruco_marker import detect_markers  # noqa: E402
from hand_sim.fixed_board import SIDE_MM, board_marker_centers_mm  # noqa: E402
from hand_sim.real_camera import load_camera  # noqa: E402
from hand_sim.webcam import grab_until, list_cameras  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--image", help="カメラの代わりに読む、保存した画像")
args = parser.parse_args()

cam = load_camera()
W, H = cam["size"]
if args.image:
    img = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
else:
    img = None
    t_end = time.time() + 60
    while img is None and time.time() < t_end:
        device = next((devs[0] for name, devs in list_cameras() if "WebCamera" in name), None)
        if device:
            try:
                # 露出を短い方から試して、板のマーカーが 4 個以上読める明るさを探す(部屋の明るさが、時間帯で変わるため)
                img, used = grab_until(device, lambda im: len([i for i in detect_markers(im, cam["K"], cam["dist"], SIDE_MM / 1000) if 1 <= i <= 6]) >= 4,
                                       settle_frames=15, timeout=8)
                print(f"露出 {used}(None = 読めなかった)")
            except Exception:  # 不安定な USB の再試行(何が起きても、時間内ならやり直す)
                pass
        if img is None:
            time.sleep(1)
    if img is None:
        sys.exit("ウェブカメラから撮れなかった")

cv2.imwrite(str(ROOT / "outputs" / "board_scale_raw.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))  # 撮った画像をそのまま保存(見つからないとき、原因を目で見るため)
found = detect_markers(img, cam["K"], cam["dist"], SIDE_MM / 1000)
design = board_marker_centers_mm()
ids = [i for i in sorted(found) if i in design]
print(f"見つかった板のマーカー: {ids}(手の板 ID 0 は {'あり' if 0 in found else 'なし'})")
print("\nマーカーの、画像の中の位置(0〜1。横, 縦)と、カメラからの距離:")
for i in sorted(found):
    c = found[i]["corners"].mean(axis=0)
    print(f"  ID {i}: 横 {c[0] / W:.2f}、縦 {c[1] / H:.2f}、距離 {np.linalg.norm(found[i]['t']) * 100:.1f} cm")
if len(ids) < 2:
    sys.exit("板のマーカーが 2 個未満。outputs/board_scale_raw.png を見て、明るさ・カメラの向き・隠れを確かめる")
print("\n2 個の中心の間隔(画像から − 設計値):")
errs = []
for a, b in itertools.combinations(ids, 2):
    d_img = np.linalg.norm(found[a]["t"] - found[b]["t"]) * 1000
    d_des = np.linalg.norm(np.array(design[a]) - np.array(design[b]))
    errs.append((d_img - d_des, d_des, a, b))
    print(f"  ID {a}–{b}: {d_img:6.1f} − {d_des:6.1f} = {d_img - d_des:+5.1f} mm({(d_img / d_des - 1) * 100:+.1f}%)")
e = np.array([x[0] for x in errs]); rel = np.array([x[0] / x[1] for x in errs])
print(f"\n間隔の誤差: 平均 {e.mean():+.2f} mm(平均 {rel.mean() * 100:+.2f}%)、ばらつき(標準偏差){e.std():.2f} mm、最大の絶対値 {np.abs(e).max():.1f} mm")
