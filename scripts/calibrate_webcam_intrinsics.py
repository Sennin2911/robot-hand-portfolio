"""【概要】capture_chessboard.py で撮ったチェスボードの画像から、ウェブカメラの内部パラメータ(焦点距離・画像の中心・歪み・視野角)を求めて、
calibration/webcam_intrinsics.json に保存する。結果として、(1) 再投影誤差(1 画素より小さければ良い)、(2) 画像のどの範囲を、盤の角がカバーしたか、
(3) 歪みを補正した画像(outputs/chessboard_undistorted.png。盤の線が、まっすぐになっているか)を出す。

使い方(この .venv で実行):
    .venv/bin/python scripts/calibrate_webcam_intrinsics.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)。
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from hand_sim.intrinsics import calibrate_intrinsics, find_corners, save_intrinsics  # noqa: E402

shots = sorted((ROOT / "outputs" / "chessboard_shots").glob("shot_*.png"))
if len(shots) < 10:
    sys.exit(f"画像が足りない({len(shots)} 枚)。先に scripts/capture_chessboard.py で、15 枚以上撮る")
corner_list, names, size = [], [], None
for path in shots:
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    size = (img.shape[1], img.shape[0])
    corners = find_corners(img)
    if corners is not None:
        corner_list.append(corners)
        names.append(path.name)
print(f"{len(shots)} 枚のうち、角が見つかったのは {len(corner_list)} 枚(画像の大きさ {size[0]} × {size[1]})")

result = calibrate_intrinsics(corner_list, size)
K, dist = result["K"], result["dist"]
print(f"\n再投影誤差 {result['rms']:.3f} 画素({'良い(1 画素未満)' if result['rms'] < 1 else '大きい。盤が反っている、ぶれている、枚数が少ない、のどれかを疑う'})")
print(f"焦点距離 fx = {K[0, 0]:.1f}、fy = {K[1, 1]:.1f} 画素(fx と fy が 1% 以内で同じなら、正方形の画素)")
print(f"画像の中心 cx = {K[0, 2]:.1f}、cy = {K[1, 2]:.1f}(画像の真ん中は {size[0] / 2:.0f}、{size[1] / 2:.0f})")
print(f"視野角: 縦 {result['fovy_deg']:.1f}°、横 {result['fovx_deg']:.1f}°(sim のカメラの fovy は、この縦の視野角を使う)")
print(f"歪みの係数 k1, k2, p1, p2, k3 = {np.round(dist, 4).tolist()}(k1 が負なら、樽型の歪み)")
worst = int(np.argmax(result["per_image_rms"]))
print(f"画像ごとの誤差: 最小 {min(result['per_image_rms']):.2f}、最大 {max(result['per_image_rms']):.2f} 画素(最大は {names[worst]})")

# 盤の角が、画像のどの範囲を通ったか(歪みは、画像の端ほど大きいので、端まで、角がカバーしていないと、端の補正が当てにならない)
all_pts = np.concatenate([c.reshape(-1, 2) for c in corner_list])
cover = [all_pts[:, 0].min() / size[0], all_pts[:, 0].max() / size[0], all_pts[:, 1].min() / size[1], all_pts[:, 1].max() / size[1]]
print(f"角が通った範囲(画像の幅・高さに対する割合): 横 {cover[0]:.2f}〜{cover[1]:.2f}、縦 {cover[2]:.2f}〜{cover[3]:.2f}(0〜1 の全体に近いほど良い。端が 0.1 / 0.9 より内側なら、端の撮影が足りない)")

out = ROOT / "calibration" / "webcam_intrinsics.json"
out.parent.mkdir(exist_ok=True)
save_intrinsics({**result, "device": "TUNSONE A03 (WebCamera 32e6:9221)", "pattern": "9x6 inner corners, 25 mm", "n_images": len(corner_list)}, out)
print(f"\n保存: {out.relative_to(ROOT)}")

# 歪みの補正の見た目: 最後の画像を補正して、元の画像と並べる
img = cv2.imread(str(shots[worst]))
undist = cv2.undistort(img, K, dist)  # cv2.undistort: 歪みを補正した画像を作る
cv2.imwrite(str(ROOT / "outputs" / "chessboard_undistorted.png"), np.concatenate([img, undist], axis=1))
print("補正の前(左)と後(右)の画像: outputs/chessboard_undistorted.png(盤の線が、補正後に、まっすぐになっているか見る)")
