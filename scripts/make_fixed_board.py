"""【概要】机に固定する ArUco の板(カメラの置き場所の較正用)の、印刷用ファイルを作る。A4 縦、ID 1〜6 の 6 個(1 辺 30 mm)。
配置は src/hand_sim/fixed_board.py と共通。紙には、原点の十字、100 mm の線、各マーカー中心の座標(板の座標。mm)を印刷する。
PDF と PNG(300 dpi)の両方を出す(PDF は、ビューアによって文字化けすることがあるため、PNG でも見られるようにする)。紙面の文字は英数字だけ。
印刷は「実際のサイズ」100%。ページ下の 100 mm の線を定規で確かめる。マーカーのまわりの白い余白は切らない(検出に必要)。

使い方(この .venv で実行):
    .venv/bin/python scripts/make_fixed_board.py        # outputs/fixed_board_A4.pdf と .png
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # 画面を出さずにファイルだけ作る設定。pyplot を import する前に書く。[頻度] ヘッドレスでの作図では定番
import matplotlib.pyplot as plt  # noqa: E402
from hand_sim.aruco_marker import make_marker_image  # noqa: E402
from hand_sim.fixed_board import (  # noqa: E402
    _PAGE_CENTERS_MM, ORIGIN_PAGE_MM, PAGE_H_MM, PAGE_W_MM, SIDE_MM, board_marker_centers_mm,
)

MM = 25.4  # 1 インチ = 25.4 mm。matplotlib の図の大きさは、インチで指定する
fig = plt.figure(figsize=(PAGE_W_MM / MM, PAGE_H_MM / MM))
ax = fig.add_axes([0, 0, 1, 1])  # ページ全体を使う軸。単位を mm にする
ax.axis("off")

centers = board_marker_centers_mm()
for i, (px, py) in _PAGE_CENTERS_MM.items():
    x0, y0 = px - SIDE_MM / 2, py - SIDE_MM / 2
    # extent = 画像を置く範囲 [左, 右, 下, 上](mm)。y 軸が下向きなので「下」の値のほうが大きい。interpolation="nearest" = 白黒の境目をぼかさない
    ax.imshow(make_marker_image(i, 600), cmap="gray", vmin=0, vmax=255, extent=[x0, x0 + SIDE_MM, y0 + SIDE_MM, y0], interpolation="nearest")
    x, y = centers[i]
    ax.text(px, py + SIDE_MM / 2 + 7, f"ID {i}  ({x:.0f}, {y:.0f}) mm", ha="center", va="center", fontsize=8)

# 原点の十字(長さ 20 mm)と、軸の矢印。板の座標: x = 右、y = 上
ox, oy = ORIGIN_PAGE_MM
ax.plot([ox - 10, ox + 10], [oy, oy], color="black", lw=1.0)
ax.plot([ox, ox], [oy - 10, oy + 10], color="black", lw=1.0)
ax.annotate("", xy=(ox + 28, oy), xytext=(ox, oy), arrowprops=dict(arrowstyle="->", color="red", lw=1.5))
ax.annotate("", xy=(ox, oy - 28), xytext=(ox, oy), arrowprops=dict(arrowstyle="->", color="red", lw=1.5))
ax.text(ox + 31, oy + 1, "x", color="red", fontsize=10, va="center")
ax.text(ox + 1, oy - 31, "y", color="red", fontsize=10, ha="center", va="bottom")
ax.text(ox + 14, oy + 8, "ORIGIN (0, 0)", fontsize=8, ha="left", va="center")

ax.text(PAGE_W_MM / 2, 15, "Fixed ArUco board (for camera calibration)  DICT_4X4_50, IDs 1-6, side 30.0 mm", ha="center", va="center", fontsize=8)
ax.text(PAGE_W_MM / 2, 274, "Print on A4 at 100% (not 'fit to page'). Do NOT cut the white margins. Glue flat on a rigid board.", ha="center", va="center", fontsize=7)
bar_y = 283.0
x_bar = (PAGE_W_MM - 100) / 2
ax.plot([x_bar, x_bar + 100], [bar_y, bar_y], color="black", lw=1.2)  # 100 mm の線。印刷後に定規で確かめる
for x in (x_bar, x_bar + 100):
    ax.plot([x, x], [bar_y - 2, bar_y + 2], color="black", lw=1.2)
ax.text(PAGE_W_MM / 2, bar_y + 6, "^ This line must be 100 mm. Check with a ruler.", ha="center", va="center", fontsize=7)
ax.set_xlim(0, PAGE_W_MM)  # imshow が軸の範囲を書き換えるので、最後に決め直す
ax.set_ylim(PAGE_H_MM, 0)  # 上が 0、下が PAGE_H_MM(紙の上から測る向き)

out_dir = ROOT / "outputs"
out_dir.mkdir(exist_ok=True)
pdf, png = out_dir / "fixed_board_A4.pdf", out_dir / "fixed_board_A4.png"
fig.savefig(pdf)
fig.savefig(png, dpi=300)
plt.close(fig)

# 自分で確かめる: 作った PNG を読み直し、ID 1〜6 が読めるか。角の位置から、マーカー中心が設計どおり(紙の mm)かも見る
import cv2  # noqa: E402
import numpy as np  # noqa: E402

img = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
corners, ids, _ = detector.detectMarkers(img)
read = sorted(ids.ravel().tolist()) if ids is not None else []
print(f"作成: {pdf.relative_to(ROOT)}、{png.relative_to(ROOT)}")
print(f"OpenCV で読めた ID: {read}(期待: [1, 2, 3, 4, 5, 6])→ {'OK' if read == [1, 2, 3, 4, 5, 6] else 'NG'}")
px_per_mm = 300 / MM
worst = 0.0
for c, i in zip(corners, ids.ravel()):
    center_mm = c.reshape(4, 2).mean(axis=0) / px_per_mm  # 4 つの角の平均 = 中心。画素 → mm
    worst = max(worst, float(np.linalg.norm(center_mm - np.array(_PAGE_CENTERS_MM[int(i)]))))
print(f"画像から読んだ中心と設計値の差の最大(0.2 mm 未満で OK): {worst:.2f} mm → {'OK' if worst < 0.2 else 'NG'}")
print("\n板の座標(原点 = 十字、x = 右、y = 上)でのマーカー中心 [mm]:")
for i, (x, y) in centers.items():
    print(f"  ID {i}: x = {x:6.1f}, y = {y:6.1f}")
