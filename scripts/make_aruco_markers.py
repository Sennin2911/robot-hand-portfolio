"""【概要】ロボットに貼る ArUco マーカーの、印刷用ファイルを作る。A4 縦に、ID 0〜(個数 − 1)のマーカーを、同じ大きさで並べる。
マーカーの 1 辺は、既定 30.0 mm(黒い枠の外側から外側まで)。PDF は、大きさ(mm)が正確になるように作る(印刷するときは、必ず「実際のサイズ」「100%」で。
「ページに合わせる」は、大きさが変わるので使わない)。印刷したら、ページの下の 100 mm の線を定規で測って、大きさを確かめる。
マーカーのまわりの白い余白(1 辺の約 0.2 倍)は、検出に必要なので、切るときも、白を残すこと。

使い方(この .venv で実行):
    .venv/bin/python scripts/make_aruco_markers.py                 # outputs/aruco_markers_A4.pdf と .png(ID 0〜5、30 mm)
    .venv/bin/python scripts/make_aruco_markers.py --side 40 --count 4
「検出に使う 1 辺の長さ」は、ここで決めた値(--side)と、同じにすること(scripts/view_real_markers.py の --side)。印刷の大きさを測り直して、違えば、測った値を使う。
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # 画面を出さずにファイルだけ作る設定。pyplot を import する前に書く(後だと効かない)。[頻度] ヘッドレスでの作図では定番
import matplotlib.pyplot as plt  # noqa: E402
from hand_sim.aruco_marker import make_marker_image  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--side", type=float, default=30.0, help="マーカーの 1 辺 [mm](黒い枠の外側から外側まで)")
parser.add_argument("--count", type=int, default=6, help="作るマーカーの個数(ID 0 から順に)")
args = parser.parse_args()

plt.rcParams["font.family"] = "Noto Sans CJK JP"  # 日本語の文字化け(□□□)を避けるフォント指定
PAGE_W, PAGE_H = 210.0, 297.0  # A4 を縦に使う [mm]
MM = 25.4  # 1 インチ = 25.4 mm。matplotlib の図の大きさは、インチで指定する
SIDE = args.side
CELL = SIDE * 1.6  # 1 個あたりの区画(マーカー + 余白 + 切り取りの線)の大きさ [mm]
COLS = int((PAGE_W - 20) // CELL)

fig = plt.figure(figsize=(PAGE_W / MM, PAGE_H / MM))
ax = fig.add_axes([0, 0, 1, 1])  # ページ全体を使う軸。座標の単位を mm にする(下の xlim / ylim)
ax.set_xlim(0, PAGE_W)
ax.set_ylim(PAGE_H, 0)  # 上が 0、下が PAGE_H(紙の上から測る向き)
ax.axis("off")
x_start = (PAGE_W - COLS * CELL) / 2
for i in range(args.count):
    r, c = divmod(i, COLS)  # divmod(a, b) = (商, 余り)。i 番目を「何行目の何列目」に直す。[頻度] 格子に並べるとき、よく使う
    x0, y0 = x_start + c * CELL + (CELL - SIDE) / 2, 15 + r * CELL + (CELL - SIDE) / 2 - 3
    # extent = 画像を置く範囲 [左, 右, 下, 上](mm)。y 軸が下向きなので、「下」の値のほうが大きい。interpolation="nearest" = ぼかさず、白黒の境目を、くっきり出す。
    ax.imshow(make_marker_image(i, 600), cmap="gray", vmin=0, vmax=255, extent=[x0, x0 + SIDE, y0 + SIDE, y0], interpolation="nearest")
    ax.text(x0 + SIDE / 2, y0 + SIDE + 4.5, f"ID {i}", ha="center", va="center", fontsize=8)
ax.set_xlim(0, PAGE_W)  # imshow が、軸の範囲を書き換えるので、決め直す
ax.set_ylim(PAGE_H, 0)
y_note = 15 + ((args.count - 1) // COLS + 1) * CELL + 8
ax.text(PAGE_W / 2, y_note, f"ArUco DICT_4X4_50、1 辺 {SIDE:.1f} mm(黒い枠の外側から外側)。A4 縦、100% で印刷(「ページに合わせる」は使わない)。白い余白を残して切る",
        ha="center", va="center", fontsize=7)
bar_y = y_note + 10
x_bar = (PAGE_W - 100) / 2
ax.plot([x_bar, x_bar + 100], [bar_y, bar_y], color="black", lw=1.2)  # 100 mm の線。印刷した後、定規で測って、100 mm になっているか確かめる
for x in (x_bar, x_bar + 100):
    ax.plot([x, x], [bar_y - 2, bar_y + 2], color="black", lw=1.2)
ax.text(PAGE_W / 2, bar_y + 6, "↑ この線が 100 mm か、定規で確かめる(違えば、印刷の倍率を直す)", ha="center", va="center", fontsize=7)

out_dir = ROOT / "outputs"
out_dir.mkdir(exist_ok=True)
pdf, png = out_dir / "aruco_markers_A4.pdf", out_dir / "aruco_markers_A4.png"
fig.savefig(pdf)  # PDF は、ベクターなので、拡大しても粗くならず、大きさが正確(画像は、そのままの大きさで埋め込まれる)
fig.savefig(png, dpi=300)
plt.close(fig)

# 自分で確かめる: 作った PNG を読み直し、全部のマーカーが、正しい ID で読めるか
import cv2  # noqa: E402

img = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
_corners, ids, _ = detector.detectMarkers(img)
read = sorted(ids.ravel().tolist()) if ids is not None else []
print(f"作成: {pdf.relative_to(ROOT)}、{png.relative_to(ROOT)}(1 辺 {SIDE:.1f} mm、{args.count} 個)")
print(f"OpenCV で読めた ID: {read}(期待: {list(range(args.count))})→ {'OK' if read == list(range(args.count)) else 'NG'}")
