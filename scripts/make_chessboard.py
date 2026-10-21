"""【概要】カメラの内部パラメータ(視野角・レンズの歪み)を求めるための、チェスボード(白黒の格子)の印刷用ファイルを作る。
A4 を横に使い、1 マス 25.0 mm の 10 × 7 マス(内側の角は 9 × 6 個)。PDF は、大きさ(mm)が正確になるように作る(印刷するときは、必ず「実際のサイズ」「100%」で。「ページに合わせる」は、大きさが変わるので使わない)。
作ったあと、OpenCV で、角が見つかるかを、自分で確かめる(findChessboardCorners)。ページの下には、印刷の大きさを定規で確かめるための、100 mm の線を入れる。

使い方(この .venv で実行):
    .venv/bin/python scripts/make_chessboard.py            # outputs/chessboard_A4.pdf と .png を作る
"""
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")  # 画面を出さずにファイルだけ作る設定。pyplot を import する前に書く(後だと効かない)。[頻度] ヘッドレスでの作図では定番
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams["font.family"] = "Noto Sans CJK JP"  # 日本語の文字化け(□□□)を避けるフォント指定
COLS, ROWS, SQUARE = 10, 7, 25.0  # マスの数(横、縦)と、1 マスの大きさ [mm]。内側の角は (COLS − 1) × (ROWS − 1) = 9 × 6 個
PAGE_W, PAGE_H = 297.0, 210.0  # A4 を横に使う [mm]
MM = 25.4  # 1 インチ = 25.4 mm。matplotlib の図の大きさは、インチで指定する

fig = plt.figure(figsize=(PAGE_W / MM, PAGE_H / MM))
ax = fig.add_axes([0, 0, 1, 1])  # ページ全体を使う軸。座標の単位を mm にする(下の xlim / ylim)
ax.set_xlim(0, PAGE_W)
ax.set_ylim(PAGE_H, 0)  # 上が 0、下が PAGE_H(紙の上から測る向き)
ax.axis("off")
x0, y0 = (PAGE_W - COLS * SQUARE) / 2, 12.0  # 盤の左上の位置 [mm]。横は中央、縦は上から 12 mm
for r in range(ROWS):
    for c in range(COLS):
        if (r + c) % 2 == 0:  # 市松模様: 行と列の和が偶数のマスを黒にする
            ax.add_patch(Rectangle((x0 + c * SQUARE, y0 + r * SQUARE), SQUARE, SQUARE, facecolor="black", edgecolor="none"))
y_text = y0 + ROWS * SQUARE + 8
ax.text(PAGE_W / 2, y_text, f"1 マス {SQUARE:.1f} mm、{COLS} × {ROWS} マス(内側の角 {COLS - 1} × {ROWS - 1} 個)。A4 横、100% で印刷(「ページに合わせる」は使わない)",
        ha="center", va="center", fontsize=9)
bar_y = y_text + 9
ax.plot([x0, x0 + 100], [bar_y, bar_y], color="black", lw=1.2)  # 100 mm の線。印刷した後、定規で測って、100 mm になっているか確かめる
for x in (x0, x0 + 100):
    ax.plot([x, x], [bar_y - 2, bar_y + 2], color="black", lw=1.2)
ax.text(x0 + 105, bar_y, "← この線が 100 mm か、定規で確かめる(違えば、印刷の倍率を直す)", va="center", fontsize=8)

out_dir = ROOT / "outputs"
out_dir.mkdir(exist_ok=True)
pdf, png = out_dir / "chessboard_A4.pdf", out_dir / "chessboard_A4.png"
fig.savefig(pdf)  # PDF は、ベクターなので、拡大しても粗くならず、大きさが正確
fig.savefig(png, dpi=200)
plt.close(fig)

# 自分で確かめる: PNG を読み直し、OpenCV で、内側の角 9 × 6 個が見つかるか
img = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)  # cv2.imread: 画像を読む。IMREAD_GRAYSCALE: 白黒(1 チャンネル)で読む
found, corners = cv2.findChessboardCorners(img, (COLS - 1, ROWS - 1))
print(f"作成: {pdf.relative_to(ROOT)}、{png.relative_to(ROOT)}")
print(f"OpenCV で、内側の角 {COLS - 1} × {ROWS - 1} 個が見つかる: {found}(角の数 {0 if corners is None else len(corners)})")
if found:
    spacing = np.linalg.norm(corners[1, 0] - corners[0, 0])  # 隣り合う角の間隔 [画素]
    print(f"角の間隔 {spacing:.1f} 画素(200 dpi で 25 mm = {25 / MM * 200:.1f} 画素)")
