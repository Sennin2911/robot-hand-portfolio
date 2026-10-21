"""【概要】カメラの内部パラメータ(視野角・レンズの歪み)を求めるための、チェスボード(白黒の格子)の印刷用ファイルを作る。
A4 を横に使い、1 マス 25.0 mm の 10 × 7 マス(内側の角は 9 × 6 個)。PDF は、大きさ(mm)が正確になるように作る(印刷するときは、必ず「実際のサイズ」「100%」で。「ページに合わせる」は、大きさが変わるので使わない)。
作ったあと、OpenCV で、角が見つかるかを、自分で確かめる(findChessboardCorners)。ページの下には、印刷の大きさを定規で確かめるための、100 mm の線を入れる。

使い方(この .venv で実行):
    .venv/bin/python scripts/make_chessboard.py            # outputs/chessboard_A4.pdf と .png を作る
    .venv/bin/python scripts/make_chessboard.py --screen 1920 1080   # 別のパソコンの画面に映す用の画像(画面の解像度と同じ大きさの PNG)を作る
画面に映す場合の注意は docs/webcam_calibration.md を参照。--screen のときは、紙の PDF は作らない。
"""
import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")  # 画面を出さずにファイルだけ作る設定。pyplot を import する前に書く(後だと効かない)。[頻度] ヘッドレスでの作図では定番
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COLS_S, ROWS_S = 10, 7  # 画面用の盤のマスの数(紙と同じ)
parser = argparse.ArgumentParser()
parser.add_argument("--screen", type=int, nargs=2, metavar=("W", "H"), help="画面に映す用の画像を作る。画面の解像度(幅 高さ。画素)を指定する")
parser.add_argument("--square", type=int, default=None, help="--screen のときの、1 マスの大きさ [画素]。省略すると、画面の 8 割に収まる最大の整数")
args = parser.parse_args()

if args.screen:
    # 画面に映す用: 白い背景の中央に、10 × 7 マスの盤を置いた画像を作る。1 マスは、整数の画素数(中間の色のにじみを避けるため。画面で、100% の大きさで映すこと)。
    sw, sh = args.screen
    sq = args.square or int(min(sw * 0.8 / COLS_S, sh * 0.8 / ROWS_S))
    img = np.full((sh, sw), 255, np.uint8)
    x0s, y0s = (sw - COLS_S * sq) // 2, (sh - ROWS_S * sq) // 2
    for r in range(ROWS_S):
        for c in range(COLS_S):
            if (r + c) % 2 == 0:
                img[y0s + r * sq:y0s + (r + 1) * sq, x0s + c * sq:x0s + (c + 1) * sq] = 0
    out = ROOT / "outputs" / f"chessboard_screen_{sw}x{sh}.png"
    out.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out), img)
    found, corners = cv2.findChessboardCorners(img, (COLS_S - 1, ROWS_S - 1))
    print(f"作成: {out.relative_to(ROOT)}(画面 {sw} × {sh}、1 マス {sq} 画素、盤 {COLS_S * sq} × {ROWS_S * sq} 画素)。OpenCV で角 {COLS_S - 1} × {ROWS_S - 1} 個が見つかる: {found}")
    raise SystemExit(0)

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
    pts = corners.reshape(-1, 2)  # OpenCV のバージョンで、形が違うので、54 × 2 にそろえる
    spacing = np.linalg.norm(pts[1] - pts[0])  # 隣り合う角の間隔 [画素]
    print(f"角の間隔 {spacing:.1f} 画素(200 dpi で 25 mm = {25 / MM * 200:.1f} 画素)")
