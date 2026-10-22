"""【概要】docs/webcam_calibration.md に載せる図(docs/img/webcam_*.png)を作る。
撮ったチェスボードの画像(outputs/chessboard_shots/shot_*.png)から、実際の較正をやり直して、その結果で描く。カメラや実機は動かさない。
図: (1) 歪みのモデルと大きさ、(2) 較正の流れ、(3) 歪みを無視すると何画素ずれるか(再投影の実例)、(4) 角が通った範囲と歪みの大きさ、
(5) 誤差の大きさ(画素 → mm)、(6) 傾けた画像が要る理由。
画像そのものは載せない(部屋や人が写っているため。点と線だけで描く)。

使い方(この .venv で実行):
    .venv/bin/python scripts/make_webcam_calibration_figures.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)。[頻度] 自作モジュールを使うスクリプトでは定番
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")  # 画面を出さずに PNG だけ作る設定。pyplot を import する前に書く(後だと効かない)。[頻度] ヘッドレスでの作図では定番
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

from hand_sim.intrinsics import find_corners, object_points  # noqa: E402

plt.rcParams["font.family"] = "Noto Sans CJK JP"  # 日本語の文字化け(□□□)を避けるフォント指定
plt.rcParams["axes.unicode_minus"] = False
OUT = ROOT / "docs" / "img"
OUT.mkdir(exist_ok=True)
W, H = 1920, 1080


def calibrate(corner_list):
    """cv2.calibrateCamera を呼んで、(K, dist, rvecs, tvecs, 画像ごとの誤差)を返す。
    rvec・tvec は「盤がカメラから見てどこにあるか」(画像ごとに別の値)。内部パラメータ K・dist は全画像で 1 組。"""
    obj = [object_points()] * len(corner_list)
    _, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj, corner_list, (W, H), None, None)
    dist = dist.ravel()  # OpenCV は (1, 5) の形で返す。(5,) の 1 次元に直す(中身は同じ)。[頻度] OpenCV の結果を使うときの定番
    errs = []
    for o, c, r, t in zip(obj, corner_list, rvecs, tvecs):
        p, _ = cv2.projectPoints(o, r, t, K, dist)
        errs.append(float(np.sqrt(np.mean(np.sum((p - c) ** 2, axis=2)))))
    return K, dist, rvecs, tvecs, errs


def distort_pixels(u, v, K, dist):
    """歪みのない理想の画素 (u, v) が、レンズを通ると、どの画素に写るかを返す(OpenCV の歪みのモデルそのまま)。
    手順: 画素 → 正規化座標(中心を 0、焦点距離で割る)→ 歪みの式 → 画素に戻す。"""
    k1, k2, p1, p2, k3 = dist
    x, y = (u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1]
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3  # 中心から離れるほど、画素が内側へ縮む(k1 が負 = 樽型)
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return xd * K[0, 0] + K[0, 2], yd * K[1, 1] + K[1, 2]


# ---- 画像から角を探して、実機と同じ手順で較正をやり直す(scripts/calibrate_webcam_intrinsics.py と同じ: 誤差が大きい画像を除いて、もう一度) ----
names, corner_list = [], []
for path in sorted((ROOT / "outputs" / "chessboard_shots").glob("shot_*.png")):
    c = find_corners(cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB))
    if c is not None:
        names.append(path.name)
        corner_list.append(c)
K, dist, rvecs, tvecs, errs = calibrate(corner_list)
limit = max(1.5, 3 * float(np.median(errs)))
keep = [i for i, e in enumerate(errs) if e <= limit]
dropped = [names[i] for i in range(len(names)) if i not in keep]
corner_list = [corner_list[i] for i in keep]
names = [names[i] for i in keep]
K, dist, rvecs, tvecs, errs = calibrate(corner_list)
rms = float(np.sqrt(np.mean(np.square(errs))))
print(f"使った画像 {len(corner_list)} 枚(除いた: {dropped})、再投影誤差 {rms:.3f} 画素、K の fx = {K[0, 0]:.1f}")
tilts = [float(np.degrees(np.arccos(min(1.0, abs(cv2.Rodrigues(r)[0][2, 2]))))) for r in rvecs]

# 歪みの大きさ(画素)を、画像全体の格子で計算しておく(図 1・4 で使う)
gu, gv = np.meshgrid(np.linspace(0, W, 97), np.linspace(0, H, 55))
du, dv = distort_pixels(gu, gv, K, dist)
shift = np.hypot(du - gu, dv - gv)  # 理想の位置から、実際に写る位置までのずれ [画素]
corner_shift = float(shift[0, 0])
print(f"画像の四隅での歪みのずれ: 約 {corner_shift:.0f} 画素、中心から 1/2 の所: 約 {float(shift[27, 72]):.0f} 画素")

# ==== 図 1: 歪みのモデル ====
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.25, 1]})
a = ax[0]
for x in np.linspace(0, W, 13):  # 理想の格子(縦線)。歪みのないカメラなら、直線のまま写る
    v = np.linspace(0, H, 100)
    a.plot([x, x], [0, H], color="#bbbbbb", lw=1)
    xd, yd = distort_pixels(np.full_like(v, x), v, K, dist)
    a.plot(xd, yd, color="#d62728", lw=1.4)
for y in np.linspace(0, H, 8):
    u = np.linspace(0, W, 100)
    a.plot([0, W], [y, y], color="#bbbbbb", lw=1)
    xd, yd = distort_pixels(u, np.full_like(u, y), K, dist)
    a.plot(xd, yd, color="#d62728", lw=1.4)
a.add_patch(Rectangle((0, 0), W, H, fill=False, ec="k", lw=1.5))
a.plot([], [], color="#bbbbbb", label="歪みのない理想の格子(直線)")
a.plot([], [], color="#d62728", label="このカメラに写る格子(樽型)")
a.set_xlim(-60, W + 60)
a.set_ylim(H + 60, -60)
a.set_aspect("equal")
a.set_title("同じ格子が、このカメラでは、こう曲がって写る\n(求めた歪みの係数で描いた。曲がり方は、実際のまま)", fontsize=10)
a.legend(loc="lower center", fontsize=8, ncol=1, framealpha=0.95)
a.set_xlabel("画像の横 [画素]")
a.set_ylabel("画像の縦 [画素]")
a = ax[1]
r = np.linspace(0, np.hypot(W / 2, H / 2), 200)  # 画像の中心からの距離 [画素]
uu, vv = K[0, 2] + r * 0.8, K[1, 2] + r * 0.6  # 斜め方向の線の上で見る
du2, dv2 = distort_pixels(uu, vv, K, dist)
a.plot(r, np.hypot(du2 - uu, dv2 - vv), color="#d62728", lw=2)
a.axvline(H / 2, color="gray", ls=":")
a.axvline(W / 2, color="gray", ls=":")
a.text(H / 2, 2, "画像の上下の端", rotation=90, va="bottom", ha="right", fontsize=8, color="gray")
a.text(W / 2, 2, "左右の端", rotation=90, va="bottom", ha="right", fontsize=8, color="gray")
a.set_xlabel("画像の中心からの距離 [画素]")
a.set_ylabel("歪みによるずれ [画素]")
a.set_title("歪みは、中心から離れるほど急に大きくなる\n(中心はほぼ 0。四隅で約 %d 画素)" % round(corner_shift), fontsize=10)
a.grid(alpha=0.3)
a.text(0.04, 0.9, "歪みの式(半径方向):\n  r' = r × (1 + k1·r^2 + k2·r^4 + k3·r^6)\n  r: 中心からの距離(焦点距離で割った値)\n求めた値: k1=%.3f  k2=%.3f  k3=%.3f" % (dist[0], dist[1], dist[4]),
       transform=a.transAxes, va="top", fontsize=8.5, bbox=dict(fc="#f5f5f5", ec="#cccccc"))
fig.tight_layout()
fig.savefig(OUT / "webcam_distortion_model.png", dpi=130)
plt.close(fig)

# ==== 図 2: 較正の流れ(箱と矢印) ====
fig, ax = plt.subplots(figsize=(13, 4.6))
ax.set_xlim(0, 13)
ax.set_ylim(0, 4.6)
ax.axis("off")


def box(x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=color, ec="#555555"))
    ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="top", fontsize=10.5, weight="bold")
    ax.text(x + w / 2, y + h - 0.7, body, ha="center", va="top", fontsize=8.8, linespacing=1.5)


def arrow(x0, y0, x1, y1, text=""):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.8, color="#333333"))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.12, text, ha="center", fontsize=8.5)


box(0.1, 2.6, 3.0, 1.8, "① 盤を撮る(25 枚)", "位置・傾き・距離を変えて\n撮る。カメラは同じ。\n盤は平らで、マスの\n大きさが全部同じ", "#e8f1fb")
box(4.0, 2.6, 3.0, 1.8, "② 角の画素を見つける", "盤の内側の角 54 個の、\n画像の中の位置を、\n1 画素より細かく求める\n(OpenCV が自動で行う)", "#e8f1fb")
box(7.9, 2.6, 5.0, 1.8, "③ 仮のパラメータで、角を画像に投影する", "盤の角は、実際の盤の上で、25 mm おきに並ぶ\n(この「正解の並び」が既知であることが、手がかり)。\n仮の焦点距離・中心・歪み、仮の盤の位置・向きで、\n「角は、画像のここに写るはず」と計算する", "#fdf2e0")
box(7.9, 0.1, 5.0, 1.8, "④ 実際の角とのずれを、小さくする", "③ の位置と、② の位置の差(再投影誤差)を、\n全画像・全部の角で足し合わせる。\nこの合計が最小になるように、パラメータを少しずつ直す\n(最小二乗法。繰り返し)", "#fdf2e0")
box(4.0, 0.1, 3.0, 1.8, "⑤ 結果", "焦点距離 fx, fy\n画像の中心 cx, cy\n歪み k1, k2, p1, p2, k3\n+ 再投影誤差(= 当てはまりの悪さ)", "#e5f5e3")
arrow(3.15, 3.5, 3.95, 3.5)
arrow(7.05, 3.5, 7.85, 3.5)
arrow(10.4, 2.55, 10.4, 1.95)
arrow(7.85, 1.0, 7.05, 1.0)
ax.text(7.45, 1.15, "収束", ha="center", fontsize=8.5)
ax.annotate("", xy=(13.0, 2.2), xytext=(13.0, 3.0), arrowprops=dict(arrowstyle="-", lw=0))
ax.text(0.1, 0.9, "盤の位置・向き(画像ごとに別)は、\n計算の途中で、画像ごとに求める。\n求めたいのは、全画像に共通の\nカメラの内部パラメータだけ。", fontsize=8.8, va="center",
        bbox=dict(fc="#f5f5f5", ec="#cccccc"))
ax.set_title("内部パラメータの求め方(歪みは、④ で、他のパラメータと一緒に求まる)", fontsize=11)
fig.savefig(OUT / "webcam_calib_flow.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# ==== 図 3: 再投影の実例(歪みを無視すると、何画素ずれるか) ====
# 歪みの影響がいちばん分かりやすい画像を選ぶ(盤が、画像の端に寄っている画像ほど、歪みを無視したときのずれが大きい)。
# 先に全画像で、歪みを無視したときの平均のずれを計算して、いちばん大きい画像の番号を取る(np.argmax = 最大の要素の番号。[頻度] 定番)
obj = object_points()
no_dist_err = []
for c, r, t in zip(corner_list, rvecs, tvecs):
    q, _ = cv2.projectPoints(obj, r, t, K, np.zeros(5))
    no_dist_err.append(float(np.mean(np.hypot(*(q.reshape(-1, 2) - c.reshape(-1, 2)).T))))
idx = int(np.argmax(no_dist_err))
det = corner_list[idx].reshape(-1, 2)
fitted, _ = cv2.projectPoints(obj, rvecs[idx], tvecs[idx], K, dist)
no_dist, _ = cv2.projectPoints(obj, rvecs[idx], tvecs[idx], K, np.zeros(5))  # 歪みの係数を 0 にして、同じ盤の位置で投影する
fitted, no_dist = fitted.reshape(-1, 2), no_dist.reshape(-1, 2)
e_fit, e_no = np.hypot(*(fitted - det).T), np.hypot(*(no_dist - det).T)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
for a, proj, e, title in [(ax[0], no_dist, e_no, "歪みを無視した(係数を 0 にした)とき"), (ax[1], fitted, e_fit, "求めた歪みを使ったとき")]:
    a.scatter(*det.T, s=26, c="#d62728", label="実際に見つかった角(画像から)", zorder=3)
    a.scatter(*proj.T, s=26, facecolors="none", edgecolors="#1f77b4", label="計算で投影した角", zorder=3)
    scale = 1 if a is ax[0] else 40  # 右は、ずれが小さすぎて見えないので、ずれの線を 40 倍に伸ばして描く
    for d, p in zip(det, proj):
        a.plot([d[0], d[0] + (p[0] - d[0]) * scale], [d[1], d[1] + (p[1] - d[1]) * scale], color="k", lw=0.9, zorder=2)
    a.set_aspect("equal")
    a.invert_yaxis()
    a.set_title(f"{title}\nずれ: 平均 {e.mean():.2f} 画素、最大 {e.max():.2f} 画素" + ("" if scale == 1 else "(ずれの線は 40 倍に伸ばした)"), fontsize=10)
    a.set_xlabel("画像の横 [画素]")
    a.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
ax[0].set_ylabel("画像の縦 [画素]")
fig.suptitle(f"{names[idx]}(歪みの影響が、いちばん大きく出た画像)の、角 54 個", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "webcam_reproj_demo.png", dpi=130)
plt.close(fig)
print(f"図 3: 歪みを無視すると、平均 {e_no.mean():.2f}、最大 {e_no.max():.2f} 画素ずれる。求めた歪みを使うと、平均 {e_fit.mean():.2f}、最大 {e_fit.max():.2f} 画素")

# ==== 図 4: 角が通った範囲と、歪みの大きさ ====
allp = np.concatenate([c.reshape(-1, 2) for c in corner_list])
fig, ax = plt.subplots(figsize=(10, 5.9))
cs = ax.contourf(gu, gv, shift, levels=[0, 5, 10, 20, 30, 50, 80, 120, 220], cmap="Oranges", alpha=0.8)
fig.colorbar(cs, ax=ax, label="歪みによるずれ [画素]", fraction=0.035)
ax.scatter(*allp.T, s=2, c="#1f4e79", label=f"盤の角(全 {len(corner_list)} 枚。{len(allp)} 点)")
for f in (0.1, 0.9):
    ax.axvline(W * f, color="gray", ls=":", lw=1)
    ax.axhline(H * f, color="gray", ls=":", lw=1)
ax.add_patch(Rectangle((0, 0), W, H, fill=False, ec="k", lw=1.5))
cov = [allp[:, 0].min() / W, allp[:, 0].max() / W, allp[:, 1].min() / H, allp[:, 1].max() / H]
ax.set_xlim(-30, W + 30)
ax.set_ylim(H + 30, -30)
ax.set_aspect("equal")
ax.set_xlabel("画像の横 [画素]")
ax.set_ylabel("画像の縦 [画素]")
ax.set_title(f"角が通った範囲: 横 {cov[0]:.2f}〜{cov[1]:.2f}、縦 {cov[2]:.2f}〜{cov[3]:.2f}(点線は 0.1 / 0.9)。色が濃いほど、歪みが大きい", fontsize=10)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "webcam_coverage.png", dpi=130)
plt.close(fig)

# ==== 図 5: 誤差の大きさ ====
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
a = ax[0]
order = np.arange(len(errs))
a.bar(order, errs, color="#4c78a8")
a.axhline(1.0, color="#d62728", ls="--")
a.text(len(errs) - 0.5, 1.03, "良い較正の目安(1 画素)", ha="right", fontsize=8.5, color="#d62728")
a.axhline(rms, color="k", lw=1)
a.text(len(errs) - 0.5, rms + 0.03, f"全体 {rms:.3f} 画素", ha="right", fontsize=8.5)
a.set_xticks(order[::2])
a.set_xticklabels([n.replace("shot_", "").replace(".png", "") for n in np.array(names)[::2]], fontsize=8)
a.set_xlabel("画像の番号(shot_NN)")
a.set_ylabel("再投影誤差 [画素]")
a.set_ylim(0, 1.2)
a.set_title("画像ごとの誤差: 全部、1 画素より、ずっと小さい" + (f"\n(誤差が大きく除いた画像: {', '.join(dropped)})" if dropped else ""), fontsize=10)
a = ax[1]
dist_cm = np.array([40, 60, 90])
mm_per_px = dist_cm * 10 / K[0, 0]  # 距離 [mm] ÷ 焦点距離 [画素] = 1 画素が表す長さ [mm](針穴カメラの相似)
labels = [f"カメラから {d} cm" for d in dist_cm]
vals = mm_per_px * rms
a.barh(labels, vals, color="#4c78a8")
for i, v in enumerate(vals):
    a.text(v + 0.2, i, f"{v:.2f} mm", va="center", fontsize=9)
a.axvline(10, color="#d62728", ls="--")
a.text(9.8, 2.45, "目標: カメラの位置の誤差\n10 mm 以内", ha="right", va="top", fontsize=8.5, color="#d62728")
a.set_xlim(0, 11)
a.set_xlabel("%.3f 画素を、実際の長さに直したもの [mm]" % rms)
a.set_title("実際の長さでは、0.1〜0.2 mm 程度", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "webcam_error_scale.png", dpi=130)
plt.close(fig)
print(f"図 5: 1 画素 = {mm_per_px[0]:.2f} / {mm_per_px[1]:.2f} / {mm_per_px[2]:.2f} mm(40 / 60 / 90 cm)。誤差 {rms:.3f} 画素 = {vals[0]:.2f} / {vals[1]:.2f} / {vals[2]:.2f} mm")

# ==== 図 6: 傾けた画像が要る理由 ====
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
a = ax[0]
# docs/webcam_calibration.md の「合成データでの結果」(焦点距離 1100 画素、6 回の試行)。範囲(最小〜最大)を、棒で描く。実測ではなく、シミュレーションの値
angles = [0, 10, 20, 30, 40]
lo = [67, 18, 0.8, 0.6, 0.4]
hi = [87, 29, 2.5, 1.1, 0.7]
for x, l, h_ in zip(angles, lo, hi):
    a.bar(x, h_ - l, bottom=l, width=6, color="#d62728" if x < 20 else "#2ca02c")
    a.text(x, h_ * 1.15, f"{l}〜{h_}%", ha="center", fontsize=9)
a.set_yscale("log")
a.set_ylim(0.2, 200)
a.set_xticks(angles)
a.set_xlabel("盤の傾き [度]")
a.set_ylabel("焦点距離 fx の誤差 [%](対数目盛)")
a.set_title("傾きがないと、焦点距離が決まらない(合成データでの確認)\n20° 以上で、誤差が 1〜2% に下がる", fontsize=10)
a.grid(alpha=0.3, which="both")
a = ax[1]
bins = np.arange(0, 50, 5)
n_small = [t for t in tilts if t < 20]
n_big = [t for t in tilts if t >= 20]
a.hist(n_small, bins=bins, color="#d62728", label=f"20° 未満({len(n_small)} 枚)")
a.hist(n_big, bins=bins, color="#2ca02c", label=f"20° 以上({len(n_big)} 枚。必要なのは 8 枚以上)")
a.set_xlabel("盤の傾き [度](画像から求めた値)")
a.set_ylabel("画像の枚数")
a.set_title("今回撮った画像の、傾きの分布", fontsize=10)
a.legend(fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "webcam_tilt_needed.png", dpi=130)
plt.close(fig)
print("図を docs/img/webcam_*.png に保存した")
