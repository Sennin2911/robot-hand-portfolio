"""【概要】実機の試験(outputs/real_trials.csv)で、立方体を「どこに、どの向きで」置いたかを、机の上の地図にして outputs/trial_coverage.png に描く。
目的: 置いた場所が偏っていないかを目で見て、次に置くとデータが増える場所(まだ試していない場所)を探す。
左: 机を上から見た図。点 = 試験の立方体の最初の位置(丸 = 成功、× = 失敗、灰色の菱形 = 動かす前に止まった)。短い線 = 立方体の向き。
    薄い網掛けのマス(4 cm 角)= まだ 1 回も試していない、届く範囲のマス。青い円弧 = 届く限界(肩の軸から 23 cm)。四角 = 置き場、小さい四角 = 板のマーカー。
右: 立方体の向き(0〜90°)の分布。空いている区間も、試す向きの候補。
無効(メモが「無効」で始まる)行は使わない。使い方: .venv/bin/python scripts/plot_trial_coverage.py
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # 画面を開かず、ファイルに描く設定。[頻度] サーバや自動実行で画像だけ作るときの定番。import pyplot より前に書く決まり
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from hand_sim.cube_world import PADS, WORKSPACE_X_M  # noqa: E402
from hand_sim.fixed_board import board_marker_corners_world  # noqa: E402

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "sans-serif"]  # 日本語の文字が出るフォント(無ければ四角になる)
AXIS_X = 0.0388  # 肩の軸の x [m](原点の 38.8 mm 前)
REACH = 0.23  # 届く限界(肩の軸から)[m]。plan_pick_test.py と同じ
CELL = 0.04  # 空き具合を見るマスの大きさ [m]
Y_MIN, Y_MAX = -0.20, 0.10  # 立方体を置く範囲の、左右(カメラに写って、置き場と重ならない範囲の目安)

rows = [r for r in csv.DictReader(open(ROOT / "outputs" / "real_trials.csv")) if not r["note"].startswith("無効") and r["true_x"] != ""]
if not rows:
    sys.exit("まだ、位置の分かる試験が無い")
X, Y = np.array([float(r["true_x"]) for r in rows]), np.array([float(r["true_y"]) for r in rows])
YAW = np.array([float(r["true_yaw_deg"]) if r["true_yaw_deg"] != "" else np.nan for r in rows])
OK = np.array([r["success"] == "True" for r in rows])
RAN = np.array([r["cycle_s"] != "" for r in rows])  # 腕を動かした試験(動かす前に止まったものを除く)

INK, MUTED, GRID = "#1f2933", "#6b7785", "#d9dee4"
C_OK, C_NG, C_SKIP = "#2a6fb0", "#c2570c", "#8a94a0"  # 成功 = 青、失敗 = 橙(形も変えて、色だけに頼らない)

fig, (ax, axh) = plt.subplots(1, 2, figsize=(13, 6.2), gridspec_kw={"width_ratios": [2.1, 1]})
fig.patch.set_facecolor("white")

# 左: 机の上の地図。横軸 = y(左が +、ロボットの左側が画面の左になるよう反転)、縦軸 = x(前)
x0, x1 = 0.08, 0.30
ax.set_xlim(Y_MAX + 0.12, Y_MIN - 0.03)  # 反転: 大きい y が左
ax.set_ylim(0.025, x1)
ax.set_aspect("equal")
nx, ny = int(round((Y_MAX - Y_MIN) / CELL)), int(round((x1 - x0) / CELL))
empty = []
for i in range(ny):
    for j in range(nx):
        cx_lo, cy_lo = x0 + i * CELL, Y_MIN + j * CELL
        cx, cy = cx_lo + CELL / 2, cy_lo + CELL / 2
        n = int(((X >= cx_lo) & (X < cx_lo + CELL) & (Y >= cy_lo) & (Y < cy_lo + CELL) & RAN).sum())
        if np.hypot(cx - AXIS_X, cy) <= REACH - 0.015 and cx >= WORKSPACE_X_M[0] + 0.02:  # 届く範囲のマスだけ
            if n == 0:
                ax.add_patch(plt.Rectangle((cy_lo, cx_lo), CELL, CELL, facecolor="none", edgecolor=GRID, hatch="///", linewidth=0))
                empty.append((cx, cy))
            ax.add_patch(plt.Rectangle((cy_lo, cx_lo), CELL, CELL, facecolor="none", edgecolor=GRID, linewidth=0.8))
            ax.text(cy, cx_lo + CELL * 0.1, f"{n}", ha="center", va="bottom", fontsize=8, color=MUTED)
th = np.radians(np.linspace(-120, 120, 200))
ax.plot(REACH * np.sin(th) * -1, AXIS_X + REACH * np.cos(th), color=C_OK, linewidth=1.2, linestyle="--")
ax.text(-0.17, AXIS_X + REACH * 0.83 + 0.025, "届く限界\n(肩の軸から 23 cm)", color=C_OK, fontsize=8, ha="left")
for color, (px, py) in PADS.items():
    if py <= Y_MAX + 0.12:
        ax.add_patch(plt.Rectangle((py - 0.03, px - 0.03), 0.06, 0.06, facecolor="none", edgecolor=INK, linewidth=1.2))
for q in board_marker_corners_world().values():
    ax.add_patch(plt.Polygon(q[:, [1, 0]], closed=True, facecolor="#cfd5dc", edgecolor="none"))
ax.add_patch(plt.Circle((0, AXIS_X), 0.012, color=INK))
ax.text(0.0, AXIS_X + 0.016, "ロボットの肩", ha="center", va="bottom", fontsize=8, color=INK)
for m, sel, col, lab in (("o", OK & RAN, C_OK, "成功"), ("X", ~OK & RAN, C_NG, "失敗(実行した)"), ("D", ~RAN, C_SKIP, "動かす前に止まった")):
    if sel.any():
        ax.scatter(Y[sel], X[sel], marker=m, s=70, c=col, edgecolors="white", linewidths=1.2, label=f"{lab} {int(sel.sum())}", zorder=3)
for x, y, yaw in zip(X, Y, YAW):  # 向きの線(世界座標の yaw。x 軸から反時計回り)
    if not np.isnan(yaw):
        d = np.radians(yaw)
        ax.plot([y - 0.012 * np.sin(d), y + 0.012 * np.sin(d)], [x - 0.012 * np.cos(d), x + 0.012 * np.cos(d)], color=INK, linewidth=1, zorder=4)
ax.set_xlabel("左 ← y [m] → 右(ロボットから見て)", color=MUTED)
ax.set_ylabel("前 x [m](肩から遠い方が上)", color=MUTED)
ax.set_title("立方体の最初の位置(丸・× = 試験、線 = 向き、網掛け = まだ 0 回のマス、数字 = そのマスの回数)", fontsize=10, color=INK, loc="left")
ax.legend(loc="upper left", fontsize=8, frameon=False)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# 右: 向きの分布
bins = np.arange(0, 91, 15)
cnt, _ = np.histogram(YAW[~np.isnan(YAW)], bins=bins)
axh.barh([f"{b}〜{b + 15}°" for b in bins[:-1]], cnt, color=C_OK, height=0.6)
for k, c in enumerate(cnt):
    axh.text(c + 0.1, k, str(int(c)), va="center", fontsize=9, color=INK)
axh.set_title("立方体の向き(0〜90°)の回数", fontsize=10, color=INK, loc="left")
axh.set_xlim(0, max(cnt.max() + 2, 4))
axh.invert_yaxis()
for s in ("top", "right"):
    axh.spines[s].set_visible(False)
fig.tight_layout()
out = ROOT / "outputs" / "trial_coverage.png"
fig.savefig(out, dpi=130)
print(f"保存: {out}(試験 {len(rows)} 回ぶん。腕を動かしたのは {int(RAN.sum())} 回)")
if empty:
    empty.sort(key=lambda p: np.hypot(p[0] - 0.19, p[1]))  # 板の真ん中に近い順
    print("まだ 0 回のマス(中心の x, y [mm])の候補: " + "、".join(f"({x * 1000:.0f}, {y * 1000:.0f})" for x, y in empty[:8]))
low = np.where(cnt == 0)[0]
if len(low):
    print("まだ 0 回の向き: " + "、".join(f"{bins[k]}〜{bins[k] + 15}°" for k in low))
