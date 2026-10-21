"""【概要】docs/offset_calibration.md に載せる図(docs/img/offset_*.png)を作る。実機は動かさない。
図は 5 枚: (1) pan のずれを真上から見た図、(2) 各関節のオフセットが手先をどちらへ動かすか、
(3) 測った姿勢で、sim の予測と実測を比べた側面図、(4) オフセットの前後の差の棒グラフ、(5) 一定のずれと姿勢で変わるずれの見分け方。
測定値は scripts/fit_joint_offsets.py の MEASURED と同じ(数字を変えたら、両方を直す)。

使い方(この .venv で実行):
    .venv/bin/python scripts/make_offset_figures.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)。
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # 画面を出さずに PNG だけ作る設定。pyplot を import する前に書く(後だと効かない)。[頻度] ヘッドレスでの作図では定番
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Arc  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from hand_sim.sim_real import JOINTS, OFFSET_DEG, real_to_sim  # noqa: E402

plt.rcParams["font.family"] = "Noto Sans CJK JP"  # 日本語の文字化け(□□□)を避けるフォント指定
plt.rcParams["axes.unicode_minus"] = False
OUT = ROOT / "docs" / "img"
OUT.mkdir(exist_ok=True)  # exist_ok=True: すでにあってもエラーにしない。[頻度] 定番

model = mujoco.MjModel.from_xml_path(str(ROOT / "models" / "so101" / "scene.xml"))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
PAN_XY = data.xanchor[model.joint("shoulder_pan").id][:2].copy()  # 肩の回転軸の水平位置

BLUE, RED, GRAY, GREEN = "#1f77b4", "#d62728", "#888888", "#2a9d4b"

# 測定値(fit_joint_offsets.py の MEASURED と同じ)。pose は実機に送った素の指令(オフセットを引かない値)。
# reliable=False は、肩の軸の位置を目測した半径(約 +1 cm の誤差を含む)。
MEAS = [
    {"name": "全関節0", "pose": [0, 0, 0, 0, 0, 10], "tip_z": 25.5},
    {"name": "点1", "pose": [0.0, -65.9, 87.0, -7.7, 0.0, 10.0], "radius": 23.5, "tip_z": 14.5, "reliable": False},
    {"name": "点2", "pose": [0.0, -100.0, 68.0, 68.3, 0.0, 10.0], "radius": 15.5, "tip_z": 13.8, "reliable": True},
    {"name": "点3", "pose": [0.0, -19.8, 55.3, -16.9, 0.0, 10.0], "radius": 30.0, "tip_z": 13.3, "reliable": False},
    {"name": "点4", "pose": [59.9, -65.9, 86.9, -7.6, 3.3, 10.0], "radius": 24.0, "tip_z": 13.7, "reliable": False},
    {"name": "点8", "pose": [0.0, -39.2, 79.9, -10.4, 0.0, 10.0], "radius": 24.0, "tip_z": 8.3, "reliable": True},
]


def arm_points(pose, apply_offset, extra_deg=None):
    """腕の 4 点 (肩の軸 → 肘の軸 → 手首の軸 → 手先) の (半径 r, 高さ z) [cm] を返す。
    r は肩の回転軸からの水平距離。apply_offset=False は「sim の予測」、True は「オフセットを足した実機の姿勢」。
    extra_deg: {関節名: 度} を、さらに足す(各関節を少し動かしたときの手先の動きを見るため)。"""
    q = real_to_sim(model, {f"{n}.pos": v for n, v in zip(JOINTS, pose)}, apply_offset=apply_offset)
    for name, deg in (extra_deg or {}).items():
        q[JOINTS.index(name)] += np.radians(deg)
    data.qpos[:6] = q
    mujoco.mj_forward(model, data)
    pts = [data.xanchor[model.joint(n).id] for n in ("shoulder_lift", "elbow_flex", "wrist_flex")]
    pts.append(data.site_xpos[model.site("gripperframe").id])
    # np.hypot(dx, dy): 水平距離。pan がずれていても、腕の面の中での距離になる
    return [(float(np.hypot(*(p[:2] - PAN_XY))) * 100, float(p[2]) * 100) for p in pts]


# ---------- (1) pan のずれ: 真上から見た図 ----------
def fig_pan():
    r, d0, tilt = 22.5, 6.07, 4.4  # 半径(弦から)、肩の軸の机の端からの距離(C の 28.5 − 22.5)、傾き [度]
    axis = np.array([0.0, d0])

    def tip(angle_deg):  # 正面(+y)から右(+x)へ angle_deg 度の向きの手先の位置
        a = np.radians(angle_deg)
        return axis + r * np.array([np.sin(a), np.cos(a)])

    fig, ax = plt.subplots(figsize=(8, 6.2))
    ax.axhline(0, color="k", lw=3)
    ax.text(-27, -1.6, "机の端", fontsize=11)
    ax.plot(*axis, "ko", ms=9)
    ax.annotate("肩の回転軸\n(端から約 6 cm)", axis, (-19, 9), arrowprops=dict(arrowstyle="->"), fontsize=10)
    A, B, C = tip(60 + tilt), tip(-60 + tilt), tip(tilt)  # 実際の位置(正面が 4.4° 右へ向いている)
    A0, B0, C0 = tip(60), tip(-60), tip(0)  # もしずれがなければ、ここに来るはず
    for P, name, dx in ((A0, "A'", 0.7), (B0, "B'", -2.6), (C0, "C'", -2.6)):
        ax.plot(*P, "o", mfc="none", mec=GRAY, ms=10)
        ax.text(P[0] + dx, P[1] + 0.8, name, color=GRAY, fontsize=10)
    ax.plot([A0[0], B0[0]], [A0[1], B0[1]], "--", color=GRAY, lw=1)
    ax.plot([axis[0], C0[0]], [axis[1], C0[1]], "--", color=GRAY, lw=1)  # 机の端に垂直な線(ずれがないときの正面)
    for P, name, dep in ((A, "A (pan +60)", 16), (B, "B (pan −60)", 19), (C, "C (pan 0)", 28.5)):
        ax.plot([axis[0], P[0]], [axis[1], P[1]], "-", color=BLUE, lw=1.2)
        ax.plot(*P, "o", color=RED, ms=9)
        ax.text(P[0] + 0.8, P[1] - 0.4, f"{name}\n端から {dep} cm", color=RED, fontsize=10)
        ax.plot([P[0], P[0]], [0, P[1]], ":", color=RED, lw=1)  # 端までの垂線(奥行き)
    ax.plot([A[0], B[0]], [A[1], B[1]], "-", color=GREEN, lw=2)
    ax.text(0, 11.5, "弦 39.0 cm → 半径 = 39.0 ÷ 1.732 = 22.5 cm", color=GREEN, fontsize=10, ha="center")
    # 正面のずれの角度を、円弧で示す(Arc: 中心・幅・高さ・開始角・終了角 [度] を指定。角度は +x 方向から反時計回り)
    ax.add_patch(Arc(axis, 20, 20, theta1=90 - tilt, theta2=90, color="k", lw=1.5))
    ax.text(2.2, d0 + 11.5, f"{tilt}°", fontsize=11)
    ax.text(-27.5, 32, f"A と B の端からの距離の差 3 cm ÷ 弦 39 cm\n→ 正面が右へ約 {tilt}° ずれている\n= pan の 0 のずれ +{tilt}°", fontsize=10, va="top")
    ax.set_xlim(-29, 29)
    ax.set_ylim(-3, 33)
    ax.set_aspect("equal")
    ax.set_xlabel("← 左       右 →   [cm]")
    ax.set_ylabel("机の端からの距離 [cm]")
    ax.set_title("真上から見た図: pan を ±60° に振った手先の位置(灰色の丸 = ずれがないときの位置)")
    fig.tight_layout()
    fig.savefig(OUT / "offset_pan_topview.png", dpi=110)
    plt.close(fig)


# ---------- (2) 各関節のオフセットが手先をどちらへ動かすか ----------
def fig_lever():
    pose = MEAS[1]["pose"]  # 点1 の姿勢
    base = arm_points(pose, apply_offset=False)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 5.4), gridspec_kw={"width_ratios": [1.5, 1]})
    ax.plot(*zip(*base), "o-", color=GRAY, lw=5, ms=8, label="sim の予測(点1 の姿勢)")
    colors = {"shoulder_lift": "#e07b00", "elbow_flex": BLUE, "wrist_flex": GREEN}
    for name, col in colors.items():
        pts = arm_points(pose, apply_offset=False, extra_deg={name: 5.0})
        ax.plot(*zip(*pts), "-", color=col, lw=1.5, label=f"{name} だけ +5°")
        dr, dz = pts[-1][0] - base[-1][0], pts[-1][1] - base[-1][1]
        bx.annotate("", (dr, dz), (0, 0), arrowprops=dict(arrowstyle="->", color=col, lw=2.5))
        bx.text(dr - 0.05, dz - 0.18, name, color=col, fontsize=10, ha="center")
    ax.axhline(0, color="k", lw=1)
    ax.set_aspect("equal")
    ax.set_xlabel("肩の軸からの水平距離 r [cm]")
    ax.set_ylabel("机の面からの高さ z [cm]")
    ax.set_title("関節を 1 つだけ +5° 動かした腕(側面図)")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    bx.plot(0, 0, "ko")
    bx.set_xlim(-1.5, 0.5)
    bx.set_ylim(-3.0, 0.4)
    bx.set_aspect("equal")
    bx.set_xlabel("手先の水平方向の動き [cm]")
    bx.set_ylabel("手先の高さ方向の動き [cm]")
    bx.set_title("手先の動き(どれも「下・やや手前」で、向きが近い)")
    bx.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "offset_lever.png", dpi=110)
    plt.close(fig)


# ---------- (3) 測った姿勢の側面図: sim の予測 / オフセットあり / 実測 ----------
def fig_poses():
    items = [m for m in MEAS if m["name"] in ("全関節0", "点1", "点2", "点3", "点8")]
    fig, axes = plt.subplots(1, len(items), figsize=(3.6 * len(items), 4.4), sharey=True)
    for ax, m in zip(axes, items):
        raw = arm_points(m["pose"], apply_offset=False)
        off = arm_points(m["pose"], apply_offset=True)
        ax.plot(*zip(*raw), "o-", color=GRAY, lw=3, ms=5, label="sim の予測(補正なし)")
        ax.plot(*zip(*off), "o-", color=BLUE, lw=2, ms=4, label="オフセットを足した姿勢")
        if "radius" in m:
            ax.plot(m["radius"], m["tip_z"], "o", color=RED, ms=10, mfc=RED if m["reliable"] else "none", mew=2, label="手先の実測" + ("" if m["reliable"] else "(半径は目測)"))
        else:
            ax.plot([off[-1][0] - 2.5, off[-1][0] + 2.5], [m["tip_z"]] * 2, "-", color=RED, lw=4, label="手先の実測(高さだけ)")
        ax.axhline(0, color="k", lw=1)
        ax.set_title(m["name"], fontsize=11)
        ax.set_aspect("equal")
        ax.set_xlabel("r [cm]")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("机の面からの高さ z [cm]")
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / "offset_poses.png", dpi=110)
    plt.close(fig)


# ---------- (4) オフセットの前後の差 ----------
def fig_residual():
    names, zb, za, rb, ra = [], [], [], [], []
    for m in MEAS:
        raw, off = arm_points(m["pose"], False)[-1], arm_points(m["pose"], True)[-1]
        names.append(m["name"])
        zb.append(m["tip_z"] - raw[1])  # 実測 − 予測
        za.append(m["tip_z"] - off[1])
        rb.append(m["radius"] - raw[0] if "radius" in m and m["reliable"] else np.nan)  # 目測の半径は誤差を含むので除く
        ra.append(m["radius"] - off[0] if "radius" in m and m["reliable"] else np.nan)
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (b, a), title in zip(axes, ((zb, za), (rb, ra)), ("手先の高さ(実測 − 予測)", "手先の半径(実測 − 予測。軸の位置が確かな点2・点8 だけ)")):
        ax.bar(x - 0.2, b, 0.4, color=GRAY, label="補正なし")
        ax.bar(x + 0.2, a, 0.4, color=BLUE, label="オフセットあり")
        ax.axhspan(-0.5, 0.5, color=GREEN, alpha=0.15, label="測定誤差の目安 ±0.5 cm")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x, names)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("差 [cm]")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "offset_residual.png", dpi=110)
    plt.close(fig)


# ---------- (5) 一定のずれ と 姿勢で変わるずれ ----------
def fig_constant_vs_varying():
    xs, dz = [], []
    for m in MEAS:
        raw = arm_points(m["pose"], False)[-1]
        xs.append(raw[0])  # 予測の半径(腕の伸び)
        dz.append(m["tip_z"] - raw[1])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    axes[0].plot(xs, dz, "o", color=RED, ms=9, label="実測 − 予測(補正なし)")
    for x, d, m in zip(xs, dz, MEAS):
        axes[0].text(x + 0.3, d + 0.15, m["name"], fontsize=9)
    axes[0].axhspan(min(dz) - 0.3, max(dz) + 0.3, color=GREEN, alpha=0.15)
    axes[0].set_title("今のデータ: ずれは +3〜4.5 cm でほぼ一定\n→ 一定のオフセットで説明できる", fontsize=11)
    xx = np.linspace(8, 28, 50)
    axes[1].plot(xx, 0.5 + 0.18 * (xx - 8) ** 1.3 / 3, "-", color=RED)
    axes[1].plot([10, 15, 20, 25, 28], [0.5 + 0.18 * (v - 8) ** 1.3 / 3 for v in (10, 15, 20, 25, 28)], "o", color=RED, ms=9)
    axes[1].set_title("(模式図)腕が伸びるほどずれが増える場合\n→ たわみなど、姿勢で変わるずれ", fontsize=11)
    for ax in axes:
        ax.set_xlabel("予測の半径(腕の伸び)[cm]")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("手先の高さの差 [cm]")
    fig.tight_layout()
    fig.savefig(OUT / "offset_constant_vs_varying.png", dpi=110)
    plt.close(fig)


for f in (fig_pan, fig_lever, fig_poses, fig_residual, fig_constant_vs_varying):
    f()
    print("作成:", f.__name__)
print("オフセット:", OFFSET_DEG)
