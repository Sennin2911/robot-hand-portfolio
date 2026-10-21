"""【概要】カメラを「任意の場所」に置いたときにも、手の目印による較正が有効かを、sim で調べる(カメラの置き場所の調査)。
真上(これまでの置き方)のほかに、前・横・斜め・後ろ・高さ・距離を変えた 12 通りにカメラを置き、それぞれで次を測る。
  (1) 手の目印が見える位置の数(較正に使える点。腕や手の陰に隠れないか)。最初は手の向きを 1 通り(16 点)で撮り、6 点未満なら、もう 1 通り(150° 回した向き)を足す
  (2) 較正が、目で見て決めた程度のずれた出発点(位置 ±10 cm、向き ±10°、視野角 ±10°)から、正しい答えに収束するか(10 通りの出発点で)
      「収束」の基準は、再投影誤差 2 画素未満 かつ **作業範囲の誤差 5 mm 未満**。作業範囲の誤差とは、立方体を置く範囲の格子点(18 点)が、本当のカメラで写る画素を、
      求めたカメラで世界座標に戻したときの誤差の平均。位置や向きの誤差そのものは見ない(位置と向きの誤差は、打ち消し合って、作業範囲では小さいことがある)。
  (3) 較正した結果で、立方体の位置を認識した誤差 [mm](ランダムな 15 配置 × 3 色)と、見つけられた割合。
      認識の方法を 2 つ比べる: 「色の塊の重心を、立方体の中心の高さに投影」(従来)と、「塊のうち明るい面(上面)だけの重心を、上面の高さに投影」。
較正のモデルは、位置 3 + 向き 3 + 視野角 1 の 7 つ(camera_pose.py)。レンズの歪みは入れていない。実機は動かさない。
結果の読み方は docs/camera_placement.md。図は docs/img/camera_placement_*.png に出る。

使い方(この .venv で実行):
    .venv/bin/python scripts/run_camera_placement_study.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)。
sys.path.insert(0, str(ROOT / "src"))
# 画面なしで描画する設定。★mujoco を import する前に書く必要がある。Linux のときだけ設定する(どちらも理由は check_install.py 参照)。
if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # 画面を出さずに PNG だけ作る設定。pyplot を import する前に書く(後だと効かない)。[頻度] ヘッドレスでの作図では定番
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402
from hand_sim.camera_pose import (  # noqa: E402
    calibrate_camera_pose, look_at_rotation, pixel_to_world_plane, project_points,
)
from hand_sim.sorting import (  # noqa: E402
    COLORS, CUBE_Z, IMG_H, IMG_W, collect_hand_marker_samples, load_sort_model, new_episode, random_layout, true_cube_pose,
)
from hand_sim.vision import _largest_component, color_mask  # noqa: E402

plt.rcParams["font.family"] = "Noto Sans CJK JP"  # 日本語の文字化け(□□□)を避けるフォント指定
plt.rcParams["axes.unicode_minus"] = False

TARGET = (0.20, -0.04, 0.02)  # カメラが見つめる点(立方体を置く範囲の中心)
FOVY = 45.0  # 縦方向の視野角 [度](これまでの sim と同じ。実機のウェブカメラは、仕様に書かれていないので、較正で求める)
# 置き場所: (名前, カメラの位置 [m]、見つめる点、画像の上にしたい向き(up))。
# 世界座標: x = ロボットの前方、y = 左、z = 上。ロボットの土台は原点にあり、立方体を置く範囲は x = 0.14〜0.26、y = −0.14〜0.06。
PLACEMENTS = [
    ("真上 60 cm(これまで)", (0.20, 0.0, 0.60), TARGET, (0, 1, 0)),
    ("真上 90 cm", (0.20, 0.0, 0.90), TARGET, (0, 1, 0)),
    ("真上 40 cm", (0.20, 0.0, 0.40), TARGET, (0, 1, 0)),
    ("前から 斜め上", (0.55, -0.04, 0.45), TARGET, (0, 0, 1)),
    ("前から 低め", (0.60, -0.04, 0.20), TARGET, (0, 0, 1)),
    ("左から", (0.20, 0.50, 0.30), TARGET, (0, 0, 1)),
    ("右から", (0.20, -0.55, 0.30), TARGET, (0, 0, 1)),
    ("斜め前 左", (0.50, 0.40, 0.40), TARGET, (0, 0, 1)),
    ("斜め前 右", (0.50, -0.45, 0.40), TARGET, (0, 0, 1)),
    ("ロボットの後ろ 上", (-0.15, 0.0, 0.50), TARGET, (0, 0, 1)),
    ("右の机の端に付けた(低め)", (0.20, -0.30, 0.15), TARGET, (0, 0, 1)),
    ("遠く 斜め上(1 m)", (0.80, 0.30, 0.70), TARGET, (0, 0, 1)),
]
N_INIT = 10  # 較正を試す、出発点の数
INIT_POS_ERR, INIT_ANG_ERR, INIT_FOV_ERR = 0.10, 10.0, 10.0  # 出発点のずれの大きさ(位置 [m]、向き [度]、視野角 [度])。目で見て決めた程度
N_LAYOUT = 15  # 認識の誤差を測る、立方体の配置の数
MIN_MARKERS = 6  # 較正に必要な目印の点数の下限(未知数 7 つ、式は 2 × 点数)
TOP_FACE_FRAC = 0.85  # 塊の中で「その色のチャンネルが、最大値のこの割合以上」の画素を、明るい面(上面)とみなす
# 作業範囲の格子点(立方体の中心と上面の高さ)。較正の良さを、「作業範囲で、どれだけ正しく座標に戻せるか」で測る。
WORK_GRID = np.array([[x, y, z] for x in (0.14, 0.20, 0.26) for y in (-0.14, -0.04, 0.06) for z in (0.015, 0.03)])

model = load_sort_model(ROOT / "models" / "so101" / "scene_sort.xml")
renderer = mujoco.Renderer(model, IMG_H, IMG_W)
cid = model.camera("overhead").id
data_cam = mujoco.MjData(model)


def place_camera(pos, target, up):
    """カメラ "overhead" を、pos に置いて target を見つめる向きにする。本当の位置・向き (pos, R) を返す。"""
    R = look_at_rotation(pos, target, up)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R.flatten())  # 回転行列(9 つの並び)→ クォータニオン
    model.cam_pos[cid], model.cam_quat[cid], model.cam_fovy[cid] = pos, quat, FOVY
    return np.array(pos, dtype=float), R


def detect_general(data, cam_pos, cam_R, fovy, top_face):
    """3 色の立方体の位置 {色: (x, y) または None} を、較正した (位置, 向き, 視野角) で出す。
    top_face=False: 色の塊全体の画素の重心を、立方体の中心の高さ(CUBE_Z)の水平な面に投影する。真上からなら、これで足りる。
    top_face=True : 塊のうち、明るい画素(上面。側面は影で暗い)だけの重心を、上面の高さ(0.03 m)に投影する。斜めから見ると、側面も写って重心がずれるので、こちらが要る。"""
    renderer.update_scene(data, camera="overhead")
    img = renderer.render()
    found = {}
    for color in COLORS:
        mask = _largest_component(color_mask(img, color))
        if mask is None or mask.sum() < 20:
            found[color] = None
            continue
        z_plane = CUBE_Z
        if top_face:
            channel = img[:, :, {"red": 0, "green": 1, "blue": 2}[color]].astype(float)
            mask = mask & (channel >= TOP_FACE_FRAC * channel[mask].max())
            z_plane = 2 * CUBE_Z  # 上面の高さ(立方体の中心の高さの 2 倍)
        v_idx, u_idx = np.nonzero(mask)
        p = pixel_to_world_plane(u_idx.mean(), v_idx.mean(), cam_pos, cam_R, fovy, IMG_W, IMG_H, z_plane)
        found[color] = None if p is None else p[:2]
    return found


def workspace_error_mm(true_pos, true_R, est_pos, est_R, est_fovy):
    """作業範囲の誤差 [mm]: 作業範囲の格子点が、本当のカメラで写る画素を、求めたカメラで世界座標に戻したときの、誤差の平均。"""
    pixels = project_points(WORK_GRID, true_pos, true_R, FOVY, IMG_W, IMG_H)
    errs = []
    for (u, v), g in zip(pixels, WORK_GRID):
        w = pixel_to_world_plane(u, v, est_pos, est_R, est_fovy, IMG_W, IMG_H, g[2])  # 格子点と同じ高さの面に戻す
        errs.append(np.inf if w is None else np.linalg.norm(w - g) * 1000)
    return float(np.mean(errs))


def random_init(true_pos, true_R, rng):
    """目で見て決めた程度にずれた、較正の出発点 (位置, 向き, 視野角) を作る。"""
    d = rng.normal(size=3)
    pos = true_pos + d / np.linalg.norm(d) * INIT_POS_ERR  # ランダムな向きに、INIT_POS_ERR だけずらす
    axis = rng.normal(size=3)
    rot = Rotation.from_rotvec(axis / np.linalg.norm(axis) * np.radians(INIT_ANG_ERR))
    return pos, (rot * Rotation.from_matrix(true_R)).as_matrix(), FOVY + rng.uniform(-INIT_FOV_ERR, INIT_FOV_ERR)


results, views = [], []
for name, pos, target, up in PLACEMENTS:
    true_pos, true_R = place_camera(pos, target, up)
    rng = np.random.default_rng(0)
    points, pixels = collect_hand_marker_samples(model, renderer)  # 較正の観測(出発点によらないので、1 回だけ集める)
    yaws = "0°"
    if len(points) < MIN_MARKERS:  # 目印が見えなさすぎる。手を 150° 回した向きでも撮って、足す(目印が、カメラの側を向く姿勢が増える)
        points, pixels = collect_hand_marker_samples(model, renderer, yaws_deg=(0.0, 150.0))
        yaws = "0°+150°"
    row = {"name": name, "n_marker": len(points), "yaws": yaws, "n_conv": 0, "ws_err": np.nan, "ws_best": np.nan, "found": 0.0,
           "err_mean": np.nan, "err_max": np.nan, "sil_mean": np.nan, "sil_max": np.nan}
    best = None
    if len(points) >= MIN_MARKERS:
        ws_list = []
        for _ in range(N_INIT):
            ip, iR, ifov = random_init(true_pos, true_R, rng)
            pos_e, R_e, fov_e, rms = calibrate_camera_pose(points, pixels, IMG_W, IMG_H, ip, iR, ifov)
            ws = workspace_error_mm(true_pos, true_R, pos_e, R_e, fov_e)
            ws_list.append(ws)
            if rms < 2.0 and ws < 5.0:  # 収束した(作業範囲で、正しい座標に戻せる答えに着いた)
                row["n_conv"] += 1
            if best is None or rms < best[0]:
                best = (rms, pos_e, R_e, fov_e)
                row["ws_best"] = ws  # 再投影誤差が最も小さかった出発点の、作業範囲の誤差(実機で「複数の出発点を試して、最も合うものを採る」ときの結果)
        row["ws_err"] = float(np.median(ws_list))
        # 認識の誤差: 最も再投影誤差の小さかった較正の結果で、ランダムな 15 配置の立方体を、2 つの方式で認識する
        errs = {True: [], False: []}
        n_found, n_all = 0, 0
        rng2 = np.random.default_rng(1)
        for _ in range(N_LAYOUT):
            data = new_episode(model, random_layout(rng2))
            for top in (True, False):
                found = detect_general(data, best[1], best[2], best[3], top)
                for c in COLORS:
                    if found[c] is not None:
                        errs[top].append(np.linalg.norm(found[c] - true_cube_pose(model, data, c)[0]) * 1000)
                    if top:
                        n_all += 1
                        n_found += found[c] is not None
        row["found"] = n_found / n_all
        if errs[True]:
            row["err_mean"], row["err_max"] = float(np.mean(errs[True])), float(np.max(errs[True]))
        if errs[False]:
            row["sil_mean"], row["sil_max"] = float(np.mean(errs[False])), float(np.max(errs[False]))
    results.append(row)
    # 図用に、この置き場所のカメラの画像(立方体を置いた状態)を撮る
    data = new_episode(model, random_layout(np.random.default_rng(5)))
    renderer.update_scene(data, camera="overhead")
    views.append((name, renderer.render().copy()))
    print(f"{name:20s} 目印 {row['n_marker']:2d} 点({yaws:7s}) 収束 {row['n_conv']:2d}/{N_INIT}  作業範囲の誤差 {row['ws_err']:5.1f} mm  "
          f"見つけた割合 {100 * row['found']:5.1f}%  認識の誤差: 上面 平均 {row['err_mean']:5.1f}・最大 {row['err_max']:5.1f} mm / 輪郭 平均 {row['sil_mean']:5.1f}・最大 {row['sil_max']:5.1f} mm", flush=True)

# 判定: 目印 6 点以上、較正が 10 通りすべて収束、立方体を全部見つけて、認識の誤差が 平均 5 mm 未満・最大 10 mm 未満 → 有効
print("\n判定(目印 6 点以上 / 10 通りの出発点で、再投影誤差が最小の答えが、作業範囲の誤差 5 mm 未満 / 立方体を全部見つける / 上面での認識の誤差 平均 5 mm 未満・最大 10 mm 未満):")
for r in results:
    ok = r["n_marker"] >= MIN_MARKERS and r["ws_best"] < 5 and r["found"] == 1.0 and r["err_mean"] < 5 and r["err_max"] < 10
    note = "" if r["n_conv"] == N_INIT else f"(出発点によっては収束しない: {r['n_conv']}/{N_INIT}。複数の出発点を試して、最も合う答えを採る)"
    print(f"  {'有効' if ok else '要注意・無効'}  {r['name']} {note}")

OUT = ROOT / "docs" / "img"
OUT.mkdir(exist_ok=True)
# 図 1: 置き場所ごとの、カメラから見た画像
fig, axes = plt.subplots(3, 4, figsize=(14, 8.2))
for ax, (name, img) in zip(axes.ravel(), views):
    ax.imshow(img)
    ax.set_title(name, fontsize=10)
    ax.axis("off")
fig.tight_layout()
fig.savefig(OUT / "camera_placement_views.png", dpi=90)
plt.close(fig)
# 図 2: 置き場所ごとの、認識の誤差(2 方式)と、較正の収束
labels = [r["name"] for r in results]
x = np.arange(len(results))
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
axes[0].bar(x - 0.2, [r["err_mean"] for r in results], 0.4, label="上面だけを使う(平均)", color="#1f77b4")
axes[0].bar(x + 0.2, [r["sil_mean"] for r in results], 0.4, label="色の塊全体を使う(平均)", color="#bbbbbb")
axes[0].axhline(10, color="k", ls="--", lw=1)
axes[0].text(len(results) - 0.5, 10.4, "掴める限界の目安 10 mm", ha="right", fontsize=8)
axes[0].set_ylabel("立方体の位置の認識の誤差 [mm]")
axes[0].set_title("較正した後の、認識の誤差(認識の方式で比べる)")
axes[0].set_xticks(x, labels, rotation=40, ha="right", fontsize=8)
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3, axis="y")
axes[1].bar(x - 0.2, [r["n_marker"] for r in results], 0.4, label="較正に使えた目印の点数", color="#2ca02c")
axes[1].bar(x + 0.2, [r["n_conv"] for r in results], 0.4, label="較正が収束した出発点(10 通り中)", color="#ff7f0e")
axes[1].axhline(MIN_MARKERS, color="k", ls="--", lw=1)
axes[1].text(len(results) - 0.5, MIN_MARKERS + 0.3, "必要な点数の下限 6", ha="right", fontsize=8)
axes[1].set_ylim(0, 15)
axes[1].set_title("目印の見えやすさ と 較正の収束")
axes[1].set_xticks(x, labels, rotation=40, ha="right", fontsize=8)
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(OUT / "camera_placement_results.png", dpi=100)
plt.close(fig)
renderer.close()
