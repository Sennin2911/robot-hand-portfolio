"""【概要】sim のグリッパーを、4 方向から近くで撮り、「指先の基準点(gripperframe)」と「手の座標軸(②)」と「手の板(ID 0)の位置・向き」を重ねて描く。
目的: 実機の CAD で測った A2〜A4 と、sim の基準点が、同じ点・同じ軸を指しているか、目で確かめる(z の差 約 12 mm の原因探し)。
描くもの: 赤い大きな球 = gripperframe(sim の指先の基準点。IK も、手先の位置の比較も、この点)/ 軸: x = 赤(指先の向き)、y = 緑(左)、z = 青(動く指のある側)/
  黄色の薄い板 = 手の板(30 mm 角、A2 の位置、A3 の向き)。オレンジの線 = 紙の「上」(A4)/ マゼンタの球 = 固定側の指のメッシュの、x が最大の点(先端のいちばん端)。
文字は日本語(PNG なので、文字化けしない。フォントは Noto Sans CJK JP)。画像は outputs/gripper_reference.png。

使い方(この .venv で実行):
    .venv/bin/python scripts/render_gripper_reference.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)
if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "egl")  # mujoco を import する「前」に設定する(import の時点で描画方式が決まるため)

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # 画面を出さずにファイルだけ作る。pyplot を import する前に書く
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from hand_sim.hand_marker import R_HAND_MARKER, T_HAND_MARKER, marker_pose_from_hand  # noqa: E402

plt.rcParams["font.family"] = "Noto Sans CJK JP"  # 日本語を表示するフォント(matplotlib の既定のフォントには、日本語が無い)

model = mujoco.MjModel.from_xml_path(str(ROOT / "models" / "so101" / "scene.xml"))
data = mujoco.MjData(model)
data.qpos[model.joint("gripper").qposadr] = 0.7  # 動く指を、少し開く(動く指の側 = ②の +z が見えるように)
mujoco.mj_forward(model, data)
sid = model.site("gripperframe").id
S = data.site_xpos[sid].copy()  # 基準点の世界座標
R = data.site_xmat[sid].reshape(3, 3).copy()  # 列 = ②の x, y, z 軸の世界での向き
R_m, p_m = marker_pose_from_hand(R, S)  # 手の板の中心・向き(世界座標)

# 固定側の指のメッシュ(当たり判定、group 2)の頂点を、②の座標で集める。x が最大の点 = 先端のいちばん端。
body = model.body("gripper").id
pts = []
for g in range(model.ngeom):
    if model.geom_bodyid[g] != body or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or model.geom_group[g] != 2:
        continue
    me = model.geom_dataid[g]
    v = model.mesh_vert[model.mesh_vertadr[me]:model.mesh_vertadr[me] + model.mesh_vertnum[me]]
    pts.append(v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])
pts = np.vstack(pts)
local = (pts - S) @ R  # ②の座標
tip = pts[np.argmax(local[:, 0])]  # x が最大の頂点
print(f"gripperframe(基準点)を原点にした、メッシュの先端の最も端の点(x 最大): {np.round((tip - S) @ R * 1000, 1)} mm(②の x, y, z)")
print(f"手の板の中心: {np.round(T_HAND_MARKER * 1000, 1)} mm(②の座標)")


def sphere(scene, pos, rgba, radius):
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, [radius, 0, 0], pos, np.eye(3).flatten(), rgba)
    scene.ngeom += 1


def line(scene, a, b, rgba, radius=0.0012):
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9), rgba)
    mujoco.mjv_connector(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE, radius, a, b)  # 2 点を結ぶ棒にする(描画用の図形を足すだけ。物理には影響しない)
    scene.ngeom += 1


def box(scene, pos, rot, size, rgba):
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_BOX, size, pos, rot.flatten(), rgba)
    scene.ngeom += 1


def draw_overlay(scene):
    box(scene, p_m + R_m[:, 2] * 0.0008, R_m, [0.015, 0.015, 0.0008], [1.0, 0.9, 0.1, 1.0])  # 手の板(30 mm 角、厚み 1.6 mm)。面の外向き(z)へ少し浮かせる
    line(scene, p_m, p_m + R_m[:, 1] * 0.025, [1.0, 0.5, 0.0, 1.0], 0.0015)  # 紙の上(A4)
    line(scene, p_m, p_m + R_m[:, 2] * 0.03, [0.6, 0.0, 0.8, 1.0], 0.0012)  # 面の垂線(A3)
    for i, rgba in enumerate(([1, 0, 0, 1], [0, 0.8, 0, 1], [0, 0.2, 1, 1])):  # 手の座標軸(x 赤、y 緑、z 青)。長さ 5 cm
        line(scene, S, S + R[:, i] * 0.05, rgba, 0.0013)
    sphere(scene, S, [1, 0, 0, 1], 0.0045)  # gripperframe(赤い球)
    sphere(scene, tip, [1, 0, 1, 1], 0.003)  # メッシュの先端の最も端(マゼンタ)


# (題, 見る向き。MuJoCo の azimuth と elevation、見る点を基準点からずらす量(②の座標、m))
center = S + R @ np.array([-0.045, 0.0, 0.0])  # グリッパーの中ほどを見る
VIEWS = [
    ("左(+y)側から。手の板の面が見える向き", 90, 0),
    ("右(−y)側から", 270, 0),
    ("真上から(x–y 面)", 90, -90),
    ("指先の側から(正面)", 180, 0),
]
model.vis.global_.offwidth, model.vis.global_.offheight = 1000, 800  # 描画に使う画面の大きさの上限(既定は 640×480 なので、大きい画像を撮るときは広げる。モデルの読み込み後、Renderer を作る前に)
renderer = mujoco.Renderer(model, height=700, width=900)
fig, axes = plt.subplots(2, 2, figsize=(14, 11.5))
for ax, (title, az, el) in zip(axes.ravel(), VIEWS):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = center
    cam.distance, cam.azimuth, cam.elevation = 0.32, az, el
    renderer.update_scene(data, camera=cam)
    draw_overlay(renderer.scene)
    ax.imshow(renderer.render())
    ax.set_title(title, fontsize=12)
    ax.axis("off")
fig.suptitle("sim のグリッパー(gripperframe は、赤い球)\n"
             "軸: x = 赤(指先の向き)/ y = 緑 / z = 青(動く指の側)。黄 = 手の板(A2〜A4)、オレンジ = 紙の上、紫 = 板の垂線、マゼンタの球 = メッシュの先端のいちばん端", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])  # rect: 図全体のうち、使う範囲(左, 下, 右, 上)。上の 7% を、全体のタイトルのために空ける
out = ROOT / "outputs" / "gripper_reference.png"
fig.savefig(out, dpi=110)
print(f"保存: {out.relative_to(ROOT)}")
