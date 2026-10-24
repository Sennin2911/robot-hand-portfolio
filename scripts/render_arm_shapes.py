"""【概要】pan 振りで記録する腕の形 A〜E を、sim で、カメラ側(ロボットの左前)から見た向きで描く。実機で形を作るときの見た目の確認用。
各形に、手の板(黄)と、gripperframe(赤い球)を重ねる。pan = 0 の形を描く(記録のときは、pan を右へ 30°、60° 回す)。文字は日本語(Noto Sans CJK JP)。
画像は outputs/arm_shapes.png。形の値は SHAPES(shoulder_lift, elbow_flex, wrist_flex の読み [度])。real_pan_sweep.py の --pose に、そのまま渡せる。

使い方(この .venv で実行):
    .venv/bin/python scripts/render_arm_shapes.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)
if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "egl")  # mujoco を import する「前」に設定する(import の時点で描画方式が決まるため)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from hand_sim.hand_marker import marker_pose_from_hand  # noqa: E402
from hand_sim.sim_real import JOINTS, real_to_sim  # noqa: E402

plt.rcParams["font.family"] = "Noto Sans CJK JP"
# 形の名前: (shoulder_lift, elbow_flex, wrist_flex の読み [度]、説明)。A は、前回の記録(休めの折りたたみ姿勢)
SHAPES = {
    "A": (-105.8, 97.4, -111.5, "休めの折りたたみ姿勢(済み)"),
    "B": (-7.0, 2.0, 95.0, "腕を前へ伸ばし、手を真下(測定の姿勢)"),
    "C": (-25.0, 2.0, 59.0, "腕を前へ高めに伸ばし、手を少し下へ"),
    "D": (-91.0, 44.0, 83.0, "腕を引き気味にして、手を下へ"),
    "E": (65.0, -52.0, -97.0, "腕を前へ倒して、手を真上へ"),
}
model = mujoco.MjModel.from_xml_path(str(ROOT / "models" / "so101" / "scene.xml"))
model.vis.global_.offwidth, model.vis.global_.offheight = 800, 800  # 描画に使う画面の大きさの上限(既定は 640×480。広げるときは、Renderer を作る前に)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=600, width=600)
sid = model.site("gripperframe").id


def add(scene, kind, size, pos, rot, rgba):
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], kind, size, pos, rot.flatten(), rgba)
    scene.ngeom += 1


fig, axes = plt.subplots(2, 3, figsize=(15, 11), gridspec_kw={"hspace": 0.22})  # hspace: 上下の画像の間のすき間(題が、上の画像に重ならないように)
for ax, (name, (lift, elbow, wflex, text)) in zip(axes.ravel(), SHAPES.items()):
    obs = {"shoulder_pan.pos": 0.0, "shoulder_lift.pos": lift, "elbow_flex.pos": elbow, "wrist_flex.pos": wflex, "wrist_roll.pos": 0.0, "gripper.pos": 10.0}
    q = real_to_sim(model, obs)
    for j, n in enumerate(JOINTS):
        data.qpos[model.joint(n).qposadr] = q[j]
    mujoco.mj_forward(model, data)
    R = data.site_xmat[sid].reshape(3, 3)
    S = data.site_xpos[sid].copy()
    R_m, p_m = marker_pose_from_hand(R, S)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.15, 0.0, 0.15]
    cam.distance, cam.azimuth, cam.elevation = 0.95, 90, 8  # 左(+y)側から。画面の右が、ロボットの正面(+x)
    renderer.update_scene(data, camera=cam)
    add(renderer.scene, mujoco.mjtGeom.mjGEOM_BOX, [0.015, 0.015, 0.0008], p_m + R_m[:, 2] * 0.0008, R_m, [1.0, 0.9, 0.1, 1.0])  # 手の板(黄)
    add(renderer.scene, mujoco.mjtGeom.mjGEOM_SPHERE, [0.008, 0, 0], S, np.eye(3), [1, 0, 0, 1])  # 手先 gripperframe(赤い球)
    ax.imshow(renderer.render())
    ax.set_title(f"{name}: {text}\nlift {lift:+.0f}、elbow {elbow:+.0f}、wrist_flex {wflex:+.0f}", fontsize=11)
    ax.axis("off")
axes.ravel()[-1].axis("off")
fig.suptitle("記録する腕の形(pan = 0、左側から見た図。画面の右が、ロボットの正面)。黄 = 手の板、赤い球 = 手先", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = ROOT / "outputs" / "arm_shapes.png"
fig.savefig(out, dpi=100)
print(f"保存: {out.relative_to(ROOT)}")
