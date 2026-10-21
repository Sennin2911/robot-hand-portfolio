"""【概要】SO-101 を sim で指定した姿勢にして、画面なし(ウィンドウを開かない)で画像 (PNG) に保存する。
真上・正面・横・斜めの 4 方向から撮って 1 枚に並べる。測る 4 点(肩・肘・手首の軸と、手先)に色つきの球の印と、机の面まで下ろした細い線を付ける。
実機の姿勢の写真と見比べたり、測った位置が合っているか確かめたりするための確認用。

使い方(この .venv で実行。引数は view_sim_pose.py と同じ):
    .venv/bin/python scripts/render_sim_pose.py --pose 0.0 -65.9 87.0 -7.7 0.0 10.0   # 実機と同じ値(度、gripper は 0〜100)
    .venv/bin/python scripts/render_sim_pose.py --q 0 0 0 0 0 0                         # sim の rad で指定
    .venv/bin/python scripts/render_sim_pose.py --pose ... --out outputs/point1.png     # 保存先を指定
出力の既定は outputs/sim_pose.png。印の位置(肩の軸からの距離・向き・高さ)も表示する。
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)。
sys.path.insert(0, str(ROOT / "src"))
# 画面なしで描画する設定。★mujoco を import する前に書く必要がある。Linux のときだけ設定する(どちらも理由は check_install.py 参照)。
if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v3 as iio  # noqa: E402  (画像ファイルの保存に使う。[頻度] 画像保存の定番)
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from hand_sim.sim_real import JOINTS, real_to_sim  # noqa: E402

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
group.add_argument("--pose", type=float, nargs=6, help="実機と同じ値で指定 (度、gripper は 0〜100)")
group.add_argument("--q", type=float, nargs=6, help="sim の角度 [rad] で指定")
parser.add_argument("--out", default=str(ROOT / "outputs" / "sim_pose.png"), help="保存する画像のパス")
args = parser.parse_args()

model = mujoco.MjModel.from_xml_path(str(ROOT / "models" / "so101" / "scene.xml"))
data = mujoco.MjData(model)
if args.pose is not None:
    q = real_to_sim(model, {f"{n}.pos": v for n, v in zip(JOINTS, args.pose)})
elif args.q is not None:
    q = np.array(args.q)
else:
    q = np.zeros(6)
q = np.clip(q, model.jnt_range[:6, 0], model.jnt_range[:6, 1])  # sim の可動域の中に収める
data.qpos[:6] = q
data.ctrl[:6] = q
mujoco.mj_forward(model, data)  # qpos から各部の位置を計算する(時間は進めない。姿勢はそのまま)

pan_axis = data.xanchor[model.joint("shoulder_pan").id][:2]
frame = data.site_xpos[model.site("gripperframe").id].copy()  # 手先。固定側の指の先端にある
# 測る 4 点: (日本語名, 画像に入れる英語名, 位置, 色 RGB)。関節の軸の位置は data.xanchor(世界座標)。
MARKS = [
    ("1 shoulder_lift の軸(マゼンタ)", "1 shoulder_lift axis", data.xanchor[model.joint("shoulder_lift").id], (255, 0, 255)),
    ("2 elbow_flex の軸(緑)", "2 elbow_flex axis", data.xanchor[model.joint("elbow_flex").id], (0, 200, 0)),
    ("3 wrist_flex の軸(青)", "3 wrist_flex axis", data.xanchor[model.joint("wrist_flex").id], (0, 80, 255)),
    ("4 手先 = 固定側の指の先端(赤)", "4 fingertip", frame, (255, 0, 0)),
]
for jp, _en, pos, _rgb in MARKS:
    d = pos[:2] - pan_axis
    extra = f"、肩の軸から {np.linalg.norm(d) * 100:5.1f} cm、向き {np.degrees(np.arctan2(d[1], d[0])):+6.1f}°(左が +)" if _en == "4 fingertip" else ""
    print(f"{jp}: 机の面から {pos[2] * 100:5.1f} cm{extra}")


def add_marker(scene, pos, rgb, radius=0.009):
    """描画シーンに印の球と、机の面(高さ 0)まで下ろした細い線を足す。モデルには入れないので、物理には影響しない。"""
    rgba = [c / 255 for c in rgb] + [1.0]
    if scene.ngeom + 2 > scene.maxgeom:
        return
    # mjv_initGeom: 描画用の図形 1 つを初期化する。[頻度] 描画に印を足したいときだけ使う(特殊)
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, [radius, 0, 0], pos, np.eye(3).flatten(), rgba)
    scene.ngeom += 1
    # mjv_connector: 2 点を結ぶ細い棒(カプセル)にする。高さを机の面から測る目安の「下げ振り」。
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9), rgba)
    mujoco.mjv_connector(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0015, pos, [pos[0], pos[1], 0.0])
    scene.ngeom += 1


# (名前, 見る点, 距離, 水平角 azimuth, 上下角 elevation)。MuJoCo の azimuth: 0 が +x 方向から見る向き、90 が +y 側から見る向き。
VIEWS = [
    ("真上", [0.15, 0.0, 0.0], 0.8, 90, -90),
    ("正面(前から)", [0.15, 0.0, 0.1], 0.8, 180, -5),
    ("横(左から)", [0.15, 0.0, 0.1], 0.8, 90, -5),
    ("斜め", [0.15, 0.0, 0.1], 0.9, 140, -25),
]
images = []
with mujoco.Renderer(model, height=480, width=480) as renderer:
    for _name, lookat, distance, azimuth, elevation in VIEWS:
        cam = mujoco.MjvCamera()
        cam.lookat[:] = lookat
        cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
        renderer.update_scene(data, camera=cam)
        for _jp, _en, pos, rgb in MARKS:
            # 軸の球は、サーボの内側に隠れないよう、見やすい側(y 方向)へ 3.5 cm ずらす(横から見て手前側)。高さ(z)は変わらない。手先はそのまま。
            shown = pos + (np.array([0.0, -0.035, 0.0]) if _en.endswith("axis") else 0.0)
            add_marker(renderer.scene, shown, rgb)
        images.append(renderer.render())

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)  # parents=True, exist_ok=True: 途中のフォルダも作り、既にあってもエラーにしない
# 凡例の帯: 画像の上に、色と名前を並べる。PIL の標準フォントは日本語が出ないので、英語名にする。
# PIL(Pillow): 画像に文字や図形を描くライブラリ。imageio が内部で使っているので、追加のインストールは不要。[頻度] 画像処理で定番
from PIL import Image, ImageDraw  # noqa: E402

body = np.hstack(images)  # np.hstack: 同じ高さの画像を横に並べて1枚にする
strip = Image.new("RGB", (body.shape[1], 28), (30, 30, 30))
draw = ImageDraw.Draw(strip)
for i, (_jp, en, _pos, rgb) in enumerate(MARKS):
    x = 10 + i * 300
    draw.ellipse([x, 8, x + 12, 20], fill=rgb)
    draw.text((x + 18, 8), en, fill=(255, 255, 255))
iio.imwrite(out, np.vstack([np.asarray(strip), body]))
print(f"保存: {out}(左から: {' / '.join(v[0] for v in VIEWS)})")
