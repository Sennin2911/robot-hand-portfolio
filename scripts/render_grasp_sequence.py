"""【概要】sim の「掴んで運んで置く」動作の途中の 6 場面を、画面なし(ウィンドウを開かない)で撮って、1 枚の画像 (PNG) に並べる。
場面: ①立方体の真上 ②降下(指は開いたまま) ③指を閉じた ④持ち上げた ⑤置き場の真上へ運んだ ⑥置いて、指を開いた。
指が立方体の両側に入っているか、持ち上げたあとに挟んだままか、を目で確かめるための確認用(数値の確認は check_so101_grasp.py)。

使い方(この .venv で実行):
    .venv/bin/python scripts/render_grasp_sequence.py                     # 既定: 立方体 (0.20, 0)、向き 30°、置き場 (0.18, 0.16)
    .venv/bin/python scripts/render_grasp_sequence.py --xy 0.15 -0.1 --yaw 45 --out outputs/grasp_b.png
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
from hand_sim.so101_grasp import GRIP_CLOSE, GRIP_OPEN, load_grasp_model, move_to, plan_for_cube, set_cube_pose  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--xy", type=float, nargs=2, default=[0.20, 0.0], help="立方体の中心の水平位置 x y [m]")
parser.add_argument("--yaw", type=float, default=30.0, help="立方体の向き [度]")
parser.add_argument("--place", type=float, nargs=2, default=[0.18, 0.16], help="置き場の水平位置 x y [m]")
parser.add_argument("--out", default=str(ROOT / "outputs" / "grasp_sequence.png"), help="保存する画像のパス")
args = parser.parse_args()

model = load_grasp_model(ROOT / "models" / "so101" / "scene_vs.xml")
data = mujoco.MjData(model)
cube = [*args.xy, 0.015]
set_cube_pose(model, data, cube, np.radians(args.yaw))
poses, grasp_yaw = plan_for_cube(model, cube, args.place, np.radians(args.yaw))
print(f"指の向き(yaw): {np.degrees(grasp_yaw):.0f}°")

W, H = 480, 360
renderer = mujoco.Renderer(model, height=H, width=W)
cam = mujoco.MjvCamera()  # 自由に動かせるカメラ。注視点・距離・向きで位置を決める
cam.lookat[:] = [0.2, 0.06, 0.06]  # カメラが見る点 [m]
cam.distance, cam.azimuth, cam.elevation = 0.7, 120, -30  # 距離 [m]、水平方向の向き [度]、見下ろす角度 [度]
frames = []


def shot(label):
    """今の場面を撮って、frames に足す。"""
    renderer.update_scene(data, camera=cam)
    frames.append(renderer.render().copy())  # .copy(): renderer は次の描画で同じ配列を使い回すので、控えを取る
    print(f"{len(frames)}. {label}: 立方体の中心 {np.round(data.xpos[model.body('cube').id], 3)} m")


move_to(model, data, poses["above"], GRIP_OPEN, 3.0)
shot("真上")
move_to(model, data, poses["grasp"], GRIP_OPEN, 1.5)
shot("降下")
move_to(model, data, poses["grasp"], GRIP_CLOSE, 1.0)
shot("閉じた")
move_to(model, data, poses["above"], GRIP_CLOSE, 1.5)
shot("持ち上げた")
move_to(model, data, poses["place_above"], GRIP_CLOSE, 2.0)
shot("運んだ")
move_to(model, data, poses["place"], GRIP_CLOSE, 1.5)
move_to(model, data, poses["place"], GRIP_OPEN, 1.0)
shot("置いて開いた")

# 3 列 × 2 行に並べる。np.concatenate(..., axis=1): 横に連結、axis=0: 縦に連結。[頻度] 画像をつなぐときの定番
rows = [np.concatenate(frames[i:i + 3], axis=1) for i in (0, 3)]
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
iio.imwrite(out, np.concatenate(rows, axis=0))
print("保存:", out)
