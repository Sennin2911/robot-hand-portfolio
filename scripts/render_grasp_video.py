"""【概要】sim の「掴んで運んで置く」動作を、画面なし(ウィンドウを開かない)で動画(mp4)に保存する。1 パターンぶん。
動作そのものは so101_grasp.pick_and_place(check_so101_grasp.py と同じ)。1 ステップごとに呼ばれる on_step で、一定間隔ごとに 1 コマ撮る。
左: 斜め上から見た図 / 右: 真上から見た図(立方体の位置と向き、指の向きが分かる)。

使い方(この .venv で実行):
    .venv/bin/python scripts/render_grasp_video.py                          # 既定: 立方体 (0.20, 0)、向き 30°、置き場 (0.18, 0.16)
    .venv/bin/python scripts/render_grasp_video.py --xy 0.15 -0.1 --yaw 45 --out outputs/grasp_b.mp4
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

import imageio.v2 as imageio  # noqa: E402  (動画の書き出しに使う。.v2 は従来からの使い方のバージョン。[頻度] 動画保存の定番)
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from hand_sim.so101_grasp import load_grasp_model, pick_and_place, set_cube_pose  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--xy", type=float, nargs=2, default=[0.20, 0.0], help="立方体の中心の水平位置 x y [m]")
parser.add_argument("--yaw", type=float, default=30.0, help="立方体の向き [度]")
parser.add_argument("--place", type=float, nargs=2, default=[0.18, 0.16], help="置き場の水平位置 x y [m]")
parser.add_argument("--fps", type=int, default=30, help="動画の 1 秒あたりのコマ数")
parser.add_argument("--out", default=str(ROOT / "outputs" / "grasp.mp4"), help="保存する動画のパス")
args = parser.parse_args()

model = load_grasp_model(ROOT / "models" / "so101" / "scene_vs.xml")
data = mujoco.MjData(model)
cube = [*args.xy, 0.015]
set_cube_pose(model, data, cube, np.radians(args.yaw))

W, H = 640, 480
renderer = mujoco.Renderer(model, height=H, width=W)
# 左のカメラ: 斜め上。右のカメラ: 真上(elevation −90 = 真下を向く)。
cam_side, cam_top = mujoco.MjvCamera(), mujoco.MjvCamera()
cam_side.lookat[:] = [0.2, 0.06, 0.06]
cam_side.distance, cam_side.azimuth, cam_side.elevation = 0.7, 120, -30
cam_top.lookat[:] = [0.2, 0.06, 0.0]
cam_top.distance, cam_top.azimuth, cam_top.elevation = 0.75, 90, -90

# 何ステップごとに 1 コマ撮るか: 1 秒 = fps コマ、1 ステップ = timestep 秒 なので、1 / (fps × timestep) ステップに 1 コマ。
render_every = int(1 / (args.fps * model.opt.timestep))
step = [0]  # リストに入れるのは、中の関数 on_step から書き換えるため(整数のままだと、関数の中で代入すると別の変数になる)
frames = []


def on_step():
    """シミュレーションが 1 ステップ進むたびに呼ばれる。render_every ステップに 1 回だけ撮る。"""
    step[0] += 1
    if step[0] % render_every:  # % は余り。0 でない(= 撮る番でない)なら何もしない
        return
    views = []
    for cam in (cam_side, cam_top):
        renderer.update_scene(data, camera=cam)
        views.append(renderer.render().copy())  # .copy(): renderer は次の描画で同じ配列を使い回すので、控えを取る
    frames.append(np.concatenate(views, axis=1))  # 左右に並べる


log = pick_and_place(model, data, cube, args.place, np.radians(args.yaw), on_step=on_step)

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
# mimsave: 画像のリストを動画(拡張子が .mp4 なら mp4)にして保存する。[頻度] 動画保存の定番
imageio.mimsave(out, frames, fps=args.fps)
print(f"保存: {out}({len(frames)} コマ、{len(frames) / args.fps:.1f} 秒)")
print("立方体の中心 [m]: 掴む前", np.round(log["before_close"], 3), "→ 持ち上げ後", np.round(log["lifted"], 3), "→ 置いた後", np.round(log["placed"], 3))
