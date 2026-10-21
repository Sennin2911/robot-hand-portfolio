"""【概要】sim の「掴んで運んで置く」動作を、MuJoCo のビューア(ウィンドウ)で実時間のまま再生する。1 パターンぶん。
動画ファイルを作らず、動きをそのまま見る用(動画にしたいときは render_grasp_video.py)。
動作が終わっても、ウィンドウは開いたまま(マウスで視点を動かして、立方体と指の位置関係を見られる)。ウィンドウを閉じると終わる。

使い方(この .venv で実行):
    .venv/bin/python scripts/view_grasp_sim.py                          # 既定: 立方体 (0.20, 0)、向き 30°、置き場 (0.18, 0.16)
    .venv/bin/python scripts/view_grasp_sim.py --speed 0.5              # 半分の速さ(ゆっくり見る)
    .venv/bin/python scripts/view_grasp_sim.py --xy 0.15 -0.1 --yaw 45
ビューアの操作: 左ドラッグ = 回す、右ドラッグ = 動かす、ホイール = 拡大縮小(この動作の再生は、途中で止められない)。
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)。
sys.path.insert(0, str(ROOT / "src"))

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402
import numpy as np  # noqa: E402
from hand_sim.so101_grasp import load_grasp_model, pick_and_place, set_cube_pose  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--xy", type=float, nargs=2, default=[0.20, 0.0], help="立方体の中心の水平位置 x y [m]")
parser.add_argument("--yaw", type=float, default=30.0, help="立方体の向き [度]")
parser.add_argument("--place", type=float, nargs=2, default=[0.18, 0.16], help="置き場の水平位置 x y [m]")
parser.add_argument("--speed", type=float, default=1.0, help="再生の速さ(1 = 実時間、0.5 = 半分、2 = 2 倍)")
args = parser.parse_args()

model = load_grasp_model(ROOT / "models" / "so101" / "scene_vs.xml")
data = mujoco.MjData(model)
cube = [*args.xy, 0.015]
set_cube_pose(model, data, cube, np.radians(args.yaw))

CHECK_EVERY = 8  # 何ステップごとに、時間合わせと画面更新の判断をするか(毎ステップだと、判断のほうが重い)
SYNC_INTERVAL = 1 / 50  # 画面を更新する間隔 [秒]。描画が重い PC でも、動きが実時間より遅くならないよう、時間で決める

with mujoco.viewer.launch_passive(model, data) as viewer:
    # 初期の視点: 斜め上から、腕と立方体が入る位置。
    viewer.cam.lookat[:] = [0.2, 0.06, 0.06]
    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 0.7, 120, -30
    start = time.perf_counter()  # perf_counter: 経過時間を測る時計。[頻度] 時間の計測では定番
    step = [0]  # リストに入れるのは、中の関数 on_step から書き換えるため

    last_sync = [start]  # 最後に画面を更新した時刻

    def on_step():
        """シミュレーションが 1 ステップ進むたびに呼ばれる。実時間に合わせる待ちと、画面の更新をする。"""
        step[0] += 1
        if step[0] % CHECK_EVERY:
            return
        if not viewer.is_running():  # ウィンドウが閉じられた
            sys.exit(0)  # sys.exit: プログラムをここで終える。[頻度] 途中で打ち切るときに使う
        # sim の時刻(data.time)に、実際の経過時間を合わせる。早く進みすぎたら、その分だけ待つ。
        wait = data.time / args.speed - (time.perf_counter() - start)
        if wait > 0:
            time.sleep(wait)
        if time.perf_counter() - last_sync[0] >= SYNC_INTERVAL:
            viewer.sync()  # 計算結果をウィンドウに反映する
            last_sync[0] = time.perf_counter()

    print("再生中...(終わっても、ウィンドウは開いたまま。閉じると終了)")
    pick_and_place(model, data, cube, args.place, np.radians(args.yaw), on_step=on_step)
    print("動作が終わった。ウィンドウを閉じると終了")
    while viewer.is_running():
        viewer.sync()
        time.sleep(0.05)
