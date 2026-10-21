"""【概要】部品仕分けの試験(run_sorting_trials.py)の「条件ごとの動き」を、MuJoCo のビューア(ウィンドウ)で実時間のまま再生する。
1 エピソード(3 色の立方体 3 個)ぶん。カメラで見つけた位置は、立方体の上に出る**水色(シアン)の球**で示す。
(手の側面にあるマゼンタ(赤紫)の球は、別物: 手に付けた較正用の目印で、モデルの一部。水色の球は、画像から出した位置を示すだけの、ビューア上の表示。)
カメラの位置がずれていると、球が立方体からずれて、指がずれた位置へ降りて、掴めない様子が見える。
1 個ごとの結果(成功・失敗の種類、認識の誤差)は、ターミナルに出る。終わっても、ウィンドウは開いたまま(閉じると終了)。

使い方(この .venv で実行):
    .venv/bin/python scripts/view_sorting_sim.py --list                 # 条件の一覧
    .venv/bin/python scripts/view_sorting_sim.py                        # 既定: baseline(ずれなし)
    .venv/bin/python scripts/view_sorting_sim.py camx15                 # カメラの位置が x に 15 mm ずれた条件
    .venv/bin/python scripts/view_sorting_sim.py camx15_cal             # 同じずれを、較正(手の目印でカメラの位置を求め直す)してから仕分け
    .venv/bin/python scripts/view_sorting_sim.py camy20 --seed 3        # 配置を変える(--seed が違うと、立方体の置き方が変わる)
    .venv/bin/python scripts/view_sorting_sim.py camx15 --speed 3       # 3 倍の速さ(1 エピソードは、実時間で約 1 分かかる)
    .venv/bin/python scripts/view_sorting_sim.py --cam-err-mm 12 -8 --noise-px 1 --calibrate   # 条件を自分で指定
"""
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)。
sys.path.insert(0, str(ROOT / "src"))
# カメラ画像の撮影(Renderer)を画面なしの方式で行う設定。★mujoco を import する前に書く必要がある。Linux のときだけ設定する(理由は check_install.py 参照)。
if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402
import numpy as np  # noqa: E402
from hand_sim.sorting import IMG_H, IMG_W, calibrate_overhead_camera, load_sort_model, new_episode, random_layout, run_episode  # noqa: E402

# 条件の名前 → (カメラの位置のずれ x, y [mm]、画素ノイズ [画素]、較正するか)。run_sorting_trials.py で測った条件と同じ。
# 名前の末尾の _cal は「較正あり」(手の目印でカメラの位置を求め直してから仕分ける)。
CONDITIONS = {
    "baseline": (0, 0, 0, False), "camx5": (5, 0, 0, False), "camx10": (10, 0, 0, False), "camx15": (15, 0, 0, False),
    "camx20": (20, 0, 0, False), "camx30": (30, 0, 0, False), "camy10": (0, 10, 0, False), "camy20": (0, 20, 0, False),
    "noise1": (0, 0, 1, False), "noise2": (0, 0, 2, False), "noise4": (0, 0, 4, False),
    "camx15_cal": (15, 0, 0, True), "camx20_cal": (20, 0, 0, True), "camx30_cal": (30, 0, 0, True), "camy20_cal": (0, 20, 0, True),
    "noise2_cal": (0, 0, 2, True), "noise4_cal": (0, 0, 4, True),
}

parser = argparse.ArgumentParser()
parser.add_argument("condition", nargs="?", default="baseline", choices=list(CONDITIONS), help="条件の名前(--list で一覧)")
parser.add_argument("--list", action="store_true", help="条件の一覧を出して終わる")
parser.add_argument("--seed", type=int, default=0, help="乱数の種(立方体の置き方が変わる)")
parser.add_argument("--speed", type=float, default=1.0, help="再生の速さ(1 = 実時間)")
parser.add_argument("--cam-err-mm", type=float, nargs=2, metavar=("DX", "DY"), help="条件を上書き: カメラの位置のずれ [mm]")
parser.add_argument("--noise-px", type=float, help="条件を上書き: 画素ノイズ [画素]")
parser.add_argument("--calibrate", action="store_true", help="条件を上書き: 較正してから仕分ける")
args = parser.parse_args()

if args.list:
    for name, (dx, dy, noise, cal) in CONDITIONS.items():
        print(f"{name:11s} カメラのずれ x={dx:2d} mm, y={dy:2d} mm / ノイズ {noise} 画素 / 較正 {'あり' if cal else 'なし'}")
    sys.exit(0)

dx, dy, noise, calibrate = CONDITIONS[args.condition]
if args.cam_err_mm is not None:
    dx, dy = args.cam_err_mm
if args.noise_px is not None:
    noise = args.noise_px
calibrate = calibrate or args.calibrate
print(f"条件: {args.condition}(カメラのずれ x={dx:g} mm, y={dy:g} mm / ノイズ {noise:g} 画素 / 較正 {'あり' if calibrate else 'なし'})、配置の種 {args.seed}")

model = load_sort_model(ROOT / "models" / "so101" / "scene_sort.xml")
renderer = mujoco.Renderer(model, IMG_H, IMG_W)
rng = np.random.default_rng(args.seed)
cam_model = None  # None のときは、カメラのずれをそのまま信じる
if calibrate:
    cam = model.camera("overhead")
    true_pos, true_fovy = np.array(cam.pos), float(model.cam_fovy[cam.id])
    est_pos, est_fovy, rms, n_used = calibrate_overhead_camera(model, renderer, true_pos + [dx / 1000, dy / 1000, 0.0], true_fovy, noise, rng)
    cam_model = (est_pos, est_fovy)
    print(f"較正: 手の目印 {n_used} 点 → カメラの位置の差 x,y = {np.linalg.norm(est_pos[:2] - true_pos[:2]) * 1000:.2f} mm、視野角 {est_fovy:.2f}°(本当の値 {true_fovy:.0f}°)")
# ビューアを開く時点で data が要るので、立方体をランダムに置いた data を、ここで作って run_episode に渡す。
data = new_episode(model, random_layout(rng))

CHECK_EVERY = 8  # 何ステップごとに、時間合わせと画面更新の判断をするか
SYNC_INTERVAL = 1 / 50  # 画面を更新する間隔 [秒]

with mujoco.viewer.launch_passive(model, data) as viewer:
    # 初期の視点: 斜め上から、腕・立方体・置き場が入る位置。
    viewer.cam.lookat[:] = [0.15, 0.04, 0.05]
    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 0.8, 100, -45
    start = time.perf_counter()  # perf_counter: 経過時間を測る時計。[頻度] 時間の計測では定番
    last_sync = [start]  # 最後に画面を更新した時刻
    step = [0]  # リストに入れるのは、中の関数 on_step から書き換えるため

    def show_marker(pos_xyz):
        """見つけた位置に、水色(シアン)の球を 1 つ出す(viewer.user_scn: ビューアに追加で描く図形の置き場)。pos_xyz が None なら消す。"""
        with viewer.lock():  # ビューアが描画中に書き換えないよう、鍵をかける。[頻度] ビューアの表示を書き換えるときの定番
            scn = viewer.user_scn
            if pos_xyz is None:
                scn.ngeom = 0
                return
            # mjv_initGeom(図形, 種類, 大きさ, 位置, 向き(3×3 の単位行列を 9 つの並びにしたもの)、色 RGBA): 図形を 1 つ作る
            mujoco.mjv_initGeom(scn.geoms[0], int(mujoco.mjtGeom.mjGEOM_SPHERE), np.array([0.008, 0, 0]), np.asarray(pos_xyz, dtype=float),
                                np.eye(3).flatten(), np.array([0.0, 1.0, 1.0, 0.9], dtype=np.float32))  # 水色(RGBA)。手の目印のマゼンタと区別する
            scn.ngeom = 1

    def on_step():
        """シミュレーションが 1 ステップ進むたびに呼ばれる。実時間に合わせる待ちと、画面の更新をする。"""
        step[0] += 1
        if step[0] % CHECK_EVERY:
            return
        if not viewer.is_running():  # ウィンドウが閉じられた
            sys.exit(0)  # sys.exit: プログラムをここで終える。[頻度] 途中で打ち切るときに使う
        wait = data.time / args.speed - (time.perf_counter() - start)  # sim の時間に、実際の経過時間を合わせる
        if wait > 0:
            time.sleep(wait)
        if time.perf_counter() - last_sync[0] >= SYNC_INTERVAL:
            viewer.sync()  # 計算結果をウィンドウに反映する
            last_sync[0] = time.perf_counter()

    def on_detect(color, det, true_xy):
        """撮影して位置を出した直後。見つけた位置に球を出し、本当の位置との差を表示する。"""
        if det is None:
            print(f"  [{color}] 見つからなかった")
            show_marker(None)
            return
        err = np.linalg.norm(np.array(det[:2]) - true_xy) * 1000
        print(f"  [{color}] 見つけた位置 ({det[0]:.3f}, {det[1]:.3f}) m、本当の位置との差 {err:.1f} mm")
        show_marker([det[0], det[1], 0.035])

    def on_record(rec):
        """1 回の試行が終わった。結果を 1 行で表示する。"""
        result = "成功" if rec["success"] else f"失敗({rec['fail_class']})"
        print(f"  → [{rec['color']}] {rec['attempt']} 回目: {result}、置き場の中心からのずれ {rec['final_err_mm']:.1f} mm、{rec['cycle_s']:.1f} 秒")

    print("再生中...(1 エピソード = 3 個。終わっても、ウィンドウは開いたまま。閉じると終了)")
    viewer.sync()
    records = run_episode(model, renderer, rng, (dx / 1000, dy / 1000), noise, 1, on_step, on_detect, on_record, data, cam_model)
    n_ok = sum(r["final_success"] for r in records if r["attempt"] == 1)
    print(f"このエピソードの結果: 3 個のうち、最終的に成功 {n_ok} 個")
    show_marker(None)
    while viewer.is_running():
        viewer.sync()
        time.sleep(0.05)
renderer.close()
