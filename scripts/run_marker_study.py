"""【概要】較正用の目印を「手に付ける」か「机(土台のそば)に固定する」かの違いを、sim で調べる。
(1) 腕の関節が、sim のモデルとずれている(実機の関節のオフセットの再現)とき、手の目印による較正と仕分けが、どうなるか。
    手の目印の位置は「指令した関節角から計算」するので、腕がずれていると、計算した位置と、カメラが見る実際の位置がずれる。
(2) 机の面に置いた固定の目印(位置は、腕と無関係に、定規などで分かる)で較正すると、どうなるか。
    (画像は合成: 本当のカメラで、目印の画素を計算する。固定の目印は、腕に依存しないので、sim で腕を動かす必要はない。)
結果の読み方は docs/marker_placement.md。実機は動かさない。

使い方(この .venv で実行):
    .venv/bin/python scripts/run_marker_study.py
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

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import hand_sim.so101_grasp as grasp  # noqa: E402
from hand_sim.calibration import calibrate_camera, project  # noqa: E402
from hand_sim.sorting import IMG_H, IMG_W, calibrate_overhead_camera, load_sort_model, run_episode  # noqa: E402
from hand_sim.vision import pixel_to_world  # noqa: E402

model = load_sort_model(ROOT / "models" / "so101" / "scene_sort.xml")
renderer = mujoco.Renderer(model, IMG_H, IMG_W)
cam = model.camera("overhead")
TRUE_POS, TRUE_FOVY = np.array(cam.pos), float(model.cam_fovy[cam.id])


def run(label, arm_error_deg, calibrate, mount_err=(0.0, 0.0, 0.0), n_episodes=12):
    """腕のずれ arm_error_deg(度、5 関節)、較正の有無、目印の位置の測り間違い mount_err [m] で、仕分けを n_episodes 回(3 個ずつ)試す。"""
    grasp.set_arm_error(arm_error_deg)
    cam_model = None
    if calibrate:
        pos, fovy, _, _ = calibrate_overhead_camera(model, renderer, TRUE_POS + [0.02, 0.0, 0.0], TRUE_FOVY, marker_pos_err=mount_err)
        cam_model = (pos, fovy)
    rng = np.random.default_rng(0)
    ok = total = 0
    det_err, final_err = [], []
    for _ in range(n_episodes):
        for rec in run_episode(model, renderer, rng, (0.0, 0.0), 0.0, 1, cam_model=cam_model):
            if rec["attempt"] == 1:
                total += 1
                ok += rec["success"]
                det_err.append(rec["det_err_mm"])
                final_err.append(rec["final_err_mm"])
    print(f"  {label}: 1 回目の成功 {ok}/{total}、認識の誤差 平均 {np.nanmean(det_err):.1f} mm、置き場の中心からのずれ 平均 {np.nanmean(final_err):.1f} mm", flush=True)
    grasp.set_arm_error(0)


print("(1) 腕の関節のずれ(度)と、手の目印による較正")
BIG, SMALL = (0, 3, -8, -4, 0), (0, 1, -2, -1, 0)  # BIG は、実機で測った関節のオフセット(補正する前の値)と同じ大きさ。SMALL は、補正した後の残りに近い大きさ
run("腕のずれなし、較正なし(カメラの位置は正しい)", 0, False)
run(f"腕のずれ大 {BIG}、較正なし", BIG, False)
run(f"腕のずれ大 {BIG}、手の目印で較正", BIG, True)
run(f"腕のずれ小 {SMALL}、較正なし", SMALL, False)
run(f"腕のずれ小 {SMALL}、手の目印で較正", SMALL, True)
run("腕のずれなし、目印の位置を 8 mm 測り間違えて較正", 0, True, (0.008, 0, 0))

print("(2) 机に固定した目印(腕と無関係)で較正する(合成データ。真上のカメラ、出発点は位置 約 40 mm ずれ・視野角 5° ずれ)")
GRID = [(x, y) for x in (0.10, 0.20, 0.30) for y in (-0.15, 0.0, 0.15)]


def workspace_error(points, noise_px, seed):
    """目印の世界座標 points から、画素を合成(本当のカメラ + 画素ノイズ)→ 較正 → 立方体の上面(高さ 3 cm)の座標の誤差 [mm](平均)。"""
    rng = np.random.default_rng(seed)
    pixels = project(points, TRUE_POS, TRUE_FOVY, IMG_W, IMG_H) + rng.normal(0, noise_px, (len(points), 2))
    pos, fovy, _ = calibrate_camera(points, pixels, IMG_W, IMG_H, TRUE_POS + [0.02, 0.01, 0.03], TRUE_FOVY + 5.0)
    errs = []
    for x in (0.14, 0.20, 0.26):
        for y in (-0.14, -0.04, 0.06):
            u, v = project(np.array([[x, y, 0.03]]), TRUE_POS, TRUE_FOVY, IMG_W, IMG_H)[0]
            est = pixel_to_world(u, v, pos, fovy, IMG_W, IMG_H, 0.03)
            errs.append(np.hypot(est[0] - x, est[1] - y) * 1000)
    return float(np.mean(errs))


for name, pts in [("机の面だけ(高さ 0 の 9 点)", np.array([[x, y, 0.0] for x, y in GRID])),
                  ("机の面 + 高さ 8 cm の台(18 点)", np.array([[x, y, z] for x, y in GRID for z in (0.0, 0.08)]))]:
    for noise in (0.0, 0.5, 1.0):
        errs = [workspace_error(pts, noise, s) for s in range(10)]
        print(f"  {name}、画素ノイズ {noise}: 立方体上面の座標の誤差 平均 {np.mean(errs):.1f} mm、最悪 {np.max(errs):.1f} mm")
renderer.close()
