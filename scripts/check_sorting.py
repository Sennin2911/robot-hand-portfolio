"""【概要】sorting.py(部品仕分け)が正しく動くかを確認する。
(1) 3 色の立方体を、ランダムな位置・向きに置いて撮影 → 色ごとに見つけられて、位置の誤差 3 mm 未満、向きの誤差 5° 未満。
    置き場の薄い色を、立方体と取り違えない(3 色とも、見つけた位置が立方体の位置で、置き場の位置ではない)。
(2) 視野の外へ出した立方体は、見つからない(None が返る)。
(3) 1 エピソード(3 個)で、全部が成功して、それぞれの置き場に入る。記録の中身(失敗の種類の欄、所要時間)が入っている。
(4) カメラの位置を 30 mm ずれて信じていても、手の目印で較正すると、位置の認識の誤差が 3 mm 未満に戻り、1 エピソードが全部成功する。
全部通れば終了コード0、そうでなければ1。
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
from hand_sim.sorting import (  # noqa: E402
    COLORS, IMG_H, IMG_W, ZONES, calibrate_overhead_camera, detect_cubes, load_sort_model, new_episode, random_layout,
    run_episode, true_cube_pose, yaw_error_deg,
)

model = load_sort_model(ROOT / "models" / "so101" / "scene_sort.xml")
renderer = mujoco.Renderer(model, IMG_H, IMG_W)
rng = np.random.default_rng(0)  # 乱数の種を固定 → 毎回同じ配置(再現性)
n_ng = 0

print("--- (1) 3 色の位置と向きを、見つけられる ---")
pos_err, yaw_err, n_miss = [], [], 0
for _ in range(20):
    layout = random_layout(rng)
    data = new_episode(model, layout)
    found = detect_cubes(model, data, renderer)
    for color in COLORS:
        if found[color] is None:
            n_miss += 1
            continue
        true_xy, true_yaw = true_cube_pose(model, data, color)
        pos_err.append(np.linalg.norm(np.array(found[color][:2]) - true_xy) * 1000)
        yaw_err.append(yaw_error_deg(found[color][2], true_yaw))
good = n_miss == 0 and max(pos_err) < 3.0 and max(yaw_err) < 5.0
n_ng += not good
print(f"{'OK' if good else 'NG'}  60 個: 見つからなかった {n_miss} 個、位置の誤差 平均 {np.mean(pos_err):.2f} mm・最大 {max(pos_err):.2f} mm、向きの誤差 平均 {np.mean(yaw_err):.2f}°・最大 {max(yaw_err):.1f}°")

print("--- (2) 視野の外の立方体は、見つからない ---")
data = new_episode(model, random_layout(rng))
data.qpos[model.jnt_qposadr[model.joint("cube_red_free").id]:][:3] = [2.0, 2.0, 0.015]  # 赤を視野の外へ(freejoint の qpos の先頭 3 つが位置)
mujoco.mj_forward(model, data)
found = detect_cubes(model, data, renderer)
good = found["red"] is None and found["green"] is not None and found["blue"] is not None
n_ng += not good
print(f"{'OK' if good else 'NG'}  赤は None、緑と青は見つかる: {[c for c in COLORS if found[c] is not None]}")

print("--- (3) 1 エピソード(3 個)が、全部成功する ---")
records = run_episode(model, renderer, np.random.default_rng(1))
good = len(records) == 3 and all(r["success"] for r in records) and all(r["fail_class"] == "" for r in records)
for r in records:
    zone = np.array(ZONES[r["color"]])
    print(f"    {r['color']}: 成功={r['success']} 認識の誤差 {r['det_err_mm']:.1f} mm、置き場の中心からのずれ {r['final_err_mm']:.1f} mm、{r['cycle_s']:.1f} 秒")
good = good and all(r["final_err_mm"] < 30 and r["cycle_s"] > 0 for r in records)
n_ng += not good
print(f"{'OK' if good else 'NG'}  3 個とも成功し、置き場の中に入った")

print("--- (4) カメラの位置が 30 mm ずれて信じていても、較正で戻る ---")
cam = model.camera("overhead")
true_pos, true_fovy = np.array(cam.pos), float(model.cam_fovy[cam.id])
est_pos, est_fovy, rms, n_used = calibrate_overhead_camera(model, renderer, true_pos + [0.03, 0.0, 0.0], true_fovy)
errs = []
for _ in range(10):
    data = new_episode(model, random_layout(rng))
    found = detect_cubes(model, data, renderer, cam_model=(est_pos, est_fovy))
    errs += [np.linalg.norm(np.array(found[c][:2]) - true_cube_pose(model, data, c)[0]) * 1000 for c in COLORS]
good = n_used >= 8 and max(errs) < 3.0
n_ng += not good
print(f"{'OK' if good else 'NG'}  較正に使えた目印 {n_used} 点、認識の誤差 平均 {np.mean(errs):.2f} mm・最大 {max(errs):.2f} mm(較正なしなら 30 mm)")
records = run_episode(model, renderer, np.random.default_rng(2), (0.03, 0.0), cam_model=(est_pos, est_fovy))
good = all(r["success"] for r in records)
n_ng += not good
print(f"{'OK' if good else 'NG'}  較正した上で、1 エピソード(3 個)が全部成功する")

renderer.close()  # 描画用の GL の部品を、終了前に明示的に片づける(しないと、終了時に EGL の警告が出ることがある)
print("全部OK" if n_ng == 0 else f"NG が {n_ng} 件")
sys.exit(0 if n_ng == 0 else 1)
