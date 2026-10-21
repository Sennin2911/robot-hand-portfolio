"""【概要】marker_calib.py(固定の目印でのカメラ較正 → 手の目印での腕のずれの推定)が正しく動くかを確認する。
(1) 机の固定の目印 10 個が、すべて見つかり、カメラの位置が 30 mm ずれた出発点から、位置の誤差 2 mm 以内に較正できる。
(2) 腕の関節がずれている(0, 3, −8, −4, 0)°とき、手の目印で、そのずれを 0.2° 以内で推定できる(ノイズなし)。
(3) 推定したずれで補正すると、腕のずれがあっても、1 エピソード(3 個)が全部成功する(補正しないと、失敗する)。
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
import hand_sim.so101_grasp as grasp  # noqa: E402
from hand_sim.marker_calib import calibrate_from_fixed_markers, detect_fixed_marker_pixels, estimate_arm_offsets, fixed_marker_positions  # noqa: E402
from hand_sim.sorting import IMG_H, IMG_W, load_sort_model, run_episode  # noqa: E402

model = load_sort_model(ROOT / "models" / "so101" / "scene_sort.xml")
renderer = mujoco.Renderer(model, IMG_H, IMG_W)
cam = model.camera("overhead")
true_pos, true_fovy = np.array(cam.pos), float(model.cam_fovy[cam.id])
ARM_ERR = (0, 3, -8, -4, 0)
n_ng = 0


def check(label, ok):
    """結果を 1 行表示し、失敗なら数える。global: 関数の外の変数 n_ng を書き換える宣言。[頻度] 小さなスクリプトでだけ使う"""
    global n_ng
    print(f"{'OK' if ok else 'NG'}  {label}")
    n_ng += 0 if ok else 1


print("--- (1) 固定の目印で、カメラを較正する ---")
n_expected = len(fixed_marker_positions(model))
n_found = len(detect_fixed_marker_pixels(model, renderer))
check(f"固定の目印が、全部見つかる(XML に {n_expected} 個、画像で {n_found} 個)", n_found == n_expected == 10)
pos, fovy, rms, n_used = calibrate_from_fixed_markers(model, renderer, true_pos + [0.03, 0.0, 0.0], true_fovy)
err_xy = np.linalg.norm(pos[:2] - true_pos[:2]) * 1000
check(f"カメラの位置(x, y)の誤差 {err_xy:.2f} mm(2 mm 未満)、再投影誤差 {rms:.2f} 画素、目印 {n_used} 個", err_xy < 2.0 and n_used == 10 and rms < 1.0)

print("--- (2) 手の目印で、腕のずれを推定する ---")
grasp.set_arm_error(ARM_ERR)
delta, rms2, n_hand = estimate_arm_offsets(model, renderer, pos, fovy, yaws_deg=(0.0, 90.0))
worst = np.abs(delta - np.array(ARM_ERR)).max()
check(f"推定 {np.round(delta, 2)} と、本当のずれ {ARM_ERR} の差の最大 {worst:.2f}°(0.2° 未満、手の目印 {n_hand} 点)", worst < 0.2)

print("--- (3) 推定したずれで補正すると、仕分けが通る ---")
records = run_episode(model, renderer, np.random.default_rng(1), (0.0, 0.0), 0.0, 0, cam_model=(pos, fovy))
good = not any(r["success"] for r in records)
check(f"補正しない: 3 個とも失敗する(腕のずれがあるので。成功 {sum(r['success'] for r in records)} 個)", good)
grasp.set_arm_compensation(delta)
records = run_episode(model, renderer, np.random.default_rng(1), (0.0, 0.0), 0.0, 0, cam_model=(pos, fovy))
check(f"補正する: 3 個とも成功する(成功 {sum(r['success'] for r in records)} 個)", all(r["success"] for r in records))
grasp.set_arm_error(0)
grasp.set_arm_compensation(0)

renderer.close()
print("全部OK" if n_ng == 0 else f"NG が {n_ng} 件")
sys.exit(0 if n_ng == 0 else 1)
