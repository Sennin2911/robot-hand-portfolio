"""【概要】手の目印によるカメラの較正(sorting.calibrate_overhead_camera)が、どんな「想定外」にどれだけ強いかを、sim で調べる。
(1) 出発点(信じているカメラの位置)がずれていても、同じ答えに収束するか。
(2) カメラが想定外に傾いている(較正のモデルは「真下を向いている」前提)とき、どうなるか。
(3) 手の目印の位置を測り間違えた(または、腕が計算どおりの位置に来ない)とき、どうなるか。
(4) 較正のときの画素の測定にノイズがあるとき、どうなるか。
指標は「立方体の位置の認識の誤差 [mm]」(本当の位置との差。ランダムに置いた立方体 15 配置 × 3 色 = 45 個の平均と最大)。
立方体を掴めるのは、この誤差が約 10 mm 以内のとき(docs/sorting_trials.md)。実機は動かさない。結果の読み方は docs/sorting_trials.md の「較正を入れた試験」。

使い方(この .venv で実行):
    .venv/bin/python scripts/run_calibration_study.py
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
    COLORS, IMG_H, IMG_W, calibrate_overhead_camera, detect_cubes, load_sort_model, new_episode, random_layout, true_cube_pose,
)

model = load_sort_model(ROOT / "models" / "so101" / "scene_sort.xml")
renderer = mujoco.Renderer(model, IMG_H, IMG_W)
cam = model.camera("overhead")
TRUE_POS, TRUE_FOVY = np.array(cam.pos), float(model.cam_fovy[cam.id])
QUAT0 = model.cam_quat[cam.id].copy()  # カメラの本当の向き(クォータニオン)。傾きを変えた後に戻すために控える


def recognition_error(cam_model, n_layouts=15, seed=0):
    """較正した (カメラの位置, 視野角) cam_model で、立方体の位置を認識したときの誤差 [mm] の (平均, 最大) を返す。"""
    rng = np.random.default_rng(seed)
    errs = []
    for _ in range(n_layouts):
        data = new_episode(model, random_layout(rng))
        found = detect_cubes(model, data, renderer, cam_model=cam_model)
        for c in COLORS:
            errs.append(np.linalg.norm(np.array(found[c][:2]) - true_cube_pose(model, data, c)[0]) * 1000)
    return float(np.mean(errs)), float(np.max(errs))


def set_tilt(deg):
    """カメラを、画像の横方向(x 軸)のまわりに deg 度、本当に傾ける(較正のモデルには知らせない)。"""
    half = np.radians(deg) / 2
    rot = np.array([np.cos(half), np.sin(half), 0.0, 0.0])  # x 軸まわりに deg 回すクォータニオン
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, QUAT0, rot)  # 2 つの回転を合成する(元の向き × 傾き)
    model.cam_quat[cam.id] = out


print("(1) 出発点(信じているカメラの位置)のずれ → 較正の結果")
for dx, dy, dz, dfov in [(0, 0, 0, 0), (15, 0, 0, 0), (30, 0, 0, 0), (0, 20, 0, 0), (30, -20, 0, 3), (20, 20, 30, 0), (40, 40, 0, -4)]:
    pos, fovy, rms, n = calibrate_overhead_camera(model, renderer, TRUE_POS + np.array([dx, dy, dz]) / 1000, TRUE_FOVY + dfov)
    print(f"  出発点のずれ x={dx:3d} y={dy:3d} z={dz:3d} mm、視野角 {dfov:+d}° → 位置の差 x,y = {np.linalg.norm(pos[:2] - TRUE_POS[:2]) * 1000:.2f} mm、"
          f"視野角 {fovy:.2f}°、再投影誤差 {rms:.2f} 画素(目印 {n} 点)")

print("(2) カメラが想定外に傾いている(出発点は 20 mm ずれ)→ 認識の誤差")
print("     傾き | 較正なし(カメラの位置は、本当の値を教える) | 較正あり")
for deg in (0, 1, 2, 3):
    set_tilt(deg)
    no_cal = recognition_error(None)  # None: 本当の XML の位置・視野角を信じる(傾きは知らない)
    pos, fovy, rms, n = calibrate_overhead_camera(model, renderer, TRUE_POS + [0.02, 0, 0], TRUE_FOVY)
    cal = recognition_error((pos, fovy))
    print(f"    {deg}° | 平均 {no_cal[0]:5.1f} mm・最大 {no_cal[1]:5.1f} mm | 平均 {cal[0]:4.1f} mm・最大 {cal[1]:4.1f} mm(再投影誤差 {rms:.2f} 画素)")
set_tilt(0)

print("(3) 手の目印の位置を、x 方向に測り間違えた → 認識の誤差")
for err_mm in (0, 3, 5, 10):
    pos, fovy, rms, n = calibrate_overhead_camera(model, renderer, TRUE_POS + [0.02, 0, 0], TRUE_FOVY, marker_pos_err=(err_mm / 1000, 0, 0))
    cal = recognition_error((pos, fovy))
    print(f"    目印の位置の誤差 {err_mm:2d} mm → 平均 {cal[0]:4.1f} mm・最大 {cal[1]:4.1f} mm")

print("(4) 較正のときの画素の測定にノイズ(5 回の較正の、平均と最悪)→ 認識の誤差(平均)")
for noise in (0, 1, 2, 4):
    errs = []
    for seed in range(5):
        pos, fovy, rms, n = calibrate_overhead_camera(model, renderer, TRUE_POS + [0.02, 0, 0], TRUE_FOVY, noise, np.random.default_rng(seed))
        errs.append(recognition_error((pos, fovy), n_layouts=8, seed=seed)[0])
    print(f"    画素ノイズ {noise} 画素 → 平均 {np.mean(errs):.1f} mm、最悪 {np.max(errs):.1f} mm")
renderer.close()
