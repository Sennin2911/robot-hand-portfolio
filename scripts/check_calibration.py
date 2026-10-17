"""【概要】カメラの較正(src/hand_sim/calibration.py)が、カメラの位置・視野角を正しく求められるかを確認する。
手順: 腕を色々な位置に動かして手先の青い目印を撮影(23 点前後)→ わざと間違えた出発点から較正 →
  (1) 求めたカメラの位置・視野角が、本当の値にどれだけ近いか
  (2) 較正前(間違った値)/ 較正後 / 本当の値 のそれぞれで、立方体の上に手先を持っていったときの水平ずれ(open-loop)
を比べる。較正後のずれが 3mm 未満で、較正前より小さければ OK。全部通れば終了コード0、そうでなければ1。
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # 理由は check_install.py を参照
if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "egl")  # ★mujoco を import する前に設定(理由は check_install.py)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from hand_sim.calibration import calibrate_camera  # noqa: E402
from hand_sim.visual_servo import (  # noqa: E402
    collect_calibration_samples, gripper_position, load_model_with_marker, locate_cube, move_arm, plan_hover)

W, H = 640, 480
HOVER_Z = 0.10
TOL = 0.003  # 較正後に求める精度 [m] (3mm)
HOME = [0, -1.7, 1.6, 1.0, 0, 0]
# 較正用に手先を動かす位置 (x, y, z) [m]。高さを3段階に変えるのが大事: 高さが違うと「画素 → 実寸」の
# 倍率が変わるので、カメラの高さと視野角(どちらも倍率に効く)を区別できる。
TARGETS = [(x, y, z) for z in (0.08, 0.15, 0.22) for x in (0.15, 0.25, 0.32) for y in (-0.15, 0.0, 0.15)]
CUBES = [(0.22, -0.12), (0.25, 0.12), (0.3, -0.05), (0.15, 0.1)]  # 較正には使っていない位置で確かめる
# 較正の出発点 = 「間違った見当」: (カメラ位置のずれ dx, dy, dz [m]、視野角 [度])
WRONG = {"横2cm+高さ3cm": ((0.02, -0.015, 0.03), 50.0), "横3cm+高さ-4cm": ((-0.03, 0.02, -0.04), 40.0)}

model = load_model_with_marker(ROOT / "models" / "so101" / "scene_vs.xml")
data = mujoco.MjData(model)
cube_adr = model.joint("cube_free").qposadr[0]
cam_id = model.camera("overhead").id
true_cam = np.array(model.camera("overhead").pos)
true_fovy = float(model.cam_fovy[cam_id])


def open_loop_gap(renderer, x, y, cam_pos, fovy):
    """このカメラ値を信じて open-loop で動かしたときの、手先と立方体の水平ずれ [m]。"""
    mujoco.mj_resetData(model, data)
    data.qpos[:6] = HOME
    data.ctrl[:6] = HOME
    data.qpos[cube_adr:cube_adr + 3] = [x, y, 0.015]
    mujoco.mj_forward(model, data)
    q, _ = plan_hover(model, locate_cube(model, data, renderer, cam_pos=cam_pos, fovy_deg=fovy), HOVER_Z, HOME[:5])
    move_arm(model, data, q)
    return float(np.linalg.norm(gripper_position(model, data)[:2] - np.array([x, y])))


n_ng = 0
with mujoco.Renderer(model, H, W) as renderer:
    mujoco.mj_resetData(model, data)
    points, pixels = collect_calibration_samples(model, data, renderer, TARGETS, HOME[:5])
    print(f"較正用のデータ: {len(points)} 点 (目標 {len(TARGETS)} 点のうち、IK が届き、目印が見えた点)")
    print(f"本当のカメラ: 位置={true_cam}  視野角={true_fovy:.2f}度\n")

    for label, (d, fovy0) in WRONG.items():
        cam0 = true_cam + np.array(d)
        cam, fovy, rms = calibrate_camera(points, pixels, W, H, cam0, fovy0)
        print(f"--- 出発点(間違った見当): {label}  位置={np.round(cam0, 3)}  視野角={fovy0:.1f}度 ---")
        print(f"  較正の結果: 位置={np.round(cam, 4)}  視野角={fovy:.2f}度  再投影誤差={rms:.2f} 画素  "
              f"(位置のずれ {np.linalg.norm(cam - true_cam) * 1000:.1f} mm)")
        for x, y in CUBES:
            g0 = open_loop_gap(renderer, x, y, cam0, fovy0)
            g1 = open_loop_gap(renderer, x, y, cam, fovy)
            gt = open_loop_gap(renderer, x, y, true_cam, true_fovy)
            ok = g1 < TOL and g1 < g0
            n_ng += 0 if ok else 1
            print(f"  {'OK' if ok else 'NG'}  立方体=({x:+.2f}, {y:+.2f})  較正前={g0 * 1000:5.1f} mm  "
                  f"較正後={g1 * 1000:5.1f} mm  (参考: 本当のカメラ値なら {gt * 1000:.1f} mm)")

print("\n全部OK" if n_ng == 0 else f"\n{n_ng} 件NGです")
sys.exit(0 if n_ng == 0 else 1)
