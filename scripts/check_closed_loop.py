"""【概要】closed-loop の視覚サーボが、open-loop より正確かを確認する。
「プログラムが信じているカメラの位置」を、本当の位置からわざとずらす(実機での較正ミスの再現)。
その状態で、同じ立方体の位置に対して open-loop(1回だけ動く)と closed-loop(撮り直して補正)を比べ、
手先と立方体の水平方向のずれ(本当の位置どうしの距離)を表示する。
closed-loop のずれが 5mm 未満、かつ open-loop より小さければ OK。全部通れば終了コード0、そうでなければ1。
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
from hand_sim.visual_servo import (  # noqa: E402
    gripper_position, load_model_with_marker, locate_cube, move_arm, plan_hover, servo_closed_loop)

W, H = 640, 480
HOVER_Z = 0.10
TOL = 0.005  # closed-loop に求める精度 [m] (5mm)
HOME = [0, -1.7, 1.6, 1.0, 0, 0]
PLACES = [(0.22, -0.12), (0.25, 0.12), (0.3, -0.05), (0.15, 0.1)]
# 立方体の位置は、(0.2, 0.0) を避けた: 2cm ずれたカメラを信じて動くと、手先の位置が立方体と 2.5cm しか離れず、
# グリッパが立方体にぶつかって弾いてしまう(open-loop でも起きる。弾かれた後は補正のしようがない)。
# 「信じているカメラの位置」のずれ (dx, dy, dz) [m]。本当の位置は (0.2, 0, 0.6)。
CAM_ERRORS = {"ずれなし": (0, 0, 0), "横に2cm": (0.02, -0.015, 0), "高さ+3cm": (0, 0, 0.03)}

model = load_model_with_marker(ROOT / "models" / "so101" / "scene_vs.xml")  # 手先に青い目印を足したモデル
data = mujoco.MjData(model)
cube_adr = model.joint("cube_free").qposadr[0]
true_cam = np.array(model.camera("overhead").pos)


def reset(x, y):
    mujoco.mj_resetData(model, data)
    data.qpos[:6] = HOME
    data.ctrl[:6] = HOME
    data.qpos[cube_adr:cube_adr + 3] = [x, y, 0.015]
    mujoco.mj_forward(model, data)


def gap_to_cube():
    """手先と立方体の、本当の水平方向のずれ [m]。立方体は腕に弾かれて動くことがあるので、今の位置と比べる。"""
    return float(np.linalg.norm(gripper_position(model, data)[:2] - data.qpos[cube_adr:cube_adr + 2]))


def cube_moved(x, y):
    """立方体が最初の位置から 2mm 以上動いたか(腕にぶつかって弾いた)。"""
    return bool(np.linalg.norm(data.qpos[cube_adr:cube_adr + 2] - np.array([x, y])) > 0.002)


n_ng = 0
with mujoco.Renderer(model, H, W) as renderer:
    for label, err in CAM_ERRORS.items():
        assumed = true_cam + np.array(err)
        print(f"--- 信じているカメラの位置: {label} ---")
        for x, y in PLACES:
            reset(x, y)  # open-loop: 1回測って1回動く
            q, _ = plan_hover(model, locate_cube(model, data, renderer, cam_pos=assumed), HOVER_Z, HOME[:5])
            move_arm(model, data, q)
            g_open = gap_to_cube()

            reset(x, y)  # closed-loop: 撮り直しながら補正
            hist = servo_closed_loop(model, data, renderer, HOVER_Z, cam_pos=assumed)
            g_closed = gap_to_cube()
            hit = cube_moved(x, y)  # closed-loop の最中に立方体を弾いていないか

            ok = g_closed < TOL and g_closed <= g_open + 0.0005 and not hit
            n_ng += 0 if ok else 1
            print(f"{'OK' if ok else 'NG'}  立方体=({x:+.3f}, {y:+.3f})  open-loop={g_open * 1000:5.1f} mm  "
                  f"closed-loop={g_closed * 1000:5.1f} mm  (撮り直し {len(hist)} 回)"
                  f"{'  ※立方体を弾いた' if hit else ''}")

print("全部OK" if n_ng == 0 else f"{n_ng} 件NGです")
sys.exit(0 if n_ng == 0 else 1)
