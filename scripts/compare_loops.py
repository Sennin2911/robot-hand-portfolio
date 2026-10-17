"""【概要】「内側だけ(1回で動く = open-loop)」と「内側+外側(撮り直して補正 = closed-loop)」を、
同じ立方体の位置・同じカメラのずれで走らせ、誤差の出方を比べる。
  内側 = ヤコビアン IK の反復(最大200回。腕は動かさず、計算だけ)。
  外側 = 動かす → 青い目印を撮り直す → 目標を直す、の繰り返し(2〜3回)。
出力: 表(ターミナル)、グラフ outputs/compare_loops.png、最終姿勢の画像 outputs/compare_loops_final.png。
(説明は docs/closed_loop_flow.md)
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # 理由は check_install.py を参照
if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "egl")  # ★mujoco を import する前に設定(理由は check_install.py)

import matplotlib  # noqa: E402
matplotlib.use("Agg")  # 画面なしで画像ファイルに保存するモード。★pyplot を import する前に書く。[頻度] ヘッドレス環境の定番
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from hand_sim.visual_servo import (  # noqa: E402
    gripper_position, load_model_with_marker, locate_cube, move_arm, plan_hover, servo_closed_loop)

W, H = 640, 480
HOVER_Z = 0.10
HOME = [0, -1.7, 1.6, 1.0, 0, 0]
CUBE = (0.25, 0.12)  # 立方体の位置 [m]
# 「信じているカメラの位置」のずれ (dx, dy, dz) [m]。本当の位置は (0.2, 0, 0.6)。
CAM_ERRORS = {"no error": (0, 0, 0), "2cm sideways": (0.02, -0.015, 0), "height +3cm": (0, 0, 0.03)}

model = load_model_with_marker(ROOT / "models" / "so101" / "scene_vs.xml")
data = mujoco.MjData(model)
cube_adr = model.joint("cube_free").qposadr[0]
true_cam = np.array(model.camera("overhead").pos)


def reset():
    mujoco.mj_resetData(model, data)
    data.qpos[:6] = HOME
    data.ctrl[:6] = HOME
    data.qpos[cube_adr:cube_adr + 3] = [*CUBE, 0.015]
    mujoco.mj_forward(model, data)


def true_gap():
    """手先と立方体の、本当の水平方向のずれ [mm]。"""
    return 1000 * float(np.linalg.norm(gripper_position(model, data)[:2] - data.qpos[cube_adr:cube_adr + 2]))


def final_crop(renderer):
    """overhead カメラの画像から、立方体のまわり 120×120 画素を切り出す(最終姿勢の確認用)。"""
    renderer.update_scene(data, camera="overhead")
    img = renderer.render()
    f = (H / 2) / np.tan(np.radians(model.cam_fovy[model.camera("overhead").id]) / 2)  # 焦点距離 [画素]
    depth = true_cam[2] - 0.03
    u = int(W / 2 + (CUBE[0] - true_cam[0]) / depth * f)  # 立方体の画素位置(pixel_to_world の逆)
    v = int(H / 2 - (CUBE[1] - true_cam[1]) / depth * f)
    return img[max(v - 60, 0):v + 60, max(u - 60, 0):u + 60]


results = {}  # 条件名 → 結果
crops = {}
with mujoco.Renderer(model, H, W) as renderer:
    for label, err in CAM_ERRORS.items():
        assumed = true_cam + np.array(err)

        # --- 内側だけ(open-loop): 1回測って、IK(最大200回)を解いて、ctrl に1回だけ入れる ---
        reset()
        ik_hist = []
        q, _ = plan_hover(model, locate_cube(model, data, renderer, cam_pos=assumed), HOVER_Z, HOME[:5], history=ik_hist)
        move_arm(model, data, q)
        open_res = {"ik": ik_hist, "gap": true_gap()}
        if label == "2cm sideways":
            crops["inner only"] = final_crop(renderer)

        # --- 内側+外側(closed-loop): 動かすたびに記録する ---
        reset()
        cycles = []

        def record(i, ik_hist):
            cycles.append({"ik": ik_hist, "gap": true_gap()})  # 動かした直後の、本当のずれ

        servo_closed_loop(model, data, renderer, HOVER_Z, cam_pos=assumed, on_step=record)
        results[label] = (open_res, cycles)
        if label == "2cm sideways":
            crops["inner + outer"] = final_crop(renderer)

# --- 表 ---
print(f"立方体の位置 = {CUBE}。「ずれ」= 手先と立方体の、本当の水平方向の距離。")
for label, (o, cycles) in results.items():
    print(f"\n=== 信じているカメラの位置: {label} ===")
    print(f"  内側だけ     : IK {len(o['ik']):3d} 回の反復(計算上の距離 {o['ik'][-1] * 1000:.2f} mm) → ctrl 1回 → 本当のずれ {o['gap']:5.1f} mm")
    for i, c in enumerate(cycles):
        print(f"  内側+外側 {i + 1}回目: IK {len(c['ik']):3d} 回の反復(計算上の距離 {c['ik'][-1] * 1000:.2f} mm) → ctrl {i + 1}回目 → 本当のずれ {c['gap']:5.1f} mm")

# --- グラフ: 上段=内側の収束(計算だけ)、下段=本当のずれ(ctrl を入れるたび) ---
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for col, (label, (o, cycles)) in enumerate(results.items()):
    ax = axes[0, col]
    ax.semilogy(np.array(o["ik"]) * 1000, "k-", lw=2, label="inner only (1 IK run)")
    for i, c in enumerate(cycles):
        ax.semilogy(np.array(c["ik"]) * 1000, "--", label=f"inner+outer: IK run #{i + 1}")
    ax.set_title(f"camera: {label}\nIK: distance to target (computed)")
    ax.set_xlabel("IK iteration"); ax.set_ylabel("mm (log)"); ax.legend(fontsize=7)
    ax = axes[1, col]
    ax.plot([1], [o["gap"]], "ks", ms=9, label="inner only")
    ax.plot(range(1, len(cycles) + 1), [c["gap"] for c in cycles], "o-", label="inner+outer")
    ax.axhline(2, color="gray", ls=":", lw=1)  # closed-loop の終了基準 2mm
    ax.set_title("true gap: hand vs cube (after each ctrl update)")
    ax.set_xlabel("number of ctrl updates"); ax.set_ylabel("mm"); ax.set_xticks(range(1, max(len(cycles), 1) + 1))
    ax.legend(fontsize=8)
fig.tight_layout()
out = ROOT / "outputs"
out.mkdir(exist_ok=True)
fig.savefig(out / "compare_loops.png", dpi=110)

# --- 最終姿勢(カメラが 2cm ずれたとき): overhead から見た、立方体のまわり ---
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
for ax, (name, img) in zip(axes, crops.items()):
    ax.imshow(img); ax.set_title(f"{name}\n(camera 2cm off, top view)"); ax.axis("off")
fig.tight_layout()
fig.savefig(out / "compare_loops_final.png", dpi=110)
print("\n保存: outputs/compare_loops.png, outputs/compare_loops_final.png")
