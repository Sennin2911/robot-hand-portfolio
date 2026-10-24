"""【概要】置き場(赤・緑・青の枠)の真上へ、手を真下に向けて送る試験の計画を作り、outputs/pad_plan.json に保存する(実行は real_run_plan.py --no-record)。
置き場は、ロボットの左側(世界座標 y = +0.12〜+0.17 m)で、いまのカメラには映らない。そのため、plan_reach_test.py と違い、カメラは使わない
(手の板の見え方も条件にしない)。指令値は、sim の IK で求め、関節の読み(--refine)で定常偏差を打ち消して送る。
安全: 経路(休めの姿勢 → 緑 → 赤 → 青 → 休めの姿勢)を sim で確認し、腕の動く部分が机から 30 mm 以上、手先が 40 mm 以上離れていなければ、計画を作らない。
手先の高さは 6 cm(置き場の枠より十分上)。その場で止まって、枠の中心の真上に来ているかを、作者が目で見て確かめる。

使い方(この .venv で実行): .venv/bin/python scripts/plan_pad_test.py
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")  # mujoco を import する「前」に設定する(import の時点で描画方式が決まるため)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from hand_sim.sim_real import real_to_sim, sim_to_real  # noqa: E402
from hand_sim.so101_ik import solve_ik_pose  # noqa: E402

TIP_Z = 0.06  # 手先の高さ [m]
# 置き場の中心(sim の世界座標 x, y [m])。板の十字から測る位置は、緑 = 左へ 170 mm・前へ +29 mm、赤 = 左へ 170 mm・前へ −41 mm、
# 青 = 左へ 120 mm・前へ +99 mm。
# 青だけ、scene_sort.xml(sim)の位置 (0.21, 0.17) から、左右を 5 cm 内側へずらした。理由: (0.21, 0.17) は肩の軸から 24 cm で、手を真下に向けて 6 cm の高さへ
# 届かない(IK が解けない。手首の曲げが可動域の端に達する)。(0.21, 0.12) なら 21 cm で届く(余裕あり)。
# 動かす順は、緑 → 赤 → 青(--only 1 2 で、青を動かさずに緑・赤だけを試せるように)。
PADS = [("green", 0.14, 0.17), ("red", 0.07, 0.17), ("blue", 0.21, 0.12)]

model = mujoco.MjModel.from_xml_path(str(ROOT / "models/so101/so101_new_calib.xml"))
data = mujoco.MjData(model)
sid = model.site("gripperframe").id
mujoco.mj_forward(model, data)
AXIS = data.xpos[model.body("shoulder").id][:2].copy()  # 肩の軸(sim の世界座標)
LIM = {"shoulder_pan": 119.1, "shoulder_lift": 106.1, "elbow_flex": 97.3, "wrist_flex": 111.9}  # 実機の可動域の半分
MARGIN = 12.0  # 可動域の端から離す [度]
GRIP_REAL = 9.5
REST = {"shoulder_pan.pos": -5.9, "shoulder_lift.pos": -103.8, "elbow_flex.pos": 96.2, "wrist_flex.pos": -108.5, "wrist_roll.pos": 0.0, "gripper.pos": 9.6}
g_lo, g_hi = model.joint("gripper").range
GRIP_Q = float(g_lo + GRIP_REAL / 100.0 * (g_hi - g_lo))
GEOMS = [g for g in range(model.ngeom) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and model.geom_group[g] == 2 and model.geom_bodyid[g] >= 3]
VERTS = {g: model.mesh_vert[model.mesh_vertadr[model.geom_dataid[g]]:model.mesh_vertadr[model.geom_dataid[g]] + model.mesh_vertnum[model.geom_dataid[g]]] for g in GEOMS}


def hand_down_R(theta_right):
    """手を真下に向けた向き。肩の軸のまわりに、右へ theta_right [rad] 回す(左なら負)。plan_reach_test.py と同じ。"""
    R0 = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    c, s = np.cos(-theta_right), np.sin(-theta_right)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ R0


def real_cmd_from_q5(q5):
    act = sim_to_real(model, np.r_[q5, GRIP_Q])
    act["gripper.pos"] = GRIP_REAL
    act["wrist_roll.pos"] = 0.0  # plan_reach_test.py と同じ理由(roll の読みを 0 にする)
    return {k: float(v) for k, v in act.items()}


def min_clearance(a, b, steps=41):
    """a から b へ、関節ごとに直線で動いたときの、(腕の動く部分の最低の高さ, 手先の最低の高さ) [m]。"""
    lo_g, lo_t = 9.9, 9.9
    for t in np.linspace(0, 1, steps):
        q = real_to_sim(model, {k: a[k] + (b[k] - a[k]) * t for k in a})
        data.qpos[:6] = q
        mujoco.mj_forward(model, data)
        lo_g = min(lo_g, min((VERTS[g] @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])[:, 2].min() for g in GEOMS))
        lo_t = min(lo_t, data.site_xpos[sid][2])
    return lo_g, lo_t


def seconds_between(a, b):
    return float(np.clip(max(abs(a[k] - b[k]) for k in a if k != "gripper.pos") / 12.0, 4.0, 14.0))  # 1 秒あたり 12° 以下


plan = {"tag": "PAD", "tip_z": TIP_Z, "assumed_start": REST, "home": REST, "camera_pos_m": None, "targets": []}
q_init = real_to_sim(model, {**REST, "shoulder_lift.pos": 13.1, "elbow_flex.pos": 1.4, "wrist_flex.pos": 82.0, "shoulder_pan.pos": 0.0})[:5]
prev = REST
for i, (color, x, y) in enumerate(PADS, start=1):
    dx, dy = x - AXIS[0], y - AXIS[1]
    r, theta_right = float(np.hypot(dx, dy)), float(np.arctan2(-dy, dx))  # 右 = −y なので、左の置き場の theta_right は負
    tgt = np.array([x, y, TIP_Z])
    q5, ok = solve_ik_pose(model, data, tgt, hand_down_R(theta_right), q_init, max_iter=400, tol_pos=3e-4, tol_rot=3e-3)
    cmd = real_cmd_from_q5(q5)
    if not (ok and abs(np.degrees(q5[4])) < 8.0 and all(abs(cmd[f"{n}.pos"]) <= LIM[n] - MARGIN for n in LIM)):
        sys.exit(f"{color}: IK が解けない、または可動域の端に近い(IK {ok}、roll {np.degrees(q5[4]):.1f}°、指令 {np.round([cmd[k] for k in cmd], 1)})。計画を作りません")
    lo_g, lo_t = min_clearance(prev, cmd)
    if lo_g < 0.030 or lo_t < 0.040:
        sys.exit(f"{color}: 経路が机に近い(腕 {lo_g * 1000:.0f} mm、手先 {lo_t * 1000:.0f} mm)。計画を作りません")
    plan["targets"].append({"id": i, "color": color, "target_xyz": tgt.tolist(), "theta_right_deg": round(float(np.degrees(theta_right)), 1), "r": r,
                            "cmd": cmd, "clearance_mm": [round(lo_g * 1000, 1), round(lo_t * 1000, 1)], "seconds": seconds_between(prev, cmd)})
    prev, q_init = cmd, q5
lo_g, lo_t = min_clearance(prev, REST)
if lo_g < 0.030:
    sys.exit("最後の点 → 休めの姿勢の経路が机に近い。計画を作りません")
plan["return_seconds"] = seconds_between(prev, REST)

print(f"計画 PAD: {len(plan['targets'])} 点(手先の高さ {TIP_Z * 100:.0f} cm)。最後 → 休めの姿勢: 腕 {lo_g * 1000:.0f} mm")
print(f"{'番号':>4s} {'色':6s}{'半径[cm]':>9s}{'方位(右が+)[°]':>15s}  腕/手先の最低の高さ[mm]  秒数  指令値(pan, lift, elbow, wrist_flex)")
for t in plan["targets"]:
    c = t["cmd"]
    print(f"{t['id']:4d} {t['color']:6s}{t['r'] * 100:9.1f}{t['theta_right_deg']:15.1f}  {t['clearance_mm']}  {t['seconds']:4.0f}  "
          f"({c['shoulder_pan.pos']:.1f}, {c['shoulder_lift.pos']:.1f}, {c['elbow_flex.pos']:.1f}, {c['wrist_flex.pos']:.1f})")
out = ROOT / "outputs" / "pad_plan.json"
out.write_text(json.dumps(plan, ensure_ascii=False, indent=2))
print(f"保存: {out.relative_to(ROOT)}")
