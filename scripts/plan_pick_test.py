"""【概要】実機で、立方体を 1 個「掴んで、置き場へ運ぶ」ための計画を作り、outputs/pick_plan_<色>.json に保存する(実行は real_pick_place.py)。
立方体の位置・向きは、detect_cubes_world.py が保存した outputs/cubes_detected.json から読む。置き場は、plan_pad_test.py と同じ位置。
掴み方・置き方の姿勢は、sim の仕分けと同じ部品(so101_grasp.plan_for_cube)で求め、sim_real.sim_to_real で実機の値にする。
段階: 1 立方体の真上へ(指を開く)→ 2 降下 → 3 指を閉じる → 4 持ち上げる → 5 置き場の真上へ → 6 降下 → 7 指を開く → 8 上がる。
安全: 腕の動く部分が、移動の途中で机から 30 mm 以上離れるかを、段階 1・4・5・8 と、最後の「休めの姿勢へ戻る」経路で、sim で確認する
(降下の段階 2・6 は、立方体・置き場に近づくので、確認の対象外。短い距離を、ゆっくり動かし、作者が目で見て確認する)。

使い方(この .venv で実行): .venv/bin/python scripts/plan_pick_test.py --color red
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")  # mujoco を import する「前」に設定する(import の時点で描画方式が決まるため)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from hand_sim.cube_world import PADS  # noqa: E402  置き場の中心(judge_real_trial.py と共有)
from hand_sim.grasp_calibration import load_site_offset  # noqa: E402
from hand_sim.sim_real import real_to_sim, sim_to_real  # noqa: E402
from hand_sim.so101_grasp import GRIP_CLOSE, GRIP_OPEN, plan_for_cube  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--color", required=True, choices=["red", "green", "blue"])
parser.add_argument("--hand-error", type=float, nargs=2, metavar=("DX_MM", "DY_MM"), default=(0.0, 0.0),
                    help="実機で測った「手先 − 目標」のずれ(世界座標 x = 前、y = 左、mm)。指定すると、そのぶん反対へずらした目標で計画する(掴む位置・置く位置の両方)。例: 手先が目標より後ろへ 8 mm、右へ 5 mm → --hand-error -8 -5")
args = parser.parse_args()

CUBE_M = 0.030
REST = {"shoulder_pan.pos": -5.9, "shoulder_lift.pos": -103.8, "elbow_flex.pos": 96.2, "wrist_flex.pos": -108.5, "wrist_roll.pos": 0.0, "gripper.pos": 9.6}
LIM = {"shoulder_pan": 119.1, "shoulder_lift": 106.1, "elbow_flex": 97.3, "wrist_flex": 111.9}  # 実機の可動域の半分
MARGIN = 8.0  # 可動域の端から離す [度]

det_file = ROOT / "outputs" / "cubes_detected.json"
if not det_file.exists():  # detect_cubes_world.py は、検出に失敗すると、古い結果を消す(古い位置で計画を作らないため)
    sys.exit("検出結果が無い(検出に失敗した、または、まだ検出していない)。.venv/bin/python scripts/detect_cubes_world.py をやり直す。暗いときは露出を上げる")
det = json.loads(det_file.read_text())["cubes"]
if args.color not in det:
    sys.exit(f"{args.color} の立方体が検出されていない(detect_cubes_world.py をもう一度実行する)")
cube = det[args.color]

model = mujoco.MjModel.from_xml_path(str(ROOT / "models/so101/so101_new_calib.xml"))
data = mujoco.MjData(model)
sid = model.site("gripperframe").id
AXIS = data.xpos[model.body("shoulder").id][:2].copy() if mujoco.mj_forward(model, data) is None else None
r_cube = float(np.hypot(cube["x_m"] - AXIS[0], cube["y_m"] - AXIS[1]))
print(f"{args.color}: 位置 ({cube['x_m'] * 1000:.0f}, {cube['y_m'] * 1000:.0f}) mm、肩の軸から {r_cube * 100:.1f} cm、向き {cube['yaw_deg']:.0f}°")
if r_cube > 0.23:
    sys.exit("肩の軸から 23 cm より遠い。手を真下に向けて届かない可能性が高いので、立方体をロボットに近づけて、検出し直す")

err = np.array(args.hand_error) * 1e-3  # 手先 − 目標 のずれ [m]。目標からこの分を引けば、手先が本来の位置に来る(ずれが同じ向き・大きさで出る前提)
if err.any():
    print(f"手先のずれの補正: 目標を ({-err[0] * 1000:+.1f}, {-err[1] * 1000:+.1f}) mm ずらして計画する")
site_offset = load_site_offset()  # 実機で掴んだときの測定から貯めた、指の面のずれ(手の座標 [m])。まだ測定が無ければ 0(update_grasp_offset.py が貯める)
if site_offset.any():
    print(f"掴みの校正(実機で掴んだ測定の中央値): 手の座標 z を {site_offset[2] * 1000:+.1f} mm ずらして計画する")
poses, hand_yaw = plan_for_cube(model, np.array([cube["x_m"] - err[0], cube["y_m"] - err[1], CUBE_M / 2]), np.array(PADS[args.color]) - err, np.radians(cube["yaw_deg"]),
                                site_offset=site_offset)
g_lo, g_hi = model.joint("gripper").range


def real_cmd(q5, grip_rad):
    """sim の腕の関節角 5 つ(rad)と指の角度(rad)→ 実機の指令値 {"名前.pos": 値}。"""
    act = sim_to_real(model, np.r_[q5, grip_rad])
    return {k: float(v) for k, v in act.items()}


G_OPEN, G_CLOSE = real_cmd(poses["grasp"], GRIP_OPEN)["gripper.pos"], real_cmd(poses["grasp"], GRIP_CLOSE)["gripper.pos"]
stages = [
    ("1 立方体の真上へ(指を開く)", "above", GRIP_OPEN),
    ("2 降下", "grasp", GRIP_OPEN),
    ("3 指を閉じる", "grasp", GRIP_CLOSE),
    ("4 持ち上げる", "above", GRIP_CLOSE),
    ("5 置き場の真上へ", "place_above", GRIP_CLOSE),
    ("6 降下", "place", GRIP_CLOSE),
    ("7 指を開く", "place", GRIP_OPEN),
    ("8 上がる", "place_above", GRIP_OPEN),
]
plan = {"tag": f"PICK_{args.color}", "assumed_start": REST, "home": REST, "color": args.color, "cube": cube, "hand_yaw_deg": float(np.degrees(hand_yaw)),
        "gripper_open": G_OPEN, "gripper_close": G_CLOSE, "stages": []}
for name, pose, grip in stages:
    cmd = real_cmd(poses[pose], grip)
    bad = [n for n in LIM if abs(cmd[f"{n}.pos"]) > LIM[n] - MARGIN]
    if bad:
        sys.exit(f"段階 {name}: {bad} が可動域の端に近い({[round(cmd[f'{n}.pos'], 1) for n in bad]})。計画を作りません")
    plan["stages"].append({"name": name, "pose": pose, "cmd": cmd})

GEOMS = [g for g in range(model.ngeom) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and model.geom_group[g] == 2 and model.geom_bodyid[g] >= 3]
VERTS = {g: model.mesh_vert[model.mesh_vertadr[model.geom_dataid[g]]:model.mesh_vertadr[model.geom_dataid[g]] + model.mesh_vertnum[model.geom_dataid[g]]] for g in GEOMS}


def clearance(a, b, steps=41):
    """a から b へ、関節ごとに直線で動くときの、腕の動く部分の最低の高さ [m]。"""
    lo = 9.9
    for t in np.linspace(0, 1, steps):
        data.qpos[:6] = real_to_sim(model, {k: a[k] + (b[k] - a[k]) * t for k in a})
        mujoco.mj_forward(model, data)
        lo = min(lo, min((VERTS[g] @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])[:, 2].min() for g in GEOMS))
    return lo


def tip_z(cmd):
    data.qpos[:6] = real_to_sim(model, cmd)
    mujoco.mj_forward(model, data)
    return float(data.site_xpos[sid][2])


def seconds_between(a, b):
    return float(np.clip(max(abs(a[k] - b[k]) for k in a if k != "gripper.pos") / 12.0, 3.0, 14.0))  # 1 秒あたり 12° 以下


chk = [("休めの姿勢 → 1", REST, plan["stages"][0]["cmd"]), ("3 → 4(持ち上げ)", plan["stages"][2]["cmd"], plan["stages"][3]["cmd"]),
       ("4 → 5(運ぶ)", plan["stages"][3]["cmd"], plan["stages"][4]["cmd"]), ("7 → 8(上がる)", plan["stages"][6]["cmd"], plan["stages"][7]["cmd"]),
       ("1 → 休めの姿勢(途中でやめたとき)", plan["stages"][0]["cmd"], REST), ("8 → 休めの姿勢", plan["stages"][7]["cmd"], REST)]
ok_all = True
for label, a, b in chk:
    c = clearance({**a, "gripper.pos": 0}, {**b, "gripper.pos": 0})
    c0 = clearance({**a, "gripper.pos": 0}, {**a, "gripper.pos": 0}, steps=1)  # 出発点の高さ
    # 掴む・置く姿勢から上がる動き(3 → 4、7 → 8)は、出発点がもともと机に近い(指の先)。「出発点より低くならない」ことを確かめる。
    ok = c >= 0.030 or (c0 < 0.030 and c >= c0 - 0.001)
    ok_all &= ok
    print(f"経路 {label}: 腕の最低の高さ {c * 1000:.0f} mm(出発点 {c0 * 1000:.0f} mm) {'OK' if ok else '危ない'}")
if not ok_all:
    sys.exit("机に近い経路がある。計画を作りません")
prev = REST
for st in plan["stages"]:
    st["seconds"] = 6.0 if st["name"][0] in "26" else (1.5 if st["name"][0] in "37" else seconds_between(prev, st["cmd"]))
    prev = st["cmd"]
plan["return_seconds"] = seconds_between(prev, REST)
print(f"手の向き(指の開閉の向き)yaw = {plan['hand_yaw_deg']:.0f}°。指の値: 開く {G_OPEN:.1f}、閉じる {G_CLOSE:.1f}")
print(f"掴む姿勢の gripperframe の高さ(sim)= {tip_z(plan['stages'][1]['cmd']) * 1000:.1f} mm(0 = 机の面)")
print(f"{'段階':28s}{'pan':>7s}{'lift':>7s}{'elbow':>7s}{'wflex':>7s}{'roll':>7s}{'grip':>6s}  秒")
for st in plan["stages"]:
    c = st["cmd"]
    print(f"{st['name']:24s}{c['shoulder_pan.pos']:7.1f}{c['shoulder_lift.pos']:7.1f}{c['elbow_flex.pos']:7.1f}{c['wrist_flex.pos']:7.1f}{c['wrist_roll.pos']:7.1f}{c['gripper.pos']:6.1f}  {st['seconds']:.1f}")
out = ROOT / "outputs" / f"pick_plan_{args.color}.json"
out.write_text(json.dumps(plan, ensure_ascii=False, indent=2))
print(f"保存: {out.relative_to(ROOT)}")
