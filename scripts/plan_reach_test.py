"""【概要】到達精度のテスト(立方体なし)の計画を作る。「手を真下に向けて、机の上 6 cm の空中の目標の点へ、手先(gripperframe)を送る」を、実機でやるための指令値と経路を決め、
outputs/reach_plan.json に保存する(実行は real_run_plan.py、結果の解析は analyze_reach_test.py)。
  最初の計画(反復 1): 肩の軸のまわりの格子(半径 15〜26 cm × 方位)の候補から、(1) IK で届く、(2) 可動域の端から 12° 以上離れる、(3) 手の板(ID 0)がカメラに見える(画像の位置・傾き)、
    (4) 前の点・最初の姿勢からの経路が、机から離れている、を満たす点を選び、広がるように 8 点にする。カメラの位置は、いま撮影して、板から求める。
  補正した計画(反復 2): --correct-from で、反復 1 の計画と、その測定結果(hand_samples.json)から、「測った差を引いた目標」で、IK をやり直す(カメラで見ながら補正する閉ループの 1 回分)。
安全: 目標の手先の高さは 6 cm(--tip-z)。経路は、sim で、腕の動く部分が机から 30 mm 以上、手先が 40 mm 以上離れることを、各区間で確認する(満たさない点は外す)。

使い方(この .venv で実行。カメラが要る):
    .venv/bin/python scripts/plan_reach_test.py                                   # 反復 1(R1)の計画
    .venv/bin/python scripts/plan_reach_test.py --correct-from outputs/reach_plan.json --tag R2 --out outputs/reach_plan2.json   # 補正した計画
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
from hand_sim.aruco_marker import detect_markers  # noqa: E402
from hand_sim.fixed_board import SIDE_MM, camera_pose_from_board  # noqa: E402
from hand_sim.hand_marker import marker_pose_from_hand  # noqa: E402
from hand_sim.real_camera import load_camera, project_points_K  # noqa: E402
from hand_sim.sim_real import JOINTS, real_to_sim, sim_to_real  # noqa: E402
from hand_sim.so101_ik import solve_ik_pose  # noqa: E402
from hand_sim.webcam import grab_until, list_cameras  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--tip-z", type=float, default=0.06, help="目標の手先(gripperframe)の高さ [m]。安全のため 5 cm 以上にする")
parser.add_argument("--n", type=int, default=8, help="目標の点の数")
parser.add_argument("--tag", default="R1", help="この計画の名前。記録するサンプルは「<tag>_<番号>」")
parser.add_argument("--correct-from", help="補正の元になる、前の計画(json)。指定すると、その測定結果で目標を補正する")
parser.add_argument("--out", default=str(ROOT / "outputs" / "reach_plan.json"))
args = parser.parse_args()
if args.tip_z < 0.05:
    sys.exit("--tip-z は 0.05 m 以上にしてください(安全のため)")

model = mujoco.MjModel.from_xml_path(str(ROOT / "models/so101/so101_new_calib.xml"))
data = mujoco.MjData(model)
sid = model.site("gripperframe").id
mujoco.mj_forward(model, data)
AXIS = data.xpos[model.body("shoulder").id][:2].copy()  # 肩の軸(sim の世界座標)
LIM = {"shoulder_pan": 119.1, "shoulder_lift": 106.1, "elbow_flex": 97.3, "wrist_flex": 111.9}  # 実機の可動域の半分(キャリブレーションから)
MARGIN = 12.0  # 可動域の端から、これだけ離す [度]
GRIP_REAL = 9.5  # gripper の実機の値(動かさない)
REST = {"shoulder_pan.pos": -5.9, "shoulder_lift.pos": -103.8, "elbow_flex.pos": 96.2, "wrist_flex.pos": -108.5, "wrist_roll.pos": 0.0, "gripper.pos": 9.6}  # 休めの姿勢(実測の値)
g_lo, g_hi = model.joint("gripper").range
GRIP_Q = float(g_lo + GRIP_REAL / 100.0 * (g_hi - g_lo))  # gripper の実機の値 → sim の角度

# 腕の動く部分(body 番号 3 以上)の当たり判定メッシュの頂点。経路で、机に近づきすぎないかの確認に使う(肩のモーターは動かないので除く)
GEOMS = [g for g in range(model.ngeom) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and model.geom_group[g] == 2 and model.geom_bodyid[g] >= 3]
VERTS = {g: model.mesh_vert[model.mesh_vertadr[model.geom_dataid[g]]:model.mesh_vertadr[model.geom_dataid[g]] + model.mesh_vertnum[model.geom_dataid[g]]] for g in GEOMS}


def hand_down_R(theta_right):
    """手を真下に向けた向き(列 = gripperframe の x, y, z 軸の世界での向き)。x = 真下、z = 放射(外向き)、y = 接線。肩の軸のまわりに、右へ theta_right [rad] 回す。"""
    R0 = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])  # y 軸まわりに 90° 回した向き(x → 真下、z → 前)
    c, s = np.cos(-theta_right), np.sin(-theta_right)  # 右へ回す = 世界の z 軸まわりに、負の向きへ回す
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return Rz @ R0


def target_xyz(r, theta_right, z):
    return np.array([AXIS[0] + r * np.cos(theta_right), AXIS[1] - r * np.sin(theta_right), z])  # 右 = −y


def real_cmd_from_q5(q5):
    """IK の結果(sim の関節角 5 つ)を、実機への指令値 {"名前.pos": 度} にする(OFFSET_DEG を引き、gripper は動かさない値)。"""
    act = sim_to_real(model, np.r_[q5, GRIP_Q])
    act["gripper.pos"] = GRIP_REAL
    # wrist_roll は、実機の読みで 0 にする。OFFSET_DEG のフィットは、roll がほぼ 0 の姿勢で求めた。また、gripperframe は roll の軸の上にほぼ乗るので、手先の位置にはほとんど効かない(数 mm 未満)。
    act["wrist_roll.pos"] = 0.0
    return {k: float(v) for k, v in act.items()}


def within_limits(cmd):
    return all(abs(cmd[f"{n}.pos"]) <= LIM[n] - MARGIN for n in LIM)


def solve(target, theta_right, q_init):
    q5, ok = solve_ik_pose(model, data, target, hand_down_R(theta_right), q_init, max_iter=400, tol_pos=3e-4, tol_rot=3e-3)
    mujoco.mj_forward(model, data)
    return q5, ok and abs(np.degrees(q5[4])) < 8.0  # wrist_roll が大きく振れた解は使わない(OFFSET の +5.7° を含めて、roll が小さい解)


def min_clearance(cmd_a, cmd_b, steps=41):
    """2 つの指令値の間を、関節ごとに直線で動いたときの、(腕の動く部分の最低の高さ, 手先の最低の高さ) [m]。"""
    lo_g, lo_t = 9.9, 9.9
    for t in np.linspace(0, 1, steps):
        obs = {k: cmd_a[k] + (cmd_b[k] - cmd_a[k]) * t for k in cmd_a}
        q = real_to_sim(model, obs)
        data.qpos[:6] = q
        mujoco.mj_forward(model, data)
        lo_g = min(lo_g, min((VERTS[g] @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])[:, 2].min() for g in GEOMS))
        lo_t = min(lo_t, data.site_xpos[sid][2])
    return lo_g, lo_t


def seconds_between(a, b):
    biggest = max(abs(a[k] - b[k]) for k in a if k != "gripper.pos")
    return float(np.clip(biggest / 12.0, 4.0, 14.0))  # 1 秒あたり 12° 以下(試した、ゆっくりした動きと同じ)


def hand_marker_view(q5, cam_pos, cam_R, K, size):
    """その関節角のとき、手の板(ID 0)が画像のどこに写るか(縦・横 0〜1)と、面の傾き [度]。"""
    data.qpos[:6] = np.r_[q5, GRIP_Q]
    mujoco.mj_forward(model, data)
    Rm, pm = marker_pose_from_hand(data.site_xmat[sid].reshape(3, 3).copy(), data.site_xpos[sid].copy())
    u, v = project_points_K(pm[None], cam_pos, cam_R, K)[0]
    to = cam_pos - pm
    tilt = np.degrees(np.arccos(np.clip(Rm[:, 2] @ to / np.linalg.norm(to), -1, 1)))
    return u / size[0], v / size[1], float(tilt)


plan = {"tag": args.tag, "tip_z": args.tip_z, "assumed_start": REST, "home": REST, "targets": []}

if args.correct_from:
    prev = json.loads(Path(args.correct_from).read_text())
    samples = {s["label"]: s for s in json.loads((ROOT / "outputs" / "hand_samples.json").read_text())}
    q_init = real_to_sim(model, {k: v for k, v in prev["targets"][0]["cmd"].items()})[:5]
    items = []
    for t in prev["targets"]:
        label = f"{prev['tag']}_{t['id']}"
        if label not in samples:
            print(f"{label}: 測定結果が無いので、補正しない(飛ばす)")
            continue
        meas = np.array(samples[label]["hand_pos_m"])
        tgt = np.array(t["target_xyz"])
        err = meas - tgt  # 測った位置 − 目標(世界座標、m)
        new_tgt = tgt - err  # 差を引いた目標(反復 1 の差が、同じ向き・大きさで再現すると仮定)
        q5, ok = solve(new_tgt, np.radians(t["theta_right_deg"]), real_to_sim(model, t["cmd"])[:5])
        cmd = real_cmd_from_q5(q5)
        if not (ok and within_limits(cmd)):
            print(f"{label}: 補正した目標の IK が解けない・可動域の端に近いので、飛ばす")
            continue
        items.append({"id": t["id"], "target_xyz": tgt.tolist(), "command_xyz": new_tgt.tolist(), "theta_right_deg": t["theta_right_deg"], "r": t["r"],
                      "cmd": cmd, "predicted_uv": t.get("predicted_uv"), "prev_error_mm": (err * 1000).tolist()})
    cands = items
    print(f"補正した計画: {len(cands)} 点。反復 1 の差(mm、測った − 目標)の平均の大きさ {np.mean([np.linalg.norm(c['prev_error_mm']) for c in cands]):.1f}")
else:
    cam = load_camera()
    K, size = cam["K"], cam["size"]
    t_cam = None
    for name, devs in list_cameras():
        if "WebCamera" in name:
            t_cam = devs[0]
    if t_cam is None:
        sys.exit("ウェブカメラが見つからない(USB を確認)")

    def good(im):
        f_ = detect_markers(im, K, cam["dist"], SIDE_MM / 1000)
        return len([i for i in f_ if 1 <= i <= 6]) >= 5

    img, used = grab_until(t_cam, good, settle_frames=15, timeout=8)
    found = detect_markers(img, K, cam["dist"], SIDE_MM / 1000)
    res = camera_pose_from_board(found, K, cam["dist"], SIDE_MM)
    if res is None:
        sys.exit("板が読めない。カメラの位置・明るさを確認する")
    cam_pos, cam_R, err_px, n_b = res
    print(f"カメラの位置(cm) {np.round(cam_pos * 100, 1)}、板 {n_b} 個、再投影誤差 {err_px:.2f} 画素、露出 {used}")
    plan["camera_pos_m"] = cam_pos.tolist()

    cands = []
    q_init = real_to_sim(model, {k: v for k, v in {**REST, "shoulder_lift.pos": 13.1, "elbow_flex.pos": 1.4, "wrist_flex.pos": 82.0, "shoulder_pan.pos": 0.0}.items()})[:5]
    for r in (0.15, 0.19, 0.23, 0.26):
        for th_deg in (-30, -15, 0, 15, 30, 45, 60):
            th = np.radians(th_deg)
            tgt = target_xyz(r, th, args.tip_z)
            q5, ok = solve(tgt, th, q_init)
            if not ok:
                continue
            cmd = real_cmd_from_q5(q5)
            u, v, tilt = hand_marker_view(q5, cam_pos, cam_R, K, size)
            if within_limits(cmd) and 0.10 < u < 0.90 and 0.17 < v < 0.85 and tilt < 52:
                cands.append({"target_xyz": tgt.tolist(), "theta_right_deg": th_deg, "r": r, "cmd": cmd, "predicted_uv": [u, v], "tilt": tilt})
    print(f"条件を満たす候補: {len(cands)} 点(IK が解け、可動域に余裕があり、手の板が見える)")
    if len(cands) < 4:
        sys.exit("候補が少なすぎる。カメラの位置・向きを確認する")
    # 広がるように選ぶ(最遠点サンプリング): 最初は中央に近い点、以降は「すでに選んだ点から最も遠い点」を足す。
    xy = np.array([c["target_xyz"][:2] for c in cands])
    chosen = [int(np.argmin(np.linalg.norm(xy - xy.mean(axis=0), axis=1)))]
    while len(chosen) < min(args.n, len(cands)):
        dmin = [min(np.linalg.norm(xy[i] - xy[j]) for j in chosen) for i in range(len(cands))]
        chosen.append(int(np.argmax(dmin)))
    cands = [cands[i] for i in chosen]
    cands.sort(key=lambda c: (c["theta_right_deg"], c["r"]))  # 方位順(動きが少なくなる)
    for i, c in enumerate(cands):
        c["id"] = i + 1

# 経路の確認: 休めの姿勢 → 点 1 → … → 点 N → 休めの姿勢。机に近づきすぎる区間があれば、その点を外す。
def check_route(items):
    prev_cmd, ok_items = REST, []
    for c in items:
        lo_g, lo_t = min_clearance(prev_cmd, c["cmd"])
        if lo_g >= 0.030 and lo_t >= 0.040:
            c["clearance_mm"] = [float(round(lo_g * 1000, 1)), float(round(lo_t * 1000, 1))]
            c["seconds"] = seconds_between(prev_cmd, c["cmd"])
            ok_items.append(c)
            prev_cmd = c["cmd"]
        else:
            print(f"点 {c['id']}: 経路が机に近い(腕 {lo_g * 1000:.0f} mm、手先 {lo_t * 1000:.0f} mm)ので外す")
    return ok_items


plan["targets"] = check_route(cands)
lo_g, lo_t = min_clearance(plan["targets"][-1]["cmd"], REST)
plan["return_seconds"] = seconds_between(plan["targets"][-1]["cmd"], REST)
print(f"最後の点 → 休めの姿勢の経路: 腕 {lo_g * 1000:.0f} mm、手先 {lo_t * 1000:.0f} mm" + ("(OK)" if lo_g >= 0.030 else "(危ない: 外す点を増やす)"))
print(f"\n計画 {args.tag}: {len(plan['targets'])} 点(手先の高さ {args.tip_z * 100:.0f} cm)")
print(f"{'番号':>4s}{'半径[cm]':>9s}{'方位(右)[°]':>12s}  目標 x, y, z [mm]            手の板(縦,横,傾き)       腕/手先の最低の高さ[mm]  動かす秒数")
for c in plan["targets"]:
    uv = c.get("predicted_uv")
    print(f"{c['id']:4d}{c['r'] * 100:9.0f}{c['theta_right_deg']:12.0f}  {np.round(np.array(c['target_xyz']) * 1000, 0)!s:27s}  "
          + (f"({uv[1]:.2f}, {uv[0]:.2f})" if uv else "-") + f"  {c['clearance_mm']}  {c['seconds']:.0f}")
Path(args.out).write_text(json.dumps(plan, ensure_ascii=False, indent=2))
print(f"\n保存: {args.out}")
