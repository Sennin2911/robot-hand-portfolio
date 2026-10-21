"""【概要】実機で測った手先・軸の位置から、sim と実機の「関節角の原点のずれ(オフセット)」を最小二乗法で求める。
考え方: 実機に送った角度 q に対して、本当の腕の形は sim の「q + オフセット」の姿勢だと仮定する。
オフセット(shoulder_lift、elbow_flex、wrist_flex の 3 つ)を動かして、sim の予測が実測に最も近くなる値を探す。
実機は動かさない。測定値は、下の MEASURED に書き足していく(点3・点4 を測ったら足す)。

使い方(この .venv で実行):
    .venv/bin/python scripts/fit_joint_offsets.py
    .venv/bin/python scripts/fit_joint_offsets.py --joints elbow_flex wrist_flex   # 肘から先だけにオフセットを許す
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)。
sys.path.insert(0, str(ROOT / "src"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from hand_sim.sim_real import JOINTS, real_to_sim  # noqa: E402

# 測定値。1 行 = 1 つの姿勢。pose は real_send_pose.py の --pose と同じ値(度、gripper は 0〜100)。
# 測った量だけを書く(測っていないものは書かない)。長さの単位は cm、向きは度。
#   radius: 肩の軸から手先までの水平距離   azimuth: 向き(正面 0、左 +)   tip_z: 手先の高さ(机の面から)
#   elbow_z / wrist_z: 肘・手首の軸の高さ(机の面から)
#   chord: pose と pose2 の 2 つの姿勢での手先どうしの水平距離(弦)。肩の軸の位置を測らなくてよい(pan だけ違う姿勢で測る)
MEASURED = [
    {"name": "全関節0", "pose": [0, 0, 0, 0, 0, 10], "elbow_z": 23.0, "wrist_z": 24.3, "tip_z": 25.5},
    {"name": "点1", "pose": [0.0, -65.9, 87.0, -7.7, 0.0, 10.0], "radius": 23.5, "tip_z": 14.5},
    {"name": "点3", "pose": [0.0, -19.8, 55.3, -16.9, 0.0, 10.0], "radius": 30.0, "tip_z": 13.3},
    {"name": "点2", "pose": [0.0, -100.0, 68.0, 68.3, 0.0, 10.0], "radius": 15.5, "tip_z": 13.8},  # 半径 = 机の端から 21.5 − 軸 6.0
    {"name": "点8", "pose": [0.0, -39.2, 79.9, -10.4, 0.0, 10.0], "radius": 24.0, "tip_z": 8.3},  # 半径 = 机の端から 30.0 − 軸 6.0
    {"name": "点4", "pose": [59.9, -65.9, 86.9, -7.6, 3.3, 10.0], "radius": 24.0, "tip_z": 13.7},
    {"name": "弦(pan ±60)", "pose": [60.0, -65.9, 87.0, -7.7, 0.0, 10.0], "pose2": [-60.0, -65.9, 87.0, -7.7, 0.0, 10.0], "chord": 39.0},
]
# オフセットを求める関節。--joints で絞れる(例: --joints elbow_flex wrist_flex は「肘から先だけ」ずれていると仮定した当てはめ)。
parser = argparse.ArgumentParser()
parser.add_argument("--joints", nargs="+", default=["shoulder_lift", "elbow_flex", "wrist_flex"],
                    choices=["shoulder_lift", "elbow_flex", "wrist_flex"])
# --no-radius: 半径の測定を使わない(肩の軸の位置を目測した半径は、約 1 cm の誤差を含むため。弦と高さだけで当てはめる)。
parser.add_argument("--no-radius", action="store_true")
args = parser.parse_args()
FIT_JOINTS = args.joints
SIGMA = 0.5  # 測定の誤差の目安 [cm](向きは、この長さを半径で割った角度を目安にする)

model = mujoco.MjModel.from_xml_path(str(ROOT / "models" / "so101" / "scene.xml"))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
pan_axis = data.xanchor[model.joint("shoulder_pan").id][:2].copy()  # 肩の回転軸の世界座標 (x, y)
fit_idx = [JOINTS.index(n) for n in FIT_JOINTS]


def predict(pose, offsets_deg):
    """実機の指令 pose に、オフセット [度] を足した姿勢での、sim の予測(cm / 度)を辞書で返す。"""
    # apply_offset=False: sim_real.py に入れたオフセットを足さない。ここでは、測定時に実機へ送った素の指令から求め直すため。
    q = real_to_sim(model, {f"{n}.pos": v for n, v in zip(JOINTS, pose)}, apply_offset=False)
    q[fit_idx] += np.radians(offsets_deg)  # 配列の [番号のリスト] で、その要素だけをまとめて取り出して足す。[頻度] numpy で定番
    data.qpos[:6] = q
    mujoco.mj_forward(model, data)
    tip = data.site_xpos[model.site("gripperframe").id]
    d = tip[:2] - pan_axis
    return {"radius": np.hypot(*d) * 100, "azimuth": np.degrees(np.arctan2(d[1], d[0])), "tip_z": tip[2] * 100,
            "elbow_z": data.xanchor[model.joint("elbow_flex").id][2] * 100,
            "wrist_z": data.xanchor[model.joint("wrist_flex").id][2] * 100, "xy": tip[:2].copy()}


def residuals(offsets_deg):
    """(予測 − 実測) を、誤差の目安 SIGMA で割ってならべた配列。これを小さくするオフセットを探す。"""
    res = []
    for m in MEASURED:
        p = predict(m["pose"], offsets_deg)
        if "pose2" in m:  # 弦: 2 つの姿勢の手先の距離を、予測の項目に足す
            p["chord"] = np.linalg.norm(p["xy"] - predict(m["pose2"], offsets_deg)["xy"]) * 100
        for key in ("radius", "azimuth", "tip_z", "elbow_z", "wrist_z", "chord"):
            if key in m and not (key == "radius" and args.no_radius):
                scale = np.degrees(SIGMA / p["radius"]) if key == "azimuth" else SIGMA
                res.append((p[key] - m[key]) / scale)
    return np.array(res)


# ガウス・ニュートン法: 残差を「オフセットの 1 次式」と見なして解き、これを繰り返す(so101_ik.py と同じ考え方)。
# ヤコビアン(オフセットを少し動かしたときの残差の変化)は、数値微分で作る。
x = np.zeros(len(FIT_JOINTS))
for _ in range(20):
    r0 = residuals(x)
    J = np.stack([(residuals(x + 1e-3 * np.eye(len(x))[i]) - r0) / 1e-3 for i in range(len(x))], axis=1)
    step, *_ = np.linalg.lstsq(J, -r0, rcond=None)  # J @ step = -r0 を最小二乗で解く。[頻度] 定番
    x += step
    if np.abs(step).max() < 1e-6:
        break

n_eq = len(residuals(x))
print(f"測定の式の数 {n_eq}、求めるオフセット {len(x)} 個(式が少ないほど、答えは当てにならない)\n")
print("求めたオフセット [度](実機の角度に足すと、sim の姿勢と合う。+ は sim の + の向き):")
for n, v in zip(FIT_JOINTS, x):
    print(f"  {n:14s} {v:+.2f}")
# 不確かさの目安: (J の転置 × J) の逆行列の対角の平方根 = 各オフセットの標準偏差(SIGMA で割ってあるので、そのまま度の単位)。
cov = np.linalg.pinv(J.T @ J)
print("  目安の幅(±1σ): " + ", ".join(f"{n} ±{np.sqrt(cov[i, i]):.1f}" for i, n in enumerate(FIT_JOINTS)))

print("\n当てはめ前後の差(予測 − 実測。cm、azimuth は度):")
print(f"{'':10s}{'項目':10s}{'前':>8s}{'後':>8s}")
for m in MEASURED:
    before, after = predict(m["pose"], np.zeros(len(x))), predict(m["pose"], x)
    if "pose2" in m:
        before["chord"] = np.linalg.norm(before["xy"] - predict(m["pose2"], np.zeros(len(x)))["xy"]) * 100
        after["chord"] = np.linalg.norm(after["xy"] - predict(m["pose2"], x)["xy"]) * 100
    for key in ("radius", "azimuth", "tip_z", "elbow_z", "wrist_z", "chord"):
        if key in m:
            print(f"{m['name']:10s}{key:10s}{before[key] - m[key]:+8.2f}{after[key] - m[key]:+8.2f}")
