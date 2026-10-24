"""【概要】record_hand_sample.py / real_pan_sweep.py で記録した、複数のサンプル(腕の形を変えたもの)から、実機の関節のずれ(OFFSET_DEG の補正量)と、板の置き方の補正を、一度に求める。
各サンプルは「実機の関節角」と「画像から出した手先の位置・向き」の組。sim で、関節角(+ OFFSET_DEG + 補正量)から手先の位置・向きを出し、画像の値との差(位置 [mm]、向き [度])が最小になる補正量を探す。
求める 7 つ: 板の補正(肩の軸まわりの向き dyaw [度]、水平の位置 dx, dy [mm])と、関節の補正量(shoulder_lift、elbow_flex、wrist_flex、wrist_roll [度])。
  pan のずれは求めない(板の向き dyaw と、pan のずれは、手先の位置の見え方が同じで、区別できないため。pan は、今の OFFSET_DEG のままとして、
  「pan のずれ + 板の向きの誤差」を、dyaw にまとめて出す。紙の置き方の誤差のほうが大きいと見ている)。
どの値が、どれだけ確かか(±)も出す。腕の形が 1 通りだと、関節の 3 つが分けられず、± が大きくなる(警告を出す)。形を変えるほど、確かになる。
最小二乗法: 差の 2 乗の和が最小になる値を探す。外れ値に引きずられないよう、ロバストな重み(soft_l1)を使う。

使い方(この .venv で実行):
    .venv/bin/python scripts/fit_real_offsets.py [--samples outputs/hand_samples.json]
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
from hand_sim.sim_real import JOINTS, OFFSET_DEG, real_to_sim  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402  (回転行列 ↔ 回転ベクトルの変換に使う)

parser = argparse.ArgumentParser()
parser.add_argument("--samples", default=str(ROOT / "outputs" / "hand_samples.json"))
parser.add_argument("--sigma-pos", type=float, default=3.0, help="位置の誤差の見積もり [mm](重み。画像の手先の位置の、ばらつきの目安)")
parser.add_argument("--sigma-rot", type=float, default=3.0, help="向きの誤差の見積もり [度]")
args = parser.parse_args()

samples = json.loads(Path(args.samples).read_text())
if len(samples) < 6:
    sys.exit(f"サンプルが {len(samples)} 個。6 個以上(腕の形を変えて)記録する")
model = mujoco.MjModel.from_xml_path(str(ROOT / "models/so101/so101_new_calib.xml"))
data = mujoco.MjData(model)
sid = model.site("gripperframe").id
mujoco.mj_forward(model, data)
AXIS = data.xpos[model.body("shoulder").id][:2].copy()  # 肩の軸(sim の世界座標)

NAMES = ["dx [mm]", "dy [mm]", "dyaw [度]", "Δshoulder_lift [度]", "Δelbow_flex [度]", "Δwrist_flex [度]", "Δwrist_roll [度]"]
JOINT_IDX = {3: 1, 4: 2, 5: 3, 6: 4}  # パラメータの番号 → JOINTS の番号(lift = 1、elbow = 2、wrist_flex = 3、wrist_roll = 4)
obs_list = [{f"{k}.pos": v for k, v in s["joints"].items()} for s in samples]
P_img = np.array([s["hand_pos_m"] for s in samples])
R_img = [np.array(s["hand_R"]) for s in samples]
q_base = [real_to_sim(model, o) for o in obs_list]  # 今の OFFSET_DEG を足した、sim の関節角 [rad]


def sim_pose(q):
    for j, name in enumerate(JOINTS):
        data.qpos[model.joint(name).qposadr] = q[j]
    mujoco.mj_forward(model, data)
    return data.site_xpos[sid].copy(), data.site_xmat[sid].reshape(3, 3).copy()


def residuals(x):
    dx, dy, dyaw = x[0] * 1e-3, x[1] * 1e-3, np.radians(x[2])
    Rz = Rotation.from_euler("z", dyaw).as_matrix()  # z 軸まわり(鉛直)の回転
    out = []
    for i in range(len(samples)):
        q = q_base[i].copy()
        for k, j in JOINT_IDX.items():
            q[j] += np.radians(x[k])
        p_sim, R_sim = sim_pose(q)
        p_corr = np.r_[AXIS, 0.0] + Rz @ (P_img[i] - np.r_[AXIS, 0.0]) + np.array([dx, dy, 0.0])  # 画像の手先を、肩の軸まわりに dyaw 回し、(dx, dy) ずらす
        R_corr = Rz @ R_img[i]
        out.extend((p_corr - p_sim) * 1000 / args.sigma_pos)
        out.extend(np.degrees(Rotation.from_matrix(R_sim.T @ R_corr).as_rotvec()) / args.sigma_rot)  # 向きの差(回転ベクトル。度)
    return np.array(out)


r0 = residuals(np.zeros(7))
res = least_squares(residuals, np.zeros(7), loss="soft_l1")
J = res.jac
cov = np.linalg.pinv(J.T @ J)  # 推定値の共分散(重みを σ にしてあるので、そのまま使える)。pinv = 擬似逆行列(分けられない方向があっても計算できる)
std = np.sqrt(np.clip(np.diag(cov), 0, None))
sv = np.linalg.svd(J, compute_uv=False)
print(f"サンプル {len(samples)} 個。差の大きさ(二乗平均の平方根、σ 単位): 補正前 {np.sqrt((r0**2).mean()):.2f} → 補正後 {np.sqrt((res.fun**2).mean()):.2f}(1 前後なら、見積もりの範囲内)")
print(f"{'パラメータ':22s}{'推定値':>10s}{'±(1σ)':>9s}   今の OFFSET_DEG")
for k, name in enumerate(NAMES):
    extra = f"{OFFSET_DEG[JOINTS[JOINT_IDX[k]]]:+.1f}" if k in JOINT_IDX else ""
    flag = "  ← 分けられていない(腕の形を増やす)" if std[k] > (5.0 if k < 3 else 2.0) else ""
    unit_fix = res.x[k]
    print(f"{name:22s}{unit_fix:+10.2f}{std[k]:9.2f}   {extra}{flag}")
# サンプルごとの残差(補正前 → 補正後)。位置 [mm]、向き [度]。どのサンプルが、合っていないかを見る
print("\nサンプルごとの差(画像 − sim。補正前 → 補正後)")
print(f"{'名前':10s}{'位置[mm]':>18s}{'向き[度]':>16s}")
for i, smp in enumerate(samples):
    b = (r0[6 * i:6 * i + 6] * np.r_[[args.sigma_pos] * 3, [args.sigma_rot] * 3])
    a = (res.fun[6 * i:6 * i + 6] * np.r_[[args.sigma_pos] * 3, [args.sigma_rot] * 3])
    print(f"{smp['label']:10s}{np.linalg.norm(b[:3]):8.1f} →{np.linalg.norm(a[:3]):6.1f}{np.linalg.norm(b[3:]):9.1f} →{np.linalg.norm(a[3:]):6.1f}")
print(f"\n(ヤコビ行列の特異値の最小/最大 = {sv.min() / sv.max():.1e}。小さいほど、分けられない組み合わせがある)")
