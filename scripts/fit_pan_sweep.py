"""【概要】pan だけを変えた複数の姿勢(record_hand_sample.py で記録)から、肩の軸の位置と、板の座標の向きのずれを、測定に頼らずに求める。
考え方: 肩以外の関節を固定して pan だけを回すと、手先は、肩の軸のまわりの水平な円の上を動く。画像から出した手先の (x, y) に円を当てはめれば、
  (1) 円の中心 = 肩の軸の位置(板の座標から見た値)。想定の原点 (0, 0) とのずれが、板の位置の測定の誤差。
  (2) 円の上の角度の変化と、pan の読みの変化を比べれば、板の向き(yaw)の誤りも分かる(pan + は右回り = 世界で見て時計回りなので、傾き −1 になるはず)。
  (3) 円の半径が、sim の手先の半径と合うか(腕の形の確認)。
肩の軸の位置は、関節のずれ(OFFSET_DEG)に依存しない(円の中心は、半径が違っても変わらない)。3 個以上の姿勢が要る。角度の幅が広いほど精度が良い(60° 以上が目安)。

使い方(この .venv で実行):
    .venv/bin/python scripts/fit_pan_sweep.py                 # outputs/hand_samples.json の、pan が違う姿勢を全部使う
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
from hand_sim.sim_real import JOINTS, real_to_sim  # noqa: E402

model = mujoco.MjModel.from_xml_path(str(ROOT / "models/so101/so101_new_calib.xml"))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
# 想定する肩の軸の位置: sim の shoulder の位置(世界座標の原点ではなく、約 38.8 mm 前。fixed_board.py の注を参照)。画像の世界座標も、sim と同じ座標にしてある。
AXIS = data.xpos[model.body("shoulder").id][:2].copy()
samples = json.loads((ROOT / "outputs" / "hand_samples.json").read_text())
if len(samples) < 3:
    sys.exit(f"姿勢が {len(samples)} 個しかない。3 個以上(pan を変えて)record_hand_sample.py で記録する")
P = np.array([s["hand_pos_m"][:2] for s in samples])  # 手先の水平位置 (N, 2) [m]
pan = np.array([s["joints"]["shoulder_pan"] for s in samples])  # pan の読み [度]
print("姿勢: " + ", ".join(f"{s['label']}(pan {p:.1f}°)" for s, p in zip(samples, pan)))
print(f"pan の幅: {pan.max() - pan.min():.1f}°(60° 以上が目安)")

# 円の当てはめ(Kåsa 法): 円の式 x² + y² = 2·a·x + 2·b·y + c を、最小二乗法で解く。(a, b) = 中心、半径 = √(c + a² + b²)。
# [頻度] 点から円を求めるときの定番。np.linalg.lstsq = 最小二乗法で連立方程式を解く関数。
A = np.column_stack([2 * P[:, 0], 2 * P[:, 1], np.ones(len(P))])
(a, b, c), *_ = np.linalg.lstsq(A, (P**2).sum(axis=1), rcond=None)
center = np.array([a, b])
radius = np.sqrt(c + a * a + b * b)
resid = np.linalg.norm(P - center, axis=1) - radius  # 各点の、円からのずれ(半径方向) [m]
print(f"\n(1) 円の中心(= 肩の軸) = ({a * 1000:+.1f}, {b * 1000:+.1f}) mm。想定(sim の肩の軸)は ({AXIS[0] * 1000:+.1f}, {AXIS[1] * 1000:+.1f})")
print(f"    → 板の位置の測定の誤差は、およそ x {(AXIS[0] - a) * 1000:+.1f} mm、y {(AXIS[1] - b) * 1000:+.1f} mm(十字の実際の位置 = 想定 + この値)")
print(f"    円からのずれ(RMS) = {np.sqrt((resid**2).mean()) * 1000:.1f} mm(小さいほど、円に乗っている)")

# (2) 円の中心から見た、各点の角度 [度](世界の x 軸から左回り)。pan との直線関係を調べる。
ang = np.degrees(np.arctan2(P[:, 1] - b, P[:, 0] - a))
ang = np.degrees(np.unwrap(np.radians(ang)))  # unwrap: 180° をまたいだ角度の飛びをつなぐ
slope, intercept = np.polyfit(pan, ang, 1)  # np.polyfit(x, y, 1) = 直線 y = 傾き·x + 切片 を当てはめる。[頻度] 直線の当てはめの定番
print(f"(2) 角度と pan の関係: 角度 = {slope:+.3f} × pan + {intercept:+.1f}°(傾きは −1 のはず)")
print(f"    pan = 0 のとき、腕は、世界の x 軸から {intercept:+.1f}° の向き(OFFSET_DEG の pan = +4.4° なら、−4.4° のはず)")

# (3) 半径: sim で、同じ関節角から出した手先の、肩の軸からの水平距離と比べる(OFFSET あり)
site_id = model.site("gripperframe").id
sim_r = []
for s in samples:
    q = real_to_sim(model, {f"{k}.pos": v for k, v in s["joints"].items()})
    for j, name in enumerate(JOINTS):
        data.qpos[model.joint(name).qposadr] = q[j]
    mujoco.mj_forward(model, data)
    sim_r.append(np.linalg.norm(data.site_xpos[site_id][:2] - AXIS))  # 肩の軸から、手先までの水平距離
print(f"(3) 半径: 画像 {radius * 1000:.1f} mm、sim {np.mean(sim_r) * 1000:.1f} mm(OFFSET あり。姿勢ごとの差 {np.ptp(sim_r) * 1000:.1f} mm)")
