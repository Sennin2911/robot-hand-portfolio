"""【概要】sim_real.py の変換が正しいかを確認する。
(1) 往復(sim → 実機 → sim)で元に戻る。 (2) 0 rad が「0 度 − オフセット」、gripper の両端が 0 と 100 になる。
(3) 符号が SIGN どおり(+ にすると実機の値も + / - なら -)になる。 (4) sim の可動域の外は端に切り詰められる。
(5) 実機の 0 を sim に直すとオフセットの角度になり、apply_offset=False ならオフセットなしの変換になる。
全部通れば終了コード0、そうでなければ1。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)。
sys.path.insert(0, str(ROOT / "src"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from hand_sim.sim_real import JOINTS, OFFSET_DEG, SIGN, real_to_sim, sim_to_real  # noqa: E402

model = mujoco.MjModel.from_xml_path(str(ROOT / "models" / "so101" / "scene.xml"))
n_ng = 0


def check(label, ok):
    """結果を1行表示し、失敗なら数える。global: 関数の外の変数 n_ng を書き換える宣言。[頻度] 小さなスクリプトでだけ使う"""
    global n_ng
    print(f"{'OK' if ok else 'NG'}  {label}")
    n_ng += 0 if ok else 1


# (1) 往復
rng = np.random.default_rng(0)  # 乱数の種を固定 → 毎回同じ値(再現性)
worst = 0.0
for _ in range(50):
    lo, hi = model.jnt_range[:6, 0], model.jnt_range[:6, 1]
    q = rng.uniform(lo, hi)  # 各関節の可動域の中の乱数
    back = real_to_sim(model, sim_to_real(model, q))
    worst = max(worst, float(np.abs(back - q).max()))
check(f"往復で元に戻る(最大のずれ {worst:.2e} rad)", worst < 1e-9)

# (2) 原点と gripper の両端
zero = sim_to_real(model, np.zeros(6))
check("腕の5関節は 0 rad → -オフセット [度]", all(abs(zero[f"{n}.pos"] + OFFSET_DEG[n]) < 1e-9 for n in JOINTS[:5]))
zero_raw = sim_to_real(model, np.zeros(6), apply_offset=False)
check("apply_offset=False なら 0 rad → 0 度", all(abs(zero_raw[f"{n}.pos"]) < 1e-9 for n in JOINTS[:5]))
glo, ghi = model.joint("gripper").range
q = np.zeros(6)
q[5] = glo
check("gripper の下限 → 0", abs(sim_to_real(model, q)["gripper.pos"] - 0.0) < 1e-9)
q[5] = ghi
check("gripper の上限 → 100", abs(sim_to_real(model, q)["gripper.pos"] - 100.0) < 1e-9)

# (3) 符号と単位(0.1 rad = 5.73 度)
for i, name in enumerate(JOINTS[:5]):
    q = np.zeros(6)
    q[i] = 0.1
    got = sim_to_real(model, q)[f"{name}.pos"]
    expected = SIGN[name] * (np.degrees(0.1) - OFFSET_DEG[name])
    check(f"{name}: +0.1 rad → {got:+.2f} 度(符号 {SIGN[name]:+d}、オフセット {OFFSET_DEG[name]:+.1f})", abs(got - expected) < 1e-9)

# (4) 範囲外は端に切り詰める
q = np.zeros(6)
q[0] = 10.0  # shoulder_pan の上限 (約1.92 rad) をはるかに超える
# 切り詰めは、オフセットを引く前の sim の角度に対して行う(切り詰めた上限 − オフセット が実機の値になる)
check("範囲外は sim の上限に切り詰める", abs(sim_to_real(model, q)["shoulder_pan.pos"] - (np.degrees(model.joint("shoulder_pan").range[1]) - OFFSET_DEG["shoulder_pan"])) < 1e-9)

# (5) 実機の 0 → sim は、オフセットの角度(符号が +1 の関節で)。apply_offset=False なら 0 rad
real_zero = {f"{n}.pos": 0.0 for n in JOINTS}
q_off, q_raw = real_to_sim(model, real_zero), real_to_sim(model, real_zero, apply_offset=False)
check("実機の 0 → sim は +オフセット [度]", all(abs(np.degrees(q_off[i]) - OFFSET_DEG[n]) < 1e-9 for i, n in enumerate(JOINTS[:5])))
check("apply_offset=False なら、実機の 0 → sim は 0 rad", np.abs(q_raw[:5]).max() < 1e-12)

print("全部OK" if n_ng == 0 else f"NG が {n_ng} 件")
sys.exit(0 if n_ng == 0 else 1)  # sys.exit(0): 成功の終了コード。[頻度] チェック用スクリプトでは定番
