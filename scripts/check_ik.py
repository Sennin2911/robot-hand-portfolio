"""【概要】arm.py の ik が正しいかを確認する。
ik が返した関節角を MuJoCo のアームに設定し、手先(site "tip")が目標位置に来るかを調べる。
(解が2通りあるので「角度が同じか」ではなく「手先が目標に届くか」で判定する)
届かない目標では ValueError が出ることも確認する。全部通れば終了コード0、そうでなければ1。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)。
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import mujoco  # noqa: E402
from hand_sim import HandEnv  # noqa: E402
from hand_sim.arm import ik, L1, L2  # noqa: E402

env = HandEnv(ROOT / "models" / "arm2link.xml")
TOL = 1e-6  # 許容する誤差 [m]

# 届く目標 (x, y) [m]。最大の到達距離ぴったり (L1+L2) の点も含める。
# 到達距離は L1+L2=0.35 (最大) 〜 |L1-L2|=0.05 (最小)。最小ぎりぎりは q2=π になり
# モデルの関節範囲(±2.8)を超えるので、ここでは (0.1, 0) までに留める。
REACHABLE = [(0.2, 0.1), (0.3, 0.0), (0.0, 0.25), (-0.1, 0.2), (0.1, -0.2),
             (L1 + L2, 0.0), (0.1, 0.0)]
# 届かない目標: 遠すぎる点 (距離>0.35) / 近すぎる点 (距離<0.05)。
UNREACHABLE = [(0.4, 0.0), (0.0, 0.36), (0.0, 0.0), (0.03, 0.0)]

n_ng = 0
try:
    for x, y in REACHABLE:
        for elbow in (+1, -1):
            q1, q2 = ik(x, y, elbow)
            env.data.qpos[:] = [q1, q2]
            mujoco.mj_forward(env.model, env.data)
            tip = env.data.site("tip").xpos[:2]
            err = np.linalg.norm(tip - np.array([x, y]))
            # elbow=+1 なら q2>=0、-1 なら q2<=0 になっているか(わずかな誤差は許す)。
            side_ok = elbow * q2 >= -1e-9
            ok = err < TOL and side_ok
            n_ng += 0 if ok else 1
            print(f"{'OK' if ok else 'NG'}  目標=({x:+.3f}, {y:+.3f}) elbow={elbow:+d}  "
                  f"q=({q1:+.3f}, {q2:+.3f})  誤差={err:.2e} m  肘の向き{'OK' if side_ok else 'NG'}")

    for x, y in UNREACHABLE:
        try:
            ik(x, y)
        except ValueError:
            print(f"OK  目標=({x:+.3f}, {y:+.3f}) は届かないので ValueError")
        else:
            n_ng += 1
            print(f"NG  目標=({x:+.3f}, {y:+.3f}) は届かないのに例外が出なかった")
except NotImplementedError as e:
    print(f"{e}. src/hand_sim/arm.py に書いてください。")
    sys.exit(1)

env.close()
print("全部OK" if n_ng == 0 else f"{n_ng} 件NGです")
sys.exit(0 if n_ng == 0 else 1)
