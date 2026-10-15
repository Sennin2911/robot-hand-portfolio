"""【概要】so101_ik.solve_ik が正しいかを確認する。
(1) 関節角をランダムに選び、順運動学で手先位置を求め、それを目標にして IK を解く
    → 「必ず届く目標」なので、手先が 1mm 以内に来て、関節が可動域内であるべき。
(2) 明らかに届かない目標では ok=False が返ること。
全部通れば終了コード0、そうでなければ1。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)。
sys.path.insert(0, str(ROOT / "src"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from hand_sim.so101_ik import N_ARM, solve_ik  # noqa: E402

model = mujoco.MjModel.from_xml_path(str(ROOT / "models" / "so101" / "scene.xml"))
data = mujoco.MjData(model)
site_id = model.site("gripperframe").id
lo, hi = model.jnt_range[:N_ARM, 0], model.jnt_range[:N_ARM, 1]
TOL = 1e-3  # 許容する誤差 [m] (1mm)
HOME = np.array([0, -1.7, 1.6, 1.0, 0])  # 出発点: たたんだ姿勢

rng = np.random.default_rng(0)  # 乱数の種を固定 → 毎回同じ目標になる(再現性)
n_ng = 0
n_tried = 0
print("--- (1) 届く目標 ---")
for i in range(40):
    # 可動域の中央付近 (中央から ±60%) で選ぶ。端ぎりぎりは解が1つしかなく難しいため。
    q_true = (lo + hi) / 2 + rng.uniform(-0.6, 0.6, N_ARM) * (hi - lo) / 2
    data.qpos[:N_ARM] = q_true
    mujoco.mj_forward(model, data)
    target = data.site_xpos[site_id].copy()  # .copy(): この後 data が書き換わるので控えを取る
    # 床より下 / 土台のすぐ近くは、実用外かつ解きにくい(特異姿勢)ので試さない。
    if target[2] < 0.02 or np.linalg.norm(target[:2]) < 0.1:
        continue
    n_tried += 1

    q, ok = solve_ik(model, data, target, HOME, tol=TOL)
    data.qpos[:N_ARM] = q
    mujoco.mj_forward(model, data)
    err = np.linalg.norm(data.site_xpos[site_id] - target)
    in_range = bool(np.all(q >= lo - 1e-9) and np.all(q <= hi + 1e-9))
    good = ok and err < TOL and in_range
    n_ng += not good
    print(f"{'OK' if good else 'NG'}  target={np.round(target, 3)} 誤差={err * 1000:.2f}mm ok={ok} 可動域内={in_range}")

print(f"({n_tried} 件を試した)")
print("--- (2) 届かない目標 ---")
for target in [(0.8, 0.0, 0.1), (0.0, 0.0, 0.9)]:
    q, ok = solve_ik(model, data, target, HOME)
    good = not ok
    n_ng += not good
    print(f"{'OK' if good else 'NG'}  target={target} ok={ok} (False が正しい)")

print("全部OK" if n_ng == 0 else f"NG が {n_ng} 件")
sys.exit(0 if n_ng == 0 else 1)
