"""【概要】so101_grasp.py の「立方体を掴んで、運んで、置く」動作が、sim で正しく動くかを確認する。
(1) 立方体の位置(机の上の数か所)と向き(0〜135°)をいろいろ変えて、掴んで運んで置けること。
    成功の基準: 持ち上げで高さ 5 cm 以上 / 運搬の終わりで置き場の 2 cm 以内 / 置いたあとで置き場の 1 cm 以内、かつ机の上(高さ 2 cm 未満)。
(2) 降りるとき(指を開いたまま)に、指が立方体に当たって動かしていないこと(動いた量が 3 mm 未満)。
(3) 腕が届かない位置では、IK が解けない例外(ValueError)になること。
(4) 認識の誤差を想定して、「立方体の位置の見積もり」を本当の位置から 10 mm ずらしても(4 方向)、掴んで置けること。
全部通れば終了コード0、そうでなければ1。
"""
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)。
sys.path.insert(0, str(ROOT / "src"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from hand_sim.so101_grasp import load_grasp_model, pick_and_place, set_cube_pose  # noqa: E402

model = load_grasp_model(ROOT / "models" / "so101" / "scene_vs.xml")
PLACE = np.array([0.18, 0.16])  # 置き場(立方体の中心の水平位置 [m])
Z = 0.015  # 立方体の中心の高さ [m](机の上に置いたとき)
n_ng = 0


def run(xy, yaw_deg):
    """立方体を (xy, 向き yaw_deg) に置いて、掴んで運んで置く。戻り値: 動作の記録の辞書。"""
    data = mujoco.MjData(model)
    set_cube_pose(model, data, [*xy, Z], np.radians(yaw_deg))
    return pick_and_place(model, data, [*xy, Z], PLACE, np.radians(yaw_deg))


print("--- (1)(2) 位置と向きを変えて、掴んで運んで置く ---")
n_try = n_ok = 0
for x, y, yaw in itertools.product((0.15, 0.22), (-0.10, 0.0, 0.08), (0, 20, 45, 70, 90, 135)):
    log = run((x, y), yaw)
    lifted = log["lifted"][2] > 0.05
    carried = np.linalg.norm(log["carried"][:2] - PLACE) < 0.02 and log["carried"][2] > 0.05
    err = np.linalg.norm(log["placed"][:2] - PLACE)
    placed = err < 0.01 and log["placed"][2] < 0.02
    undisturbed = np.linalg.norm(log["before_close"][:2] - [x, y]) < 0.003
    good = lifted and carried and placed and undisturbed
    n_try += 1
    n_ok += good
    n_ng += not good
    print(f"{'OK' if good else 'NG'}  立方体 ({x:.2f}, {y:+.2f}) 向き {yaw:3d}°: 持ち上げ={lifted} 運搬={carried} 置き誤差={err * 1000:.1f}mm 降下で動かさない={undisturbed}")
print(f"({n_ok} / {n_try} 件が成功)")

print("--- (3) 届かない位置 ---")
for xy in [(0.45, 0.0), (0.02, 0.0)]:
    try:
        run(xy, 0)
        good = False
    except ValueError as e:
        good = True
    n_ng += not good
    print(f"{'OK' if good else 'NG'}  立方体 {xy}: IK が解けない例外になる(ValueError)のが正しい")

print("--- (4) 位置の見積もりが 10 mm ずれていても掴める ---")
TOL_MM = 10
for xy, ang in itertools.product([(0.15, -0.10), (0.20, 0.0), (0.22, 0.05)], (0, 90, 180, 270)):
    data = mujoco.MjData(model)
    set_cube_pose(model, data, [*xy, Z], 0.0)
    est = np.array(xy) + TOL_MM / 1000 * np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))])  # ずらした見積もり(ang は、ずらす向き [度])
    log = pick_and_place(model, data, [*est, Z], PLACE, 0.0)
    good = log["lifted"][2] > 0.05 and np.linalg.norm(log["placed"][:2] - PLACE) < 0.02
    n_ng += not good
    print(f"{'OK' if good else 'NG'}  立方体 {xy}、見積もりを {ang:3d}° の向きへ {TOL_MM} mm ずらす: 持ち上げ={log['lifted'][2] > 0.05} 置き誤差={np.linalg.norm(log['placed'][:2] - PLACE) * 1000:.1f}mm")

print("全部OK" if n_ng == 0 else f"NG が {n_ng} 件")
sys.exit(0 if n_ng == 0 else 1)
