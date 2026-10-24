"""【概要】hand_marker.py(板の付き方と、マーカー → 手先の変換)が正しいかを確認する。
 (1) 向きの行列が正しい回転(右手系、長さ 1、直交)  (2) 作者の測定どおり(垂線 = ②の +y、紙の上 = ②の -z)
 (3) sim のいろいろな関節角で、手先 → マーカー → 手先 と往復して元に戻る
 (4) sim のモデルの上で、マーカーの中心が固定側の部品のすぐ近く(板の取り付け位置として自然)で、面が +y 側(外)を向いている
全部通れば終了コード0、そうでなければ1。(4) は目安の確認(板そのものは sim に無い)。

使い方(この .venv で実行):
    MUJOCO_GL=egl .venv/bin/python scripts/check_hand_marker.py
"""
import os
import sys
from pathlib import Path

# mujoco を import する「前」に設定する必要がある(import の時点で描画方式が決まるため)。ここでは描画しないが、他のスクリプトと揃えている。
os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
# src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)。
sys.path.insert(0, str(ROOT / "src"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from hand_sim.hand_marker import (  # noqa: E402
    R_HAND_MARKER, T_HAND_MARKER, hand_pose_from_marker, marker_pose_from_hand,
)

n_ng = 0


def check(label, ok):
    """結果を 1 行表示し、失敗なら数える。global: 関数の外の変数 n_ng を書き換える宣言。[頻度] 小さなスクリプトでだけ使う"""
    global n_ng
    print(f"{'OK' if ok else 'NG'}  {label}")
    n_ng += 0 if ok else 1


R = R_HAND_MARKER
check("(1) 回転行列: R^T R = 単位行列、行列式 = +1(右手系)",
      np.allclose(R.T @ R, np.eye(3)) and np.isclose(np.linalg.det(R), 1.0))
check("(2) マーカーの垂線(3 列目)= ②の +y", np.allclose(R[:, 2], [0, 1, 0]))
check("(2) 紙の上(2 列目)= ②の -z", np.allclose(R[:, 1], [0, 0, -1]))
check("(2) 紙の右(1 列目)= ②の +x(右手系から決まる)", np.allclose(R[:, 0], [1, 0, 0]))
check("(2) 位置 = (-94.43, -20.5, -48.01) mm", np.allclose(T_HAND_MARKER * 1e3, [-94.43, -20.5, -48.01]))

model = mujoco.MjModel.from_xml_path(str(ROOT / "models/so101/so101_new_calib.xml"))
data = mujoco.MjData(model)
site_id = model.site("gripperframe").id
rng = np.random.default_rng(0)  # 乱数の種を固定(毎回同じ結果にするため)
worst_p, worst_r = 0.0, 0.0
for _ in range(50):
    for j in range(5):  # 腕の 5 関節を、可動範囲の中でランダムに動かす(6 番目はグリッパー)
        lo, hi = model.jnt_range[j]
        data.qpos[model.jnt_qposadr[j]] = rng.uniform(lo, hi)
    mujoco.mj_forward(model, data)  # 関節角から、各部の位置を計算し直す
    R_h = data.site_xmat[site_id].reshape(3, 3).copy()
    p_h = data.site_xpos[site_id].copy()
    R_m, p_m = marker_pose_from_hand(R_h, p_h)
    R_h2, p_h2 = hand_pose_from_marker(R_m, p_m)
    worst_p = max(worst_p, np.linalg.norm(p_h2 - p_h))
    worst_r = max(worst_r, np.abs(R_h2 - R_h).max())
check(f"(3) 往復で元に戻る(50 姿勢。位置の誤差の最大 {worst_p:.1e} m、向きの誤差の最大 {worst_r:.1e})", worst_p < 1e-9 and worst_r < 1e-9)

# (4) 基準の姿勢(全関節 0)で、②の座標のメッシュの頂点を集め、マーカーの中心との位置関係を見る
mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)
R_h = data.site_xmat[site_id].reshape(3, 3)
p_h = data.site_xpos[site_id]
gripper_body = model.body("gripper").id
pts = []
for g in range(model.ngeom):
    if model.geom_bodyid[g] != gripper_body or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or model.geom_group[g] != 2:
        continue  # 固定側の部品の、当たり判定のメッシュ(group 2)だけを使う
    mesh = model.geom_dataid[g]
    v = model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
    world = v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]  # メッシュの頂点を世界座標に直す
    pts.append((world - p_h) @ R_h)  # さらに ②(手の基準)の座標に直す
pts = np.vstack(pts)
c = T_HAND_MARKER
dist = np.linalg.norm(pts - c, axis=1).min()
print(f"      マーカー中心から、固定側の部品の頂点までの最短距離: {dist*1e3:.1f} mm(②の座標で)")
check("(4) マーカー中心が、固定側の部品から 30 mm 以内(板の取り付け位置として自然)", dist < 0.030)
print("      参考: 固定側の部品の範囲 [mm]  最小", (pts.min(0) * 1e3).round(0), " 最大", (pts.max(0) * 1e3).round(0))

print("\n全部 OK" if n_ng == 0 else f"\nNG が {n_ng} 件")
sys.exit(0 if n_ng == 0 else 1)
