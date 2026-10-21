"""【概要】camera_pose.py(任意の位置・向きのカメラの、投影・較正)が正しいかを、画像なしの合成データで確認する。
(1) look_at_rotation が、正しい向きの回転行列(直交で、見つめる点が画像の中心に写る)を作る。
(2) 画素 → 世界座標(pixel_to_world_plane)が、project_points の逆になる(往復で元に戻る)。
(3) 本当のカメラで「目印の画素」を合成し、目で見たようにずれた出発点から、較正が本当の位置・向き・視野角を取り戻す。
全部通れば終了コード0、そうでなければ1。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)。
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from hand_sim.camera_pose import (  # noqa: E402
    calibrate_camera_pose, look_at_rotation, pixel_to_world_plane, project_points, rotation_angle_deg,
)
from scipy.spatial.transform import Rotation  # noqa: E402

W, H = 640, 480
n_ng = 0


def check(label, ok):
    """結果を 1 行表示し、失敗なら数える。global: 関数の外の変数 n_ng を書き換える宣言。[頻度] 小さなスクリプトでだけ使う"""
    global n_ng
    print(f"{'OK' if ok else 'NG'}  {label}")
    n_ng += 0 if ok else 1


# (1) look_at_rotation
pos, target = np.array([0.55, -0.04, 0.45]), np.array([0.20, -0.04, 0.02])
R = look_at_rotation(pos, target)
check("回転行列が直交している(R^T R = I、行列式 +1)", np.allclose(R.T @ R, np.eye(3)) and np.isclose(np.linalg.det(R), 1.0))
center = project_points(target[None, :], pos, R, 45.0, W, H)[0]
check(f"見つめる点が、画像の中心 (320, 240) に写る(実際: {center.round(2)})", np.allclose(center, [W / 2, H / 2], atol=1e-6))

# (2) 往復
pts = np.array([[0.14, -0.14, 0.03], [0.26, 0.06, 0.03], [0.20, -0.04, 0.015]])
pix = project_points(pts, pos, R, 45.0, W, H)
back = np.array([pixel_to_world_plane(u, v, pos, R, 45.0, W, H, p[2]) for (u, v), p in zip(pix, pts)])
check(f"画素 → 世界座標が、投影の逆になる(最大の差 {np.abs(back - pts).max():.1e} m)", np.allclose(back, pts, atol=1e-9))

# (3) 較正が、本当のカメラを取り戻す。目印の世界座標は、作業範囲の上空に 3 通りの高さで 4 × 4 の格子状に置く(合成)。
rng = np.random.default_rng(0)
markers = np.array([[x, y, z] for x in (0.12, 0.18, 0.24, 0.30) for y in (-0.14, -0.05, 0.04, 0.10) for z in (0.07, 0.10, 0.13)])
for name, true_pos, up in [("斜め前から", (0.55, -0.04, 0.45), (0, 0, 1)), ("真上から", (0.20, 0.0, 0.60), (0, 1, 0)), ("左から", (0.2, 0.5, 0.3), (0, 0, 1))]:
    true_R = look_at_rotation(true_pos, target, up)
    true_fov = 50.0
    pixels = project_points(markers, true_pos, true_R, true_fov, W, H)
    ok_all = True
    for _ in range(5):  # 出発点を 5 通り(位置 10 cm、向き 10°、視野角 ±8° ずらす)
        d = rng.normal(size=3)
        ip = np.array(true_pos) + d / np.linalg.norm(d) * 0.10
        axis = rng.normal(size=3)
        iR = (Rotation.from_rotvec(axis / np.linalg.norm(axis) * np.radians(10)) * Rotation.from_matrix(true_R)).as_matrix()
        pos_e, R_e, fov_e, rms = calibrate_camera_pose(markers.copy(), pixels, W, H, ip, iR, true_fov + rng.uniform(-8, 8))
        ok_all &= bool(np.linalg.norm(pos_e - true_pos) < 1e-3 and rotation_angle_deg(R_e, true_R) < 0.01 and abs(fov_e - true_fov) < 0.01 and rms < 1e-3)
    check(f"較正が本当の位置・向き・視野角を取り戻す({name}、出発点 5 通り、ノイズなし)", ok_all)

print("全部OK" if n_ng == 0 else f"NG が {n_ng} 件")
sys.exit(0 if n_ng == 0 else 1)
