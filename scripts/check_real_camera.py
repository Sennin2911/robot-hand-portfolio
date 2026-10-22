"""【概要】real_camera.py(実機のカメラの、歪みの補正と、K を使った画素 → 世界座標)が正しいかを確認する。
(1) 画素の往復: distort_pixels と undistort_points が、互いの逆になる。
(2) K 版の換算: project_points_K と pixel_to_world_plane_K が、互いの逆になる。cx, cy が真ん中の K なら、camera_pose.py の関数と同じ値になる。
    実機の K(cx, cy が真ん中からずれている)では、camera_pose.py の関数を使うと、何 mm ずれるかも表示する。
(3) 合成画像: 歪ませた画像の端に、赤い正方形を置き、undistort_image で直したあとの色検出が、本当の位置に戻る。
(4) 実機の画像: 撮ったチェスボードの画像(outputs/chessboard_shots/)を、直すと、盤の線が、まっすぐになる(画像が無ければ飛ばす)。
全部通れば終了コード0、そうでなければ1。

使い方(この .venv で実行):
    .venv/bin/python scripts/check_real_camera.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)。
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from hand_sim.camera_pose import look_at_rotation, pixel_to_world_plane  # noqa: E402
from hand_sim.intrinsics import PATTERN, find_corners  # noqa: E402
from hand_sim.real_camera import (  # noqa: E402
    distort_pixels, load_camera, pixel_to_world_plane_K, project_points_K, undistort_image, undistort_points,
)
from hand_sim.vision import find_cube_pose  # noqa: E402

n_ng = 0


def check(label, ok):
    """結果を 1 行表示し、失敗なら数える。global: 関数の外の変数 n_ng を書き換える宣言。[頻度] 小さなスクリプトでだけ使う"""
    global n_ng
    print(f"{'OK' if ok else 'NG'}  {label}")
    n_ng += 0 if ok else 1


cam = load_camera()
K, dist = cam["K"], cam["dist"]
W, H = cam["size"]

# ---- (1) 画素の往復 ----
rng = np.random.default_rng(0)  # 乱数の種を決める(毎回同じ点が出る)。[頻度] 確認用の乱数では定番
pts = np.stack([rng.uniform(0, W, 200), rng.uniform(0, H, 200)], axis=1)  # 画像全体に散らした、歪みのない理想の点
du, dv = distort_pixels(pts[:, 0], pts[:, 1], K, dist)
back = undistort_points(np.stack([du, dv], axis=1), K, dist)
err = np.abs(back - pts).max()
check(f"(1) 歪ませて、直すと、元に戻る(画像全体の 200 点で、最大のずれ {err:.4f} 画素 < 0.01)", err < 0.01)
moved = np.hypot(du - pts[:, 0], dv - pts[:, 1])
print(f"    (参考: 歪みで動いた量は、最大 {moved.max():.0f} 画素、中央値 {np.median(moved):.0f} 画素)")

# ---- (2) K 版の換算 ----
cam_pos = np.array([0.30, -0.10, 0.60])  # 机の斜め上 60 cm
cam_R = look_at_rotation(cam_pos, np.array([0.20, 0.0, 0.0]))  # 机の上の (0.2, 0) あたりを見る
table = np.array([[0.20, 0.0, 0.0], [0.15, 0.08, 0.0], [0.28, -0.07, 0.0], [0.22, 0.05, 0.03]])  # 机の上(と、立方体の上面 3 cm)の点
px = project_points_K(table, cam_pos, cam_R, K)
back_w = np.array([pixel_to_world_plane_K(u, v, cam_pos, cam_R, K, p[2]) for (u, v), p in zip(px, table)])
err = np.abs(back_w - table).max()
check(f"(2a) 世界 → 画素 → 世界で、元に戻る(最大のずれ {err * 1000:.4f} mm < 0.001)", err < 1e-6)
K_center = np.array([[1368.0, 0, W / 2], [0, 1368.0, H / 2], [0, 0, 1.0]])  # 中心が真ん中で、fx = fy の K(camera_pose.py の仮定どおり)
fovy_c = float(np.degrees(2 * np.arctan(H / (2 * K_center[1, 1]))))
px_c = project_points_K(table, cam_pos, cam_R, K_center)
w_old = np.array([pixel_to_world_plane(u, v, cam_pos, cam_R, fovy_c, W, H, p[2]) for (u, v), p in zip(px_c, table)])
w_new = np.array([pixel_to_world_plane_K(u, v, cam_pos, cam_R, K_center, p[2]) for (u, v), p in zip(px_c, table)])
err = np.abs(w_old - w_new).max()
check(f"(2b) 中心が真ん中の K では、camera_pose.py の関数と同じ値(差 {err * 1000:.4f} mm < 0.001)", err < 1e-6)
w_wrong = np.array([pixel_to_world_plane(u, v, cam_pos, cam_R, cam["fovy_deg"], W, H, p[2]) for (u, v), p in zip(px, table)])
gap = np.hypot(*(w_wrong - table)[:, :2].T) * 1000
print(f"    (参考: 実機の K で写った画素を、cx, cy が真ん中・fx = fy の仮定(camera_pose.py)で戻すと、机の上で {gap.min():.1f}〜{gap.max():.1f} mm ずれる。K 版を使う理由)")
check("(2c) その仮定のずれが、1 mm 以上ある(K 版が、必要なこと)", gap.max() >= 1.0)

# ---- (3) 合成画像: 歪ませた画像の端に、赤い正方形を置く ----
# 手順: 歪みのない画像 ideal を作り、赤い正方形を、本当の位置 (u0, v0) に置く。歪んだ画像は、歪んだ画素 (ud, vd) ごとに、
# 「その画素が、理想の画像のどこにあたるか」を cv2.undistortPoints で求め、cv2.remap で、その位置の色を取ってきて作る(remap: 画素ごとに、取ってくる位置を指定して、画像を作り直す)。
gx, gy = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
ideal_xy = undistort_points(np.stack([gx.ravel(), gy.ravel()], axis=1), K, dist).reshape(H, W, 2).astype(np.float32)
for name, (u0, v0) in {"画像の中心付近": (960, 540), "右の端": (1700, 300), "左下の隅に近い所": (260, 900)}.items():
    ideal = np.full((H, W, 3), 128, np.uint8)
    ideal[v0 - 20:v0 + 20, u0 - 20:u0 + 20] = (230, 20, 20)  # 40 × 40 画素の赤い正方形(RGB)
    distorted = cv2.remap(ideal, ideal_xy[:, :, 0], ideal_xy[:, :, 1], cv2.INTER_LINEAR)
    raw = find_cube_pose(distorted, "red")
    fixed = find_cube_pose(undistort_image(distorted, K, dist), "red")
    e_raw = np.hypot(raw[0] - (u0 - 0.5), raw[1] - (v0 - 0.5))  # 正方形の中心は、画素の添字 u0-20〜u0+19 の真ん中 = u0 - 0.5
    e_fix = np.hypot(fixed[0] - (u0 - 0.5), fixed[1] - (v0 - 0.5))
    print(f"    {name}: 歪んだままの検出は {e_raw:.1f} 画素ずれ、直してからの検出は {e_fix:.2f} 画素ずれ")
    check(f"(3) {name}: 直してから検出すると、本当の位置に戻る(ずれ {e_fix:.2f} 画素 < 1.0)", e_fix < 1.0)

# ---- (4) 実機の画像: 盤の線が、まっすぐになるか ----
shots = sorted((ROOT / "outputs" / "chessboard_shots").glob("shot_*.png"))
if not shots:
    print("    (4) 実機の画像(outputs/chessboard_shots/)が無いので、飛ばす")
else:
    def line_deviation(corners):
        """盤の各行(横に 9 個の角)に、直線を当てはめ、角が直線から外れる量の最大値 [画素] を返す。まっすぐなら 0 に近い。"""
        pts = corners.reshape(PATTERN[1], PATTERN[0], 2)  # (縦の行数, 横の個数, 2)
        worst = 0.0
        for row in pts:
            d = row - row.mean(axis=0)
            direction = np.linalg.svd(d)[2][0]  # 角の並びに、いちばん合う直線の向き(特異値分解の第 1 成分)。[頻度] 直線の当てはめの定番
            normal = np.array([-direction[1], direction[0]])
            worst = max(worst, float(np.abs(d @ normal).max()))
        return worst

    before, after = [], []
    for path in shots:
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        c0 = find_corners(img)
        c1 = find_corners(undistort_image(img, K, dist))
        if c0 is not None and c1 is not None:
            before.append(line_deviation(c0))
            after.append(line_deviation(c1))
    print(f"    {len(before)} 枚で比較: 盤の線の曲がり(最大)の中央値 = 直す前 {np.median(before):.2f} 画素 → 直した後 {np.median(after):.2f} 画素")
    check("(4) 直すと、盤の線の曲がりが、小さくなる(中央値)", np.median(after) < np.median(before))
    check(f"(4) 直した後の、盤の線の曲がりが、1 画素未満(中央値 {np.median(after):.2f})", np.median(after) < 1.0)

print(f"\n{'全部 OK' if n_ng == 0 else f'NG が {n_ng} 件'}")
sys.exit(1 if n_ng else 0)
