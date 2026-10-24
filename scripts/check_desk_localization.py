"""【概要】画像から「机の上の位置」を出す精度を、板のマーカーで見積もる(立方体の位置の精度の見積もり)。
やり方(1 個外して予測する検証): 板のマーカー(ID 1〜6)のうち、いくつかを外し、残りでカメラの位置・向きを求める。外したマーカーの中心の画素から、机の面(高さ 0)の世界座標を出し、
設計値(印刷した配置)と比べる。カメラの位置の誤差、レンズの補正の誤差、画素の検出の誤差が、まとめて入った、実際の「立方体の位置の出し方」と同じ手順の誤差になる。
(板の置き方そのものの誤差は、外したものと残りで共通なので、入らない。入るのは、画像から位置を出す手順の誤差。)
外す個数を 1〜2 個にして、すべての組み合わせを試し、複数枚の画像(露出を変えず、時間をおいて撮る)で平均する。

使い方(この .venv で実行。カメラが要る。板が、手や腕で隠れていない状態で):
    .venv/bin/python scripts/check_desk_localization.py [--frames 5]
"""
import argparse
import itertools
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)

import numpy as np  # noqa: E402
from hand_sim.aruco_marker import detect_markers  # noqa: E402
from hand_sim.fixed_board import SIDE_MM, board_marker_centers_world, camera_pose_from_board  # noqa: E402
from hand_sim.real_camera import load_camera, pixel_to_world_plane_K, undistort_points  # noqa: E402
from hand_sim.webcam import grab_until, list_cameras  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--frames", type=int, default=5, help="撮る枚数")
args = parser.parse_args()

cam = load_camera()
K, dist = cam["K"], cam["dist"]
design = board_marker_centers_world((0.1108, 0.0), -np.pi / 2, 0.0)  # マーカー中心の設計値(世界座標、m)。fixed_board.BOARD_XY_M と同じ置き方
device = next((devs[0] for name, devs in list_cameras() if "WebCamera" in name), None)
if device is None:
    sys.exit("ウェブカメラが見つからない")


def diagonal_center(c):
    """四角形の 4 つの角(左上、右上、右下、左下の順)の、対角線の交点。透視で歪んだ正方形でも、実際の中心が写る点になる(4 つの角の平均は、ずれる)。"""
    p1, p2, p3, p4 = c
    # p1 + s(p3 − p1) = p2 + t(p4 − p2) を、s, t について解く(2 本の対角線の交点)。np.linalg.solve(A, b) = 連立方程式 A x = b の解
    A = np.array([p3 - p1, -(p4 - p2)]).T
    s, _ = np.linalg.solve(A, p2 - p1)
    return p1 + s * (p3 - p1)


errs_1, errs_2, per_marker = [], [], {i: [] for i in range(1, 7)}
errs_ex = {"手前の列(1, 3, 5)で出して、奥の列(2, 4, 6)を予測": [], "奥の列(2, 4, 6)で出して、手前の列(1, 3, 5)を予測": []}  # 外挿の検証: 一部のマーカーだけでカメラの位置を出し、離れたマーカーを予測する(板の外の立方体に近い条件)
EXTRA = [("手前の列(1, 3, 5)で出して、奥の列(2, 4, 6)を予測", (1, 3, 5), (2, 4, 6)), ("奥の列(2, 4, 6)で出して、手前の列(1, 3, 5)を予測", (2, 4, 6), (1, 3, 5))]
for k in range(args.frames):
    img, _ = grab_until(device, lambda im: len([i for i in detect_markers(im, K, dist, SIDE_MM / 1000) if 1 <= i <= 6]) == 6, settle_frames=15, timeout=8)
    found = detect_markers(img, K, dist, SIDE_MM / 1000)
    ids = [i for i in range(1, 7) if i in found]
    if len(ids) < 6:
        print(f"{k + 1} 枚目: 板のマーカーが {len(ids)} 個しか読めない(飛ばす)。腕や手が隠していないか確認する")
        continue
    for n_out, bucket in ((1, errs_1), (2, errs_2)):
        for out in itertools.combinations(ids, n_out):
            rest = {i: found[i] for i in ids if i not in out}
            pose = camera_pose_from_board(rest, K, dist, SIDE_MM)
            cam_pos, cam_R = pose[0], pose[1]
            for i in out:
                u, v = undistort_points(diagonal_center(found[i]["corners"]), K, dist)  # 歪んだ画素 → 歪みのない画素
                p = pixel_to_world_plane_K(u, v, cam_pos, cam_R, K, 0.0)  # 机の面(紙の表面、高さ 0)との交点
                e = (p - design[i]) * 1000  # 画像から出した位置 − 設計値 [mm]
                bucket.append(np.linalg.norm(e[:2]))
                if n_out == 1:
                    per_marker[i].append(e[:2])
    for name, use, test in EXTRA:
        pose = camera_pose_from_board({i: found[i] for i in use}, K, dist, SIDE_MM)
        for i in test:
            u, v = undistort_points(diagonal_center(found[i]["corners"]), K, dist)
            p = pixel_to_world_plane_K(u, v, pose[0], pose[1], K, 0.0)
            errs_ex[name].append(np.linalg.norm(((p - design[i]) * 1000)[:2]))
    time.sleep(1.0)

if not errs_1:
    sys.exit("使える画像が無かった")
a, b = np.array(errs_1), np.array(errs_2)
print(f"画像 {args.frames} 枚で、外した 1 個の位置を、残りの 5 個で出した誤差(机の面。水平): 平均 {a.mean():.2f} mm、最大 {a.max():.2f} mm(のべ {len(a)} 回)")
print(f"外した 2 個の位置を、残りの 4 個で出した誤差:                                  平均 {b.mean():.2f} mm、最大 {b.max():.2f} mm(のべ {len(b)} 回)")
print("\nマーカーごとの誤差(1 個外したとき。x, y の平均 [mm] と、ばらつき(標準偏差の大きさ)):")
for i, lst in per_marker.items():
    if lst:
        arr = np.array(lst)
        print(f"  ID {i}: 平均 ({arr[:, 0].mean():+.2f}, {arr[:, 1].mean():+.2f})、ばらつき {np.linalg.norm(arr.std(axis=0)):.2f} mm")
print("\n外挿の検証(一部の列だけでカメラの位置を出して、離れた列を予測。板の外の立方体に近い条件):")
for name, lst in errs_ex.items():
    if lst:
        print(f"  {name}: 平均 {np.mean(lst):.2f} mm、最大 {np.max(lst):.2f} mm")
print("\n注意: 立方体の中心は、上面(高さ 25〜30 mm)の中心から、高さの半分を引いて出す。高さの見積もりが 3 mm 違うと、視線が約 45° なので、水平の位置が約 3 mm ずれる。")
