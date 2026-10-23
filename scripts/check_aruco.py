"""【概要】aruco_marker.py(ArUco マーカーの作成と検出)が正しいかを、合成画像で確認する。
手順: 世界座標の既知の位置・向きにマーカーを置き、実機のカメラ(webcam_intrinsics.json の K と歪み)で撮った画像を作る
(理想の画像 → 実機のレンズの歪みを加える)。それを detect_markers に渡し、返ってきた位置・向きが、置いた値と合うかを見る。
 (1) ID が読める  (2) 位置の誤差 [mm]  (3) 向きの誤差 [度]  (4) 画素ノイズを足しても読める  (5) 印刷用の画像が、そのまま読める
全部通れば終了コード0、そうでなければ1。

使い方(この .venv で実行):
    .venv/bin/python scripts/check_aruco.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)。
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from hand_sim.aruco_marker import (  # noqa: E402
    CV_TO_CAM, detect_markers, make_marker_image, marker_object_points, marker_pose_in_world,
)
from hand_sim.camera_pose import look_at_rotation  # noqa: E402
from hand_sim.real_camera import load_camera, undistort_points  # noqa: E402

n_ng = 0


def check(label, ok):
    """結果を 1 行表示し、失敗なら数える。global: 関数の外の変数 n_ng を書き換える宣言。[頻度] 小さなスクリプトでだけ使う"""
    global n_ng
    print(f"{'OK' if ok else 'NG'}  {label}")
    n_ng += 0 if ok else 1


cam = load_camera()
K, dist = cam["K"], cam["dist"]
W, H = cam["size"]
SIDE = 0.030  # マーカーの 1 辺 [m]。黒い枠の外側から外側まで
PAD = 1.4  # 画像の 1 辺 = マーカーの何倍か(まわりに白い余白を付ける。印刷物と同じ)

# 「歪んだ画像の画素 → 理想の画像のどこか」の対応表(check_real_camera.py と同じ作り方。remap で歪んだ画像を作るのに使う)
gx, gy = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
ideal_xy = undistort_points(np.stack([gx.ravel(), gy.ravel()], axis=1), K, dist).reshape(H, W, 2).astype(np.float32)

cam_pos = np.array([0.30, -0.10, 0.60])  # 机の斜め上 60 cm
cam_R = look_at_rotation(cam_pos, np.array([0.20, 0.0, 0.0]))


def render(marker_id, R_w, p_w, noise=0.0, seed=0):
    """世界座標の向き R_w・位置 p_w に置いたマーカーを、実機のカメラで撮った画像(RGB)を作る。"""
    # 世界 → OpenCV のカメラ座標。CV_TO_CAM は、自分自身が逆行列(y, z の符号を反転するだけ)なので、同じ行列で往復できる。
    R_cv = CV_TO_CAM @ cam_R.T @ R_w
    t_cv = CV_TO_CAM @ cam_R.T @ (p_w - cam_pos)
    pad_pts = marker_object_points(SIDE * PAD)  # 余白を含む四角の 4 つの角
    px, _ = cv2.projectPoints(pad_pts, cv2.Rodrigues(R_cv)[0], t_cv, K, None)  # 歪みなしで画素に投影
    src_px = 140
    marker = make_marker_image(marker_id, int(src_px / PAD))
    sheet = np.full((src_px, src_px), 255, np.uint8)  # 白い余白つきの、印刷物にあたる画像
    off = (src_px - marker.shape[0]) // 2
    sheet[off:off + marker.shape[0], off:off + marker.shape[1]] = marker
    src_corners = np.array([[0, 0], [src_px, 0], [src_px, src_px], [0, src_px]], dtype=np.float32)
    Hm = cv2.getPerspectiveTransform(src_corners, px.reshape(4, 2).astype(np.float32))  # 印刷物の画素 → 画像の画素 の対応(ホモグラフィ)
    ideal = cv2.warpPerspective(sheet, Hm, (W, H), flags=cv2.INTER_AREA, borderValue=120)  # 背景は中間の灰色
    distorted = cv2.remap(ideal, ideal_xy[:, :, 0], ideal_xy[:, :, 1], cv2.INTER_LINEAR)
    if noise > 0:
        rng = np.random.default_rng(seed)
        distorted = np.clip(distorted + rng.normal(0, noise, distorted.shape), 0, 255).astype(np.uint8)
    return cv2.cvtColor(distorted, cv2.COLOR_GRAY2RGB)


def pose_facing_camera(p_w, tilt_deg=0.0, spin_deg=0.0):
    """位置 p_w のマーカーの向き。基準 = 面がカメラのほうを向く向き。tilt_deg = 面の x 軸まわりに傾ける、spin_deg = 面に垂直な軸まわりに回す。"""
    # look_at_rotation(目の位置, 見る先) は、z 軸が「見る先 → 目」の向きの回転を返す。目 = カメラ、見る先 = マーカーにすると、
    # z 軸(面の垂直)が マーカー → カメラ の向きになり、面はカメラのほうを向く(引数を逆にすると、面が裏を向いて、鏡に映ったように見える)。
    R0 = look_at_rotation(cam_pos, p_w)

    def rot(axis, deg):
        a = np.radians(deg)
        c, s = np.cos(a), np.sin(a)
        return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]
    return R0 @ rot("x", tilt_deg) @ rot("z", spin_deg)


def angle_between(Ra, Rb):
    """2 つの回転行列の間の角度 [度]。"""
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


cases = {
    "真正面・中央付近 (ID 3)": (3, np.array([0.20, 0.0, 0.0]), 0, 0),
    "30° 傾ける (ID 7)": (7, np.array([0.22, 0.03, 0.02]), 30, 20),
    "50° 傾ける (ID 12)": (12, np.array([0.18, -0.02, 0.04]), 50, -40),
    "画像の端に近い (ID 0)": (0, np.array([0.40, 0.12, 0.0]), 20, 10),
    "遠い 90 cm (ID 5)": (5, np.array([0.10, 0.05, -0.20]), 10, 0),
}
for name, (mid, p, tilt, spin) in cases.items():
    R_w = pose_facing_camera(p, tilt, spin)
    img = render(mid, R_w, p)
    found = detect_markers(img, K, dist, SIDE)
    if mid not in found:
        check(f"{name}: ID が読める(見つかった ID: {sorted(found)})", False)
        continue
    R_est, p_est = marker_pose_in_world(found[mid]["R"], found[mid]["t"], cam_pos, cam_R)
    # 位置の誤差を、「カメラから見て横方向」と「奥行き方向(カメラに向かう方向)」に分ける。
    # マーカー 1 枚では、横は画素の位置から正確に出るが、奥行きは「見かけの大きさ」から出すので、角の位置のわずかなずれが効きやすい。
    ray = (p - cam_pos) / np.linalg.norm(p - cam_pos)  # カメラ → マーカーの向きの単位ベクトル
    err = p_est - p
    e_depth = abs(err @ ray) * 1000
    e_side = np.linalg.norm(err - (err @ ray) * ray) * 1000  # 誤差から、奥行き方向の成分を引いた残り = 横方向
    e_ang = angle_between(R_w, R_est)
    dist_cam = np.linalg.norm(p - cam_pos) * 100
    print(f"    {name}: カメラから {dist_cam:.0f} cm、横の誤差 {e_side:.2f} mm、奥行きの誤差 {e_depth:.2f} mm、向きの誤差 {e_ang:.2f}°")
    # 基準は、30 mm のマーカーを 60〜90 cm で撮ったときの実測(奥行き 0.4〜4 mm、向き: 傾けると 1° 未満、真正面だと 3〜5°)に、余裕を持たせた値。
    # 真正面で向きの誤差が大きいのは、PnP の性質(平らな面を正面から見ると、傾きの向きが決まりにくい)。
    check(f"{name}: ID が読め、横 {e_side:.2f} mm < 1.0、奥行き {e_depth:.2f} mm < 6.0、向き {e_ang:.2f}° < 6.0",
          e_side < 1.0 and e_depth < 6.0 and e_ang < 6.0)

# (4) 画素ノイズ(標準偏差 4。8 ビットの 0〜255 で)を足しても読める
p = np.array([0.22, 0.03, 0.02])
R_w = pose_facing_camera(p, 30, 20)
errs = []
for seed in range(5):
    found = detect_markers(render(3, R_w, p, noise=4.0, seed=seed), K, dist, SIDE)
    if 3 in found:
        errs.append(np.linalg.norm(marker_pose_in_world(found[3]["R"], found[3]["t"], cam_pos, cam_R)[1] - p) * 1000)
print(f"    ノイズあり 5 回: 見つかった {len(errs)} 回、位置の誤差 {', '.join(f'{e:.2f}' for e in errs)} mm")
check("(4) ノイズを足しても、5 回とも読め、位置の誤差が 3 mm 未満", len(errs) == 5 and max(errs) < 3.0)

# (5) 印刷用の画像が、そのまま読める(余白つきの白い紙に貼った状態)
sheet = np.full((300, 300), 255, np.uint8)
sheet[50:250, 50:250] = make_marker_image(9, 200)
sheet_found = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)).detectMarkers(sheet)[1]
check(f"(5) 印刷用に作った ID 9 の画像が、ID 9 として読める(読めた ID: {None if sheet_found is None else sheet_found.ravel().tolist()})",
      sheet_found is not None and sheet_found.ravel().tolist() == [9])

print(f"\n{'全部 OK' if n_ng == 0 else f'NG が {n_ng} 件'}")
sys.exit(1 if n_ng else 0)
