"""【概要】実機のウェブカメラの画像を、sim と同じ「歪みのない(ピンホール)カメラ」の画像として扱うための部品。
チェスボードの較正(calibration/webcam_intrinsics.json)で求めた内部パラメータ(カメラ行列 K と歪みの係数 dist)を使う。
  - load_camera: 較正の結果(K, dist)を JSON から読む。
  - undistort_image: 歪んだ画像を、歪みのない画像に直す(直した画像は、同じ K のピンホールカメラで撮ったのと同じになる)。
  - undistort_points: 歪んだ画像の中の画素位置 (u, v) だけを、歪みのない画像での位置に直す(画像全体を直すより速い)。
  - distort_pixels: 逆向き。歪みのない画素位置が、このレンズでは、どの画素に写るか(合成データを作るときに使う)。
  - project_points_K / pixel_to_world_plane_K: camera_pose.py の project_points / pixel_to_world_plane の「K 版」。
    camera_pose.py は「画像の中心 = 画素の真ん中、縦横の焦点距離は同じ」と仮定している。実機は、cx, cy が真ん中から数〜17 画素ずれていて(fx と fy も少し違う)、
    この差は、カメラから 60 cm で、数 mm〜7 mm の位置のずれになる。そのまま使うと、掴む位置が、その分ずれる。K 版は、cx, cy, fx, fy を、そのまま使う。
流れ(実機): カメラ画像 → undistort_image → vision.py の色検出(sim と同じ) → (u, v) → pixel_to_world_plane_K → 世界座標。
カメラの座標系は camera_pose.py と同じ(MuJoCo と同じ): 見る向きが −z、画像の右が +x、画像の上が +y。回転行列 cam_R は「カメラ → 世界」。
カメラの置き場所(cam_pos, cam_R)は、まだ求めていない(次の工程: カメラの固定位置の決定と、目印での較正)。
"""
import json
from pathlib import Path

import cv2  # OpenCV: 画像処理・カメラの較正のライブラリ。[頻度] 画像処理では定番
import numpy as np

ROOT = Path(__file__).resolve().parents[2]  # このファイル(src/hand_sim/real_camera.py)から 2 つ上 = リポジトリの直下。[頻度] 設定ファイルの場所を決めるとき、よく使う
CALIB_PATH = ROOT / "calibration" / "webcam_intrinsics.json"


def load_camera(path=CALIB_PATH):
    """scripts/calibrate_webcam_intrinsics.py が保存した JSON を読み、{"K": 3×3, "dist": 5 個, "size": (幅, 高さ), "fovy_deg": ...} を返す。
    K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]](単位は画素)。dist = [k1, k2, p1, p2, k3](歪みの係数)。"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {"K": np.array(data["K"]), "dist": np.array(data["dist"]).ravel(), "size": tuple(data["image_size"]), "fovy_deg": data["fovy_deg"]}


def undistort_image(img, K, dist):
    """歪んだ画像 img(高さ × 幅 × 3)を、歪みのない画像に直して返す。大きさ・K は、元と同じ(新しい K は渡さない)。
    cv2.undistort の、新しいカメラ行列を省略した形 = 元の K のまま。樽型の歪みでは、直した画像の、どの画素も、元の画像の中の画素から取れるので、黒い欠けは出ない
    (その代わり、画像の端の絵が、中心から離れる向きに引き伸ばされる)。[頻度] 実機のカメラの画像を扱うときの定番"""
    return cv2.undistort(img, K, dist)


def undistort_points(uv, K, dist):
    """歪んだ画像の中の画素位置 uv((N, 2) の配列、または (u, v) の 1 点)を、歪みのない画像での画素位置に直して返す(同じ形)。
    歪みの式は、逆向きに解けない(式で書けない)ので、繰り返し計算で近づける。cv2.undistortPoints の既定は 5 回で、歪みの大きい画像の端では足りない(0.5 画素ほど残る)。
    そこで、回数と止める条件(criteria)を指定して、100 回まで・誤差 1e-10 で止める(OpenCV 5.0。古い版には、別名の undistortPointsIter があった)。P=K を渡すと、結果を、正規化座標ではなく画素に戻してくれる。"""
    pts = np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2)  # OpenCV の点の形は (N, 1, 2)
    criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-10)  # (止める条件の種類, 最大の回数, 誤差)。[頻度] OpenCV の繰り返し計算で、よく出てくる書き方
    out = cv2.undistortPoints(pts, K, dist, None, None, K, criteria).reshape(-1, 2)  # 引数: 点, K, dist, dst(出力先。None = 新しく作る), R(回転。None = 使わない), P(画素に戻す行列), 止める条件
    return out[0] if np.asarray(uv).ndim == 1 else out


def distort_pixels(u, v, K, dist):
    """歪みのない理想の画素 (u, v) が、レンズを通ると、どの画素に写るかを返す(OpenCV の歪みのモデルそのまま)。u, v は、数でも配列でもよい。
    手順: 画素 → 正規化座標(中心を 0、焦点距離で割る)→ 歪みの式 → 画素に戻す。歪みの式の意味は docs/webcam_calibration.md の 0-1。"""
    k1, k2, p1, p2, k3 = dist
    x, y = (u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1]
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3  # 中心から離れるほど、画素が内側へ縮む(k1 が負 = 樽型)
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return xd * K[0, 0] + K[0, 2], yd * K[1, 1] + K[1, 2]


def project_points_K(points, cam_pos, cam_R, K):
    """世界座標の点 (N×3) が、歪みのない画像のどの画素 (u, v) に写るかを、(N×2) の配列で返す。カメラの後ろにある点は nan。
    camera_pose.project_points の K 版(視野角と画像の大きさの代わりに、fx, fy, cx, cy をそのまま使う)。"""
    p_cam = (np.asarray(points, dtype=float) - cam_pos) @ cam_R  # 世界 → カメラの座標系(R.T @ (p − c) を、行ごとに計算した形)
    depth = -p_cam[:, 2]  # カメラの前方への距離(カメラは −z を見る)
    with np.errstate(divide="ignore", invalid="ignore"):  # 割り算の警告(depth が 0 や負)を出さない
        u = K[0, 0] * p_cam[:, 0] / depth + K[0, 2]
        v = -K[1, 1] * p_cam[:, 1] / depth + K[1, 2]  # v は下へ増える。カメラの y は上へ増えるので、符号が逆
    u[depth <= 0] = np.nan
    v[depth <= 0] = np.nan
    return np.stack([u, v], axis=1)


def pixel_to_world_plane_K(u, v, cam_pos, cam_R, K, z_plane):
    """歪みのない画像の画素 (u, v) の方向へ光線を伸ばし、高さ z_plane [m] の水平な面と交わる点の世界座標 (x, y, z) を返す。交わらなければ None。
    camera_pose.pixel_to_world_plane の K 版。歪んだ画像の画素を渡すときは、先に undistort_points で直す(または、画像を undistort_image で直してから色検出する)。"""
    ray_cam = np.array([(u - K[0, 2]) / K[0, 0], -(v - K[1, 2]) / K[1, 1], -1.0])  # カメラの座標系での光線の向き
    ray = cam_R @ ray_cam  # 世界座標での向き
    if abs(ray[2]) < 1e-9:  # 水平に近すぎて、面と交わらない
        return None
    t = (z_plane - cam_pos[2]) / ray[2]  # カメラから、面までの、光線の長さ(の倍率)
    if t <= 0:  # 面が、カメラの後ろ側
        return None
    return np.asarray(cam_pos, dtype=float) + t * ray
