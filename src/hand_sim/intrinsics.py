"""【概要】カメラの内部パラメータ(焦点距離、画像の中心、レンズの歪み)を、チェスボードの画像から求める(OpenCV を使う)。
カメラの内部の性質は、カメラをどこに置いても変わらない。置き場所(外部パラメータ)の較正(camera_pose.py、marker_calib.py)の前に、先に求めておく。
  - find_corners: 画像から、チェスボードの内側の角(9 × 6 個)を見つける(見つからなければ None)。
  - calibrate_intrinsics: 複数の画像の角から、カメラ行列 K(焦点距離 fx, fy、中心 cx, cy)と歪み dist を求める。
  - fovy_deg: カメラ行列から、縦の視野角を出す(sim のカメラの fovy と同じ意味)。
  - save_intrinsics / load_intrinsics: 結果を JSON で保存・読み込み。
チェスボードは scripts/make_chessboard.py で作る(1 マス 25 mm、内側の角 9 × 6 個)。
"""
import json

import cv2  # OpenCV。画像処理・カメラの較正の定番ライブラリ。[頻度] カメラを扱うときの定番
import numpy as np

PATTERN = (9, 6)  # 内側の角の数(横、縦)
SQUARE_M = 0.025  # 1 マスの大きさ [m]


def object_points():
    """チェスボード上の、内側の角の 3 次元の位置 (54 × 3、単位 m)。盤の面を z = 0、左上の角を原点にする。"""
    grid = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
    grid[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2)  # np.mgrid: 格子の座標を作る。(0,0), (1,0), (2,0), ...
    return grid * SQUARE_M


def find_corners(image):
    """画像(RGB、または、白黒)から、チェスボードの内側の角を探す。見つかれば (54 × 1 × 2) の画素位置(小数の精度に直したもの)、見つからなければ None。"""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # cv2.cvtColor: 色の形式を変える。RGB → 白黒
    found, corners = cv2.findChessboardCorners(gray, PATTERN, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        return None
    # cornerSubPix: 角の位置を、1 画素より細かい精度(サブピクセル)に直す。較正の精度は、ここで決まる。
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    # OpenCV のバージョンで、戻り値の形が違う(4.x は 54 × 1 × 2、5.x は 54 × 2)ので、54 × 1 × 2 の float32 にそろえる。
    return np.asarray(refined, dtype=np.float32).reshape(-1, 1, 2)


def calibrate_intrinsics(corner_list, image_size):
    """角の見つかった画像ぶんの corner_list(find_corners の結果のリスト)と、画像の大きさ (幅, 高さ) から、内部パラメータを求める。

    戻り値: 辞書 {"K": カメラ行列(3×3)、"dist": 歪みの係数、"rms": 再投影誤差 [画素]、"per_image_rms": 画像ごとの誤差、"image_size": (幅, 高さ)、
                  "fovy_deg": 縦の視野角、"fovx_deg": 横の視野角}。
    再投影誤差: 求めた内部パラメータで、盤の角を画像に投影し直したときの、実際の角の位置との差(二乗平均平方根)。1 画素より小さければ、良い較正。
    """
    objpoints = [object_points()] * len(corner_list)
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, corner_list, image_size, None, None)
    per_image = []
    for obj, img, rvec, tvec in zip(objpoints, corner_list, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)  # 求めた値で、角を画像に投影し直す
        per_image.append(float(np.sqrt(np.mean(np.sum((projected - img) ** 2, axis=2)))))
    w, h = image_size
    return {"K": K, "dist": dist.ravel(), "rms": float(rms), "per_image_rms": per_image, "image_size": (int(w), int(h)),
            "fovy_deg": fovy_deg(K, h), "fovx_deg": float(np.degrees(2 * np.arctan(w / (2 * K[0, 0]))))}


def fovy_deg(K, height):
    """カメラ行列 K と画像の高さ [画素] から、縦の視野角 [度] を出す。sim のカメラの fovy と同じ意味(焦点距離 f = (高さ / 2) / tan(fovy / 2))。"""
    return float(np.degrees(2 * np.arctan(height / (2 * K[1, 1]))))


def save_intrinsics(result, path):
    """calibrate_intrinsics の結果(numpy 配列を含む)を、JSON に保存する。JSON は、数・文字・リストしか入れられないので、配列は .tolist() でリストに直す。"""
    data = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in result.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_intrinsics(path):
    """save_intrinsics で保存した JSON を読み、K と dist を numpy 配列に戻して返す。"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    data["K"] = np.array(data["K"])
    data["dist"] = np.array(data["dist"])
    return data
