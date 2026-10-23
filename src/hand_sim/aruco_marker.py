"""【概要】ロボットに貼る ArUco マーカー(白黒の四角い目印)の、作成と検出。
ArUco マーカー = 黒い枠の中に、白黒の格子で ID を入れた、正方形の目印。1 枚の画像から「ID」と「4 つの角の画素位置」が取れ、
大きさ(1 辺の長さ)が分かっていれば、カメラから見た、マーカーの位置と向き(姿勢)まで求まる。
  - make_marker_image: 印刷用のマーカー画像(白黒)を作る。
  - detect_markers: 画像から、マーカーを探し、{ID: {"corners": 角の画素, "R": 向き, "t": 位置}} を返す(歪んだ実機の画像のまま渡してよい)。
  - marker_pose_in_world: カメラ座標のマーカーの姿勢を、世界座標に直す(カメラの置き場所 cam_pos, cam_R が分かっているとき)。
座標の向きの注意: OpenCV のカメラ座標は「右が +x、下が +y、前が +z」。camera_pose.py / MuJoCo は「右が +x、上が +y、前が −z」で、y と z の符号が逆。
detect_markers が返す R, t は OpenCV 側(solvePnP の結果そのまま)。marker_pose_in_world が、この符号の違いを直す。
"""
import cv2  # OpenCV: 画像処理のライブラリ。aruco(マーカー)の機能もここに入っている。[頻度] 画像処理では定番
import numpy as np

DICTIONARY = cv2.aruco.DICT_4X4_50  # マーカーの種類(辞書)。4×4 の格子で ID 0〜49 の 50 種類。格子が粗いほど、小さく・遠くても読める。[頻度] 数が足りる限り、小さい辞書を選ぶのが定番


def _dictionary():
    return cv2.aruco.getPredefinedDictionary(DICTIONARY)


def make_marker_image(marker_id, side_px=400, border_bits=1):
    """ID が marker_id のマーカーを、1 辺 side_px 画素の白黒画像(高さ × 幅、0 = 黒、255 = 白)で返す。
    border_bits = 外側の黒い枠の太さ(格子の 1 マス分が 1)。印刷するときは、まわりに白い余白(マーカーの 1 マス分以上)を残すこと(余白が無いと、枠が見つからない)。"""
    return cv2.aruco.generateImageMarker(_dictionary(), marker_id, side_px, borderBits=border_bits)


def marker_object_points(side_m):
    """マーカーの 4 つの角の、マーカー自身の座標系での位置 [m]。原点 = マーカーの中心、x = 右、y = 上、z = 手前(マーカーの面に垂直)。
    角の順番は、ArUco の検出結果の並びと同じ(左上、右上、右下、左下)。solvePnP に「どの 3D 点が、どの画素に写ったか」を教えるために使う。"""
    h = side_m / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)


def detect_markers(img_rgb, K, dist, side_m):
    """画像(高さ × 幅 × 3、RGB)から ArUco マーカーを全部探し、{ID: {"corners": (4, 2) の画素, "R": 3×3, "t": 3}} を返す。見つからなければ空の辞書。
    img_rgb は、歪んだままの実機の画像でよい(歪みは solvePnP に dist を渡して、中で考慮する)。side_m = マーカーの 1 辺 [m](黒い枠の外側から外側まで)。
    R, t = マーカー → カメラ(OpenCV の向き)の回転と位置。t はマーカーの中心の、カメラから見た位置 [m]。"""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)  # 検出は白黒の画像で行う
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX  # 角の位置を、1 画素より細かく(サブピクセルで)求める。位置の精度のために、ほぼ必須
    detector = cv2.aruco.ArucoDetector(_dictionary(), params)
    corners, ids, _rejected = detector.detectMarkers(gray)  # _rejected: マーカーに似ていたが違った候補。使わないので、頭に _ を付ける習慣
    found = {}
    if ids is None:
        return found
    obj = marker_object_points(side_m)
    for c, marker_id in zip(corners, ids.ravel()):
        c = c.reshape(4, 2).astype(np.float64)
        # IPPE_SQUARE: 「正方形の 4 つの角」専用の解き方。角が 4 つしか無いときに、ふつうの解法より安定する。
        ok, rvec, tvec = cv2.solvePnP(obj, c, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            continue
        R, _ = cv2.Rodrigues(rvec)  # 回転ベクトル(3 個の数)→ 回転行列(3×3)。[頻度] OpenCV の姿勢を扱うとき、よく使う
        found[int(marker_id)] = {"corners": c, "R": R, "t": tvec.ravel()}
    return found


# OpenCV のカメラ座標(右 x、下 y、前 z)→ camera_pose.py のカメラ座標(右 x、上 y、後ろ −z)。y と z の符号を反転する行列。
CV_TO_CAM = np.diag([1.0, -1.0, -1.0])


def marker_pose_in_world(R_cv, t_cv, cam_pos, cam_R):
    """detect_markers の R, t(OpenCV のカメラ座標)を、世界座標の (向き 3×3, 位置 3) に直す。
    cam_pos, cam_R = カメラの世界での位置と向き(camera_pose.py と同じ。cam_R は「カメラ → 世界」)。"""
    R_cam = CV_TO_CAM @ R_cv  # マーカーの向きを、camera_pose.py のカメラ座標で表す
    t_cam = CV_TO_CAM @ t_cv
    return cam_R @ R_cam, np.asarray(cam_pos, dtype=float) + cam_R @ t_cam
