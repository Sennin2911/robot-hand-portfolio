"""【概要】机に固定する ArUco の板(カメラの置き場所の較正用)の、マーカーの配置。
A4 縦の紙に、ArUco(ID 1〜6、1 辺 30 mm)を 2 列 × 3 行で並べる。ID 0 は、手の板が使っているので避ける。
板の座標(ここで決める): 原点 = 紙に印刷した「原点の十字」、x = 紙の右、y = 紙の上(紙を読む向きで)、z = 紙の表から外へ(机に置けば上)。単位 mm。
  - 印刷物の中の位置関係は、この配置の値で決まる(プリンタの誤差は 0.2 mm 程度。印刷後に 100 mm の線で確かめる)。
  - 作者が測るのは「板の座標 → ロボットの基準(世界座標)」だけ。板の原点の十字の位置(x, y)・向き・高さ。
    すると、各マーカーの世界座標は board_marker_centers_world で出せる。
  - 配置は make_fixed_board.py(印刷用)と共通。ここを変えたら、印刷し直す。
"""
import cv2  # OpenCV。ここでは solvePnP(「3D の点が、画像のどこに写ったか」から、カメラの位置・向きを求める関数)を使う
import numpy as np

# 板の置き方(作者の測定 2026-10-24): 十字の世界座標 (x, y) = (72, 0) mm。向きは、ロボットの後ろから見て紙の上下が正しく読める向き(= −π/2)の想定。
# 高さ 0(机に直接)・水平の想定。向きと高さは、作者に確認する(未確認)。
BOARD_XY_M = (0.072, 0.0)
BOARD_YAW_RAD = -np.pi / 2
BOARD_Z_M = 0.0

SIDE_MM = 30.0  # マーカーの 1 辺 [mm](黒い枠の外側から外側まで)。印刷後にノギスで測り直し、違えば測った値を使う
PAGE_W_MM, PAGE_H_MM = 210.0, 297.0  # A4 縦
ORIGIN_PAGE_MM = (30.0, 250.0)  # 原点の十字の、紙の左上からの位置 (右へ, 下へ) [mm]

# 各マーカー中心の、紙の左上からの位置 (右へ, 下へ) [mm]。ID: (右へ, 下へ)
# 列の間隔 90 mm、行の間隔 80 mm。マーカーどうしを離すほど、カメラの位置の計算が安定する(近すぎると奥行きが決まりにくい)。
_PAGE_CENTERS_MM = {
    1: (60.0, 50.0), 2: (150.0, 50.0),
    3: (60.0, 130.0), 4: (150.0, 130.0),
    5: (60.0, 210.0), 6: (150.0, 210.0),
}


def board_marker_centers_mm():
    """{ID: (x, y)}。板の座標(原点 = 十字、x = 右、y = 上)での、マーカー中心の位置 [mm]。
    紙の座標は「下へ」が増える向きなので、y は「原点の下へ − マーカーの下へ」で、上向きに直している。"""
    ox, oy = ORIGIN_PAGE_MM
    return {i: (px - ox, oy - py) for i, (px, py) in _PAGE_CENTERS_MM.items()}


def board_marker_centers_world(board_xy_m, board_yaw_rad=0.0, board_z_m=0.0):
    """板の置き方から、各マーカー中心の世界座標 [m] を出す。{ID: (3,) の配列}。
    board_xy_m = 板の原点(十字)の、世界座標での (x, y) [m]。board_yaw_rad = 板の x 軸が、世界の x 軸から左回りに何 rad 回っているか
    (板の下の辺を、土台の側面に平行に置けば 0 または ±π/2 など)。board_z_m = 板の表面の高さ [m]。板は水平に置く前提。"""
    c, s = np.cos(board_yaw_rad), np.sin(board_yaw_rad)
    out = {}
    for i, (x, y) in board_marker_centers_mm().items():
        out[i] = np.array([board_xy_m[0] + c * x * 1e-3 - s * y * 1e-3,
                           board_xy_m[1] + s * x * 1e-3 + c * y * 1e-3,
                           board_z_m])
    return out


def board_marker_corners_world(side_mm=SIDE_MM, board_xy_m=BOARD_XY_M, board_yaw_rad=BOARD_YAW_RAD, board_z_m=BOARD_Z_M):
    """各マーカーの 4 つの角の世界座標 [m]。{ID: (4, 3)}。角の順番は ArUco の検出結果と同じ(紙の向きで、左上、右上、右下、左下)。
    マーカーは紙の上で正立(上 = 紙の上)に印刷してあるので、板の座標では、中心から (−h, +h)、(+h, +h)、(+h, −h)、(−h, −h)(h = 1 辺の半分)。"""
    h = side_mm / 2
    c, s = np.cos(board_yaw_rad), np.sin(board_yaw_rad)
    out = {}
    for i, (cx, cy) in board_marker_centers_mm().items():
        pts = []
        for dx, dy in ((-h, h), (h, h), (h, -h), (-h, -h)):
            x, y = (cx + dx) * 1e-3, (cy + dy) * 1e-3  # 板の座標 [m]
            pts.append([board_xy_m[0] + c * x - s * y, board_xy_m[1] + s * x + c * y, board_z_m])
        out[i] = np.array(pts)
    return out


def camera_pose_from_board(found, K, dist, side_mm=SIDE_MM, **board):
    """detect_markers の結果 found(ID → 角の画素)のうち、板の ID(1〜6)を全部使って、カメラの世界座標での位置と向きを求める。
    戻り値: (cam_pos (3,) [m], cam_R (3×3), 再投影誤差の平均 [画素], 使ったマーカーの数)。板のマーカーが 1 つも無ければ None。
    cam_R は camera_pose.py と同じ約束(カメラ座標は右 x、上 y、後ろ z → 世界)。aruco_marker.marker_pose_in_world にそのまま渡せる。
    1 個のマーカー(4 点)で解くより、6 個(24 点)を一度に使うほうが、位置の誤差が小さい。"""
    corners = board_marker_corners_world(side_mm, **board)
    ids = [i for i in sorted(found) if i in corners]
    if not ids:
        return None
    obj = np.vstack([corners[i] for i in ids])
    img_pts = np.vstack([found[i]["corners"] for i in ids])
    # SOLVEPNP_ITERATIVE: 最小二乗で、画素のずれが最小になる位置・向きを探す。点が 4 つ以上あれば使える。[頻度] 点が多いときの定番
    ok, rvec, tvec = cv2.solvePnP(obj, img_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    err = float(np.linalg.norm(proj.reshape(-1, 2) - img_pts, axis=1).mean())
    R_wc, _ = cv2.Rodrigues(rvec)  # 世界 → カメラ(OpenCV の向き)の回転
    t = tvec.ravel()
    cv_to_cam = np.diag([1.0, -1.0, -1.0])  # OpenCV のカメラ座標 ↔ camera_pose.py のカメラ座標(y と z の符号を反転。逆行列も同じ)
    return -R_wc.T @ t, R_wc.T @ cv_to_cam, err, len(ids)
