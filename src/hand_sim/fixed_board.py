"""【概要】机に固定する ArUco の板(カメラの置き場所の較正用)の、マーカーの配置。
A4 縦の紙に、ArUco(ID 1〜6、1 辺 30 mm)を 2 列 × 3 行で並べる。ID 0 は、手の板が使っているので避ける。
板の座標(ここで決める): 原点 = 紙に印刷した「原点の十字」、x = 紙の右、y = 紙の上(紙を読む向きで)、z = 紙の表から外へ(机に置けば上)。単位 mm。
  - 印刷物の中の位置関係は、この配置の値で決まる(プリンタの誤差は 0.2 mm 程度。印刷後に 100 mm の線で確かめる)。
  - 作者が測るのは「板の座標 → ロボットの基準(世界座標)」だけ。板の原点の十字の位置(x, y)・向き・高さ。
    すると、各マーカーの世界座標は board_marker_centers_world で出せる。
  - 配置は make_fixed_board.py(印刷用)と共通。ここを変えたら、印刷し直す。
"""
import numpy as np

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
