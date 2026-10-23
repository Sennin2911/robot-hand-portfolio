"""【概要】ロボットの手に貼った、色付きの丸いシール(赤・黄・緑・青)を目印にして、手の面の位置と向きを求める。
ArUco(aruco_marker.py)の代わりに、手元のシールを使う方式。1 枚のシールでは位置(中心)しか分からないが、
「平らな面に貼った 4 枚が、互いに何 mm 離れているか」が分かっていれば、4 つの中心の画素位置から、ArUco と同じ solvePnP で、面の位置と向きが出る。
色が違うので、どのシールがどれか(対応づけ)で迷わない。
  - find_color_dots: 画像から、色ごとに、いちばん大きい塊の中心(画素)と面積を探す。
  - detect_dot_marker: 4 色以上の中心から、面の位置・向き R, t を求める(戻り値の形と座標の向きは aruco_marker.detect_markers と同じ)。
  - DEFAULT_LAYOUT: シールの貼り方(面の座標系での、各色の中心の位置 [mm])。実際に貼った位置を測って、この値に差し替える。
面の座標系: 原点 = 4 枚の真ん中あたり、x = 面をカメラ側から見て右、y = 上、z = 面から手前(カメラのほう)へ垂直。aruco_marker.marker_object_points と同じ向き。
世界座標への換算は、aruco_marker.marker_pose_in_world がそのまま使える。
"""
import cv2  # OpenCV: 画像処理のライブラリ。[頻度] 画像処理では定番
import numpy as np

from hand_sim.real_camera import undistort_points

# 色ごとの、色相(H)の範囲。OpenCV の H は 0〜179(色の種類を、円周上の角度で表した値。赤 0 付近、黄 30 付近、緑 60 付近、青 120 付近)。
# 赤は、H が 0 をまたぐので、2 つの範囲に分ける。彩度 S(色の濃さ)と明るさ V の下限は、灰色・黒・白(S が低い)を除くため。
# 彩度・明るさの下限は、手元のシールを撮ったスマホの写真(docs/img/robot-arm-marking.jpg)の実測で決めた: シールの S は 150〜215、肌・木の机は S が 45〜60 なので、下限 130 で肌・机を外せる。
# 緑・紺は暗く写り(V が 43〜55)、下限を 30 にしないと拾えない。スマホの写真は、ウェブカメラより暗く写っているので、実機のウェブカメラの画像で、確かめ直す(view_real_dots.py)。
HUE_RANGES = {
    "red": [(0, 8), (172, 179)],
    "yellow": [(20, 36)],
    "green": [(40, 90)],
    "blue": [(100, 130)],
}
MIN_SATURATION, MIN_VALUE = 130, 30

# 既定の貼り方: 2×2、間隔 30 mm。{色: (x, y)}(面の座標系、mm)。x = 右、y = 上。
DEFAULT_LAYOUT = {"red": (-15.0, 15.0), "yellow": (15.0, 15.0), "green": (-15.0, -15.0), "blue": (15.0, -15.0)}


def color_mask(img_rgb, color):
    """画像(高さ × 幅 × 3、RGB)の中で、color に当たる画素が 255、他が 0 の画像(高さ × 幅)を返す。"""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)  # RGB → HSV(色相・彩度・明るさ)。色の種類を、明るさと分けて扱える。[頻度] 色で物を探すときの定番
    mask = np.zeros(img_rgb.shape[:2], np.uint8)
    for lo, hi in HUE_RANGES[color]:
        mask |= cv2.inRange(hsv, (lo, MIN_SATURATION, MIN_VALUE), (hi, 255, 255))  # |= は「または」を足し込む。inRange: 範囲に入る画素を 255 にする
    return mask


def find_color_dots(img_rgb, colors=("red", "yellow", "green", "blue"), min_area=15):
    """色ごとに、いちばん大きい塊を探し、{色: {"uv": (u, v), "area": 面積 [画素]}} を返す。min_area 画素未満の塊、見つからない色は、含めない。
    中心 uv は、塊の画素の位置の平均(重心)。画素の中心を 0 とする座標(OpenCV の角の座標と同じ)。"""
    found = {}
    for color in colors:
        mask = color_mask(img_rgb, color)
        # connectedComponentsWithStats: つながった塊ごとに、番号を付け、面積と重心を返す。0 番は背景。[頻度] 塊を数えるときの定番
        n, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n <= 1:
            continue
        best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))  # 背景(0 番)を除いて、面積が最大の塊
        area = int(stats[best, cv2.CC_STAT_AREA])
        if area >= min_area:
            found[color] = {"uv": centroids[best].copy(), "area": area}
    return found


def detect_dot_marker(img_rgb, K, dist, layout=None, min_area=15):
    """画像から色のシールを探し、4 色以上が見つかれば、面の位置・向きを返す。見つからなければ None。
    戻り値: {"R": 3×3, "t": 3, "dots": find_color_dots の結果, "reproj_px": 再投影の誤差(各点の平均、画素)}。R, t = 面 → カメラ(OpenCV の向き)。
    layout = {色: (x, y) mm}(省略すると DEFAULT_LAYOUT)。img_rgb は、歪んだままの実機の画像でよい(中心の画素だけ、歪みを直す)。"""
    layout = layout or DEFAULT_LAYOUT
    dots = find_color_dots(img_rgb, tuple(layout), min_area)
    use = [c for c in layout if c in dots]  # 見つかった色(layout の順)
    if len(use) < 4:  # 平らな面の姿勢を決めるには、4 点が要る(3 点以下では解が定まらない)
        return None
    obj = np.array([[layout[c][0] / 1000, layout[c][1] / 1000, 0.0] for c in use])  # 面の座標系での位置 [m]。z = 0(平らな面)
    uv = undistort_points(np.array([dots[c]["uv"] for c in use]), K, dist)  # 歪みを直した位置(以降は、歪みのないカメラとして解く)
    # IPPE: 「平らな面の上の点」専用の解き方。点が 4 つ以上あるとき、安定する。
    ok, rvec, tvec = cv2.solvePnP(obj, uv, K, None, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)  # 求めた姿勢で、点を画像に投影し直す
    reproj = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - uv, axis=1)))  # 実際の位置とのずれ(画素)。大きいと、色の取り違え・貼り位置の測り違いを疑う
    R, _ = cv2.Rodrigues(rvec)
    return {"R": R, "t": tvec.ravel(), "dots": {c: dots[c] for c in use}, "reproj_px": reproj}
