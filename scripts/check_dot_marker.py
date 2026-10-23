"""【概要】dot_marker.py(色の丸いシール 4 枚で、手の面の位置・向きを出す)が正しいかを、合成画像で確認する。
手順: 世界座標の既知の位置・向きに、シール(2×2、間隔 30 mm)を貼った面を置き、実機のカメラ(webcam_intrinsics.json の K と歪み)で撮った画像を作る
(理想の画像 → 実機のレンズの歪みを加える → 画素ノイズ)。dot_marker.detect_dot_marker に渡し、返った位置・向きを、置いた値と比べる。
シールの直径(5 mm と 10 mm)、カメラからの距離(約 40・60・90 cm)、面の傾き(0・30・50°)を変えて、見つかるか・精度がどうなるかを表にする。
 (1) 全部の場合で 4 色が見つかる(10 mm)  (2) 精度が基準内(10 mm)  (3) 5 mm の結果(見つからない場合・誤差。基準は 10 mm と別)  (4) 貼り位置の測り誤差が、姿勢の誤差に、どう効くか
全部通れば終了コード0、そうでなければ1。

使い方(この .venv で実行):
    .venv/bin/python scripts/check_dot_marker.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)。
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from hand_sim.aruco_marker import CV_TO_CAM, marker_pose_in_world  # noqa: E402
from hand_sim.camera_pose import look_at_rotation  # noqa: E402
from hand_sim.dot_marker import DEFAULT_LAYOUT, detect_dot_marker  # noqa: E402
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
DOT_RGB = {"red": (220, 30, 30), "yellow": (240, 210, 20), "green": (20, 150, 40), "blue": (30, 60, 200)}  # シールの色(RGB)
FACE_GRAY = 60  # 手の面(背景)の明るさ。暗い灰色

gx, gy = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
ideal_xy = undistort_points(np.stack([gx.ravel(), gy.ravel()], axis=1), K, dist).reshape(H, W, 2).astype(np.float32)  # 歪んだ画像の作成用(check_aruco.py と同じ)

cam_pos = np.array([0.30, -0.10, 0.60])
cam_R = look_at_rotation(cam_pos, np.array([0.20, 0.0, 0.0]))


def pose_facing_camera(p_w, tilt_deg=0.0, spin_deg=0.0):
    """位置 p_w の面の向き。基準 = 面がカメラのほうを向く。tilt_deg = 面の x 軸まわりに傾ける、spin_deg = 面に垂直な軸まわりに回す。
    look_at_rotation(目の位置, 見る先) は、z 軸が「見る先 → 目」の向き。目 = カメラにすると、面の垂直(z)が カメラのほうを向く。"""
    R0 = look_at_rotation(cam_pos, p_w)

    def rot(axis, deg):
        a = np.radians(deg)
        c, s = np.cos(a), np.sin(a)
        return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]
    return R0 @ rot("x", tilt_deg) @ rot("z", spin_deg)


def render(R_w, p_w, diameter_mm, layout=DEFAULT_LAYOUT, noise=0.0, seed=0):
    """面(位置 p_w、向き R_w)に、直径 diameter_mm のシールを layout の位置に貼って、実機のカメラで撮った画像(RGB)を作る。"""
    R_cv = CV_TO_CAM @ cam_R.T @ R_w
    t_cv = CV_TO_CAM @ cam_R.T @ (p_w - cam_pos)
    rvec = cv2.Rodrigues(R_cv)[0]
    ideal = np.full((H, W, 3), FACE_GRAY, np.uint8)
    theta = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    r = diameter_mm / 2000  # 半径 [m]
    SHIFT = 4  # fillPoly の座標を、1/16 画素の細かさで渡す(縁を、画素より細かい位置で描くため)
    for color, (x, y) in layout.items():
        circle = np.stack([x / 1000 + r * np.cos(theta), y / 1000 + r * np.sin(theta), np.zeros_like(theta)], axis=1)  # 面の座標系での円の縁
        px, _ = cv2.projectPoints(circle, rvec, t_cv, K, None)  # 歪みなしで画素に投影(傾くと、楕円になる)
        poly = np.round(px.reshape(-1, 2) * (1 << SHIFT)).astype(np.int32)
        cv2.fillPoly(ideal, [poly], DOT_RGB[color], lineType=cv2.LINE_AA, shift=SHIFT)  # LINE_AA = 縁をなめらかに(混ざった色の画素ができる)
    distorted = cv2.remap(ideal, ideal_xy[:, :, 0], ideal_xy[:, :, 1], cv2.INTER_LINEAR)
    if noise > 0:
        rng = np.random.default_rng(seed)
        distorted = np.clip(distorted + rng.normal(0, noise, distorted.shape), 0, 255).astype(np.uint8)
    return distorted


def angle_between(Ra, Rb):
    """2 つの回転行列の間の角度 [度]。"""
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def evaluate(img, R_w, p_w, layout=DEFAULT_LAYOUT):
    """検出して、(横の誤差 mm, 奥行きの誤差 mm, 向きの誤差 度, 最小の面積 画素) を返す。4 色が見つからなければ None。"""
    res = detect_dot_marker(img, K, dist, layout)
    if res is None:
        return None
    R_est, p_est = marker_pose_in_world(res["R"], res["t"], cam_pos, cam_R)
    ray = (p_w - cam_pos) / np.linalg.norm(p_w - cam_pos)
    err = p_est - p_w
    e_depth = abs(err @ ray) * 1000
    e_side = np.linalg.norm(err - (err @ ray) * ray) * 1000  # 誤差から、奥行き方向の成分を引いた残り = 横方向
    return e_side, e_depth, angle_between(R_w, R_est), min(d["area"] for d in res["dots"].values())


places = {  # カメラからの距離が約 40・60・90 cm になる面の位置
    "約 40 cm": np.array([0.28, -0.08, 0.22]),
    "約 60 cm": np.array([0.20, 0.0, 0.0]),
    "約 90 cm": np.array([0.05, 0.10, -0.30]),
}
tilts = (0, 30, 50)
results = {}  # (直径, 距離の名前, 傾き) → evaluate の結果
print("    直径 / 距離 / 傾き → 横の誤差 mm, 奥行きの誤差 mm, 向きの誤差 °, 最小の塊の面積 画素(None = 4 色が見つからない)")
for diameter in (10, 5):
    for pname, p in places.items():
        for tilt in tilts:
            R_w = pose_facing_camera(p, tilt, 20)
            r = evaluate(render(R_w, p, diameter), R_w, p)
            results[(diameter, pname, tilt)] = r
            text = "None" if r is None else f"{r[0]:.2f}, {r[1]:.2f}, {r[2]:.2f}°, {r[3]}"
            print(f"    {diameter:2d} mm / {pname} / 傾き {tilt:2d}° → {text}")

ten = [v for (d, _, _), v in results.items() if d == 10]
check("(1) 10 mm: 9 通り全部で、4 色が見つかる", all(v is not None for v in ten))
if all(v is not None for v in ten):
    worst = [max(v[i] for v in ten) for i in range(3)]
    print(f"    10 mm の最悪値: 横 {worst[0]:.2f} mm、奥行き {worst[1]:.2f} mm、向き {worst[2]:.2f}°")
    # 基準は、先に決めたものではなく、上の実測に余裕を持たせた値(ArUco の check_aruco.py と、同じ考え方)。
    check("(2) 10 mm: 横 < 1.0 mm、奥行き < 6.0 mm、向き < 6.0°", worst[0] < 1.0 and worst[1] < 6.0 and worst[2] < 6.0)
five = {k: v for k, v in results.items() if k[0] == 5}
lost = [f"{k[1]}・傾き {k[2]}°" for k, v in five.items() if v is None]
check(f"(3) 5 mm: 4 色が見つかった場合の数 = {len(five) - len(lost)} / {len(five)}(見つからない: {', '.join(lost) if lost else 'なし'})。参考の結果。見つからない場合があっても、NG にはしない", True)
five_ok = [v for v in five.values() if v is not None]
if five_ok:
    print(f"    5 mm の最悪値(見つかった場合のみ): 横 {max(v[0] for v in five_ok):.2f} mm、奥行き {max(v[1] for v in five_ok):.2f} mm、向き {max(v[2] for v in five_ok):.2f}°、最小の面積 {min(v[3] for v in five_ok)} 画素")

# (4) 貼り位置の測り誤差: 各シールの位置が、実際より ±0.5 mm(乱数)ずれた値で解いたとき、姿勢の誤差はどうなるか(10 mm、60 cm、傾き 30°)
p, R_w = places["約 60 cm"], pose_facing_camera(places["約 60 cm"], 30, 20)
img = render(R_w, p, 10)
rng = np.random.default_rng(1)
sens = {}  # 測り誤差(標準偏差 mm) → (横, 奥行き, 向き) の、20 回の平均と最大
for sigma in (0.1, 0.2, 0.5):
    sides, depths, angles = [], [], []
    for _ in range(20):
        wrong = {c: (x + rng.normal(0, sigma), y + rng.normal(0, sigma)) for c, (x, y) in DEFAULT_LAYOUT.items()}  # 各シールの位置に、測り誤差を足す
        r = evaluate(img, R_w, p, wrong)
        sides.append(r[0]); depths.append(r[1]); angles.append(r[2])
    sens[sigma] = (np.mean(sides), max(sides), np.mean(depths), max(depths), np.mean(angles), max(angles))
    s = sens[sigma]
    print(f"    貼り位置の測り誤差 {sigma} mm(標準偏差)、20 回: 横 {s[0]:.2f}(最大 {s[1]:.2f}) mm、奥行き {s[2]:.2f}(最大 {s[3]:.2f}) mm、向き {s[4]:.1f}°(最大 {s[5]:.1f}°)")
# 測り誤差が 0.1 mm なら、10 mm のシールで、誤差は小さい。0.5 mm では、奥行きの誤差が 1 cm 級になる(上の表示。貼り位置を定規で測るだけでは足りない、という結果)。
# 基準は、先に決めたものではなく、この結果を見てから、「0.1 mm で、ほぼ使える」ことを確かめる形にした。0.2 mm 以上の結果は、参考として表示するだけ。
check("(4) 貼り位置を 0.1 mm で測れれば、横 < 1.0 mm、奥行き < 8 mm(平均)", sens[0.1][0] < 1.0 and sens[0.1][2] < 8.0)
check("(4) 測り誤差が大きいほど、奥行きの誤差が大きくなる(0.1 < 0.2 < 0.5 mm の順。平均)", sens[0.1][2] < sens[0.2][2] < sens[0.5][2])

print(f"\n{'全部 OK' if n_ng == 0 else f'NG が {n_ng} 件'}")
sys.exit(1 if n_ng else 0)
