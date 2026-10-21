"""【概要】intrinsics.py(チェスボードからのカメラ内部パラメータの較正)が正しいかを、合成データで確認する。
(1) scripts/make_chessboard.py が作った画像から、内側の角 9 × 6 個が見つかる。
(2) 本当の内部パラメータ(焦点距離・歪み)を決めておき、盤を色々な位置・傾きに置いたときの角の画素を、計算で作って(ノイズ 0.3 画素)、較正が、本当の値を取り戻す。
    → 実機で、盤を 20 枚ほど撮ったときの、較正の精度の目安にもなる。
全部通れば終了コード0、そうでなければ1。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)。
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from hand_sim.intrinsics import calibrate_intrinsics, find_corners, object_points  # noqa: E402

n_ng = 0


def check(label, ok):
    """結果を 1 行表示し、失敗なら数える。global: 関数の外の変数 n_ng を書き換える宣言。[頻度] 小さなスクリプトでだけ使う"""
    global n_ng
    print(f"{'OK' if ok else 'NG'}  {label}")
    n_ng += 0 if ok else 1


# (1) 印刷用の画像から、角が見つかる(scripts/make_chessboard.py が作る PNG。無ければ、このチェックは飛ばす)
png = ROOT / "outputs" / "chessboard_A4.png"
if png.exists():
    img = cv2.cvtColor(cv2.imread(str(png)), cv2.COLOR_BGR2RGB)
    corners = find_corners(img)
    check(f"印刷用のチェスボードの画像から、角が見つかる(54 個。見つかった数 {0 if corners is None else len(corners)})", corners is not None and len(corners) == 54)
else:
    print("(スキップ) outputs/chessboard_A4.png が無い。先に scripts/make_chessboard.py を実行する")

# (2) 合成データでの較正
W, H = 1920, 1080
K_true = np.array([[1100.0, 0, 970.0], [0, 1090.0, 535.0], [0, 0, 1]])  # 焦点距離約 1100 画素(横の視野角 約 80°)、中心は、画像の中心から少しずれる
dist_true = np.array([-0.30, 0.12, 0.001, -0.0005, -0.03])  # k1, k2, p1, p2, k3。広角レンズによくある、樽型の歪み(k1 が負)
rng = np.random.default_rng(0)
obj = object_points()
corner_list = []
while len(corner_list) < 20:
    # 盤を、カメラの前 0.4〜0.9 m に、傾き(±35°)をつけて置き、画像の中に収まる姿勢だけ使う
    rvec = rng.uniform(-0.6, 0.6, 3)
    tvec = np.array([rng.uniform(-0.15, 0.15), rng.uniform(-0.08, 0.08), rng.uniform(0.4, 0.9)])
    pts, _ = cv2.projectPoints(obj, rvec, tvec, K_true, dist_true)
    pts = pts.reshape(-1, 2)
    if pts[:, 0].min() < 20 or pts[:, 0].max() > W - 20 or pts[:, 1].min() < 20 or pts[:, 1].max() > H - 20:
        continue
    corner_list.append((pts + rng.normal(0, 0.3, pts.shape)).astype(np.float32).reshape(-1, 1, 2))  # 角の検出誤差を想定したノイズ
result = calibrate_intrinsics(corner_list, (W, H))
K, dist = result["K"], result["dist"]
check(f"焦点距離 fx = {K[0, 0]:.1f}(本当の値 1100.0、差 1% 未満)", abs(K[0, 0] - 1100) / 1100 < 0.01)
check(f"焦点距離 fy = {K[1, 1]:.1f}(本当の値 1090.0、差 1% 未満)", abs(K[1, 1] - 1090) / 1090 < 0.01)
check(f"画像の中心 cx, cy = ({K[0, 2]:.1f}, {K[1, 2]:.1f})(本当の値 970, 535、差 5 画素未満)", abs(K[0, 2] - 970) < 5 and abs(K[1, 2] - 535) < 5)
check(f"歪み k1 = {dist[0]:.3f}(本当の値 -0.300、差 0.03 未満)", abs(dist[0] - (-0.30)) < 0.03)
check(f"再投影誤差 {result['rms']:.2f} 画素(ノイズ 0.3 画素なので、0.5 未満)", result["rms"] < 0.5)
print(f"  縦の視野角 {result['fovy_deg']:.1f}°、横の視野角 {result['fovx_deg']:.1f}°")

print("全部OK" if n_ng == 0 else f"NG が {n_ng} 件")
sys.exit(0 if n_ng == 0 else 1)
