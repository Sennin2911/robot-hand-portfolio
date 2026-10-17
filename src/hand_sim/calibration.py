"""【概要】カメラの較正(キャリブレーション): 「カメラの位置と視野角」を、観測データから求める(第2週-6)。
pixel_to_world は「カメラの位置が分かっている」前提で、画素を世界座標に直す。実機ではその位置を測って求める必要がある。
ここでは、次の方法で求める(固定カメラ = eye-to-hand の、位置が既知の目印を使う定番の手順)。
  1. 腕を色々な位置に動かし、手先の青い目印を撮影する。
  2. 各回で、目印の世界座標は「関節角からの順運動学」で分かる(カメラを使わない = カメラの位置に依存しない値)。
  3. 同じ目印の画素位置 (u, v) も測れる。
  4. 「カメラの位置・視野角をこう仮定すると、世界座標の点は画素のどこに写るはずか」を計算し、
     実際の画素とのずれ(再投影誤差)が最小になる位置・視野角を、最小二乗法で探す。
カメラの向きは「真下を向き、画像の右 = 世界の +x、上 = +y」と分かっている前提(scene_vs.xml と同じ)。
向きも未知の場合は、より一般的な手法(OpenCV の solvePnP など)になる(docs/calibration.md を参照)。
"""
import numpy as np
from scipy.optimize import least_squares


def project(points, cam_pos, fovy_deg, width, height):
    """世界座標の点 (N×3) が、画像のどの画素 (u, v) に写るかを計算する(pixel_to_world の逆向きの計算)。

    points : 世界座標 (x, y, z) [m] を N 個並べた配列 (N, 3)。
    戻り値 : (N, 2) の配列。各行が (u, v) [画素]。
    """
    points = np.asarray(points, dtype=float)
    f = (height / 2) / np.tan(np.radians(fovy_deg) / 2)  # 焦点距離 [画素]
    depth = cam_pos[2] - points[:, 2]  # カメラから各点までの距離(真下を向いているので高さの差)
    u = width / 2 + f * (points[:, 0] - cam_pos[0]) / depth
    v = height / 2 - f * (points[:, 1] - cam_pos[1]) / depth  # v は下へ増える。世界の y は上へ増えるので符号が逆
    return np.stack([u, v], axis=1)  # np.stack: 配列を新しい軸で重ねる。[頻度] よく使う


def calibrate_camera(points, pixels, width, height, init_cam_pos, init_fovy_deg):
    """目印の世界座標 points (N×3) と画素位置 pixels (N×2) から、カメラの位置と視野角を求める。

    init_cam_pos, init_fovy_deg : 探索の出発点(だいたいの見当でよい)。
    戻り値: (cam_pos, fovy_deg, rms) rms = 再投影誤差の二乗平均平方根 [画素](小さいほど観測とよく合う)。

    scipy.optimize.least_squares: 「残差(ずれ)の二乗和を最小にする未知数」を反復で探す。[頻度] 較正・フィッティングの定番。
    未知数は p = (カメラ x, y, z, 視野角) の4つ。
    """
    pixels = np.asarray(pixels, dtype=float)

    def residual(p):
        # 仮定した p での画素位置 - 実際に測った画素位置。これを全点・u と v 並べて1本の配列にする(ravel)。
        return (project(points, p[:3], p[3], width, height) - pixels).ravel()

    p0 = np.array([*init_cam_pos, init_fovy_deg], dtype=float)  # *: リストの中身を展開して並べる
    sol = least_squares(residual, p0)
    rms = float(np.sqrt(np.mean(sol.fun ** 2)))
    return sol.x[:3], float(sol.x[3]), rms
