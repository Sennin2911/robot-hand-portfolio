"""【概要】カメラを「任意の位置・向き」に置いたときの較正と、画素 → 世界座標の換算。
これまでの calibration.py は、カメラが「真上から真下を向く」前提で、位置と視野角だけを求めた。
ここでは、向き(傾き)も未知として、「位置 3 + 向き 3 + 視野角 1 = 7 つの値」を、手の目印の観測から最小二乗法で求める。
  - project_points: 世界座標の点が、画像のどの画素に写るか(カメラの位置・向き・視野角から計算)。
  - calibrate_camera_pose: 目印の世界座標と画素から、カメラの位置・向き・視野角を求める。
  - pixel_to_world_plane: 画素 → 世界座標。画素の方向へ伸ばした光線が、指定した高さの水平な面と交わる点を求める。
  - look_at_rotation: 「カメラの位置と、見つめる点」から、カメラの向き(回転行列)を作る(カメラを置く場所を決めるのに使う)。
カメラの座標系は MuJoCo と同じ: 見る向きが −z、画像の右が +x、画像の上が +y。回転行列 R は「カメラの座標系 → 世界座標系」(列 = カメラの各軸が世界でどちらを向くか)。
結果の読み方は docs/camera_placement.md を参照。レンズの歪みは扱わない(実機では、チェスボードの撮影で、別に求める)。
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation  # 3次元の回転を扱う道具。回転ベクトル・行列・クォータニオンを相互に変換できる。[頻度] ロボット・カメラの計算で定番


def look_at_rotation(cam_pos, target, up=(0.0, 0.0, 1.0)):
    """カメラが cam_pos にあり、target を見つめるときの、カメラの向き(3×3 の回転行列、カメラ → 世界)を返す。
    up: 世界で「上」と見なす向き。画像の上方向が、できるだけ up に近くなるようにする。"""
    forward = np.asarray(target, dtype=float) - np.asarray(cam_pos, dtype=float)
    forward /= np.linalg.norm(forward)  # /= : 長さ 1 にそろえる(割り算して上書き)。[頻度] よく使う
    z_axis = -forward  # カメラは自分の −z 方向を見る
    x_axis = np.cross(forward, up)  # 画像の右 = 前方向 × 上
    x_axis /= np.linalg.norm(x_axis)  # 真下を向くときは up と平行で、外積が 0 になる(その場合は、呼び出し側が up を変える)
    y_axis = np.cross(z_axis, x_axis)  # 画像の上
    return np.stack([x_axis, y_axis, z_axis], axis=1)


def project_points(points, cam_pos, cam_R, fovy_deg, width, height):
    """世界座標の点 (N×3) が、画像のどの画素 (u, v) に写るかを、(N×2) の配列で返す。カメラの後ろにある点は nan になる。"""
    p_cam = (np.asarray(points, dtype=float) - cam_pos) @ cam_R  # 世界 → カメラの座標系(R.T @ (p − c) を、行ごとに計算した形)
    depth = -p_cam[:, 2]  # カメラの前方への距離(カメラは −z を見る)
    f = (height / 2) / np.tan(np.radians(fovy_deg) / 2)  # 焦点距離 [画素]
    with np.errstate(divide="ignore", invalid="ignore"):  # 割り算の警告(depth が 0 や負)を出さない
        u = width / 2 + f * p_cam[:, 0] / depth
        v = height / 2 - f * p_cam[:, 1] / depth  # v は下へ増える。カメラの y は上へ増えるので、符号が逆
    u[depth <= 0] = np.nan
    v[depth <= 0] = np.nan
    return np.stack([u, v], axis=1)


def pixel_to_world_plane(u, v, cam_pos, cam_R, fovy_deg, width, height, z_plane):
    """画素 (u, v) の方向へ光線を伸ばし、高さ z_plane [m] の水平な面と交わる点の世界座標 (x, y, z) を返す。交わらなければ None。"""
    f = (height / 2) / np.tan(np.radians(fovy_deg) / 2)
    ray_cam = np.array([(u - width / 2) / f, -(v - height / 2) / f, -1.0])  # カメラの座標系での光線の向き
    ray = cam_R @ ray_cam  # 世界座標での向き
    if abs(ray[2]) < 1e-9:  # 水平に近すぎて、面と交わらない
        return None
    t = (z_plane - cam_pos[2]) / ray[2]  # カメラから、面までの、光線の長さ(の倍率)
    if t <= 0:  # 面が、カメラの後ろ側
        return None
    return np.asarray(cam_pos, dtype=float) + t * ray


def calibrate_camera_pose(points, pixels, width, height, init_pos, init_R, init_fovy_deg, fix_fovy=False, robust=True):
    """目印の世界座標 points (N×3) と画素 pixels (N×2) から、カメラの位置・向き・視野角を求める。

    init_pos, init_R, init_fovy_deg: 探索の出発点(だいたいの見当でよい。実機なら、目で見て決めた位置・向き)。
    向きは、出発点の回転 init_R に、小さな回転(回転ベクトル r = 軸 × 角度 [rad])を掛けた形で探す(r が 0 のとき、出発点のまま)。
    戻り値: (位置, 回転行列, 視野角 [度], 再投影誤差 rms [画素])。
    未知数は 7 つ(位置 3、回転ベクトル 3、視野角 1)。目印が 4 点以上(8 つの式)で解け、広い範囲・複数の高さに散らすほど安定する。
    fix_fovy: True なら、視野角を未知数にせず、init_fovy_deg のまま使う(未知数は 6 つ)。視野角は、カメラの内部の性質なので、
              チェスボードの撮影(内部パラメータの較正)で別に求めておけば、カメラが遠いときの「距離と視野角の見分けにくさ」を避けられる。
    robust: True なら、大きく外れた観測(目印が一部だけ隠れて、中心がずれた点など)の影響を小さくする(soft_l1 損失、2 画素を境にする)。
            ふつうの最小二乗法は、外れた 1 点に全体が引きずられる。外れ値は、目印が見える範囲が増えるほど(複数の向き・斜めのカメラで)出やすい。
    """
    pixels = np.asarray(pixels, dtype=float)
    R0 = Rotation.from_matrix(init_R)

    def unpack(p):
        fovy = init_fovy_deg if fix_fovy else p[6]
        return p[:3], (Rotation.from_rotvec(p[3:6]) * R0).as_matrix(), fovy  # Rotation * Rotation: 回転の合成

    def residual(p):
        pos, R, fovy = unpack(p)
        r = project_points(points, pos, R, fovy, width, height) - pixels
        return np.nan_to_num(r, nan=1e3).ravel()  # カメラの後ろに回った点には、大きな残差を与えて、そこへ行かせない

    p0 = np.array([*init_pos, 0.0, 0.0, 0.0, init_fovy_deg], dtype=float)
    sol = least_squares(residual, p0, loss="soft_l1" if robust else "linear", f_scale=2.0)  # f_scale: 残差がこの大きさ(画素)を超えたら、影響を弱める
    pos, R, fovy = unpack(sol.x)
    rms = float(np.sqrt(np.median(residual(sol.x) ** 2))) if robust else float(np.sqrt(np.mean(sol.fun ** 2)))  # robust のときは、外れ値に左右されない中央値で、誤差の大きさを出す
    return pos, R, float(fovy), rms


def rotation_angle_deg(R_a, R_b):
    """2 つの回転行列の間の角度 [度]。向きの誤差の大きさを測る。"""
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(R_a.T @ R_b).as_rotvec())))
