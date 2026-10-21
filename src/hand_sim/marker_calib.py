"""【概要】机に固定した目印(シアンの球)でカメラを較正し、そのカメラで手の目印を見て、腕の関節のずれを測る。
流れ:
  1. 固定の目印(位置は、ロボットの座標で分かっている。腕に依存しない)を撮影 → カメラの位置・視野角を求める(calibrate_from_fixed_markers)。
  2. 較正したカメラで、手の目印を撮影する。手の目印が写る画素は、「実際の腕の姿勢」で決まる。
     「指令した関節角 + 関節ごとのずれ」で計算した位置を画素に直し、実際の画素との差が最小になる「ずれ」を探す(estimate_arm_offsets)。
     → 腕の関節のずれが、カメラで測れる(工程1 で、定規で測ったオフセットを、カメラで置き換えるもの)。
  3. 求めたずれを、制御側の補正(so101_grasp.set_arm_compensation。実機では sim_real.py の OFFSET_DEG)に使う。
カメラは真上から真下を向く前提(calibration.py と同じ)。固定の目印の見分けは、sim では「だいたいの位置からの最近傍」で行う(実機は、マーカーの ID で見分ける)。
結果の読み方は docs/marker_placement.md。
"""
import mujoco
import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares

from hand_sim.calibration import calibrate_camera, project
from hand_sim.so101_grasp import HOME_Q
from hand_sim.sorting import IMG_H, IMG_W, collect_hand_marker_samples, new_episode, random_layout
from hand_sim.vision import color_mask


def fixed_marker_positions(model):
    """固定の目印(geom の名前が fixed_marker_<番号>)の、世界座標 (N×3) を、番号順に返す。位置の「正解」は、XML に書いてある値。"""
    positions = []
    i = 0
    while True:
        try:
            positions.append(model.geom(f"fixed_marker_{i}").pos.copy())  # 世界に直接置いた geom なので、pos が世界座標
        except KeyError:  # 番号が尽きた(存在しない名前を引くと KeyError)
            break
        i += 1
    return np.array(positions)


def detect_fixed_marker_pixels(model, renderer, noise_px=0.0, rng=None):
    """固定の目印(シアンの塊)の画素 (N×2) を、見つけた順に返す。どの目印かの見分けは、ここではしない(match_markers で行う)。腕は、待機の姿勢(視野の外)。"""
    data = new_episode(model, {c: (0.2 + 0.03 * k, -0.02, 0.0) for k, c in enumerate(("red", "green", "blue"))})  # 立方体を置く(固定の目印の見え方に影響しない)
    renderer.update_scene(data, camera="overhead")
    labels, n = ndimage.label(color_mask(renderer.render(), "cyan"))
    pixels = []
    for k in range(1, n + 1):
        v_idx, u_idx = np.nonzero(labels == k)
        if len(v_idx) < 10:
            continue
        u, v = (u_idx.max() + u_idx.min()) / 2, (v_idx.max() + v_idx.min()) / 2
        if noise_px > 0:
            u, v = u + rng.normal(0, noise_px), v + rng.normal(0, noise_px)
        pixels.append((u, v))
    return np.array(pixels)


def match_markers(known, pixels, cam_pos, fovy_deg):
    """見つけた塊の画素 pixels を、分かっている目印の位置 known (N×3) に割り当てる。戻り値: 割り当てた (known の添字のリスト、pixels の添字のリスト)。

    「そのカメラ(cam_pos, fovy_deg)で見たときの、各目印の予想の画素」に一番近い塊を、各目印に割り当てる(1 つの塊は、1 つの目印にだけ)。
    カメラの値がずれていても、目印どうしの間隔(約 10 cm = 約 100 画素)より小さければ、取り違えない。実機は、マーカーの ID で見分ける。
    """
    expected = project(known, cam_pos, fovy_deg, IMG_W, IMG_H)
    dist = np.hypot(expected[:, None, 0] - pixels[None, :, 0], expected[:, None, 1] - pixels[None, :, 1])  # (目印の数 × 塊の数) の距離の表
    pairs = sorted((dist[i, j], i, j) for i in range(len(known)) for j in range(len(pixels)))  # 距離の近い順
    used_k, used_p, ki, pj = set(), set(), [], []
    for d, i, j in pairs:
        if d > 60 or i in used_k or j in used_p:  # 60 画素より遠い組は、割り当てない
            continue
        used_k.add(i)
        used_p.add(j)
        ki.append(i)
        pj.append(j)
    return ki, pj


def calibrate_from_fixed_markers(model, renderer, belief_pos, belief_fovy, noise_px=0.0, rng=None, position_err=None):
    """固定の目印で、カメラの位置と視野角を求める。戻り値: (位置、視野角 [度]、再投影誤差 [画素]、使えた目印の数)。

    見分けと較正を、交互に繰り返す(「信じているカメラ」が、本当の値からずれていても、最初の見分けに間違いが混ざることがあるため):
    信じているカメラで見分ける → 較正する → 求めたカメラで見分け直す → 較正し直す …(割り当てが変わらなくなるまで、最大 5 回)。
    position_err: (N×3) 目印の位置の測り間違い [m]。これを足した位置を「分かっている位置」として使う(板の位置を定規で測った誤差の再現)。
    """
    pixels = detect_fixed_marker_pixels(model, renderer, noise_px, rng)
    known = fixed_marker_positions(model)
    known_used = known if position_err is None else known + position_err
    pos, fovy = np.asarray(belief_pos, dtype=float), float(belief_fovy)
    last = None
    for _ in range(5):
        ki, pj = match_markers(known_used, pixels, pos, fovy)
        if (ki, pj) == last:
            break
        last = (ki, pj)
        pos, fovy, rms = calibrate_camera(known_used[ki], pixels[pj], IMG_W, IMG_H, pos, fovy)
    return pos, fovy, rms, len(ki)


def estimate_arm_offsets(model, renderer, cam_pos, fovy_deg, yaws_deg=(0.0,), noise_px=0.0, rng=None, fit_joints=(0, 1, 2, 3, 4)):
    """較正したカメラ (cam_pos, fovy_deg) で、手の目印を見て、腕の関節のずれ [度](長さ 5)を求める。

    観測: 手を、指を真下に向けて、いろいろな位置へ「指令」して撮影する(collect_hand_marker_samples)。実際の腕は、指令からずれた姿勢になっている。
    モデル: 「指令した関節角 q に、ずれ δ を足した姿勢」の、手の目印の位置を、順運動学で計算 → カメラで投影 → 画素。
            実際の画素との差が最小になる δ を、最小二乗法で探す(fit_joints で、求める関節を選べる。他の関節のずれは 0 とする)。
    戻り値: (δ [度、長さ 5]、再投影誤差 rms [画素]、使えた点の数)。ずれの定義は、「実際の関節角 = 指令 + δ」(sim_real.py の OFFSET_DEG と同じ向き)。
    """
    points, pixels, qs = collect_hand_marker_samples(model, renderer, noise_px, rng, yaws_deg=yaws_deg, with_q=True)
    data = mujoco.MjData(model)
    marker_id = model.site("hand_marker_site").id
    cam_pos = np.asarray(cam_pos, dtype=float)
    idx = list(fit_joints)

    def predicted_pixels(delta_rad):
        out = []
        for q in qs:
            data.qpos[:5] = np.asarray(q) + delta_rad
            mujoco.mj_forward(model, data)
            out.append(project(data.site_xpos[marker_id][None, :], cam_pos, fovy_deg, IMG_W, IMG_H)[0])
        return np.array(out)

    def residual(p):
        delta = np.zeros(5)
        delta[idx] = p
        return (predicted_pixels(delta) - pixels).ravel()

    sol = least_squares(residual, np.zeros(len(idx)))
    delta = np.zeros(5)
    delta[idx] = sol.x
    return np.degrees(delta), float(np.sqrt(np.mean(sol.fun ** 2))), len(qs)


def calibrate_system(model, renderer, belief_offset=(0.03, 0.0, 0.0), noise_px=0.0, board_err_mm=0.0, rng=None, yaws_deg=(0.0, 90.0)):
    """固定の目印でカメラを較正し、そのカメラで手の目印を見て腕のずれを測る(1 → 2 の流れ)。結果を辞書で返す。

    belief_offset: 「信じているカメラの位置」の、本当の位置からのずれ [m](較正の出発点)。
    noise_px: 画素に足すランダムなずれ(標準偏差、画素)。board_err_mm: 固定の目印の位置の誤差(標準偏差 [mm]。目印ごとにばらつく。板の位置を測った誤差の再現)。
    戻り値: {"cam_model": (位置, 視野角)、"arm_offsets_deg": 求めた腕のずれ(長さ 5)、"cam_err_mm": カメラの位置(x, y)の本当の値との差、"n_fixed": 使えた固定の目印の数、"n_hand": 使えた手の目印の数}。
    使うときは、so101_grasp.set_arm_compensation(arm_offsets_deg) で補正し、cam_model を、仕分け(run_episode)に渡す。
    """
    cam = model.camera("overhead")
    true_pos, true_fovy = np.array(cam.pos), float(model.cam_fovy[cam.id])
    board_err = rng.normal(0, board_err_mm / 1000, (len(fixed_marker_positions(model)), 3)) if board_err_mm > 0 else None
    pos, fovy, _, n_fixed = calibrate_from_fixed_markers(model, renderer, true_pos + np.asarray(belief_offset), true_fovy, noise_px, rng, board_err)
    delta, _, n_hand = estimate_arm_offsets(model, renderer, pos, fovy, yaws_deg=yaws_deg, noise_px=noise_px, rng=rng)
    return {"cam_model": (pos, fovy), "arm_offsets_deg": delta, "cam_err_mm": float(np.linalg.norm(pos[:2] - true_pos[:2]) * 1000),
            "n_fixed": n_fixed, "n_hand": n_hand}
