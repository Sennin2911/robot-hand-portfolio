"""【概要】部品仕分け(新計画の工程3)。机の上にランダムに置いた 3 色の立方体を、真上のカメラで見つけて、
色ごとの置き場へ運ぶ。1 回ごとの結果と失敗の種類を記録して、成功率と失敗の内訳を出せるようにする。
流れ(1 個ぶん): 待機の姿勢に戻る → 撮影して 3 色の立方体の位置と向きを出す → so101_grasp.pick_and_place で掴んで運ぶ
              → 結果を分類(成功 / 未検出 / 到達不可 / 把持失敗 / 運搬中の落下 / 置き場の外)。掴めなかったら、もう 1 回だけやり直す。
カメラ(vision.py の色検出)と、掴む動作(so101_grasp.py)をつなぐだけのファイル。シーンは models/so101/scene_sort.xml。
詳しい考え方と結果の読み方は docs/sorting_trials.md を参照。単位は m(長さ)、rad(角度)。
"""
import mujoco
import numpy as np

from hand_sim.calibration import calibrate_camera
from hand_sim.so101_grasp import (
    GRIP_OPEN, HOME_Q, go_home, grasp_rotation, load_grasp_model, pick_and_place, set_cube_pose,
)
from hand_sim.so101_ik import solve_ik_pose
from hand_sim.visual_servo import CUBE_TOP_Z
from hand_sim.vision import _largest_component, color_mask, find_cube_pose, pixel_to_world

COLORS = ("red", "green", "blue")
# 置き場の中心 (x, y) [m] と、板の半分の大きさ。scene_sort.xml の pad_* と同じ値にする(食い違うと、成功の判定がずれる)。
ZONES = {"red": (0.07, 0.17), "green": (0.14, 0.17), "blue": (0.21, 0.17)}
ZONE_HALF = 0.03
CUBE_Z = 0.015  # 立方体の中心の高さ(机の上に置いたとき)[m]

# 立方体をランダムに置く範囲 (x, y) [m]。置き場(y ≥ 0.14)から離す。腕の届く範囲(土台から約 0.27 m)に収める。
X_RANGE, Y_RANGE, R_MAX = (0.14, 0.26), (-0.14, 0.06), 0.27
MIN_GAP = 0.06  # 立方体の中心どうしの最小の距離 [m]。近すぎると、指が隣の立方体に当たる

IMG_W, IMG_H = 640, 480  # 撮影する画像の大きさ [画素]
MOVED_MM = 5.0  # 他の立方体が、これ以上(mm)動いたら「他の立方体を動かした」と記録する
FAIL_CLASSES = ("未検出", "到達不可", "把持失敗", "運搬中の落下", "置き場の外")


def load_sort_model(xml_path):
    """仕分け用のシーンを読み込み、指の当たり判定を掴む用に作り直した model を返す(so101_grasp.load_grasp_model を使う)。"""
    return load_grasp_model(xml_path, cube_geoms=tuple(f"cube_{c}_geom" for c in COLORS), hand_marker=True)


def random_layout(rng):
    """3 色の立方体の、ランダムな配置 {色: (x, y, yaw[rad])} を返す。範囲内で、互いに MIN_GAP 以上離す。
    rng: numpy の乱数生成器(np.random.default_rng(種))。種を決めておけば、同じ配置を再現できる。"""
    placed = []
    while len(placed) < len(COLORS):
        x, y = rng.uniform(*X_RANGE), rng.uniform(*Y_RANGE)  # 範囲の中の一様な乱数
        if np.hypot(x, y) > R_MAX:  # 腕が届かない
            continue
        if any(np.hypot(x - px, y - py) < MIN_GAP for px, py, _ in placed):  # any(...): 1 つでも True なら True。[頻度] 定番
            continue
        placed.append((x, y, rng.uniform(0, np.pi / 2)))  # 向きは 0〜90°(正方形は 90° ごとに同じ形)
    return dict(zip(COLORS, placed))  # dict(zip(キー, 値)): 2 つの並びから辞書を作る。[頻度] 定番


def new_episode(model, layout):
    """待機の姿勢の腕と、layout の立方体で、新しい data を作って返す。"""
    data = mujoco.MjData(model)
    data.qpos[:5] = HOME_Q  # 腕を待機の姿勢から始める(qpos は今の関節角、ctrl は目標角。両方に入れると、姿勢がそのまま保たれる)
    data.ctrl[:5] = HOME_Q
    data.qpos[5] = data.ctrl[5] = GRIP_OPEN
    for color, (x, y, yaw) in layout.items():
        set_cube_pose(model, data, [x, y, CUBE_Z], yaw, cube=f"cube_{color}")
    return data


def true_cube_pose(model, data, color):
    """立方体の本当の位置 (x, y) [m] と向き yaw [rad、0〜90° 未満](sim の正解。認識の誤差を測るためだけに使う)。"""
    body = model.body(f"cube_{color}")
    R = data.xmat[body.id].reshape(3, 3)
    yaw = np.arctan2(R[1, 0], R[0, 0]) % (np.pi / 2)  # 回転行列から z 軸まわりの角度を出す。% (π/2): 90° ごとに同じ形なので、0〜90° にそろえる
    return data.xpos[body.id][:2].copy(), yaw


def yaw_error_deg(a, b):
    """向き a, b [rad] の差の大きさ [度、0〜45]。90° ごとに同じ形なので、差は 90° の周期で測る。"""
    d = (a - b) % (np.pi / 2)
    return float(np.degrees(min(d, np.pi / 2 - d)))


def detect_cubes(model, data, renderer, cam_err=(0.0, 0.0), noise_px=0.0, rng=None, cam_model=None):
    """真上のカメラで撮影して、3 色の立方体の {色: (x, y, yaw[rad])} を返す。見つからない色は、値が None。

    cam_err : プログラムが信じているカメラの位置の、本当の位置からのずれ (dx, dy) [m]。実機ではカメラの位置を測って求めるので、ずれる。
              ずれた分だけ、見つけた位置が全部まとめてずれる(較正の誤差を再現する)。
    noise_px: 画素位置にわざと足すランダムなずれの大きさ(標準偏差、画素)。画像のぼやけ・影・光の変化の代わり。
    cam_model: (カメラの位置 (x, y, z)、視野角 [度])。較正で求めた値を渡すと、cam_err の代わりに、これを「信じているカメラ」として使う。
    """
    renderer.update_scene(data, camera="overhead")
    img = renderer.render()
    cam = model.camera("overhead")
    cam_pos = np.array(cam.pos) + [cam_err[0], cam_err[1], 0.0]  # 「信じているカメラの位置」
    fovy = model.cam_fovy[cam.id]
    if cam_model is not None:  # 較正で求めた値を使う
        cam_pos, fovy = np.asarray(cam_model[0], dtype=float), float(cam_model[1])
    found = {}
    for color in COLORS:
        pose = find_cube_pose(img, color)
        if pose is None:
            found[color] = None
            continue
        u, v, yaw_deg = pose
        if noise_px > 0:
            u, v = u + rng.normal(0, noise_px), v + rng.normal(0, noise_px)  # rng.normal(平均, 標準偏差): 正規分布の乱数
        x, y = pixel_to_world(u, v, cam_pos, fovy, IMG_W, IMG_H, CUBE_TOP_Z)  # 画素 → 世界座標(上面の高さで換算)
        found[color] = (x, y, np.radians(yaw_deg))
    return found


def run_attempt(model, data, renderer, color, cam_err=(0.0, 0.0), noise_px=0.0, rng=None, on_step=None, on_detect=None, cam_model=None):
    """color の立方体を、1 回だけ、見つけて → 掴んで → 置き場へ運ぶ。結果の辞書(記録 1 行ぶん)を返す。

    記録: 成功したか(success)、失敗の種類(fail_class。成功なら "")、認識の誤差(det_err_mm、det_yaw_err_deg)、
          最終の位置の置き場の中心からのずれ(final_err_mm)、他の立方体を動かした量(moved_others_mm)、所要時間(cycle_s、sim の時間)。
    on_step  : シミュレーションの 1 ステップごとに呼ぶ関数(ビューアの表示用)。None なら何もしない。
    on_detect: 撮影して位置を出した直後に、on_detect(color, 見つけた (x, y, yaw) または None, 本当の (x, y)) を呼ぶ(ビューアで見つけた位置を示す用)。
    """
    t0 = data.time
    go_home(model, data, 2.0, on_step)  # 腕を視野の外へ。立方体を隠さない
    others_before = {c: true_cube_pose(model, data, c)[0] for c in COLORS if c != color}
    true_xy, true_yaw = true_cube_pose(model, data, color)
    rec = {"color": color, "true_x": true_xy[0], "true_y": true_xy[1], "true_yaw_deg": np.degrees(true_yaw),
           "success": False, "fail_class": "", "det_err_mm": np.nan, "det_yaw_err_deg": np.nan,
           "final_err_mm": np.nan, "moved_others_mm": 0.0, "cycle_s": 0.0}

    det = detect_cubes(model, data, renderer, cam_err, noise_px, rng, cam_model)[color]
    if on_detect is not None:
        on_detect(color, det, true_xy)
    if det is None:
        rec["fail_class"] = "未検出"
    else:
        x, y, yaw = det
        rec["det_err_mm"] = float(np.linalg.norm(np.array([x, y]) - true_xy) * 1000)
        rec["det_yaw_err_deg"] = yaw_error_deg(yaw, true_yaw)
        try:
            log = pick_and_place(model, data, [x, y, CUBE_Z], ZONES[color], yaw, on_step=on_step, cube=f"cube_{color}")
        except ValueError:  # so101_grasp.plan_pick_place: どの向きでも IK が解けない
            rec["fail_class"] = "到達不可"
        else:
            zone = np.array(ZONES[color])
            final_xy, _ = true_cube_pose(model, data, color)
            rec["final_err_mm"] = float(np.linalg.norm(final_xy - zone) * 1000)
            carry_ok = log["carried"][2] > 0.05 and np.linalg.norm(log["carried"][:2] - zone) < 0.03
            in_zone = bool(np.all(np.abs(final_xy - zone) < ZONE_HALF)) and data.xpos[model.body(f"cube_{color}").id][2] < 0.02
            # 成功の判定は「立方体が、自分の色の置き場に入ったか」(in_zone)を、最初に見る。持ち上げた高さの基準(5 cm)だけで失敗にすると、
            # 指が立方体の低い所を掴んで、4.8 cm までしか上がらなくても、置き場まで運べた(= 掴めた)場合を、失敗にしてしまった。
            if in_zone:
                rec["success"] = True
            elif log["lifted"][2] < 0.05:
                rec["fail_class"] = "把持失敗"
            elif not carry_ok:
                rec["fail_class"] = "運搬中の落下"
            else:
                rec["fail_class"] = "置き場の外"
    rec["moved_others_mm"] = max([float(np.linalg.norm(true_cube_pose(model, data, c)[0] - p) * 1000) for c, p in others_before.items()] + [0.0])
    rec["cycle_s"] = data.time - t0
    return rec


def run_episode(model, renderer, rng, cam_err=(0.0, 0.0), noise_px=0.0, retries=1, on_step=None, on_detect=None, on_record=None, data=None, cam_model=None):
    """3 色の立方体をランダムに置いて、全部を置き場へ運ぶ(1 エピソード)。記録の辞書のリストを返す。

    掴めなかった(成功でない)立方体は、最大 retries 回だけやり直す(撮り直して、やり直す)。未検出と到達不可は、やり直しても同じなのでやり直さない。
    各記録には、エピソードの番号は付いていない(呼び出し側で足す)。attempt が何回目か、final_success が最終的に成功したかを付ける。
    on_step、on_detect: run_attempt と同じ(ビューアの表示用)。on_record: 1 回の試行が終わるたびに、その記録を渡して呼ぶ関数(途中経過の表示用)。
    data: 立方体を置いた状態の data(new_episode で作ったもの)。None なら、ここでランダムな配置を作る。
          ビューアは、開く時点で data が要るので、呼び出し側で作って渡せるようにしてある。
    cam_model: 較正で求めた (カメラの位置、視野角)。渡すと、cam_err の代わりに使う(detect_cubes を参照)。
    """
    if data is None:
        data = new_episode(model, random_layout(rng))
    records = []
    for color in rng.permutation(COLORS):  # rng.permutation: 順番をランダムに並べ替える(置く順番もばらつかせる)
        color = str(color)
        for attempt in range(1, retries + 2):
            rec = run_attempt(model, data, renderer, color, cam_err, noise_px, rng, on_step, on_detect, cam_model)
            rec["attempt"] = attempt
            if on_record is not None:
                on_record(rec)
            records.append(rec)
            if rec["success"] or rec["fail_class"] in ("未検出", "到達不可"):
                break
        final = records[-1]["success"]
        for r in records:
            if r["color"] == color:
                r["final_success"] = final  # この色が、やり直しも含めて最終的に成功したか
    return records


# ---- カメラの較正(手の目印を使う)----
# 較正のために手を動かす位置 (x, y) [m]: 立方体を置く範囲の全体に、高さ(下の CALIB_Z)を 2 通り(目印の高さの違いが、カメラの高さと視野角を分ける手がかりになる)。
CALIB_XY = [(0.14, -0.12), (0.14, 0.04), (0.20, -0.12), (0.20, 0.0), (0.20, 0.06), (0.26, -0.10), (0.26, 0.0), (0.24, 0.06)]
CALIB_Z = (0.03, 0.08)  # 手先(gripperframe)の高さ。これより高いと、手を真下に向けたままでは、手首が可動域の端に届かず、IK が解けない


def collect_hand_marker_samples(model, renderer, noise_px=0.0, rng=None, marker_pos_err=(0.0, 0.0, 0.0), camera="overhead", yaws_deg=(0.0,), with_q=False):
    """手の目印の観測を集める: 手を CALIB_XY × CALIB_Z の位置へ、指を真下に向けて動かし、各位置で撮影する。

    戻り値: (points, pixels)。points = 目印の世界座標 (N×3、関節角から計算。カメラに依存しない)、pixels = 目印の画素 (N×2)。
    目印が、腕の陰などで見えない位置は飛ばす。腕は physics で動かさず、qpos を直接書いて姿勢を作る(較正に必要なのは、姿勢が分かることだけ)。
    noise_px: 画素に足すランダムなずれ(標準偏差)。marker_pos_err: 目印の世界座標に足す一定のずれ [m](測り間違いの再現)。
    camera: 撮影するカメラの名前(既定は真上のカメラ。置き場所の調査では、同じ名前のカメラを、別の位置・向きに置き直して使う)。
    yaws_deg: 手の向き(指が開閉する向きの角度、0° は世界の +x 向き)の候補。目印は手の片側に付いているので、カメラの反対側を向くと、手の陰に隠れる。
              複数の向きで撮ると、どちら側のカメラからも、目印が見える位置が増える(wrist_roll の可動域の都合で、150° 程度まで)。
    """
    data = mujoco.MjData(model)
    marker_id = model.site("hand_marker_site").id
    import hand_sim.so101_grasp as grasp  # ARM_ERROR(試験用の腕のずれ)の「今の値」を読むため。from import だと、設定前の値が固定される
    points, pixels, qs = [], [], []
    q = np.array(HOME_Q, dtype=float)
    for yaw_deg, (x, y), z in [(yaw, xy, z) for yaw in yaws_deg for z in CALIB_Z for xy in CALIB_XY]:
        R = grasp_rotation(np.radians(yaw_deg))  # 手を真下に向けた向き(指の開閉の向きが yaw_deg)
        q, ok = solve_ik_pose(model, data, [x, y, z], R, q)
        if not ok:
            continue
        data.qpos[:5] = q
        mujoco.mj_forward(model, data)
        marker_world = data.site_xpos[marker_id].copy()  # 目印の世界座標は、「指令した関節角 q」から計算する(ロボットが信じている位置)
        data.qpos[:5] = q + grasp.ARM_ERROR  # 実際の腕は、指令からずれた姿勢になる(通常は 0 なので、同じ)。カメラが見るのは、実際の姿勢
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        mask = _largest_component(color_mask(renderer.render(), "magenta"))
        if mask is None or mask.sum() < 10:  # 目印が見えない点は飛ばす
            continue
        v_idx, u_idx = np.nonzero(mask)
        u, v = (u_idx.max() + u_idx.min()) / 2, (v_idx.max() + v_idx.min()) / 2
        if noise_px > 0:
            u, v = u + rng.normal(0, noise_px), v + rng.normal(0, noise_px)
        points.append(marker_world + marker_pos_err)  # 指令した関節角から計算した目印の世界座標。+ で新しい配列ができる
        pixels.append((u, v))
        qs.append(q.copy())
    return (np.array(points), np.array(pixels), qs) if with_q else (np.array(points), np.array(pixels))


def calibrate_overhead_camera(model, renderer, belief_pos, belief_fovy, noise_px=0.0, rng=None, marker_pos_err=(0.0, 0.0, 0.0)):
    """手の目印を使って、真上のカメラの位置と視野角を求める。(カメラの位置の「信じている値」は、出発点としてだけ使う。)

    手順(calibration.py の説明と同じ):
      1. 手を、CALIB_XY × CALIB_Z の位置へ、指を真下に向けて動かす(関節角から、手の目印の世界座標が計算で分かる)。
      2. その状態を撮影して、マゼンタの目印が画像のどの画素に写るかを測る(collect_hand_marker_samples)。
      3. 「カメラがこの位置・視野角なら、目印の世界座標は、この画素に写るはず」の計算と、実際の画素の差が最小になる位置・視野角を探す。
    戻り値: (カメラの位置 (x, y, z), 視野角 [度], 再投影誤差 rms [画素], 使えた点の数)。
    belief_pos, belief_fovy: 探索の出発点(だいたいの見当。実機なら、定規で測ったカメラの位置)。
    noise_px、marker_pos_err: collect_hand_marker_samples と同じ。
    """
    points, pixels = collect_hand_marker_samples(model, renderer, noise_px, rng, marker_pos_err)
    cam_pos, fovy, rms = calibrate_camera(points, pixels, IMG_W, IMG_H, belief_pos, belief_fovy)
    return cam_pos, fovy, rms, len(points)
