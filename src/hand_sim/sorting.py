"""【概要】部品仕分け(新計画の工程3)。机の上にランダムに置いた 3 色の立方体を、真上のカメラで見つけて、
色ごとの置き場へ運ぶ。1 回ごとの結果と失敗の種類を記録して、成功率と失敗の内訳を出せるようにする。
流れ(1 個ぶん): 待機の姿勢に戻る → 撮影して 3 色の立方体の位置と向きを出す → so101_grasp.pick_and_place で掴んで運ぶ
              → 結果を分類(成功 / 未検出 / 到達不可 / 把持失敗 / 運搬中の落下 / 置き場の外)。掴めなかったら、もう 1 回だけやり直す。
カメラ(vision.py の色検出)と、掴む動作(so101_grasp.py)をつなぐだけのファイル。シーンは models/so101/scene_sort.xml。
詳しい考え方と結果の読み方は docs/sorting_trials.md を参照。単位は m(長さ)、rad(角度)。
"""
import mujoco
import numpy as np

from hand_sim.so101_grasp import GRIP_OPEN, HOME_Q, go_home, load_grasp_model, pick_and_place, set_cube_pose
from hand_sim.visual_servo import CUBE_TOP_Z
from hand_sim.vision import find_cube_pose, pixel_to_world

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
    return load_grasp_model(xml_path, cube_geoms=tuple(f"cube_{c}_geom" for c in COLORS))


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


def detect_cubes(model, data, renderer, cam_err=(0.0, 0.0), noise_px=0.0, rng=None):
    """真上のカメラで撮影して、3 色の立方体の {色: (x, y, yaw[rad])} を返す。見つからない色は、値が None。

    cam_err : プログラムが信じているカメラの位置の、本当の位置からのずれ (dx, dy) [m]。実機ではカメラの位置を測って求めるので、ずれる。
              ずれた分だけ、見つけた位置が全部まとめてずれる(較正の誤差を再現する)。
    noise_px: 画素位置にわざと足すランダムなずれの大きさ(標準偏差、画素)。画像のぼやけ・影・光の変化の代わり。
    """
    renderer.update_scene(data, camera="overhead")
    img = renderer.render()
    cam = model.camera("overhead")
    cam_pos = np.array(cam.pos) + [cam_err[0], cam_err[1], 0.0]  # 「信じているカメラの位置」
    fovy = model.cam_fovy[cam.id]
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


def run_attempt(model, data, renderer, color, cam_err=(0.0, 0.0), noise_px=0.0, rng=None, on_step=None, on_detect=None):
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

    det = detect_cubes(model, data, renderer, cam_err, noise_px, rng)[color]
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
            if log["lifted"][2] < 0.05:
                rec["fail_class"] = "把持失敗"
            elif not carry_ok:
                rec["fail_class"] = "運搬中の落下"
            elif not in_zone:
                rec["fail_class"] = "置き場の外"
            else:
                rec["success"] = True
    rec["moved_others_mm"] = max([float(np.linalg.norm(true_cube_pose(model, data, c)[0] - p) * 1000) for c, p in others_before.items()] + [0.0])
    rec["cycle_s"] = data.time - t0
    return rec


def run_episode(model, renderer, rng, cam_err=(0.0, 0.0), noise_px=0.0, retries=1, on_step=None, on_detect=None, on_record=None, data=None):
    """3 色の立方体をランダムに置いて、全部を置き場へ運ぶ(1 エピソード)。記録の辞書のリストを返す。

    掴めなかった(成功でない)立方体は、最大 retries 回だけやり直す(撮り直して、やり直す)。未検出と到達不可は、やり直しても同じなのでやり直さない。
    各記録には、エピソードの番号は付いていない(呼び出し側で足す)。attempt が何回目か、final_success が最終的に成功したかを付ける。
    on_step、on_detect: run_attempt と同じ(ビューアの表示用)。on_record: 1 回の試行が終わるたびに、その記録を渡して呼ぶ関数(途中経過の表示用)。
    data: 立方体を置いた状態の data(new_episode で作ったもの)。None なら、ここでランダムな配置を作る。
          ビューアは、開く時点で data が要るので、呼び出し側で作って渡せるようにしてある。
    """
    if data is None:
        data = new_episode(model, random_layout(rng))
    records = []
    for color in rng.permutation(COLORS):  # rng.permutation: 順番をランダムに並べ替える(置く順番もばらつかせる)
        color = str(color)
        for attempt in range(1, retries + 2):
            rec = run_attempt(model, data, renderer, color, cam_err, noise_px, rng, on_step, on_detect)
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
