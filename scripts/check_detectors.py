"""【概要】検出器(画像 → 物体の位置)を差し替えても、視覚サーボが動くことを確認する(第2週-7)。
検出器の約束: detector(img) -> (u, v) または None(vision.py の冒頭を参照)。
3種類の検出器を、制御側のコードを一切変えずに差し替えて試す:
  A. 色検出(bbox)   : 今までの find_red_centroid。赤い画素すべての範囲の中心。
  B. 色検出(最大の塊): find_red_blob_centroid。一番大きなつながりだけを使う。
  C. 「物体検出モデル」のまね: 画像 → 枠 (x0, y0, x1, y1) を返す関数を bbox_to_center で包んだもの。
                    本物の YOLO などの代わりに、色で枠を作る関数で代用している(差し替えの形を見せるのが目的)。
試験1: 普通の画像で、3つとも立方体の位置を 3mm 以内で当てるか。
試験2: 画像の隅に小さな赤いごみを足す。A は大きく外れ、B と C は耐えるはず(差し替える意味の確認)。
試験3: closed-loop の視覚サーボ全体を、B の検出器で動かして(カメラ位置も 2cm ずらす)届くか。
全部通れば終了コード0、そうでなければ1。
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # 理由は check_install.py を参照
if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "egl")  # ★mujoco を import する前に設定(理由は check_install.py)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402
from hand_sim.vision import (  # noqa: E402
    bbox_to_center, find_blue_blob_centroid, find_blue_centroid, find_red_blob_centroid, find_red_centroid)
from hand_sim.visual_servo import (  # noqa: E402
    gripper_position, load_model_with_marker, locate_cube, servo_closed_loop)

W, H = 640, 480
HOVER_Z = 0.10
TOL = 0.003  # 3mm
HOME = [0, -1.7, 1.6, 1.0, 0, 0]
PLACES = [(0.22, -0.12), (0.25, 0.12), (0.3, -0.05), (0.15, 0.1)]


def fake_model_bbox(img):
    """「物体検出モデル」のまね。画像から、赤い物体の枠 (x0, y0, x1, y1) を返す(無ければ None)。
    本物のモデルなら、ここでモデルの推論を呼んで、一番自信のある枠を返す。
    ごみに強いモデルを想定して、一番大きな赤い塊だけの枠にしている。
    """
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    mask = (r > 100) & (g < 0.4 * r) & (b < 0.4 * r)
    labels, n = ndimage.label(mask)
    if n == 0:
        return None
    best = int(np.argmax(ndimage.sum(mask, labels, index=range(1, n + 1)))) + 1
    v_idx, u_idx = np.nonzero(labels == best)
    return (u_idx.min(), v_idx.min(), u_idx.max(), v_idx.max())


def with_distractor(detector):
    """検出器を包んで、画像の左上に小さな赤いごみ(4×4 画素)を足してから検出するようにする。
    関数を受け取って関数を返す書き方(デコレータの形)。[頻度] 検出器を差し替えるときの定番ではないが、テストでは便利
    """
    def wrapped(img):
        img = img.copy()  # 元の画像を壊さない
        img[10:14, 10:14] = [255, 0, 0]
        return detector(img)
    return wrapped


DETECTORS = {
    "A 色検出(bbox)": find_red_centroid,
    "B 色検出(最大の塊)": find_red_blob_centroid,
    "C モデルのまね(枠→中心)": lambda img: bbox_to_center(fake_model_bbox(img)),
}

model = load_model_with_marker(ROOT / "models" / "so101" / "scene_vs.xml")
data = mujoco.MjData(model)
cube_adr = model.joint("cube_free").qposadr[0]
true_cam = np.array(model.camera("overhead").pos)


def reset(x, y):
    mujoco.mj_resetData(model, data)
    data.qpos[:6] = HOME
    data.ctrl[:6] = HOME
    data.qpos[cube_adr:cube_adr + 3] = [x, y, 0.015]
    mujoco.mj_forward(model, data)


def vision_error(renderer, detector, x, y):
    """検出器で求めた立方体の位置と、本当の位置の距離 [m]。見つからなければ None。"""
    reset(x, y)
    xy = locate_cube(model, data, renderer, detector=detector)
    return None if xy is None else float(np.linalg.norm(np.asarray(xy) - np.array([x, y])))


n_ng = 0
with mujoco.Renderer(model, H, W) as renderer:
    print("--- 試験1: 普通の画像(視覚の誤差 [mm]) ---")
    for name, det in DETECTORS.items():
        errs = [vision_error(renderer, det, x, y) for x, y in PLACES]
        ok = all(e is not None and e < TOL for e in errs)
        n_ng += 0 if ok else 1
        print(f"{'OK' if ok else 'NG'}  {name:<18} " + "  ".join("なし" if e is None else f"{e * 1000:.1f}" for e in errs))

    print("\n--- 試験2: 画像の隅に赤いごみを足す(視覚の誤差 [mm]) ---")
    for name, det in DETECTORS.items():
        errs = [vision_error(renderer, with_distractor(det), x, y) for x, y in PLACES]
        good = all(e is not None and e < TOL for e in errs)
        bad = all(e is None or e > 0.02 for e in errs)  # 2cm 以上外れる(= ごみに引きずられる)
        expect_good = not name.startswith("A")  # A は外れるのが「想定どおり」
        ok = good if expect_good else bad
        n_ng += 0 if ok else 1
        note = "耐えた" if good else ("外れた(想定どおり)" if bad else "中途半端")
        print(f"{'OK' if ok else 'NG'}  {name:<18} " + "  ".join("なし" if e is None else f"{e * 1000:.1f}" for e in errs) + f"  → {note}")

    print("\n--- 試験3: closed-loop 全体を B の検出器で動かす(カメラ位置を横に 2cm ずらす) ---")
    assumed = true_cam + np.array([0.02, -0.015, 0])
    for x, y in PLACES:
        reset(x, y)
        servo_closed_loop(model, data, renderer, HOVER_Z, cam_pos=assumed,
                          cube_detector=find_red_blob_centroid, marker_detector=find_blue_blob_centroid)
        gap = float(np.linalg.norm(gripper_position(model, data)[:2] - data.qpos[cube_adr:cube_adr + 2]))
        ok = gap < TOL
        n_ng += 0 if ok else 1
        print(f"{'OK' if ok else 'NG'}  立方体=({x:+.2f}, {y:+.2f})  水平ずれ={gap * 1000:.1f} mm")

print("\n全部OK" if n_ng == 0 else f"\n{n_ng} 件NGです")
sys.exit(0 if n_ng == 0 else 1)
