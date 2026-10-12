"""【概要】vision.py の2つの関数が正しいかを確認する。
赤い立方体を色々な位置に置き、overhead カメラで撮影 → find_red_centroid で画素位置を求める
→ pixel_to_world で世界座標(x, y)[m]に直し、本当の位置(MuJoCo が知っている値)と比べる。
立方体が画面の外にあるとき find_red_centroid が None を返すことも確認する。
全部通れば終了コード0、そうでなければ1。
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)。
sys.path.insert(0, str(ROOT / "src"))
# ★mujoco を import する前に書く(理由は check_install.py を参照)。
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from hand_sim.vision import find_red_centroid, pixel_to_world  # noqa: E402

W, H = 640, 480
TOL = 0.005  # 許容する誤差 [m] (5mm)
CUBE_TOP_Z = 0.03  # 立方体の上面の高さ [m] (一辺3cmで床に置いてあるので)

model = mujoco.MjModel.from_xml_path(str(ROOT / "models" / "so101" / "scene_vs.xml"))
data = mujoco.MjData(model)
cam = model.camera("overhead")

# 立方体の freejoint の qpos は「x, y, z, 回転(クォータニオン4個)」の7個。
# その先頭の番号を調べておき、そこへ直接 (x, y, z) を書き込んで立方体を置く。
cube_adr = model.joint("cube_free").qposadr[0]
# アームは邪魔にならないよう、たたんだ姿勢(home)に固定して写す。
HOME = [0, -1.7, 1.6, 1.0, 0, 0]

# 試す立方体の位置 (x, y) [m]。画面の中央付近から端まで。
PLACES = [(0.22, -0.12), (0.2, 0.0), (0.25, 0.12), (0.12, -0.2), (0.3, 0.2), (0.1, 0.15)]

n_ng = 0
try:
    with mujoco.Renderer(model, H, W) as renderer:
        def shoot(x, y):
            """立方体を (x, y) に置いて overhead カメラで撮影し、画像を返す。"""
            mujoco.mj_resetData(model, data)
            data.qpos[:6] = HOME
            data.qpos[cube_adr:cube_adr + 3] = [x, y, 0.015]
            mujoco.mj_forward(model, data)  # 時間は進めず、位置から見た目だけ更新する
            renderer.update_scene(data, camera="overhead")
            return renderer.render()

        for x, y in PLACES:
            img = shoot(x, y)
            uv = find_red_centroid(img)
            if uv is None:
                n_ng += 1
                print(f"NG  置いた位置=({x:+.3f}, {y:+.3f}) 赤い物体が見つからなかった")
                continue
            est = pixel_to_world(uv[0], uv[1], cam.pos, model.cam_fovy[cam.id], W, H, CUBE_TOP_Z)
            err = np.linalg.norm(np.asarray(est) - np.array([x, y]))
            ok = err < TOL
            n_ng += 0 if ok else 1
            print(f"{'OK' if ok else 'NG'}  置いた位置=({x:+.3f}, {y:+.3f})  "
                  f"画素=({uv[0]:6.1f}, {uv[1]:6.1f})  推定=({est[0]:+.3f}, {est[1]:+.3f})  誤差={err * 1000:.1f} mm")

        # 画面の外(カメラの視野の外)に置いたら None が返るはず。
        img = shoot(2.0, 2.0)
        uv = find_red_centroid(img)
        ok = uv is None
        n_ng += 0 if ok else 1
        print(f"{'OK' if ok else 'NG'}  画面の外に置いたとき None が返る (返った値: {uv})")
except NotImplementedError as e:
    print(f"{e}. src/hand_sim/vision.py に書いてください。")
    sys.exit(1)

print("全部OK" if n_ng == 0 else f"{n_ng} 件NGです")
sys.exit(0 if n_ng == 0 else 1)
