"""【概要】SO-101 のモデル (models/so101/scene.xml) を読み込み、関節とアクチュエータの
一覧を表示し、いくつかの姿勢を画像 (outputs/so101_poses.png) に保存する。
どの関節が何番で、動く範囲がどれくらいかを、実機を買う前に把握するための確認用。
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# 他のスクリプトと同じ形(詳しい意味は check_install.py を参照)。
sys.path.insert(0, str(ROOT / "src"))
# 画面なしで描画する設定。★mujoco を import する前に書く必要がある(理由は check_install.py 参照)。
os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v3 as iio  # noqa: E402  (画像ファイルの保存に使う。[頻度] 画像保存の定番)
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

model = mujoco.MjModel.from_xml_path(str(ROOT / "models" / "so101" / "scene.xml"))
data = mujoco.MjData(model)

# --- 1. 関節とアクチュエータの一覧 ---
# このモデルのアクチュエータは <position>(位置アクチュエータ)。実機のサーボ(STS3215)と同じく
# 「目標角度 ctrl [rad] を渡すと、モデルの内部で PD 制御してくれる」タイプ。
# 自作 PD(controller.py)でトルクを計算した2リンクアームの <motor> とは違う点に注意。
# gainprm[0] = kp、biasprm[2] = -kv(MuJoCo の仕様。位置アクチュエータは力 = kp*(目標-現在) - kv*速度)。
print(f"{'関節名':<14}{'下限[rad]':>10}{'上限[rad]':>10}{'kp':>8}{'kv':>7}{'最大トルク[N·m]':>16}")
for i in range(model.nu):
    # actuator_trnid[i][0] = このアクチュエータが動かす関節の番号。
    joint = model.joint(model.actuator_trnid[i][0])
    act = model.actuator(i)
    print(f"{joint.name:<14}{joint.range[0]:>10.2f}{joint.range[1]:>10.2f}"
          f"{act.gainprm[0]:>8.1f}{-act.biasprm[2]:>7.2f}{act.forcerange[1]:>16.2f}")

# --- 2. 姿勢を変えて描画 ---
# 各姿勢は「6関節の目標角度 [rad]」。順番は上の表と同じ
# (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper)。
POSES = {
    "zero(各関節の可動域の中央)": [0, 0, 0, 0, 0, 0],
    "lift+elbow": [0, -0.8, 0.8, 0.5, 0, 0],
    "pan+gripper open": [0.8, -0.5, 1.0, 0.0, 0, 1.2],
}

# 自由カメラ(名前付きカメラがこのモデルには無いので、位置を数値で指定する)。
cam = mujoco.MjvCamera()
cam.lookat[:] = [0.2, 0.0, 0.1]  # 見る点 [m]
cam.distance = 0.9  # その点からカメラまでの距離 [m]
cam.azimuth = 140  # 水平方向の回り込み角 [度]
cam.elevation = -25  # 上下方向の角度 [度](マイナスで見下ろす)

images = []
with mujoco.Renderer(model, height=480, width=480) as renderer:
    # with ... as: 使い終わったら自動で片付け(close)してくれる書き方。[頻度] ファイルやリソースの扱いで定番。
    for name, target in POSES.items():
        mujoco.mj_resetData(model, data)
        data.ctrl[:] = target
        # 目標角度に落ち着くまで 2 秒分進める(timestep 0.002 秒 × 1000 回)。
        for _ in range(1000):
            mujoco.mj_step(model, data)
        renderer.update_scene(data, camera=cam)
        images.append(renderer.render())
        print(f"{name}: 到達した関節角 [rad] = {np.round(data.qpos[:model.nu], 2)}")

out = ROOT / "outputs" / "so101_poses.png"
# np.hstack: 同じ高さの画像を横に並べて1枚にする。[頻度] 配列を並べる定番(縦は vstack)。
iio.imwrite(out, np.hstack(images))
print(f"保存: {out}")
