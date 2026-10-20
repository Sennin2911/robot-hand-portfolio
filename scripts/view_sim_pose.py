"""【概要】SO-101 を sim で指定した姿勢にして MuJoCo のビューアに表示し、
その姿勢を実機に送るときの値(度、gripper は 0〜100)と、実機に送るコマンドをターミナルに出す。
sim の見た目と実機の腕の形を見比べるための準備用。実機は動かさない(送るのは real_send_pose.py)。

使い方(この .venv で実行):
    .venv/bin/python scripts/view_sim_pose.py                      # 全関節 0 rad の姿勢
    .venv/bin/python scripts/view_sim_pose.py --q 0.5 0 0 0 0 0.5  # 6関節の rad を順に指定
    .venv/bin/python scripts/view_sim_pose.py --no-view            # ウィンドウを開かず、値だけ表示
関節の並び: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)。
sys.path.insert(0, str(ROOT / "src"))

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402
import numpy as np  # noqa: E402
from hand_sim.sim_real import JOINTS, sim_to_real  # noqa: E402

parser = argparse.ArgumentParser()
# nargs=6: 値をちょうど6個受け取る。type=float: 文字列でなく小数にする。
parser.add_argument("--q", type=float, nargs=6, default=[0.0] * 6, help="6関節の角度 [rad]。sim の関節の並び順")
parser.add_argument("--no-view", action="store_true", help="ビューアを開かず、変換した値だけ表示する")
args = parser.parse_args()

model = mujoco.MjModel.from_xml_path(str(ROOT / "models" / "so101" / "scene.xml"))
data = mujoco.MjData(model)
q = np.array(args.q)

real = sim_to_real(model, q)  # sim の rad → 実機の値(sim の可動域の外は端に切り詰められる)
values = [real[f"{n}.pos"] for n in JOINTS]
print("sim の姿勢 [rad] :", {n: round(float(v), 3) for n, v in zip(JOINTS, q)})
print("実機の値 (度/0-100):", {n: round(v, 1) for n, v in zip(JOINTS, values)})
print("実機に送るコマンド (lerobot の環境で実行。腕を支えてから):")
print(f"  python {ROOT}/scripts/real_send_pose.py --pose {' '.join(f'{v:.1f}' for v in values)}")

if not args.no_view:
    # qpos は今の関節角、ctrl は <position> アクチュエータの目標角。両方に入れておくと、姿勢が保たれる。
    data.qpos[:6] = q
    data.ctrl[:6] = q
    mujoco.mj_forward(model, data)
    # launch_passive: 自分でシミュレーションを回しつつ、ビューアに表示する。
    # `with ... as viewer:` は、ブロックを抜けるとウィンドウを自動で閉じる書き方。[頻度] 定番
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()  # 計算結果をウィンドウに反映する
            time.sleep(model.opt.timestep)  # 実時間に合わせて待つ(待たないと一瞬で進んでしまう)
