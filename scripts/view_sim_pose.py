"""【概要】SO-101 を sim で指定した姿勢にして MuJoCo のビューアに表示し、
その姿勢を実機に送るときの値(度、gripper は 0〜100)と、実機に送るコマンドをターミナルに出す。
sim の見た目と実機の腕の形を見比べるための準備用。実機は動かさない(送るのは real_send_pose.py)。

使い方(この .venv で実行):
    .venv/bin/python scripts/view_sim_pose.py                          # 全関節 0 の姿勢
    .venv/bin/python scripts/view_sim_pose.py --pose 30 0 30 0 60 60   # 実機と同じ値(度、gripper は 0〜100)で指定
    .venv/bin/python scripts/view_sim_pose.py --q 0.5 0 0 0 0 0.5      # sim の rad で指定
    .venv/bin/python scripts/view_sim_pose.py --no-view                # ウィンドウを開かず、値だけ表示
--pose は real_send_pose.py の --pose と同じ値・同じ意味。--pose と --q は同時に指定できない。
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
from hand_sim.sim_real import JOINTS, real_to_sim, sim_to_real  # noqa: E402

parser = argparse.ArgumentParser()
# add_mutually_exclusive_group: このグループの中のオプションは、どれか1つしか指定できない。[頻度] たまに使う
group = parser.add_mutually_exclusive_group()
# nargs=6: 値をちょうど6個受け取る。type=float: 文字列でなく小数にする。
group.add_argument("--pose", type=float, nargs=6, help="実機と同じ値で指定 (度、gripper は 0〜100)")
group.add_argument("--q", type=float, nargs=6, help="sim の角度 [rad] で指定")
parser.add_argument("--no-view", action="store_true", help="ビューアを開かず、変換した値だけ表示する")
args = parser.parse_args()

model = mujoco.MjModel.from_xml_path(str(ROOT / "models" / "so101" / "scene.xml"))
data = mujoco.MjData(model)
if args.pose is not None:
    # 実機の値 → sim の rad。キーは real_to_sim が読む "関節名.pos" の形にする。
    q = real_to_sim(model, {f"{n}.pos": v for n, v in zip(JOINTS, args.pose)})
elif args.q is not None:
    q = np.array(args.q)
else:
    q = np.zeros(6)

# 画面に出す姿勢も、sim の可動域の中に収める(値の表示と画面が食い違わないように)。
q = np.clip(q, model.jnt_range[:6, 0], model.jnt_range[:6, 1])
real = sim_to_real(model, q)  # sim の rad → 実機の値(sim の可動域の外は端に切り詰められる)
values = [real[f"{n}.pos"] for n in JOINTS]
if args.pose is not None:
    # sim の可動域の方が実機より狭いので、指定した値が切り詰められることがある。そのときは知らせる。
    for n, want, got in zip(JOINTS, args.pose, values):
        if abs(want - got) > 1e-6:
            print(f"注意: {n} は sim の可動域の外なので、{want} を {got:.1f} に切り詰めた")
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
