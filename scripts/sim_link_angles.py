"""【概要】sim の姿勢での「各リンクの傾き」を表示する。実機のリンクの傾きを、スマホの傾斜計アプリや
分度器で測って比べ、関節角(特に elbow_flex)が sim と実機で合っているかを調べるための確認用。
傾きは、真横から見て「手先の方向(前)を 0 度、上向きを + 」とした角度(-180〜180)。前に伸びていれば 0 度前後、真上なら 90 度、
後ろに倒れて上を向いていれば 90 度より大きい(例: 142 度 = 水平から 38 度の高さで、後ろへ向かう)。「水平から何度」(0〜90)だけを知りたいときは、
括弧内の「水平からの鋭角」を見る。リンクの向きは、関節の位置どうしを結ぶ線で決める:
  上腕 = shoulder_lift の軸 → elbow_flex の軸、前腕 = elbow_flex の軸 → wrist_flex の軸、手 = wrist_flex の軸 → 手先(gripperframe)。

使い方(この .venv で実行。引数は view_sim_pose.py と同じ):
    .venv/bin/python scripts/sim_link_angles.py --pose 0 0 0 0 0 10
    .venv/bin/python scripts/sim_link_angles.py --pose 0.0 -65.9 87.0 -7.7 0.0 10.0
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)。
sys.path.insert(0, str(ROOT / "src"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from hand_sim.sim_real import JOINTS, real_to_sim  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--pose", type=float, nargs=6, required=True, help="実機と同じ値で指定 (度、gripper は 0〜100)")
args = parser.parse_args()

model = mujoco.MjModel.from_xml_path(str(ROOT / "models" / "so101" / "scene.xml"))
data = mujoco.MjData(model)
data.qpos[:6] = real_to_sim(model, {f"{n}.pos": v for n, v in zip(JOINTS, args.pose)})
mujoco.mj_forward(model, data)

anchor = lambda name: data.xanchor[model.joint(name).id]  # lambda: 名前のない小さな関数。[頻度] ちょっとした一行の関数でよく使う
points = {"肩(shoulder_lift の軸)": anchor("shoulder_lift"), "肘(elbow_flex の軸)": anchor("elbow_flex"),
          "手首(wrist_flex の軸)": anchor("wrist_flex"), "手先(gripperframe)": data.site_xpos[model.site("gripperframe").id]}
names = list(points)
links = [("上腕", names[0], names[1]), ("前腕", names[1], names[2]), ("手", names[2], names[3])]
print("各点の高さ [cm](机の面が 0):", {n: round(float(p[2]) * 100, 1) for n, p in points.items()})
dirs = {}
# 「前」= 肩の回転軸から手先へ向かう水平方向(shoulder_pan が 0 でなくても、腕の面の中で測れる)。
forward = points[names[3]][:2] - data.xanchor[model.joint("shoulder_pan").id][:2]
forward = forward / np.linalg.norm(forward)
for label, a, b in links:
    v = points[b] - points[a]
    dirs[label] = v
    angle = np.degrees(np.arctan2(v[2], np.dot(v[:2], forward)))  # np.dot: 内積(2つのベクトルの「向きの重なり」)。[頻度] ベクトル計算で定番
    acute = np.degrees(np.arctan2(abs(v[2]), np.linalg.norm(v[:2])))
    print(f"{label}: 長さ {np.linalg.norm(v) * 100:5.1f} cm、前を 0°・上を + とした向き {angle:+7.1f}°(水平からの鋭角 {acute:.1f}°)")
u, f = dirs["上腕"], dirs["前腕"]
interior = np.degrees(np.arccos(np.dot(-u, f) / np.linalg.norm(u) / np.linalg.norm(f)))  # 肘の内角: 肘から肩へ向かう線と、肘から手首へ向かう線の間の角
print(f"肘の内角(上腕と前腕の間の角): {interior:.1f}°")
