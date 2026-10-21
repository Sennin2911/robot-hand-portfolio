"""【概要】手先位置の実測(新計画の工程1)に使う 10 点を決め、各点の IK の解、実機に送るコマンド、
sim が予測する手先の位置を表にして docs/hand_position_measurement.md に書き出す。実機は動かさない。
点の決め方: 肩の回転軸を中心に「半径 r [cm]、向き a [度]、手先の高さ z [cm]」で指定する(実機で測りやすい形)。
その位置に届く関節角は solve_ik が求める。実機の可動域(端から MARGIN 内側)に収まらない点は、使わず理由を表示する。

使い方(この .venv で実行):
    .venv/bin/python scripts/plan_measure_points.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)。
sys.path.insert(0, str(ROOT / "src"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from hand_sim.sim_real import JOINTS, sim_to_real  # noqa: E402
from hand_sim.visual_servo import forward_tip, plan_hover  # noqa: E402

HOME = [0, -1.7, 1.6, 1.0, 0]  # IK の出発点: たたんだ姿勢(check_visual_servo.py と同じ)
GRIPPER = 10.0  # gripper の値(実機の 0〜100)。端(0, 100)の近くは real_send_pose.py が拒否するので避ける
MARGIN = 3.0  # real_move_joints.py と同じ。実機の可動域の端から、これだけ内側に収める [度]
# 実機の可動域(中央からの片側 [度])。~/.cache/huggingface/lerobot/calibration/robots/so_follower/my_follower_arm.json
# の range_min / range_max から (max - min) / 2 * 360 / 4095 で求めた値(real_move_joints.py の joint_limits と同じ式)。
REAL_HALF = {"shoulder_pan": 119.1, "shoulder_lift": 106.1, "elbow_flex": 97.3, "wrist_flex": 111.9, "wrist_roll": 180.0}

# 測る 10 点: (半径 r [cm], 向き a [度], 手先 gripperframe の高さ z [cm])。範囲は r=15〜28 cm、向き ±60°(作者が指定)。
# 向き a は、肩の軸から見て正面が 0、左が +、右が −(sim の y 軸の向き)。実機の shoulder_pan は右が + なので、符号が逆になる。
POINTS = [
    (22, 0, 10), (15, 0, 10), (28, 0, 10),            # 正面で、手前・中間・奥
    (22, -60, 10), (22, 60, 10),                       # 左右の端
    (28, 60, 10), (15, -60, 10),                       # 角(遠い左、近い右)
    (22, 0, 5), (15, 30, 5), (28, -30, 5),             # 低い高さ(掴むときに近い)
]

model = mujoco.MjModel.from_xml_path(str(ROOT / "models" / "so101" / "scene.xml"))
scratch = mujoco.MjData(model)
mujoco.mj_forward(model, scratch)
pan_axis = scratch.xanchor[model.joint("shoulder_pan").id][:2]  # 肩の回転軸の世界座標 (x, y)。測る基準

glo, ghi = model.joint("gripper").range
q_grip = glo + GRIPPER / 100.0 * (ghi - glo)

rows, cmds = [], []
for no, (r, a, z) in enumerate(POINTS, start=1):
    target = np.array([pan_axis[0] + r / 100 * np.cos(np.radians(a)), pan_axis[1] + r / 100 * np.sin(np.radians(a)), z / 100])
    q, ok = plan_hover(model, target[:2], target[2], HOME)
    tip = forward_tip(model, q)
    err = np.linalg.norm(tip - target)
    real = sim_to_real(model, np.append(q, q_grip))
    values = [real[f"{n}.pos"] for n in JOINTS]
    over = [n for n, v in zip(JOINTS[:5], values) if abs(v) > REAL_HALF[n] - MARGIN]
    if not ok or over:
        print(f"{no}: r={r} a={a} z={z} は使えない(IK {'失敗' if not ok else '成功'}、実機の可動域の端に近い関節: {over})")
        continue
    pose = " ".join(f"{round(v, 1) + 0.0:.1f}" for v in values)  # round(...) + 0.0: "-0.0" と表示されるのを避ける
    rows.append((no, r, a, z, err, pose))
    cmds.append(f"# 点{no}: r={r} cm, 向き={a:+d}°, z={z} cm\npython ~/ドキュメント/robot-hand-portfolio/scripts/real_send_pose.py --pose {pose} --hold 30")

lines = [
    "# 手先位置の実測(新計画の工程1)",
    "",
    "sim が予測する手先の位置と、実機で測った位置を比べる。実機は `scripts/real_send_pose.py` で動かす(`plan_measure_points.py` が生成)。",
    "",
    "## 測り方",
    "- **基準は肩の回転軸**(真上から見て、回る部分の中心)。机の面から高さを測る。",
    "- **測る点は「手先」= 固定側の指(動かない方の指)の先端**。sim の `gripperframe`(IK の目標の点)は、この指の先端にある(最も近いメッシュの頂点まで約 0.4 cm)。",
    "- 動く方の指や手首、指のつけ根ではない(取り違えると、半径・高さが大きくずれる)。sim の画像は `scripts/render_sim_pose.py` で出せる(赤い球が手先)。",
    "- 向きは、正面(軸から前方)を 0 として、**左が +、右が −**(sim の向き)。実機の `shoulder_pan` は右が + なので、コマンドの値とは符号が逆。",
    "- 差 = 実測 − 予測。予測は、半径・向き・高さの列(sim の IK が届いた位置)。",
    f"- 実機の値の gripper は {GRIPPER:.0f}(端から離れた値)で固定。",
    "- 高さの基準は机の面(sim の床 z=0)。sim の土台の底は z=−0.2 cm で、床にほぼ接している。",
    "- 土台の後端から肩の軸までは、sim で約 7.0 cm(作者が実機で測ったのは約 6 cm)。",
    "",
    "## 測定表",
    "| 点 | 予測の半径 [cm] | 予測の向き [度] | 予測の高さ [cm] | IK の誤差 [mm] | 実測の半径 | 実測の向き | 実測の高さ | 差(半径/向き/高さ) | メモ |",
    "|---|---|---|---|---|---|---|---|---|---|",
]
for no, r, a, z, err, _ in rows:
    lines.append(f"| {no} | {r} | {a:+d} | {z} | {err * 1000:.2f} | | | | | |")
lines += ["", "## 実機に送るコマンド(LeRobot の環境。腕を支えてから。30 秒保つので、その間に測る)", "```", *cmds, "```", ""]
out = ROOT / "docs" / "hand_position_measurement.md"
out.write_text("\n".join(lines), encoding="utf-8")  # write_text: 文字列をファイルに書く。[頻度] 小さなファイルの書き出しで定番
print(f"{len(rows)} 点を {out.relative_to(ROOT)} に書き出した")
for no, r, a, z, err, pose in rows:
    print(f"  点{no:2d}: r={r:2d} a={a:+3d} z={z:2d}  IK誤差 {err * 1000:.2f} mm  --pose {pose}")
