"""【概要】2リンクアームを MuJoCo のビューア(GUIウィンドウ)で動かす。
目標点(オレンジの球)が円を描いて動き、アームの手先(黄色い点)がそれを追いかける。
手先を追わせるには、目標点の位置 (x, y) から ik で目標の関節角を求め、
PD制御(controller.py)でその角度に合わせるトルクを計算し、モータ(ctrl)に渡す。

ビューアの操作: スペース=一時停止, Backspace=リセット, 右ドラッグ=カメラ移動
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)。
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import mujoco.viewer  # noqa: E402
from hand_sim import HandEnv  # noqa: E402
from hand_sim.arm import ik  # noqa: E402
from hand_sim.controller import PDController  # noqa: E402

env = HandEnv(ROOT / "models" / "arm2link.xml")
# PD制御のゲイン。kp は大きいほど速く目標に着くが行き過ぎやすい。kv は大きいほど行き過ぎを抑える。
# ただし kv を大きくしすぎると計算が不安定になる(詳しくは controller.py)。
pd = PDController(kp=10.0, kv=0.5)

# 目標点が描く円: 中心 (0.2, 0) m、半径 0.1 m、周期 6 秒。
# 肩からの距離が 0.1〜0.3 m に収まるので、アームの届く範囲 (0.05〜0.35 m) の中に入る。
CENTER = np.array([0.2, 0.0])
RADIUS = 0.1
PERIOD = 6.0


def goal_at(t):
    """時刻 t [秒] での目標点 (x, y) [m] を返す。円周上を反時計回りに動く。"""
    angle = 2 * np.pi * t / PERIOD
    return CENTER + RADIUS * np.array([np.cos(angle), np.sin(angle)])


def ctrl_for(x, y):
    """目標点 (x, y) に手先を届かせるためのトルク指令 [N·m] を返す。"""
    # ik は (q1, q2) のタプルを返す。`q1, q2 = ...` で2つの変数に分けて受け取る(タプルの展開)。
    # [頻度] 定番。ここで得た (q1, q2) は「目標の関節角」であって、モータへの指令ではない。
    q1, q2 = ik(x, y)
    # 目標角・今の角度・今の角速度から、PD制御でトルクを計算する。
    return pd.torque([q1, q2], env.joint_pos(), env.joint_vel())


# launch_passive の意味や `with ... as viewer:` の書き方は view_grasp.py を参照。
with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    while viewer.is_running():
        env.reset()
        t0 = time.time()                      # 再生開始時の実時間
        while viewer.is_running():
            # シミュレーション時間が実時間に追いつくまでステップを進める。
            while env.data.time < time.time() - t0:
                x, y = goal_at(env.data.time)
                # mocap_pos: mocap の body の位置を入れる配列(後述)。[0] は mocap body の1つ目
                # (このモデルでは "goal" だけ)。物理計算は関与せず、代入した位置にそのまま置かれる。
                # 高さ z=0.05 はアームと同じ高さ(arm2link.xml の link1 の高さ)。
                env.data.mocap_pos[0] = [x, y, 0.05]
                env.step(ctrl_for(x, y))
            viewer.sync()
            time.sleep(0.005)
