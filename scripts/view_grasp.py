"""【概要】つかむ動作をMuJoCoのビューア(GUIウィンドウ)でリアルタイム再生する。ウィンドウを閉じるまで繰り返す。

ビューアの操作: スペース=一時停止, Backspace=リセット, 右ドラッグ=カメラ移動
"""
import sys
import time
from pathlib import Path

# src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# mujoco.viewer: GUIウィンドウで表示する機能。
import mujoco.viewer  # noqa: E402
from hand_sim import HandEnv  # noqa: E402
from hand_sim.grasp import PHASES, ctrl_at, LIFT  # noqa: E402

env = HandEnv()
total = sum(d for _, d in PHASES) + 1.0  # 全体の時間 + 最後に1秒止める

# launch_passive: ウィンドウを開くが、シミュレーションの進行は私たちのコードが握る方式。
# `with ... as viewer:` は「ブロックを抜けるときに後始末(ウィンドウを閉じる)を自動で行う」書き方。
# [頻度] ファイルやウィンドウなど「後片付けが要るもの」を扱うときの定番。
with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    while viewer.is_running():                # ウィンドウが開いている間、動作を繰り返す
        env.reset()
        t0 = time.time()                      # 再生開始時の実時間(壁時計)
        while viewer.is_running() and env.data.time < total:
            # シミュレーション時間が実時間に追いつくまでステップを進める。
            # (こうしないと、PCの速さによって再生が速すぎたり遅すぎたりする)
            while env.data.time < time.time() - t0 and env.data.time < total:
                # env.step()には配列（ここでは[0.0, 0.0, 0.0, 0.0, 0.0]）が入る
                # env.step(ctrl_at(env.data.time))
                env.step(LIFT)
            viewer.sync()                     # 最新の状態をウィンドウに反映
            time.sleep(0.005)                 # 0.005秒休んでCPUを使いすぎないようにする
