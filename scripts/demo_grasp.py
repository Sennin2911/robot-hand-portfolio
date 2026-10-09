"""【概要】つかむ動作(下降 → 把持 → 持ち上げ)を画面なしで実行し、動画(mp4)に保存する。
成功(物体が持ち上がった)なら終了コード0、失敗なら1で終わる。自動テストにも使える。
"""
import os
import sys
from pathlib import Path

# プロジェクト直下のフォルダ。.parents[1] は「親フォルダを1つ上へ」(scripts → 直下)。
ROOT = Path(__file__).resolve().parents[1]
# src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)。
sys.path.insert(0, str(ROOT / "src"))
# 画面なし描画の設定。mujoco を読み込む前に必要。
os.environ.setdefault("MUJOCO_GL", "egl")

# imageio: 画像・動画の読み書きライブラリ。.v2 は従来からの使い方のバージョン。
import imageio.v2 as imageio  # noqa: E402
from hand_sim import HandEnv  # noqa: E402
from hand_sim.grasp import PHASES, SUCCESS_Z, ctrl_at  # noqa: E402

env = HandEnv()
env.reset()
fps = 30  # 動画の1秒あたりのコマ数
# 何ステップごとに1コマ撮るか: 1/(30コマ × 0.002秒) ≒ 16ステップ。int() で整数にする。
render_every = int(1 / (fps * env.dt))
# 全体の時間 = 各区間の時間の合計。
# sum(d for _, d in PHASES) は「PHASESの各要素から時間 d だけ取り出して足す」書き方
# (ジェネレータ式)。[頻度] 定番。
total = sum(d for _, d in PHASES)
frames = []  # 撮影したコマ(画像)を貯めるリスト
for i in range(int(total / env.dt)):         # 全体の時間ぶんステップを繰り返す
    env.step(ctrl_at(i * env.dt))             # 今の時刻の指令値を与えて1ステップ進める
    if i % render_every == 0:                 # % は余り。「render_every回に1回」だけ撮影する
        frames.append(env.render("front"))

# 最後のコマを画像として保存(動画を開かなくても結果を確認できる)。
imageio.imwrite(ROOT / "outputs" / "demo_grasp_last.png", frames[-1])  # [-1] = 最後の要素
z = env.body_pos("target")[2]                 # 物体の位置 [x,y,z] のうち高さ z
ok = z > SUCCESS_Z
# 条件式 A if 条件 else B: 条件によって値を切り替える書き方。[頻度] 定番
print(f"target z = {z:.3f} m -> {'GRASP SUCCESS' if ok else 'grasp failed'}")  # :.3f は小数点以下3桁
out = ROOT / "outputs" / "demo_grasp.mp4"
imageio.mimsave(out, frames, fps=fps)         # コマのリストを動画として保存
print("saved", out)
env.close()
sys.exit(0 if ok else 1)                       # 成功なら0、失敗なら1で終了
