"""【概要】インストールの動作確認。モデル読込 → シミュレーション → 画像描画 の順に試す。"""
import os
import sys
from pathlib import Path

# sys.path: Pythonが「import したいファイル」を探しに行くフォルダの一覧。
# .insert(0, ...) でその先頭に src フォルダを加え、`from hand_sim import ...` を可能にする。
#   Path(__file__)  : このスクリプト自身の場所
#   .resolve()      : 絶対パスに直す
#   .parents[1]     : 親フォルダを1つ上へ(scripts → プロジェクト直下)
# [頻度] 「パッケージとしてインストールしない簡易プロジェクト」では定番だが、
#        本格的なプロジェクトでは pip install -e . で代用することが多い。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# 画面なしで描画するための設定。MUJOCO_GL=egl は「GPUで直接描画する方式」の指定。
# setdefault: すでに設定済みなら上書きしない(ユーザーの指定を優先する)。
# ★mujoco を import する前に設定する必要があるので、下の import より先に書く。
# sys.platform: 実行中のOSの名前。Linux なら "linux"、Mac なら "darwin"、Windows なら "win32"。
#   [頻度] OSごとに処理を変えたいときの定番。このプロジェクトでは、他のスクリプトも同じ形にしている。
# if で Linux のときだけ設定する理由: EGL は Linux 用の仕組みで、Mac には無い。
#   Mac で "egl" を指定すると描画に失敗するので、Mac では何も設定せず MuJoCo の標準に任せる。
if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402  (import が途中にあっても警告を出さない印。上記の理由で順番を変えている)
from hand_sim import HandEnv  # noqa: E402

env = HandEnv()
env.reset()
# 500ステップ(= 0.002秒 × 500 = 1秒)進める。data.ctrl は現在の指令値で、ここでは変えずに渡す。
env.step(env.data.ctrl, n_substeps=500)
# f"..." は f文字列: {} の中に変数や式を埋め込める。[頻度] 現在のPythonでは定番。
print(f"mujoco {mujoco.__version__}: nu={env.nu}, nbody={env.model.nbody}, step OK")
try:  # try/except: 失敗するかもしれない処理を試し、失敗したときの処理を except に書く。
    img = env.render()
    print(f"render OK: {img.shape}")  # .shape = 配列の形(高さ, 幅, 色の数)
    env.close()
except Exception as e:
    print(f"render FAILED: {e}")
    sys.exit(1)  # 終了コード1 = 失敗。0 が成功。他のプログラムが成否を判定できる。
