# 【概要】このファイルがあるフォルダ(hand_sim)を「Pythonのパッケージ」にする目印。
#   これがあると、他のファイルから `from hand_sim import HandEnv` のように読み込める。
#   [頻度] パッケージには必ず置く定番のファイル。中身が空のことも多い。

# 同じフォルダの env.py から HandEnv クラスを読み込む。
# 先頭の「.」は「同じフォルダの」という意味(相対import)。[頻度] パッケージ内では定番。
from .env import HandEnv

# `from hand_sim import *` としたときに公開する名前の一覧。
# [頻度] ライブラリではよく書くが、小さなプロジェクトでは省略されることも多い。
__all__ = ["HandEnv"]
