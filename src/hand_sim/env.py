"""【概要】MuJoCoシミュレータを扱いやすくする薄いラッパー(HandEnv クラス)。

シミュレータの操作(進める・状態を読む・描画する)をここにまとめておく。
制御のコードは HandEnv だけを使うようにしておけば、将来、実機ロボットを動かす
部品に差し替えても、制御側のコードをほとんど変えずに済む。
"""
# pathlib.Path: ファイルのパスを文字列より安全に扱う標準機能。[頻度] 現在のPythonでは定番。
from pathlib import Path

# mujoco: MuJoCo本体のPythonライブラリ。
import mujoco
# numpy: 数値の配列(ベクトル)を高速に扱う定番ライブラリ。慣習として np と略す。
import numpy as np

# デフォルトで読み込むモデルファイルの場所。
# __file__          : このファイル自身(env.py)の場所。
# .resolve()        : 相対パスを、省略のない絶対パスに直す。
# .parents[2]       : 親フォルダをたどる。[0]=hand_sim, [1]=src, [2]=プロジェクト直下。
# / "models" / ...  : Pathでは「/」でフォルダ名をつなげられる。
# [頻度] 「このファイルからの相対位置でデータを探す」書き方は定番。
# 要約：現在のファイルの2階層上のmodelsフォルダにあるscene.xmlファイルを読み込む。
DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models" / "scene.xml"


class HandEnv:
    """ハンドのシミュレーション環境。1つのシミュレータ(model + data)を持つ。"""

    # __init__ : HandEnv() を作ったときに最初に1回呼ばれる「準備用」の特別な関数。
    # self     : 「このオブジェクト自身」。self.xxx に入れた値は、他のメソッドからも使える。
    def __init__(self, model_path=DEFAULT_MODEL):
        # model: 物体の形・関節・重さなど「変わらない設定」。XMLファイルから読み込む。
        # (Pathを文字列に直して渡す必要があるので str() を使っている)
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        # data : 今の位置・速度・時刻など「刻々と変わる状態」。
        self.data = mujoco.MjData(self.model)
        # dt: 1ステップで進む時間[秒]。モデルの timestep(0.002秒)と同じ値。
        self.dt = self.model.opt.timestep
        # 描画用の部品。最初に描画するときに作る(使わないなら作らない)。
        self._renderer = None

    # @property: メソッドを「self.nu」のように、かっこ無しの値として読めるようにする。
    # [頻度] よく使われる。
    @property
    def nu(self):
        """操作できるアクチュエータ(モータ)の数。"""
        return self.model.nu

    def reset(self):
        """シミュレーションを初期状態に戻す。ここでの初期に戻すのはqposの値のみでdataはそのまま"""
        mujoco.mj_resetData(self.model, self.data)
        # mj_forward: 時間を進めずに、qposからワールド座標系での再計算を行う。
        mujoco.mj_forward(self.model, self.data)

    def step(self, ctrl, n_substeps=1):
        """位置指令 ctrl(長さ nu の配列)を与えて、n_substeps 回シミュレーションを進める。"""
        # [:] = は「配列の中身を丸ごと上書き」。data.ctrl は「モータへの指令値」。
        self.data.ctrl[:] = ctrl
        # for _ in range(n): 「回数だけ繰り返したい、番号は使わない」ときの定番。
        # _ は「使わない変数」という慣習の名前。
        for _ in range(n_substeps):
            # mj_step: mj_forwardと異なり、計算するがctrlの値も考慮してシミュレーションを1ステップ(dt秒)も進める。
            mujoco.mj_step(self.model, self.data)

    def joint_pos(self):
        """各関節の現在位置(角度[rad]、スライドは[m])を返す。"""
        # qpos: 全関節の位置の配列。先頭から nu 個がハンドの関節(物体より前に定義してあるため)。
        # .copy(): コピーを返す。元のデータを呼び出し側が誤って書き換えるのを防ぐ。
        return self.data.qpos[: self.model.nu].copy()

    def body_pos(self, name):
        """名前で指定した物体(body)の位置 [x, y, z] (単位m) を返す。"""
        # xpos：各ボディ・jointの世界座標
        return self.data.body(name).xpos.copy()

    def render(self, camera="front", width=640, height=480):
        """指定カメラからの画像(高さ×幅×RGB の配列)を返す。画面には表示しない。"""
        # 描画部品が未作成、またはサイズが変わったときだけ作り直す。
        # (毎回作ると遅く、GPUのメモリも無駄になる)
        if self._renderer is None or (self._renderer.width, self._renderer.height) != (width, height):
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        # update_scene: 今のシミュレータの状態を、描画部品に反映する。
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def close(self):
        """描画部品を片付ける。使い終わったら呼ぶ(呼ばないと終了時にエラー表示が出ることがある)。"""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
