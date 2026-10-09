"""【概要】制御の応答(ステップ応答)を数値で評価する関数をまとめたファイル。
シミュレータには依存せず、時刻と角度の配列だけを受け取る。
(将来、実機のログを同じ関数で評価できるようにするため。)
"""
import numpy as np


def analyze_step(t, q, target, tol=0.02):
    """1つの関節のステップ応答を評価して (overshoot, settling_time) を返す。どちらも float 1つ。

    t: 時刻の配列 [秒]、q: 同じ長さの関節角の配列 [rad]、target: 目標角 [rad]。
    q[0] を出発点とし、「出発点から target へ、一気に目標を切り替えた」応答とみなす。

    overshoot (行き過ぎ量) [rad]:
        目標を行き過ぎた量の最大値。向きは「出発点→目標」の向きで測るので、
        目標が出発点より小さい(下がる向きの)場合も、超えたぶんが正の値になる。超えなければ 0。
    settling_time (整定時間) [秒]:
        「それ以降はずっと |q − target| ≤ tol × |target − q[0]|(帯の中)」になる最初の時刻。
        最初から帯の中なら t[0]。最後まで帯に入りっぱなしにならなければ np.nan。
    """
    # 配列にしておくと、全サンプルをまとめて計算できる(リストのままだと引き算などができない)。
    t = np.asarray(t, dtype=float)
    q = np.asarray(q, dtype=float)
    step = target - q[0]            # 出発点から目標までの変化量(負なら下がる向き)
    direction = np.sign(step)       # np.sign: 正なら +1、負なら -1、0 なら 0。「どちら向きに動くか」

    # --- 行き過ぎ量 ---
    # (q − target) × direction が正の部分が、「目標を向きの先へ超えた量」。
    #  向きを掛けるので、正の目標でも負の目標でも同じ式で測れる。
    #  [間違えやすい点] `q > target` と書くと、目標が正のときしか「超えた」の意味にならない。
    # 全部負(=一度も超えていない)なら max(0.0, ...) で 0 にする。
    overshoot = max(0.0, float(np.max((q - target) * direction)))

    # --- 整定時間 ---
    # 「帯の外にいるサンプル」の添字をすべて集める。帯の幅は出発点から目標までの距離の tol 倍。
    #  [間違えやすい点] 帯は abs(q − target) で判定する。`target × (1±tol)` の形だと、
    #  目標が負のとき下限が上限より大きくなり、帯が空になる。
    # np.where(条件)[0]: 条件を満たす添字の配列。[頻度] NumPy で定番。
    outside = np.where(np.abs(q - target) > tol * abs(step))[0]

    if len(outside) == 0:
        # 最初から最後まで帯の中。
        settling_time = t[0]
    elif outside[-1] == len(q) - 1:
        # [-1] は「最後の要素」。帯の外にいる最後のサンプルが、配列の最後 = 最後まで整定しなかった。
        settling_time = np.nan
    else:
        # 「帯の外にいた最後のサンプル」の次のサンプルから、ずっと帯の中。その時刻が整定時間。
        #  [間違えやすい点] 「最初に帯に入った時刻」だと、あとで帯から外れる応答を見逃す。
        #  また `t[最後に中にいた] − t[中に入った]` は「中にいた時間の長さ」で、整定時間ではない。
        settling_time = t[outside[-1] + 1]

    # float() は numpy の数を普通の Python の小数に直す(表示や保存で扱いやすくするため)。
    return overshoot, float(settling_time)
