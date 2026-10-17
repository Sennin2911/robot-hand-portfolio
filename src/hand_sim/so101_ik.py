"""【概要】SO-101 の数値逆運動学(IK)。手先(site "gripperframe")を目標の位置 (x, y, z) [m] に
持っていく関節角を、ヤコビアンを使った反復計算で求める。
2リンク平面アームの arm.ik は数式で一発で解いたが、5自由度の3次元アームは式で解きにくい。
そこで「今の姿勢から、手先を目標へ少しずつ近づける」を繰り返す方法(数値IK)を使う。
詳しい考え方は docs/numerical_ik.md を参照。
"""
import mujoco
import numpy as np

N_ARM = 5  # IK で動かす関節の数 (shoulder_pan 〜 wrist_roll)。gripper(6番目)は手先の位置に影響しないので除く。


def solve_ik(model, data, target, q_init, max_iter=200, tol=1e-4, damping=0.05, max_step=0.2, history=None):
    """手先を target [m] に届かせる関節角 (長さ5) を求める。

    model, data : MuJoCo の model / data。data の中身(qpos)は、この関数の中で書き換わる。
    target      : 手先の目標位置 (x, y, z) [m]。世界座標。
    q_init      : 反復の出発点となる関節角 (長さ5) [rad]。
    tol         : 手先と目標の距離がこれ未満になったら成功 [m]。
    damping     : 減衰(λ)。特異姿勢で暴れないための安全弁。大きいほど慎重(遅い)。
    max_step    : 1回の更新で関節を動かす最大量 [rad]。
    history     : リストを渡すと、反復ごとの「手先と目標の距離」[m] を追記する(収束の様子を見る用)。None なら記録しない。

    戻り値: (q, ok)  q = 求めた関節角(長さ5、関節の可動域内)、ok = tol 以内に届いたか(True/False)。
    """
    site_id = model.site("gripperframe").id
    # 関節の可動域 (下限, 上限)。model.jnt_range の先頭5行が arm の5関節に対応する。
    lo, hi = model.jnt_range[:N_ARM, 0], model.jnt_range[:N_ARM, 1]
    q = np.clip(np.array(q_init, dtype=float), lo, hi)  # np.array(...): コピーを作る(元の q_init を壊さないため)
    jacp = np.zeros((3, model.nv))  # 位置のヤコビアン用の入れ物 (3行 × 全自由度の列)

    for _ in range(max_iter):
        data.qpos[:N_ARM] = q
        mujoco.mj_forward(model, data)  # qpos から手先の位置などを計算し直す(順運動学)
        err = np.asarray(target) - data.site_xpos[site_id]  # 手先から目標へ向かうベクトル [m]
        if history is not None:
            history.append(float(np.linalg.norm(err)))
        if np.linalg.norm(err) < tol:
            return q, True

        # ヤコビアン J: 「関節を少し動かすと手先がどう動くか」の表 (3行 × 5列)。
        # mj_jacSite は jacp (位置用) に結果を書き込む。第5引数 None = 回転用は要らない。
        mujoco.mj_jacSite(model, data, jacp, None, site_id)
        J = jacp[:, :N_ARM]

        # 関節をどれだけ動かすか dq (長さ5) を求める。減衰付き最小二乗法 (damped least squares):
        #     dq = J^T (J J^T + λ² I)^-1 err        (λ = damping、I = 3×3 の単位行列)
        # 単純に J の逆行列を使うと、腕が伸び切った姿勢(特異姿勢)で dq が巨大になって暴れる。
        # λ² I を足すと、そういう姿勢でも dq が有限に収まる。[頻度] IK では定番の安全策。
        #   J.T   : 転置(行と列の入れ替え)。[頻度] numpy で非常によく使う
        #   @     : 行列の積。[頻度] numpy で定番(* は要素ごとの積で別物)
        #   np.eye(3) : 3×3 の単位行列 I。[頻度] 線形代数でよく使う
        #   np.linalg.solve(A, b) : A x = b の x を求める。逆行列を作って掛けるより正確で速いので定番。
        A = J @ J.T + damping**2 * np.eye(3)  # 3×3
        dq = J.T @ np.linalg.solve(A, err)  # (5×3) @ (3,) → 長さ5

        # 1回で大きく動きすぎると不安定になるので、最大の変化量を max_step に抑える。
        biggest = np.max(np.abs(dq))
        if biggest > max_step:
            dq *= max_step / biggest  # *= : 左の変数に掛けて上書き。[頻度] よく使う
        q = np.clip(q + dq, lo, hi)  # 可動域を超えないよう切り詰める (np.clip: 範囲外を端の値にする)

    return q, False
