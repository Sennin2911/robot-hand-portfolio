"""【概要】sim(MuJoCo)の関節角 [rad] と、実機(LeRobot)の値 [度、gripper は 0〜100] を相互に変換する。
合わせるのは「単位」「+ の向き(符号)」「gripper の尺度」「関節ごとの一定の角度のずれ(オフセット)」の4つ。
オフセットは、実機で手先の位置を測って求めた値(求め方は docs/offset_calibration.md、符号などは docs/sim_real_mapping.md)。
sim の目標をそのまま実機に送るための橋渡しで、sim の可動域の外に出る値は端に切り詰める。
"""
import numpy as np

# 関節の並び順。sim の qpos / ctrl の並び、LeRobot の "xxx.pos" のキーの名前と同じ。
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# 実機の + の向きが sim と同じなら +1、逆なら -1。実機で 1 関節ずつ動かして確かめた結果(2026-10-20)。
# wrist_roll は、実機の + が「土台側から手先を見て反時計回り」(作者の確認)で、sim の + と同じ向き。
SIGN = {
    "shoulder_pan": +1,
    "shoulder_lift": +1,
    "elbow_flex": +1,
    "wrist_flex": +1,
    "wrist_roll": +1,
    "gripper": +1,  # sim も実機も、+ の方向が「開く」
}

# 実機の角度に足すと、sim の姿勢と合う一定のずれ [度]。実機で手先の位置を測り、最小二乗法で求めた(2026-10-21)。
#   sim の角度 [度] = 符号 × 実機の値 + OFFSET_DEG   (実機の値 = 符号 × (sim の角度 − OFFSET_DEG))
# 例: shoulder_pan が +4.4 → 実機に 0 を送ると、腕は sim の「+4.4°」の向き(右へ 4.4°)を向く。
# lift・elbow・wrist の 3 つは、測った姿勢の範囲では手先を ±0.7 cm で合わせるが、3 つの配分は幅(±1〜3°)が大きい。
# 掴む動作で位置のずれが目立つ姿勢が出たら、その姿勢を測り足して求め直す。値を求める手順: scripts/fit_joint_offsets.py
OFFSET_DEG = {
    "shoulder_pan": +4.4,  # 机の端と土台を平行に置き、pan ±60° で印をつけた左右の奥行きの差(16 cm と 19 cm)から求めた
    "shoulder_lift": +3.0,
    "elbow_flex": -7.8,
    "wrist_flex": -3.7,
    "wrist_roll": 0.0,  # 未測定(手先の位置にはほとんど効かないので、今は 0)
    "gripper": 0.0,  # gripper は尺度の変換だけで、オフセットは使わない
}


def _sim_range(model, name):
    """sim でその関節が動ける範囲 (下限, 上限) [rad]。MJCF の range から取る。"""
    # model.joint("名前"): 名前で関節を引く。.range は [下限, 上限] の配列。
    lo, hi = model.joint(name).range
    return float(lo), float(hi)


def sim_to_real(model, q_sim, apply_offset=True):
    """sim の関節角 q_sim (長さ6、単位 rad) を、実機に送る辞書 {"shoulder_pan.pos": 度, ...} にする。

    腕の5関節: 度 = 符号 × (rad を度に直した値 − OFFSET_DEG)。
    apply_offset=False にすると、オフセットを引かない(オフセットを求めるスクリプトが、測定時の素の指令を扱うときに使う)。
    gripper  : sim の可動域 (下限〜上限) を、実機の 0〜100 に直線で対応させる(下限 → 0、上限 → 100)。
    sim の可動域の外の値は、先に端へ切り詰める(sim の範囲の方が実機より狭いので、安全側になる)。
    """
    action = {}
    for name, q in zip(JOINTS, q_sim):  # zip(a, b): 2つの並びを先頭から順に組にして回す。[頻度] 定番
        lo, hi = _sim_range(model, name)
        q = float(np.clip(q, lo, hi))  # np.clip(値, 下限, 上限): 範囲の外なら端の値にする。[頻度] 定番
        if name == "gripper":
            value = (q - lo) / (hi - lo) * 100.0
        else:
            offset = OFFSET_DEG[name] if apply_offset else 0.0
            value = SIGN[name] * (np.degrees(q) - offset)  # np.degrees: rad → 度。SIGN は ±1 なので、掛けると逆変換にもなる
        action[f"{name}.pos"] = float(value)
    return action


def real_to_sim(model, obs, apply_offset=True):
    """実機の観測 obs ({"shoulder_pan.pos": 度, ...}) を、sim の関節角 (長さ6の配列、rad) にする。
    apply_offset=True のとき、実機の値に OFFSET_DEG を足す(= 実機の腕が本当に取っている姿勢にする)。"""
    q = np.zeros(len(JOINTS))
    for i, name in enumerate(JOINTS):  # enumerate: (番号, 要素) の組で回す。[頻度] 定番
        value = obs[f"{name}.pos"]
        lo, hi = _sim_range(model, name)
        if name == "gripper":
            q[i] = lo + value / 100.0 * (hi - lo)
        else:
            offset = OFFSET_DEG[name] if apply_offset else 0.0
            q[i] = np.radians(SIGN[name] * value + offset)
    return q
