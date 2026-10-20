"""【概要】sim(MuJoCo)の関節角 [rad] と、実機(LeRobot)の値 [度、gripper は 0〜100] を相互に変換する。
合わせるのは「単位」「+ の向き(符号)」「gripper の尺度」の3つ。0 の位置は、どちらも
「可動域の中央が 0」なので、腕の5関節ではずらさない(考え方は docs/sim_real_mapping.md)。
sim の目標をそのまま実機に送るための橋渡しで、sim の可動域の外に出る値は端に切り詰める。
"""
import numpy as np

# 関節の並び順。sim の qpos / ctrl の並び、LeRobot の "xxx.pos" のキーの名前と同じ。
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# 実機の + の向きが sim と同じなら +1、逆なら -1。実機で 1 関節ずつ動かして確かめた結果(2026-10-20)。
# wrist_roll は「半時計回り」としか聞いていない(どちらから見たかが未確認)ので、仮に +1 にしている。
SIGN = {
    "shoulder_pan": +1,
    "shoulder_lift": +1,
    "elbow_flex": +1,
    "wrist_flex": +1,
    "wrist_roll": +1,
    "gripper": +1,  # sim も実機も、+ の方向が「開く」
}


def _sim_range(model, name):
    """sim でその関節が動ける範囲 (下限, 上限) [rad]。MJCF の range から取る。"""
    # model.joint("名前"): 名前で関節を引く。.range は [下限, 上限] の配列。
    lo, hi = model.joint(name).range
    return float(lo), float(hi)


def sim_to_real(model, q_sim):
    """sim の関節角 q_sim (長さ6、単位 rad) を、実機に送る辞書 {"shoulder_pan.pos": 度, ...} にする。

    腕の5関節: 度 = 符号 × rad を度に直した値。
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
            value = SIGN[name] * np.degrees(q)  # np.degrees: rad → 度
        action[f"{name}.pos"] = float(value)
    return action


def real_to_sim(model, obs):
    """実機の観測 obs ({"shoulder_pan.pos": 度, ...}) を、sim の関節角 (長さ6の配列、rad) にする。"""
    q = np.zeros(len(JOINTS))
    for i, name in enumerate(JOINTS):  # enumerate: (番号, 要素) の組で回す。[頻度] 定番
        value = obs[f"{name}.pos"]
        lo, hi = _sim_range(model, name)
        if name == "gripper":
            q[i] = lo + value / 100.0 * (hi - lo)
        else:
            q[i] = np.radians(SIGN[name] * value)  # 符号は ±1 なので、掛けると逆変換にもなる
    return q
