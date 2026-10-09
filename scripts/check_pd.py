"""【概要】controller.py の PDController が正しいかを確認する。
 1. 式の確認: 手計算できる値で torque() の答えを比べる。
 2. 動作の確認: アームを目標角度へ動かし、最終的にぴったり止まり、行き過ぎが小さいかを見る。
全部通れば終了コード0、そうでなければ1。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)。
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from hand_sim import HandEnv  # noqa: E402
from hand_sim.controller import PDController  # noqa: E402

n_ng = 0


def check(name, ok, detail=""):
    """判定結果を表示して、NG の数を数える。"""
    global n_ng  # 関数の外の変数 n_ng を書き換えるための宣言。[頻度] あまり使わない(簡単なスクリプト向け)。
    n_ng += 0 if ok else 1
    print(f"{'OK' if ok else 'NG'}  {name}  {detail}")


try:
    # 1. 式の確認(手計算: 関節1 = 10×(1.0−0.5) − 0.5×0.2 = 4.9、関節2 = 10×(0−0.5) − 0.5×(−0.4) = −4.8)
    pd = PDController(kp=10.0, kv=0.5)
    got = np.asarray(pd.torque([1.0, 0.0], [0.5, 0.5], [0.2, -0.4]), dtype=float)
    check("式: 目標との差と速度から計算", np.allclose(got, [4.9, -4.8]), f"got={got.round(4)} 期待=[4.9 -4.8]")
    got = np.asarray(pd.torque([0.3, 0.3], [0.3, 0.3], [0.0, 0.0]), dtype=float)
    check("式: 目標に着いて止まっていればトルク0", np.allclose(got, 0.0), f"got={got.round(4)}")

    # 2. 動作の確認: 3秒間、目標角度 (0.8, -1.0) へ動かす。
    env = HandEnv(ROOT / "models" / "arm2link.xml")
    target = np.array([0.8, -1.0])
    max_over = np.zeros(2)  # 目標を行き過ぎた量の最大値
    for _ in range(int(3.0 / env.dt)):
        env.step(pd.torque(target, env.joint_pos(), env.joint_vel()))
        # 目標の向きに対して「超えた量」: 目標が正なら q - target、負なら target - q。
        max_over = np.maximum(max_over, (env.joint_pos() - target) * np.sign(target))
    err = np.abs(env.joint_pos() - target)
    check("動作: 3秒後に目標へ到達(誤差 < 0.01 rad)", bool(np.all(err < 0.01)), f"誤差={err.round(4)}")
    check("動作: 行き過ぎが小さい(< 0.05 rad)", bool(np.all(max_over < 0.05)), f"行き過ぎ={max_over.round(4)}")
    env.close()
except NotImplementedError as e:
    print(f"{e}. src/hand_sim/controller.py に書いてください。")
    sys.exit(1)

print("全部OK" if n_ng == 0 else f"{n_ng} 件NGです")
sys.exit(0 if n_ng == 0 else 1)
