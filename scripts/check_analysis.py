"""【概要】analysis.py の analyze_step が正しいかを、手計算できる小さなデータで確認する。
全部通れば終了コード0、そうでなければ1。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)。
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from hand_sim.analysis import analyze_step  # noqa: E402

n_ng = 0


def check(name, got, expect_over, expect_settle):
    """analyze_step の結果が、期待する行き過ぎ量・整定時間と一致するかを判定して表示する。"""
    global n_ng  # 関数の外の変数 n_ng を書き換えるための宣言。[頻度] あまり使わない(簡単なスクリプト向け)。
    over, settle = got
    ok_over = np.isclose(over, expect_over)
    # np.isclose(nan, nan) は False になるので、nan どうしは別に比べる。
    ok_settle = (np.isnan(settle) and np.isnan(expect_settle)) or np.isclose(settle, expect_settle)
    ok = bool(ok_over and ok_settle)
    n_ng += 0 if ok else 1
    print(f"{'OK' if ok else 'NG'}  {name}  行き過ぎ={over:.3f}(期待 {expect_over})  "
          f"整定時間={settle}(期待 {expect_settle})")


try:
    t = np.arange(6.0)   # 時刻 0,1,2,3,4,5 [秒]

    # 1. 行き過ぎる: 1.2 まで行き過ぎ(0.2)、t=3 から ±0.02 以内に入りっぱなし。
    check("目標1.0に対し行き過ぎる", analyze_step(t, np.array([0, 0.5, 1.2, 1.0, 1.0, 1.0]), 1.0), 0.2, 3.0)
    # 2. 行き過ぎない: 0.99 は ±0.02 以内なので t=3 で整定。
    check("行き過ぎない", analyze_step(t, np.array([0, 0.5, 0.9, 0.99, 1.0, 1.0]), 1.0), 0.0, 3.0)
    # 3. 一度入っても、また外に出る: 最後に外にいるのは t=4 なので、整定は t=5。
    check("途中で一度入って、また外れる", analyze_step(t, np.array([0, 1.0, 1.0, 1.0, 0.5, 1.0]), 1.0), 0.0, 5.0)
    # 4. 最後まで範囲に入らない: 整定時間は nan。
    check("最後まで整定しない", analyze_step(t, np.array([0, 0.2, 0.4, 0.6, 0.8, 0.9]), 1.0), 0.0, np.nan)
    # 5. 負の目標(下がる向き): 行き過ぎは「目標を超えて、さらに下がった量」。-1.1 で 0.1。
    check("負の向きに行き過ぎる", analyze_step(t, np.array([0, -0.5, -1.1, -1.0, -1.0, -1.0]), -1.0), 0.1, 3.0)
except NotImplementedError as e:
    print(f"{e}. src/hand_sim/analysis.py に書いてください。")
    sys.exit(1)

print("全部OK" if n_ng == 0 else f"{n_ng} 件NGです")
sys.exit(0 if n_ng == 0 else 1)
