"""【概要】arm.py の fk が正しいかを、MuJoCo が計算した手先位置(site "tip")と比べて確認する。
一致すれば OK、ずれれば NG と、ずれの大きさを表示する。終了コードは全部一致なら0、それ以外は1。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)。
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import mujoco  # noqa: E402
from hand_sim import HandEnv  # noqa: E402
from hand_sim.arm import fk  # noqa: E402

# HandEnv は model_path を渡せば、scene.xml 以外のモデルも読める。
env = HandEnv(ROOT / "models" / "arm2link.xml")

# 確認する関節角 (q1, q2) [rad]。特別な姿勢 + 適当な姿勢。
CASES = [(0, 0), (np.pi / 2, 0), (0, np.pi / 2), (0.5, 0.8), (-1.0, 1.5), (2.0, -2.0)]
TOL = 1e-6  # 許容する誤差 [m]。これより小さいずれなら「一致」とみなす。

n_ng = 0
for q1, q2 in CASES:
    # qpos(関節の位置)に角度を直接入れ、mj_forward で手先などの座標を計算し直す。
    # (時間は進めない。reset() のコメントにある mj_forward と同じ使い方)
    env.data.qpos[:] = [q1, q2]
    mujoco.mj_forward(env.model, env.data)
    expected = env.data.site("tip").xpos[:2]   # [:2] は x, y だけ取り出す(z は高さで不要)

    try:
        got = np.asarray(fk(q1, q2), dtype=float)
    except NotImplementedError as e:
        # try/except: 例外(エラー)が起きても止まらず、except 側の処理に移る書き方。
        # [頻度] 定番。ここでは「未実装」を分かりやすい表示にするために使っている。
        print(f"{e}. src/hand_sim/arm.py に書いてください。")
        sys.exit(1)

    err = np.linalg.norm(got - expected)       # 2点間の距離 = ずれの大きさ [m]
    ok = err < TOL
    n_ng += 0 if ok else 1
    print(f"{'OK' if ok else 'NG'}  q=({q1:+.3f}, {q2:+.3f})  "
          f"fk={got.round(4)}  MuJoCo={expected.round(4)}  誤差={err:.2e} m")

env.close()
print("全部一致" if n_ng == 0 else f"{n_ng} 件ずれています")
sys.exit(0 if n_ng == 0 else 1)
