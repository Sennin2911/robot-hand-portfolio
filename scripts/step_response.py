"""【概要】PD制御のステップ応答を調べる。目標角を一気に切り替えたとき、アームの関節が
どう動くか(行き過ぎ・収束の速さ・発散)を、kp と kv の組み合わせごとに比べる。
結果は outputs/ に保存する:
  step_response.png   … 関節角の時間変化のグラフ
  step_response_*.csv … 組み合わせごとのログ(時刻・角度・角速度・トルク指令)
実行すると、行き過ぎ量と整定時間の一覧表も表示する。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)。
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402
# Agg: 画面に出さず、画像ファイルに書き出すだけの描画方式(画面のない環境でも動く)。
# pyplot を import する「前」に指定する必要がある(後だと反映されないことがある)。
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from hand_sim import HandEnv  # noqa: E402
from hand_sim.analysis import analyze_step  # noqa: E402
from hand_sim.controller import PDController  # noqa: E402

TARGET = np.array([0.8, -1.0])   # 目標の関節角 [q1, q2] [rad]。出発点は両方 0。
DURATION = 3.0                   # 何秒間動かすか [秒]
# 比べる (kp, kv) の組み合わせ。kv=0 は「ブレーキなし」、(20, 2) は大きすぎる kv の例。
GAINS = [(10, 0.0), (10, 0.5), (10, 1.0), (5, 0.2), (20, 0.5), (20, 2.0)]

env = HandEnv(ROOT / "models" / "arm2link.xml")
out_dir = ROOT / "outputs"
out_dir.mkdir(exist_ok=True)   # フォルダがなければ作る(あっても何もしない)

fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)   # 縦に2枚のグラフ(q1用, q2用)
print(f"{'kp':>5} {'kv':>5} | {'q1 行き過ぎ':>10} {'q1 整定':>8} | {'q2 行き過ぎ':>10} {'q2 整定':>8}  [rad, 秒]")

for kp, kv in GAINS:
    pd = PDController(kp=kp, kv=kv)
    env.reset()
    log = []   # 1ステップごとに [時刻, q1, q2, 角速度1, 角速度2, トルク指令1, トルク指令2] を貯める
    for _ in range(int(DURATION / env.dt)):
        q, qd = env.joint_pos(), env.joint_vel()
        tau = pd.torque(TARGET, q, qd)
        # 「トルク指令を出す前」の状態を記録する(その時刻の状態と、そこで出した指令の対応が取れる)。
        log.append([env.data.time, *q, *qd, *tau])   # * はリストや配列の中身を並べて展開する書き方
        env.step(tau)
    log = np.array(log)
    t = log[:, 0]   # [:, 0] は「全行の0列目」= 時刻の列
    np.savetxt(out_dir / f"step_response_kp{kp}_kv{kv}.csv", log, delimiter=",",
               header="t,q1,q2,qd1,qd2,tau1,tau2", comments="")

    # 評価(analysis.py の analyze_step が未実装のときは、表の数値を「-」にして続ける)
    try:
        r1 = analyze_step(t, log[:, 1], TARGET[0])
        r2 = analyze_step(t, log[:, 2], TARGET[1])
        print(f"{kp:>5} {kv:>5} | {r1[0]:>10.3f} {r1[1]:>8.2f} | {r2[0]:>10.3f} {r2[1]:>8.2f}")
    except NotImplementedError:
        print(f"{kp:>5} {kv:>5} | {'-':>10} {'-':>8} | {'-':>10} {'-':>8}")

    for ax, col in zip(axes, (1, 2)):
        ax.plot(t, log[:, col], label=f"kp={kp}, kv={kv}")

# 目標値の破線とグラフの体裁(日本語フォントが無い環境でも文字化けしないよう、グラフ内は英語)
for ax, target, name in zip(axes, TARGET, ("q1", "q2")):
    ax.axhline(target, color="k", linestyle="--", linewidth=1, label="target")
    ax.set_ylabel(f"{name} [rad]")
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=8, ncol=2)
axes[1].set_xlabel("time [s]")
fig.suptitle("Step response of PD control (2-link arm)")
fig.tight_layout()
fig.savefig(out_dir / "step_response.png", dpi=120)
print("saved", out_dir / "step_response.png")
env.close()
