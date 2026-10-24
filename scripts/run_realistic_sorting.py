"""【概要】sim の仕分け試験に、実機で測った誤差を入れて、成功率の見込みを出す(docs/sorting_real_errors.md)。
2 種類を実行する:
  (A) 条件: 実機で測った誤差(到達・位置の出し方・向きなど。docs/reach_test.md)を中心に、「向きの偏りが外れた場合」「2 倍悪い」「4 倍悪い」。
  (B) 掃引: 誤差を 1 つずつ(位置のランダムなずれ、位置の偏り(x 方向・y 方向)、掴む高さの偏り、向きのずれ、向きの偏り)増やして、成功率が落ち始める大きさを探す。
立方体の配置は、全条件で同じ(配置用と、誤差用の乱数を分けてある)なので、条件どうしを比べられる。成功率には、95% の信頼区間(ウィルソンの方法)を付ける。
結果は outputs/realistic_sorting.json に保存し、図 outputs/realistic_sorting.png を作る。

使い方(この .venv で実行。全部で 20〜30 分):
    .venv/bin/python scripts/run_realistic_sorting.py [--episodes 100] [--sweep-episodes 40]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)
if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "egl")  # mujoco を import する「前」に設定する(import の時点で描画方式が決まるため)

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # 画面を出さずにファイルだけ作る
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from hand_sim.sorting import IMG_H, IMG_W, load_sort_model, new_episode, random_layout, run_episode  # noqa: E402

plt.rcParams["font.family"] = "Noto Sans CJK JP"  # 日本語を表示するフォント

parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=100, help="(A) 条件ごとのエピソード数(1 エピソード = 3 個)")
parser.add_argument("--sweep-episodes", type=int, default=40, help="(B) 掃引の 1 点ごとのエピソード数")
args = parser.parse_args()

model = load_sort_model(ROOT / "models" / "so101" / "scene_sort.xml")
renderer = mujoco.Renderer(model, IMG_H, IMG_W)


def wilson(k, n, z=1.96):
    """成功 k 回 / n 回 の、95% 信頼区間(ウィルソンの方法)。成功率が 0% や 100% でも、区間が 0 幅にならない。[頻度] 成功率を報告するときの定番"""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def run(err_model, episodes, seed=0):
    """1 つの条件で、episodes エピソードを実行して、集計の辞書を返す。配置は seed だけで決まる(誤差の乱数とは別)。"""
    rng_layout = np.random.default_rng(seed)
    rng_noise = np.random.default_rng(seed + 1000)
    first_ok = final_ok = n = 0
    fails = {}
    det_err = []
    for _ in range(episodes):
        data = new_episode(model, random_layout(rng_layout))
        recs = run_episode(model, renderer, rng_noise, retries=1, data=data, err_model=err_model)
        for r in recs:
            if r["attempt"] == 1:
                n += 1
                first_ok += bool(r["success"])
                final_ok += bool(r["final_success"])
                det_err.append(r["det_err_mm"])
            if not r["success"]:
                fails[r["fail_class"]] = fails.get(r["fail_class"], 0) + 1  # 失敗の種類ごとに数える(やり直しの失敗も数える)
    return {"n": n, "first": first_ok / n, "first_ci": wilson(first_ok, n), "final": final_ok / n, "final_ci": wilson(final_ok, n),
            "fails": fails, "det_err_mm": float(np.nanmean(det_err))}


def em(sigma=0.0, bias=(0.0, 0.0), z=0.0, ysig=0.0, ybias=0.0):
    return {"pos_sigma_mm": sigma, "pos_bias_mm": bias, "z_bias_mm": z, "yaw_sigma_deg": ysig, "yaw_bias_deg": ybias}


# (A) 実機で測った誤差(docs/reach_test.md): 到達(関節の読みで補正後)偏り (0.8, −1.1) mm・ばらつき 1.6 mm、高さの偏り −2 mm、画像から位置を出す誤差 約 1 mm(+ 高さ・重心の誤差)、
#     向き(手の向きの残り 1〜3°、立方体の向きの検出)。「向きの偏り 6°」は、wrist_roll の +5.7° が関節のずれだった場合の、グリッパーの回り。
CONDITIONS = [
    ("誤差なし", em()),
    ("実機の測定値(中心)", em(2.0, (1.0, -1.0), -2.0, 3.0, 0.0)),
    ("向きの偏りが外れた場合(+6°)", em(2.0, (1.0, -1.0), -2.0, 3.0, 6.0)),
    ("2 倍悪い", em(4.0, (2.0, -2.0), -4.0, 6.0, 0.0)),
    ("4 倍悪い", em(8.0, (4.0, -4.0), -8.0, 12.0, 0.0)),
]
SWEEPS = {
    "位置のランダムなずれ σ [mm](x, y それぞれ)": ("sigma", [0, 2, 4, 6, 8, 10, 12], lambda v: em(sigma=v)),
    "位置の偏り [mm](x 方向)": ("bias_x", [0, 2, 4, 6, 8, 10, 12, 14], lambda v: em(bias=(v, 0.0))),
    "位置の偏り [mm](y 方向)": ("bias_y", [0, 2, 4, 6, 8, 10, 12, 14], lambda v: em(bias=(0.0, v))),
    "掴む高さの偏り [mm](負 = 低い)": ("z", [-12, -8, -4, 0, 4, 8, 12], lambda v: em(z=v)),
    "向きのランダムなずれ σ [度]": ("yaw_sigma", [0, 3, 6, 10, 15, 20], lambda v: em(ysig=v)),
    "向きの偏り [度]": ("yaw_bias", [0, 3, 6, 10, 15, 20, 30], lambda v: em(ybias=v)),
}

t0 = time.time()
result = {"conditions": {}, "sweeps": {}}
print(f"(A) 条件(各 {args.episodes} エピソード = {args.episodes * 3} 個)")
for name, e in CONDITIONS:
    r = run(e, args.episodes)
    result["conditions"][name] = {"err_model": e, **r}
    print(f"  {name:28s} 1 回目 {r['first'] * 100:5.1f}%({r['first_ci'][0] * 100:.0f}〜{r['first_ci'][1] * 100:.0f}%)、やり直し込み {r['final'] * 100:5.1f}%、位置の誤差 {r['det_err_mm']:.1f} mm、失敗 {r['fails']}  [{time.time() - t0:.0f} 秒]")
print(f"\n(B) 掃引(1 点あたり {args.sweep_episodes} エピソード = {args.sweep_episodes * 3} 個)")
for title, (key, values, make) in SWEEPS.items():
    rows = []
    for v in values:
        r = run(make(v), args.sweep_episodes)
        rows.append({"value": v, **r})
    result["sweeps"][title] = rows
    print(f"  {title}: " + "  ".join(f"{row['value']}→{row['first'] * 100:.0f}%" for row in rows) + f"  [{time.time() - t0:.0f} 秒]")

(ROOT / "outputs" / "realistic_sorting.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))

# 図: 掃引(6 枚)。縦軸 = 1 回目の成功率(誤差の棒は 95% 信頼区間)。実機で測った値の位置に、縦線を引く。
MEASURED = {"位置のランダムなずれ σ [mm](x, y それぞれ)": 2.0, "位置の偏り [mm](x 方向)": 1.0, "位置の偏り [mm](y 方向)": 1.0,
            "掴む高さの偏り [mm](負 = 低い)": -2.0, "向きのランダムなずれ σ [度]": 3.0, "向きの偏り [度]": 0.0}
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, (title, rows) in zip(axes.ravel(), result["sweeps"].items()):
    xs = [r["value"] for r in rows]
    ys = np.array([r["first"] for r in rows]) * 100
    lo = np.clip(ys - np.array([r["first_ci"][0] for r in rows]) * 100, 0, None)  # 丸め誤差で、わずかに負になるのを防ぐ(誤差の棒は、0 以上)
    hi = np.clip(np.array([r["first_ci"][1] for r in rows]) * 100 - ys, 0, None)
    ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", capsize=3, color="#2a6fb0")
    ax.axvline(MEASURED[title], color="#d9534f", ls="--", label="実機で測った値(目安)")
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("1 回目の成功率 [%]")
    ax.set_ylim(-3, 103)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
fig.suptitle("sim の仕分け: 誤差を 1 つずつ増やしたときの成功率(青 = 結果、赤の破線 = 実機で測った値)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(ROOT / "outputs" / "realistic_sorting.png", dpi=100)
print(f"\n保存: outputs/realistic_sorting.json、outputs/realistic_sorting.png  [{time.time() - t0:.0f} 秒]")
