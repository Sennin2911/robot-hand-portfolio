"""【概要】docs/sorting_trials.md に載せる図(docs/img/sorting_*.png)を、試験の記録 (outputs/sorting_*.csv) から作る。実機・sim は動かさない。
図: (1) 条件ごとの成功率(1 回目 / やり直しも含めて)、(2) 条件ごとの失敗の内訳、(3) 認識の誤差と成功・失敗の関係。
記録は scripts/run_sorting_trials.py が作る。

使い方(この .venv で実行):
    .venv/bin/python scripts/make_sorting_figures.py
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 画面を出さずに PNG だけ作る設定。pyplot を import する前に書く(後だと効かない)。[頻度] ヘッドレスでの作図では定番
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams["font.family"] = "Noto Sans CJK JP"  # 日本語の文字化け(□□□)を避けるフォント指定
plt.rcParams["axes.unicode_minus"] = False
OUT = ROOT / "docs" / "img"
CLASSES = ["未検出", "到達不可", "把持失敗", "運搬中の落下", "置き場の外"]
# 表示する条件: (ファイル名の tag, 図の目盛りの名前)。ファイルが無い条件は飛ばす。
CONDITIONS = [("baseline", "ずれなし"), ("camx5", "x 5 mm"), ("camx10", "x 10 mm"), ("camx15", "x 15 mm"), ("camx20", "x 20 mm"),
              ("camx30", "x 30 mm"), ("camy10", "y 10 mm"), ("camy20", "y 20 mm"), ("noise1", "ノイズ 1 画素"), ("noise2", "ノイズ 2 画素"),
              ("noise4", "ノイズ 4 画素")]


def load(tag):
    """outputs/sorting_<tag>.csv を、辞書のリストとして読む。無ければ None。"""
    path = ROOT / "outputs" / f"sorting_{tag}.csv"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:  # CSV の値は全部文字列なので、使う列を数に直す
        r["success"] = r["success"] == "True"
        r["final_success"] = r["final_success"] == "True"
        r["det_err_mm"] = float(r["det_err_mm"]) if r["det_err_mm"] not in ("", "nan") else np.nan
    return rows


data = [(label, load(tag)) for tag, label in CONDITIONS]
data = [(label, rows) for label, rows in data if rows]
labels = [label for label, _ in data]
x = np.arange(len(data))

# (1) 成功率
first = [100 * np.mean([r["success"] for r in rows if r["attempt"] == "1"]) for _, rows in data]
final = [100 * np.mean([r["final_success"] for r in rows if r["attempt"] == "1"]) for _, rows in data]
fig, ax = plt.subplots(figsize=(11, 4.2))
ax.bar(x - 0.2, first, 0.4, color="#d62728", label="1 回目で成功")
ax.bar(x + 0.2, final, 0.4, color="#1f77b4", label="やり直しも含めて成功")
for i, (a, b) in enumerate(zip(first, final)):
    ax.text(i - 0.2, a + 1, f"{a:.0f}", ha="center", fontsize=8)
    ax.text(i + 0.2, b + 1, f"{b:.0f}", ha="center", fontsize=8)
ax.set_xticks(x, labels, rotation=20)
ax.set_ylabel("成功率 [%](立方体 102 個あたり)")
ax.set_ylim(0, 125)
ax.set_title("条件ごとの成功率(カメラの位置のずれ・画像のノイズを変えた)")
ax.legend(loc="upper right", ncol=2)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(OUT / "sorting_success.png", dpi=110)
plt.close(fig)

# (2) 失敗の内訳(試行ごと)
fig, ax = plt.subplots(figsize=(11, 4.2))
bottom = np.zeros(len(data))
colors = ["#7f7f7f", "#bcbd22", "#d62728", "#ff7f0e", "#9467bd"]
for cls, col in zip(CLASSES, colors):
    counts = np.array([sum(r["fail_class"] == cls for r in rows) for _, rows in data])
    ax.bar(x, counts, 0.6, bottom=bottom, color=col, label=cls)
    bottom += counts
ax.set_xticks(x, labels, rotation=20)
ax.set_ylabel("失敗の回数(試行ごと)")
ax.set_title("失敗の内訳")
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(OUT / "sorting_failures.png", dpi=110)
plt.close(fig)

# (3) 認識の誤差と成功の関係: 全条件の 1 回目の試行を、誤差の区間ごとに集計する
errs, oks = [], []
for _, rows in data:
    for r in rows:
        if r["attempt"] == "1" and not np.isnan(r["det_err_mm"]):
            errs.append(r["det_err_mm"])
            oks.append(r["success"])
errs, oks = np.array(errs), np.array(oks)
edges = [0, 2, 5, 10, 15, 20, 30, 45]
fig, ax = plt.subplots(figsize=(7.5, 4.2))
centers, rates, counts = [], [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    sel = (errs >= lo) & (errs < hi)
    if sel.sum() >= 3:
        centers.append(f"{lo}〜{hi}")
        rates.append(100 * oks[sel].mean())
        counts.append(int(sel.sum()))
ax.bar(centers, rates, color="#1f77b4")
for i, (r, n) in enumerate(zip(rates, counts)):
    ax.text(i, r + 1.5, f"{r:.0f}%\n(n={n})", ha="center", fontsize=8)
ax.set_ylim(0, 120)
ax.set_xlabel("認識の位置の誤差 [mm]")
ax.set_ylabel("1 回目で成功した割合 [%]")
ax.set_title("認識の誤差が大きいほど、失敗する")
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(OUT / "sorting_error_vs_success.png", dpi=110)
plt.close(fig)
print("作成: sorting_success.png、sorting_failures.png、sorting_error_vs_success.png")
