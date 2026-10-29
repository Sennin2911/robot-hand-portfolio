"""【概要】連続仕分け(run_real_sort.sh)の記録 outputs/real_trials_sort.csv を、エピソードごとにまとめる。1 個ずつの成功率(主な指標)と、3 個全部が入ったエピソードの割合(補助)を出す。
使い方: .venv/bin/python scripts/summarize_sort.py [--log real_trials_sort.csv] [--episode N]   (--episode を付けると、そのエピソードだけ)
無効(メモが「無効」で始まる行)は、数えない。"""
import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--log", default="real_trials_sort.csv", help="outputs/ の中の CSV 名")
parser.add_argument("--episode", type=int, help="このエピソードだけ表示する")
args = parser.parse_args()


def wilson(k, n, z=1.96):
    """成功 k 回 / n 回の、95% 信頼区間(Wilson 法)。少ない回数でも、0% や 100% に張りつかない。"""
    if n == 0:
        return 0.0, 0.0
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


rows = [r for r in csv.DictReader(open(ROOT / "outputs" / args.log)) if not r["note"].startswith("無効")]
episodes = defaultdict(list)
for r in rows:
    m = re.search(r"episode=(\d+)", r["note"])  # re.search: 文字列の中から、パターンに合う最初の部分を探す。(\d+) は数字の並び。[頻度] 文字列から数字を取り出す定番
    if m:
        episodes[int(m.group(1))].append(r)
shown = {args.episode: episodes[args.episode]} if args.episode else dict(sorted(episodes.items()))
for ep, rs in shown.items():
    print(f"エピソード {ep}: " + "  ".join(f"{r['color']}={'○' if r['success'] == 'True' else '×(' + r['fail_class'] + ')'}" for r in rs)
          + f"  → {sum(r['success'] == 'True' for r in rs)}/{len(rs)}")
all_rows = [r for rs in episodes.values() for r in rs]
if all_rows:
    k, n = sum(r["success"] == "True" for r in all_rows), len(all_rows)
    lo, hi = wilson(k, n)
    print(f"\n1 個ずつの成功率(主な指標): {k}/{n} = {100 * k / n:.0f}%(95% 区間 {100 * lo:.0f}〜{100 * hi:.0f}%)")
    full = [ep for ep, rs in episodes.items() if len(rs) == 3 and all(r["success"] == "True" for r in rs)]
    print(f"3 個全部が入ったエピソード(補助): {len(full)}/{len(episodes)}")
    for c in ("red", "green", "blue"):
        rc = [r for r in all_rows if r["color"] == c]
        if rc:
            print(f"  {c}: {sum(r['success'] == 'True' for r in rc)}/{len(rc)}")
