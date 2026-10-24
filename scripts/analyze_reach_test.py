"""【概要】到達精度のテストの結果を解析する。各点で「カメラの画像から出した手先の位置」と「目標の位置」の差(到達の誤差)を、点ごとと全体で出す。
計画を複数渡すと(反復 1 と、補正した反復 2)、同じ目標について、誤差が補正で減ったかを比べる。
  誤差 = 測った手先の位置 − 目標の位置(世界座標 mm)。測定そのものにも、カメラと板の誤差(2〜4 mm 程度)が入っている。
  水平(x, y)と高さ(z)を分けて出す。掴む動作では、水平の誤差が、立方体を外す原因になる(許容は約 14 mm)。

使い方(この .venv で実行):
    .venv/bin/python scripts/analyze_reach_test.py                                                   # 反復 1 だけ
    .venv/bin/python scripts/analyze_reach_test.py outputs/reach_plan.json outputs/reach_plan2.json  # 反復 1 と 2 を比べる
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
plan_paths = sys.argv[1:] or [str(ROOT / "outputs" / "reach_plan.json")]  # sys.argv[1:]: コマンドラインで、スクリプト名の後ろに書いた引数の並び。[頻度] 簡単なスクリプトの引数の取り方の定番
plans = [json.loads(Path(p).read_text()) for p in plan_paths]
samples = {s["label"]: s for s in json.loads((ROOT / "outputs" / "hand_samples.json").read_text())}


def errors(plan):
    """{番号: 誤差ベクトル [mm]}。測定結果が無い点は含めない。"""
    out = {}
    for t in plan["targets"]:
        s = samples.get(f"{plan['tag']}_{t['id']}")
        if s is not None:
            out[t["id"]] = (np.array(s["hand_pos_m"]) - np.array(t["target_xyz"])) * 1000
    return out


errs = [errors(p) for p in plans]
for p, e in zip(plans, errs):
    print(f"\n=== 計画 {p['tag']}(手先の高さ {p['tip_z'] * 100:.0f} cm)。測定できた点: {len(e)} / {len(p['targets'])} ===")
    print(f"{'番号':>4s}{'半径[cm]':>9s}{'方位[°]':>8s}   誤差 x, y, z [mm]            水平 [mm]   全体 [mm]")
    for t in p["targets"]:
        if t["id"] in e:
            v = e[t["id"]]
            print(f"{t['id']:4d}{t['r'] * 100:9.0f}{t['theta_right_deg']:8.0f}   {np.round(v, 1)!s:27s}{np.linalg.norm(v[:2]):9.1f}{np.linalg.norm(v):11.1f}")
    if e:
        V = np.array(list(e.values()))
        h = np.linalg.norm(V[:, :2], axis=1)
        print(f"  水平の誤差: 平均 {h.mean():.1f} mm、最大 {h.max():.1f} mm  |  高さの誤差: 平均 {V[:, 2].mean():+.1f} mm(±{V[:, 2].std():.1f})  |  全体の平均の大きさ {np.linalg.norm(V, axis=1).mean():.1f} mm")
        print(f"  偏り(平均のベクトル): x {V[:, 0].mean():+.1f}、y {V[:, 1].mean():+.1f}、z {V[:, 2].mean():+.1f} mm(これが 0 から離れていれば、全体にずれている)")

if len(plans) == 2:
    common = sorted(set(errs[0]) & set(errs[1]))
    if common:
        a = np.array([np.linalg.norm(errs[0][i][:2]) for i in common])
        b = np.array([np.linalg.norm(errs[1][i][:2]) for i in common])
        print(f"\n=== {plans[0]['tag']} → {plans[1]['tag']}(同じ目標 {len(common)} 点)。水平の誤差: 平均 {a.mean():.1f} → {b.mean():.1f} mm、最大 {a.max():.1f} → {b.max():.1f} mm ===")
        for i, x, y in zip(common, a, b):
            print(f"  点 {i}: {x:5.1f} → {y:5.1f} mm")
