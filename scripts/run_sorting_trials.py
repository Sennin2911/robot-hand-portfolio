"""【概要】部品仕分けの試験(新計画の工程3)。机の上にランダムに置いた 3 色の立方体を、カメラで見つけて、色ごとの置き場へ運ぶ。
これをエピソード(3 個ぶん)ごとに繰り返し、1 回ごとの記録を CSV に、成功率と失敗の内訳を画面に出す。
条件(カメラの位置のずれ、画像のノイズ)を変えて、失敗が出る条件を探すのにも使う。

使い方(この .venv で実行):
    .venv/bin/python scripts/run_sorting_trials.py                              # 既定: 34 エピソード(= 102 個)、カメラのずれなし
    .venv/bin/python scripts/run_sorting_trials.py --cam-err-mm 10 0            # カメラの位置を x 方向に 10 mm ずらして信じる
    .venv/bin/python scripts/run_sorting_trials.py --noise-px 2 --tag noise2    # 画素位置に標準偏差 2 画素のノイズ
記録は outputs/sorting_<tag>.csv に出る(--tag を省略すると baseline)。
"""
import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)。
sys.path.insert(0, str(ROOT / "src"))
# 画面なしで描画する設定。★mujoco を import する前に書く必要がある。Linux のときだけ設定する(どちらも理由は check_install.py 参照)。
if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from hand_sim.sorting import FAIL_CLASSES, IMG_H, IMG_W, MOVED_MM, load_sort_model, run_episode  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=34, help="エピソード数(1 エピソード = 3 個。34 で 102 個)")
parser.add_argument("--seed", type=int, default=0, help="乱数の種(同じ種なら、同じ配置になる)")
parser.add_argument("--cam-err-mm", type=float, nargs=2, default=[0.0, 0.0], metavar=("DX", "DY"), help="信じているカメラの位置の、本当の位置からのずれ [mm]")
parser.add_argument("--noise-px", type=float, default=0.0, help="画素位置に足すランダムなずれ(標準偏差)[画素]")
parser.add_argument("--retries", type=int, default=1, help="掴めなかったときの、やり直しの回数")
parser.add_argument("--tag", default="baseline", help="記録のファイル名に付ける名前")
args = parser.parse_args()

model = load_sort_model(ROOT / "models" / "so101" / "scene_sort.xml")
renderer = mujoco.Renderer(model, IMG_H, IMG_W)
rng = np.random.default_rng(args.seed)
cam_err = (args.cam_err_mm[0] / 1000, args.cam_err_mm[1] / 1000)  # mm → m

records = []
for ep in range(args.episodes):
    for rec in run_episode(model, renderer, rng, cam_err, args.noise_px, args.retries):
        rec["episode"] = ep
        records.append(rec)

out = ROOT / "outputs" / f"sorting_{args.tag}.csv"
out.parent.mkdir(parents=True, exist_ok=True)
fields = ["episode", "color", "attempt", "success", "final_success", "fail_class", "true_x", "true_y", "true_yaw_deg",
          "det_err_mm", "det_yaw_err_deg", "final_err_mm", "moved_others_mm", "cycle_s"]
with open(out, "w", newline="", encoding="utf-8") as f:  # newline="": CSV の改行が二重にならないための指定。[頻度] CSV の書き出しでは定番
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows({k: rec[k] for k in fields} for rec in records)

renderer.close()  # 描画用の GL の部品を、終了前に明示的に片づける(しないと、終了時に EGL の警告が出ることがある)

# ---- 集計 ----
n_attempt = len(records)
firsts = [r for r in records if r["attempt"] == 1]  # 1 回目だけ = 立方体 1 個につき 1 件
n_cube = len(firsts)
ok_first = sum(r["success"] for r in firsts)
ok_final = sum(r["final_success"] for r in firsts)
print(f"条件: カメラのずれ {args.cam_err_mm} mm、ノイズ {args.noise_px} 画素、やり直し {args.retries} 回まで、乱数の種 {args.seed}")
print(f"立方体 {n_cube} 個(試行 {n_attempt} 回)")
print(f"  1 回目で成功: {ok_first} / {n_cube} = {100 * ok_first / n_cube:.1f}%")
print(f"  やり直しも含めて成功: {ok_final} / {n_cube} = {100 * ok_final / n_cube:.1f}%")
print("失敗の内訳(試行ごと。やり直しの失敗も数える):")
for cls in FAIL_CLASSES:
    n = sum(r["fail_class"] == cls for r in records)
    if n:
        print(f"  {cls}: {n} 回")
if not any(r["fail_class"] for r in records):
    print("  (失敗なし)")
det = np.array([r["det_err_mm"] for r in records if not np.isnan(r["det_err_mm"])])
if len(det):
    print(f"認識の誤差: 平均 {det.mean():.1f} mm、最大 {det.max():.1f} mm / 向きの誤差: 平均 {np.nanmean([r['det_yaw_err_deg'] for r in records]):.1f}°")
print(f"所要時間(sim の時間、1 個あたり): 平均 {np.mean([r['cycle_s'] for r in records]):.1f} 秒")
moved = sum(r["moved_others_mm"] > MOVED_MM for r in records)
print(f"他の立方体を {MOVED_MM:.0f} mm 以上動かした: {moved} 回")
print("記録:", out.relative_to(ROOT))
