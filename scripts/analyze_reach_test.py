"""【概要】到達精度のテストの結果を解析する。各点で「カメラの画像から出した手先の位置」と「目標の位置」の差(到達の誤差)を、点ごとと全体で出す。
計画を複数渡すと(反復 1、関節の読みで補正した R1b、カメラで補正した R2 など)、同じ目標について、前の計画からの誤差の変化を比べる。
誤差は 2 つに分けて出す: (a) 関節の読みの差による分 = sim(実際の関節の読み)− 目標(サーボが指令に届かない分。重さによる定常偏差)、(b) 残り = 画像 − sim(実際の読み)(sim と実機の形の違い・測定の誤差)。
  誤差 = 測った手先の位置 − 目標の位置(世界座標 mm)。測定そのものにも、カメラと板の誤差(2〜4 mm 程度)が入っている。
  水平(x, y)と高さ(z)を分けて出す。掴む動作では、水平の誤差が、立方体を外す原因になる(許容は約 14 mm)。

使い方(この .venv で実行):
    .venv/bin/python scripts/analyze_reach_test.py                                                   # 反復 1 だけ
    .venv/bin/python scripts/analyze_reach_test.py outputs/reach_plan.json outputs/reach_plan2.json  # 反復 1 と 2 を比べる
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")  # mujoco を import する「前」に設定する(import の時点で描画方式が決まるため)
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)
from hand_sim.sim_real import JOINTS, real_to_sim  # noqa: E402

_model = mujoco.MjModel.from_xml_path(str(ROOT / "models/so101/so101_new_calib.xml"))
_data = mujoco.MjData(_model)
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


def sim_hand(obs):
    """実機の関節角 obs({"名前.pos": 度})から、sim の順運動学で出した手先の位置 [mm]。"""
    q = real_to_sim(_model, obs)
    for j, n in enumerate(JOINTS):
        _data.qpos[_model.joint(n).qposadr] = q[j]
    mujoco.mj_forward(_model, _data)
    return _data.site_xpos[_model.site("gripperframe").id].copy() * 1000


def decompose(plan):
    """誤差を 2 つに分ける: (a) 関節の読みの差による分 = sim(実際の読み) − 目標(サーボが指令に届かない分)、(b) 残り = 画像 − sim(実際の読み)(sim と実機の形の違い・測定の誤差)。"""
    out = {}
    for t in plan["targets"]:
        s = samples.get(f"{plan['tag']}_{t['id']}")
        if s is not None:
            p_act = sim_hand({k + ".pos": v for k, v in s["joints"].items()})
            tgt = np.array(t["target_xyz"]) * 1000
            out[t["id"]] = (p_act - tgt, np.array(s["hand_pos_m"]) * 1000 - p_act)
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

for p in plans:
    dec = decompose(p)
    if dec:
        A = np.array([a for a, _ in dec.values()])
        B = np.array([b for _, b in dec.values()])
        print(f"\n{p['tag']} の誤差の分解(平均の大きさ): 関節の読みの差による分(サーボが指令に届かない) 水平 {np.linalg.norm(A[:, :2], axis=1).mean():.1f} mm、高さ {np.abs(A[:, 2]).mean():.1f} mm"
              f" / 残り(形の違い・測定) 水平 {np.linalg.norm(B[:, :2], axis=1).mean():.1f} mm、高さ {np.abs(B[:, 2]).mean():.1f} mm")

if len(plans) >= 2:
    for k in range(1, len(plans)):
        a_, b_ = errs[k - 1], errs[k]
        common = sorted(set(a_) & set(b_))
        if common:
            a = np.array([np.linalg.norm(a_[i][:2]) for i in common]); b = np.array([np.linalg.norm(b_[i][:2]) for i in common])
            az = np.array([abs(a_[i][2]) for i in common]); bz = np.array([abs(b_[i][2]) for i in common])
            print(f"\n=== {plans[k - 1]['tag']} → {plans[k]['tag']}(同じ目標 {len(common)} 点)。水平の誤差: 平均 {a.mean():.1f} → {b.mean():.1f} mm、最大 {a.max():.1f} → {b.max():.1f} mm。"
                  f"高さの誤差(絶対値): 平均 {az.mean():.1f} → {bz.mean():.1f} mm ===")
            for i, x, y in zip(common, a, b):
                print(f"  点 {i}: 水平 {x:5.1f} → {y:5.1f} mm")
