"""【概要】実機で立方体を掴んだとき、「閉じた指の中で、立方体の中心が、手(gripperframe)の座標のどこにあるか」を、カメラで測った値から、
sim との差(= 実機の固定側の指の面の位置のずれ)を求めて、calibration/grasp_site_offset.json に貯める。次の計画(plan_pick_test.py)は、この値を自動で使う。

考え方: 指を閉じると、動く側の指が、立方体を固定側の指へ押しつける。だから、閉じたあとの立方体は、固定側の指の面から、半辺(15 mm)の位置に止まる
(指の開閉の向き z の位置が決まる。sim の値は CUBE_IN_HAND の z = 14.8 mm)。この位置は、掴む前の位置のずれによらない「物理的な基準」になる。
実機で測った z が sim の値と違うぶんが、実機の指の面と sim の指の面の差。差を、掴む前の位置(CUBE_IN_SITE)にも同じだけ足せば、
固定側の指の余裕(sim で決めた 16 mm)が、実機でも保たれる。
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]  # Path(__file__).resolve(): このファイルの絶対パス。.parents[2]: 2 つ上のフォルダ(プロジェクトの直下)。[頻度] 定番
FILE = ROOT / "calibration" / "grasp_site_offset.json"
KEEP = 10  # 直近の何回ぶんの測定で中央値を取るか
LIMIT_M = 0.012  # 適用するずれの上限 [m]。これより大きい値は、測り間違いを疑って、上限で止める
REJECT_M = 0.025  # 1 回の測定が、これより大きく sim とずれていたら、測り間違いとして、貯めない [m]


def _read():
    return json.loads(FILE.read_text()) if FILE.exists() else {"samples": []}


def load_site_offset():
    """計画に使う、手の座標(x, y, z)のずれ [m] を、長さ 3 の配列で返す。まだ測定が無ければ 0(sim どおり)。"""
    samples = _read()["samples"]
    if not samples:
        return np.zeros(3)
    # np.median: 中央値。外れ値(1 回の測り間違い)に引きずられにくい。[頻度] 測定値をまとめるときの定番
    z = float(np.median([s["delta_z_m"] for s in samples[-KEEP:]]))
    return np.array([0.0, 0.0, float(np.clip(z, -LIMIT_M, LIMIT_M))])


def add_sample(delta_z_m, info):
    """1 回の測定(実機の z − sim の z)を貯めて、貯めたあとの、計画に使う値 [m] を返す。
    貯めない(測り間違いが疑わしい)ときは、(None, 理由) を返す。info: あとで見返すための記録(色、位置、どの手の記録かなど)。"""
    if abs(delta_z_m) > REJECT_M:
        return None, f"差が {delta_z_m * 1000:+.1f} mm と大きすぎる(±{REJECT_M * 1000:.0f} mm まで)。立方体が指に隠れた、掴めていないなどを疑って、貯めない"
    data = _read()
    data["samples"].append({"delta_z_m": float(delta_z_m), **info})
    FILE.parent.mkdir(exist_ok=True)
    FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    return load_site_offset()[2], None
