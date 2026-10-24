# 実機の到達精度のテスト(立方体なし。2026-10-24)

「手先を、机の上の指定した点へ送ったとき、実際にどれだけ近づくか」を、実機で測った。立方体を掴む精度の土台になる(許容は約 14 mm)。

## やり方
1. **目標**: 手を真下に向け、手先(gripperframe)を、机の上 6 cm の空中の 8 点へ。肩の軸から半径 15〜23 cm、方位は右へ 15〜60°(`scripts/plan_reach_test.py`)。
   カメラが、ロボットの左前にあり、手の板(ID 0)は左向きなので、右側の点だけがカメラに見える。左側は、今のカメラの位置では測れない。
2. **指令値**: sim の IK(位置と向きを合わせる `solve_ik_pose`)で関節角を求め、`OFFSET_DEG` を引いて、実機の値にする。経路は、sim で、腕の動く部分が机から 52 mm 以上、手先が 60 mm 離れることを確認した。
3. **測定**: 着いたら、手の板と、机の板(ID 1〜6)で、手先の世界座標での位置を出す(`scripts/real_run_plan.py` → `record_hand_sample.py`)。
4. **誤差** = 測った手先の位置 − 目標(`scripts/analyze_reach_test.py`)。

## 結果
| 条件 | 水平の誤差(平均 / 最大) | 高さの誤差(絶対値の平均) | 全体(平均) |
|---|---|---|---|
| **R1**: 指令どおりに送るだけ | 4.2 / 5.2 mm | 7.2 mm | 8.4 mm |
| **R1b**: 動いたあと、関節の読みと指令の差を、指令に足して送り直す(`--refine 1`) | **1.6 / 3.5 mm** | 2.1 mm | 3.0 mm |
| **R2**: R1b に加えて、カメラで測った差を、目標から引く | 2.3 / 4.6 mm | **1.0 mm** | 2.6 mm |

## 分かったこと
- **誤差の主役は、サーボが指令に届かないこと(重さによる定常偏差)**。elbow の読みが、指令より +1.7〜1.9° 手前で止まる。R1 の誤差の約 6 割(水平 4.4 mm、高さ 5.2 mm)がこれ。
  関節の読み(エンコーダ)で、差を足して送り直す(R1b)と、ほぼ消える(水平 0.9 mm、高さ 0.7 mm)。**カメラは要らない。**
- **カメラによる 1 点ごとの補正は、水平では逆効果**(1.6 → 2.3 mm)。残った誤差(sim と実機の形の違い、測定の誤差)は、カメラの測定のノイズ(2〜3 mm。手の板は、小さく、手先から 10 cm 離れ、傾いているため)と同程度で、引くとノイズを足してしまう。
  高さは、全点に共通の偏り(−2 mm)があったので、改善した(2.1 → 1.0 mm)。使うなら、全点に共通の偏りだけを、1 つの定数として補正する。
- 誤差の分解(R1b): 関節の読みの差による分 水平 0.9 / 高さ 0.7 mm、残り(形の違い・測定)水平 2.0 / 高さ 1.6 mm。
- 掴むときの見込み: 立方体の位置の見積もりの誤差(次の章)と、到達の誤差(約 2 mm)を足しても、数 mm。許容(約 14 mm)に対して、十分な余裕。

## 画像から、机の上の位置を出す精度(`scripts/check_desk_localization.py`)
板のマーカーを、1〜2 個外して、残りでカメラの位置を求め、外したマーカーの中心の画素から、机の面(高さ 0)の位置を出して、設計値と比べた(のべ 30〜150 回)。立方体の位置を、画像から出す手順と同じ。
| 検証 | 水平の誤差(平均 / 最大) |
|---|---|
| 1 個外して、残り 5 個で予測(内挿) | 0.19〜0.22 / 0.38〜0.42 mm |
| 2 個外して、残り 4 個で予測 | 0.23〜0.26 / 0.62〜0.64 mm |
| 手前の列 3 個で、奥の列 3 個を予測(外挿) | 0.58 / 0.88 mm |
| 奥の列 3 個で、手前の列 3 個を予測(外挿) | 0.92 / 1.76 mm |

- 机の面の上の位置は、**1 mm 前後**で出せる。カメラの位置の精度、レンズの補正、板の角の検出は、立方体の位置を出す精度の上限にならない。
- 立方体では、これに、**高さの見積もり**が加わる。視線が約 45° なので、高さが 3 mm 違うと、水平の位置が約 3 mm ずれる。色の塊の中心(重心)の精度も効く(側面の混入。`camera_placement.md`)。
- 手の板の測定(手先の位置)に出る 2〜3 mm のノイズは、カメラの位置ではなく、手の板そのもの(小さい、傾いている、手先まで 10 cm の梃子)の誤差が主。

## 使い方
```
.venv/bin/python scripts/plan_reach_test.py                       # R1 の計画(カメラが要る)
~/lerobot-work/lerobot/.venv/bin/python scripts/real_run_plan.py                                   # 実行(休めの姿勢から。腕を支える)
.venv/bin/python scripts/plan_reach_test.py --clone-from outputs/reach_plan.json --tag R1b --out outputs/reach_plan_r1b.json
~/lerobot-work/lerobot/.venv/bin/python scripts/real_run_plan.py --plan outputs/reach_plan_r1b.json --refine 1
.venv/bin/python scripts/plan_reach_test.py --correct-from outputs/reach_plan_r1b.json --tag R2 --out outputs/reach_plan2.json
.venv/bin/python scripts/analyze_reach_test.py outputs/reach_plan.json outputs/reach_plan_r1b.json outputs/reach_plan2.json
.venv/bin/python scripts/check_desk_localization.py
```
安全: 経路は、番号の順に動くことを前提に、sim で確認してある(`--only` は先頭から続く番号だけ)。手先の高さは 5 cm 以上。腕を支えて、動く範囲に物が無いことを確認してから Enter。
