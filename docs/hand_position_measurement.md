# 手先位置の実測(新計画の工程1)

sim が予測する手先の位置と、実機で測った位置を比べる。実機は `scripts/real_send_pose.py` で動かす(`plan_measure_points.py` が生成)。

## 測り方
- **基準は肩の回転軸**(真上から見て、回る部分の中心)。机の面から高さを測る。
- **測る点は「手先」= 固定側の指(動かない方の指)の先端**。sim の `gripperframe`(IK の目標の点)は、この指の先端にある(最も近いメッシュの頂点まで約 0.4 cm)。
- 動く方の指や手首、指のつけ根ではない(取り違えると、半径・高さが大きくずれる)。sim の画像は `scripts/render_sim_pose.py` で出せる(赤い球が手先)。
- 向きは、正面(軸から前方)を 0 として、**左が +、右が −**(sim の向き)。実機の `shoulder_pan` は右が + なので、コマンドの値とは符号が逆。
- 差 = 実測 − 予測。予測は、半径・向き・高さの列(sim の IK が届いた位置)。
- 実機の値の gripper は 10(端から離れた値)で固定。
- 高さの基準は机の面(sim の床 z=0)。sim の土台の底は z=−0.2 cm で、床にほぼ接している。
- 土台の後端から肩の軸までは、sim で約 7.0 cm(作者が実機で測ったのは約 6 cm)。

## 測定表
| 点 | 予測の半径 [cm] | 予測の向き [度] | 予測の高さ [cm] | IK の誤差 [mm] | 実測の半径 | 実測の向き | 実測の高さ | 差(半径/向き/高さ) | メモ |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 22 | +0 | 10 | 0.10 | | | | | |
| 2 | 15 | +0 | 10 | 0.03 | | | | | |
| 3 | 28 | +0 | 10 | 0.03 | | | | | |
| 4 | 22 | -60 | 10 | 0.07 | | | | | |
| 5 | 22 | +60 | 10 | 0.07 | | | | | |
| 6 | 28 | +60 | 10 | 0.09 | | | | | |
| 7 | 15 | -60 | 10 | 0.07 | | | | | |
| 8 | 22 | +0 | 5 | 0.08 | | | | | |
| 9 | 15 | +30 | 5 | 0.07 | | | | | |
| 10 | 28 | -30 | 5 | 0.05 | | | | | |

## 実機に送るコマンド(LeRobot の環境。腕を支えてから。30 秒保つので、その間に測る)
```
# 点1: r=22 cm, 向き=+0°, z=10 cm
python ~/ドキュメント/robot-hand-portfolio/scripts/real_send_pose.py --pose 0.0 -65.9 87.0 -7.7 0.0 10.0 --hold 30
# 点2: r=15 cm, 向き=+0°, z=10 cm
python ~/ドキュメント/robot-hand-portfolio/scripts/real_send_pose.py --pose 0.0 -100.0 68.0 68.3 0.0 10.0 --hold 30
# 点3: r=28 cm, 向き=+0°, z=10 cm
python ~/ドキュメント/robot-hand-portfolio/scripts/real_send_pose.py --pose 0.0 -19.8 55.3 -16.9 0.0 10.0 --hold 30
# 点4: r=22 cm, 向き=-60°, z=10 cm
python ~/ドキュメント/robot-hand-portfolio/scripts/real_send_pose.py --pose 59.9 -65.9 86.9 -7.6 3.3 10.0 --hold 30
# 点5: r=22 cm, 向き=+60°, z=10 cm
python ~/ドキュメント/robot-hand-portfolio/scripts/real_send_pose.py --pose -59.9 -65.9 86.9 -7.5 -3.4 10.0 --hold 30
# 点6: r=28 cm, 向き=+60°, z=10 cm
python ~/ドキュメント/robot-hand-portfolio/scripts/real_send_pose.py --pose -59.9 -19.9 55.3 -16.9 -3.2 10.0 --hold 30
# 点7: r=15 cm, 向き=-60°, z=10 cm
python ~/ドキュメント/robot-hand-portfolio/scripts/real_send_pose.py --pose 59.8 -97.3 66.6 68.3 3.6 10.0 --hold 30
# 点8: r=22 cm, 向き=+0°, z=5 cm
python ~/ドキュメント/robot-hand-portfolio/scripts/real_send_pose.py --pose 0.0 -39.2 79.9 -10.4 0.0 10.0 --hold 30
# 点9: r=15 cm, 向き=+30°, z=5 cm
python ~/ドキュメント/robot-hand-portfolio/scripts/real_send_pose.py --pose -29.9 -98.0 82.8 60.6 -1.7 10.0 --hold 30
# 点10: r=28 cm, 向き=-30°, z=5 cm
python ~/ドキュメント/robot-hand-portfolio/scripts/real_send_pose.py --pose 30.0 -4.3 51.5 -17.7 1.5 10.0 --hold 30
```
