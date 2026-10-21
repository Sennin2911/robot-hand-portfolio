# 手先位置の実測(新計画の工程1)

sim が予測する手先の位置と、実機で測った位置を比べる。実機は `scripts/real_send_pose.py` で動かす(`plan_measure_points.py` が生成)。

## 測り方
- **基準は肩の回転軸**(真上から見て、回る部分の中心)。机の面から高さを測る。
- **実測する点は「指先」= 固定側の指の先端**(グリッパを下から見て、動かない方の指のいちばん先)。
- IK の目標は `gripperframe`(指先から軸に沿って約 2.8 cm 手前の点)。定規で測りにくいので、実測は指先にして、sim の指先の位置を予測として使う。
- 向きは、正面(軸から前方)を 0 として、**左が +、右が −**(sim の向き)。実機の `shoulder_pan` は右が + なので、コマンドの値とは符号が逆。
- 差 = 実測 − 予測。予測は「指先の予測」の列。向きは指が軸からわずかに横にずれているため、0 にならない。
- 実機の値の gripper は 10(端から離れた値)で固定。
- 高さの基準は机の面(sim の床 z=0)。sim の土台の底が z=0 に合っているかは未確認なので、最初の点で高さの差が一定に出たら、その分は土台の高さの違いを疑う。
- 土台の後端から肩の軸までは、sim で約 6.1 cm(作者が実機で測った約 6 cm と一致)。

## 測定表
| 点 | IK の目標 gripperframe(半径 / 向き / 高さ) | 指先の予測 半径 [cm] | 向き [度] | 高さ [cm] | 実測の半径 | 実測の向き | 実測の高さ | 差(半径/向き/高さ) | メモ |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 22 cm / +0° / 10 cm | 23.9 | +2.7 | 5.6 | | | | | |
| 2 | 15 cm / +0° / 10 cm | 15.0 | +4.3 | 5.2 | | | | | |
| 3 | 28 cm / +0° / 10 cm | 29.5 | +2.2 | 5.4 | | | | | |
| 4 | 22 cm / -60° / 10 cm | 23.8 | -57.8 | 5.5 | | | | | |
| 5 | 22 cm / +60° / 10 cm | 23.9 | +63.2 | 5.6 | | | | | |
| 6 | 28 cm / +60° / 10 cm | 29.5 | +62.6 | 5.5 | | | | | |
| 7 | 15 cm / -60° / 10 cm | 14.9 | -56.6 | 5.1 | | | | | |
| 8 | 22 cm / +0° / 8 cm | 23.3 | +2.8 | 3.4 | | | | | |
| 9 | 15 cm / +30° / 8 cm | 14.7 | +34.8 | 3.2 | | | | | |
| 10 | 28 cm / -30° / 8 cm | 29.1 | -28.0 | 3.3 | | | | | |

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
# 点8: r=22 cm, 向き=+0°, z=8 cm
python ~/ドキュメント/robot-hand-portfolio/scripts/real_send_pose.py --pose 0.0 -56.3 85.0 -8.6 0.0 10.0 --hold 30
# 点9: r=15 cm, 向き=+30°, z=8 cm
python ~/ドキュメント/robot-hand-portfolio/scripts/real_send_pose.py --pose -29.9 -98.8 73.3 66.2 -1.7 10.0 --hold 30
# 点10: r=28 cm, 向き=-30°, z=8 cm
python ~/ドキュメント/robot-hand-portfolio/scripts/real_send_pose.py --pose 30.0 -14.2 54.3 -17.1 1.5 10.0 --hold 30
```
