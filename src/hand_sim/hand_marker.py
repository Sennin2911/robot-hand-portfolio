"""【概要】手(固定側の指)に付けた ArUco 板の「付き方」と、マーカーの姿勢 → 手先(gripperframe)の姿勢の変換。
3 つの座標系をつなぐ:
  ② 手の基準 = gripperframe(固定側の指の先端。x = 指先が向く方向、y = 左(指先を遠ざけ、動く指を上にしたとき)、z = 動く指がある方向)
  ③ マーカーの軸 = 黒枠の中心。x = 紙の右、y = 紙の上、z = 紙の表から外へ出る向き(aruco_marker.marker_object_points と同じ約束)
  ① 世界座標 = 肩の軸の真下の机の点。x = 正面、y = 左、z = 上
作者の測定(2026-10-24): マーカー中心は ② で (-94.43, +20.5, -48.01) mm(y は符号を反転して確定)、面の垂線 = ②の +y、紙の上 = ②の -z。
  - hand_pose_from_marker: ①で見たマーカーの姿勢 → ①での手先の姿勢。
  - marker_pose_from_hand: 逆(手先の姿勢 → マーカーの姿勢)。sim で「マーカーがどこに写るはずか」を出すのに使う。
"""
import numpy as np

# マーカーの中心の位置 [m]。② の座標で表した値(作者の測定 A2)。
# y の符号は 2026-10-24 に反転した(+20.5)。測ったとき、②の y の向きが分かっておらず、逆に読んでいた。作者が「板は固定側の指の左(+y)側」と確認し、
# pan 振りの画像でも、反転すると手先の水平の差が 30〜45 mm → 2〜8 mm に縮んだ。
T_HAND_MARKER = np.array([-94.43, 20.5, -48.01]) * 1e-3

# マーカーの向き。列 = マーカーの x, y, z 軸を、② の座標で表したもの(= 「③ → ②」の回転行列)。
#   マーカーの x(紙の右)= ②の +x  /  y(紙の上)= ②の -z(測定 A4)  /  z(垂線)= ②の +y(測定 A3)
# 右手系になっている(x × y = z)ことは、check_hand_marker.py でも確かめる。
R_HAND_MARKER = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, -1.0, 0.0],
])
# 行列の見方: 1 列目 = (1,0,0) = x 軸、2 列目 = (0,0,-1) = y 軸、3 列目 = (0,1,0) = z 軸。


def hand_pose_from_marker(R_w_marker, p_w_marker):
    """①(世界)で見たマーカーの向き R_w_marker (3×3) と中心の位置 p_w_marker (3, m) から、①での手先の向き (3×3) と位置 (3, m) を返す。
    手先の向きの列は、gripperframe の x, y, z 軸が世界でどちらを向くか(so101_ik.py の target_R と同じ約束)。
    計算: マーカーの向き = 手先の向き @ R_HAND_MARKER なので、手先の向き = マーカーの向き @ R_HAND_MARKER の転置(.T = 逆回転)。
    @ は行列の積(掛け算)。[頻度] numpy の姿勢計算では定番。"""
    R_w_hand = np.asarray(R_w_marker) @ R_HAND_MARKER.T
    p_w_hand = np.asarray(p_w_marker, dtype=float) - R_w_hand @ T_HAND_MARKER
    return R_w_hand, p_w_hand


def marker_pose_from_hand(R_w_hand, p_w_hand):
    """hand_pose_from_marker の逆。①での手先の向き・位置から、①でのマーカーの向き・中心の位置を返す。"""
    R_w_hand = np.asarray(R_w_hand)
    return R_w_hand @ R_HAND_MARKER, np.asarray(p_w_hand, dtype=float) + R_w_hand @ T_HAND_MARKER
