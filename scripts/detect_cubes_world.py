"""【概要】ウェブカメラで 1 枚撮り、机の板からカメラの位置を求め、赤・緑・青の立方体の、机の上の位置(世界座標 x, y [m])と向き yaw を出して、
outputs/cubes_detected.json に保存する(掴む計画 plan_pick_test.py が読む)。画像は outputs/real_detect.png(印と探す範囲を描いたもの)。
立方体の中心の高さは、1 辺(30 mm)の半分(15 mm)の面に当てて換算する。板のマーカーが 4 個未満しか見えなければ、何も保存しない。

使い方(この .venv で実行): .venv/bin/python scripts/detect_cubes_world.py [--extend-down 120] [--exposure 600 --gain 0]
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from hand_sim.aruco_marker import detect_markers  # noqa: E402
from hand_sim.fixed_board import SHOULDER_AXIS_X_SIM_M, SIDE_MM, board_marker_corners_world, camera_pose_from_board  # noqa: E402
from hand_sim.cube_world import fit_cube_yaw, marker_mask, search_hull, search_roi, white_balance_by_paper  # noqa: E402
from hand_sim.real_camera import load_camera, pixel_to_world_plane_K, project_points_K, undistort_image  # noqa: E402
from hand_sim.vision import find_cube_pose_hsv  # noqa: E402
from hand_sim.webcam import grab_frames, list_cameras, set_manual_exposure  # noqa: E402

# コマンドラインの引数を受け取る仕組み。[頻度] スクリプトに「設定を変えて実行したい」ときの定番
parser = argparse.ArgumentParser()
parser.add_argument("--extend-down", type=int, default=0, metavar="PX",
                    help="探す範囲(黄色の枠)の下の辺を、画像の下方向へ PX 画素広げる(ロボットの左側 = 画像の下。既定 0)。"
                         "画像の下の端は、レンズの較正が届いていない範囲なので、広げるほど位置の精度が落ちうる")
# os.environ.get("名前", "既定値"): 環境変数を読む。無ければ既定値。[頻度] 「シェルスクリプト越しに設定を渡したい」ときの定番
# (run_real_trial.sh は、このスクリプトに引数を渡せないので、DETECT_EXPOSURE=600 scripts/run_real_trial.sh green のように環境変数で渡す)
parser.add_argument("--exposure", type=int, default=int(os.environ.get("DETECT_EXPOSURE", "300")), help="露出(0.1 ミリ秒単位。既定 300。環境変数 DETECT_EXPOSURE でも指定できる)")
parser.add_argument("--gain", type=int, default=int(os.environ.get("DETECT_GAIN", "32")), help="ゲイン(0〜63。既定 32。環境変数 DETECT_GAIN でも指定できる)")
parser.add_argument("--frames", type=int, default=int(os.environ.get("DETECT_FRAMES", "5")),
                    help="続けて撮る枚数(既定 5。環境変数 DETECT_FRAMES でも指定できる)。画素ごとの中央値を 1 枚にして使う。1 コマごとの細かいノイズで、塊の縁が揺れ、向きの検出が変わるのを減らす。1 なら、従来どおり 1 枚")
parser.add_argument("--out-json", default="cubes_detected.json", help="結果を保存する outputs/ の中のファイル名(試験の「掴む前」「置いた後」を別々に残すために変える。既定は plan_pick_test.py が読む名前)")
parser.add_argument("--out-png", default="real_detect.png", help="印と探す範囲を描いた画像を保存する outputs/ の中のファイル名")
args = parser.parse_args()

CUBE_M = 0.030  # 立方体の 1 辺(ノギスで測った値)
# 肩の軸からの水平距離 r による、掴める範囲の目安(実機の試験 60〜109 の結果から。docs/next_steps.md の「試験の範囲の限界」)。
# r < 8 cm: 手を真下に向ける姿勢が取れず「届かない」(試験 99、101 の 2 件とも失敗)。 8〜10 cm: 取れる日もあるが不安定(9 件中 4 件成功)。 10 cm 以上: 問題なし(r 10〜21 cm の有効な試験は、動作の失敗なし)。
# 遠い側の限界は未測定(実績は r 約 21 cm まで)。
R_UNREACHABLE_M = 0.08
R_UNSTABLE_M = 0.10
R_FAR_UNTESTED_M = 0.28
YAW_SPREAD_MAX_DEG = 8.0  # 1 コマごとに向きを当てはめたときの、中央の向きからの最大のずれの上限(これを超えたら、向きが揺れていると警告)
# 前回の検出結果を、先に消す。今回の検出が途中で失敗しても(暗くて見つからない、カメラが読めない、マーカーが足りないなど)、
# 古い位置が残って、plan_pick_test.py がそれで計画を作ってしまわないため。成功した色だけが、最後に書き直される。
# unlink(missing_ok=True): ファイルを消す。無くてもエラーにしない。[頻度] 「あれば消す」の定番
(ROOT / "outputs" / args.out_json).unlink(missing_ok=True)
cam = load_camera()
device = next(devs[0] for name, devs in list_cameras() if "WebCamera" in name)
# 撮る前に、露出・ホワイトバランス・ゲインを毎回この値に固定する。カメラの設定は、前に動いたスクリプトの値が残るため
# (record_hand_sample.py は、手の板を読むために露出を 100〜1600 の間で変え、その値を残す)。設定しないと、実行のたびに画像の明るさが変わる。
# 既定値は、立方体の色検出が合う条件(露出 300、ホワイトバランス 4500、ゲイン 32)。暗いときは --exposure / --gain(または環境変数)で変える。[頻度] 色で検出するスクリプトでは、毎回固定するのが定番
set_manual_exposure(device, exposure=args.exposure, white_balance=4500, gain=args.gain)
frames = grab_frames(max(args.frames, 1), device)
# np.median(axis=0): 枚数の方向(0 番目の軸)で、画素ごとの中央値を取る。ノイズで 1 コマだけ外れた値を捨てられる。[頻度] 複数枚の画像を重ねてノイズを減らすときの定番
fixed = undistort_image(np.median(frames, axis=0).astype(np.uint8), cam["K"], cam["dist"])
board = {i: m for i, m in detect_markers(fixed, cam["K"], np.zeros(5), SIDE_MM / 1000).items() if 1 <= i <= 6}
if len(board) < 4:
    sys.exit(f"板のマーカーが {len(board)} 個しか見えない(4 個以上要る)。カメラ・明るさ・立方体が隠していないかを確認する")
cam_pos, cam_R, reproj, n = camera_pose_from_board(board, cam["K"], np.zeros(5), SIDE_MM)
print(f"板のマーカー {n} 個、再投影誤差 {reproj:.2f} 画素")

# 探す範囲: 板の 6 個のマーカーと、3 つの置き場を含む枠(立方体がマーカーを隠していても、範囲が変わらない。詳しくは cube_world.search_hull)
hull = search_hull(fixed.shape, cam_pos, cam_R, cam["K"], extend_down=args.extend_down)
roi = search_roi(fixed.shape, cam_pos, cam_R, cam["K"], extend_down=args.extend_down)
# 上: 板のマーカーの黒い四角は探さない(余裕 4 画素。黒に青みが残ると「青」と誤検出したため)。ロボットの土台など、立方体を置けない範囲も探さない(詳しくは marker_mask、workspace_mask)

# 色の判定用に、板の白い紙が白になるよう色を補正した画像を使う(照明やカメラの設定で色が偏っても、立方体の色相が判定の範囲に入るように)。表示は、元の画像のまま
balanced, gains = white_balance_by_paper(fixed, cam_pos, cam_R, cam["K"])
print(f"色の補正(紙を白にする倍率 R, G, B): {np.round(gains, 2)}")

annotated = cv2.cvtColor(fixed, cv2.COLOR_RGB2BGR)
cv2.polylines(annotated, [hull.astype(np.int32)], True, (0, 255, 255), 2)
draw = {"red": (0, 0, 255), "green": (0, 200, 0), "blue": (255, 0, 0)}
out = {}
board_mask = marker_mask(fixed.shape, cam_pos, cam_R, cam["K"])  # 板のマーカーの黒い四角(立方体を探す範囲から除いている部分)
# 1 コマごとの、色の塊(向きの揺れを見るため)。画素ごとの中央値の画像を使う本番の判定とは別に、各コマで同じ色判定をする
frame_masks = {c: [] for c in ("red", "green", "blue")}
if len(frames) > 1:
    for f in frames:
        bal_f, _g = white_balance_by_paper(undistort_image(f, cam["K"], cam["dist"]), cam_pos, cam_R, cam["K"])
        for c in frame_masks:
            frame_masks[c].append(find_cube_pose_hsv(bal_f, c, roi, return_mask=True)[1])  # [1]: (姿勢, 塊) のうち塊だけ。見つからなければ None
for color in ("red", "green", "blue"):
    pose, mask = find_cube_pose_hsv(balanced, color, roi, return_mask=True)
    if pose is None:
        print(f"{color:5s}: 見つからない")
        continue
    u, v, yaw = pose
    xy = pixel_to_world_plane_K(u, v, cam_pos, cam_R, cam["K"], CUBE_M / 2)[:2]
    # 向き: 立方体の 3 次元の形を画像に映して、見えている塊と重なりが最大になる向きを探す(cube_world.fit_cube_yaw)。
    # 旧方式(塊の中で明るい上位 40% の画素 = 上面、と見なして向きを出す)は、上面と側面が同じ色で明るさでは分けられず、同じ画像でも 41° ↔ 67° と揺れた。
    # 位置(x, y)は、これまでどおり塊の外接長方形の中心(定規で測った位置との差 2 mm 前後)。形の当てはめの中心とは、数 mm 違う。
    yaw_world, _fit_center, fit_iou, yaw_contrast = fit_cube_yaw(mask, cam_pos, cam_R, cam["K"], pixel_to_world_plane_K(u, v, cam_pos, cam_R, cam["K"], CUBE_M / 2)[:2])
    # 向きの揺れ: 撮った 1 コマごとに、同じ当てはめをして、上で出した向きからのずれの最大を見る(0〜90° は 90° で一周するので、差は 90° で折り返して測る)。
    # 同じ置き方で撮り直すと向きが 20° 近く変わる(緑)のを、1 回の検出の中で見つけるため。コマが 1 枚なら測れない(None)。
    spread = None
    if len(frame_masks) > 1:
        yaws = []
        for fm in frame_masks[color]:
            if fm is not None:
                yaws.append(fit_cube_yaw(fm, cam_pos, cam_R, cam["K"], pixel_to_world_plane_K(u, v, cam_pos, cam_R, cam["K"], CUBE_M / 2)[:2])[0])
        if len(yaws) > 1:
            diffs = [abs((y - yaw_world + 45) % 90 - 45) for y in yaws]  # (差 + 45) を 90 で割った余り − 45: 差を −45〜45° に収める。%: 割った余り。[頻度] 角度の折り返しの定番
            spread = float(max(diffs))
            if spread > YAW_SPREAD_MAX_DEG:
                print(f"  (注意: {color} の向きが揺れている。撮った {len(yaws)} コマごとの向きが、最大 {spread:.0f}° ずれた(目安 {YAW_SPREAD_MAX_DEG:.0f}°)。±10〜20° の誤差がありうる。置き直す、または撮り直す)")
    if fit_iou < 0.5:
        print(f"  (注意: {color} の形の当てはめの重なり IoU = {fit_iou:.2f} と低い。向きが当てにならない。塊が欠けている・他の物と混ざっている可能性)")
    # 立方体が板のマーカーの上(または、すぐ隣)にあるか: 塊を 6 画素太らせて、マーカーの領域と重なる画素を数える。
    # 塊は、マーカーの領域を除いて作っているので、マーカーに乗った立方体は、その部分が直線で欠ける。すると、向きが 0° / 90° に偏り、位置も数 mm〜10 mm ずれる(試験 84、85、91、92 は、4 件とも向きが 4〜6° / 82〜88° だった)。
    touch = int((cv2.dilate(mask.astype(np.uint8), np.ones((13, 13), np.uint8)).astype(bool) & board_mask).sum())
    on_marker = touch > 40
    if on_marker:
        print(f"  (注意: {color} の立方体は、板のマーカーに重なっている・接している(接する画素 {touch})。位置と向きが信用できない。マーカーから 30 mm 以上離して置き直す)")
    # 肩の軸からの距離。sim の世界座標の原点は、肩の軸の 38.8 mm 後ろ(fixed_board.py)なので、その分を引いてから測る。math.hypot(a, b): √(a²+b²)。[頻度] 距離の計算の定番
    r = math.hypot(xy[0] - SHOULDER_AXIS_X_SIM_M, xy[1])
    if r < R_UNREACHABLE_M:
        reach = "unreachable"
        print(f"  (範囲外: {color} は肩の軸から {r * 100:.1f} cm。{R_UNREACHABLE_M * 100:.0f} cm より近く、手を真下に向けられず届かない。{R_UNSTABLE_M * 100:.0f} cm 以上に置き直す)")
    elif r < R_UNSTABLE_M:
        reach = "unstable"
        print(f"  (近すぎ: {color} は肩の軸から {r * 100:.1f} cm。{R_UNREACHABLE_M * 100:.0f}〜{R_UNSTABLE_M * 100:.0f} cm は不安定(実績 9 件中 4 件成功)。{R_UNSTABLE_M * 100:.0f} cm 以上に置き直すと確実)")
    elif r > R_FAR_UNTESTED_M:
        reach = "far_untested"
        print(f"  (遠い: {color} は肩の軸から {r * 100:.1f} cm。{R_FAR_UNTESTED_M * 100:.0f} cm より遠い範囲は試験していない)")
    else:
        reach = "ok"
    out[color] = {"reach": reach, "reach_r_m": round(float(r), 4), "on_marker": on_marker, "x_m": float(xy[0]), "y_m": float(xy[1]), "yaw_deg": yaw_world, "yaw_iou": round(fit_iou, 2), "yaw_contrast": round(yaw_contrast, 3), "yaw_spread_deg": None if spread is None else round(spread, 1), "uv": [float(u), float(v)]}
    print(f"{color:5s}: 世界座標 x = {xy[0] * 1000:6.1f} mm、y = {xy[1] * 1000:6.1f} mm、向き(世界座標、0〜90°)= {yaw_world:.0f}°(画素 ({u:.0f}, {v:.0f}))")
    cv2.drawMarker(annotated, (int(u), int(v)), draw[color], cv2.MARKER_CROSS, 40, 3)
    cv2.putText(annotated, color, (int(u) + 25, int(v) - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, draw[color], 2)
cv2.imwrite(str(ROOT / "outputs" / args.out_png), annotated)
if out:
    (ROOT / "outputs" / args.out_json).write_text(json.dumps({"cam_pos_m": cam_pos.tolist(), "board_markers": n, "board_reproj_px": float(reproj), "cubes": out}, ensure_ascii=False, indent=2))
    print(f"保存: outputs/{args.out_json}、outputs/{args.out_png}")
