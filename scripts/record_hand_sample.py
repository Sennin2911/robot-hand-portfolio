"""【概要】腕を手で支えて作った 1 つの姿勢について、「関節角(実機から読む)」と「手先の位置(カメラの画像から)」を 1 組、outputs/hand_samples.json に足す。
手順(1 回の実行で全部やる): 関節角を読む → 撮影 → 関節角をもう一度読む。前後の角度が 0.5° 以上違えば、撮影中に腕が動いたので、記録しない。
関節角の読み取りは real_read_joints.py(LeRobot の環境で動くので、別のプロセスとして呼ぶ)。指令は出さず、トルクにも触れない。
カメラが USB の不調で、認識・切断を繰り返していても、認識された合間に撮れるまで、何度か試す(最大 60 秒)。
記録した組は、fit_pan_sweep.py(肩の軸の位置を求める)や、compare 系の確認に使う。

使い方(この .venv で実行):
    .venv/bin/python scripts/record_hand_sample.py --label pan0
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from hand_sim.aruco_marker import detect_markers, marker_pose_in_world  # noqa: E402
from hand_sim.fixed_board import SIDE_MM, camera_pose_from_board  # noqa: E402
from hand_sim.hand_marker import hand_pose_from_marker  # noqa: E402
from hand_sim.cube_world import locate_cube  # noqa: E402
from hand_sim.real_camera import load_camera, undistort_image  # noqa: E402
from hand_sim.webcam import grab_until, list_cameras  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--label", required=True, help="この姿勢の名前(例: pan0、pan15)")
parser.add_argument("--exposure", type=int, default=100, help="露出時間(0.1 ミリ秒単位)")
parser.add_argument("--lerobot-python", default=str(Path.home() / "lerobot-work/lerobot/.venv/bin/python"), help="LeRobot の環境の python")
parser.add_argument("--detect-cube", choices=["red", "green", "blue"],
                    help="同じ画像で、この色の立方体の位置(世界座標)も測って、記録に足す(cube_pos_m、cube_area_px)。掴んだ瞬間の、手と立方体の位置関係を調べるために使う(update_grasp_offset.py)")
parser.add_argument("--joints-file-only", action="store_true",
                    help="関節角は読まず、outputs/real_joints.json の値をそのまま使う(real_pan_sweep.py が、腕を動かしたあとに呼ぶ。2 つのプロセスが、同時に実機の通信路を開かないため)")
args = parser.parse_args()

JOINTS_FILE = ROOT / "outputs" / "real_joints.json"
SAMPLES_FILE = ROOT / "outputs" / "hand_samples.json"


def read_joints():
    """real_read_joints.py を実行して、関節角 {名前: 値} を返す(別プロセス = LeRobot の環境)。--joints-file-only のときは、保存済みのファイルを読むだけ。"""
    if args.joints_file_only:
        return json.loads(JOINTS_FILE.read_text())["joints"]
    subprocess.run([args.lerobot_python, str(ROOT / "scripts" / "real_read_joints.py")], check=True, capture_output=True)  # check=True: 失敗したら例外にする
    return json.loads(JOINTS_FILE.read_text())["joints"]


def capture():
    """ウェブカメラから 1 枚撮る。USB が切れていれば、認識されるまで待って、やり直す(最大 60 秒)。"""
    t_end = time.time() + 60
    while time.time() < t_end:
        device = next((devs[0] for name, devs in list_cameras() if "WebCamera" in name), None)
        if device:
            try:
                # 露出を短い方から試して、手の板(ID 0)と、板のマーカー 4 個以上が読める明るさを探す(部屋の明るさが、時間帯で変わるため)
                def good(im):
                    ids = detect_markers(im, load_camera()["K"], load_camera()["dist"], SIDE_MM / 1000)
                    return 0 in ids and len([i for i in ids if 1 <= i <= 6]) >= 4
                img, _used = grab_until(device, good, exposures=(args.exposure, 200, 400, 800, 1600), settle_frames=15, timeout=8)
                return img
            except Exception:  # 切断・タイムアウトなど。何が起きても、時間内ならやり直す(「何でも握りつぶす」書き方は普通は避けるが、ここは不安定な機器の再試行のため)
                pass
        time.sleep(1)
    sys.exit("ウェブカメラから撮れなかった(USB の接続を確認する)")


before = read_joints()
img = capture()
after = read_joints()
moved = max(abs(before[k] - after[k]) for k in before if k != "gripper")
if moved > 0.5:
    sys.exit(f"撮影中に腕が動いた(関節角の変化 最大 {moved:.2f}°)。記録しない。腕を支えたまま、もう一度")

cam = load_camera()
found = detect_markers(img, cam["K"], cam["dist"], SIDE_MM / 1000)
result = camera_pose_from_board(found, cam["K"], cam["dist"], SIDE_MM)
if result is None or 0 not in found:
    sys.exit(f"板(ID 1〜6)か手の板(ID 0)が見つからない(見つかった ID: {sorted(found)})。記録しない")
cam_pos, cam_R, err, n = result
R_m, p_m = marker_pose_in_world(found[0]["R"], found[0]["t"], cam_pos, cam_R)
R_h, p_h = hand_pose_from_marker(R_m, p_m)

cube_pos_m, cube_area = None, 0
if args.detect_cube:
    # 歪みを直した画像で、色で立方体を探す(板のマーカーの検出は歪み補正なしで dist を渡す方式なので、ここだけ別に補正する)
    xy, cube_area = locate_cube(undistort_image(img, cam["K"], cam["dist"]), args.detect_cube, cam_pos, cam_R, cam["K"])
    if xy is not None:
        cube_pos_m = [float(xy[0]), float(xy[1]), 0.015]  # 高さは、中心(半辺)の 15 mm と仮定した値

samples = json.loads(SAMPLES_FILE.read_text()) if SAMPLES_FILE.exists() else []
samples = [s for s in samples if s["label"] != args.label]  # 同じ名前は、上書きする
samples.append({"label": args.label, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "joints": after, "hand_pos_m": p_h.tolist(), "marker_pos_m": p_m.tolist(),
                "hand_R": R_h.tolist(), "camera_pos_m": cam_pos.tolist(), "board_reproj_px": err, "board_markers": n, "frame": "sim_world",
                "cube_pos_m": cube_pos_m, "cube_area_px": cube_area})  # frame: 座標の約束(sim の世界座標。原点は肩の軸の 38.8 mm 後ろ)
SAMPLES_FILE.write_text(json.dumps(samples, ensure_ascii=False, indent=2))
print(f"記録: {args.label}  関節角(度) " + ", ".join(f"{k} {v:.1f}" for k, v in after.items()))
if args.detect_cube:
    print("  立方体: " + (f"({cube_pos_m[0] * 1000:.1f}, {cube_pos_m[1] * 1000:.1f}) mm、面積 {cube_area} 画素" if cube_pos_m else "見つからない"))
print(f"  手先(世界座標、mm) = {np.round(p_h * 1000, 1)}、板の再投影誤差 {err:.2f} 画素({n} 個)、撮影中の関節角の変化 最大 {moved:.2f}°。記録済みの組: {len(samples)} 個")
