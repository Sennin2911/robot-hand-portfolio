"""【概要】実機の手先の位置・向きを、(A) カメラの画像と、(B) 実機の関節角から sim で計算した値 の 2 通りで出し、差を表示する。腕のずれの確認が目的。
  (A) 画像: 机の板(ID 1〜6)でカメラの位置を求め、手の板(ID 0)の姿勢から、手先(gripperframe)の世界座標での位置・向きを出す。
  (B) sim : outputs/real_joints.json(real_read_joints.py で読んだ関節角)を、sim の関節角に直し、順運動学(関節角 → 手先の位置)で手先の位置・向きを出す。
      OFFSET_DEG を足した場合(実機の腕が本当に取っている姿勢)と、足さない場合の両方を出す。
撮影と関節角の読み取りの間に、腕を動かさないこと(動かしていれば、差は無意味)。

使い方(この .venv で実行。先に LeRobot の環境で real_read_joints.py を実行しておく):
    .venv/bin/python scripts/compare_hand_real_sim.py --manual --exposure 100
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")  # mujoco を import する「前」に設定する(import の時点で描画方式が決まるため)。ここでは描画しないが、他と揃えている
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)

import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from hand_sim.aruco_marker import detect_markers, marker_pose_in_world  # noqa: E402
from hand_sim.camera_pose import rotation_angle_deg  # noqa: E402
from hand_sim.fixed_board import SIDE_MM, camera_pose_from_board  # noqa: E402
from hand_sim.hand_marker import hand_pose_from_marker  # noqa: E402
from hand_sim.real_camera import load_camera  # noqa: E402
from hand_sim.sim_real import JOINTS, real_to_sim  # noqa: E402
from hand_sim.webcam import grab_frame, list_cameras, set_manual_exposure  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--image", help="カメラの代わりに読む、保存した画像")
parser.add_argument("--device", help="カメラのファイル名。省略すると、名前に WebCamera を含むカメラを探す")
parser.add_argument("--manual", action="store_true", help="露出とホワイトバランスを固定にする")
parser.add_argument("--exposure", type=int, default=100, help="--manual のときの露出時間(0.1 ミリ秒単位)")
args = parser.parse_args()

cam = load_camera()
if args.image:
    img = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
else:
    device = args.device or next((devs[0] for name, devs in list_cameras() if "WebCamera" in name), None)
    if device is None:
        sys.exit("ウェブカメラが見つからない")
    if args.manual:
        set_manual_exposure(device, args.exposure)
    img = grab_frame(device)

found = detect_markers(img, cam["K"], cam["dist"], SIDE_MM / 1000)
result = camera_pose_from_board(found, cam["K"], cam["dist"], SIDE_MM)
if result is None or 0 not in found:
    sys.exit(f"板(ID 1〜6)か手の板(ID 0)が見つからない。見つかった ID: {sorted(found)}")
cam_pos, cam_R, err, n = result
R_m, p_m = marker_pose_in_world(found[0]["R"], found[0]["t"], cam_pos, cam_R)
R_img, p_img = hand_pose_from_marker(R_m, p_m)  # (A) 画像から出した手先
print(f"(A) 画像: 板 {n} 個で再投影誤差 {err:.2f} 画素。手先 = {np.round(p_img * 1000, 1)} mm")

joints = json.loads((ROOT / "outputs" / "real_joints.json").read_text())
print(f"関節角(読んだ時刻 {joints['time']}): " + ", ".join(f"{k} {v:.1f}" for k, v in joints["joints"].items()))
obs = {f"{k}.pos": v for k, v in joints["joints"].items()}  # {"shoulder_pan.pos": 度, ...}。real_to_sim が受け取る形
model = mujoco.MjModel.from_xml_path(str(ROOT / "models/so101/so101_new_calib.xml"))
data = mujoco.MjData(model)
site_id = model.site("gripperframe").id
for label, apply_offset in (("OFFSET_DEG を足す", True), ("OFFSET_DEG を足さない", False)):
    q = real_to_sim(model, obs, apply_offset=apply_offset)
    for j, name in enumerate(JOINTS):
        data.qpos[model.joint(name).qposadr] = q[j]  # model.joint(名前).qposadr = その関節の角度が入っている qpos の番号
    mujoco.mj_forward(model, data)  # 関節角から、各部の位置を計算する(順運動学)
    p_sim = data.site_xpos[site_id].copy()
    R_sim = data.site_xmat[site_id].reshape(3, 3).copy()
    diff = (p_img - p_sim) * 1000  # 画像 − sim [mm]
    print(f"(B) sim({label}): 手先 = {np.round(p_sim * 1000, 1)} mm  → 画像 − sim = {np.round(diff, 1)} mm(距離 {np.linalg.norm(diff):.1f} mm)、"
          f"向きの差 {rotation_angle_deg(R_img, R_sim):.1f}°")
