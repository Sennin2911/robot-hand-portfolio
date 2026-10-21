"""【概要】実機のウェブカメラの接続を確認する。つながっているカメラの一覧を出し、画像を 1 枚撮って、保存し、明るさ・色の統計と、撮影の速さを表示する。
画像は outputs/webcam_test.png に出る(outputs/ は git に入らない。カメラに写ったものが、記録に残らないように)。

使い方(この .venv で実行):
    .venv/bin/python scripts/check_webcam.py                      # 既定: /dev/video2(最初に見つかったウェブカメラ)
    .venv/bin/python scripts/check_webcam.py --device /dev/video2 --manual   # 露出と色を固定にしてから撮る
    .venv/bin/python scripts/check_webcam.py --size 1280 720
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)。
sys.path.insert(0, str(ROOT / "src"))

import imageio.v3 as iio  # noqa: E402
from hand_sim.webcam import grab_frame, list_cameras, set_manual_exposure  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--device", default=None, help="カメラのファイル名。省略すると、MiraBox(キャプチャカード)以外で、最初に見つかったカメラ")
parser.add_argument("--size", type=int, nargs=2, default=[1920, 1080], metavar=("W", "H"), help="画像の大きさ(幅 高さ)")
parser.add_argument("--exposure", type=int, default=300, help="--manual のときの露出時間(0.1 ミリ秒単位、1〜5000)。大きいほど明るい")
parser.add_argument("--gain", type=int, default=32, help="--manual のときのゲイン(0〜63、このカメラの既定は 32)。大きいほど明るいが、ざらつく")
parser.add_argument("--manual", action="store_true", help="撮る前に、露出とホワイトバランスを固定にする")
args = parser.parse_args()

print("つながっているカメラ:")
cameras = list_cameras()
for name, devices in cameras:
    print(f"  {name}: {', '.join(devices)}")
device = args.device
if device is None:
    # キャプチャカード(HDMI 入力用。ウェブカメラではない)を除いて、最初のカメラの、最初の /dev/video を使う。
    candidates = [d for name, ds in cameras if "MiraBox" not in name and "Capture" not in name for d in ds[:1]]
    if not candidates:
        print("ウェブカメラが見つからない(USB につないで、もう一度)")
        sys.exit(1)
    device = candidates[0]
print(f"使うカメラ: {device}")

if args.manual:
    set_manual_exposure(device, args.exposure, gain=args.gain)
    print("露出とホワイトバランスを固定にした")

t0 = time.perf_counter()
frame = grab_frame(device, tuple(args.size))
dt = time.perf_counter() - t0
out = ROOT / "outputs" / "webcam_test.png"
out.parent.mkdir(parents=True, exist_ok=True)
iio.imwrite(out, frame)
h, w, _ = frame.shape
print(f"画像: {w} × {h}、起動から 30 コマ読むまで {dt:.2f} 秒(30 fps なら約 1 秒)")
print(f"明るさ(0〜255): 平均 {frame.mean():.0f}、最小 {frame.min()}、最大 {frame.max()}。色の平均(R, G, B): {frame[:, :, 0].mean():.0f}, {frame[:, :, 1].mean():.0f}, {frame[:, :, 2].mean():.0f}")
print(f"保存: {out.relative_to(ROOT)}")
