"""【概要】実機のウェブカメラから、画像を 1 枚取り込む(OpenCV を使わず、ffmpeg を呼ぶだけの小さな部品)。
Linux では、ウェブカメラは /dev/video<番号> という名前のファイルに見える(UVC という標準方式)。ffmpeg の v4l2 入力で、そこから画像を読む。
- list_cameras: つながっているカメラの一覧(v4l2-ctl --list-devices の結果を読む)。
- grab_frame: 画像を 1 枚(高さ × 幅 × 3 の RGB、uint8)。カメラを起動した直後は、露出や色が落ち着いていないので、最初の数コマは捨てる。
- set_manual_exposure: 露出と色(ホワイトバランス)を、自動から固定に切り替える(色で立方体を見つけるので、明るさ・色が、毎回、同じに写るようにする)。
sim の画像(mujoco.Renderer の戻り値)と同じ形(RGB, uint8)で返すので、vision.py の色検出に、そのまま渡せる。
"""
import re
import subprocess

import imageio.v3 as iio  # 画像の読み込み。[頻度] 画像入出力の定番
import numpy as np


def list_cameras():
    """つながっているカメラを、[(名前, [/dev/video の名前, ...]), ...] で返す。v4l2-ctl(v4l-utils)が要る。"""
    out = subprocess.run(["v4l2-ctl", "--list-devices"], capture_output=True, text=True).stdout  # subprocess.run: 外部のコマンドを実行して、結果を受け取る。[頻度] 外部ツールを呼ぶときの定番
    cameras, name = [], None
    for line in out.splitlines():
        if line and not line.startswith(("\t", " ")):  # 行頭が空白でない行 = カメラの名前の行
            name = line.strip().rstrip(":")
            cameras.append((name, []))
        elif line.strip().startswith("/dev/video"):
            cameras[-1][1].append(line.strip())
    return cameras


def grab_frame(device="/dev/video2", size=(1920, 1080), settle_frames=30, timeout=20):
    """device から画像を 1 枚取り込み、(高さ × 幅 × 3) の RGB 配列で返す。settle_frames コマ読んで捨ててから、最後の 1 コマを使う(自動露出が落ち着くのを待つ)。
    size: (幅, 高さ)。MJPG(圧縮)で読む。このカメラは、MJPG なら 1920×1080 でも 30 fps、YUYV(無圧縮)だと 5 fps に落ちる(v4l2-ctl --list-formats-ext)。"""
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "v4l2", "-input_format", "mjpeg", "-video_size", f"{size[0]}x{size[1]}", "-framerate", "30",
           "-i", device, "-frames:v", str(settle_frames), "-f", "image2pipe", "-vcodec", "png", "-update", "1", "-"]  # 最後のコマを、PNG として標準出力へ
    result = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if result.returncode != 0 or not result.stdout:
        raise RuntimeError(f"{device} から画像を取り込めなかった: {result.stderr.decode(errors='replace')[:200]}")
    return np.asarray(iio.imread(result.stdout, extension=".png"))[:, :, :3]  # [:, :, :3]: アルファ(透明度)があれば捨てて、RGB の 3 チャンネルにする


def grab_frames(n, device="/dev/video2", size=(1920, 1080), settle_frames=30, timeout=30):
    """device から、続けて n 枚の画像を取り込み、(n × 高さ × 幅 × 3) の RGB 配列で返す。settle_frames コマを捨ててから、続く n コマを使う(ffmpeg を 1 回だけ起動する)。
    なぜ grab_frame を n 回呼ばないか: 1 回ごとに、カメラを開いて settle_frames コマを捨てるので、n 倍の時間がかかる。ここは 1 回で済ませる。"""
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "v4l2", "-input_format", "mjpeg", "-video_size", f"{size[0]}x{size[1]}", "-framerate", "30",
           "-i", device, "-frames:v", str(settle_frames + n), "-vf", f"select=gte(n\\,{settle_frames})", "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]  # select=gte(n,30): 30 コマ目より後だけ出力。rawvideo: 圧縮なしの画素を、そのまま並べて出す
    result = subprocess.run(cmd, capture_output=True, timeout=timeout)
    frame_bytes = size[0] * size[1] * 3
    if result.returncode != 0 or len(result.stdout) < n * frame_bytes:
        raise RuntimeError(f"{device} から {n} 枚を取り込めなかった: {result.stderr.decode(errors='replace')[:200]}")
    return np.frombuffer(result.stdout[:n * frame_bytes], np.uint8).reshape(n, size[1], size[0], 3)  # frombuffer: バイト列を、コピーせずに配列として見る。[頻度] 生の画像データを扱うときの定番


def set_manual_exposure(device="/dev/video2", exposure=300, white_balance=4500, gain=None):
    """露出とホワイトバランスを、自動 → 固定にする(v4l2-ctl)。exposure の単位は、0.1 ミリ秒(このカメラの exposure_time_absolute。1〜5000)。
    なぜ固定するか: 自動のままだと、立方体の位置や、光の当たり方で、色の見え方が変わり、色検出の条件(vision.py の color_mask)が成り立たなくなる。"""
    controls = ["auto_exposure=1", f"exposure_time_absolute={exposure}", "white_balance_automatic=0", f"white_balance_temperature={white_balance}"]  # auto_exposure=1: 手動。3: 自動
    if gain is not None:
        controls.append(f"gain={gain}")
    subprocess.run(["v4l2-ctl", "-d", device, "-c", ",".join(controls)], check=True)


def grab_until(device, is_good, exposures=(100, 200, 400, 800, 1600), **grab_kwargs):
    """露出を、短い方から順に試して、is_good(画像) が True になる最初の画像を返す。(画像, そのときの露出) を返す。どれもだめなら (最後の画像, None)。
    なぜ要るか: 部屋の明るさは、時間帯で変わる(昼は露出 100 でよく、夕方は暗くて読めない)。マーカーが読める明るさを、毎回、自動で探す。
    is_good: 画像(RGB)を受け取り、「使える画像か」を True / False で返す関数(例: 「マーカーが 5 個以上読める」)。[頻度] 関数を引数に渡す書き方は、Python で定番"""
    img = None
    for exposure in exposures:
        set_manual_exposure(device, exposure)
        img = grab_frame(device, **grab_kwargs)
        if is_good(img):
            return img, exposure
    return img, None
