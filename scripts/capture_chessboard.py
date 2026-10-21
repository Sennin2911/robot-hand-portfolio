"""【概要】カメラの内部パラメータの較正に使う、チェスボードの画像を撮る。3 つの撮り方がある:
  --test : 試しに撮る。Enter キーを押すたびに 1 枚撮り、角が見つかったか・明るさ・ぼけ・盤の大きさを表示して、outputs/chessboard_test.png に保存する。
           **毎回、同じファイルに上書きされ、較正用の保存先(outputs/chessboard_shots/)には保存されない**(較正には使われない)。較正用の撮影は --enter。
           盤の持ち方・明るさ・照明の反射を、本番の前に確かめる用。q + Enter で終わる。
  --enter: 本番。Enter キーを押すたびに 1 枚撮る。角が見つかったものだけを保存して、数える。盤を動かす位置の指示が、ターミナルに出る。q + Enter で終わる。
  (どちらも付けない)本番。一定の間隔(既定 8 秒)で、自動で撮る。両手で盤を持つときは、こちらが楽。
撮った画像は outputs/chessboard_shots/ に保存する(outputs/ は git に入らない)。ウィンドウは開かない(ターミナルだけで操作する)。

使い方(この .venv で実行):
    .venv/bin/python scripts/capture_chessboard.py --test --manual --exposure 300       # 試しの撮影(Enter で撮る)
    .venv/bin/python scripts/capture_chessboard.py --enter --manual --exposure 300      # 本番、Enter で撮る
    .venv/bin/python scripts/capture_chessboard.py --manual --exposure 300 --interval 8 # 本番、8 秒ごとに自動で撮る
撮り終わったら: .venv/bin/python scripts/calibrate_webcam_intrinsics.py
"""
import argparse
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# src フォルダを import の探索先に追加(詳しい意味は check_install.py を参照)。
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
from hand_sim.intrinsics import find_corners  # noqa: E402
from hand_sim.webcam import set_manual_exposure  # noqa: E402

# 盤を動かす位置の指示(画像の中での位置と、傾き)。順に、画像の隅々まで盤が通るようにする(歪みは、画像の端ほど大きいので、端の角を多く使うのが大事)。
HINTS = [
    "画像の真ん中に、正面向きで", "画像の左上の隅に近づけて", "画像の右上の隅に近づけて", "画像の左下の隅に近づけて", "画像の右下の隅に近づけて",
    "画像の左端に、少し右へ傾けて", "画像の右端に、少し左へ傾けて", "画像の上の端に、手前へ傾けて", "画像の下の端に、奥へ傾けて",
    "カメラから遠く(画像の 1/4 の大きさ)に", "カメラから近く(画像の半分の大きさ)に", "真ん中で、左へ 30° 回して", "真ん中で、右へ 30° 回して",
    "左上で、斜めに(右上がりに 20°)", "右下で、斜めに(左上がりに 20°)", "左へ寄せて、手前に傾けて", "右へ寄せて、手前に傾けて",
    "上へ寄せて、右へ傾けて", "下へ寄せて、左へ傾けて", "真ん中で、手前へ 30° 傾けて", "真ん中で、奥へ 30° 傾けて", "左下で、奥へ傾けて", "右上で、奥へ傾けて",
    "画像の左端いっぱいに", "画像の右端いっぱいに",
]

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="/dev/video2", help="カメラのファイル名")
parser.add_argument("--test", action="store_true", help="試しの撮影(Enter で撮る。較正用には数えず、毎回 outputs/chessboard_test.png に上書き保存)")
parser.add_argument("--enter", action="store_true", help="Enter キーを押すたびに撮る(省略すると、一定の間隔で、自動で撮る)")
parser.add_argument("--shots", type=int, default=20, help="保存する画像の枚数(本番のとき)")
parser.add_argument("--interval", type=float, default=8.0, help="自動で撮るときの間隔 [秒]。この間に、盤を次の位置へ動かして、止める")
parser.add_argument("--max-attempts", type=int, default=None, help="自動で撮るときの、撮影を試す回数の上限(角が見つからなくても数える)。省略すると、枚数の 3 倍")
parser.add_argument("--size", type=int, nargs=2, default=[1920, 1080], metavar=("W", "H"))
parser.add_argument("--exposure", type=int, default=300, help="--manual のときの露出時間(0.1 ミリ秒単位、1〜5000)。大きいほど明るい")
parser.add_argument("--gain", type=int, default=32, help="--manual のときのゲイン(0〜63、このカメラの既定は 32)。大きいほど明るいが、ざらつく")
parser.add_argument("--manual", action="store_true", help="露出とホワイトバランスを固定にする(盤の白と黒が、つぶれない明るさに、先に決めておく)")
args = parser.parse_args()

if args.manual:
    set_manual_exposure(args.device, args.exposure, gain=args.gain)
out_dir = ROOT / "outputs" / "chessboard_shots"
out_dir.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)  # cv2.VideoCapture: カメラから、コマを読む入口。CAP_V4L2: Linux のカメラの標準方式
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # 圧縮方式 MJPG(1080p でも 30 fps)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.size[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.size[1])
cap.set(cv2.CAP_PROP_FPS, 30)
if not cap.isOpened():
    sys.exit(f"{args.device} を開けなかった(カメラがつながっているか、他のアプリが使っていないか)")


class LatestFrame:
    """カメラのコマを、裏(別のスレッド)で読み続けて、一番新しいコマだけを持つ。
    Enter を待つ間(input() で止まっている間)も、カメラは動き続けて、古いコマがたまる。裏で読み捨てておけば、Enter の瞬間の、新しいコマが取れる。
    threading.Thread: プログラムの中で、同時に動く、もう 1 つの流れ(スレッド)を作る。[頻度] 待ち時間のある処理で、たまに使う"""

    def __init__(self, capture):
        self.capture, self.frame, self.count, self.lock, self.running = capture, None, 0, threading.Lock(), True
        self.thread = threading.Thread(target=self._loop, daemon=True)  # daemon=True: メインの処理が終わったら、一緒に終わる
        self.thread.start()

    def _loop(self):
        while self.running:
            ok, frame = self.capture.read()
            if ok:
                with self.lock:  # 読み書きが重ならないよう、鍵をかける。[頻度] スレッドで、同じ変数を使うときの定番
                    self.frame, self.count = frame, self.count + 1

    def latest(self):
        """今の最新のコマ(コピー)を返す。起動直後で、まだ無ければ、来るまで待つ。"""
        while True:
            with self.lock:
                if self.frame is not None:
                    return self.frame.copy()
            time.sleep(0.02)

    def stop(self):
        self.running = False
        self.thread.join(timeout=2)


def next_shot_number(directory):
    """directory にある shot_NN.png の、最大の番号 + 1 を返す(無ければ 1)。もう一度実行したときに、前の画像を上書きしないため。"""
    numbers = [int(p.stem.split("_")[1]) for p in directory.glob("shot_*.png") if p.stem.split("_")[1].isdigit()]  # p.stem: 拡張子を除いたファイル名
    return max(numbers, default=0) + 1


def evaluate(frame):
    """1 枚のコマを調べて、(角, 表示する文) を返す。角は、見つかれば (54 × 1 × 2)、無ければ None。"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = gray.mean()
    sharp = cv2.Laplacian(gray, cv2.CV_64F).var()  # 画像の変化の大きさの分散。大きいほど、くっきり(ぼけていると小さい)
    corners = find_corners(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # OpenCV は、画像を、青・緑・赤の順で持つ。find_corners は RGB を受け取るので、並べ替える
    text = f"明るさ 平均 {brightness:.0f}(100〜150 が目安)、くっきり度 {sharp:.0f}(小さいと、ぼけ・ぶれ)"
    if corners is None:
        return None, text + "\n  × 角が見つからなかった(盤が全部写っているか、ぶれていないか、反射していないか、傾けすぎていないか)"
    x, y = corners[:, 0, 0], corners[:, 0, 1]
    size = (x.max() - x.min()) / frame.shape[1]  # 盤の横の幅が、画像の幅の何割か
    return corners, text + f"\n  ○ 角 54 個が見つかった。盤の範囲: 横 {x.min():.0f}〜{x.max():.0f}、縦 {y.min():.0f}〜{y.max():.0f} 画素(盤の幅は、画像の幅の {size:.0%})"


reader = LatestFrame(cap)
time.sleep(1.5)  # 起動直後は、露出が落ち着いていない。少し待つ
try:
    if args.test:
        print("試しの撮影。盤を持って、Enter で撮る(q + Enter で終わる)。結果は outputs/chessboard_test.png に保存される。")
        while True:
            if input("\nEnter で撮影 > ").strip().lower() == "q":
                break
            frame = reader.latest()
            _, text = evaluate(frame)
            cv2.imwrite(str(ROOT / "outputs" / "chessboard_test.png"), frame)
            print("  " + text)
    elif args.enter:
        saved = 0
        first = next_shot_number(out_dir)  # 前の実行で撮った画像の、続きの番号から保存する(上書きしない)
        print(f"本番(Enter で撮る)。角が見つかったものだけ保存して、数える。この実行で {args.shots} 枚撮ると終わる(q + Enter で途中終了)。保存は shot_{first:02d}.png から。")
        while saved < args.shots:
            print(f"\n[{saved + 1}/{args.shots}] 次: {HINTS[saved % len(HINTS)]}")
            if input("盤を動かして止めたら、Enter > ").strip().lower() == "q":
                break
            frame = reader.latest()
            corners, text = evaluate(frame)
            print("  " + text)
            if corners is not None:
                name = f"shot_{first + saved:02d}.png"
                saved += 1
                cv2.imwrite(str(out_dir / name), frame)
                print(f"  → 保存 {name}")
    else:
        saved, attempt = 0, 0
        first = next_shot_number(out_dir)
        max_attempts = args.max_attempts or args.shots * 3
        print(f"{args.shots} 枚を、{args.interval:.0f} 秒ごとに自動で撮る(角が見つかったものだけ数える)。盤を、指示の位置へ動かして、撮影まで止めておく。")
        while saved < args.shots and attempt < max_attempts:
            print(f"\n[{saved + 1}/{args.shots}] 次: {HINTS[attempt % len(HINTS)]}")
            time.sleep(args.interval)
            attempt += 1
            frame = reader.latest()
            corners, text = evaluate(frame)
            print("  " + text)
            if corners is not None:
                name = f"shot_{first + saved:02d}.png"
                saved += 1
                cv2.imwrite(str(out_dir / name), frame)
                print(f"  → 保存 {name}")
        if saved < args.shots:
            print(f"\n撮影の上限({max_attempts} 回)に達した。角が見つかったのは {saved} 枚だけ")
except (EOFError, KeyboardInterrupt):  # 入力が閉じた、または Ctrl+C
    print("\n中断した")
finally:
    reader.stop()
    cap.release()
if not args.test:
    n = len(list(out_dir.glob("shot_*.png")))
    print(f"\n{out_dir.relative_to(ROOT)} に {n} 枚ある。次: .venv/bin/python scripts/calibrate_webcam_intrinsics.py")
