"""【概要】カメラの内部パラメータの較正に使う、チェスボードの画像を、一定の間隔で自動的に撮る。
盤を、カメラの前で、色々な位置・距離・傾きに動かしながら、撮影の合図(音と、画面の表示)に合わせて、止める。
撮った画像は、角(9 × 6 個)が見つかったものだけを保存する(outputs/chessboard_shots/。outputs/ は git に入らない)。
ウィンドウは開かない(ターミナルに、撮れたか、次にどこへ盤を動かすかの指示が出る)。

使い方(この .venv で実行):
    .venv/bin/python scripts/capture_chessboard.py                # 既定: 20 枚、8 秒おき
    .venv/bin/python scripts/capture_chessboard.py --shots 25 --interval 6 --manual   # 25 枚、6 秒おき、露出と色を固定
撮り終わったら: .venv/bin/python scripts/calibrate_webcam_intrinsics.py
"""
import argparse
import sys
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
parser.add_argument("--shots", type=int, default=20, help="保存する画像の枚数")
parser.add_argument("--interval", type=float, default=8.0, help="撮影の間隔 [秒]。この間に、盤を次の位置へ動かして、止める")
parser.add_argument("--max-attempts", type=int, default=None, help="撮影を試す回数の上限(角が見つからなくても数える)。省略すると、枚数の 3 倍。盤が写っていないときに、いつまでも続かないようにする")
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

saved, attempt = 0, 0
max_attempts = args.max_attempts or args.shots * 3
print(f"{args.shots} 枚撮る(角が見つかったものだけ数える)。盤を、下の指示の位置へ動かして、{args.interval:.0f} 秒ごとの撮影まで、止めておく。")
try:
    while saved < args.shots and attempt < max_attempts:
        hint = HINTS[attempt % len(HINTS)]
        print(f"\n[{saved + 1}/{args.shots}] 次: {hint}")
        end = time.time() + args.interval
        while time.time() < end:  # 待つ間も、カメラのコマを読み捨てる(読まないと、古いコマがたまる)
            cap.read()
        for _ in range(3):
            cap.read()  # 撮影の直前に、最新のコマに入れ替える
        ok, frame = cap.read()
        attempt += 1
        if not ok:
            print("  コマを読めなかった。もう一度")
            continue
        corners = find_corners(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # OpenCV は、画像を、青・緑・赤の順で持つ。find_corners は RGB を受け取るので、並べ替える
        if corners is None:
            print("  × 角が見つからなかった(盤が全部写っているか、ぶれていないか、反射していないか)。やり直し")
            continue
        saved += 1
        cv2.imwrite(str(out_dir / f"shot_{saved:02d}.png"), frame)
        x, y = corners[:, 0, 0], corners[:, 0, 1]
        print(f"  ○ 保存 shot_{saved:02d}.png(盤の範囲: 横 {x.min():.0f}〜{x.max():.0f}、縦 {y.min():.0f}〜{y.max():.0f} 画素)")
finally:
    cap.release()
if saved < args.shots:
    print(f"\n撮影の上限({max_attempts} 回)に達した。角が見つかったのは {saved} 枚だけ")
print(f"\n{saved} 枚を {out_dir.relative_to(ROOT)} に保存した。次: .venv/bin/python scripts/calibrate_webcam_intrinsics.py")
