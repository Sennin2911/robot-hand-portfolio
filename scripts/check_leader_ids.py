"""【概要】リーダーアームのサーボ 6 個を全部つないだ状態で、ID 1〜6 がそれぞれ応答するか(ping)と、今の位置を表示する。
読むだけで、指令もトルクの操作もしない。組み立ての前後に、配線と ID の確認に使う。

使い方(LeRobot の環境で実行する。この .venv ではない):
    ~/lerobot-work/lerobot/.venv/bin/python scripts/check_leader_ids.py --port /dev/ttyACM0
    # 位置の表示を続けて、手で関節を回したときに、どの ID の数字が変わるかを見る(Ctrl+C で終了)
    ~/lerobot-work/lerobot/.venv/bin/python scripts/check_leader_ids.py --port /dev/ttyACM0 --watch
見方: 6 個とも「応答あり」で、モデル番号が 777(STS3215)なら OK。--watch で関節を 1 つずつ回して、
      その関節の名前の数字(下の JOINTS の対応)だけが変わるか確認する。位置の単位は「ステップ」(0〜4095 で 1 周)。
"""
import argparse  # コマンドラインの引数(--port など)を読む標準ライブラリ。[頻度] スクリプトを作るとき定番
import time

# ID と関節名の対応。lerobot-setup-motors で付けた ID(gripper 6 〜 shoulder_pan 1)と同じ
JOINTS = {1: "shoulder_pan", 2: "shoulder_lift", 3: "elbow_flex", 4: "wrist_flex", 5: "wrist_roll", 6: "gripper"}
STS3215_MODEL = 777  # STS3215 のモデル番号(ping で返ってくる値)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # ArgumentParser = 引数の定義を集める入れ物
    ap.add_argument("--port", required=True, help="リーダーの基板のポート(例 /dev/ttyACM0)。lerobot-find-port で調べる")
    ap.add_argument("--watch", action="store_true", help="位置を続けて表示する(Ctrl+C で終了)。手で回して、どの ID が動くか見る")
    args = ap.parse_args()

    # lerobot の部品は、実行するときだけ必要なので、ここで import する
    from lerobot.motors import Motor, MotorNormMode, SerialMotorsBus

    motors = {name: Motor(i, "sts3215", MotorNormMode.RANGE_M100_100) for i, name in JOINTS.items()}  # 辞書の内包表記: 6 個ぶんの Motor を一度に作る。[頻度] よく使う
    bus = SerialMotorsBus(port=args.port, motors=motors)
    bus.connect(handshake=False)  # handshake=False: 応答しない ID があってもエラーで止めず、下で 1 個ずつ調べる
    try:
        ok = 0
        print("ID  関節            応答")
        for i, name in JOINTS.items():
            try:
                model = bus.ping(i)  # ping = 「いますか?」と呼びかける。応答があればモデル番号、無ければ None
            except Exception:
                model = None
            if model == STS3215_MODEL:
                status, ok = "あり(STS3215)", ok + 1
            elif model is None:
                status = "応答なし"
            else:
                status = f"応答あり、モデル番号 {model}(STS3215 は {STS3215_MODEL})"
            print(f"{i:2d}  {name:14s}  {status}")
        print(f"\n{ok}/6 個が応答した。" + ("すべて OK。" if ok == 6 else "応答しない ID は、ケーブルの差し込み、電源(12 V)、ID の設定を確認する。"))

        if args.watch:
            print("\n位置を表示する。1 つの関節を手で回して、どの ID の数字が変わるか見る(Ctrl+C で終了)。")
            try:
                while True:
                    pos = []
                    for i, name in JOINTS.items():
                        try:
                            pos.append(f"{i}:{bus.read('Present_Position', name, normalize=False):4d}")  # normalize=False: キャリブレーション無しの生の値(ステップ)で読む
                        except Exception:
                            pos.append(f"{i}:----")
                    print("  ".join(pos), end="\r")  # end="\r" = 改行せず、同じ行を上書きする
                    time.sleep(0.1)
            except KeyboardInterrupt:  # Ctrl+C で終了したときの処理。[頻度] 無限ループを止める定番
                print()
    finally:
        bus.disconnect(disable_torque=False)  # トルクの設定には触れずに閉じる


if __name__ == "__main__":  # このファイルを直接実行したときだけ main() を呼ぶ決まり文句。[頻度] 定番
    main()
