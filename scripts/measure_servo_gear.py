"""【概要】サーボ(STS3215)を 1 個だけ基板につないで、最大の速さで 90° 往復させ、かかった時間を測る。
目的: リーダーアーム用のサーボが、ギア比 1/147・1/191・1/345 のどれかを見分ける(型番のラベルが無いため)。
      ギア比が小さいほど、同じ指令でも出力軸が速く回る(1/147 は 1/345 の約 2.3 倍、1/191 は約 1.8 倍)。
      ギア比はレジスタに入っていないので、ソフトで直接は読めない。速さで見分ける。

使い方(LeRobot の環境で実行する。この .venv ではない):
    # 1 個ずつ基板につないで、ラベル(自分が付ける名前)を変えて実行する
    ~/lerobot-work/lerobot/.venv/bin/python scripts/measure_servo_gear.py --port /dev/ttyACM1 --label A
    # 6 個ぶん測り終えたら、速い順の表と 3 グループへの分け方を見る
    ~/lerobot-work/lerobot/.venv/bin/python scripts/measure_servo_gear.py --summary

注意:
  - **--port は必ず指定する(既定値なし)。フォロワーのポートに間違って実行しないため。** 調べ方: lerobot-find-port
  - 負荷をかけない(ホーンに何も付けない、または軽いものだけ)。サーボ本体は机にテープで止めるか、手で押さえる。
  - 動き出しの勢いが強い。動いている間、出力軸を握らず、指を近づけない。
  - 6 個すべて、同じ電源電圧で測る(電圧が違うと速さが変わる)。
  - 単位: 位置は「ステップ」(0〜4095 で 1 周 360°。1 ステップ = 約 0.088°)。90° = 1024 ステップ。
"""
import argparse  # コマンドラインの引数(--port など)を読む標準ライブラリ。[頻度] スクリプトを作るとき定番
import csv
import statistics
import time
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "outputs" / "servo_gear_measure.csv"  # Path(__file__).resolve().parents[1] = このファイルの 1 つ上のフォルダ(プロジェクトの根)。[頻度] 定番
STEPS_PER_REV = 4096  # 1 周のステップ数(STS3215)
MOVE_STEPS = STEPS_PER_REV // 4  # 90° = 1024 ステップ。// は小数点以下を切り捨てる割り算
SPEED_MAX = 3400  # Goal_Speed(単位: ステップ/秒)の最大値。モーターの限界まで回すため、最大を指令する
TOLERANCE = 12  # 目標との差がこのステップ(約 1°)以内になったら「着いた」とみなす
TIMEOUT_S = 4.0  # 1 回の動きの最長時間。これを超えたら着かなかったとして中止する
CSV_HEADER = ["time", "label", "id", "runs", "mean_total_s", "mean_cruise_deg_per_s"]


def move_and_time(bus, name, target):
    """target(ステップ)へ動かし、(着くまでの秒数, 途中の速さ[度/秒]) を返す。
    途中の速さ = 動きの 20%〜80% の区間だけで測る。動き出しと止まるときの加減速を除くため。"""
    p_start = bus.read("Present_Position", name, normalize=False)  # normalize=False: キャリブレーション無しの生の値(ステップ)で読む
    travel = target - p_start
    samples = []  # (経過秒, 位置) の記録
    t0 = time.perf_counter()  # perf_counter = 経過時間を測る用の高精度の時計。[頻度] 時間を測るときの定番
    bus.write("Goal_Position", name, target, normalize=False)
    while True:
        t = time.perf_counter() - t0
        pos = bus.read("Present_Position", name, normalize=False)
        samples.append((t, pos))
        if abs(pos - target) <= TOLERANCE:
            break
        if t > TIMEOUT_S:
            raise RuntimeError(f"{TIMEOUT_S} 秒たっても目標に着かない(位置 {pos}、目標 {target})。サーボに触れていないか確認。")
    total = t

    # 20%〜80% の区間(動いた量が travel の 0.2〜0.8 倍の間)の、最初と最後のサンプルを探す
    def progress(sample):
        return (sample[1] - p_start) / travel  # 0 = 出発、1 = 到着。 [1] はタプルの 2 つ目(位置)

    inside = [s for s in samples if 0.2 <= progress(s) <= 0.8]  # リストの内包表記: 条件に合うものだけの新しいリストを作る。[頻度] よく使う
    if len(inside) >= 2:
        (ta, pa), (tb, pb) = inside[0], inside[-1]  # inside[-1] = 最後の要素。[頻度] 定番
        cruise = abs(pb - pa) / (tb - ta) * 360 / STEPS_PER_REV if tb > ta else float("nan")
    else:
        cruise = float("nan")  # サンプルが足りない(速すぎる、読み取りが遅い)
    return total, cruise


def run_measure(args):
    # lerobot は、測定のときだけ必要。--summary のときは lerobot が無い環境でも動くよう、ここで import する
    from lerobot.motors import Motor, MotorNormMode, SerialMotorsBus

    name = "m"
    bus = SerialMotorsBus(port=args.port, motors={name: Motor(args.id, "sts3215", MotorNormMode.RANGE_M100_100)})
    bus.connect()  # ポートを開いて、サーボが応答するか確かめる(応答が無ければここで止まる)
    try:
        lo = bus.read("Min_Position_Limit", name, normalize=False)  # 動かせる範囲(新品は 0〜4095)
        hi = bus.read("Max_Position_Limit", name, normalize=False)
        p0 = bus.read("Present_Position", name, normalize=False)
        # +90° 側に動かせればそちらへ、無理なら -90° 側へ
        if p0 + MOVE_STEPS <= hi:
            far = p0 + MOVE_STEPS
        elif p0 - MOVE_STEPS >= lo:
            far = p0 - MOVE_STEPS
        else:
            raise RuntimeError(f"位置 {p0} から 90° 動かす余裕が無い(範囲 {lo}〜{hi})。出力軸を手で回して真ん中あたりへ。")
        print(f"サーボ ID {args.id}: 今の位置 {p0}、往復する先 {far}(90° = {abs(far - p0)} ステップ)")
        input("負荷が無く、本体が固定され、指が離れていることを確認して Enter(止めるときは Ctrl+C): ")

        bus.write("Torque_Enable", name, 0, normalize=False)  # 設定を書く前にトルクを切る
        if args.set_gains:
            # 位置制御のゲイン。このキットのサーボは P=0 のため、指令を出しても力が出ず動かなかった(2026-10-30)。
            # 値は LeRobot がフォロワーに接続のたびに書くのと同じ(P=16, I=0, D=32)。EEPROM(電源を切っても残る領域)に書かれる。
            # リーダーとして手で動かすだけなら不要。このスクリプトで動かすときだけ必要。
            for reg, val in (("P_Coefficient", 16), ("I_Coefficient", 0), ("D_Coefficient", 32)):
                bus.write(reg, name, val, normalize=False)
            print("ゲインを書いた: " + ", ".join(f"{r}={bus.read(r, name, normalize=False)}" for r in ("P_Coefficient", "I_Coefficient", "D_Coefficient")))
        bus.write("Goal_Time", name, 0, normalize=False)  # 0 = 時間指定なし(速さの指定を使う)
        bus.write("Acceleration", name, 254, normalize=False)  # 加速度も最大。LeRobot の推奨と同じ
        bus.write("Goal_Speed", name, SPEED_MAX, normalize=False)  # 最大の速さ。これより遅いと、ギア比によらず同じ速さになってしまう
        # トルク上限(0〜1000)。このキットのサーボは、電源を入れた直後は 0 で、トルクを入れても力が出ず動かなかった(2026-10-30)。
        # このレジスタは電源を切ると元に戻る型(SRAM)なので、毎回ここで書く。サーボ本体の恒久的な設定(EEPROM)は変えない。
        bus.write("Torque_Limit", name, 1000, normalize=False)
        bus.write("Torque_Enable", name, 1, normalize=False)

        totals, cruises = [], []
        # 往復を args.repeat 回。行き(p0 → far)と帰り(far → p0)の両方を測る
        for i in range(args.repeat):
            for target, label in ((far, "行き"), (p0, "帰り")):
                total, cruise = move_and_time(bus, name, target)
                totals.append(total)
                cruises.append(cruise)
                print(f"  {i + 1} 回目 {label}: {total:.3f} 秒、途中の速さ {cruise:6.1f} 度/秒")
                time.sleep(0.5)  # 止まってから次へ(揺れが収まるのを待つ)
    finally:
        # try/finally: エラーや Ctrl+C でも必ず実行される。トルクを切ってから通信を閉じる。[頻度] 機器を扱うときの定番
        try:
            bus.write("Torque_Enable", name, 0, normalize=False)
        finally:
            bus.disconnect(disable_torque=False)

    mean_total = statistics.mean(totals)  # 平均
    mean_cruise = statistics.mean(c for c in cruises if c == c)  # c == c は「nan でない」の判定(nan は自分自身と等しくない)。()の中は内包表記の仲間(ジェネレータ式)
    print(f"[{args.label}] 平均 {mean_total:.3f} 秒、途中の平均の速さ {mean_cruise:.1f} 度/秒")

    OUT.parent.mkdir(exist_ok=True)
    new_file = not OUT.exists()
    with OUT.open("a", newline="") as f:  # with ... as f: 使い終わったら自動で閉じる。"a" = 末尾に追記。[頻度] ファイルを扱うときの定番
        w = csv.writer(f)
        if new_file:
            w.writerow(CSV_HEADER)
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), args.label, args.id, len(totals),
                    f"{mean_total:.4f}", f"{mean_cruise:.1f}"])
    print(f"記録: {OUT}")


def run_summary():
    if not OUT.exists():
        raise SystemExit(f"{OUT} がまだ無い。先に測定する。")
    latest = {}  # ラベルごとに、いちばん新しい行だけを使う(測り直したときは後ろが勝つ)
    with OUT.open(newline="") as f:
        for row in csv.DictReader(f):  # DictReader: 1 行を {列名: 値} の辞書にして読む
            latest[row["label"]] = row
    rows = sorted(latest.values(), key=lambda r: -float(r["mean_cruise_deg_per_s"]))  # 速い順。key=... は並べ替えの基準。-は大きい順にするため
    slowest = float(rows[-1]["mean_cruise_deg_per_s"])
    print("速い順(途中の速さ。いちばん遅いものを 1 とした比)")
    for r in rows:
        v = float(r["mean_cruise_deg_per_s"])
        print(f"  {r['label']:>6s}  {v:7.1f} 度/秒  比 {v / slowest:4.2f}  (平均 {float(r['mean_total_s']):.3f} 秒)")
    print("\n目安: 1/147 ≈ 2.3、1/191 ≈ 1.8、1/345 = 1.0。リーダーは 1/147 が 3 個、1/191 が 2 個、1/345 が 1 個のはず。")
    if len(rows) == 6:
        print("6 個そろった。上の比が、速い方から 3 個(≈2.3)・2 個(≈1.8)・1 個(1.0)に分かれていれば、内訳が合っている。")
    else:
        print(f"今は {len(rows)} 個。6 個そろうまで測る。")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # ArgumentParser = 引数の定義を集める入れ物
    ap.add_argument("--port", help="サーボをつないだ基板のポート(例 /dev/ttyACM1)。必須。lerobot-find-port で調べる")
    ap.add_argument("--id", type=int, default=1, help="サーボの ID(出荷時は 1)")
    ap.add_argument("--label", help="このサーボの名前(A〜F など。袋やサーボにも同じ印を付けておく)")
    ap.add_argument("--repeat", type=int, default=3, help="往復の回数(既定 3)")
    ap.add_argument("--set-gains", action="store_true", help="測定の前に P=16, I=0, D=32 をサーボに書く(P=0 のサーボは動かないため)")
    ap.add_argument("--summary", action="store_true", help="測定せず、記録済みの結果を速い順に表示する")
    args = ap.parse_args()
    if args.summary:
        run_summary()
        return
    if not args.port or not args.label:
        ap.error("--port と --label は必須(フォロワーのポートに間違って実行しないため、--port に既定値は無い)")
    run_measure(args)


if __name__ == "__main__":  # このファイルを直接実行したときだけ main() を呼ぶ決まり文句。[頻度] 定番
    main()
