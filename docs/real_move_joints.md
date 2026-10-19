# 実機を1関節ずつ動かす ― `scripts/real_move_joints.py` の解説

実機の SO-101 を、1関節ずつ「+delta → 戻る → -delta → 戻る」と少しだけ動かす確認用スクリプト。
目的は、各関節の**動く向き(符号)**と、**指令どおりに動くか**(指令と実測の差)を、小さな動きで確かめること。
この `.venv` ではなく、LeRobot の環境(`~/lerobot-work/lerobot` の `.venv`)で実行する。

## 実行
    cd ~/lerobot-work/lerobot && source .venv/bin/activate
    python ~/ドキュメント/robot-hand-portfolio/scripts/real_move_joints.py --joint wrist_roll --delta 30 --seconds 3

| 引数 | 意味 | 既定 |
|---|---|---|
| `--joint` | 動かす関節。省略すると全関節を順に | 全関節 |
| `--delta` | 動かす量(腕の5関節は度、gripper は 0〜100 の量) | 5 |
| `--seconds` | 片道にかける秒数 | 2 |

安全:
- 動かす前に、腕を手で、**関節が端に当たっていない中ほどの姿勢**にする(休めの姿勢では片側にしか動けない)。
- 周りに物がないこと、ケーブルが引っ張られないことを確認する。
- 終了(切断)するとトルクが切れて**腕が落ちる**。片手で支える。
- 動かす順は、`wrist_roll` → `shoulder_pan` → `gripper` → `wrist_flex` → 最後に `elbow_flex`、`shoulder_lift`(腕全体の重さがかかる)。

## 全体の流れ
    接続 → 今の姿勢を読む → (関節ごとに) +delta へ動かす → 実測を読む → 戻る → -delta も同様 → 切断

## 1. 可動域を求める(`joint_limits`)
キャリブレーションで各関節を端から端まで動かしたとき、サーボの**生のカウント値**(0〜4095 が1周 = 360°)が保存される。
LeRobot は範囲の**中央が 0 度**になるように決めている。

    度 = (カウント − 範囲の中央) × 360 / 4095

そのため可動域は左右対称になる(今回のキャリブレーション結果):

| 関節 | 可動域(度) |
|---|---|
| shoulder_pan | ±119.1 |
| shoulder_lift | ±106.1 |
| elbow_flex | ±97.3 |
| wrist_flex | ±111.9 |
| wrist_roll | ±180.0 |
| gripper | 0〜100(別の尺度) |

目標が端から `MARGIN`(3°)以内になる向きは、動かさず飛ばす。端に押し当てるとサーボに無理な力がかかるため。

## 2. 滑らかに動かす(`move_to`)
目標を一度に送らず、`RATE_HZ`(50Hz)で少しずつずらして送る。

    steps = 秒数 × 50          例: 2秒 → 100 ステップ
    各ステップの目標 = start + (goal − start) × (i / steps)

例えば 3.7° → 8.7° なら、3.75、3.80、…、8.7 と 0.02 秒ごとに送る。最初から 8.7 を送ると、サーボが全力で急いで動くため。
動かさない関節は start と goal を同じ値にして、その位置を保つ。

## 3. `send_action` の中で起きること
`send_action` は `Goal_Position`(目標位置)をサーボに書き込む。実際の制御はサーボ内部の基板が行う
(P=16、I=0、D=32 の位置制御)。Python 側は目標を送るだけで、トルクは計算しない。
sim の `<position>` アクチュエータ(`data.ctrl` に目標角を入れると PD が MuJoCo 内部で動く)と同じ構造。

## 4. 「差」の意味
`差 = 実測 − 指令`。`wrist_roll` の ±5° では ±0.4° だった。
- 目標に近づくほどサーボの力は弱まるので、数分の1°の誤差は残る。
- 重力がかかる `shoulder_lift`、`elbow_flex` は差が大きくなりやすい。
- 後で sim と実機の差(ギャップ)を測るときの、**基本の誤差の大きさ**になる。

## 5. `try/finally`
途中でエラーや Ctrl+C があっても、`finally` の `robot.disconnect()` は必ず実行される(機器を扱うときの定番)。

## sim との対応
| | sim | 実機 |
|---|---|---|
| 目標の渡し方 | `data.ctrl[i] = 角度 [rad]` | `send_action({"joint.pos": 角度 [度]})` |
| 制御 | MuJoCo 内部の PD | サーボ内部の P 制御 |
| 現在の角度 | `data.qpos` | `get_observation()` |
| 単位 | rad | 度(gripper は 0〜100) |

次の「sim↔実機の変換」は、この表の**単位・0 の位置・+ の向き**を合わせる作業。

## 結果の記録(実機で動かしたら埋める)
| 関節 | + の向き(腕がどちらに動くか) | 差(度) | メモ |
|---|---|---|---|
| shoulder_pan | | | |
| shoulder_lift | | | |
| elbow_flex | | | |
| wrist_flex | | | |
| wrist_roll | (±5° で差 ±0.4°。向きは未記録) | ±0.4 | |
| gripper | | | |
