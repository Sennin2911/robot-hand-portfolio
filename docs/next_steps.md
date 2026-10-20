# 進捗と次にやること

別のPC(Mac など)や、新しい Claude Code のセッションから作業を再開するための引き継ぎメモ。
**作業を進めたら、この文書も更新する**(日付を付けて、実態とずれた記述を残さない)。

最終更新: 2026-10-20(続き)

## 目的
「カメラで対象を捉え、sim で検証した制御を、実機(SO-101 + STS3215 サーボ)に移す」ポートフォリオ。
流れは、第1週=sim と制御の土台 → 第2週=sim で視覚サーボ → 第3週=実機に移植 → 第4週=sim と実機の差(ギャップ)を計測。

## 進め方(重要)
作者は MuJoCo も Python も初心者。ただし 2026-10-18 に方針を変更した:
**コードは Claude が実装する(コメントは CLAUDE.md のルールで丁寧に残す)。作者は制御の考え方を理解することに集中する。**
(理由: 制御の概要は分かっており、実装を自分で書くこと自体の学習価値は低いと作者が判断したため。)

- 新しい課題は、(1)説明の md(`docs/`)、(2)実装、(3)`scripts/check_*.py` の答え合わせ、の形で進める。
  実装後は check を実行して結果を報告し、考え方・数値の意味は説明で補う。
- 結果を説明するときは、実行結果のどの出力が、どの行から来たかを対応づける。
- コードのコメントは `CLAUDE.md` のルールに従う(日本語、初心者が知らない記法の意味と使用頻度を書く)。
- コミットは小さく頻繁に。push は依頼があったときだけ。

## 済んだこと
| 週 | 内容 | 場所 |
|---|---|---|
| 第1週 | 2リンク平面アームの順運動学・逆運動学 | `src/hand_sim/arm.py`、`scripts/check_arm.py`、`check_ik.py` |
| 第1週 | 自作 PD 制御(トルク駆動)とステップ応答の実験 | `src/hand_sim/controller.py`、`analysis.py`、`docs/pd_step_response.md` |
| 第1週 | SO-101 の MuJoCo モデルの取り込み(公式 `TheRobotStudio/SO-ARM100` の `Simulation/SO101/`) | `models/so101/`(出典・ライセンスは `models/so101/README.md`)、`scripts/inspect_so101.py` |
| 第2週 | 視覚サーボ用のシーン(俯瞰カメラ `overhead` + 赤い立方体) | `models/so101/scene_vs.xml` |
| 第2週 | **色検出(`find_red_centroid`)と画素→世界座標(`pixel_to_world`)** | `src/hand_sim/vision.py`、解説 `docs/vision_basics.md`、答え合わせ `scripts/check_vision.py`(全OK) |
| 第2週 | **視覚サーボ(open-loop と closed-loop)**。closed-loop は手先の青い目印も見て、カメラの位置のずれを打ち消す。`check_closed_loop.py` 全OK | `src/hand_sim/visual_servo.py`、解説 `docs/visual_servo.md`、流れの整理 `docs/closed_loop_flow.md`、`scripts/check_visual_servo.py`、`scripts/check_closed_loop.py` |
| 第2週 | **検出器の差し替え口**(色検出 / 最大の塊 / 枠を返すモデルのまね。制御側は不変) | `src/hand_sim/vision.py`、`visual_servo.py` の `detector` 引数、解説 `docs/detector_swap.md`、`scripts/check_detectors.py` |
| 第2週 | **カメラの較正(位置・視野角)** | `src/hand_sim/calibration.py`、解説 `docs/calibration.md`、`scripts/check_calibration.py` |
| 第2週 | **SO-101 の数値IK**(核心の `dq = ...` は Claude が実装。`check_so101_ik.py` 全OK) | `src/hand_sim/so101_ik.py`、解説 `docs/numerical_ik.md`、`scripts/check_so101_ik.py` |

SO-101 の要点: 6関節(`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`)、
`<position>` アクチュエータ(kp=998、kv=2.73、最大トルク 3.35 N·m)。
目標角度 `ctrl` [rad] を渡すと、モデルの内部で PD 制御される。第1週の `<motor>`(トルク指令)とは違う点に注意。

## 第2週の項目(すべて完了。2026-10-18)
各項目は、解説 md・実装・答え合わせ(`scripts/check_*.py`)の形で進めた。

1. ~~SO-101 の逆運動学(IK)~~ **完了。** `so101_ik.py` の `solve_ik`(ヤコビアン+減衰付き最小二乗法)。
2. ~~視覚サーボを1周つなぐ~~ **完了(open-loop と closed-loop)。** `src/hand_sim/visual_servo.py`、答え合わせ `scripts/check_visual_servo.py`(全OK、水平ずれ 0.4〜1.4mm)。
   撮影 → 色検出 → 世界座標 → IK → `ctrl` を1回通す方式。手先は立方体の真上 10cm で止める(掴むところまでは未実装)。
   残る課題: 降下して掴む動作(ゲート1に必須かは未決定)。
3. ~~ハンドアイ・キャリブレーションの考え方を sim で通す~~ **完了(固定カメラ = eye-to-hand。位置と視野角のみ)。** `src/hand_sim/calibration.py`、解説 `docs/calibration.md`、`scripts/check_calibration.py`(全OK)。
   カメラの向きは「真下」と分かっている前提、レンズ歪みは無い前提。実機では OpenCV の `solvePnP` / `calibrateHandEye` などで向きまで求める必要がある。
   固定カメラ(eye-to-hand)か手首カメラ(eye-in-hand)かは**未決定**のまま(今は固定カメラで進めている)。
4. ~~検出を、既製の物体検出モデルに差し替えられる形に保つ~~ **完了(形だけ。本物のモデルは未使用)。** 検出器の約束 `detector(img) -> (u, v) | None`、解説 `docs/detector_swap.md`、`scripts/check_detectors.py`(全OK)。

ゲート1(第2週末): sim で視覚サーボが通ること。通ったら実機の機材を発注する。通らなければ、第3週を sim の補強に充てる。
**判定待ち:** 第2週の4項目は完了し、sim で「立方体の真上に 1〜2 mm で届く」(カメラ位置が 2 cm ずれていても、closed-loop または較正で)ところまで通った。
ただし「掴む」ところは未実装。ゲート1の条件を「届く」までとするか、「掴む」までとするかは、**作者の判断待ち**。発注は作者が決める(購入するキットの型番も未確認。下の「未決定」を参照)。

## 未決定・未確認のこと
- **機材は購入済み、3Dプリントの出力も完了(2026-10-19 時点)。型番は、作者の認識では SO-101。**
  ただし、使った STL のフォルダ名が `STL/SO101/`(`STL/SO100/` ではない)かは、まだ目で確認していない。組み立て前に確認する。
  フォロワーのみで足りる(リーダーは遠隔操作の教示用で、視覚サーボには不要)。
- **組み立ての参考:** LeRobot の公式ドキュメント(https://huggingface.co/docs/lerobot/so101)と、
  TheRobotStudio/SO-ARM100 のリポジトリ(部品表・3Dプリント・組み立て)。**サーボの ID 設定は、組み立てより前**(組み立てると、コネクタに手が届きにくい)。
  具体的なコマンドは、公式ページの最新を見る。
- **LeRobot は、この `.venv`(Python 3.14)に入れない。** 要求される Python のバージョンを公式ガイドで確認し、別の環境(`uv` が使える)に入れる。
  Feetech サーボ用の追加機能は、LeRobot のフォルダで `pip install -e ".[feetech]"`(公式の最新の手順を確認)。
- **実機の進捗(2026-10-19):** LeRobot 0.6.2 を `~/lerobot-work/lerobot`(Python 3.12 の別環境)に導入済み。基板のポートは `/dev/ttyACM0`。
  フォロワーのサーボ ID を設定済み(shoulder_pan=1、shoulder_lift=2、elbow_flex=3、wrist_flex=4、wrist_roll=5、gripper=6)。
  ID 設定のあと、数珠つなぎにして PING で確認済み(ID 1〜6 が1つずつ応答、重なり・余りなし)。ただし、**どの物理サーボがどの ID か**は未確認
  (ID 設定のとき、手順書の「1つだけ接続」ではなく、1つずつ足していった)。組み立て前に、サーボを1つずつ基板につないで PING し、ID のラベルを貼る。
  次は、サーボに ID のラベルを貼る → 組み立て(Joint 1〜5、Gripper)→ 数珠つなぎ → `lerobot-calibrate`。
  Linux では USB を抜き差しすると、`sudo chmod 666 /dev/ttyACM0` の設定が戻る。
- **実機の進捗(2026-10-20):** 組み立てとキャリブレーション(`my_follower_arm`)が完了。関節角の読み取りは正常。
  `scripts/real_move_joints.py`(解説 `docs/real_move_joints.md`)で1関節ずつ ±delta 動かす確認を開始(`wrist_roll` ±5° は差 ±0.4°)。
  次は、各関節の + の向きを記録 → sim の角度(rad)と実機の値(度)の変換(符号・原点・gripper の 0〜100)。
  5関節+gripper の ±30 の差を記録済み(`docs/real_move_joints.md`。基本の誤差は ±0.2〜0.5°、重力がかかる関節は −30 側で最大 +2.0°)。
  実機の + の向きは、全関節とも sim と同じ(符号はすべて +1。wrist_roll は土台側から見て反時計回りと作者が確認)。
  変換コード `src/hand_sim/sim_real.py` と `scripts/check_sim_real.py`(全OK)を実装済み。
  sim の姿勢を見る `scripts/view_sim_pose.py`(実機に送る値とコマンドも表示)と、実機を指定姿勢へ動かす `scripts/real_send_pose.py` を作成(実機での実行は未)。
  `real_send_pose.py --pose 30 0 30 0 60 60` で実機が指定どおりに動くことを確認済み(2026-10-21)。`view_sim_pose.py --pose ...` も同じ値で指定できる。
  sim のウィンドウ(`view_sim_pose.py --pose 30 0 30 0 60 60`)の姿勢と、実機の腕の形が同じ見た目になることも確認済み(作者、2026-10-21)。
  これで sim↔実機の変換(単位・符号・0 の位置・gripper の尺度)は、実機で合っている。オフセットは不要。
  次は、sim で求めた目標(例: 視覚サーボの IK の解)を `sim_to_real` で変換して実機に送る。まず「立方体の真上に手先を持っていく」姿勢から。
  `scripts/plan_hover_pose.py`(IK → 実機の値と、sim の予測の手先位置を表示)を作成。既定(立方体 (0.22, -0.12) の真上 10 cm)は
  `--pose 33.5 -69.8 89.0 -6.7 1.8 0.0`。実機で手先の高さ・肩の軸からの距離・向きを測り、予測(10.0 cm / 21.7 cm / 右へ 33.5°)と比べる(未実施)。
  変換の整理は `docs/sim_real_mapping.md`(sim の可動域との比較まで済み。+ の向きの表は実機で確認して埋める)。
  変換コード(`src/hand_sim/sim_real.py`)は、向きが揃ってから実装する。
  **ゲート1は「届く」まで通った時点で通過とし、実機の作業を進める**(作者の了承、2026-10-20)。掴む動作は後回し。
- 実機のサーボとの通信方法は、LeRobot を使う方向(組み立ての公式ガイドが LeRobot 前提のため)。直接シリアル通信にするかは、必要になったら検討する。
- LeRobot ではグリッパを 0〜100 の直動量で扱うが、モデルでは回転関節(rad)。変換は未実装。
- **Mac での動作は未確認**(手順は `README.md` の「Mac で動かす」)。

## 外出先(Mac)でできること
- `README.md` の手順で環境を作り、`check_vision.py` などを動かして確認する。
- 上の「次にやること」の1(数値 IK の考え方の予習)。
- ドキュメント(`docs/`)の読み書き、疑問点の整理。
- 機材の調査(価格、納期、BOM。公式 README の BOM が参考になる)。
