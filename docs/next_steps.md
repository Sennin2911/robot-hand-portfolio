# 進捗と次にやること

別のPC(Mac など)や、新しい Claude Code のセッションから作業を再開するための引き継ぎメモ。
**作業を進めたら、この文書も更新する**(日付を付けて、実態とずれた記述を残さない)。

最終更新: 2026-10-21

## 目的
**2026-10-21 に方針を変更した。** 正式な計画は、リポジトリ直下の `ポートフォリオ計画 — 部品仕分けセル(SO-101).md`(以下「新計画」)。
カメラで部品(2〜3色の立方体)を見つけて種類別に仕分けるロボットセルを SO-101 で作り、sim と実機で試験した成功率・失敗の内訳・改善を
「導入検証報告書」の形で示す(狙いは東京周辺のロボット SIer・導入支援)。これまでの視覚サーボと sim↔実機の変換は、その土台に使う。
旧計画(`最短ポートフォリオ計画 — 視覚サーボ・ロボットハンド.md`、4週間でハンドに移植)は、経緯の記録として残す。

新計画の工程(日付は置かない):
1. 手先位置の実測(5〜10姿勢で、sim の予測と実測の差を表にする)
   → 測る 10 点は `scripts/plan_measure_points.py` が決め、`docs/hand_position_measurement.md` に表とコマンドを出力済み
     (肩の軸から半径 15〜28 cm、向き ±60°、手先の高さ 10 cm と 5 cm。作者が範囲を指定)。**実機での測定は未実施(作者の番)。**
2. sim で掴む(真上 → 降下 → 閉じる → 持ち上げる → 置き場へ → 開く、の状態機械)
3. sim で仕分けと試験(色違いの立方体2〜3種、ランダム配置、100回。失敗を分類してログに残す)
4. 実機に固定カメラを付けて通す(テーブル面で較正をやり直す)
5. 実機で試験と分析(30〜50回、sim と失敗の内訳を比べる)
6. 改善を1サイクル回す(いちばん多い失敗に対策し、再試験して前後を比べる)
範囲外: 障害物回避、経路計画、YOLO の再学習、手首カメラ。カメラは固定(eye-to-hand)に決定。

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
**判定済み(2026-10-20、下の実機の進捗を参照):** 第2週の4項目は完了し、sim で「立方体の真上に 1〜2 mm で届く」(カメラ位置が 2 cm ずれていても、closed-loop または較正で)ところまで通った。
ただし「掴む」ところは未実装。ゲート1は「届く」までで通過とした(機材は購入・組み立て済み)。

## 未決定・未確認のこと
- **機材は購入済み、3Dプリントの出力も完了(2026-10-19 時点)。型番は、作者の認識では SO-101。**
  ただし、使った STL のフォルダ名が `STL/SO101/`(`STL/SO100/` ではない)かは、まだ確認していない(**組み立て後のいま、sim との寸法のずれの調査に必要**)。
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
- **実機の進捗(2026-10-20〜21):** 組み立てとキャリブレーション(`my_follower_arm`)が完了。以下を実機で確認済み。
  - 1関節ずつ ±30 の確認(`scripts/real_move_joints.py`、解説 `docs/real_move_joints.md`)。基本の誤差は ±0.2〜0.5°、重力がかかる関節(shoulder_lift、elbow_flex、wrist_flex)は −30 側で最大 +2.0°。
  - sim↔実機の変換(`src/hand_sim/sim_real.py`、答え合わせ `scripts/check_sim_real.py` 全OK、整理 `docs/sim_real_mapping.md`)。実機の + の向きは全関節とも sim と同じ(符号 +1。wrist_roll は土台側から見て反時計回り)。
    gripper は sim の可動域(−10°〜100°)を実機の 0〜100 に直線で対応させる。
  - sim の姿勢と実機の腕の形は、見た目で一致(`--pose 30 0 30 0 60 60`、全関節 0、IK の解の姿勢、点2・点7)。
  - 使うスクリプト: `scripts/view_sim_pose.py`(sim で姿勢を見る。`--pose` は実機と同じ値)、`scripts/real_send_pose.py`(実機を指定姿勢へ。LeRobot の環境で実行。目標が可動域の端 3° 以内だと拒否)、
    `scripts/plan_hover_pose.py`(位置 → IK → 実機の値)、`scripts/plan_measure_points.py`(測定 10 点)、`scripts/render_sim_pose.py`(姿勢を画像に。測る 4 点に色つきの球)、`scripts/sim_link_angles.py`(sim の各リンクの傾き)。
- **手先位置の実測(新計画の工程1)— 測定中(2026-10-21):** 測る 10 点と表は `docs/hand_position_measurement.md`(`plan_measure_points.py` が生成。実測を書き込むなら、再生成で消えるので別に控える)。
  測る点は、固定側の指の先端(= sim の `gripperframe`)。肩の回転軸を基準に、半径・向き・机の面からの高さを測る。
  - 点1(予測: 半径 22.0 cm、高さ 10.0 cm)の実測は、半径 23.5 cm、高さ 14.5 cm(+1.5 cm、+4.5 cm)。
  - 全関節 0 の姿勢(`--pose 0 0 0 0 0 10`)で、軸と手先の高さを測った(机の面から): ① shoulder_lift の軸 11.7(sim 11.7)、② elbow_flex の軸 23.0(22.9)、③ wrist_flex の軸 24.3(23.4)、④ 手先 25.5(22.6)。
    **土台と上腕の寸法は合っている。ずれは肘から先で増える(+0.1 → +0.9 → +2.9 cm)。**
  - 逆算した関節のオフセット(参考値。測定誤差 ±0.5 cm は ±2° 相当): shoulder_lift −1.8°、elbow_flex −1.5°、wrist_flex −3.8°(マイナスは、sim より手が上を向く側)。点1に当てはめても高さ +2.4 cm 分しか埋まらず、確定ではない。
  - スマホの水平器(Level Tool)で測った各リンクの傾きは、sim の「リンクの長い辺の向き」と、全関節 0 で上腕 約5°・前腕 ±2°・手 1° 以内、点1で上腕 1°・前腕 2° 以内で一致。
    注意: 関節の軸どうしを結ぶ線と、リンクの長い辺は、向きが違う(上腕で約 9°)。比べるときは長い辺どうしで。
  - **次にやること:** (1) 全関節 0 の姿勢で、水平距離を定規で測る: ② 肘の軸 → ③ 手首の軸(sim 13.5 cm)、③ → ④ 手先(sim 15.8 cm)。STL の寸法のずれの確認。
    (2) **印刷に使った STL のフォルダが `STL/SO101/` か**(SO100 のファイルが混ざっていないか)を確認する。 (3) ずれが確定したら、`sim_real.py` にオフセットを足すか、sim のモデルを実機に合わせる。
    (4) 残りの点(点3・点4 など)の実測を進める。
- **ゲート1は「届く」まで通った時点で通過とし、実機の作業を進める**(作者の了承、2026-10-20)。掴む動作は後回し(新計画の工程2)。
- 実機のサーボとの通信方法は、LeRobot を使う方向(組み立ての公式ガイドが LeRobot 前提のため)。直接シリアル通信にするかは、必要になったら検討する。
- **Mac での動作は未確認**(手順は `README.md` の「Mac で動かす」)。

## 外出先(Mac)でできること
- `README.md` の手順で環境を作り、`check_vision.py` などを動かして確認する。
- 上の「次にやること」の1(数値 IK の考え方の予習)。
- ドキュメント(`docs/`)の読み書き、疑問点の整理。
- 機材の調査(価格、納期、BOM。公式 README の BOM が参考になる)。
