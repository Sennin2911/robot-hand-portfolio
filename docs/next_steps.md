# 進捗と次にやること

別のPC(Mac など)や、新しい Claude Code のセッションから作業を再開するための引き継ぎメモ。
**作業を進めたら、この文書も更新する**(日付を付けて、実態とずれた記述を残さない)。

最終更新: 2026-10-18

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
- **購入するキットの型番。** 検討中の商品が SO-ARM100 か SO-101 か、3Dプリント部品が含まれるかは、商品ページでの確認が必要。
  印刷用の STL は型番で違う(`STL/SO101/` と `STL/SO100/`)。違うものを印刷すると作り直しになる。
  フォロワーのみで足りる(リーダーは遠隔操作の教示用で、視覚サーボには不要)。
- 3Dプリントを外注するなら、納期から逆算して第2週中に依頼する。
- 実機のサーボとの通信方法(LeRobot を使うか、直接シリアル通信か)は未調査。
- LeRobot ではグリッパを 0〜100 の直動量で扱うが、モデルでは回転関節(rad)。変換は未実装。
- **Mac での動作は未確認**(手順は `README.md` の「Mac で動かす」)。

## 外出先(Mac)でできること
- `README.md` の手順で環境を作り、`check_vision.py` などを動かして確認する。
- 上の「次にやること」の1(数値 IK の考え方の予習)。
- ドキュメント(`docs/`)の読み書き、疑問点の整理。
- 機材の調査(価格、納期、BOM。公式 README の BOM が参考になる)。
