# カメラの認識・キャリブレーション(較正)まわりの文書一覧

「カメラで何を・どうやって測るか」が分からなくなったときの、読む順番つきの目次。
(用語: キャリブレーション = 較正 = カメラの数値を測って、画素とロボットの座標を結びつける作業)

## 全体像: 3 種類の「合わせる」作業がある
| 何を合わせるか | 測るもの | 道具 | 状態 |
|---|---|---|---|
| ① カメラ自身の性質(内部パラメータ) | 焦点距離 fx, fy、中心 cx, cy、歪み | チェスボード | **完了**(誤差 0.275 画素) |
| ② カメラの置き場所(外部パラメータ) | カメラの位置と向き(机・ロボットから見て) | 固定の目印、または手の目印 | **未**(K 版の部品を作る) |
| ③ 腕のずれ(関節角のオフセット) | sim と実機の、関節ごとの角度の差 | 手の ArUco(外せる板) | 板の余白を直して再撮影の段階 |

流れ: ① → 画像を補正 → ② → 立方体の画素が、ロボット座標に直る → 掴める。③ は腕そのものの確認で、別の作業。

## 読む順番(おすすめ)
1. **`vision_basics.md`**(347 行): 「色で探す」「最大の塊」の考え方。画素 → 世界座標の換算(`pixel_to_world`)の基本。**まずここ**。
2. **`calibration.md`**(133 行): 較正とは何か、なぜ要るか。sim での考え方(カメラの位置・視野角を、手の目印から求める)。
3. **`webcam_calibration.md`**(225 行): ① の実機版。チェスボードで内部パラメータを求める手順、0 章が図解、5 章が「画像の補正と換算」(`real_camera.py`)。
4. **`marker_placement.md`**(160 行): ② の目印を、手に付けるか・机に固定するか。
5. **`camera_placement.md`**(104 行): カメラを置く場所の調査(sim で 12 通り)。結論は、前側の斜め上、高さ 40〜90 cm、後ろ側は不可。
6. **`aruco_markers.md`**(53 行): 手に付ける ArUco の検出。**白い余白(5 mm 以上)が要る**理由もここ。
7. **`dot_markers.md`**(50 行): 色シール方式(保留中)。貼り位置の測り誤差に弱い。

## 関連(読まなくても進められる)
- `detector_swap.md`: 検出器(色 / 塊 / モデル)を差し替えられる形にした話。
- `visual_servo.md`、`closed_loop_flow.md`: 視覚サーボ(目で見ながら手を動かす)の流れ。
- `sorting_trials.md`: sim での仕分け試験。5 章が「較正を入れた試験」(カメラが 15 mm ずれると失敗、較正で戻る)。
- `offset_calibration.md`、`sim_real_mapping.md`: ③ の背景(関節ごとのオフセット、sim と実機の変換)。

## 疑問からひける早見表
| 疑問 | 読む所 |
|---|---|
| 画素の位置が、なぜ世界の mm に直せるのか | `vision_basics.md` → `webcam_calibration.md` 5 章 |
| fx, fy, cx, cy, 歪みって何 | `webcam_calibration.md` 0 章(図) |
| 内部パラメータと外部パラメータの違い | 上の表(①と②)、`calibration.md` |
| カメラの位置は、どうやって測る | `calibration.md`、`marker_placement.md` |
| どこにカメラを置けばいい | `camera_placement.md` 結論と 5 章 |
| なぜ手に目印を付ける / 机に固定する | `marker_placement.md` |
| ArUco が読めない | `aruco_markers.md`(余白、大きさ 40 画素以上、明るさ) |
| 立方体の色検出が合わない | `vision_basics.md`、`scripts/view_real_detection.py` |

## コードの場所
| 役割 | ファイル |
|---|---|
| 色検出・画素 → 世界座標(sim) | `src/hand_sim/vision.py` |
| 実機カメラの補正・K を使った換算 | `src/hand_sim/real_camera.py` |
| 任意の位置・向きのカメラの投影と較正 | `src/hand_sim/camera_pose.py`、`src/hand_sim/calibration.py` |
| ArUco / 色シールの検出 | `src/hand_sim/aruco_marker.py`、`src/hand_sim/dot_marker.py` |
| 実機で見るスクリプト | `scripts/view_real_detection.py`、`view_real_markers.py`、`view_real_dots.py` |
| 答え合わせ | `scripts/check_vision.py`、`check_calibration.py`、`check_intrinsics.py`、`check_real_camera.py`、`check_aruco.py` |

## 分からない所があったら
どの文書の、どの部分が分からないかを教えてもらえれば、図や具体例で説明し直す。
