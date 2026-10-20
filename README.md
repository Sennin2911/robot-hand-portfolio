# Robot Hand Portfolio

カメラで対象を捉え → シミュレータで検証した制御を → 実機ロボットハンドへ移す、までを目指すプロジェクト。

## ロードマップ
SO-101 で、固定カメラで部品(色違いの立方体)を見つけて種類別に仕分けるロボットセルを作り、
sim と実機で試験した成功率・失敗の内訳・改善を「導入検証報告書」の形でまとめる(計画: `ポートフォリオ計画 — 部品仕分けセル(SO-101).md`)。
1. **[done]** sim の土台と視覚サーボ(順・逆運動学、PD制御、色検出、カメラ較正、SO-101 の数値 IK)
2. **[done]** 実機の組み立て、キャリブレーション、sim↔実機の変換、IK 解の姿勢の再現
3. 手先位置の実測(sim の予測との差)
4. sim で掴む → 仕分けと試験(100回)
5. 実機に固定カメラを付けて通す → 試験と分析 → 改善を1サイクル
進捗の詳細は `docs/next_steps.md`。

## セットアップ
システムに `python3-venv` が無い環境でも動く手順 (sudo 不要):

```bash
python3 -m venv --without-pip .venv
pip3 --python .venv/bin/python install -r requirements.txt
```

## 動作確認
```bash
.venv/bin/python scripts/check_install.py   # 読込・ステップ・EGL描画
.venv/bin/python scripts/demo_grasp.py      # 把持デモ -> outputs/demo_grasp.mp4
```
GUIで把持動作をリアルタイム再生 (ビューア):
```bash
.venv/bin/python scripts/view_grasp.py
```
ヘッドレス描画は `MUJOCO_GL=egl` (Linux のときだけスクリプト側で設定する。Mac では設定しない。詳しくは下の「Mac で動かす」)。GUI で見るなら `.venv/bin/python -m mujoco.viewer --mjcf=models/scene.xml`。

## Mac で動かす
Linux 以外でも、同じコードで動く作り(`MUJOCO_GL=egl` は Linux のときだけ設定する)。
ただし作者は Mac で動作を確認していない。うまく動かないときは、下の「困ったとき」を見る。

**1. 取得とセットアップ** (Python 3.10 以上が目安。`python3 --version` で確認)
```bash
git clone git@github.com:Sennin2911/robot-hand-portfolio.git
cd robot-hand-portfolio
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```
- Linux の `.venv/` は Mac では使えない(`.venv/` は git に入っていないので、Mac で作り直すのが正しい)。
- Mac の Python には `pip` が最初から付くので、`--without-pip` は不要で、`pip3 --python ...` の形も使わない。
  上のように `.venv/bin/python -m pip install ...` と書けばよい。
- Python が無いとき: `brew install python`(Homebrew を入れてある場合)か、python.org のインストーラを使う。

**2. 動作確認** (画面なし。Linux と同じコマンド)
```bash
.venv/bin/python scripts/check_install.py    # 最後に "render OK" と出ればよい
.venv/bin/python scripts/check_vision.py     # 第2週の答え合わせ。"全部OK" と出れば完成
.venv/bin/python scripts/inspect_so101.py    # SO-101 の関節一覧 + outputs/so101_poses.png
```

**3. GUI(ビューア)で見る**: Mac では `python` ではなく **`mjpython`** で実行する必要がある
(MuJoCo の仕様。macOS で `mujoco` を pip で入れると `.venv/bin/mjpython` も入るはず。この Linux には無いので未確認)。
```bash
.venv/bin/mjpython scripts/view_grasp.py
.venv/bin/mjpython scripts/view_arm.py
.venv/bin/mjpython -m mujoco.viewer --mjcf=models/so101/scene_vs.xml
```

**困ったとき**
| 症状 | 試すこと |
|---|---|
| `render FAILED` や描画のエラー | 環境変数 `MUJOCO_GL` が設定されていないか確認(`echo $MUJOCO_GL`)。`egl` になっていたら `unset MUJOCO_GL` |
| ビューアで `launch_passive` のエラー | `python` ではなく `mjpython` で実行しているか確認 |
| `pip install` が失敗する | `python3 --version` が 3.10 以上か確認 |

## 構成
- `models/scene.xml` — 床・2指ハンド(手のひら上下スライド+各指2関節)・把持対象(箱)・カメラ2台(`front`, `top`)
- `src/hand_sim/env.py` — `HandEnv`: reset / step(位置指令) / 状態取得 / render。制御側はこのIFだけに依存させ、後で実機ドライバに差し替える
- `src/hand_sim/grasp.py` — 把持動作の定義 (CUIデモとGUIで共通)
- `scripts/` — 動作確認とデモ (`demo_grasp.py`=動画出力, `view_grasp.py`=GUI)
- `docs/simulation_guide.md` — 入門ガイド (MuJoCoの基本・設計のポイント・env.py/grasp.pyの解説・練習問題)
- `outputs/` — 生成物 (git管理外)

## 注意
- 現在のハンドは仮のモデル。実機が決まったら MJCF を実機の寸法・関節・アクチュエータ特性に合わせて差し替える。
- 把持パラメータは掃引で決めた値 (曲げが大きいと物体を掬い上げてしまい、挟持にならない)。

## 用語・Q&A (初心者向けメモ)

### VMware(仮想マシン)は必要?
不要。MuJoCo は Python のパッケージで、普通の Linux に `pip` で入る。`.venv/` は Python の
「仮想環境」(プロジェクト専用のライブラリ置き場) で、VMware の仮想マシンとは別物。

### なぜ venv を使う?
システムの Python を汚さず、プロジェクトごとにライブラリのバージョンを分けるため。
Python のバージョンが非対応だからではない (この環境の Python 3.14 でも MuJoCo は動いた)。
Ubuntu 系はシステムの Python への直接 `pip install` を制限していることも多く、venv が標準的な方法。

### GUI と CUI(画面なし) はどちらで動かす?
| | 向く場面 |
|---|---|
| GUI (ビューア) | モデルの形や関節の動きを目で確認、接触がおかしいときの原因調査 |
| CUI (画面なし) | 自動テスト、パラメータ掃引、繰り返し検証、動画・グラフの保存 |

このプロジェクトは **CUI を主役**にする。理由: 同じ条件で再現でき、成功・失敗を数値で判定でき、
結果を動画で残せる (「sim で検証 → 実機へ」に必要)。GUI はデバッグの補助として使う。
CUI でも画面に出さずに描画して動画にできる (`demo_grasp.py`)。

### GUI(ビューア)で確認する
```bash
.venv/bin/python scripts/view_grasp.py                                # 把持動作を再生
.venv/bin/python -m mujoco.viewer --mjcf=models/scene.xml             # モデルを自由に触る
```
スペース=一時停止 / Backspace=リセット / 右ドラッグ=カメラ移動。
起動時の `libdecor-gtk.so` と `OpenGL error 0x502` の警告は、Wayland 環境でよく出るもので、
表示が崩れていなければ無視してよい (ウィンドウの枠の装飾が付かないだけ)。
ウィンドウが開かない場合は X11 セッションでの起動や、`MUJOCO_GL=glfw` の指定を試す。

### 用語
- **ビューア**: MuJoCo 付属の、シミュレーションを 3D 表示するウィンドウ型アプリ。
- **Wayland / X11**: Linux が画面を表示するための仕組み (表示サーバ) の種類。X11 が古く、Wayland が新しい。
  このPCは Wayland。MuJoCo のビューアは X11 系の仕組み (GLFW) を使うため、相性の問題が出ることがある。
- **セッション**: ログインしてから使っているデスクトップ環境のこと。「Wayland セッション」= Wayland で動いているデスクトップ。
- **EGL**: 画面を使わず GPU で直接描画する方式。`MUJOCO_GL=egl` で指定する (画面なしの動画出力で使用)。
- **MJCF**: MuJoCo のモデルを書く XML 形式 (`models/scene.xml`)。

## コードを読むときの前提
作者は MuJoCo・Python の初心者。コード中の日本語コメントに、概要と各記法の意味・使用頻度を書く方針 (詳細は `CLAUDE.md`)。
