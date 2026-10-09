# Robot Hand Portfolio

カメラで対象を捉え → シミュレータで検証した制御を → 実機ロボットハンドへ移す、までを目指すプロジェクト。

## ロードマップ
1. **[done] シミュレータと制御の土台** (MuJoCo, 2指ハンド + 物体, スクリプト把持)
2. カメラによる対象認識・位置推定 (sim内カメラ → 実カメラ)
3. 認識結果を使った把持計画・制御 (sim で検証)
4. 実機ハンドへの移植 (制御インタフェース共通化, sim2real ギャップの評価)

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
ヘッドレス描画は `MUJOCO_GL=egl` (スクリプト側で設定済み)。GUI で見るなら `.venv/bin/python -m mujoco.viewer --mjcf=models/scene.xml`。

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
