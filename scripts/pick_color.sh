#!/usr/bin/env bash
# 【概要】任意の場所に置いた立方体を、「画像で検出 → 位置から計画を作る → 実機で掴んで運ぶ」まで、1 コマンドで行う。
# 使い方: scripts/pick_color.sh 色 [real_pick_place.py に渡す引数]
#   例: scripts/pick_color.sh green --refine 1
#       scripts/pick_color.sh green --until 2 --refine 1     (降下まで)
# 流れ: (1) detect_cubes_world.py(撮影して、3 色の位置・向きを保存)→ (2) plan_pick_test.py(その位置への計画。届かなければここで止まる。腕は動かない)
#       → (3) real_pick_place.py(実機。腕を支えて Enter、など。引数はそのまま渡す。--record を自動で付ける)
#       → (4) update_grasp_offset.py(掴んだ直後の、手と立方体の位置を測って、指の面のずれを貯める。次の (2) が自動で使う。掴むところまで進まなければ、何もしない)
# set -e: どれかのコマンドが失敗したら、そこで止める(失敗したのに、次の「腕を動かす」へ進まないため)。[頻度] 安全のため、手順をつなぐスクリプトでは定番
set -e
cd "$(dirname "$0")/.."  # このスクリプトの場所から見た、プロジェクトの直下へ移る
COLOR="$1"
shift
.venv/bin/python scripts/detect_cubes_world.py
MUJOCO_GL=egl .venv/bin/python scripts/plan_pick_test.py --color "$COLOR"
~/lerobot-work/lerobot/.venv/bin/python scripts/real_pick_place.py --color "$COLOR" --record "$@"
.venv/bin/python scripts/update_grasp_offset.py --color "$COLOR" || echo "(校正は更新されなかった。上の理由を確認する。掴みの動作そのものには影響しない)"  # || : 前のコマンドが失敗したときだけ、右を実行する。校正の失敗で、全体を失敗扱いにしないため
