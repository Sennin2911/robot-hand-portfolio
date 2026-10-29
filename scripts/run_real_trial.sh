#!/usr/bin/env bash
# 【概要】実機の試験を 1 回ぶん行い、成否を outputs/real_trials.csv に記録する(新計画の工程 5)。
# 使い方: scripts/run_real_trial.sh 色 [real_pick_place.py に渡す引数]
#   例: scripts/run_real_trial.sh green                   (最初の数回向け。終了時に、腕を支える確認が出る)
#       scripts/run_real_trial.sh green --no-confirm-end  (慣れたあと。最初の Enter 1 回だけで、最後まで自動)
# 記録先を分けたいとき: TRIAL_LOG=real_trials_phase6.csv scripts/run_real_trial.sh green(outputs/ の中のファイル名。無ければ real_trials.csv。試験番号は 1 から数え直す)
# 準備: 立方体を机に置く(置き場には置かない)。腕は休めの姿勢にする。試験が終わったら、立方体を置き場から取り、別の場所に置き直して、もう一度実行する。
# 流れ: (1) 検出(掴む前)→ (2) 計画 → (3) 実機(--auto。最初の Enter 1 回だけ)→ (4) 休めの姿勢へ戻ったあとの検出 → (5) 判定と記録。
#       (1)(2) で失敗したら、動かさずに「検出失敗」「届かない」として記録して終わる。(3) の途中で安全装置が働いたら、真上へ戻って中止し、「安全停止」「把持失敗」として記録する。
# set -e は使わない: 途中で失敗しても、失敗した理由を記録するところまで進めたいため(腕を動かす (3) の前の失敗では、(3) へ進まないように、自分で止める)。
cd "$(dirname "$0")/.."  # このスクリプトの場所から見た、プロジェクトの直下へ移る
COLOR="$1"
shift
CONFIRM_END="--confirm-end"
ARGS=()
for a in "$@"; do  # 引数を 1 つずつ見る。--no-confirm-end だけ、このスクリプトで使う(real_pick_place.py には渡さない)
    if [ "$a" = "--no-confirm-end" ]; then CONFIRM_END=""; else ARGS+=("$a"); fi
done
rm -f outputs/trial_before.json outputs/trial_before.png outputs/trial_detect.log outputs/trial_after.json outputs/cubes_detected.json outputs/real_trial_result.json  # 前の試験の結果を、先に消す(取り違えないため)

# tee: 画面に出しながら、同じ内容をファイルにも保存する。[頻度] 「画面で見たいし、あとで原因も調べたい」ときの定番。judge_real_trial.py が、検出失敗の原因をこのログから記録する
# ${PIPESTATUS[0]}: パイプの左側(python)の終了コード。普通の $? は右側(tee)のものになるため、こちらを使う
.venv/bin/python scripts/detect_cubes_world.py --out-json trial_before.json --out-png trial_before.png 2>&1 | tee outputs/trial_detect.log
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    .venv/bin/python scripts/judge_real_trial.py --color "$COLOR" --stage-failed detect; exit 1
fi
# 狙う色が「範囲外」(肩に近すぎて届かない。detect_cubes_world.py の reach 判定)なら、動かす前に止める。試験としては記録しない(範囲外は、成功率の対象に入れないため)
# python -c "…": 1 行だけの Python をその場で実行する。[頻度] シェルの中で JSON を 1 項目だけ読むときの定番。.get(キー, 既定値): キーが無ければ既定値を返す(色が見つからなかった場合も、エラーにならず空になる)
REACH=$(.venv/bin/python -c "import json,sys; d=json.load(open('outputs/trial_before.json')); print(d['cubes'].get(sys.argv[1], {}).get('reach', ''))" "$COLOR" 2>/dev/null)
if [ "$REACH" = "unreachable" ]; then
    echo "【中止】$COLOR の立方体は、肩に近すぎて届かない範囲(範囲外)です。肩の軸から 10 cm 以上に置き直してください。腕は動かしていません。記録もしていません。"
    exit 1
fi
cp outputs/trial_before.json outputs/cubes_detected.json 2>/dev/null  # plan_pick_test.py は、この名前のファイルを読む。 (立方体が 1 つも見つからなければファイルが無く、cp は失敗するが、次の plan_pick_test.py が「検出結果が無い」で止まる)
MUJOCO_GL=egl .venv/bin/python scripts/plan_pick_test.py --color "$COLOR"
RC=$?  # 直前のコマンドの終了コード(0 = 成功、3 = 隣の立方体が近すぎて、どの向きでも掴めない、その他 = 届かないなど)
if [ $RC -ne 0 ]; then
    if [ $RC -eq 3 ]; then STAGE=neighbor; else STAGE=plan; fi
    .venv/bin/python scripts/judge_real_trial.py --color "$COLOR" --stage-failed $STAGE; exit 1
fi

~/lerobot-work/lerobot/.venv/bin/python scripts/real_pick_place.py --color "$COLOR" --auto --refine 1 $CONFIRM_END "${ARGS[@]}"

sleep 2  # 腕が止まって、カメラの前から外れるのを待つ
.venv/bin/python scripts/detect_cubes_world.py --out-json trial_after.json --out-png trial_after.png
.venv/bin/python scripts/judge_real_trial.py --color "$COLOR"
