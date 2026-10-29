#!/usr/bin/env bash
# 【概要】立方体を、既定では赤 → 緑 → 青の順に(色と順番は --colors で変えられる)、続けて置き場へ運ぶ「連続仕分け」を 1 エピソードぶん行う(run_real_trial.sh の中の real_pick_place.py が、3 色を 1 回の接続で続ける)。
# 使い方: 立方体を 3 個とも机に置く(置き場には置かない)。腕は休めの姿勢にする。
#   scripts/run_real_sort.sh --speed 5.5 --settle 0.5 --no-confirm-end    (引数は、run_real_trial.sh にそのまま渡す。エピソード番号は、記録から自動で決める)
#   scripts/run_real_sort.sh --from-rest ...                              (従来の方式: 1 個運ぶたびに、休めの姿勢へ戻って、次を休めの姿勢から始める)
#   scripts/run_real_sort.sh --colors green red ...                       (運ぶ色と順番を決める。1〜3 色。既定は red green blue。腕が立方体を隠す場面を作りたいときなど)
#   scripts/run_real_sort.sh --force-hidden ...                           (確認用: 検出が隠れたことにして、休めの姿勢へ戻って検出し直す動きを、必ず通す)
# 既定(置いた姿勢から続ける方式): 赤を置いたら、休めの姿勢へ戻らず、その姿勢のまま検出して、次の立方体の真上へ向かう。
#   腕が立方体を隠して見えない(前に見えていた色が見えなくなった、または検出できなかった)ときだけ、休めの姿勢へ戻って、検出し直す。全部見えていれば、そのまま続ける。
# 記録: outputs/real_trials_sort.csv(環境変数 TRIAL_LOG で変えられる)。色ごとに 1 行。メモ欄に "episode=N" が入る。まとめは scripts/summarize_sort.py。
# 決めごと(模倣学習・VLA と同じ条件で比べるため): 色の順は固定(赤、緑、青)。失敗した色は、やり直さず、飛ばして次の色へ進む(1 エピソードから、3 個ぶんの独立した結果を取る)。
#   置き場に入った立方体は、「ほかの立方体」として数える(動かしたら「他を動かした」)。動かす前の Enter は、最初の 1 回だけ(--from-rest のときは、色ごとに 1 回)。
# set -e は使わない: ある色が失敗しても、次の色へ進みたいため。
cd "$(dirname "$0")/.."
export TRIAL_LOG="${TRIAL_LOG:-real_trials_sort.csv}"
LOGFILE="outputs/$TRIAL_LOG"
# エピソード番号: 記録に既にあるメモの "episode=N" の最大 + 1。grep -o: 一致した部分だけ出す。sort -n | tail -1: 数字として並べて、最後(最大)を取る。[頻度] 「最大の番号 + 1」を出す定番
LAST=$( [ -f "$LOGFILE" ] && grep -o 'episode=[0-9]*' "$LOGFILE" | cut -d= -f2 | sort -n | tail -1 )
EPISODE=$(( ${LAST:-0} + 1 ))
export TRIAL_NOTE="episode=$EPISODE"
# オプションを読む: --colors の後ろの色の並び、--from-rest、--force-hidden。残りは、run_real_trial.sh へそのまま渡す。
# while [ $# -gt 0 ] / shift: 引数を 1 つずつ取り出して進める定番の書き方(case: 値によって分岐する。[頻度] シェルスクリプトの引数処理の定番)
COLORS=()
FROM_REST=0
REST_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --colors) shift
            while [ $# -gt 0 ] && [[ "$1" =~ ^(red|green|blue)$ ]]; do COLORS+=("$1"); shift; done ;;
        --from-rest) FROM_REST=1; shift ;;
        --force-hidden) REST_ARGS+=("--force-hidden"); shift ;;
        *) REST_ARGS+=("$1"); shift ;;
    esac
done
[ ${#COLORS[@]} -eq 0 ] && COLORS=(red green blue)
echo "=== エピソード $EPISODE: ${COLORS[*]}(記録: $LOGFILE)==="
if [ $FROM_REST -eq 1 ]; then
    for COLOR in "${COLORS[@]}"; do
        echo; echo "--- $COLOR (エピソード $EPISODE) ---"
        scripts/run_real_trial.sh "$COLOR" "${REST_ARGS[@]}"
    done
else
    scripts/run_real_trial.sh "${COLORS[0]}" --then "${COLORS[@]:1}" "${REST_ARGS[@]}"
fi
echo; .venv/bin/python scripts/summarize_sort.py --log "$TRIAL_LOG" --episode "$EPISODE"
