#!/usr/bin/env bash
# 【概要】3 色の立方体を、赤 → 緑 → 青の順に、続けて置き場へ運ぶ「連続仕分け」を 1 エピソードぶん行う(run_real_trial.sh を 3 回呼ぶ)。
# 使い方: 立方体を 3 個とも机に置く(置き場には置かない)。腕は休めの姿勢にする。
#   scripts/run_real_sort.sh                      (エピソード番号は、記録から自動で決める)
#   scripts/run_real_sort.sh --speed 5.5 --settle 0.5 --no-confirm-end    (引数は、run_real_trial.sh にそのまま渡す)
# 記録: outputs/real_trials_sort.csv(環境変数 TRIAL_LOG で変えられる)。色ごとに 1 行。メモ欄に "episode=N" が入る。まとめは scripts/summarize_sort.py。
# 決めごと(模倣学習・VLA と同じ条件で比べるため): 色の順は固定(赤、緑、青)。失敗した色は、やり直さず、飛ばして次の色へ進む(1 エピソードから、3 個ぶんの独立した結果を取る)。
#   各色の前に、検出をやり直す(前の色を運ぶ間に、ほかの立方体が動いていないかを、毎回見るため)。置き場に入った立方体は、「ほかの立方体」として数える(動かしたら「他を動かした」)。
# set -e は使わない: ある色が失敗しても、次の色へ進みたいため。
cd "$(dirname "$0")/.."
export TRIAL_LOG="${TRIAL_LOG:-real_trials_sort.csv}"
LOGFILE="outputs/$TRIAL_LOG"
# エピソード番号: 記録に既にあるメモの "episode=N" の最大 + 1。grep -o: 一致した部分だけ出す。sort -n | tail -1: 数字として並べて、最後(最大)を取る。[頻度] 「最大の番号 + 1」を出す定番
LAST=$( [ -f "$LOGFILE" ] && grep -o 'episode=[0-9]*' "$LOGFILE" | cut -d= -f2 | sort -n | tail -1 )
EPISODE=$(( ${LAST:-0} + 1 ))
echo "=== エピソード $EPISODE: 赤 → 緑 → 青(記録: $LOGFILE)==="
for COLOR in red green blue; do
    echo; echo "--- $COLOR (エピソード $EPISODE) ---"
    TRIAL_NOTE="episode=$EPISODE" scripts/run_real_trial.sh "$COLOR" "$@"
done
echo; .venv/bin/python scripts/summarize_sort.py --log "$TRIAL_LOG" --episode "$EPISODE"
