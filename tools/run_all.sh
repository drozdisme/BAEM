#!/usr/bin/env bash
# tools/run_all.sh — весь конвейер одной командой (CSV + настоящий трансформер).
#   setup → скачать CSV → подготовить → собрать → обучить → оценить
#
# Запуск из корня исходников baem/:   bash tools/run_all.sh
#
# Переменные (необязательно):
#   REPO        датасет (RZ412/PokerBench)
#   STAGE       preflop | postflop | both   (по умолчанию both)
#   EPOCHS LR BATCH LIMIT  гиперпараметры обучения
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
T="$ROOT/tools"; DATA="$ROOT/data"; BIN="$ROOT/bin"; mkdir -p "$DATA"

REPO="${REPO:-RZ412/PokerBench}"
STAGE="${STAGE:-both}"
EPOCHS="${EPOCHS:-30}"; LR="${LR:-0.0005}"; BATCH="${BATCH:-64}"; LIMIT="${LIMIT:-0}"

echo "######## 1/5 setup ########"
bash "$T/setup.sh"

echo "######## 2/5 download CSV ########"
SNAP=$(python3 "$T/download_pokerbench.py" --repo "$REPO" | tail -1)
echo "snapshot: $SNAP"

find_csv() { find "$SNAP" -iname "$1" | head -1; }
PRE_TR=$(find_csv "*preflop*train*game_scenario*.csv");  PRE_TE=$(find_csv "*preflop*test*game_scenario*.csv")
PST_TR=$(find_csv "*postflop*train*game_scenario*.csv"); PST_TE=$(find_csv "*postflop*test*game_scenario*.csv")

echo "######## 3/5 prepare ########"
: > "$DATA/prepared_train.txt"; : > "$DATA/prepared_test.txt"
prep(){ python3 "$T/prepare_pokerbench_csv.py" --csv "$1" --out "$2" --append --limit "$LIMIT"; }
prep_te(){ python3 "$T/prepare_pokerbench_csv.py" --csv "$1" --out "$2" --append; }
if [ "$STAGE" = preflop ] || [ "$STAGE" = both ]; then
    [ -n "$PRE_TR" ] && prep "$PRE_TR" "$DATA/prepared_train.txt"
    [ -n "$PRE_TE" ] && prep_te "$PRE_TE" "$DATA/prepared_test.txt"
fi
if [ "$STAGE" = postflop ] || [ "$STAGE" = both ]; then
    [ -n "$PST_TR" ] && prep "$PST_TR" "$DATA/prepared_train.txt"
    [ -n "$PST_TE" ] && prep_te "$PST_TE" "$DATA/prepared_test.txt"
fi
echo "train lines: $(wc -l < "$DATA/prepared_train.txt")  test lines: $(wc -l < "$DATA/prepared_test.txt")"

echo "######## 4/5 build ########"
bash "$T/build.sh"

echo "######## 5/5 train ########"
"$BIN/train_pokerbench" --data "$DATA/prepared_train.txt" --out "$ROOT/agent_weights.bin" \
    --epochs "$EPOCHS" --lr "$LR" --batch "$BATCH"

echo "######## eval ########"
if [ -s "$DATA/prepared_test.txt" ]; then
    "$BIN/play_pokerbench" --weights "$ROOT/agent_weights.bin" --eval "$DATA/prepared_test.txt"
fi
echo
echo "ГОТОВО. Веса: $ROOT/agent_weights.bin"
echo "Сыграть ход: echo '0 0 150 0 1 12 25 0 0' | $BIN/play_pokerbench --weights $ROOT/agent_weights.bin --stdin"
