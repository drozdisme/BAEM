#!/usr/bin/env bash
# tools/run_v2.sh — ПОЛНЫЙ v2-конвейер (сайзинг + рука×борд) на реальном PokerBench.
#   setup → скачать CSV → подготовить v2 → собрать → обучить → турнир
#
#   bash tools/run_v2.sh
# Переменные: REPO, EPOCHS, LR, BATCH, LIMIT
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
T="$ROOT/tools"; DATA="$ROOT/data"; BIN="$ROOT/bin"; mkdir -p "$DATA"
REPO="${REPO:-RZ412/PokerBench}"; EPOCHS="${EPOCHS:-40}"; LR="${LR:-0.0006}"
BATCH="${BATCH:-256}"; LIMIT="${LIMIT:-0}"

echo "#### setup ####"; bash "$T/setup.sh"
echo "#### download ####"
SNAP=$(python3 "$T/download_pokerbench.py" --repo "$REPO" | tail -1); echo "snapshot: $SNAP"
fc(){ find "$SNAP" -iname "$1" | head -1; }
PRE_TR=$(fc "*preflop*train*game_scenario*.csv");  PRE_TE=$(fc "*preflop*test*game_scenario*.csv")
PST_TR=$(fc "*postflop*train*game_scenario*.csv"); PST_TE=$(fc "*postflop*test*game_scenario*.csv")

echo "#### prepare v2 ####"
: > "$DATA/prepared_v2_train.txt"; : > "$DATA/prepared_v2_test.txt"
P(){ python3 "$T/prepare_pokerbench_v2.py" --csv "$1" --out "$2" --append --limit "$LIMIT"; }
Pte(){ python3 "$T/prepare_pokerbench_v2.py" --csv "$1" --out "$2" --append; }
[ -n "$PRE_TR" ] && P "$PRE_TR" "$DATA/prepared_v2_train.txt"
[ -n "$PST_TR" ] && P "$PST_TR" "$DATA/prepared_v2_train.txt"
[ -n "$PRE_TE" ] && Pte "$PRE_TE" "$DATA/prepared_v2_test.txt"
[ -n "$PST_TE" ] && Pte "$PST_TE" "$DATA/prepared_v2_test.txt"
echo "train lines: $(wc -l < "$DATA/prepared_v2_train.txt")"

echo "#### build ####"; bash "$T/build.sh"
echo "#### train v2 ####"
"$BIN/train_v2" --data "$DATA/prepared_v2_train.txt" --out "$ROOT/policy_v2.bin" \
    --epochs "$EPOCHS" --lr "$LR" --batch "$BATCH"

echo "#### АРЕНА: твоя модель vs боты ####"
"$BIN/arena_v2" --seat "model2x:$ROOT/policy_v2.bin" --seat call --seat random --seat tight \
    --hands 100000 --seed 7
echo
echo "ГОТОВО. Веса агента: $ROOT/policy_v2.bin"
echo "Дуэли:  $BIN/arena_v2 --seat model2:$ROOT/policy_v2.bin --seat tight --hands 200000"
