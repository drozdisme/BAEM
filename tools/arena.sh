#!/usr/bin/env bash
# tools/arena.sh — быстрый запуск турнира агентов за одним столом.
# По умолчанию: твой обученный агент против call/random/tight на 100k раздач.
#
#   bash tools/arena.sh                              # дефолтный стол
#   bash tools/arena.sh --seat model:a.bin --seat model:b.bin --hands 200000
#   HANDS=500000 bash tools/arena.sh                 # больше раздач = уже доверит. интервал
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
BIN="$ROOT/bin"
[ -x "$BIN/table_play" ] || bash "$ROOT/tools/build.sh"

if [ "$#" -gt 0 ]; then
    "$BIN/table_play" "$@"
else
    HANDS="${HANDS:-100000}"
    W="${WEIGHTS:-$ROOT/agent_weights.bin}"
    [ -f "$W" ] || { echo "Нет весов $W — сначала обучи (tools/run_all.sh)"; exit 1; }
    "$BIN/table_play" \
        --seat "model:$W" --seat call --seat random --seat tight \
        --hands "$HANDS" --seed 7
fi
