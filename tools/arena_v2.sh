#!/usr/bin/env bash
# tools/arena_v2.sh — турнир ПОЛНОГО BAEM-агента (model2x: прайор + онлайн-эксплойт).
#   bash tools/arena_v2.sh                 # BAEM vs call/random/tight/aggro
#   bash tools/arena_v2.sh ab              # честное A/B: фикс vs адаптив (раздельные столы)
#   bash tools/arena_v2.sh --seat ...      # произвольный стол
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"; BIN="$ROOT/bin"
[ -x "$BIN/arena_v2" ] || bash "$ROOT/tools/build.sh"
W="${WEIGHTS:-$ROOT/policy_v2.bin}"; H="${HANDS:-120000}"
[ -f "$W" ] || { echo "нет $W — сначала обучи: bash tools/run_v2.sh"; exit 1; }

if [ "${1:-}" = "ab" ]; then
  echo "=== A/B (раздельные столы, одинаковое поле и сид) ==="
  echo "-- ФИКС. ПРАЙОР (model2) против поля --"
  "$BIN/arena_v2" --seat "model2:$W"  --seat call --seat random --seat tight --hands "$H" --seed 31 | tail -5
  echo "-- ПОЛНЫЙ BAEM (model2x) против того же поля --"
  "$BIN/arena_v2" --seat "model2x:$W" --seat call --seat random --seat tight --hands "$H" --seed 31 | tail -5
elif [ "$#" -gt 0 ]; then
  "$BIN/arena_v2" "$@"
else
  "$BIN/arena_v2" --seat "model2x:$W" --seat call --seat random --seat tight --seat aggro --hands "$H" --seed 7
fi
