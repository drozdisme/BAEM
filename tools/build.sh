#!/usr/bin/env bash
# tools/build.sh — сборка инструментов БЕЗ cmake.
# Если есть third_party/unified_ml → собирает с НАСТОЯЩИМ трансформером
# (-DHAVE_UNIFIED_ML). Иначе — автономный MLP-fallback.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TOOLS="$ROOT/tools"
OUT="$ROOT/bin"
UML="$ROOT/third_party/unified_ml"
mkdir -p "$OUT"

ARCH="-O3"
if echo 'int main(){}' | g++ -std=c++20 -march=native -x c++ - -o /dev/null 2>/dev/null; then
    ARCH="-O3 -march=native"
fi

DEFS=""; INCS="-I$ROOT"; LIB=""
if [ -d "$UML/include" ]; then
    echo "[build] unified_ml найден → НАСТОЯЩИЙ трансформер"
    DEFS="-DHAVE_UNIFIED_ML"
    INCS="$INCS -I$UML/include -I$UML/sle_engine/include"
    LIBA="$OUT/libuml.a"
    if [ ! -f "$LIBA" ]; then
        echo "[build] компилирую unified_ml в статическую библиотеку (один раз)..."
        OBJDIR="$OUT/uml_obj"; mkdir -p "$OBJDIR"
        SRCS=$(ls "$UML"/src/autograd/*.cpp "$UML"/src/core/*.cpp \
                  "$UML"/src/models/transformer/*.cpp "$UML"/src/ucao/*.cpp \
               | grep -v 'symbolic.cpp')
        for s in $SRCS; do
            o="$OBJDIR/$(echo "$s" | md5sum | cut -c1-12).o"
            g++ -std=c++20 $ARCH -fopenmp -I"$UML/include" -I"$UML/sle_engine/include" \
                -c "$s" -o "$o"
        done
        ar rcs "$LIBA" "$OBJDIR"/*.o
        echo "[build] libuml.a готова"
    fi
    LIB="$LIBA -fopenmp"
else
    echo "[build] unified_ml НЕ найден → MLP-fallback (быстро, но слабее)"
fi

echo "[build] train_pokerbench ..."
g++ -std=c++20 $ARCH $DEFS $INCS "$TOOLS/train_pokerbench.cpp" $LIB -o "$OUT/train_pokerbench"
echo "[build] play_pokerbench ..."
g++ -std=c++20 $ARCH $DEFS $INCS "$TOOLS/play_pokerbench.cpp"  $LIB -o "$OUT/play_pokerbench"
echo "[build] table_play (арена v1) ..."
g++ -std=c++20 $ARCH $DEFS $INCS "$TOOLS/table_play.cpp"       $LIB -o "$OUT/table_play"

# ── v2: сайзинг + фичи рука×борд. По умолчанию БЫСТРЫЙ MLP-прайор (лучше на этих фичах).
#    Трансформер-прайор — опционально: USE_TF_PRIOR=1 bash tools/build.sh (медленнее и хуже на seq_len=1).
V2DEFS="-I$ROOT"; V2LIB=""; V2NAME="MLP"
if [ "${USE_TF_PRIOR:-0}" = "1" ] && [ -d "$UML/include" ]; then
    V2DEFS="-DHAVE_UNIFIED_ML -DUSE_TF_PRIOR $INCS"; V2LIB="$LIB"; V2NAME="трансформер(эксперимент)"
fi
echo "[build] train_v2 ($V2NAME) ..."
g++ -std=c++20 $ARCH $V2DEFS "$TOOLS/train_v2.cpp"  $V2LIB -o "$OUT/train_v2"
echo "[build] arena_v2 ..."
g++ -std=c++20 $ARCH $V2DEFS "$TOOLS/arena_v2.cpp"  $V2LIB -o "$OUT/arena_v2"
echo "[build] probe_v2 ..."
g++ -std=c++20 $ARCH $V2DEFS "$TOOLS/probe_v2.cpp"  $V2LIB -o "$OUT/probe_v2"

echo "[build] cfr_solver (GTO-движок, CFR) ..."
g++ -std=c++20 $ARCH "$TOOLS/cfr_solver.cpp" -o "$OUT/cfr_solver"
echo "[build] pushfold_solver (NLHE GTO push/fold) ..."
g++ -std=c++20 $ARCH "$TOOLS/pushfold_solver.cpp" -o "$OUT/pushfold_solver"
echo "[build] lbr_eval (эксплуатируемость агента, LBR) ..."
g++ -std=c++20 $ARCH "$TOOLS/lbr_eval.cpp" -o "$OUT/lbr_eval"
echo "[build] river_solver (постфлоп-CFR, риверный солвер) ..."
g++ -std=c++20 $ARCH "$TOOLS/river_solver.cpp" -o "$OUT/river_solver"
echo "[build] turn_solver (turn-CFR + chance-слой) ..."
g++ -std=c++20 $ARCH "$TOOLS/turn_solver.cpp" -o "$OUT/turn_solver"
echo "[build] resolve (in-play re-solver, 4c) ..."
g++ -std=c++20 $ARCH "$TOOLS/resolve.cpp" -o "$OUT/resolve"
echo "[build] blueprint (префлоп-CFR ядро, 4d) ..."
g++ -std=c++20 $ARCH "$TOOLS/blueprint.cpp" -o "$OUT/blueprint"

echo "[build] OK -> $OUT/{train_pokerbench, play_pokerbench, table_play, train_v2, arena_v2, probe_v2, cfr_solver, pushfold_solver, lbr_eval, river_solver, turn_solver, resolve, blueprint}"
