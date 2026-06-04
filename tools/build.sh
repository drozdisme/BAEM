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

echo "[build] OK -> $OUT/{train_pokerbench, play_pokerbench}"
