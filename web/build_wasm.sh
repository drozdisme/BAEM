#!/bin/bash
# Build BAEM poker engine to WebAssembly
# Requirements: Emscripten SDK (emcc in PATH)
# Install: https://emscripten.org/docs/getting_started/downloads.html
# Usage: cd baem/web && bash build_wasm.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="$SCRIPT_DIR"

echo "[build] BAEM Poker Engine → WebAssembly"
echo "[build] source: wasm_bridge.cpp"

emcc "$SCRIPT_DIR/wasm_bridge.cpp" \
  -I"$SCRIPT_DIR/.." \
  -std=c++20 \
  -O2 \
  -s WASM=1 \
  -s EXPORTED_FUNCTIONS='["_engine_init","_engine_new_hand","_engine_action","_engine_get_state","_engine_reset_game"]' \
  -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap"]' \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s INITIAL_MEMORY=33554432 \
  -s ENVIRONMENT=web \
  -s NO_EXIT_RUNTIME=1 \
  -s MODULARIZE=0 \
  --no-entry \
  -o "$OUT_DIR/baem_engine.js"

echo "[build] OK → baem_engine.js + baem_engine.wasm"
echo ""
echo "Deploy these files alongside index.html:"
echo "  baem_engine.js"
echo "  baem_engine.wasm"
echo "  index.html"
echo ""
echo "For local testing (COOP/COEP headers required):"
echo "  python3 -m http.server --bind 127.0.0.1 8080"
echo "  Then open http://localhost:8080"