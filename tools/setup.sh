#!/usr/bin/env bash
# tools/setup.sh — проверка инструментов и установка Python-зависимостей.
set -euo pipefail
echo "[setup] g++ ..."
command -v g++ >/dev/null || { echo "ERROR: установите g++: sudo apt-get install -y g++"; exit 1; }
GCCVER=$(g++ -dumpversion | cut -d. -f1)
[ "$GCCVER" -ge 10 ] || { echo "ERROR: нужен g++ >= 10 (C++20). Текущая: $(g++ -dumpversion)"; exit 1; }
echo "[setup] g++ $(g++ -dumpversion) OK"
echo "[setup] python3 ..."
command -v python3 >/dev/null || { echo "ERROR: нет python3"; exit 1; }
echo "[setup] pip deps (datasets, huggingface_hub, pandas, pyarrow) ..."
python3 -m pip install --quiet --upgrade --break-system-packages \
    "datasets>=2.0" "huggingface_hub>=0.20" pandas pyarrow 2>&1 | tail -2 || \
  python3 -m pip install --quiet --upgrade \
    "datasets>=2.0" "huggingface_hub>=0.20" pandas pyarrow 2>&1 | tail -2 || \
  echo "WARN: pip не прошёл начисто (ок, если данные уже локально)."
echo "[setup] done."
